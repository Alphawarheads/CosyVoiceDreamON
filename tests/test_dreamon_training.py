"""CPU training/checkpoint tests; optional YAML/parquet integration checks."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from cosyvoice.dataset.dreamon_processor import make_token_batches, pack_token_batch, read_token_samples
from cosyvoice.llm.dreamon_training import DreamOnSpeechTrainer, build_training_model, load_cosyvoice_speech_components
from test_dreamon_speech import TinyTokenizer, tiny_bridge


def training_batch():
    return pack_token_batch([
        {"utt": "short", "text_ids": torch.tensor([1, 2]), "speech_ids": torch.tensor([0, 1, 2])},
        {"utt": "long", "text_ids": torch.tensor([1, 2, 3]), "speech_ids": torch.tensor([2, 3, 4, 5])},
    ])


class TrainingTests(unittest.TestCase):
    def test_freeze_switch_updates_only_selected_modules_with_separate_lrs(self):
        for freeze in (True, False):
            with self.subTest(freeze_dreamon=freeze):
                torch.manual_seed(123)
                trainer = DreamOnSpeechTrainer(tiny_bridge(), freeze_dreamon=freeze,
                                                dreamon_lr=0.002).train()
                before = {name: p.detach().clone() for name, p in trainer.named_parameters()}
                optimizer = torch.optim.AdamW(trainer.optimizer_parameter_groups(), lr=0.01)
                self.assertEqual([g["lr"] for g in optimizer.param_groups],
                                 [0.01] if freeze else [0.01, 0.002])
                trainer(training_batch(), "cpu")["loss"].backward()
                self.assertEqual(trainer.bridge.backbone.embedding.weight.grad is None, freeze)
                optimizer.step()
                changed = {name for name, p in trainer.named_parameters() if not torch.equal(before[name], p)}
                expected = {"bridge.speech_in_proj.weight", "bridge.speech_out_proj.weight"}
                if not freeze:
                    expected.add("bridge.backbone.embedding.weight")
                self.assertEqual(changed, expected)

    def test_unfrozen_bf16_backbone_uses_fp32_trainable_weights(self):
        bridge = tiny_bridge().bfloat16()
        trainer = DreamOnSpeechTrainer(bridge, freeze_dreamon=False).train()
        self.assertTrue(all(p.dtype == torch.float32 for p in trainer.parameters() if p.requires_grad))
        self.assertEqual(bridge.speech_decoder.weight.dtype, torch.bfloat16)
        trainer(training_batch(), "cpu")["loss"].backward()
        self.assertTrue(torch.isfinite(bridge.backbone.embedding.weight.grad).all())
        with self.assertRaisesRegex(ValueError, "boolean"):
            DreamOnSpeechTrainer(tiny_bridge(), freeze_dreamon="false")

    def test_full_checkpoint_round_trip_and_refrozen_export_preserve_backbone(self):
        torch.manual_seed(123)
        trainer = DreamOnSpeechTrainer(tiny_bridge(), freeze_dreamon=False).train()
        optimizer = torch.optim.AdamW(trainer.optimizer_parameter_groups(), lr=0.01)
        trainer(training_batch(), "cpu")["loss"].backward()
        optimizer.step()
        payload = trainer.training_checkpoint({"epoch": 2, "step": 8})
        self.assertEqual(payload["checkpoint_scope"], "backbone_and_projections")
        self.assertFalse(payload["freeze_dreamon"])
        # Use identical frozen CosyVoice modules, a fresh (different) DreamOn core.
        restored = copy.deepcopy(trainer.bridge)
        with torch.no_grad():
            restored.backbone.embedding.weight.zero_()
            restored.speech_in_proj.weight.zero_()
            restored.speech_out_proj.weight.zero_()
        restored.backbone_checkpoint_required = False
        frozen_trainer = DreamOnSpeechTrainer(restored, freeze_dreamon=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.pt"
            torch.save(payload, path)
            frozen_trainer.load_training_checkpoint(path)
        self.assertFalse(any(p.requires_grad for p in restored.backbone.parameters()))
        torch.testing.assert_close(restored.backbone.embedding.weight, trainer.bridge.backbone.embedding.weight)
        reference = trainer.bridge(torch.tensor([[1, 2]]), torch.tensor([[1]]), torch.tensor([[-1, -1]]))
        actual = restored(torch.tensor([[1, 2]]), torch.tensor([[1]]), torch.tensor([[-1, -1]]))
        torch.testing.assert_close(actual, reference)
        refrozen = frozen_trainer.training_checkpoint({"epoch": 0, "step": 1})
        self.assertTrue(refrozen["freeze_dreamon"])
        self.assertEqual(refrozen["checkpoint_scope"], "backbone_and_projections")
        torch.testing.assert_close(refrozen["backbone_state_dict"]["embedding.weight"],
                                   payload["backbone_state_dict"]["embedding.weight"])

    def test_bad_full_checkpoint_cannot_partly_load_projections(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge(), freeze_dreamon=False)
        payload = trainer.training_checkpoint({"epoch": 0, "step": 1})
        original = trainer.bridge.speech_in_proj.weight.detach().clone()
        payload["state_dict"]["speech_in_proj.weight"].zero_()
        payload["backbone_state_dict"]["embedding.weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            trainer.load_training_checkpoint(payload)
        torch.testing.assert_close(trainer.bridge.speech_in_proj.weight, original)
        del payload["backbone_state_dict"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            trainer.load_training_checkpoint(payload)

    def test_legacy_projection_checkpoint_can_initialize_unfrozen_training(self):
        for version in (1, 2):
            with self.subTest(version=version):
                source = tiny_bridge().initial_adapter_checkpoint()
                source["format_version"] = version
                source.pop("checkpoint_scope")
                source.pop("freeze_dreamon")
                trainer = DreamOnSpeechTrainer(tiny_bridge(), freeze_dreamon=False)
                trainer.load_training_checkpoint(source)
                self.assertTrue(all(p.requires_grad for p in trainer.bridge.backbone.parameters()))
                self.assertIn("backbone_state_dict", trainer.training_checkpoint({"epoch": 0, "step": 1}))

    def test_factory_passes_freeze_switch_without_loading_real_weights(self):
        for freeze in (True, False):
            with self.subTest(freeze=freeze), \
                    patch("cosyvoice.llm.dreamon_training.load_cosyvoice_speech_components", return_value={}), \
                    patch("cosyvoice.llm.dreamon_training.DreamOnSpeechLM.from_local_weights",
                          return_value=tiny_bridge().bfloat16()):
                trainer = build_training_model("dreamon", "cosyvoice", gradient_checkpointing=False,
                                                freeze_dreamon=freeze, dreamon_lr=0.002)
            self.assertEqual(trainer.bridge.freeze_dreamon, freeze)
            self.assertEqual(trainer.dreamon_lr, 0.002)

    def test_original_training_contract_and_optimizer_update(self):
        torch.manual_seed(123)
        trainer = DreamOnSpeechTrainer(tiny_bridge()).train()
        before = {name: p.detach().clone() for name, p in trainer.named_parameters()}
        optimizer = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=0.01)
        result = trainer(training_batch(), torch.device("cpu"))
        self.assertEqual(set(result), {"loss", "acc", "mask_fraction"})
        self.assertTrue(all(value.ndim == 0 and torch.isfinite(value) for value in result.values()))
        result["loss"].backward()
        optimizer.step()
        changed = {name for name, p in trainer.named_parameters() if not torch.equal(before[name], p)}
        self.assertEqual(changed, {"bridge.speech_in_proj.weight", "bridge.speech_out_proj.weight"})

    def test_padding_never_reaches_attention_or_loss(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge()).eval()
        original = training_batch()
        modified = copy.deepcopy(original)
        # IDs far outside either vocabulary: only padding positions are changed.
        modified["text_token"][0, 2] = 999999
        modified["speech_token"][0, 3] = -999999
        expected = trainer(original, "cpu")
        actual = trainer(modified, "cpu")
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_validation_masks_stable_under_rng_and_batch_reordering(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge()).eval()
        batch = training_batch()
        first = trainer(batch, "cpu")
        torch.manual_seed(999)
        torch.rand(20)
        reversed_batch = {key: (value.flip(0) if isinstance(value, torch.Tensor)
                                else value[::-1] if key == "utts" else value)
                          for key, value in batch.items()}
        second = trainer(reversed_batch, "cpu")
        for key in first:
            torch.testing.assert_close(first[key], second[key])

    def test_visible_prefix_is_disjoint_from_masked_target(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge(), prompt_probability=1,
                                        max_prompt_fraction=0.5, full_mask_probability=1).train()
        original = torch.tensor([[0, 1, 2, 3, 4, 5]])
        prompt, target, canvas, mask = trainer._corrupt(original, "utt")
        self.assertGreater(prompt.numel(), 0)
        torch.testing.assert_close(torch.cat((prompt, target), dim=1), original)
        self.assertTrue(mask.all())
        self.assertTrue(canvas.eq(-1).all())
        torch.testing.assert_close(original, torch.tensor([[0, 1, 2, 3, 4, 5]]))

    def test_loss_only_supervises_masked_positions(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge()).train()
        batch = {"utts": ["a"], "text_tokenizer": "dreamon", "text_token": torch.tensor([[1]]),
                 "text_token_len": torch.tensor([1]), "speech_token": torch.tensor([[1, 2, 3]]),
                 "speech_token_len": torch.tensor([3])}
        logits = torch.randn(1, 3, 10, requires_grad=True)
        corrupted = (torch.empty(1, 0, dtype=torch.long), batch["speech_token"],
                     torch.tensor([[-1, 2, -1]]), torch.tensor([[True, False, True]]))
        with patch.object(trainer, "_corrupt", return_value=corrupted), \
                patch.object(trainer.bridge, "forward", return_value=logits) as forward:
            result = trainer(batch, "cpu")
        expected = F.cross_entropy(logits[0, [0, 2], :7], torch.tensor([1, 3]))
        torch.testing.assert_close(result["loss"], expected)
        torch.testing.assert_close(forward.call_args.args[2], corrupted[2])
        result["loss"].backward()
        self.assertTrue(logits.grad[0, 1].eq(0).all())
        self.assertTrue(logits.grad[..., 7:].eq(0).all())

    def test_wrong_tokenizer_is_rejected(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge())
        batch = training_batch()
        batch["text_tokenizer"] = "cosyvoice"
        with self.assertRaisesRegex(ValueError, "DreamOn data pipeline"):
            trainer(batch, "cpu")

    def test_fp32_projections_with_bf16_frozen_modules(self):
        bridge = tiny_bridge()
        for module in (bridge.backbone, bridge.speech_embedding, bridge.speech_decoder, bridge.task_embedding):
            module.bfloat16()
        trainer = DreamOnSpeechTrainer(bridge).train()
        loss = trainer(training_batch(), "cpu")["loss"]
        loss.backward()
        self.assertEqual(bridge.speech_in_proj.weight.grad.dtype, torch.float32)
        self.assertEqual(bridge.speech_out_proj.weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_refuses_to_omit_unfrozen_base_weights(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge())
        trainer.bridge.speech_decoder.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "additional unfrozen"):
            trainer.training_checkpoint({"epoch": 0, "step": 1})

    def test_training_checkpoint_round_trip_into_inference(self):
        trainer = DreamOnSpeechTrainer(tiny_bridge()).train()
        optimizer = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=0.01)
        trainer(training_batch(), "cpu")["loss"].backward()
        optimizer.step()
        payload = trainer.training_checkpoint({"epoch": 0, "step": 1})
        self.assertEqual(set(payload["state_dict"]), {"speech_in_proj.weight", "speech_out_proj.weight"})
        reference_tokens, _ = trainer.bridge.generate("ab", "c", torch.tensor([[1]]), speech_tokens=4)
        restored = copy.deepcopy(trainer.bridge)
        with torch.no_grad():
            restored.speech_in_proj.weight.zero_()
            restored.speech_out_proj.weight.zero_()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_0_whole.pt"
            torch.save(payload, checkpoint)
            restored.load_adapter_checkpoint(checkpoint)
        actual, diagnostics = restored.generate("ab", "c", torch.tensor([[1]]), speech_tokens=4)
        torch.testing.assert_close(actual, reference_tokens)
        self.assertEqual(diagnostics["adaptation_status"], "training_checkpoint")

    def test_bad_checkpoint_does_not_partially_modify_parameters(self):
        bridge = tiny_bridge()
        original = bridge.speech_in_proj.weight.detach().clone()
        checkpoint = bridge.initial_adapter_checkpoint()
        checkpoint["state_dict"]["speech_in_proj.weight"].zero_()
        checkpoint["state_dict"]["speech_out_proj.weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            bridge.load_adapter_checkpoint(checkpoint)
        torch.testing.assert_close(bridge.speech_in_proj.weight, original)
        with self.assertRaises(ValueError):
            bridge.load_adapter_checkpoint({"llm_decoder.weight": torch.zeros(10, 2)})

    def test_load_cosyvoice_components_directly_from_checkpoint(self):
        source = tiny_bridge()
        with tempfile.TemporaryDirectory() as directory:
            torch.save({"speech_embedding.weight": source.speech_embedding.weight.detach(),
                        "llm_decoder.weight": source.speech_decoder.weight.detach(),
                        "llm_decoder.bias": source.speech_decoder.bias.detach(),
                        "llm_embedding.weight": source.task_embedding.weight.detach()}, Path(directory) / "llm.pt")
            result = load_cosyvoice_speech_components(directory, speech_vocab_size=7)
        torch.testing.assert_close(result["speech_decoder"].weight, source.speech_decoder.weight)
        torch.testing.assert_close(result["speech_embedding"].weight, source.speech_embedding.weight)

    def test_jsonl_pipeline_batches_and_filters_by_total_context(self):
        records = [{"utt": "a", "text": "ab", "speech_token": [1, 2, 3]},
                   {"utt": "b", "text": "c", "speech_token": [2]},
                   {"utt": "too_long", "text": "x" * 100, "speech_token": [1]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            batches = list(make_token_batches(read_token_samples([{"src": str(path)}]),
                                              TinyTokenizer, batch_size=2, max_sequence_tokens=20, mode="dev"))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["utts"], ["a", "b"])
        self.assertEqual(batches[0]["speech_token_len"].tolist(), [3, 1])
        self.assertEqual(batches[0]["text_tokenizer"], "dreamon")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow not installed")
    def test_parquet_list_column_and_tar_extension(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parquet_000.tar"
            pq.write_table(pa.table({"utt": ["a"], "text": ["ab"], "speech_token": [[1, 2, 3]],
                                    "audio_data": [b"not decoded"]}), path)
            records = list(read_token_samples([{"src": str(path)}]))
        self.assertEqual(records[0]["speech_token"], [1, 2, 3])
        self.assertNotIn("audio_data", records[0])

    @unittest.skipUnless(importlib.util.find_spec("hyperpyyaml"), "HyperPyYAML not installed")
    def test_training_yaml_constructs_model_and_callable_pipeline(self):
        from hyperpyyaml import load_hyperpyyaml
        trainer = DreamOnSpeechTrainer(tiny_bridge())
        path = Path(__file__).resolve().parents[1] / "configs/dreamon_cosyvoice_train.yaml"
        factory_calls = []

        def factory(**kwargs):
            # HyperPyYAML !apply expects a real function, not a MagicMock.
            factory_calls.append(kwargs)
            return trainer

        with patch("cosyvoice.llm.dreamon_training.build_training_model", new=factory):
            with path.open(encoding="utf-8") as handle:
                config = load_hyperpyyaml(handle, overrides={"flow": None, "hift": None, "hifigan": None})
            self.assertTrue(factory_calls[-1]["freeze_dreamon"])
            with path.open(encoding="utf-8") as handle:
                load_hyperpyyaml(handle, overrides={"freeze_dreamon": False, "dreamon_lr": 0.000002})
            self.assertFalse(factory_calls[-1]["freeze_dreamon"])
            self.assertEqual(factory_calls[-1]["dreamon_lr"], 0.000002)
        self.assertIs(config["llm"], trainer)
        self.assertEqual(config["model_family"], "dreamon_speech_adapter")
        self.assertTrue(all(callable(stage) for stage in config["data_pipeline"]))
        self.assertEqual(config["data_pipeline"][1].func.__name__, "make_token_batches")


if __name__ == "__main__":
    unittest.main()
