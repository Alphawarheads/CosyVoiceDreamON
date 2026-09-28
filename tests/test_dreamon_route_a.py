"""Route A: duration supervision, checkpoint and inference regression tests."""
import copy
import unittest
from unittest.mock import patch
import torch
from cosyvoice.llm.dreamon_route_a import (
    SpeechDurationPredictor, build_route_a_model, predict_speech_length)
from test_dreamon_lora import lora_bridge
from test_dreamon_training import training_batch


def model():
    with patch("cosyvoice.llm.dreamon_route_a.load_cosyvoice_speech_components", return_value={}), \
         patch("cosyvoice.llm.dreamon_route_a.DreamOnSpeechLM.from_local_weights",
               return_value=lora_bridge(True)):
        return build_route_a_model("dreamon", "cosyvoice", gradient_checkpointing=False,
                                   use_lora=True, lora_rank=2, lora_alpha=4., lora_dropout=0.,
                                   reference_dropout=0.)


class RouteATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def test_objective_gradients_and_frozen_base(self):
        m = model().train()
        b = training_batch()
        b["reference_rate"] = [4., None]
        original = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = torch.optim.AdamW(m.optimizer_parameter_groups(), lr=.001)
        result = m(b, "cpu")
        torch.testing.assert_close(result["loss"],
                                   result["token_ce"] + .1 * result["duration_loss"])
        result["loss"].backward()
        for n, p in m.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, n)
                self.assertTrue(torch.isfinite(p.grad).all(), n)
            else:
                self.assertIsNone(p.grad, n)
        opt.step()
        for n, p in m.named_parameters():
            if not p.requires_grad:
                torch.testing.assert_close(p, original[n])
        self.assertTrue(any(not torch.equal(p, original[n])
                            for n, p in m.named_parameters() if "duration_predictor" in n))

    def test_duration_branch_does_not_update_speech_backbone(self):
        m = model().train()
        b = training_batch(); b["reference_rate"] = [4., 5.]
        m(b, "cpu")["duration_loss"].backward()
        for n, p in m.named_parameters():
            if "duration_predictor" not in n:
                self.assertIsNone(p.grad, n)

    def test_checkpoint_roundtrip_and_inference_without_target_audio(self):
        m = model().eval()
        b = training_batch(); b["reference_rate"] = [4., None]
        expected = m(b, "cpu")
        payload = m.training_checkpoint(dict(epoch=0, step=1))
        self.assertEqual(payload["format_version"], 5)
        self.assertIn("backbone_state_dict", payload)
        bridge = lora_bridge(True)
        # Match the external frozen CosyVoice components, as real loading does.
        for key in ("speech_embedding", "speech_decoder", "task_embedding"):
            getattr(bridge, key).load_state_dict(getattr(m.bridge, key).state_dict())
        bridge.speech_in_proj.float(); bridge.speech_out_proj.float()
        bridge.load_adapter_checkpoint(payload)
        bridge.eval()
        pred, info = predict_speech_length(bridge, "hello", "hello", 20, maximum=3)
        self.assertTrue(1 <= pred <= 3)
        self.assertFalse(info["target_audio_used"])
        self.assertEqual(bridge.speech_conditioning, "text_only")
        self.assertEqual(next(bridge.duration_predictor.parameters()).dtype, torch.float32)
        ids = torch.tensor([[1, 2]])
        emb1 = m.bridge.backbone.get_input_embeddings()(ids)
        emb2 = bridge.backbone.get_input_embeddings()(ids)
        torch.testing.assert_close(m.bridge.duration_predictor(emb1, 4.),
                                   bridge.duration_predictor(emb2, 4.))
        for k, v in payload["backbone_state_dict"].items():
            torch.testing.assert_close(bridge.backbone.state_dict()[k], v)

    def test_corrupt_duration_checkpoint_rejected_before_projection_mutation(self):
        m = model()
        payload = m.training_checkpoint(dict(epoch=0, step=1))
        payload["state_dict"]["speech_in_proj.weight"].zero_()
        first = next(iter(payload["route_a"]["duration_state_dict"].values()))
        first.flatten()[0] = float("nan")
        original = m.bridge.speech_in_proj.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "duration tensor"):
            m.bridge.load_adapter_checkpoint(payload)
        torch.testing.assert_close(m.bridge.speech_in_proj.weight, original)

    def test_legacy_checkpoint_does_not_offer_learned_duration(self):
        bridge = lora_bridge()
        with self.assertRaisesRegex(ValueError, "Route A checkpoint"):
            predict_speech_length(bridge, "hello")
        payload = bridge.initial_adapter_checkpoint()
        other = lora_bridge()
        other.load_adapter_checkpoint(payload)
        self.assertFalse(hasattr(other, "duration_predictor"))

    def test_duration_module_can_fit_length_labels(self):
        p = SpeechDurationPredictor(6, hidden_size=8)
        x = torch.randn(1, 4, 6)
        label = torch.tensor(50.).log()
        opt = torch.optim.Adam(p.parameters(), lr=.01)
        initial = (p(x, 3.) - label).abs().item()
        for _ in range(60):
            opt.zero_grad()
            loss = (p(x, 3.) - label).square()
            loss.backward(); opt.step()
        self.assertLess((p(x, 3.) - label).abs().item(), initial * .15)


if __name__ == "__main__":
    unittest.main()
