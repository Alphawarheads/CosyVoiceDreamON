"""Small CPU tests; no transformers, audio dependencies, GPU or model weights.

Run: python -m unittest discover -s tests -p test_dreamon_speech.py -v
"""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM, validate_speech_tokens


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # Distinct from hypothetical CosyVoice IDs: generate must use this tokenizer.
        return [ord(character) % 5 for character in text]


class TinyBackbone(nn.Module):
    def __init__(self, positions_only=False, nonfinite=False):
        super().__init__()
        self.embedding = nn.Embedding(16, 6)
        self.config = SimpleNamespace(max_position_embeddings=128)
        self.positions_only = positions_only
        self.nonfinite = nonfinite

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, position_ids, use_cache, return_dict):
        assert not use_cache
        assert attention_mask.dtype == torch.bool
        assert bool(attention_mask.all())
        assert attention_mask.shape == (1, 1, inputs_embeds.shape[1], inputs_embeds.shape[1])
        if self.positions_only:
            hidden = position_ids.unsqueeze(-1).expand_as(inputs_embeds).float()
        else:
            # Global context ensures projected reference/task embeddings get gradients.
            hidden = inputs_embeds + inputs_embeds.mean(dim=1, keepdim=True)
        if self.nonfinite:
            hidden = hidden * float("nan")
        return SimpleNamespace(last_hidden_state=hidden)


def tiny_bridge(**kwargs):
    return DreamOnSpeechLM(
        backbone=TinyBackbone(**kwargs), tokenizer=TinyTokenizer(),
        speech_embedding=nn.Embedding(10, 2), speech_decoder=nn.Linear(2, 10),
        task_embedding=nn.Embedding(2, 2), mask_token_id=7, bos_token_id=8, eos_token_id=9,
        speech_vocab_size=7, seed=123, max_sequence_tokens=128,
    )


class SpeechBridgeTests(unittest.TestCase):
    def test_reject_codec_controls_empty_float_and_wrong_batch(self):
        invalid = [torch.empty(1, 0, dtype=torch.int32),
                   torch.tensor([[7]]), torch.tensor([[-1]]),
                   torch.tensor([[1.0]]), torch.tensor([[1], [2]])]
        for tokens in invalid:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    validate_speech_tokens(tokens, vocab_size=7)
        validate_speech_tokens(torch.tensor([[0, 6]], dtype=torch.int32), vocab_size=7)

    def test_dreamon_one_position_shift(self):
        model = tiny_bridge(positions_only=True)
        with torch.no_grad():
            model.speech_out_proj.weight.zero_()
            model.speech_out_proj.weight[0, 0] = 1
            model.speech_decoder.weight.zero_()
            model.speech_decoder.bias.zero_()
            model.speech_decoder.weight[0, 0] = 1
        # BOS + two text IDs + TASK + three reference tokens = 7 prefix positions.
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[0, 1, 2]]),
                       torch.tensor([[-1, -1, -1]]))
        torch.testing.assert_close(logits[0, :, 0], torch.tensor([6., 7., 8.]))

    def test_control_classes_cannot_escape_and_final_partial_step_finishes(self):
        model = tiny_bridge()
        with torch.no_grad():
            model.speech_decoder.weight.zero_()
            model.speech_decoder.bias.zero_()
            model.speech_decoder.bias[6] = 10
            model.speech_decoder.bias[7:] = 1000  # Forbidden classes dominate raw logits.
        tokens, stats = model.generate("ab", "c", torch.tensor([[1, 2]]),
                                       speech_tokens=5, tokens_per_step=2)
        torch.testing.assert_close(tokens, torch.full((1, 5), 6, dtype=torch.int32))
        self.assertEqual([step["remaining_masks"] for step in stats["steps"]], [3, 1, 0])
        self.assertEqual(stats["text_tokens"], 3)
        positions = [p for step in stats["steps"] for p in step["positions"]]
        self.assertEqual(sorted(positions), list(range(5)))
        self.assertEqual(stats["adaptation_status"], "untrained_random_projections")

    def test_seeded_sampling_and_projections_are_reproducible(self):
        first, second = tiny_bridge(), tiny_bridge()
        torch.testing.assert_close(first.speech_in_proj.weight, second.speech_in_proj.weight)
        torch.testing.assert_close(first.speech_out_proj.weight, second.speech_out_proj.weight)
        prompt = torch.empty(1, 0, dtype=torch.int64)
        a, _ = first.generate("abc", "", prompt, speech_tokens=6, temperature=0.7, seed=42)
        b, _ = first.generate("abc", "", prompt, speech_tokens=6, temperature=0.7, seed=42)
        torch.testing.assert_close(a, b)

    def test_nonfinite_logits_abort_before_decoding(self):
        model = tiny_bridge(nonfinite=True)
        with self.assertRaises(FloatingPointError):
            model.generate("a", "", torch.tensor([[1]]), speech_tokens=3)

    def test_invalid_sampling_settings_and_context_overflow(self):
        model = tiny_bridge()
        for settings in ({"speech_tokens": 0}, {"tokens_per_step": 0},
                         {"temperature": float("nan")}, {"temperature": -1}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    model.generate("a", "", torch.tensor([[1]]), **settings)
        with self.assertRaisesRegex(ValueError, "Sequence has"):
            model.generate("a" * 120, "", torch.tensor([[1, 2, 3]]), speech_tokens=8)

    def test_future_adapter_training_gets_gradients_but_frozen_weights_do_not(self):
        model = tiny_bridge()
        logits = model(torch.tensor([[1, 2]]), torch.tensor([[1, 2]]),
                       torch.tensor([[-1, 3, -1]]))
        logits.square().mean().backward()
        for projection in (model.speech_in_proj, model.speech_out_proj):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.embedding.weight.grad)
        self.assertIsNone(model.speech_embedding.weight.grad)
        self.assertIsNone(model.speech_decoder.weight.grad)

    def test_copied_speech_weights_survive_removing_original_llm(self):
        original = SimpleNamespace(speech_embedding=nn.Embedding(10, 2),
                                   llm_decoder=nn.Linear(2, 10), llm_embedding=nn.Embedding(2, 2))
        copied = DreamOnSpeechLM.copy_cosyvoice_components(original)
        expected = original.speech_embedding.weight.detach().clone()
        with torch.no_grad():
            original.speech_embedding.weight.zero_()
        del original
        torch.testing.assert_close(copied["speech_embedding"].weight, expected)


if __name__ == "__main__":
    unittest.main()
