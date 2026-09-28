"""CPU LoRA gradient, dtype, checkpoint and generation regression tests."""

import copy
import argparse
import ast
import importlib.util
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cosyvoice.llm.dreamon_lora import LoRALinear, lora_config, prepare_lora
from cosyvoice.llm.dreamon_training import DreamOnSpeechTrainer, build_training_model
from test_dreamon_speech import TinyBackbone, tiny_bridge
from test_dreamon_training import training_batch


class AttentionBackbone(TinyBackbone):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(6, 6)
        self.k_proj = nn.Linear(6, 6)
        self.v_proj = nn.Linear(6, 6)
        self.o_proj = nn.Linear(6, 6)
        self.checkpointing = False

    def compute(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x) + self.k_proj(x) + self.v_proj(x)))

    def forward(self, **kwargs):
        x = super().forward(**kwargs).last_hidden_state
        hidden = checkpoint(self.compute, x, use_reentrant=False) if self.checkpointing else self.compute(x)
        return SimpleNamespace(last_hidden_state=hidden)


def lora_bridge(bf16=False):
    model = tiny_bridge()
    model.backbone = AttentionBackbone()
    model.backbone.requires_grad_(False)
    model.eval()
    return model.bfloat16() if bf16 else model


def trainer_for(bridge, **kwargs):
    return DreamOnSpeechTrainer(bridge, use_lora=True, lora_rank=2,
                               lora_alpha=4, lora_dropout=0.0, lora_lr=0.002, **kwargs)


class LoRATests(unittest.TestCase):
    def test_cli_lora_options_are_parsed(self):
        path = Path(__file__).resolve().parents[1] / 'cosyvoice/bin/train.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_args')
        context = {'argparse': argparse, 'deepspeed': None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), context)
        argv = ['train', '--model', 'llm', '--config', 'config.yaml', '--train_data', 'train.list',
                '--cv_data', 'dev.list', '--model_dir', 'exp', '--use_lora', 'true',
                '--freeze_dreamon', 'true', '--lora_rank', '16', '--lora_alpha', '32',
                '--lora_dropout', '0.05', '--lora_lr', '0.0001',
                '--lora_target_modules', 'q_proj', 'v_proj']
        with patch.object(sys, 'argv', argv):
            args = context['get_args']()
        self.assertEqual(args.use_lora, 'true')
        self.assertEqual(args.lora_rank, 16)
        self.assertEqual(args.lora_target_modules, ['q_proj', 'v_proj'])

    @unittest.skipUnless(importlib.util.find_spec('hyperpyyaml'), 'HyperPyYAML not installed')
    def test_yaml_lora_options_reach_real_factory(self):
        from hyperpyyaml import load_hyperpyyaml
        path = Path(__file__).resolve().parents[1] / 'configs/dreamon_cosyvoice_train.yaml'
        received = []
        def factory(**options):
            received.append(options.copy())
            options['gradient_checkpointing'] = False
            return build_training_model(**options)
        with patch('cosyvoice.llm.dreamon_training.build_training_model', new=factory), \
                patch('cosyvoice.llm.dreamon_training.load_cosyvoice_speech_components', return_value={}), \
                patch('cosyvoice.llm.dreamon_training.DreamOnSpeechLM.from_local_weights', return_value=lora_bridge(True)):
            with path.open(encoding='utf-8') as handle:
                config = load_hyperpyyaml(handle, overrides={'use_lora': True, 'lora_rank': 2,
                                                            'lora_alpha': 4, 'lora_lr': 0.002})
        self.assertTrue(config['llm'].bridge.use_lora)
        self.assertEqual(config['llm'].bridge.lora_config['rank'], 2)
        self.assertEqual(config['llm'].lora_lr, 0.002)
        self.assertEqual(received[0]['lora_target_modules'], ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    def test_zero_initialization_and_low_rank_formula(self):
        linear = nn.Linear(6, 4)
        x = torch.randn(2, 5, 6, requires_grad=True)
        layer = LoRALinear(linear, rank=2, alpha=4, dropout=0.0)
        torch.testing.assert_close(layer(x), linear(x))
        with torch.no_grad():
            layer.lora_B.weight.normal_()
        expected = torch.nn.functional.linear(x, linear.weight + 2 * layer.lora_B.weight @ layer.lora_A.weight, linear.bias)
        torch.testing.assert_close(layer(x), expected)
        layer(x).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_only_lora_and_projections_update_with_checkpointing(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                torch.manual_seed(45)
                bridge = lora_bridge(bf16)
                bridge.backbone.checkpointing = True
                trainer = trainer_for(bridge).train()
                before = {name: p.detach().clone() for name, p in trainer.named_parameters()}
                optimizer = torch.optim.AdamW(trainer.optimizer_parameter_groups(), lr=0.01)
                self.assertEqual([g['lr'] for g in optimizer.param_groups], [0.01, 0.002])
                self.assertEqual(optimizer.param_groups[1]['name'], 'dreamon_lora')
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    result = trainer(training_batch(), 'cpu')
                    result['loss'].backward()
                    for name, p in trainer.named_parameters():
                        if p.requires_grad:
                            self.assertEqual(p.dtype, torch.float32)
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                        else:
                            self.assertIsNone(p.grad, name)
                    optimizer.step()
                for name, p in trainer.named_parameters():
                    if not p.requires_grad:
                        self.assertTrue(torch.equal(before[name], p), name)
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in trainer.named_parameters() if '.lora_A.' in n))
                self.assertTrue(any(not torch.equal(before[n], p) for n, p in trainer.named_parameters() if '.lora_B.' in n))
                self.assertFalse(torch.equal(before['bridge.speech_in_proj.weight'], bridge.speech_in_proj.weight))
                self.assertFalse(torch.equal(before['bridge.speech_out_proj.weight'], bridge.speech_out_proj.weight))
                self.assertEqual(bridge.backbone.q_proj.base_layer.weight.dtype,
                                 torch.bfloat16 if bf16 else torch.float32)

    def test_full_checkpoint_automatically_restores_lora_for_generation(self):
        torch.manual_seed(1)
        base = lora_bridge()
        source = trainer_for(copy.deepcopy(base)).train()
        optimizer = torch.optim.AdamW(source.optimizer_parameter_groups(), lr=0.01)
        source(training_batch(), 'cpu')['loss'].backward()
        optimizer.step()
        source.eval()
        payload = source.training_checkpoint({'epoch': 0, 'step': 1})
        self.assertEqual(payload['format_version'], 4)
        self.assertEqual(payload['checkpoint_scope'], 'backbone_and_projections')
        self.assertIn('q_proj.base_layer.weight', payload['backbone_state_dict'])
        self.assertIn('q_proj.lora_B.weight', payload['backbone_state_dict'])
        restored = copy.deepcopy(base)
        with torch.no_grad():
            restored.backbone.q_proj.weight.zero_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epoch_0_whole.pt'
            torch.save(payload, path)
            restored.load_adapter_checkpoint(path)
        self.assertIsInstance(restored.backbone.q_proj, LoRALinear)
        self.assertFalse(restored.backbone.q_proj.training)
        self.assertFalse(any(p.requires_grad for p in restored.backbone.parameters()))
        inputs = (torch.tensor([[1, 2]]), torch.tensor([[1]]), torch.tensor([[-1, -1, -1]]))
        torch.testing.assert_close(restored(*inputs), source.bridge(*inputs))
        expected, _ = source.bridge.generate('ab', 'c', torch.tensor([[1]]), speech_tokens=4)
        actual, stats = restored.generate('ab', 'c', torch.tensor([[1]]), speech_tokens=4)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(stats['lora_config']['rank'], 2)
        self.assertIn('backbone_state_dict', restored.initial_adapter_checkpoint())
        # Resume preserves the requested trainability and exact adapter weights.
        resumed = trainer_for(copy.deepcopy(base))
        resumed.load_training_checkpoint(payload)
        self.assertTrue(resumed.bridge.backbone.q_proj.lora_B.weight.requires_grad)
        torch.testing.assert_close(resumed.bridge.backbone.q_proj.lora_B.weight,
                                   source.bridge.backbone.q_proj.lora_B.weight)

    def test_frozen_projection_checkpoint_initializes_lora(self):
        source = lora_bridge()
        source.speech_in_proj.weight.data.fill_(0.3)
        for version in (1, 2, 3):
            payload = source.initial_adapter_checkpoint()
            payload['format_version'] = version
            trainer = trainer_for(lora_bridge())
            trainer.load_training_checkpoint(payload)
            self.assertTrue(trainer.bridge.use_lora)
            torch.testing.assert_close(trainer.bridge.speech_in_proj.weight, source.speech_in_proj.weight)
            self.assertTrue(trainer.bridge.backbone.q_proj.lora_B.weight.eq(0).all())

    def test_bad_lora_checkpoint_is_atomic(self):
        source = trainer_for(lora_bridge())
        for fault in ('nan', 'missing', 'config'):
            with self.subTest(fault=fault):
                payload = source.training_checkpoint({'epoch': 0, 'step': 1})
                payload['state_dict']['speech_in_proj.weight'].zero_()
                if fault == 'nan':
                    payload['backbone_state_dict']['q_proj.lora_B.weight'][0, 0] = float('nan')
                elif fault == 'missing':
                    del payload['backbone_state_dict']['q_proj.lora_A.weight']
                else:
                    payload['lora_config']['target_modules'] = ['missing_proj']
                destination = lora_bridge()
                before = {name: p.detach().clone() for name, p in destination.named_parameters()}
                with self.assertRaises(ValueError):
                    destination.load_adapter_checkpoint(payload)
                self.assertIsNone(destination.lora_config)
                self.assertIsInstance(destination.backbone.q_proj, nn.Linear)
                for name, p in destination.named_parameters():
                    self.assertTrue(torch.equal(before[name], p), name)

    def test_validation_conflicts_and_no_duplicate_injection(self):
        with self.assertRaisesRegex(ValueError, 'requires freeze_dreamon'):
            trainer_for(lora_bridge(), freeze_dreamon=False)
        for kwargs in ({'rank': 0}, {'alpha': float('nan')}, {'dropout': 1.0},
                       {'target_modules': ['q_proj', 'q_proj']}):
            with self.assertRaises(ValueError):
                lora_config(**kwargs)
        with self.assertRaisesRegex(ValueError, 'not found'):
            prepare_lora(AttentionBackbone(), lora_config(2, target_modules=['missing']), 1)
        bridge = trainer_for(lora_bridge()).bridge
        count = len(list(bridge.parameters()))
        bridge.configure_training(True, True, 2, 4, 0.0)
        self.assertEqual(len(list(bridge.parameters())), count)
        with self.assertRaisesRegex(ValueError, 'differs'):
            bridge.configure_training(True, True, 3, 4, 0.0)
        bridge.configure_training(True, False)
        self.assertFalse(any(p.requires_grad for p in bridge.backbone.parameters()))
        self.assertEqual(bridge.initial_adapter_checkpoint()['format_version'], 4)


if __name__ == '__main__':
    unittest.main()
