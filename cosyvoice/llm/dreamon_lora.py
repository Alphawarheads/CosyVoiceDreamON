"""Standard unmerged LoRA for DreamOn linear layers, using only PyTorch.

Base weights retain their loading dtype. FP32 A/B parameters use the caller's
AMP context. Full checkpoints keep base weights plus adapters without merging.
"""

import math

import torch
from torch import nn


DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def lora_config(rank=16, alpha=32.0, dropout=0.05, target_modules=DEFAULT_TARGETS):
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank must be a positive integer.")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive.")
    if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if (not isinstance(target_modules, (list, tuple)) or not target_modules
            or any(not isinstance(name, str) or not name or "." in name for name in target_modules)
            or len(set(target_modules)) != len(target_modules)):
        raise ValueError("LoRA target_modules must be unique non-empty linear-layer leaf names.")
    return {"rank": rank, "alpha": float(alpha), "dropout": float(dropout),
            "target_modules": sorted(target_modules)}


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base_layer, nn.Linear) or rank > min(base_layer.weight.shape):
            raise ValueError("LoRA needs a Linear layer with rank <= both weight dimensions.")
        self.base_layer = base_layer
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(base_layer.weight.device)
        self.lora_B.to(base_layer.weight.device)
        self.train(base_layer.training)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, inputs):
        original = self.base_layer(inputs)
        update = self.lora_B(self.lora_A(self.dropout(inputs.to(self.lora_A.weight.dtype))))
        return original + (update * self.scaling).to(original.dtype)


def prepare_lora(backbone, config, seed):
    """Build replacements without changing the backbone or its gradient flags."""
    layers = {name: layer for name, layer in backbone.named_modules()
              if name.rsplit(".", 1)[-1] in config["target_modules"]}
    found = {name.rsplit(".", 1)[-1] for name in layers}
    if found != set(config["target_modules"]):
        raise ValueError(f"LoRA target modules not found: {sorted(set(config['target_modules']) - found)}")
    if any(not isinstance(layer, nn.Linear) for layer in layers.values()):
        raise ValueError("LoRA targets must be plain Linear layers; adapters cannot be nested.")
    if any(config["rank"] > min(layer.weight.shape) for layer in layers.values()):
        raise ValueError("LoRA rank exceeds a target layer dimension.")
    # CPU initialization remains identical across DDP ranks and preserves RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return {name: LoRALinear(layer, config["rank"], config["alpha"], config["dropout"])
                for name, layer in layers.items()}


def install_lora(backbone, replacements, trainable=False):
    for name, layer in replacements.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        layer.base_layer.requires_grad_(False)
        layer.lora_A.requires_grad_(trainable)
        layer.lora_B.requires_grad_(trainable)
        setattr(parent, leaf, layer)
