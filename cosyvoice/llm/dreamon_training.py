"""Masked speech-token training through the existing CosyVoice Executor/DDP loop.

By default only the projections train. freeze_dreamon=False also trains and
saves the DreamOn backbone. use_lora=True trains internal low-rank adapters
and projections while keeping the base weights frozen. CosyVoice modules stay frozen.
This is fixed-canvas masked CE, not the DreamOn expansion/deletion objective.
"""

import hashlib
import logging
import math
import os
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from cosyvoice.llm.dreamon_lora import DEFAULT_TARGETS
from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM, validate_speech_tokens


def load_cosyvoice_speech_components(model_dir, speech_vocab_size=6561):
    """Read llm.pt directly without constructing/loading a second transformer."""
    state = torch.load(Path(model_dir) / "llm.pt", map_location="cpu", weights_only=True, mmap=True)
    names = ("speech_embedding.weight", "llm_decoder.weight", "llm_decoder.bias", "llm_embedding.weight")
    if any(name not in state for name in names):
        raise ValueError("llm.pt does not contain the expected CosyVoice2 speech modules.")
    rows, hidden = state[names[0]].shape
    if rows != speech_vocab_size + 3:
        raise ValueError("The checkpoint does not use the CosyVoice2 speech vocabulary.")
    components = {"speech_embedding": nn.Embedding(rows, hidden),
                  "speech_decoder": nn.Linear(hidden, rows),
                  "task_embedding": nn.Embedding(2, hidden)}
    components["speech_embedding"].load_state_dict({"weight": state[names[0]]}, strict=True)
    components["speech_decoder"].load_state_dict(
        {"weight": state[names[1]], "bias": state[names[2]]}, strict=True)
    components["task_embedding"].load_state_dict({"weight": state[names[3]]}, strict=True)
    return components


def build_training_model(dreamon_model_dir, cosyvoice_model_dir, dtype="bf16",
                         gradient_checkpointing=True, max_sequence_tokens=2048,
                         seed=1986, freeze_dreamon=True, dreamon_lr=1e-5, **training_options):
    if dtype not in ("bf16", "fp32"):
        raise ValueError("Adapter training supports bf16 or fp32 backbone weights.")
    bridge = DreamOnSpeechLM.from_local_weights(
        dreamon_model_dir, load_cosyvoice_speech_components(cosyvoice_model_dir),
        device="cpu", dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype],
        max_sequence_tokens=max_sequence_tokens, seed=seed,
    )
    if gradient_checkpointing:
        bridge.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DreamOnSpeechTrainer(bridge, seed=seed, freeze_dreamon=freeze_dreamon,
                                 dreamon_lr=dreamon_lr, **training_options)
    counts = model.trainable_parameter_counts()
    total = sum(p.numel() for p in model.parameters())
    logging.info("DreamOn freeze_dreamon=%s use_lora=%s; trainable parameters: %s (%s/%s, %.3f%%)",
                 freeze_dreamon, model.bridge.use_lora, counts, sum(counts.values()), total,
                 100 * sum(counts.values()) / total)
    if model.bridge.use_lora:
        logging.info("LoRA config: %s; lr=%g; frozen backbone dtype=%s", model.bridge.lora_config,
                     model.lora_lr, model.bridge.backbone.get_input_embeddings().weight.dtype)
    model.base_model_paths = {"dreamon": str(Path(dreamon_model_dir).resolve()),
                              "cosyvoice": str(Path(cosyvoice_model_dir).resolve())}
    # Same initial projections on all ranks, independent corruption per rank.
    torch.manual_seed(seed + int(os.environ.get("RANK", "0")))
    random.seed(seed + int(os.environ.get("RANK", "0")))
    return model


class DreamOnSpeechTrainer(nn.Module):
    """Adapt CosyVoice's model(batch, device) -> loss_dict contract.

    Each variable-length example is processed separately, then losses are
    averaged per utterance. Padding is sliced away before attention or masking.
    Batches >1 work; batch_size=1 is the conservative memory default for 7B.
    """

    def __init__(self, bridge, mask_min=0.1, mask_max=1.0, full_mask_probability=0.25,
                 prompt_probability=0.5, max_prompt_fraction=0.3,
                 validation_mask_ratio=0.5, seed=1986, freeze_dreamon=True, dreamon_lr=1e-5,
                 use_lora=False, lora_rank=16, lora_alpha=32.0, lora_dropout=0.05,
                 lora_target_modules=DEFAULT_TARGETS, lora_lr=1e-4):
        super().__init__()
        if not 0 < mask_min <= mask_max <= 1:
            raise ValueError("Require 0 < mask_min <= mask_max <= 1.")
        if not 0 <= full_mask_probability <= 1 or not 0 <= prompt_probability <= 1:
            raise ValueError("Mask/prompt probabilities must lie in [0, 1].")
        if not 0 <= max_prompt_fraction < 1 or not 0 < validation_mask_ratio <= 1:
            raise ValueError("Invalid prompt fraction or validation mask ratio.")
        self.bridge = bridge
        if not math.isfinite(dreamon_lr) or dreamon_lr <= 0:
            raise ValueError("dreamon_lr must be finite and positive.")
        self.dreamon_lr = dreamon_lr
        if not math.isfinite(lora_lr) or lora_lr <= 0:
            raise ValueError("lora_lr must be finite and positive.")
        self.lora_lr = lora_lr
        self.bridge.configure_training(freeze_dreamon, use_lora, lora_rank, lora_alpha,
                                       lora_dropout, lora_target_modules)
        self.mask_min, self.mask_max = mask_min, mask_max
        self.full_mask_probability = full_mask_probability
        self.prompt_probability, self.max_prompt_fraction = prompt_probability, max_prompt_fraction
        self.validation_mask_ratio = validation_mask_ratio
        self.seed = seed
        self.base_model_paths = {}

    def trainable_parameter_counts(self):
        return {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                for name, module in self.bridge.named_children()}

    def optimizer_parameter_groups(self):
        """Use the configured optimizer LR for projections and a separate core LR."""
        groups = [{"params": list(self.bridge.speech_in_proj.parameters())
                             + list(self.bridge.speech_out_proj.parameters()),
                   "name": "speech_projections"}]
        backbone = [p for p in self.bridge.backbone.parameters() if p.requires_grad]
        if backbone:
            groups.append({"params": backbone,
                           "lr": self.lora_lr if self.bridge.use_lora else self.dreamon_lr,
                           "name": "dreamon_lora" if self.bridge.use_lora else "dreamon"})
        selected = {id(p) for group in groups for p in group["params"]}
        if selected != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("Optimizer groups do not cover exactly the trainable parameters.")
        return groups

    def _corrupt(self, speech, utt):
        """Split an optional visible prefix, then mask targets without leakage."""
        # CPU draws keep validation invariant under rank, device and batch order.
        rng = None
        if not self.training:
            digest = hashlib.sha256(f"{self.seed}:{utt}".encode("utf-8")).digest()
            rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        prefix_length = 0
        if torch.rand((), generator=rng).item() < self.prompt_probability:
            limit = int(speech.shape[1] * self.max_prompt_fraction)
            if limit:
                prefix_length = int(torch.randint(1, limit + 1, (), generator=rng))
        # Text describes the entire utterance, including this visible audio prefix;
        # inference likewise concatenates reference transcript and target text.
        prompt = speech[:, :prefix_length]
        target = speech[:, prefix_length:]
        if not self.training:
            rate = self.validation_mask_ratio
        elif torch.rand((), generator=rng).item() < self.full_mask_probability:
            rate = 1.0
        else:
            rate = self.mask_min + (self.mask_max - self.mask_min) * torch.rand((), generator=rng).item()
        count = max(1, round(target.numel() * rate))
        positions = torch.randperm(target.numel(), generator=rng)[:count]
        mask = torch.zeros(target.shape, dtype=torch.bool)
        mask[0, positions] = True
        mask = mask.to(speech.device)
        canvas = target.clone().masked_fill(mask, self.bridge.MASK)
        return prompt, target, canvas, mask

    def forward(self, batch, device):
        required = ("utts", "text_token", "text_token_len", "speech_token", "speech_token_len")
        if any(key not in batch for key in required):
            raise ValueError("DreamOn training needs utterance IDs, DreamOn text IDs and offline speech tokens/lengths.")
        if batch.get("text_tokenizer") != "dreamon":
            raise ValueError("Use the DreamOn data pipeline; CosyVoice text IDs cannot be reused.")
        device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        if self.bridge.speech_in_proj.weight.device != device:
            raise ValueError("Move the training model to the requested device before forward().")
        batch_size = len(batch["utts"])
        if batch_size == 0:
            raise ValueError("Empty training batch.")
        for key in ("text_token", "speech_token"):
            if batch[key].ndim != 2 or batch[key].shape[0] != batch_size:
                raise ValueError(f"Invalid batch shape: {key}")
            if batch[key].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{key} must contain integer token IDs.")
        for key in ("text_token_len", "speech_token_len"):
            if batch[key].shape != (batch_size,) or batch[key].dtype not in (torch.int32, torch.int64):
                raise ValueError(f"Invalid length tensor: {key}")

        losses, accuracies, fractions = [], [], []
        for index, utt in enumerate(batch["utts"]):
            text_length = int(batch["text_token_len"][index])
            speech_length = int(batch["speech_token_len"][index])
            if (not 0 < text_length <= batch["text_token"].shape[1]
                    or not 0 < speech_length <= batch["speech_token"].shape[1]):
                raise ValueError(f"Invalid text/speech length for {utt}.")
            text = batch["text_token"][index:index + 1, :text_length].to(device)
            speech = batch["speech_token"][index:index + 1, :speech_length].to(device)
            validate_speech_tokens(speech, self.bridge.speech_vocab_size)
            prompt, target, canvas, loss_mask = self._corrupt(speech, utt)
            logits = self.bridge(text, prompt, canvas)[..., :self.bridge.speech_vocab_size].float()
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"Non-finite speech logits while training {utt}.")
            # Only corrupted target positions contribute; no prompt or padding loss.
            selected_logits, labels = logits[loss_mask], target[loss_mask].long()
            losses.append(F.cross_entropy(selected_logits, labels))
            accuracies.append(selected_logits.argmax(-1).eq(labels).float().mean())
            fractions.append(loss_mask.float().mean())
        return {"loss": torch.stack(losses).mean(), "acc": torch.stack(accuracies).mean(),
                "mask_fraction": torch.stack(fractions).mean()}

    def training_checkpoint(self, info):
        payload = self.bridge.initial_adapter_checkpoint()
        if info["epoch"] >= 0:
            payload["adaptation_status"] = "training_checkpoint"
        payload.update(epoch=info["epoch"], step=info["step"],
                       base_model_paths=self.base_model_paths,
                       training_objective="fixed_canvas_masked_speech_ce")
        payload["training_options"] = {
            "mask_min": self.mask_min, "mask_max": self.mask_max,
            "full_mask_probability": self.full_mask_probability,
            "prompt_probability": self.prompt_probability,
            "max_prompt_fraction": self.max_prompt_fraction,
            "validation_mask_ratio": self.validation_mask_ratio,
            "seed": self.seed, "max_sequence_tokens": self.bridge.max_sequence_tokens,
            "freeze_dreamon": self.bridge.freeze_dreamon, "dreamon_lr": self.dreamon_lr,
            "use_lora": self.bridge.use_lora, "lora_config": self.bridge.lora_config, "lora_lr": self.lora_lr,
        }
        return payload

    def load_training_checkpoint(self, path):
        # As with the original DDP .pt format, optimizer state is not included.
        return self.bridge.load_adapter_checkpoint(path)
