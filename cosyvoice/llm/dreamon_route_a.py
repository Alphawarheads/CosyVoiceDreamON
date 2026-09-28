"""Route A: independently learned duration, fixed-canvas speech denoising.

The duration branch sees target text and optionally ANOTHER utterance's rate.
Ground-truth target length is a loss label only, never a predictor input.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from cosyvoice.llm.dreamon_training import DreamOnSpeechTrainer, load_cosyvoice_speech_components
from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM
import os
import random


class SpeechDurationPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, initial_rate=4.0):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or not math.isfinite(initial_rate) or initial_rate <= 0:
            raise ValueError("Invalid duration architecture.")
        self.config = dict(input_size=input_size, hidden_size=hidden_size, initial_rate=initial_rate)
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        # mean/max contextual text pooling, log text count, reference log rate, availability.
        self.output = nn.Sequential(nn.Linear(hidden_size * 2 + 3, hidden_size), nn.SiLU(),
                                    nn.Linear(hidden_size, 1))
        nn.init.normal_(self.output[-1].weight, std=0.001)
        nn.init.constant_(self.output[-1].bias, math.log(initial_rate))

    def forward(self, text_embeddings, reference_rate=None):
        if text_embeddings.ndim != 3 or text_embeddings.shape[0] != 1 or text_embeddings.shape[1] < 1:
            raise ValueError("Duration prediction expects one non-empty text sequence.")
        # Keep this inexpensive branch FP32 even during speech-model AMP.
        with torch.autocast(device_type=text_embeddings.device.type, enabled=False):
            x = self.input_proj(self.input_norm(text_embeddings.detach().float()))
            x = x + self.conv2(F.silu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
            count = x.shape[1]
            if reference_rate is None:
                log_rate, available = 0.0, 0.0
            else:
                if not math.isfinite(float(reference_rate)) or reference_rate <= 0:
                    raise ValueError("Reference rate must be finite and positive.")
                log_rate, available = math.log(reference_rate), 1.0
            extra = x.new_tensor([[math.log(count), log_rate, available]])
            features = torch.cat((x.mean(1), x.amax(1), extra), dim=-1)
            return self.output(features).squeeze() + math.log(count)


@torch.inference_mode()
def predict_speech_length(bridge, text, prompt_text="", prompt_speech_tokens=0, maximum=750):
    if not hasattr(bridge, "duration_predictor"):
        raise ValueError("Learned length requires a Route A checkpoint with a duration predictor.")
    ids = bridge.encode_text(text)
    reference_rate = None
    if prompt_text and prompt_speech_tokens > 0:
        reference_count = len(bridge.tokenizer.encode(prompt_text, add_special_tokens=False))
        if reference_count:
            reference_rate = prompt_speech_tokens / reference_count
    embedding = bridge.backbone.get_input_embeddings()(ids)
    log_length = bridge.duration_predictor(embedding, reference_rate)
    if not torch.isfinite(log_length).item():
        raise FloatingPointError("Non-finite learned duration.")
    # Route A speech conditioning is text-only; reference audio still conditions Flow/HiFT.
    context_limit = bridge.max_sequence_tokens - ids.numel() - 3
    limit = min(maximum, context_limit)
    if limit < 1:
        raise ValueError("Target text leaves no room for speech in the context.")
    raw_log = float(log_length)
    value = math.exp(min(30.0, max(-30.0, raw_log)))
    rounded = max(1, round(value))
    length = min(rounded, limit)
    return length, {
        "length_mode": "learned", "predicted_tokens_float": value,
        "length_clipped": rounded > limit or value < 1,
        "maximum_allowed_tokens": limit, "reference_rate_used": reference_rate,
        "target_audio_used": False,
    }


class RouteATrainer(DreamOnSpeechTrainer):
    def optimizer_parameter_groups(self):
        # Parent intentionally rejects unknown trainable modules; construct explicit groups here.
        groups = [
            {"params": list(self.bridge.speech_in_proj.parameters()) +
                       list(self.bridge.speech_out_proj.parameters()), "name": "speech_projections"},
            {"params": [p for p in self.bridge.backbone.parameters() if p.requires_grad],
             "lr": self.lora_lr, "name": "dreamon_lora"},
            {"params": list(self.bridge.duration_predictor.parameters()),
             "lr": self.duration_lr, "name": "duration_predictor"},
        ]
        chosen = [id(p) for g in groups for p in g["params"]]
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if len(chosen) != len(set(chosen)) or set(chosen) != expected:
            raise ValueError("Route A optimizer coverage mismatch.")
        return groups

    def forward(self, batch, device):
        losses = super().forward(batch, device)
        duration_losses, relative_errors, absolute_errors = [], [], []
        for index in range(len(batch["utts"])):
            ntext = int(batch["text_token_len"][index])
            target_length = int(batch["speech_token_len"][index])
            ids = batch["text_token"][index:index + 1, :ntext].to(self.bridge.speech_in_proj.weight.device)
            rate = batch["reference_rate"][index]
            if self.training and torch.rand(()).item() < self.reference_dropout:
                rate = None
            embeddings = self.bridge.backbone.get_input_embeddings()(ids)
            log_n = self.bridge.duration_predictor(embeddings, rate)
            label = log_n.new_tensor(math.log(target_length))
            duration_losses.append(F.smooth_l1_loss(log_n, label))
            predicted = log_n.detach().clamp(-20, 20).exp()
            absolute_errors.append((predicted - target_length).abs())
            relative_errors.append((predicted / target_length - 1).abs())
        token_ce = losses["loss"]
        duration_loss = torch.stack(duration_losses).mean()
        losses.update(token_ce=token_ce, duration_loss=duration_loss,
                      duration_relative_error=torch.stack(relative_errors).mean(),
                      duration_mae_tokens=torch.stack(absolute_errors).mean(),
                      loss=token_ce + self.duration_loss_weight * duration_loss)
        return losses

    def training_checkpoint(self, info):
        payload = super().training_checkpoint(info)
        payload["training_objective"] = "route_a_fixed_canvas_ce_plus_log_duration"
        payload["training_options"].update(
            duration_lr=self.duration_lr, duration_loss_weight=self.duration_loss_weight,
            reference_dropout=self.reference_dropout, speech_conditioning="text_only",
            duration_conditioning="text_and_other_same_speaker_utterance_rate")
        return payload


def build_route_a_model(dreamon_model_dir, cosyvoice_model_dir, duration_lr=1e-3,
                        duration_loss_weight=0.1, reference_dropout=0.25,
                        initial_rate=4.0, **kwargs):
    for value in (duration_lr, duration_loss_weight):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Duration LR and loss weight must be positive.")
    if not 0 <= reference_dropout <= 1:
        raise ValueError("Invalid reference dropout.")
    if kwargs.get("prompt_probability", 0.0) != 0:
        raise ValueError("Route A uses full target text/speech without a same-utterance prefix.")
    kwargs["prompt_probability"] = 0.0
    dtype = kwargs.pop("dtype", "bf16")
    checkpointing = kwargs.pop("gradient_checkpointing", True)
    maximum = kwargs.pop("max_sequence_tokens", 2048)
    seed = kwargs.get("seed", 1986)
    if dtype != "bf16":
        raise ValueError("Route A training uses BF16 frozen backbone and FP32 trainables.")
    if not kwargs.get("freeze_dreamon", True) or not kwargs.get("use_lora", True):
        raise ValueError("Route A currently supports a frozen backbone with LoRA.")
    kwargs.update(freeze_dreamon=True, use_lora=True)
    bridge = DreamOnSpeechLM.from_local_weights(
        dreamon_model_dir, load_cosyvoice_speech_components(cosyvoice_model_dir),
        device="cpu", dtype=torch.bfloat16, max_sequence_tokens=maximum, seed=seed)
    if checkpointing:
        bridge.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = RouteATrainer(bridge, **kwargs)
    model.base_model_paths = {"dreamon": str(dreamon_model_dir), "cosyvoice": str(cosyvoice_model_dir)}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(kwargs.get("seed", 1986))
        model.bridge.duration_predictor = SpeechDurationPredictor(
            model.bridge.backbone.get_input_embeddings().embedding_dim,
            initial_rate=initial_rate).float()
    model.duration_lr = duration_lr
    model.duration_loss_weight = duration_loss_weight
    model.reference_dropout = reference_dropout
    model.bridge.speech_conditioning = "text_only"
    torch.manual_seed(seed + int(os.environ.get("RANK", "0")))
    random.seed(seed + int(os.environ.get("RANK", "0")))
    return model
