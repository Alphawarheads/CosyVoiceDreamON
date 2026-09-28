"""Experimental DreamOn backbone with CosyVoice2 speech embeddings and head.

The two projections start untrained. Producing valid codec IDs does not imply
intelligible speech. This implements fixed-canvas denoising, not DreamOn's
text-vocabulary expansion/deletion algorithm. No original checkpoint is edited.
"""

import copy
from pathlib import Path

import torch
from torch import nn

from cosyvoice.llm.dreamon_lora import (DEFAULT_TARGETS, LoRALinear, install_lora,
                                        lora_config, prepare_lora)


def validate_speech_tokens(tokens, vocab_size=6561, allow_empty=False):
    """Reject invalid IDs before embedding or passing them to the audio backend."""
    if tokens.ndim != 2 or tokens.shape[0] != 1:
        raise ValueError("Speech tokens must have shape [1, length].")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("Speech tokens must be int32 or int64, not rounded floats.")
    if tokens.numel() == 0:
        if allow_empty:
            return
        raise ValueError("Empty speech tokens would invoke the original LLM in tts().")
    if tokens.min().item() < 0 or tokens.max().item() >= vocab_size:
        raise ValueError(f"Speech IDs must be in [0, {vocab_size - 1}].")


class DreamOnSpeechLM(nn.Module):
    """Single-utterance speech bridge; use generate(), then tts(source_speech_token=...).

    Native DreamOn text embeddings preserve its tokenizer/embedding pairing.
    CosyVoice speech and task embeddings are projected into DreamOn's space.
    The original DreamOn text LM head is not retained.
    """

    MASK = -1  # Canvas sentinel only; never sent to an embedding or the codec.

    def __init__(self, backbone, tokenizer, speech_embedding, speech_decoder,
                 task_embedding, mask_token_id, bos_token_id, eos_token_id,
                 speech_vocab_size=6561, seed=1986, max_sequence_tokens=2048):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.speech_embedding = speech_embedding
        self.speech_decoder = speech_decoder
        self.task_embedding = task_embedding
        self.speech_vocab_size = speech_vocab_size
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.seed = seed
        self.max_sequence_tokens = max_sequence_tokens
        self.adaptation_status = "untrained_random_projections"

        dream_embedding = self.backbone.get_input_embeddings()
        dream_size = dream_embedding.embedding_dim
        cosy_size = speech_embedding.embedding_dim
        if speech_embedding.num_embeddings != speech_vocab_size + 3:
            raise ValueError("Expected CosyVoice2 speech vocabulary plus 3 special IDs.")
        if (speech_decoder.in_features != cosy_size
                or speech_decoder.out_features != speech_vocab_size + 3):
            raise ValueError("CosyVoice speech embedding/head shapes disagree.")
        if task_embedding.num_embeddings != 2 or task_embedding.embedding_dim != cosy_size:
            raise ValueError("Expected CosyVoice2 SOS/TASK embedding of shape [2, hidden].")
        for token_id in (mask_token_id, bos_token_id, eos_token_id):
            if token_id is None or not 0 <= token_id < dream_embedding.num_embeddings:
                raise ValueError("DreamOn MASK/BOS/EOS ID is missing or out of range.")
        if not 1 <= max_sequence_tokens <= backbone.config.max_position_embeddings:
            raise ValueError("max_sequence_tokens exceeds the DreamOn context limit.")

        # CPU initialization is reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.speech_in_proj = nn.Linear(cosy_size, dream_size, bias=False)
            self.speech_out_proj = nn.Linear(dream_size, cosy_size, bias=False)
            nn.init.xavier_uniform_(self.speech_in_proj.weight)
            nn.init.xavier_uniform_(self.speech_out_proj.weight)
        # A later trainer can explicitly unfreeze more modules. The projections
        # remain trainable; forward() deliberately does not use inference_mode.
        for module in (self.backbone, self.speech_embedding,
                       self.speech_decoder, self.task_embedding):
            module.requires_grad_(False)
        self.freeze_dreamon = True
        self.lora_config = None
        self.use_lora = False
        # Keep a fine-tuned backbone in subsequent checkpoints even if re-frozen.
        self.backbone_checkpoint_required = False
        self.eval()

    def configure_training(self, freeze_dreamon=True, use_lora=False, lora_rank=16,
                           lora_alpha=32.0, lora_dropout=0.05, lora_target_modules=DEFAULT_TARGETS):
        """Call before constructing the optimizer/DDP wrapper.

        LoRA freezes the original core and trains only A/B plus speech projections.
        Disabling LoRA on an adapted model freezes its adapters but keeps their effect.
        """
        if not isinstance(freeze_dreamon, bool) or not isinstance(use_lora, bool):
            raise ValueError("freeze_dreamon and use_lora must be boolean values.")
        if use_lora and not freeze_dreamon:
            raise ValueError("LoRA requires freeze_dreamon=true; false selects full fine-tuning.")
        if use_lora:
            config = lora_config(lora_rank, lora_alpha, lora_dropout, lora_target_modules)
            if self.lora_config is not None and self.lora_config != config:
                raise ValueError("LoRA configuration differs from the installed adapters.")
            if self.lora_config is None:
                install_lora(self.backbone, prepare_lora(self.backbone, config, self.seed))
                self.lora_config = config
        self.freeze_dreamon = freeze_dreamon
        self.use_lora = use_lora
        self.backbone.requires_grad_(not freeze_dreamon)
        if not freeze_dreamon:
            self.backbone.float()
            self.backbone_checkpoint_required = True
        if use_lora:
            for layer in self.backbone.modules():
                if isinstance(layer, LoRALinear):
                    layer.lora_A.float().requires_grad_(True)
                    layer.lora_B.float().requires_grad_(True)
            self.backbone_checkpoint_required = True
        for module in (self.speech_in_proj, self.speech_out_proj):
            module.float().requires_grad_(True)
        for module in (self.speech_embedding, self.speech_decoder, self.task_embedding):
            module.requires_grad_(False)

    @staticmethod
    def copy_cosyvoice_components(cosyvoice_llm):
        """Copy only small speech modules, allowing the Qwen backbone to be freed."""
        return {
            "speech_embedding": copy.deepcopy(cosyvoice_llm.speech_embedding).cpu(),
            "speech_decoder": copy.deepcopy(cosyvoice_llm.llm_decoder).cpu(),
            "task_embedding": copy.deepcopy(cosyvoice_llm.llm_embedding).cpu(),
        }

    @classmethod
    def from_local_weights(cls, model_dir, components, device="cuda:0",
                           dtype=torch.bfloat16, **kwargs):
        from transformers import AutoModel, AutoTokenizer

        model_dir = str(Path(model_dir).expanduser().resolve(strict=True))
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
        )
        dream = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
            torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        bridge = cls(
            backbone=dream.get_decoder(), tokenizer=tokenizer, **components,
            mask_token_id=dream.config.mask_token_id,
            bos_token_id=dream.config.bos_token_id,
            eos_token_id=dream.config.eos_token_id, **kwargs,
        )
        # Only the backbone is held by bridge; the large text head can be freed.
        del dream
        return bridge.to(device=device, dtype=dtype).eval()

    def encode_text(self, text, prompt_text=""):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Provide a non-empty target text string.")
        if not isinstance(prompt_text, str):
            raise ValueError("prompt_text must be the reference audio transcript.")
        # Match CosyVoice's prompt/target ordering while using DreamOn's tokenizer.
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        ids += self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long,
                            device=self.speech_in_proj.weight.device)

    def forward_features(self, text_ids, prompt_speech_tokens, canvas):
        """Return hidden features aligned to each canvas position.

        DreamOn predicts position i using the hidden state at i-1 (see its
        generation_utils.py logits shift). Keep this convention in future losses.
        The visible prompt and resolved canvas tokens participate bidirectionally.
        """
        validate_speech_tokens(prompt_speech_tokens, self.speech_vocab_size, allow_empty=True)
        if canvas.ndim != 2 or canvas.shape[0] != 1 or canvas.shape[1] == 0:
            raise ValueError("Canvas must have shape [1, positive_length].")
        if canvas.dtype not in (torch.int32, torch.int64):
            raise ValueError("Canvas must contain integer IDs.")
        if canvas.min().item() < self.MASK or canvas.max().item() >= self.speech_vocab_size:
            raise ValueError("Canvas accepts only MASK=-1 or valid speech IDs.")
        if text_ids.ndim != 2 or text_ids.shape[0] != 1:
            raise ValueError("Text IDs must have shape [1, length].")

        device = self.speech_in_proj.weight.device
        text_ids = text_ids.to(device=device, dtype=torch.long)
        prompt_speech_tokens = prompt_speech_tokens.to(device=device, dtype=torch.long)
        canvas = canvas.to(device=device, dtype=torch.long)
        embed = self.backbone.get_input_embeddings()
        bos = embed.weight[self.bos_token_id].view(1, 1, -1)
        eos = embed.weight[self.eos_token_id].view(1, 1, -1)
        mask = embed.weight[self.mask_token_id].view(1, 1, -1)
        # Keep trainable projections in FP32 while the frozen 7B backbone may be
        # BF16. Explicit casts also make inference without autocast work.
        def project_speech(values):
            return self.speech_in_proj(values.to(self.speech_in_proj.weight.dtype)).to(embed.weight.dtype)

        task = project_speech(self.task_embedding.weight[1].view(1, 1, -1))
        prompt = project_speech(self.speech_embedding(prompt_speech_tokens))
        canvas_emb = project_speech(self.speech_embedding(canvas.clamp_min(0)))
        canvas_emb = torch.where(canvas.eq(self.MASK).unsqueeze(-1), mask, canvas_emb)
        prefix = torch.cat((bos, embed(text_ids), task, prompt), dim=1)
        inputs = torch.cat((prefix, canvas_emb, eos), dim=1)
        length = inputs.shape[1]
        if length > self.max_sequence_tokens:
            raise ValueError(f"Sequence has {length} tokens; limit is {self.max_sequence_tokens}. "
                             "Shorten the text, reference audio, or speech canvas.")
        # SDPA uses True=visible. No causal mask or KV cache is appropriate here.
        attention_mask = torch.ones((1, 1, length, length), dtype=torch.bool, device=device)
        hidden = self.backbone(
            inputs_embeds=inputs, attention_mask=attention_mask,
            position_ids=torch.arange(length, device=device).unsqueeze(0),
            use_cache=False, return_dict=True,
        ).last_hidden_state
        start = prefix.shape[1]
        aligned_hidden = hidden[:, start - 1:start + canvas.shape[1] - 1]
        return aligned_hidden

    def forward(self, text_ids, prompt_speech_tokens, canvas):
        aligned_hidden = self.forward_features(text_ids, prompt_speech_tokens, canvas)
        speech_hidden = self.speech_out_proj(aligned_hidden.to(self.speech_out_proj.weight.dtype))
        return self.speech_decoder(speech_hidden.to(self.speech_decoder.weight.dtype))

    @torch.inference_mode()
    def generate(self, text, prompt_text, prompt_speech_tokens, speech_tokens=100,
                 tokens_per_step=1, temperature=0.0, seed=None, progress=None):
        """Fill a fixed canvas in negative-entropy order; never emit control IDs."""
        if not isinstance(speech_tokens, int) or not 1 <= speech_tokens < self.max_sequence_tokens:
            raise ValueError("speech_tokens must be positive and below the context limit.")
        if not isinstance(tokens_per_step, int) or tokens_per_step < 1:
            raise ValueError("tokens_per_step must be a positive integer.")
        if not 0 <= temperature < float("inf"):
            raise ValueError("temperature must be finite and non-negative.")
        self.eval()
        text_ids = self.encode_text(text, prompt_text)
        device = self.speech_in_proj.weight.device
        canvas = torch.full((1, speech_tokens), self.MASK, dtype=torch.long, device=device)
        rng = torch.Generator(device=device).manual_seed(self.seed if seed is None else seed)
        history = []
        while canvas.eq(self.MASK).any().item():
            remaining = canvas[0].eq(self.MASK).nonzero(as_tuple=True)[0]
            all_logits = self(text_ids, prompt_speech_tokens, canvas)
            if not torch.isfinite(all_logits).all().item():
                raise FloatingPointError("Non-finite DreamOn speech logits. Try --math-sdpa "
                                         "or --dtype fp32; no token IDs were decoded.")
            # Slice before softmax: EOS/FILL/reserved classes cannot reach the codec.
            logits = all_logits[0, remaining, :self.speech_vocab_size].float()
            scaled = logits / temperature if temperature > 0 else logits
            log_probs = scaled.log_softmax(dim=-1)
            probs = log_probs.exp()
            if not torch.isfinite(log_probs).all().item():
                raise FloatingPointError("Non-finite sampling probabilities; increase temperature.")
            confidence = (probs * log_probs).sum(dim=-1)
            count = min(tokens_per_step, remaining.numel())
            selected = confidence.topk(count).indices
            if temperature == 0:
                predictions = logits[selected].argmax(dim=-1)
            else:
                predictions = torch.multinomial(probs[selected], 1, generator=rng).squeeze(-1)
            positions = remaining[selected]
            canvas[0, positions] = predictions
            step = {
                "step": len(history) + 1,
                "remaining_masks": int(remaining.numel() - count),
                "positions": positions.cpu().tolist(),
                "token_ids": predictions.cpu().tolist(),
                "mean_entropy": float(-confidence.mean().item()),
                "logit_min": float(logits.min().item()),
                "logit_max": float(logits.max().item()),
            }
            history.append(step)
            if progress is not None:
                progress(step)

        tokens = canvas.to(device="cpu", dtype=torch.int32)
        validate_speech_tokens(tokens, self.speech_vocab_size)
        return tokens, {
            "adaptation_status": self.adaptation_status,
            "generation_mode": "fixed_canvas_negative_entropy",
            "lora_config": self.lora_config,
            "logit_alignment": "hidden_at_i_minus_1_predicts_token_at_i",
            "text_tokenizer": "DreamOn",
            "text_tokens": text_ids.shape[1],
            "prompt_speech_tokens": prompt_speech_tokens.shape[1],
            "speech_tokens": speech_tokens,
            "unique_tokens": tokens.unique().numel(),
            "adjacent_repeat_ratio": (float(tokens[:, 1:].eq(tokens[:, :-1]).float().mean())
                                      if speech_tokens > 1 else 0.0),
            "all_logits_finite": True,
            "steps": history,
        }

    def checkpoint_trainable_prefixes(self):
        """Subclasses extending this must serialize and validate their extra state."""
        return ("speech_in_proj.", "speech_out_proj.", "backbone.", "duration_predictor.")

    def initial_adapter_checkpoint(self):
        """Export projections and, when needed, the complete DreamOn backbone."""
        allowed = self.checkpoint_trainable_prefixes()
        if any(p.requires_grad and not name.startswith(allowed) for name, p in self.named_parameters()):
            raise ValueError("Checkpoint cannot save additional unfrozen speech modules.")
        include_backbone = (self.backbone_checkpoint_required
                            or any(p.requires_grad for p in self.backbone.parameters()))
        payload = {
            "format_version": 4 if self.lora_config is not None else 3,
            "model_family": "dreamon_speech_adapter",
            "checkpoint_scope": "backbone_and_projections" if include_backbone else "projection_weights_only",
            "freeze_dreamon": self.freeze_dreamon,
            "adaptation_status": self.adaptation_status,
            "seed": self.seed,
            "speech_vocab_size": self.speech_vocab_size,
            "mask_token_id": self.mask_token_id,
            "logit_alignment": "previous_position",
            "cosyvoice_hidden_size": self.speech_in_proj.in_features,
            "dreamon_hidden_size": self.speech_in_proj.out_features,
            "state_dict": {
                "speech_in_proj.weight": self.speech_in_proj.weight.detach().cpu().clone(),
                "speech_out_proj.weight": self.speech_out_proj.weight.detach().cpu().clone(),
            },
        }
        if self.lora_config is not None:
            payload["lora_config"] = copy.deepcopy(self.lora_config)
            payload["use_lora"] = self.use_lora
        if include_backbone:
            payload["backbone_state_dict"] = {
                name: value.detach().to(device="cpu", copy=True)
                for name, value in self.backbone.state_dict().items()
            }
        if hasattr(self, "duration_predictor"):
            payload["bridge_format_version"] = payload["format_version"]
            payload["format_version"] = 5
            payload["route_a"] = {
                "duration_config": dict(self.duration_predictor.config),
                "speech_conditioning": "text_only",
                "duration_state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.duration_predictor.state_dict().items()
                },
            }
        return payload

    def load_adapter_checkpoint(self, checkpoint):
        """Load legacy projections or projections plus a fine-tuned backbone.

        Loading weights does not change the training freeze setting. Validate all
        tensors first, so an incompatible backbone cannot partly update a model.
        """
        payload = (torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                   if not isinstance(checkpoint, dict) else checkpoint)
        if isinstance(payload, dict) and payload.get("format_version") == 5:
            from cosyvoice.llm.dreamon_route_a import SpeechDurationPredictor
            route = payload.get("route_a")
            if (not isinstance(route, dict) or route.get("speech_conditioning") != "text_only"
                    or payload.get("bridge_format_version") not in (3, 4)):
                raise ValueError("Invalid Route A checkpoint metadata.")
            config = route.get("duration_config", {})
            if config.get("input_size") != self.backbone.get_input_embeddings().embedding_dim:
                raise ValueError("Duration input size differs from DreamOn.")
            duration = SpeechDurationPredictor(**config)
            state = route.get("duration_state_dict", {})
            expected = duration.state_dict()
            if set(state) != set(expected):
                raise ValueError("Invalid duration checkpoint keys.")
            for name, tensor in state.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                        or not tensor.is_floating_point() or not torch.isfinite(tensor).all().item()):
                    raise ValueError(f"Invalid duration tensor: {name}")
            if hasattr(self, "duration_predictor") and self.duration_predictor.config != config:
                raise ValueError("Duration configuration differs from training.")
            duration.load_state_dict(state, strict=True)
            base_payload = dict(payload)
            base_payload["format_version"] = payload["bridge_format_version"]
            # Loading the legacy portion also validates the full backbone and projections.
            result = self.load_adapter_checkpoint(base_payload)
            if hasattr(self, "duration_predictor"):
                if self.duration_predictor.config != config:
                    raise ValueError("Duration configuration differs from training.")
                self.duration_predictor.load_state_dict(state, strict=True)
            else:
                self.duration_predictor = duration.to(
                    device=self.speech_in_proj.weight.device, dtype=torch.float32).eval()
            self.speech_conditioning = "text_only"
            return payload
        if (not isinstance(payload, dict) or payload.get("format_version") not in (1, 2, 3, 4)
                or "state_dict" not in payload):
            raise ValueError("Expected a DreamOn speech adapter checkpoint, not llm.pt or DreamOn weights.")
        has_backbone = "backbone_state_dict" in payload
        if payload["format_version"] < 3 and has_backbone:
            raise ValueError("Backbone weights require checkpoint format version 3.")
        if payload["format_version"] >= 3:
            scope = "backbone_and_projections" if has_backbone else "projection_weights_only"
            if (payload.get("checkpoint_scope") != scope
                    or not isinstance(payload.get("freeze_dreamon"), bool)
                    or (not payload["freeze_dreamon"] and not has_backbone)):
                raise ValueError("Checkpoint scope/freeze_dreamon metadata is inconsistent.")
        pending_lora = {}
        loaded_lora_config = None
        if payload["format_version"] == 4:
            raw_config = payload.get("lora_config")
            if (not has_backbone or not isinstance(raw_config, dict)
                    or set(raw_config) != {"rank", "alpha", "dropout", "target_modules"}
                    or not isinstance(payload.get("use_lora"), bool)):
                raise ValueError("LoRA checkpoint requires full backbone weights and LoRA configuration.")
            loaded_lora_config = lora_config(**raw_config)
            if self.lora_config is not None and self.lora_config != loaded_lora_config:
                raise ValueError("Checkpoint LoRA configuration differs from the training configuration.")
            if self.lora_config is None:
                pending_lora = prepare_lora(self.backbone, loaded_lora_config, self.seed)
        elif "lora_config" in payload:
            raise ValueError("LoRA metadata requires checkpoint format version 4.")
        elif has_backbone and self.lora_config is not None:
            raise ValueError("Load a legacy full backbone before enabling LoRA, or initialize LoRA from a frozen projection checkpoint.")
        if payload["format_version"] >= 2:
            expected = {"model_family": "dreamon_speech_adapter",
                        "speech_vocab_size": self.speech_vocab_size,
                        "mask_token_id": self.mask_token_id, "logit_alignment": "previous_position"}
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f"Incompatible adapter {key}: {payload.get(key)!r}, expected {value!r}.")
        for key, expected in (("cosyvoice_hidden_size", self.speech_in_proj.in_features),
                              ("dreamon_hidden_size", self.speech_in_proj.out_features)):
            if payload.get(key) != expected:
                raise ValueError(f"Adapter {key} does not match the loaded base models.")
        targets = {"speech_in_proj.weight": self.speech_in_proj.weight,
                   "speech_out_proj.weight": self.speech_out_proj.weight}
        state = payload["state_dict"]
        if not isinstance(state, dict) or set(state) != set(targets):
            raise ValueError("Adapter checkpoint must contain exactly the two projection weights.")
        if has_backbone:
            backbone_state = payload["backbone_state_dict"]
            backbone_targets = dict(self.backbone.state_dict())
            # Validate virtual wrapped state before modifying the live architecture.
            for name, layer in pending_lora.items():
                prefix = name + "."
                for key in list(backbone_targets):
                    if key.startswith(prefix):
                        del backbone_targets[key]
                backbone_targets.update({prefix + key: value for key, value in layer.state_dict().items()})
            if not isinstance(backbone_state, dict) or set(backbone_state) != set(backbone_targets):
                raise ValueError("Backbone checkpoint keys do not match the loaded DreamOn model.")
            targets.update({f"backbone.{name}": value for name, value in backbone_targets.items()})
            state = {**state, **{f"backbone.{name}": value for name, value in backbone_state.items()}}
        # Validate everything before mutating any parameter or buffer.
        for name, target in targets.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.is_floating_point() != target.is_floating_point()
                    or not torch.isfinite(value).all().item()):
                raise ValueError(f"Invalid adapter tensor: {name}")
            if (value.is_floating_point() and value.numel()
                    and torch.finfo(value.dtype).max > torch.finfo(target.dtype).max
                    and value.abs().max().item() > torch.finfo(target.dtype).max):
                raise ValueError(f"Adapter tensor exceeds target dtype range: {name}")
        if pending_lora:
            install_lora(self.backbone, pending_lora, trainable=False)
            self.lora_config = loaded_lora_config
            # A full-training destination must retain its requested trainability.
            if not self.freeze_dreamon:
                self.backbone.requires_grad_(True)
        with torch.no_grad():
            for name, target in targets.items():
                target.copy_(state[name])
        if has_backbone:
            self.backbone_checkpoint_required = True
        self.adaptation_status = payload.get("adaptation_status", "loaded_adapter_checkpoint")
        return payload
