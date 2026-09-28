"""Route B0: speech denoising with native canvas expansion/deletion.

No duration/rate predictor. The 3-way action distribution is factorized from
the conditional CosyVoice speech vocabulary; this differs from the original
DreamOn single augmented-vocabulary head and is an explicit experimental choice.
"""
import hashlib
import math
import os
import random
import torch
from torch import nn
from torch.nn import functional as F
from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM, validate_speech_tokens
from cosyvoice.llm.dreamon_training import DreamOnSpeechTrainer, load_cosyvoice_speech_components

FILL, EXPAND, DELETE = 0, 1, 2


def corrupt_speech(tokens, mask_ratio, rng, max_canvas, merge_rounds=None, surplus_count=None):
    """Mask true tokens, merge hidden spans, append removable surplus slots.

    Each item retains its latent true span solely to construct supervision.
    Span lengths/labels are NEVER inputs to the backbone or generator.
    Multi-round merges provide compressed canvases, not just N/2..N examples.
    """
    values = tokens.reshape(-1).tolist()
    if not values or not 0 < mask_ratio <= 1 or max_canvas < len(values):
        raise ValueError("Invalid corruption input/context.")
    masked = set(rng.sample(range(len(values)), max(1, round(len(values) * mask_ratio))))
    units = [(v if i not in masked else -1, (v,)) for i, v in enumerate(values)]
    # Include unedited denoising as an anchor, plus 1..4 compression levels.
    rounds = rng.choice([0, 0, 1, 2, 3, 4]) if merge_rounds is None else merge_rounds
    if not isinstance(rounds, int) or not 0 <= rounds <= 4:
        raise ValueError("merge_rounds must be in 0..4.")
    probability = rng.choice([0.5, 1.0])
    for _ in range(rounds):
        merged, i = [], 0
        while i < len(units):
            if (i + 1 < len(units) and units[i][0] == units[i+1][0] == -1
                    and rng.random() < probability):
                merged.append((-1, units[i][1] + units[i+1][1])); i += 2
            else:
                merged.append(units[i]); i += 1
        units = merged
    surplus_limit = min(max_canvas - len(units), 64, max(1, round(len(units)*0.3)))
    surplus = rng.randint(0, surplus_limit) if surplus_count is None else surplus_count
    if not isinstance(surplus, int) or not 0 <= surplus <= max_canvas - len(units):
        raise ValueError("Invalid surplus count.")
    units += [(-1, ())] * surplus
    canvas = torch.tensor([[u[0] for u in units]], dtype=torch.long, device=tokens.device)
    actions = torch.tensor([[(-100 if u[0] != -1 else
                             DELETE if len(u[1]) == 0 else EXPAND if len(u[1]) > 1 else FILL)
                             for u in units]], device=tokens.device)
    labels = torch.tensor([[(u[1][0] if u[0] == -1 and len(u[1]) == 1 else -100)
                            for u in units]], device=tokens.device)
    return canvas, actions, labels, dict(
        original_tokens=len(values), original_mask_fraction=len(masked)/len(values),
        canvas_tokens=len(units), merge_rounds=rounds,
        latent_span_lengths=[len(u[1]) for u in units])


class EditGenerationError(RuntimeError):
    def __init__(self, reason, report):
        super().__init__(reason)
        self.report = dict(report, status="failed", failure_reason=reason)


class DynamicSpeechLM(DreamOnSpeechLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden = self.speech_in_proj.out_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.edit_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.edit_head.float()
        self.speech_conditioning = "text_only"

    def configure_training(self, *args, **kwargs):
        super().configure_training(*args, **kwargs)
        self.edit_head.float().requires_grad_(True)

    def edit_forward(self, text_ids, canvas):
        prompt = canvas[:, :0]
        hidden = self.forward_features(text_ids, prompt, canvas)
        speech_hidden = self.speech_out_proj(hidden.to(self.speech_out_proj.weight.dtype))
        speech = self.speech_decoder(speech_hidden.to(self.speech_decoder.weight.dtype))
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            action = self.edit_head(hidden.float())
        return speech[..., :self.speech_vocab_size], action

    @torch.inference_mode()
    def generate_dynamic(self, text, initial_masks=64, maximum=750, max_steps=2048,
                         temperature=0.8, action_temperature=0.8, seed=None):
        if (not isinstance(initial_masks, int) or initial_masks < 1 or
                not isinstance(max_steps, int) or max_steps < 1 or
                not isinstance(maximum, int) or maximum < initial_masks):
            raise ValueError("Invalid initial mask count, maximum or step budget.")
        if any(not math.isfinite(t) or t < 0 for t in [temperature, action_temperature]):
            raise ValueError("Temperatures must be finite and nonnegative.")
        self.eval()
        ids = self.encode_text(text)
        limit = min(maximum, self.max_sequence_tokens - ids.numel() - 3)
        if initial_masks > limit:
            raise ValueError("Initial canvas exceeds context/token cap.")
        device = ids.device
        rng = torch.Generator(device=device).manual_seed(self.seed if seed is None else seed)
        canvas = torch.full((1, initial_masks), self.MASK, dtype=torch.long, device=device)
        report = dict(status="running", length_mode="dynamic", target_audio_used=False,
                      duration_predictor_used=False, initial_masks=initial_masks, maximum=limit,
                      temperature=temperature, action_temperature=action_temperature,
                      expand_count=0, delete_count=0, fill_count=0, steps=[])
        seen = {}
        for step in range(max_steps):
            positions = canvas[0].eq(self.MASK).nonzero().flatten()
            if positions.numel() == 0:
                break
            # With deterministic actions, returning to an identical canvas without
            # emitting speech is an exact loop, not merely slow convergence.
            state_key = tuple(canvas[0].tolist())
            if action_temperature == 0 and state_key in seen:
                report["cycle_first_seen_step"] = seen[state_key]
                raise EditGenerationError("deterministic_edit_cycle", report)
            seen[state_key] = step
            speech, action = self.edit_forward(ids, canvas)
            s, a = speech[0, positions].float(), action[0, positions].float()
            if not torch.isfinite(s).all() or not torch.isfinite(a).all():
                raise EditGenerationError("nonfinite_logits", report)
            # Select the most certain ACTION first. Using joint speech-token
            # confidence would systematically prefer controls over 6561-way speech.
            ap = (a / (action_temperature or 1.)).softmax(-1)
            entropy = -(ap * ap.clamp_min(1e-20).log()).sum(-1)
            j = int(entropy.argmin())
            pos = int(positions[j])
            act = (int(torch.multinomial(ap[j], 1, generator=rng)) if action_temperature
                   else int(a[j].argmax()))
            before = canvas.shape[1]
            if act == EXPAND:
                if before >= limit:
                    raise EditGenerationError("canvas_cap_reached", report)
                replacement = canvas.new_full((1, 2), self.MASK)
                report["expand_count"] += 1
            elif act == DELETE:
                replacement = canvas[:, :0]
                report["delete_count"] += 1
            else:
                token = (int(torch.multinomial((s[j]/temperature).softmax(-1), 1, generator=rng))
                         if temperature else int(s[j].argmax()))
                replacement = canvas.new_tensor([[token]])
                report["fill_count"] += 1
            canvas = torch.cat([canvas[:, :pos], replacement, canvas[:, pos+1:]], dim=1)
            report["steps"].append(dict(step=step+1, position=pos, action=["fill","expand","delete"][act],
                                       length_before=before, length_after=canvas.shape[1],
                                       action_probability=float(ap[j,act]),
                                       remaining_masks=int(canvas.eq(self.MASK).sum())))
            expected_length = initial_masks + report["expand_count"] - report["delete_count"]
            if canvas.shape[1] != expected_length:
                raise AssertionError("Edit length conservation failed.")
            if canvas.numel() == 0:
                raise EditGenerationError("empty_output", report)
        if canvas.eq(self.MASK).any():
            raise EditGenerationError("edit_step_budget_exhausted", report)
        validate_speech_tokens(canvas, self.speech_vocab_size)
        report.update(status="complete", forward_passes=len(report["steps"]),
                      speech_tokens=canvas.numel(), unique_tokens=canvas.unique().numel(),
                      adjacent_repeat_ratio=float(canvas[:,1:].eq(canvas[:,:-1]).float().mean())
                      if canvas.numel()>1 else 0.)
        return canvas.cpu().int(), report

    def checkpoint_trainable_prefixes(self):
        return super().checkpoint_trainable_prefixes() + ("edit_head.",)

    def initial_adapter_checkpoint(self):
        payload = super().initial_adapter_checkpoint()
        if "route_a" in payload:
            raise ValueError("B cannot contain a duration predictor.")
        payload["bridge_format_version"] = payload["format_version"]
        payload["format_version"] = 6
        payload["route_b"] = dict(version="b0_factorized_edit", speech_conditioning="text_only",
            actions=["fill", "expand", "delete"], loss="joint_action_plus_conditional_speech_nll",
            corruption="masked_spans_0_to_4_binary_merges_plus_tail_surplus",
            implementation_revision="b0.1_cycle_diagnostics_weighted_metrics",
            default_action_temperature=0.8,
            edit_state_dict={k:v.detach().cpu().clone() for k,v in self.edit_head.state_dict().items()})
        return payload

    def load_adapter_checkpoint(self, checkpoint):
        payload = (torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                   if not isinstance(checkpoint, dict) else checkpoint)
        route = payload.get("route_b", {})
        if (payload.get("format_version") != 6 or payload.get("bridge_format_version") not in (3,4)
                or route.get("version") != "b0_factorized_edit"
                or route.get("actions") != ["fill","expand","delete"]
                or route.get("speech_conditioning") != "text_only" or "route_a" in payload):
            raise ValueError("Expected a compatible Route B checkpoint.")
        state = route.get("edit_state_dict", {})
        expected = self.edit_head.state_dict()
        if set(state) != set(expected):
            raise ValueError("Invalid edit head checkpoint keys.")
        for k,v in state.items():
            if (not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                    or not v.is_floating_point() or not torch.isfinite(v).all()):
                raise ValueError("Invalid edit head tensor: " + k)
        base = dict(payload, format_version=payload["bridge_format_version"])
        super().load_adapter_checkpoint(base)
        self.edit_head.float().load_state_dict(state, strict=True)
        self.speech_conditioning = "text_only"
        return payload


class RouteBTrainer(DreamOnSpeechTrainer):
    def optimizer_parameter_groups(self):
        groups = [
            dict(params=list(self.bridge.speech_in_proj.parameters())+list(self.bridge.speech_out_proj.parameters()),
                 name="speech_projections"),
            dict(params=[p for p in self.bridge.backbone.parameters() if p.requires_grad],
                 name="dreamon_lora", lr=self.lora_lr),
            dict(params=list(self.bridge.edit_head.parameters()), name="edit_head", lr=self.lora_lr)]
        ids = [id(p) for g in groups for p in g["params"]]
        if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("Route B optimizer coverage mismatch.")
        return groups

    def forward(self, batch, device):
        if batch.get("text_tokenizer") != "dreamon" or not batch["utts"]:
            raise ValueError("Expected nonempty DreamOn token batch.")
        results = []
        for i, utt in enumerate(batch["utts"]):
            nt, ns = int(batch["text_token_len"][i]), int(batch["speech_token_len"][i])
            if not 0 < nt <= batch["text_token"].shape[1] or not 0 < ns <= batch["speech_token"].shape[1]:
                raise ValueError("Invalid text/speech length.")
            dev = self.bridge.speech_in_proj.weight.device
            ids = batch["text_token"][i:i+1,:nt].to(dev)
            speech = batch["speech_token"][i:i+1,:ns].to(dev)
            validate_speech_tokens(speech, self.bridge.speech_vocab_size)
            if self.training:
                rng = random
                ratio = 1. if rng.random() < self.full_mask_probability else rng.uniform(self.mask_min,self.mask_max)
            else:
                digest = hashlib.sha256(f"{self.seed}:{utt}".encode()).digest()
                rng = random.Random(int.from_bytes(digest[:8], "little"))
                ratio = self.validation_mask_ratio
            kwargs = (dict(merge_rounds=0, surplus_count=0)
                      if not self.training and not getattr(self, "validation_edits", True) else {})
            canvas, actions, labels, info = corrupt_speech(
                speech, ratio, rng, self.bridge.max_sequence_tokens-nt-3, **kwargs)
            s, a = self.bridge.edit_forward(ids, canvas)
            masked, fills = actions.ne(-100), labels.ne(-100)
            nmask = masked.sum()
            action_sum = F.cross_entropy(a[masked].float(), actions[masked], reduction="sum")
            # Zero keeps the projection branch in DDP's graph even in an all-control example.
            token_sum = (F.cross_entropy(s[fills].float(), labels[fills], reduction="sum")
                         if fills.any() else s.sum()*0.)
            # Exact NLL of factorization p(s)=p(FILL)*p(s|FILL); no target-length loss.
            loss = (action_sum+token_sum)/nmask
            action_pred = a.argmax(-1)
            token_correct = s.argmax(-1)[fills].eq(labels[fills]).sum().float()
            record = dict(loss=loss, token_ce=token_sum/fills.sum().clamp_min(1),
                          action_ce=action_sum/nmask,
                          acc=token_correct/fills.sum().clamp_min(1),
                          action_acc=action_pred[masked].eq(actions[masked]).float().mean(),
                          mask_fraction=loss.new_tensor(info["original_mask_fraction"]),
                          canvas_ratio=loss.new_tensor(canvas.numel()/ns),
                          token_correct=token_correct, token_count=fills.sum().float(),
                          masked_count=nmask.float(), token_ce_sum=token_sum.detach(),
                          action_ce_sum=action_sum.detach())
            for action, name in enumerate(["fill","expand","delete"]):
                target = actions.eq(action)
                record[name+"_correct"] = (action_pred.eq(action)&target).sum().float()
                record[name+"_count"] = target.sum().float()
                record[name+"_predicted"] = (action_pred.eq(action)&masked).sum().float()
            results.append(record)
        return {k:torch.stack([r[k] for r in results]).mean() for k in results[0]}

    def training_checkpoint(self, info):
        payload = super().training_checkpoint(info)
        payload["training_objective"] = "route_b_joint_edit_and_speech_nll"
        return payload


def build_route_b_model(dreamon_model_dir, cosyvoice_model_dir, max_sequence_tokens=2048,
                        seed=1986, lora_rank=16, lora_alpha=32., lora_lr=1e-4):
    bridge = DynamicSpeechLM.from_local_weights(
        dreamon_model_dir, load_cosyvoice_speech_components(cosyvoice_model_dir),
        device="cpu", dtype=torch.bfloat16, max_sequence_tokens=max_sequence_tokens, seed=seed)
    bridge.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model = RouteBTrainer(bridge, seed=seed, freeze_dreamon=True, use_lora=True,
                         lora_rank=lora_rank, lora_alpha=lora_alpha, lora_lr=lora_lr,
                         lora_dropout=.05, prompt_probability=0., validation_mask_ratio=1.)
    model.base_model_paths = dict(dreamon=str(dreamon_model_dir), cosyvoice=str(cosyvoice_model_dir))
    torch.manual_seed(seed+int(os.environ.get("RANK","0")))
    random.seed(seed+int(os.environ.get("RANK","0")))
    return model
