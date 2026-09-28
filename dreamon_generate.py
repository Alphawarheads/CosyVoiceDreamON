#!/usr/bin/env python3
"""Batch-generate paired evaluation audio with original CosyVoice2 or DreamOn.

Example: python dreamon_generate.py --meta data/eval/meta.lst --backend dreamon
         --adapter-checkpoint exp/dreamon_frozen/epoch_0_whole.pt --output-dir outputs/frozen
"""

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from dreamon_cosyvoice_test import ROOT, decode_and_save, load_config, write_report
from cosyvoice.utils.dreamon_eval import read_eval_meta, write_eval_meta, prompt_rate_length, utterance_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--backend", choices=("cosyvoice", "dreamon"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; existing directories are rejected")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dreamon_cosyvoice.yaml")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--cosyvoice-model-dir")
    parser.add_argument("--dreamon-model-dir")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--tokens-per-step", type=int)
    parser.add_argument("--speech-tokens", dest="initial_speech_tokens", type=int)
    parser.add_argument("--math-sdpa", action="store_true", default=None)
    parser.add_argument("--length-mode", choices=("auto", "dynamic", "learned", "prompt-rate", "fixed", "oracle"), default="auto")
    parser.add_argument("--max-speech-tokens", type=int, default=750)
    parser.add_argument("--initial-masks", type=int, default=64)
    parser.add_argument("--max-edit-steps", type=int, default=2048)
    parser.add_argument("--action-temperature", type=float, default=0.8,
                        help="Route B action sampling temperature; 0 enables greedy diagnostics")
    parser.add_argument("--limit", type=int, help="debug subset; the selected meta file is saved alongside outputs")
    parser.add_argument("--check-only", action="store_true", help="validate metadata/config only; no models or audio generation")
    return parser.parse_args()


def run(args, rows, config, output_dir, report):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    if config["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM, validate_speech_tokens

    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable.")
        torch.cuda.set_device(device)
        if args.backend == "dreamon" and config["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported; select --dtype fp16 or fp32 for inference.")
    elif config["dtype"] != "fp32":
        raise ValueError("CPU generation requires --dtype fp32.")
    if config["math_sdpa"]:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def seed_all(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(value)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    cosyvoice = CosyVoice2(config["cosyvoice_model_dir"], fp16=False,
                          load_jit=False, load_trt=False, load_vllm=False)
    bridge = None
    if args.backend == "dreamon":
        components = DreamOnSpeechLM.copy_cosyvoice_components(cosyvoice.model.llm)
        cosyvoice.model.llm = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]
        bridge_class = DreamOnSpeechLM
        checkpoint_payload = None
        if config["adapter_checkpoint"]:
            checkpoint_payload = torch.load(config["adapter_checkpoint"], map_location="cpu",
                                            weights_only=True, mmap=True)
            if checkpoint_payload.get("format_version") == 7:
                from cosyvoice.llm.dreamon_route_c import DynamicSpeechLM
                bridge_class = DynamicSpeechLM
            elif checkpoint_payload.get("format_version") == 6:
                from cosyvoice.llm.dreamon_route_b import DynamicSpeechLM
                bridge_class = DynamicSpeechLM
        bridge = bridge_class.from_local_weights(
            config["dreamon_model_dir"], components, device=device, dtype=dtype,
            seed=config["seed"], max_sequence_tokens=config["max_sequence_tokens"],
        )
        del components
        if config["adapter_checkpoint"]:
            bridge.speech_in_proj.float()
            bridge.speech_out_proj.float()
            bridge.load_adapter_checkpoint(checkpoint_payload)
        del checkpoint_payload
        report["adaptation_status"] = bridge.adaptation_status
        report["loaded_finetuned_backbone"] = bridge.backbone_checkpoint_required
        report["lora_config"] = bridge.lora_config
        selected_mode = args.length_mode
        if selected_mode == "auto":
            selected_mode = ("dynamic" if hasattr(bridge, "generate_dynamic") else
                             "learned" if hasattr(bridge, "duration_predictor") else "prompt-rate")
        if hasattr(bridge, "generate_dynamic") != (selected_mode == "dynamic"):
            raise ValueError("Route B/C checkpoints require dynamic mode; dynamic requires a Route B/C checkpoint.")
        if selected_mode == "learned" and not hasattr(bridge, "duration_predictor"):
            raise ValueError("Learned length requires a Route A checkpoint.")
        args.length_mode = selected_mode
        config["dynamic_length"] = selected_mode == "dynamic"
        if selected_mode == "dynamic":
            report["edit_generation_config"] = dict(
                initial_masks=args.initial_masks, max_steps=args.max_edit_steps,
                speech_temperature=config["temperature"], action_temperature=args.action_temperature)
        report["length_mode"] = selected_mode
        report["speech_conditioning"] = getattr(bridge, "speech_conditioning", "reference_and_text")
        bridge.eval()


    for index, row in enumerate(rows, 1):
        report["current_utt"] = row.utt
        write_report(output_dir / "run.json", report)
        seed = utterance_seed(config["seed"], row.utt)
        seed_all(seed)
        # Same raw transcript, reference and frontend for both models; no target audio conditioning.
        conditioning = cosyvoice.frontend.frontend_zero_shot(
            row.text, row.prompt_text, str(row.prompt_wav), cosyvoice.sample_rate, "")
        length, clipped = None, False
        length_info = {}
        if bridge is not None:
            if args.length_mode == "dynamic":
                length_info = {"length_mode": "dynamic", "initial_masks": args.initial_masks, "target_audio_used": False}
            elif args.length_mode == "learned":
                from cosyvoice.llm.dreamon_route_a import predict_speech_length
                length, length_info = predict_speech_length(
                    bridge, row.text, row.prompt_text,
                    conditioning["llm_prompt_speech_token"].numel(), args.max_speech_tokens)
                clipped = length_info["length_clipped"]
            elif args.length_mode == "fixed":
                length = config["initial_speech_tokens"]
            elif args.length_mode == "oracle":
                # Explicit diagnostic only: use target length, never its token values as input.
                reference_tokens, _ = cosyvoice.frontend._extract_speech_token(str(row.target_wav))
                length = reference_tokens.numel()
                if not 0 < length <= args.max_speech_tokens:
                    raise ValueError(f"{row.utt}: oracle token length is empty or exceeds --max-speech-tokens.")
                del reference_tokens
            else:
                length, clipped = prompt_rate_length(
                    row.text, row.prompt_text, conditioning["llm_prompt_speech_token"].numel(), args.max_speech_tokens)
        synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            generation_report = {}
            if bridge is None:
                ids = list(cosyvoice.model.llm.inference(
                    text=conditioning["text"], text_len=conditioning["text_len"].clone(),
                    prompt_text=conditioning["prompt_text"], prompt_text_len=conditioning["prompt_text_len"].clone(),
                    prompt_speech_token=conditioning["llm_prompt_speech_token"],
                    prompt_speech_token_len=conditioning["llm_prompt_speech_token_len"].clone(),
                    embedding=conditioning["llm_embedding"],
                ))
                tokens = torch.tensor([ids], dtype=torch.int32)
            elif args.length_mode == "dynamic":
                from cosyvoice.llm.dreamon_route_b import EditGenerationError
                try:
                    reference_kwargs = (dict(prompt_text=row.prompt_text,
                        prompt_speech_tokens=conditioning["llm_prompt_speech_token"])
                        if getattr(bridge, "speech_conditioning", None) == "reference_text_and_speech" else {})
                    tokens, generation_report = bridge.generate_dynamic(
                        row.text, initial_masks=args.initial_masks, maximum=args.max_speech_tokens,
                        max_steps=args.max_edit_steps, temperature=config["temperature"],
                        action_temperature=args.action_temperature, seed=seed, **reference_kwargs)
                except EditGenerationError as exc:
                    write_report(output_dir / "tokens" / f"{row.utt}.failure.json", exc.report)
                    report["failed_utterances"] = report.get("failed_utterances", 0) + 1
                    report["attempted"] = index
                    with (output_dir / "utterances.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"utt": row.utt, "seed": seed, "status": "failed",
                                                 "failure_reason": str(exc)}) + "\n")
                    write_report(output_dir / "run.json", report)
                    print(f"[{index}/{len(rows)}] {row.utt}: generation failed: {exc}", flush=True)
                    continue
                write_report(output_dir / "tokens" / f"{row.utt}.trajectory.json", generation_report)
            else:
                text_only = getattr(bridge, "speech_conditioning", None) == "text_only"
                speech_prompt = (conditioning["llm_prompt_speech_token"][:, :0]
                                 if text_only else conditioning["llm_prompt_speech_token"])
                tokens, generation_report = bridge.generate(
                    row.text, "" if text_only else row.prompt_text, speech_prompt,
                    speech_tokens=length, tokens_per_step=config["tokens_per_step"],
                    temperature=config["temperature"], seed=seed)
            validate_speech_tokens(tokens)
            seed_all(seed)  # Flow randomness independent of the language-model sampling.
            audio = decode_and_save(cosyvoice, conditioning, tokens, output_dir / "wavs" / f"{row.utt}.wav")
        synchronize()
        elapsed = time.monotonic() - started
        torch.save(tokens, output_dir / "tokens" / f"{row.utt}.pt")
        record = {"utt": row.utt, "seed": seed, "speech_tokens": tokens.numel(), "length_clipped": clipped,
                  "length_prediction": length_info,
                  "edit_counts": {k:generation_report[k] for k in ("expand_count","delete_count","fill_count") if k in generation_report},
                  "generation_seconds": elapsed, "rtf": elapsed / audio["duration_seconds"], "audio": audio}
        with (output_dir / "utterances.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        report["completed"] += 1
        report["attempted"] = index
        write_report(output_dir / "run.json", report)
        print(f"[{index}/{len(rows)}] {row.utt}: {audio['duration_seconds']:.2f}s audio", flush=True)
    report["status"] = "complete_with_failures" if report.get("failed_utterances", 0) else "complete"
    write_report(output_dir / "run.json", report)


def main():
    args = parse_args()
    rows = read_eval_meta(args.meta)
    if args.max_speech_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Token/sample limits must be positive.")
    if args.limit:
        rows = rows[:args.limit]
    overrides = {name: getattr(args, name) for name in (
        "config", "adapter_checkpoint", "cosyvoice_model_dir", "dreamon_model_dir", "dtype", "device",
        "seed", "temperature", "tokens_per_step", "initial_speech_tokens", "math_sdpa")}
    overrides.update(text=rows[0].text, prompt_text=rows[0].prompt_text, prompt_wav=str(rows[0].prompt_wav))
    config = load_config(SimpleNamespace(**overrides))
    if args.backend == "cosyvoice":
        if args.adapter_checkpoint:
            raise ValueError("--adapter-checkpoint is only for --backend dreamon.")
        config["adapter_checkpoint"] = None
    if args.backend == "dreamon" and args.length_mode == "learned" and not config["adapter_checkpoint"]:
        raise ValueError("Learned length requires --adapter-checkpoint.")
    if args.backend == "dreamon" and args.length_mode == "oracle":
        if any(row.target_wav is None or not row.target_wav.is_file() for row in rows):
            raise ValueError("Oracle length requires a valid target WAV in every metadata row.")
    if args.length_mode == "fixed" and config["initial_speech_tokens"] > args.max_speech_tokens:
        raise ValueError("--speech-tokens exceeds --max-speech-tokens.")
    if args.check_only:
        print(f"Metadata/config valid: {len(rows)} utterances. Model dependencies/GPU were not checked.")
        return 0
    output_dir = args.output_dir.resolve()
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "wavs").mkdir()
    (output_dir / "tokens").mkdir()
    write_eval_meta(output_dir / "meta.lst", rows)
    report = {"status": "running", "backend": args.backend, "config": config, "expected": len(rows), "completed": 0,
              "length_mode": args.length_mode if args.backend == "dreamon" else "autoregressive_eos",
              "uses_target_length": args.backend == "dreamon" and args.length_mode == "oracle",
              "text_normalization": "raw_metadata_transcripts", "max_speech_tokens": args.max_speech_tokens}
    write_report(output_dir / "run.json", report)
    try:
        run(args, rows, config, output_dir, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_report(output_dir / "run.json", report)
        raise
    return 2 if report.get("failed_utterances", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
