#!/usr/bin/env python3
"""Evaluate masked speech tokens, or ASR-score completed generation runs.

Token reconstruction metrics and ASR pronunciation metrics are separate protocols.
Neither is a MOS/speaker-similarity measurement or the official CosyVoice benchmark.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime
import itertools
import json
import logging
import os
from pathlib import Path
import sys
import unicodedata
import uuid


ROOT = Path(__file__).resolve().parent


def normalize_text(text, language):
    text = unicodedata.normalize("NFKC", text).casefold()
    cleaned = []
    for char in text:
        if unicodedata.category(char)[0] in ("P", "S"):
            cleaned.append(" " if language == "en" and char not in ("'", "’") else "")
        else:
            cleaned.append(char)
    text = " ".join("".join(cleaned).split())
    return text if language == "en" else "".join(text.split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def transcript_errors(reference, hypothesis, language):
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref_chars:
        raise ValueError("Reference transcript is empty after normalization.")
    scores = {"reference_normalized": ref, "hypothesis_normalized": hyp,
              "character_errors": edit_distance(ref_chars, hyp_chars),
              "reference_characters": len(ref_chars)}
    if language == "en":
        scores.update(word_errors=edit_distance(ref.split(), hyp.split()), reference_words=len(ref.split()))
    return scores


def aggregate_asr(rows):
    if not rows:
        raise ValueError("No ASR results to score.")
    result = {}
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        counts = {key: sum(row.get(key, 0) for row in selected)
                  for key in ("character_errors", "reference_characters", "word_errors", "reference_words")}
        score = {"utterances": len(selected), **counts,
                 "cer": counts["character_errors"] / counts["reference_characters"]}
        if counts["reference_words"]:
            score["wer"] = counts["word_errors"] / counts["reference_words"]
        result[model] = score
    return result


def read_shard_list(path):
    # Same convention as the training recipe: relative shard paths use cwd.
    with Path(path).open(encoding="utf-8") as handle:
        paths = [Path(line.strip()).resolve(strict=True) for line in handle if line.strip()]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("data.list must contain non-duplicated JSONL/parquet paths.")
    return [{"src": str(path)} for path in paths]


def evaluate_token_batches(trainer, batches, device, use_amp=False, on_result=None):
    import torch

    trainer.eval()
    device = torch.device(device)
    sums = {"loss": 0.0, "acc": 0.0, "masked_ce_sum": 0.0, "masked_correct_sum": 0.0,
            "masked_tokens": 0, "utterances": 0}
    seen = set()
    with torch.inference_mode():
        for batch in batches:
            if len(batch["utts"]) != 1:
                raise ValueError("Standalone evaluation uses batch_size=1 for exact aggregation.")
            utt = batch["utts"][0]
            if utt in seen:
                raise ValueError(f"Duplicate evaluation utterance: {utt}")
            seen.add(utt)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
            with context:
                values = trainer(batch, device)
            # Eval corruption is deterministic by utterance ID; recover its count.
            speech = batch["speech_token"][:, :int(batch["speech_token_len"][0])].to(device)
            prompt, target, _, mask = trainer._corrupt(speech, utt)
            count = int(mask.sum())
            loss, acc = float(values["loss"]), float(values["acc"])
            row = {"utt": utt, "loss": loss, "acc": acc, "masked_tokens": count,
                   "prompt_tokens": prompt.numel(), "target_tokens": target.numel()}
            sums["loss"] += loss
            sums["acc"] += acc
            sums["masked_ce_sum"] += loss * count
            sums["masked_correct_sum"] += acc * count
            sums["masked_tokens"] += count
            sums["utterances"] += 1
            if on_result:
                on_result(row)
    if not sums["utterances"]:
        raise ValueError("No usable evaluation utterances.")
    return {"utterances": sums["utterances"], "masked_tokens": sums["masked_tokens"],
            "loss": sums["loss"] / sums["utterances"], "acc": sums["acc"] / sums["utterances"],
            "token_weighted_ce": sums["masked_ce_sum"] / sums["masked_tokens"],
            "token_weighted_accuracy": sums["masked_correct_sum"] / sums["masked_tokens"]}


def run_token_eval(args, emit):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    import torch
    from cosyvoice.dataset.dreamon_processor import make_token_batches, read_token_samples
    from cosyvoice.llm.dreamon_training import build_training_model

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unavailable; use --dtype fp32.")
    elif device.type != "cpu" or args.dtype != "fp32":
        raise ValueError("CPU evaluation requires --dtype fp32.")
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    trainer = build_training_model(
        args.dreamon_model_dir, args.cosyvoice_model_dir, dtype=args.dtype,
        freeze_dreamon=True, gradient_checkpointing=False,
        max_sequence_tokens=args.max_sequence_tokens, seed=args.seed,
        prompt_probability=0.5 if args.protocol == "reconstruction" else 0.0,
        max_prompt_fraction=0.3,
        validation_mask_ratio=0.5 if args.protocol == "reconstruction" else 1.0,
    )
    if args.checkpoint:
        trainer.load_training_checkpoint(args.checkpoint)
    trainer.to(device).eval()
    shards = read_shard_list(args.data_list)
    coverage = {"records_read": 0}

    def records():
        for row in read_token_samples(shards, mode="dev"):
            coverage["records_read"] += 1
            yield row

    batches = make_token_batches(records(), lambda: trainer.bridge.tokenizer, mode="dev",
                                 batch_size=1, max_sequence_tokens=args.max_sequence_tokens)
    if args.max_samples:
        batches = itertools.islice(batches, args.max_samples)
    metrics = evaluate_token_batches(trainer, batches, device,
                                     use_amp=device.type == "cuda" and args.dtype == "bf16", on_result=emit)
    coverage["filtered_records"] = coverage["records_read"] - metrics["utterances"]
    return {"protocol": args.protocol, "seed": args.seed, "metrics": metrics, "coverage": coverage,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "adaptation_status": trainer.bridge.adaptation_status,
            "base_model_paths": trainer.base_model_paths, "shards": shards,
            "mask_ratio": trainer.validation_mask_ratio, "prompt_probability": trainer.prompt_probability,
            "max_prompt_fraction": trainer.max_prompt_fraction,
            "note": "loss/acc are utterance means; full-mask is one forward pass, not iterative generation or WER."}


def generation_items(runs):
    items, seen = [], set()
    for run in runs:
        run = Path(run).resolve(strict=True)
        if run in seen:
            raise ValueError(f"Duplicate generation run: {run}")
        seen.add(run)
        report = json.loads((run / "diagnostics.json").read_text(encoding="utf-8"))
        if report.get("status") != "interface_smoke_test_completed":
            raise ValueError(f"Generation did not complete successfully: {run}")
        reference = report["config"]["text"]  # Never include the reference speaker's prompt text.
        for model, filename in (("baseline", "baseline.wav"), ("dreamon", "dreamon_stitched.wav")):
            items.append({"run": str(run), "model": model, "reference": reference,
                          "wav": str((run / filename).resolve(strict=True))})
    return items


def run_asr_eval(args, emit):
    items = generation_items(args.runs)
    import whisper

    model = whisper.load_model(args.whisper_model, device=args.device,
                               download_root=str(ROOT / ".cache/whisper"))
    rows = []
    for item in items:
        # No target transcript/prompt supplied to ASR, avoiding evaluation leakage.
        output = model.transcribe(item["wav"], language=args.language, task="transcribe",
                                  temperature=0.0, condition_on_previous_text=False,
                                  fp16=str(args.device).startswith("cuda"), verbose=None)
        row = {**item, "hypothesis": output["text"],
               **transcript_errors(item["reference"], output["text"], args.language)}
        rows.append(row)
        emit(row)
    return {"asr_model": args.whisper_model, "language": args.language, "metrics": aggregate_asr(rows),
            "normalization": "NFKC + casefold; punctuation/symbol removal; no number expansion or script conversion",
            "note": "Rates are edit totals/reference totals, not percentages; this is not the official benchmark."}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tokens = commands.add_parser("tokens", help="evaluate a data.list with deterministic masking")
    tokens.add_argument("--data-list", required=True)
    tokens.add_argument("--checkpoint", help="omit to evaluate untrained projections")
    tokens.add_argument("--dreamon-model-dir", default=str(ROOT / "DreamOn-v0-7B"))
    tokens.add_argument("--cosyvoice-model-dir", default=str(ROOT / "CosyVoice2-0.5B"))
    tokens.add_argument("--protocol", choices=("reconstruction", "full-mask"), default="reconstruction")
    tokens.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    tokens.add_argument("--seed", type=int, default=1986)
    tokens.add_argument("--max-sequence-tokens", type=int, default=2048)
    tokens.add_argument("--max-samples", type=int, default=0, help="0 evaluates all usable records")
    tokens.add_argument("--math-sdpa", action="store_true")
    asr = commands.add_parser("asr", help="ASR-score the two WAVs in one or more generation runs")
    asr.add_argument("--runs", nargs="+", required=True, help="directories containing diagnostics.json and WAVs")
    asr.add_argument("--language", choices=("en", "zh"), required=True)
    asr.add_argument("--whisper-model", default="small", help="Whisper name or local .pt path")
    for command in (tokens, asr):
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output-dir", type=Path, help="new directory; existing results are not overwritten")
    args = parser.parse_args()
    if args.command == "tokens" and (args.max_samples < 0 or args.max_sequence_tokens < 4):
        parser.error("max-samples must be >=0 and max-sequence-tokens must be >=4")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    name = f"{args.command}-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    directory = args.output_dir or ROOT / "outputs/dreamon_eval" / name
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "command": args.command, "arguments": vars(args).copy()}
    report["arguments"]["output_dir"] = str(directory.resolve())
    summary_path = directory / "summary.json"
    exit_code = 0
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as handle:
        def emit(row):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            logging.info("Evaluated %s", row.get("utt", row.get("wav")))

        try:
            report.update(run_token_eval(args, emit) if args.command == "tokens" else run_asr_eval(args, emit))
            report["status"] = "completed"
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            logging.exception("Evaluation failed")
            exit_code = 1
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Evaluation report: {summary_path}")
    if exit_code == 0:
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
