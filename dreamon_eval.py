#!/usr/bin/env python3
"""Score generated WAVs using the upstream SEED-TTS ASR or WavLM evaluator.

Runs the upstream Python programs directly, with complete-coverage checks.
No TTS weights are loaded. Use a separate environment for evaluator dependencies.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cosyvoice.utils.dreamon_eval import read_eval_meta, scoring_pairs, summarize_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--seed-eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new scores directory")
    parser.add_argument("--metric", choices=("wer", "sim"), default="wer")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--speaker-checkpoint", type=Path, help="SEED's wavlm_large_finetune.pth, for SIM")
    parser.add_argument("--check-only", action="store_true", help="check files/coverage; do not load ASR/SIM models")
    args = parser.parse_args()
    rows = read_eval_meta(args.meta)
    pairs = scoring_pairs(rows, args.wav_dir)
    repo = args.seed_eval_dir.resolve(strict=True)
    if args.metric == "wer":
        program = repo / "run_wer.py"
    else:
        program = repo / "thirdparty/UniSpeech/downstreams/speaker_verification/verification_pair_list_v2.py"
        if args.speaker_checkpoint is None or not args.speaker_checkpoint.is_file():
            raise ValueError("SIM requires --speaker-checkpoint pointing to wavlm_large_finetune.pth.")
    if not program.is_file():
        raise FileNotFoundError(program)
    if args.check_only:
        print(f"Complete audio coverage: {len(pairs)} utterances. Evaluator dependencies/models were not checked.")
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pair_file, raw_file = output / "pairs.txt", output / "raw_scores.txt"
    pair_file.write_text("\n".join(pairs) + "\n", encoding="utf-8")
    if args.metric == "wer":
        command = [sys.executable, str(program), str(pair_file), str(raw_file), args.language]
    else:
        command = [sys.executable, str(program), str(pair_file), "--model_name", "wavlm_large",
                   "--checkpoint", str(args.speaker_checkpoint.resolve()), "--scores", str(raw_file),
                   "--wav1_start_sr", "0", "--wav2_start_sr", "0", "--wav1_end_sr", "-1",
                   "--wav2_end_sr", "-1", "--device", "cuda:0"]
    report = {"status": "running", "meta": str(args.meta.resolve()), "wav_dir": str(args.wav_dir.resolve()),
              "expected_utterances": len(rows), "command": command, "language": args.language,
              "aggregation": "utterance_mean_all_samples",
              "metric": ("CER" if args.language == "zh" else "WER") if args.metric == "wer" else "SIM_WavLM"}

    def save_report():
        (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    save_report()
    try:
        subprocess.run(command, cwd=program.parent, check=True)
        value = summarize_scores(raw_file, args.metric, len(rows))
        report.update(status="complete", scored_utterances=len(rows), value=value * 100 if args.metric == "wer" else value,
                      units="percent" if args.metric == "wer" else "cosine_similarity")
        print(f"{report['metric']}: {report['value']:.4f} ({report['units']})")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_report()
        raise
    save_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
