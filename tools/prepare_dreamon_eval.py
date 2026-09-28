#!/usr/bin/env python3
"""Make a reproducible local diagnostic set from prepared LibriTTS metadata.

Reference and target are different utterances from the same held-out speaker.
This is a custom diagnostic protocol, not the CosyVoice paper's published list.
"""

import argparse
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosyvoice.utils.dreamon_eval import EvalUtterance, write_eval_meta


def read_mapping(path):
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split(maxsplit=1)
        if key in mapping:
            raise ValueError(f"Duplicate utterance in {path}: {key}")
        mapping[key] = value
    return mapping


def prepare_rows(data_dir, limit):
    text = read_mapping(data_dir / "text")
    speakers = read_mapping(data_dir / "utt2spk")
    wavs = read_mapping(data_dir / "wav.scp")
    if set(text) != set(speakers) or set(text) != set(wavs):
        raise ValueError("wav.scp, text and utt2spk must contain the same utterances.")
    grouped = defaultdict(list)
    for utt in sorted(text):
        # Match original prepare_data.py: relative WAV paths are relative to cwd.
        wav = Path(wavs[utt]).expanduser().resolve(strict=True)
        with wave.open(str(wav), "rb") as handle:
            seconds = handle.getnframes() / handle.getframerate()
        if 1 <= seconds <= 20 and text[utt].strip():
            grouped[speakers[utt]].append((utt, wav, seconds))
    per_speaker = []
    for speaker in sorted(grouped):
        items = grouped[speaker]
        prompt = next((item for item in items if 2 <= item[2] <= 8), None)
        if prompt is None:
            continue
        per_speaker.append([
            EvalUtterance(utt, text[prompt[0]], prompt[1], text[utt], wav)
            for utt, wav, _ in items if utt != prompt[0] and text[utt] != text[prompt[0]]
        ])
    rows = []
    for column in zip_longest(*per_speaker):
        for row in column:
            if row is not None:
                rows.append(row)
                if len(rows) == limit:
                    return rows
    if not rows:
        raise ValueError("No suitable same-speaker reference/target pairs; check WAV durations and transcripts.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="prepared dev-clean or test-clean metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100, help="maximum utterances, sampled round-robin across speakers")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Use a new metadata filename: {args.output}")
    rows = prepare_rows(args.data_dir, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_eval_meta(args.output, rows)
    print(f"Saved {len(rows)} local diagnostic pairs to {args.output}. This is not an official benchmark list.")


if __name__ == "__main__":
    main()
