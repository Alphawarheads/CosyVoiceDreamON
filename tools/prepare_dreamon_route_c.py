#!/usr/bin/env python3
"""Pair an existing LibriTTS token cache for Route C without extracting tokens again."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cosyvoice.dataset.dreamon_reference import pair_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-sequence-tokens", type=int, default=2048)
    p.add_argument("--max-reference-tokens", type=int, default=250)
    p.add_argument("--seed", type=int, default=1986)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.source_cache.read_bytes()
    source = json.loads(raw)
    for split in ("train", "dev"):
        info = source["summary"][split]
        if hashlib.sha256(Path(info["source_list"]).read_bytes()).hexdigest() != info["list_sha256"]:
            raise ValueError("Source list changed; rebuild original token cache.")
    result = pair_cache(source, args.max_sequence_tokens, args.max_reference_tokens, args.seed)
    result["pairing"].update(source_cache=str(args.source_cache.resolve()), source_sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting an existing experiment cache.
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(result["pairing_summary"], indent=2))


if __name__ == "__main__":
    main()
