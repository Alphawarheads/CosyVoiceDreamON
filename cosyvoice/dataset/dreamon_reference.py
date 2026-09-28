"""Route C: deterministic, split-local, different-utterance reference pairing."""
from collections import defaultdict
import hashlib
import random
import re
from bisect import bisect_right

CONDITIONING = "reference_text_and_speech"


def speaker_id(record):
    # This preparation command is intentionally specific to LibriTTS IDs.
    match = re.fullmatch(r"(\d+)[_-]\d+[_-]\d+[_-]\d+", record["utt"])
    if match is None:
        raise ValueError("Expected LibriTTS speaker/chapter/utterance ID: " + record["utt"])
    return match.group(1)


def pair_cache(source, max_sequence_tokens=2048, max_reference_tokens=250, seed=1986):
    if max_sequence_tokens < 4 or max_reference_tokens < 1:
        raise ValueError("Invalid context/reference limits.")
    result = dict(format="route_c_paired_v1", summary=source["summary"],
                  pairing=dict(seed=seed, max_sequence_tokens=max_sequence_tokens,
                               max_reference_tokens=max_reference_tokens,
                               surplus_reserve=64, speech_conditioning=CONDITIONING,
                               policy="fixed_same_speaker_different_utterance_within_split"),
                  reference_pool={}, pairing_summary={})
    for split in ("train", "dev"):
        records = source[split]
        pool = defaultdict(list)
        for r in records:
            if 0 < len(r["speech"]) <= max_reference_tokens:
                pool[speaker_id(r)].append(r)
        # The all split has hundreds of thousands of targets: select from a
        # per-speaker length index rather than rescanning every utterance.
        indexed = {}
        for speaker, examples in pool.items():
            examples.sort(key=lambda r:(len(r["text_ids"])+len(r["speech"]), r["utt"]))
            indexed[speaker] = ([len(r["text_ids"])+len(r["speech"]) for r in examples], examples)
        paired, used, dropped = [], {}, 0
        for target in records:
            lengths, examples = indexed.get(speaker_id(target), ([], []))
            allowance = (max_sequence_tokens - len(target["text_ids"])
                         - len(target["speech"]) - 3 - 64)
            limit = bisect_right(lengths, allowance)
            if not limit:
                dropped += 1
                continue
            digest = hashlib.sha256(f"{seed}:{split}:{target['utt']}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "little"))
            offset = rng.randrange(limit)
            reference = None
            for shift in range(limit):
                candidate = examples[(offset + shift) % limit]
                if (candidate["utt"] != target["utt"]
                        and candidate["text"].strip() != target["text"].strip()):
                    reference = candidate
                    break
            if reference is None:
                dropped += 1
                continue
            r = {k:target[k] for k in ("utt", "text", "text_ids", "speech")}
            r.update(ref_utt=reference["utt"], ref_text=reference["text"],
                     ref_text_ids=reference["text_ids"], ref_speech=reference["speech"])
            paired.append(r)
            used[reference["utt"]] = {k:reference[k] for k in ("utt", "text", "text_ids", "speech")}
        result[split] = paired
        result["reference_pool"][split] = used
        result["pairing_summary"][split] = dict(source_targets=len(records), paired_targets=len(paired),
                                               dropped_no_compatible_reference=dropped,
                                               unique_references=len(used))
    validate_reference_cache(result, max_sequence_tokens)
    return result


def validate_reference_cache(cache, max_sequence_tokens):
    if cache.get("format") != "route_c_paired_v1":
        raise ValueError("Route C requires a paired cache; run tools/prepare_dreamon_route_c.py.")
    split_ids, split_speakers = {}, {}
    for split in ("train", "dev"):
        records = cache[split]
        refs = cache["reference_pool"][split]
        if not records or len({r["utt"] for r in records}) != len(records):
            raise ValueError("Empty split or duplicate target IDs.")
        split_ids[split] = {r["utt"] for r in records} | set(refs)
        split_speakers[split] = {speaker_id(r) for r in records} | {speaker_id(r) for r in refs.values()}
        for r in records:
            ref = refs.get(r["ref_utt"])
            if (ref is None or r["utt"] == r["ref_utt"] or speaker_id(r) != speaker_id(ref)
                    or r["text"].strip() == r["ref_text"].strip()):
                raise ValueError("Invalid or leaking reference pair: " + r["utt"])
            for key in ("text", "text_ids", "speech"):
                if r["ref_" + key] != ref[key]:
                    raise ValueError("Reference text/tokens do not match the source reference.")
            for key in ("text_ids", "ref_text_ids", "speech", "ref_speech"):
                values = r[key]
                if not values or any(type(v) is not int or v < 0 for v in values):
                    raise ValueError("Invalid token list: " + key)
                if "speech" in key and max(values) >= 6561:
                    raise ValueError("Speech token outside CosyVoice2 vocabulary.")
            if sum(len(r[k]) for k in ("text_ids", "ref_text_ids", "speech", "ref_speech")) + 3 > max_sequence_tokens:
                raise ValueError("Paired sample exceeds context; reprepare with the training context limit.")
    if split_ids["train"] & split_ids["dev"] or split_speakers["train"] & split_speakers["dev"]:
        raise ValueError("Train/dev target or reference leakage.")
