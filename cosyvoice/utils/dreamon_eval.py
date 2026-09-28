"""Dependency-free metadata and score helpers for paired TTS evaluation."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class EvalUtterance:
    utt: str
    prompt_text: str
    prompt_wav: Path
    text: str
    target_wav: Path | None = None


def read_eval_meta(path):
    """Read SEED's 4/5-column format, resolving audio paths beside the meta file."""
    path = Path(path).resolve(strict=True)
    rows, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) not in (4, 5) or any(not part for part in fields[:4]):
            raise ValueError(f"{path}:{number}: expected utt|prompt_text|prompt_wav|text[|target_wav].")
        utt = fields[0].removesuffix(".wav")
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", utt) or utt in seen:
            raise ValueError(f"Unsafe or duplicate output ID at line {number}: {utt!r}")
        if any("\t" in part for part in fields):
            raise ValueError(f"Tabs are not supported in evaluation metadata: line {number}.")
        seen.add(utt)

        def audio_path(value):
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()

        prompt = audio_path(fields[2])
        if not prompt.is_file():
            raise FileNotFoundError(prompt)
        target = audio_path(fields[4]) if len(fields) == 5 and fields[4] else None
        if target == prompt:
            raise ValueError(f"{utt}: reference audio must differ from the target recording.")
        rows.append(EvalUtterance(utt, fields[1], prompt, fields[3], target))
    if not rows:
        raise ValueError("Evaluation metadata is empty.")
    return rows


def write_eval_meta(path, rows):
    lines = []
    for row in rows:
        fields = [row.utt, row.prompt_text, str(row.prompt_wav.resolve()), row.text]
        if row.target_wav is not None:
            fields.append(str(row.target_wav.resolve()))
        if any(any(char in field for char in "|\n\r\t") for field in fields):
            raise ValueError(f"Unsupported delimiter in metadata for {row.utt}.")
        lines.append("|".join(fields))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def utterance_seed(seed, utt):
    return int.from_bytes(hashlib.sha256(f"{seed}:{utt}".encode()).digest()[:4], "little")


def prompt_rate_length(text, prompt_text, prompt_tokens, maximum):
    """Heuristic only: Chinese characters / other words, at reference speech rate."""
    def units(value):
        return len(re.findall(r"[\u3400-\u9fff]|[^\W_]+", value))

    target_units, reference_units = units(text), units(prompt_text)
    if min(target_units, reference_units, prompt_tokens, maximum) < 1:
        raise ValueError("Duration estimation needs text units and non-empty reference tokens.")
    estimate = max(1, round(prompt_tokens * target_units / reference_units))
    return min(estimate, maximum), estimate > maximum


def scoring_pairs(rows, wav_dir):
    """Require complete coverage; upstream scorers otherwise silently skip files."""
    wav_dir = Path(wav_dir).resolve(strict=True)
    pairs = []
    for row in rows:
        generated = wav_dir / f"{row.utt}.wav"
        if not generated.is_file() or generated.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty generated audio: {generated}")
        if any(char in str(generated) for char in "|\n\r\t"):
            raise ValueError(f"Unsupported delimiter in generated audio path: {generated}")
        pairs.append(f"{generated}|{row.prompt_wav}|{row.text}")
    return pairs


def summarize_scores(raw_path, metric, expected_count):
    """SEED-compatible utterance mean, with no removal of high-error examples."""
    values = []
    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or (metric == "sim" and line.startswith("avg score:")):
            continue
        fields = line.split("\t")
        if len(fields) != (7 if metric == "wer" else 2):
            raise ValueError("Unexpected upstream score format.")
        value = float(fields[1])
        if not math.isfinite(value) or (metric == "wer" and value < 0):
            raise ValueError("Non-finite or invalid evaluation score.")
        values.append(value)
    if len(values) != expected_count or not values:
        raise ValueError(f"Incomplete evaluation: scored {len(values)} of {expected_count} utterances.")
    return sum(values) / len(values)
