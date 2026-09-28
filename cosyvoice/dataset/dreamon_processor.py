"""Token-only processors compatible with CosyVoice Dataset/Processor.

Read existing CosyVoice parquet shards (even when named .tar) or JSONL shards.
Speech tokens must have been extracted with speech_tokenizer_v2.onnx. No audio
decoding, speaker embeddings or mel features are needed for adapter training.
"""

import json
import logging
from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence

from cosyvoice.llm.dreamon_speech import validate_speech_tokens


def load_dreamon_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(model_dir).resolve(strict=True)),
                                         trust_remote_code=True, local_files_only=True)


def read_token_samples(data, mode="train"):
    for shard in data:
        path = Path(shard["src"])
        if path.suffix.lower() == ".jsonl":
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except ValueError as error:
                            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
                        if not isinstance(record, dict):
                            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                        count += 1
                        yield record
            if count == 0:
                raise ValueError(f"Empty training shard: {path}")
        else:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            columns = [key for key in ("utt", "text", "speech_token", "speech_tokenizer")
                       if key in parquet.schema_arrow.names]
            if not {"utt", "text", "speech_token"}.issubset(columns):
                raise ValueError(f"{path} needs utt/text/speech_token columns. "
                                 "Extract offline tokens with speech_tokenizer_v2.onnx first.")
            count = 0
            for records in parquet.iter_batches(batch_size=64, columns=columns):
                for record in records.to_pylist():
                    count += 1
                    yield record
            if count == 0:
                raise ValueError(f"Empty training shard: {path}")


def _prepare_token_sample(sample, tokenizer, max_sequence_tokens, speech_vocab_size):
    """Validate and tokenize one record, returning None when it is filtered."""
    if not isinstance(sample.get("utt"), str) or not sample["utt"]:
        raise ValueError("Every training record needs a non-empty string utt ID.")
    if not isinstance(sample.get("text"), str) or not sample["text"].strip():
        raise ValueError(f"Missing transcript for {sample['utt']}.")
    if sample.get("speech_tokenizer", "cosyvoice2") != "cosyvoice2":
        raise ValueError(f"{sample['utt']} has speech tokens from a different codec.")
    if sample.get("speech_token") is None:
        raise ValueError(f"Missing offline speech tokens for {sample['utt']}.")
    speech_ids = torch.as_tensor(sample["speech_token"])
    if speech_ids.ndim != 1:
        raise ValueError(f"speech_token must be a flat list for {sample['utt']}.")
    if speech_ids.numel() == 0:
        return None
    validate_speech_tokens(speech_ids.unsqueeze(0), speech_vocab_size)
    text_ids = torch.tensor(tokenizer.encode(sample["text"], add_special_tokens=False), dtype=torch.long)
    # BOS + TASK + EOS, plus all text and speech (reference is a prefix split).
    if text_ids.numel() == 0 or text_ids.numel() + speech_ids.numel() + 3 > max_sequence_tokens:
        return None
    return {"utt": sample["utt"], "text_ids": text_ids, "speech_ids": speech_ids.long()}


def count_usable_samples_by_shard(shards, get_tokenizer, max_sequence_tokens=2048,
                                  speech_vocab_size=6561):
    """Return exact usable sample counts for progress reporting."""
    tokenizer = get_tokenizer()
    counts = []
    total = len(shards)
    for index, path in enumerate(shards, 1):
        count = 0
        for sample in read_token_samples(iter(({"src": path},)), mode="count"):
            if _prepare_token_sample(sample, tokenizer, max_sequence_tokens,
                                     speech_vocab_size) is not None:
                count += 1
        counts.append(count)
        if index == total or index % 25 == 0:
            logging.info("Counted usable DreamOn samples in %s/%s shards", index, total)
    return counts


def pack_token_batch(samples):
    text = [sample["text_ids"] for sample in samples]
    speech = [sample["speech_ids"] for sample in samples]
    return {
        "utts": [sample["utt"] for sample in samples],
        "text_tokenizer": "dreamon",
        "text_token": pad_sequence(text, batch_first=True, padding_value=0),
        "text_token_len": torch.tensor([ids.numel() for ids in text], dtype=torch.int32),
        "speech_token": pad_sequence(speech, batch_first=True, padding_value=0),
        "speech_token_len": torch.tensor([ids.numel() for ids in speech], dtype=torch.int32),
    }


def make_token_batches(data, get_tokenizer, batch_size=1, max_sequence_tokens=2048,
                       shuffle_buffer_size=64, speech_vocab_size=6561, mode="train"):
    if batch_size < 1 or shuffle_buffer_size < 1:
        raise ValueError("batch_size and shuffle_buffer_size must be positive.")
    tokenizer = get_tokenizer()
    buffer, pending = [], []
    accepted, dropped = 0, 0

    def drain():
        if mode == "train":
            random.shuffle(buffer)
        while buffer:
            pending.append(buffer.pop())
            if len(pending) == batch_size:
                yield pack_token_batch(pending)
                pending.clear()

    for sample in data:
        prepared = _prepare_token_sample(sample, tokenizer, max_sequence_tokens,
                                         speech_vocab_size)
        if prepared is None:
            dropped += 1
            continue
        accepted += 1
        buffer.append(prepared)
        if len(buffer) >= (shuffle_buffer_size if mode == "train" else 1):
            yield from drain()
    yield from drain()
    if pending:
        yield pack_token_batch(pending)
    if dropped:
        logging.warning("DreamOn token pipeline dropped %s empty/overlength utterances", dropped)
    if accepted == 0:
        raise ValueError("No usable DreamOn training samples in this worker; check tokens and context limit.")
