"""Four-GPU Route A trainer using the existing LibriTTS parquet lists.

Example: python -m torch.distributed.run --standalone --nproc_per_node=4
         -m cosyvoice.bin.train_route_a --model-dir exp/NEW_TAG
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import logging

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from cosyvoice.dataset.dreamon_processor import (
    load_dreamon_tokenizer, read_token_samples, _prepare_token_sample)
from cosyvoice.llm.dreamon_route_a import build_route_a_model, predict_speech_length

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("loss", "token_ce", "duration_loss", "acc", "mask_fraction",
           "duration_relative_error", "duration_mae_tokens")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--train-data", type=Path, default=Path("/home/lize/AudioData/libritts/prepared/lists/train_100.data.list"))
    p.add_argument("--cv-data", type=Path, default=Path("/home/lize/AudioData/libritts/prepared/lists/dev_all.data.list"))
    p.add_argument("--dreamon-model-dir", type=Path, default=ROOT / "DreamOn-v0-7B")
    p.add_argument("--cosyvoice-model-dir", type=Path, default=ROOT / "CosyVoice2-0.5B")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--projection-lr", type=float, default=1e-4)
    p.add_argument("--duration-lr", type=float, default=1e-3)
    p.add_argument("--duration-loss-weight", type=float, default=.1)
    p.add_argument("--max-sequence-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1986)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=0, help="0: all configured epochs")
    return p


def prepare(args):
    args.model_dir.mkdir(parents=True, exist_ok=True)
    dest = args.model_dir / "data_cache.json"
    if dest.exists():
        raise FileExistsError("Data cache exists; choose a new experiment directory.")
    tokenizer = load_dreamon_tokenizer(args.dreamon_model_dir)
    splits, summary = {}, {}
    for name, listing in (("train", args.train_data), ("dev", args.cv_data)):
        paths = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
        records, rejected, seen = [], 0, set()
        for path in paths:
            for raw in read_token_samples(iter([{"src": path}])):
                sample = _prepare_token_sample(raw, tokenizer, args.max_sequence_tokens, 6561)
                if sample is None:
                    rejected += 1
                    continue
                if sample["utt"] in seen:
                    raise ValueError("Duplicate utterance: " + sample["utt"])
                seen.add(sample["utt"])
                records.append(dict(utt=sample["utt"], text=raw["text"],
                                    text_ids=sample["text_ids"].tolist(),
                                    speech=sample["speech_ids"].tolist()))
        records.sort(key=lambda r: r["utt"])
        groups = {}
        for index, record in enumerate(records):
            # LibriTTS utt IDs are speaker_chapter_utterance, as produced by preprocessing.
            speaker = record["utt"].replace("-", "_").split("_")[0]
            groups.setdefault(speaker, []).append(index)
        for indices in groups.values():
            for position, index in enumerate(indices):
                other = indices[(position + 1) % len(indices)] if len(indices) > 1 else None
                record = records[index]
                record["reference_index"] = other
                record["reference_rate"] = (len(records[other]["speech"]) / len(records[other]["text_ids"])
                                            if other is not None else None)
                if other == index:
                    raise AssertionError("Target leakage into reference rate.")
        splits[name] = records
        summary[name] = dict(usable=len(records), dropped=rejected, speakers=len(groups),
                             source_list=str(listing.resolve()),
                             list_sha256=hashlib.sha256(listing.read_bytes()).hexdigest(),
                             unconditioned=sum(r["reference_rate"] is None for r in records))
        print("PREPARED", name, summary[name], flush=True)
    if set(r["utt"] for r in splits["train"]) & set(r["utt"] for r in splits["dev"]):
        raise ValueError("Train/dev overlap.")
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Empty data split.")
    rates = sorted(len(r["speech"]) / len(r["text_ids"]) for r in splits["train"])
    splits["initial_rate"] = rates[len(rates) // 2]
    splits["summary"] = summary
    write_json(dest, splits)
    write_json(args.model_dir / "data_summary.json", summary)


def batch(record):
    return dict(utts=[record["utt"]], text_tokenizer="dreamon",
                text_token=torch.tensor([record["text_ids"]], dtype=torch.long),
                text_token_len=torch.tensor([len(record["text_ids"])]),
                speech_token=torch.tensor([record["speech"]], dtype=torch.long),
                speech_token_len=torch.tensor([len(record["speech"])]),
                reference_rate=[record["reference_rate"]])


@torch.inference_mode()
def evaluate(model, records, local, rank, world, ratio):
    model.eval()
    model.validation_mask_ratio = ratio
    sums = torch.zeros(len(METRICS) + 1, device=local, dtype=torch.float64)
    for index in range(rank, len(records), world):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(batch(records[index]), local)
        sums[:-1] += torch.stack([result[k].detach().double() for k in METRICS])
        sums[-1] += 1
    dist.all_reduce(sums)
    values = sums[:-1] / sums[-1].clamp_min(1)
    return dict(zip(METRICS, values.tolist())) | {"utterances": int(sums[-1].item()), "mask_ratio": ratio}


@torch.inference_mode()
def generation_probe(model, records, output, step):
    model.eval()
    candidates = [r for r in records if 40 <= len(r["speech"]) <= 100][:2]
    results = []
    for r in candidates:
        ref = records[r["reference_index"]] if r["reference_index"] is not None else None
        length, info = predict_speech_length(
            model.bridge, r["text"], ref["text"] if ref else "",
            len(ref["speech"]) if ref else 0, maximum=350)
        for mode, n in (("learned", length), ("oracle", len(r["speech"]))):
            for temp in (0.0, 0.8):
                tokens, report = model.bridge.generate(
                    r["text"], "", torch.empty((1, 0), dtype=torch.long),
                    speech_tokens=n, temperature=temp, seed=1986)
                name = r["utt"] + "_" + mode + "_t" + str(temp)
                torch.save(tokens, output / (name + ".pt"))
                results.append(dict(utt=r["utt"], text=r["text"], mode=mode, temperature=temp,
                                    tokens=n, target_tokens=len(r["speech"]), length_prediction=info,
                                    unique_tokens=report["unique_tokens"],
                                    adjacent_repeat_ratio=report["adjacent_repeat_ratio"]))
    write_json(output / "generation_probe.json", {"step": step, "results": results,
               "note": "Token diagnostics only; no claim of audible/intelligible speech."})


def train(args):
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=45))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache = json.loads((args.model_dir / "data_cache.json").read_text())
    records, dev = cache["train"], cache["dev"]
    if (args.model_dir / "status.json").exists():
        raise FileExistsError("Training directory already contains a run; choose a fresh tag.")
    print("PHASE loading_model rank", rank, flush=True)
    model = build_route_a_model(
        args.dreamon_model_dir, args.cosyvoice_model_dir, dtype="bf16",
        gradient_checkpointing=True, max_sequence_tokens=args.max_sequence_tokens,
        seed=args.seed, freeze_dreamon=True, use_lora=True,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=.05,
        lora_lr=args.lora_lr, duration_lr=args.duration_lr,
        duration_loss_weight=args.duration_loss_weight, initial_rate=cache["initial_rate"],
        mask_min=.1, mask_max=1., full_mask_probability=.25, validation_mask_ratio=1.)
    model.cuda(local)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(),
                                  lr=args.projection_lr, weight_decay=.01)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], find_unused_parameters=False, gradient_as_bucket_view=True)
    print("PHASE ddp_ready rank", rank, flush=True)
    sampler = DistributedSampler(records, num_replicas=world, rank=rank, seed=args.seed,
                                 shuffle=True, drop_last=False)
    writer = SummaryWriter(str(args.model_dir / "tensorboard")) if rank == 0 else None
    params = [p for p in model.parameters() if p.requires_grad]
    counts = model.trainable_parameter_counts()
    if rank == 0:
        manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        manifest.update(world_size=world, gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        batch_size_per_gpu=1, effective_batch_size=world, accum_grad=1,
                        trainable_parameters=counts, train_samples=len(records), dev_samples=len(dev),
                        steps_per_epoch=len(sampler), initial_rate=cache["initial_rate"],
                        initialization="original_pretrained_DreamOn_and_CosyVoice_plus_fresh_LoRA_projections_duration",
                        checkpoint_scope="full_DreamOn_LoRA_projections_duration; frozen_CosyVoice_from_original",
                        reference_rate="another same-speaker utterance within the corresponding split",
                        speech_conditioning="text_only; reference_audio_only_for_duration_and_Flow_HiFT",
                        early_stopping=False)
        write_json(args.model_dir / "manifest.json", manifest)
    dist.barrier()
    step = 0
    started = time.monotonic()
    validation_time = 0.
    best_ce = float("inf")
    # Balanced short/long fixed diagnostic subset; full dev-clean+dev-other at each epoch.
    subset_indices = sorted(random.Random(args.seed).sample(range(len(dev)), min(80, len(dev))))
    cv_subset = [dev[i] for i in subset_indices]
    with (args.model_dir / "metrics.jsonl").open("a") if rank == 0 else open(os.devnull, "w") as metrics_file:
        try:
            for epoch in range(args.epochs):
                sampler.set_epoch(epoch)
                epoch_started = time.monotonic()
                for bidx, index in enumerate(sampler, 1):
                    ddp.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        losses = ddp(batch(records[index]), local)
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError("Non-finite training loss.")
                    losses["loss"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Non-finite gradient.")
                    optimizer.step()
                    step += 1
                    if step <= 3 or step % 10 == 0:
                        values = torch.stack([losses[k].detach() for k in METRICS] + [norm.detach()])
                        dist.all_reduce(values)
                        values /= world
                        if rank == 0:
                            result = dict(zip(METRICS + ("grad_norm",), values.tolist()))
                            elapsed_training = max(.001, time.monotonic() - started - validation_time)
                            eta = (len(sampler) - bidx) * elapsed_training / step
                            record = dict(status="training", epoch=epoch, batch=bidx, total_batches=len(sampler),
                                          step=step, epoch_eta_seconds=eta, elapsed_seconds=time.monotonic()-started,
                                          **result)
                            print("TRAIN Epoch", epoch, "Batch", str(bidx)+"/"+str(len(sampler)),
                                  json.dumps(result), "ETA(s)", round(eta), flush=True)
                            write_json(args.model_dir / "status.json", record)
                            metrics_file.write(json.dumps(record)+"\n"); metrics_file.flush()
                            for key, value in result.items():
                                writer.add_scalar("TRAIN/"+key, value, step)
                            for group in optimizer.param_groups:
                                writer.add_scalar("LR/"+group["name"], group["lr"], step)
                    end = bidx == len(sampler) or (args.max_steps and step >= args.max_steps)
                    save_now = step in (100, 1000) or end
                    if save_now:
                        check_started = time.monotonic()
                        if rank == 0:
                            write_json(args.model_dir / "status.json",
                                       dict(status="validating", epoch=epoch, step=step, full_dev=bool(end)))
                        cv_records = dev if end else cv_subset
                        cv_full = evaluate(model, cv_records, local, rank, world, 1.)
                        cv_half = evaluate(model, cv_subset, local, rank, world, .5)
                        dist.barrier()
                        if rank == 0:
                            name = "epoch_%d_whole" % epoch if end else "step_%06d" % step
                            if shutil.disk_usage(args.model_dir).free < 35 * 1024**3:
                                raise RuntimeError("Less than 35 GiB free; stopping before checkpoint write.")
                            payload = model.training_checkpoint(dict(epoch=epoch, step=step))
                            payload["optimizer_steps_completed"] = step
                            dest = args.model_dir / (name+".pt")
                            temporary = dest.with_suffix(".pt.tmp")
                            torch.save(payload, temporary); temporary.replace(dest); del payload
                            state = dict(optimizer=optimizer.state_dict(), epoch=epoch, step=step)
                            temp = args.model_dir / (name+"_optimizer.pt.tmp")
                            torch.save(state, temp); temp.replace(args.model_dir / (name+"_optimizer.pt"))
                            result = dict(epoch=epoch, step=step, checkpoint=str(dest),
                                          CV_full=cv_full, CV_half=cv_half,
                                          epoch_elapsed_seconds=time.monotonic()-epoch_started)
                            write_json(args.model_dir / (name+".json"), result)
                            for label, vals in (("CV_full", cv_full), ("CV_half", cv_half)):
                                for key in METRICS:
                                    writer.add_scalar(label+"/"+key, vals[key], step)
                            if end and cv_full["token_ce"] < best_ce:
                                best_ce = cv_full["token_ce"]
                                write_json(args.model_dir / "best_checkpoint.json",
                                           dict(checkpoint=str(dest), full_mask_token_ce=best_ce, epoch=epoch))
                            print("CHECKPOINT", json.dumps(result), flush=True)
                            probe_dir = args.model_dir / "probes" / name
                            probe_dir.mkdir(parents=True)
                            generation_probe(model, dev, probe_dir, step)
                            writer.flush()
                        dist.barrier()
                        validation_time += time.monotonic()-check_started
                    if args.max_steps and step >= args.max_steps:
                        break
                if args.max_steps and step >= args.max_steps:
                    break
            if rank == 0:
                write_json(args.model_dir / "status.json",
                           dict(status="complete", step=step, epochs_completed=epoch+1,
                                elapsed_seconds=time.monotonic()-started))
        except Exception:
            write_json(args.model_dir / ("failure_rank_%d.json" % rank),
                       dict(status="failed", step=step, error=traceback.format_exc()))
            if rank == 0:
                write_json(args.model_dir / "status.json",
                           dict(status="failed", step=step, error=traceback.format_exc()))
            raise
        finally:
            if writer:
                writer.close()
            dist.destroy_process_group()


def main():
    args = parser().parse_args()
    if args.epochs < 1 or args.max_steps < 0:
        raise ValueError("Invalid epoch/step count.")
    os.chdir(ROOT)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prepare_only:
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
