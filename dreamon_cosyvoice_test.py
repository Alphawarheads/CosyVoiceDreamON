#!/usr/bin/env python3
"""Compare CosyVoice2 with a DreamOn speech bridge, using local weights.

python dreamon_cosyvoice_test.py --check-only
python dreamon_cosyvoice_test.py --text "你好，欢迎来到语音实验。"
python dreamon_cosyvoice_test.py --adapter-checkpoint exp/dreamon_speech/epoch_0_whole.pt

Outputs live in a new timestamped directory per run. The experiment first saves
the original LLM's tokens/audio, then frees Qwen and generates speech tokens with
DreamOn. The same reference conditioning, Flow and HiFT decode both sequences.
This is a fixed-length experiment. Without --adapter-checkpoint, projections
are random; loading a trained checkpoint does not automatically verify quality.
"""

import argparse
from datetime import datetime
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import struct
import sys
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "Matcha-TTS"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dreamon_cosyvoice.yaml")
    for name in ("text", "prompt-text", "prompt-wav", "dreamon-model-dir",
                 "cosyvoice-model-dir", "output-dir", "device"):
        parser.add_argument("--" + name)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--speech-tokens", dest="initial_speech_tokens", type=int)
    parser.add_argument("--tokens-per-step", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--adapter-checkpoint", help="DreamOn experiment checkpoint (projections, optionally backbone)")
    parser.add_argument("--math-sdpa", action="store_true", default=None)
    parser.add_argument("--check-only", action="store_true", help="Check files/dependencies; load no models.")
    return parser.parse_args()


def load_config(args):
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is missing. Use the project's Python environment.") from error
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    config.setdefault("adapter_checkpoint", None)
    for key, value in vars(args).items():
        if key not in ("config", "check_only") and value is not None:
            config[key] = value
    required = {
        "dreamon_model_dir", "cosyvoice_model_dir", "output_dir", "text", "prompt_text",
        "prompt_wav", "device", "dtype", "math_sdpa", "seed", "speech_vocab_size",
        "cosyvoice_hidden_size", "dreamon_hidden_size", "max_sequence_tokens",
        "initial_speech_tokens", "tokens_per_step", "temperature", "dynamic_length", "stream",
        "adapter_checkpoint",
    }
    if set(config) != required:
        raise ValueError(f"Config missing keys: {sorted(required - set(config))}; "
                         f"unknown keys: {sorted(set(config) - required)}")
    for key in ("dreamon_model_dir", "cosyvoice_model_dir", "output_dir", "prompt_wav"):
        path = Path(config[key]).expanduser()
        config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    if config["adapter_checkpoint"] is not None:
        path = Path(config["adapter_checkpoint"]).expanduser()
        config["adapter_checkpoint"] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    for key in ("text", "prompt_text"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be non-empty; prompt_text must match the reference audio.")
    for key in ("speech_vocab_size", "cosyvoice_hidden_size", "dreamon_hidden_size",
                "max_sequence_tokens", "initial_speech_tokens", "tokens_per_step"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["initial_speech_tokens"] >= config["max_sequence_tokens"]:
        raise ValueError("The speech canvas must leave context space for text/reference audio.")
    if config["speech_vocab_size"] != 6561:
        raise ValueError("This experiment targets the CosyVoice2 codec with 6561 speech IDs.")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if (not isinstance(config["temperature"], (int, float))
            or not math.isfinite(config["temperature"]) or config["temperature"] < 0):
        raise ValueError("temperature must be finite and non-negative.")
    if config["dtype"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("dtype must be bf16, fp16 or fp32.")
    if not isinstance(config["device"], str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", config["device"]):
        raise ValueError("device must be cpu, cuda, or cuda:N.")
    for key in ("math_sdpa", "dynamic_length", "stream"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a YAML boolean.")
    if config["dynamic_length"] or config["stream"]:
        raise ValueError("This experiment supports only dynamic_length: false and stream: false.")
    return config


def check_environment(config):
    """Read metadata only, so missing dependencies are reported before 7B loading."""
    issues = []
    dream = Path(config["dreamon_model_dir"])
    cosy = Path(config["cosyvoice_model_dir"])
    files = [dream / name for name in (
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "modeling_dream.py", "configuration_dream.py", "generation_utils.py",
        "tokenization_dream.py", "tokenizer_config.json", "vocab.json", "merges.txt",
    )]
    files += [cosy / name for name in (
        "cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx",
        "speech_tokenizer_v2.onnx", "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/model.safetensors", "CosyVoice-BlankEN/tokenizer_config.json",
        "CosyVoice-BlankEN/vocab.json", "CosyVoice-BlankEN/merges.txt",
    )]
    files.append(Path(config["prompt_wav"]))
    if config["adapter_checkpoint"] is not None:
        files.append(Path(config["adapter_checkpoint"]))
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"Missing or empty file: {path}")

    dimensions = {}
    try:
        dream_config = json.loads((dream / "config.json").read_text(encoding="utf-8"))
        cosy_config = json.loads((cosy / "CosyVoice-BlankEN/config.json").read_text(encoding="utf-8"))
        for name, actual in (("dreamon_hidden_size", dream_config["hidden_size"]),
                             ("cosyvoice_hidden_size", cosy_config["hidden_size"])):
            dimensions[name] = actual
            if config[name] != actual:
                issues.append(f"{name}: configured {config[name]}, checkpoint {actual}")
        index = json.loads((dream / "model.safetensors.index.json").read_text(encoding="utf-8"))
        for shard in set(index["weight_map"].values()):
            path = dream / shard
            if not path.is_file() or path.stat().st_size < 1024:
                issues.append(f"Missing/invalid DreamOn weight shard: {path}")
        # Check the actual embedding tensor header, not just the model config.
        tensor_name = "model.embed_tokens.weight"
        with (dream / index["weight_map"][tensor_name]).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if not 0 < header_size < 100_000_000:
                raise ValueError("Invalid safetensors header (possibly a Git LFS pointer).")
            header = json.loads(handle.read(header_size))
        dimensions["dreamon_embedding_shape"] = header[tensor_name]["shape"]
        if dimensions["dreamon_embedding_shape"] != [dream_config["vocab_size"], dream_config["hidden_size"]]:
            issues.append("DreamOn embedding tensor shape disagrees with config.json.")
    except (OSError, ValueError, KeyError, struct.error) as error:
        issues.append(f"Checkpoint metadata check failed: {error}")

    versions = {}
    packages = {
        "torch": "torch", "torchaudio": "torchaudio", "transformers": "transformers",
        "accelerate": "accelerate", "numpy": "numpy", "soundfile": "soundfile",
        "hyperpyyaml": "HyperPyYAML", "onnxruntime": "onnxruntime",
        "whisper": "openai-whisper", "modelscope": "modelscope", "matcha": None,
    }
    for module, distribution in packages.items():
        if importlib.util.find_spec(module) is None:
            hint = (" Populate third_party/Matcha-TTS with the Matcha-TTS source "
                    "(the archive's submodule directory may be empty)." if module == "matcha" else "")
            issues.append(f"Missing Python module: {module}.{hint}")
        if distribution:
            try:
                versions[module] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                versions[module] = "unknown"
    torch_version = versions.get("torch", "unknown").split("+")[0]
    numbers = tuple(int(n) for n in re.findall(r"\d+", torch_version)[:3])
    if numbers and numbers < (2, 5, 1):
        issues.append("Use torch >= 2.5.1 for this DreamOn experiment; the original "
                      "CosyVoice requirements pin 2.3.1. Keep torchaudio matched to torch.")
    if (versions.get("torchaudio", "unknown") != "unknown" and torch_version != "unknown"
            and versions["torchaudio"].split("+")[0] != torch_version):
        issues.append("torch and torchaudio release versions must match.")
    return {"issues": issues, "versions": versions, "dimensions": dimensions}


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def decode_and_save(cosyvoice, conditioning, tokens, path):
    import soundfile as sf
    import torch
    from cosyvoice.llm.dreamon_speech import validate_speech_tokens

    validate_speech_tokens(tokens)
    # Only flow conditions are passed: no text or LLM prompt can generate a fallback.
    flow_inputs = {key: conditioning[key] for key in (
        "flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding",
    )}
    chunks = [item["tts_speech"] for item in cosyvoice.model.tts(
        **flow_inputs, source_speech_token=tokens, stream=False,
    )]
    if not chunks:
        raise RuntimeError("The audio decoder produced no chunks.")
    waveform = torch.cat(chunks, dim=1).detach().cpu().float()
    if waveform.numel() == 0 or not torch.isfinite(waveform).all().item():
        raise FloatingPointError("The audio decoder produced empty or non-finite audio.")
    peak = float(waveform.abs().max())
    sf.write(str(path), waveform.squeeze(0).numpy(), cosyvoice.sample_rate, subtype="PCM_16")
    return {"file": str(path), "sample_rate": cosyvoice.sample_rate,
            "duration_seconds": waveform.shape[1] / cosyvoice.sample_rate,
            "peak": peak, "rms": float(waveform.square().mean().sqrt()),
            "samples_outside_pcm_range": int(waveform.abs().gt(1).sum()),
            "speech_token_count": tokens.numel()}


def run_experiment(config, output_dir, report, report_path):
    # Set these before importing transformers, and keep generated caches local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    if config["device"] == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from cosyvoice.llm.dreamon_speech import DreamOnSpeechLM, validate_speech_tokens

    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        torch.cuda.set_device(device)
        report["gpu"] = torch.cuda.get_device_name(device)
        if config["dtype"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support BF16; use --dtype fp16 or fp32.")
    elif config["dtype"] != "fp32":
        raise ValueError("CPU execution requires --dtype fp32 and substantial RAM for 7B weights.")
    if config["math_sdpa"]:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config["dtype"]]

    def seed_all():
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])

    report["stage"] = "load_cosyvoice"
    write_report(report_path, report)
    print("[1/4] Loading local CosyVoice2 and extracting reference conditioning...", flush=True)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    cosyvoice = CosyVoice2(config["cosyvoice_model_dir"], fp16=False,
                          load_jit=False, load_trt=False, load_vllm=False)
    # HyperPyYAML seeds RNGs during construction; apply the experiment seed AFTER it.
    seed_all()
    conditioning = cosyvoice.frontend.frontend_zero_shot(
        config["text"], config["prompt_text"], config["prompt_wav"], cosyvoice.sample_rate, "",
    )
    report["stage"] = "baseline"
    write_report(report_path, report)
    print("[2/4] Generating original CosyVoice tokens and baseline.wav...", flush=True)
    # Run the LLM synchronously so exceptions propagate instead of dying in a thread.
    # CosyVoice inference mutates text_len, so pass a clone.
    with torch.inference_mode():
        baseline_ids = list(cosyvoice.model.llm.inference(
            text=conditioning["text"], text_len=conditioning["text_len"].clone(),
            prompt_text=conditioning["prompt_text"],
            prompt_text_len=conditioning["prompt_text_len"].clone(),
            prompt_speech_token=conditioning["llm_prompt_speech_token"],
            prompt_speech_token_len=conditioning["llm_prompt_speech_token_len"].clone(),
            embedding=conditioning["llm_embedding"],
        ))
        baseline_tokens = torch.tensor([baseline_ids], dtype=torch.int32)
        validate_speech_tokens(baseline_tokens, config["speech_vocab_size"])
        torch.save(baseline_tokens, output_dir / "baseline_tokens.pt")
        seed_all()
        report["baseline"] = decode_and_save(cosyvoice, conditioning, baseline_tokens,
                                              output_dir / "baseline.wav")
    write_report(report_path, report)

    components = DreamOnSpeechLM.copy_cosyvoice_components(cosyvoice.model.llm)
    # The experiment uses source_speech_token exclusively after this point.
    cosyvoice.model.llm = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["stage"] = "load_dreamon"
    write_report(report_path, report)
    print("[3/4] Loading DreamOn and speech projections...", flush=True)
    bridge = DreamOnSpeechLM.from_local_weights(
        config["dreamon_model_dir"], components, device=device, dtype=dtype,
        speech_vocab_size=config["speech_vocab_size"], seed=config["seed"],
        max_sequence_tokens=config["max_sequence_tokens"],
    )
    del components
    if config["adapter_checkpoint"] is not None:
        bridge.speech_in_proj.float()
        bridge.speech_out_proj.float()
        bridge.load_adapter_checkpoint(config["adapter_checkpoint"])
    report["adaptation_status"] = bridge.adaptation_status
    report["loaded_finetuned_backbone"] = bridge.backbone_checkpoint_required
    print(f"  Adapter status: {bridge.adaptation_status}", flush=True)
    if bridge.backbone_checkpoint_required:
        # The supplied checkpoint already holds all weights; avoid duplicating 7B.
        report["checkpoint_source"] = config["adapter_checkpoint"]
    else:
        adapter_name = "adapter_loaded.pt" if config["adapter_checkpoint"] else "adapter_initial.pt"
        torch.save(bridge.initial_adapter_checkpoint(), output_dir / adapter_name)
    report["stage"] = "dreamon_generation"
    write_report(report_path, report)

    def progress(step):
        # Persist partial progress, including a useful trace if a later step fails.
        report["last_denoising_step"] = step
        if step["step"] == 1 or step["step"] % 10 == 0 or step["remaining_masks"] == 0:
            print(f"  Denoising step {step['step']}: {step['remaining_masks']} masks left", flush=True)
            write_report(report_path, report)

    started = time.monotonic()
    tokens, diagnostics = bridge.generate(
        config["text"], config["prompt_text"], conditioning["llm_prompt_speech_token"],
        speech_tokens=config["initial_speech_tokens"], tokens_per_step=config["tokens_per_step"],
        temperature=config["temperature"], seed=config["seed"], progress=progress,
    )
    diagnostics["generation_seconds"] = time.monotonic() - started
    report["dreamon"] = diagnostics
    torch.save(tokens, output_dir / "dreamon_tokens.pt")
    write_report(report_path, report)
    del bridge
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[4/4] Decoding DreamOn speech tokens through the same Flow and HiFT...", flush=True)
    report["stage"] = "decode_dreamon"
    write_report(report_path, report)
    seed_all()
    with torch.inference_mode():
        report["dreamon_audio"] = decode_and_save(cosyvoice, conditioning, tokens,
                                                   output_dir / "dreamon_stitched.wav")
    report["status"] = "interface_smoke_test_completed"
    report["stage"] = "done"
    report["speech_quality_verified"] = False
    write_report(report_path, report)
    print(f"Saved experiment to: {output_dir}", flush=True)
    print("Interface test completed. Speech quality still requires listening/evaluation.")


def main():
    args = parse_args()
    try:
        config = load_config(args)
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    try:
        checks = check_environment(config)
    except Exception as error:
        print(f"Environment check failed: {error}", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 2 if checks["issues"] else 0
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output_dir = Path(config["output_dir"]) / name
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "diagnostics.json"
    report = {"status": "started", "stage": "preflight", "config": config,
              "python": sys.version, "checks": checks,
              "adaptation_status": ("pending_adapter_load" if config["adapter_checkpoint"]
                                    else "untrained_random_projections"),
              "speech_quality_verified": False}
    write_report(report_path, report)
    if checks["issues"]:
        report["status"] = "preflight_failed"
        write_report(report_path, report)
        print("Preflight failed:\n- " + "\n- ".join(checks["issues"]), file=sys.stderr)
        print(f"Diagnostics: {report_path}", file=sys.stderr)
        return 2
    try:
        run_experiment(config, output_dir, report, report_path)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(report_path, report)
        traceback.print_exc()
        print(f"Failed during {report['stage']}. Diagnostics: {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
