#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# Supply --train_data and --cv_data pointing to lists of JSONL/parquet shards.
# NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash examples/dreamon/run_train.sh ...
"${PYTHON_BIN:-python}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m cosyvoice.bin.train \
  --train_engine torch_ddp --model llm \
  --config configs/dreamon_cosyvoice_train.yaml \
  --model_dir exp/dreamon_speech \
  --tensorboard_dir tensorboard/dreamon_speech \
  --num_workers 0 --prefetch 2 --use_amp \
  "$@"
