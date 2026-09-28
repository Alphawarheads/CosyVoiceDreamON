#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD:$PWD/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# Avoid the NCCL peer-to-peer initialization hang previously reproduced on this host.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-4}" -m cosyvoice.bin.train_route_b "$@"
