#!/usr/bin/env bash
# One entry point: prepare LibriTTS -> train -> generate -> WER evaluation.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}; DATA_ROOT=${DATA_ROOT:-$ROOT/data/libritts}
DATASETS=${DATASETS:-train-clean-100,train-clean-360,train-other-500}; DEV_DATASET=${DEV_DATASET:-dev-clean}
EVAL_DATASET=${EVAL_DATASET:-test-clean}; STAGE=${STAGE:-all}; FREEZE_DREAMON=${FREEZE_DREAMON:-true}
MODEL_DIR=${MODEL_DIR:-$ROOT/exp/dreamon_frozen}; GEN_DIR=${GEN_DIR:-$ROOT/outputs/dreamon_eval}
SEED_EVAL_DIR=${SEED_EVAL_DIR:-$ROOT/../seed-tts-eval}; MAX_EVAL=${MAX_EVAL:-100}; NPROC_PER_NODE=${NPROC_PER_NODE:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }; command -v "$PYTHON_BIN" >/dev/null || die "Activate the Python/CUDA environment first."
IFS=',' read -r -a TRAIN_PARTS <<< "$DATASETS"

prepare_part(){
  local part=$1 src="$DATA_ROOT/$1" meta="$DATA_ROOT/meta/$1" pq="$DATA_ROOT/parquet/$1"; mkdir -p "$meta" "$pq"
  [[ -d "$src" ]] || die "Missing $src; extract the archive there first."
  [[ -f "$meta/wav.scp" ]] || "$PYTHON_BIN" examples/libritts/cosyvoice/local/prepare_data.py --src_dir "$src" --des_dir "$meta"
  [[ -f "$meta/utt2speech_token.pt" ]] || "$PYTHON_BIN" tools/extract_speech_token.py --dir "$meta" --onnx_path "$ROOT/CosyVoice2-0.5B/speech_tokenizer_v2.onnx"
  [[ -f "$pq/data.list" ]] || "$PYTHON_BIN" tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 1 --src_dir "$meta" --des_dir "$pq"
}
prepare(){
  mkdir -p "$DATA_ROOT/lists"; for p in "${TRAIN_PARTS[@]}" "$DEV_DATASET" "$EVAL_DATASET"; do prepare_part "$p"; done
  : > "$DATA_ROOT/lists/train.data.list"; for p in "${TRAIN_PARTS[@]}"; do cat "$DATA_ROOT/parquet/$p/data.list" >> "$DATA_ROOT/lists/train.data.list"; done
  cp "$DATA_ROOT/parquet/$DEV_DATASET/data.list" "$DATA_ROOT/lists/dev.data.list"; cp "$DATA_ROOT/parquet/$EVAL_DATASET/data.list" "$DATA_ROOT/lists/eval.data.list"
  "$PYTHON_BIN" tools/prepare_dreamon_eval.py --data-dir "$DATA_ROOT/meta/$EVAL_DATASET" --output "$DATA_ROOT/lists/eval.meta.lst" --limit "$MAX_EVAL"
}
train(){ [[ -f "$DATA_ROOT/lists/train.data.list" ]] || prepare; local suffix=frozen; [[ "$FREEZE_DREAMON" == true ]] || suffix=unfrozen
  NPROC_PER_NODE="$NPROC_PER_NODE" bash examples/dreamon/run_train.sh --freeze_dreamon "$FREEZE_DREAMON" --model_dir "$MODEL_DIR" --tensorboard_dir "$ROOT/tensorboard/dreamon_$suffix" --train_data "$DATA_ROOT/lists/train.data.list" --cv_data "$DATA_ROOT/lists/dev.data.list"; }
generate(){ "$PYTHON_BIN" dreamon_generate.py --meta "$DATA_ROOT/lists/eval.meta.lst" --backend dreamon --adapter-checkpoint "$MODEL_DIR/epoch_0_whole.pt" --output-dir "$GEN_DIR" --length-mode prompt-rate --max-speech-tokens 750; }
evaluate(){ [[ -d "$SEED_EVAL_DIR" ]] || die "Clone https://github.com/BytedanceSpeech/seed-tts-eval and set SEED_EVAL_DIR."; "$PYTHON_BIN" dreamon_eval.py --meta "$GEN_DIR/meta.lst" --wav-dir "$GEN_DIR/wavs" --seed-eval-dir "$SEED_EVAL_DIR" --output-dir "$GEN_DIR/wer" --metric wer --language en; }
case "$STAGE" in prepare) prepare;; train) train;; generate) generate;; eval) evaluate;; all) prepare; train; generate; evaluate;; *) die "Use STAGE=prepare|train|generate|eval|all";; esac
