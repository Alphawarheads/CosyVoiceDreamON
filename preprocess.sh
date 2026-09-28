#!/usr/bin/env bash
# Prepare LibriTTS for DreamOn x CosyVoice2. No training or generation here.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); cd "$ROOT"
python_bin=${PYTHON_BIN:-python}; source_dir=${1:-/home/lize/AudioData/libritts}
raw_dir=${RAW_DIR:-$source_dir/raw}; data_dir=${DATA_DIR:-$source_dir/prepared}
num_threads=${NUM_THREADS:-4}; num_processes=${NUM_PROCESSES:-1}
cosyvoice_model_dir=${COSYVOICE_MODEL_DIR:-$ROOT/CosyVoice2-0.5B}
train_parts=(train-clean-100 train-clean-360 train-other-500); dev_parts=(dev-clean dev-other)
parts=("${train_parts[@]}" "${dev_parts[@]}")
die(){ printf 'ERROR: %s\n' "$*" >&2; exit 1; }; log(){ printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
[[ ${1:-} != --help && ${1:-} != -h ]] || { echo "Usage: bash preprocess.sh [/home/lize/AudioData/libritts]"; exit 0; }
command -v "$python_bin" >/dev/null || die "Activate the Python environment first."
[[ -s $cosyvoice_model_dir/speech_tokenizer_v2.onnx ]] || die "Missing speech_tokenizer_v2.onnx"
for value in "$num_threads" "$num_processes"; do [[ $value =~ ^[1-9][0-9]*$ ]] || die "Worker counts must be positive."; done
for path in "$ROOT" "$source_dir" "$raw_dir" "$data_dir"; do [[ ! $path =~ [[:space:]] ]] || die "Path contains spaces: $path"; done
mkdir -p "$raw_dir" "$data_dir/lists"

validate_list(){ "$python_bin" - "$1" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
p=Path(sys.argv[1]); names=[x for x in p.read_text().splitlines() if x]
if not names or len(names)!=len(set(names)): raise ValueError(f"Empty/duplicate list: {p}")
rows=0
for name in names:
    q=Path(name)
    if not q.is_absolute() or not q.is_file(): raise ValueError(f"Missing/non-absolute shard: {q}")
    f=pq.ParquetFile(q)
    if not {'utt','text','speech_token'}.issubset(f.schema_arrow.names): raise ValueError(f"Missing columns: {q}")
    rows += f.metadata.num_rows
print(f"Validated {len(names)} shards / {rows} rows: {p}")
PY
}
extract_part(){
  local part=$1 archive="$source_dir/$1.tar.gz" target="$raw_dir/LibriTTS/$1" marker="$raw_dir/LibriTTS/$1/.dreamon_extract_done"
  [[ -f $marker ]] && return; [[ -s $archive ]] || die "Missing archive: $archive"; log "Extract $part"
  "$python_bin" - "$archive" "$raw_dir" "$part" <<'PY'
from pathlib import Path
import sys,tarfile
a,d,s=sys.argv[1:]; root=Path(d).resolve(); root.mkdir(parents=True,exist_ok=True)
with tarfile.open(a,'r|gz') as h:
    for m in h:
        target=(root/m.name).resolve()
        if not target.is_relative_to(root/'LibriTTS') or not (m.isfile() or m.isdir()): raise ValueError(m.name)
        h.extract(m,root,filter='data' if hasattr(tarfile,'data_filter') else None)
part=root/'LibriTTS'/s
if not any(part.glob('*/*/*.wav')): raise ValueError(f'No WAV files in {part}')
(part/'.dreamon_extract_done').write_text(str(Path(a).resolve())+'\n')
PY
}
prepare_part(){
  local part=$1 src="$raw_dir/LibriTTS/$1" dest="$data_dir/$1"; mkdir -p "$dest"
  if [[ ! -f $dest/.metadata_done ]]; then log "Metadata $part"; "$python_bin" examples/libritts/cosyvoice/local/prepare_data.py --src_dir "$src" --des_dir "$dest"; for f in wav.scp text utt2spk spk2utt; do [[ -s $dest/$f ]] || die "Empty $dest/$f"; done; touch "$dest/.metadata_done"; fi
  if [[ ! -f $dest/.tokens_done || ! -s $dest/utt2speech_token.pt ]]; then log "Speech tokens $part"; "$python_bin" tools/extract_speech_token.py --dir "$dest" --onnx_path "$cosyvoice_model_dir/speech_tokenizer_v2.onnx" --num_thread "$num_threads"; [[ -s $dest/utt2speech_token.pt ]] || die "Token extraction failed: $part"; touch "$dest/.tokens_done"; fi
  if [[ ! -f $dest/.parquet_done || ! -s $dest/parquet/data.list ]]; then log "Parquet $part"; mkdir -p "$dest/parquet"; "$python_bin" tools/make_parquet_list.py --src_dir "$dest" --des_dir "$dest/parquet" --num_processes "$num_processes"; validate_list "$dest/parquet/data.list"; touch "$dest/.parquet_done"; else validate_list "$dest/parquet/data.list"; fi
}
for part in "${parts[@]}"; do extract_part "$part"; prepare_part "$part"; done
write_combo(){ local name=$1; shift; local out="$data_dir/lists/train_${name}.data.list"; : > "$out.tmp"; for p in "$@"; do cat "$data_dir/$p/parquet/data.list" >> "$out.tmp"; done; mv -- "$out.tmp" "$out"; validate_list "$out"; }
write_combo 100 train-clean-100; write_combo 360 train-clean-360; write_combo 500 train-other-500; write_combo all "${train_parts[@]}"
dev_list="$data_dir/lists/dev_all.data.list"; : > "$dev_list.tmp"; for p in "${dev_parts[@]}"; do cat "$data_dir/$p/parquet/data.list" >> "$dev_list.tmp"; done; mv -- "$dev_list.tmp" "$dev_list"; validate_list "$dev_list"
printf '%s\n' "train_100=$data_dir/lists/train_100.data.list" "train_360=$data_dir/lists/train_360.data.list" "train_500=$data_dir/lists/train_500.data.list" "train_all=$data_dir/lists/train_all.data.list" "dev_all=$dev_list" > "$data_dir/lists/README.txt"
log "Preprocessing complete: $data_dir/lists"
