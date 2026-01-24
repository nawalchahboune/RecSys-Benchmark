#!/bin/bash

#SBATCH --job-name=gen_all_items
#SBATCH --partition=Odyssey
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/TASTE/reproduce/dataprocess/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/TASTE/reproduce/dataprocess/outputs/%x-%j.err

#!/bin/bash
# ...existing code...
set -euo pipefail
mkdir -p /Odyssey/private/n23chahb/compet/RecSys-Benchmark/TASTE/reproduce/dataprocess/outputs

PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
echo "Using Python: $PY"

DATASET="ml-100k"
SRC_ITEM="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/benchmark_splits/${DATASET}/item"
DATA_DIR="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/TASTE/data/${DATASET}"
mkdir -p "$DATA_DIR"
ITEM_FILE="${DATA_DIR}/item.txt"

# Export so Python sees them
export SRC_ITEM ITEM_FILE

# Convert RecBole item -> item.txt (item_id \t title)
$PY - <<'PY'
import os, pandas as pd
src = os.environ["SRC_ITEM"]
dst = os.environ["ITEM_FILE"]
df = pd.read_csv(src, sep="\t")
title_col = "movie_title" if "movie_title" in df.columns else ("title" if "title" in df.columns else None)
if title_col is None:
    raise SystemExit(f"Title column not found in {src}. Available: {list(df.columns)}")
df_out = df[["item_id", title_col]].rename(columns={title_col: "title"})
df_out.to_csv(dst, sep="\t", index=False, header=False)
print(f"Wrote {dst} with {len(df_out)} rows")
PY

TOKENIZER="google-t5/t5-base"
SCRIPT="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/TASTE/gen_all_items.py"

$PY "$SCRIPT" \
  --data_name ml \
  --item_file "$ITEM_FILE" \
  --output "item_name.jsonl" \
  --output_dir "$DATA_DIR" \
  --tokenizer "$TOKENIZER"
# ...existing code...