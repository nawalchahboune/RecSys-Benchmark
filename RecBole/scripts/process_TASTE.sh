#!/bin/bash
#SBATCH --job-name=process_data_TASTE
#SBATCH --partition=Odyssey
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.err

set -euo pipefail
mkdir -p /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs

# Sélection Python (sans activation)
PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
echo "Using Python: $PY"

export PYTHONPATH="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole:${PYTHONPATH:-}"

# Configs
model="SASRecF"
dataset="ml-100k"
source_dir="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole"
data_config="${source_dir}/configs/datasets/ml.yaml"
eval_config="${source_dir}/configs/eval.yaml"
model_config="${source_dir}/configs/models/SASRecF.yaml"
output_dir="benchmark_splits/${dataset}"
DATA_ROOT="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/data-ml"

cd "$source_dir"
ls -l "$DATA_ROOT/${dataset}/"{${dataset}.inter,${dataset}.item,${dataset}.user}

# Préprocess (export des splits)
$PY -u export_data_splits.py \
  --model "$model" \
  --dataset_name "$dataset" \
  --output_dir "$output_dir" \
  --config_file_list "$data_config" "$eval_config" "$model_config"