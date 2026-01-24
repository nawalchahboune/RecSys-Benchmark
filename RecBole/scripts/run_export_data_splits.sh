#!/bin/bash

#SBATCH --job-name=export_ml100k
#SBATCH --partition=Odyssey
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p outputs

# Sélection de Python (venv partagé)
PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Python introuvable: $PY" >&2
  exit 1
fi
echo "Using Python: $PY"

# RecBole local
# export PYTHONPATH="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole:$PYTHONPATH"
export PYTHONPATH="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole:${PYTHONPATH:-}"


model="SASRecF"
dataset_type="ml"
dataset="ml-100k"
exp_name="${model}_${dataset}"
nproc=2

source_dir="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole"

data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"
model_config="${source_dir}/configs/models/${model}.yaml"

output_dir="benchmark_splits/${dataset}"

cd "$source_dir"

# Vérifier les fichiers atomiques attendus sous data_path/${dataset}/
DATA_ROOT="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/data-ml"
ls -l "$DATA_ROOT/${dataset}/"{${dataset}.inter,${dataset}.item,${dataset}.user}

# Export des splits
# $PY -u export_data_splits.py \
#   --model "$model" \
#   --dataset_name "$dataset" \
#   --output_dir "$output_dir" \
#   --config_file_list "$data_config" "$eval_config"

$PY -u export_data_splits.py \
  --model="$model" \
  --dataset_name="ml-100k" \
  --output_dir="benchmark_splits/ml-100k" \
  --config_file_list "$data_config" "$eval_config" "$model_config"
