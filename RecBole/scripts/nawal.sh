#!/bin/bash
#SBATCH --job-name=sasrecf_ml100k_train
#SBATCH --partition=Odyssey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -D /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.err

set -euo pipefail
mkdir -p scripts/outputs

PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
echo "Using Python: $PY"

export WANDB_DISABLED=true
export WANDB_MODE=disabled

srun --cpu-bind=none -u $PY run_recbole.py \
  --model SASRecF \
  --dataset ml-100k \
  --exp_name SASRecF_ml-100k \
  --nproc 1 \
  --config_files configs/models/SASRecF.yaml configs/datasets/ml.yaml configs/eval.yaml