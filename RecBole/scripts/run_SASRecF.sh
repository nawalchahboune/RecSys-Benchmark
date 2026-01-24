#!/bin/bash
#SBATCH --job-name=sasrecf_amzn_train
#SBATCH --partition=Odyssey
#SBATCH --gres=gpu:1        # ou gpu:1 pour “n’importe quel GPU”; autres choix: gpu:h100:1, gpu:l40s:1, gpu:rtx8000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.err




set -euo pipefail
mkdir -p /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs

PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
echo "Using Python: $PY"

export PYTHONPATH="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole:${PYTHONPATH:-}"

model="SASRecF"
dataset="amzn"
source_dir="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole"
model_config="${source_dir}/configs/models/SASRecF.yaml"
data_config="${source_dir}/configs/datasets/amzn.yaml"
eval_config="${source_dir}/configs/eval.yaml"
exp_name="SASRecF_${dataset}"

DATA_ROOT="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/Amazon_Beauty"
ls -l "$DATA_ROOT/${dataset}/"{${dataset}.inter,${dataset}.item}


# ...existing code...
$PY - <<'PY'
import sys, torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('torch version:', torch.__version__)
print('CUDA build:', getattr(torch.version, 'cuda', None))
sys.exit(0 if torch.cuda.is_available() else 1)
PY

# Stop if CUDA not available in this venv
if [[ $? -ne 0 ]]; then
  echo "CUDA not available in this Python env. Install GPU wheels for PyTorch."
  exit 1
fi
# ...existing code...

cd "$source_dir"
$PY run_recbole.py \
  --model "$model" \
  --dataset "$dataset" \
  --exp_name "$exp_name" \
  --nproc 1 \
  --config_files "$model_config" "$data_config" "$eval_config"