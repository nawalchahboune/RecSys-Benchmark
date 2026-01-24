#!/bin/bash
#SBATCH --job-name=sasrecf_ml100k_eval
#SBATCH --partition=Odyssey
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --time=10:00:00
#SBATCH -D /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.err

set -euo pipefail
mkdir -p /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs

PY="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/python"
if [[ ! -x "$PY" ]]; then PY="python3"; fi
echo "Using Python: $PY"

export PYTHONPATH="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole:${PYTHONPATH:-}"

model="SASRecF"
dataset="ml-1m"
source_dir="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole"
model_config="${source_dir}/configs/models/SASRecF.yaml"
data_config="${source_dir}/configs/datasets/ml.yaml"
eval_config="${source_dir}/configs/eval.yaml"
exp_name="SASRecF_${dataset}"

RESUME_PATH="${RESUME_PATH:-${1:-}}"

DATA_ROOT="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/data-ml"
ls -l "$DATA_ROOT/${dataset}/"{${dataset}.inter,${dataset}.item,${dataset}.user}


OUT_BIN="${source_dir}/submissions/${exp_name}-$(date +%Y%m%d-%H%M%S).bin"
mkdir -p "${source_dir}/submissions"

export SAVE_SUBMISSION_PATH="${OUT_BIN}"
export SUBMISSION_K="10"  # adapte K


# CKPT=""
# if [[ -n "$RESUME_PATH" && -f "$RESUME_PATH" ]]; then
#   CKPT="$RESUME_PATH"
# else
#   CKPT_DIRS=( "${source_dir}/saved" "${source_dir}/logs" )
#   CKPT=$(find "${CKPT_DIRS[@]}" -type f -name "*.pth" -printf "%T@ %p\n" 2>/dev/null \
#     | sort -nr | awk '{print $2}' \
#     | grep -E "/${exp_name}.*\.pth$|/SASRecF.*${dataset}.*\.pth$" | head -n1)
#   if [[ -z "$CKPT" ]]; then
#     CKPT=$(find "${CKPT_DIRS[@]}" -type f -name "*.pth" -printf "%T@ %p\n" 2>/dev/null \
#       | sort -nr | awk '{print $2}' | head -n1)
#   fi
# fi


CKPT=""
if [[ -n "$RESUME_PATH" && -f "$RESUME_PATH" ]]; then
  CKPT="$RESUME_PATH"
fi
if [[ -z "${CKPT}" ]]; then
  echo "Missing checkpoint path"; exit 1
fi

if [[ -z "${CKPT}" ]]; then
  echo "Aucun checkpoint .pth trouvé. Passez RESUME_PATH en argument."
  exit 1
fi



TMP_EVAL="${source_dir}/configs/eval_only.tmp.yaml"
cat > "${TMP_EVAL}" <<EOF
eval_only: True
resume_path: "${CKPT}"
show_progress: False
use_gpu: False
enable_amp: False
enable_scaler: False
save_submission_path: "${OUT_BIN}"
topk: [10]
EOF
echo "Submission path: ${OUT_BIN}"


cd "$source_dir"
$PY run_recbole.py \
  --model "$model" \
  --dataset "$dataset" \
  --exp_name "$exp_name" \
  --nproc 1 \
  --config_files "$model_config" "$data_config" "$eval_config" "$TMP_EVAL"

