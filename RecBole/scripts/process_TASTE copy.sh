#!/bin/bash
#SBATCH --job-name=process_data_TASTE
#SBATCH --partition=Odyssey
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.out
#SBATCH --error=/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs/%x-%j.err
# SBATCH --mail-type=END
# SBATCH --mail-user="votre.email@exemple.com"

set -euo pipefail

# Dossiers de sortie pour logs
mkdir -p /Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole/scripts/outputs

# Activer le venv du projet
source /Odyssey/private/n23chahb/compet/RecSys-Benchmark/.venv/bin/activate

# Configs
model="SASRec"             # Laissez SASRec pour que Recbole détecte un modèle supporté
dataset_type="ml-100k"
dataset="ml"

# Chemins locaux
source_dir="/Odyssey/private/n23chahb/compet/RecSys-Benchmark/RecBole"
exp_name="/Odyssey/private/n23chahb/compet/checkpoints/SASRecF_${dataset}"
nproc=1

model_config="${source_dir}/configs/models/${model}.yaml"
data_config="${source_dir}/configs/datasets/${dataset_type}.yaml"
eval_config="${source_dir}/configs/eval.yaml"

cd "$source_dir"

# Astuce: si vous avez modifié run_recbole.py pour utiliser gen_dataset_TASTE local,
# assurez-vous que l'import correspond (ligne commentée dans le fichier).
python run_recbole.py \
  --dataset "$dataset" \
  --exp_name "$exp_name" \
  --nproc "$nproc" \
  --config_files "${model_config} ${data_config} ${eval_config}" \
  --data_preprocess