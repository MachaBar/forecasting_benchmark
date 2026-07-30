#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=tsfm
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail
mkdir -p jobs
source .venv/bin/activate

#!/usr/bin/env bash
# Lance tous les modèles de fondation sur un dataset donné.
#
# Prérequis : configs/dataset/<DATASET>.yaml doit exister (copie et adapte
# configs/dataset/cer_bis.yaml : path, path_client_split, context_length, ...).
#
# Usage :
#   bash scripts/run_foundation_models.sh <dataset_name>
#   bash scripts/run_foundation_models.sh ideal

DATASET="${1:?Usage: $0 <dataset_name>  (ex: ideal)}"

MODELS=(
    chronos
    tsicl
    patchtst_fm
    tirex
)

echo "=== Foundation models × dataset=${DATASET} ==="

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo ">>> Running model=${MODEL} on dataset=${DATASET}"
    # python -m src.baselines.run_foundation \
    #     dataset="${DATASET}" \
    #     model="${MODEL}"

    #--- Variante multirun (balayage context_length/prediction_length) ---
    srun python -u -m src.baselines.run_foundation --multirun \
        dataset="${DATASET}" \
        model="${MODEL}" \
        dataset.context_length=144,336,672,1440,2200 \
        dataset.prediction_length=96,336
done

echo ""
echo "=== Terminé. Résultats sous outputs/<model>/${DATASET}/ ==="


