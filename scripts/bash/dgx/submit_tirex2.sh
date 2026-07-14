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

export HF_HUB_OFFLINE=1
# srun python -u -m src.baselines.run_foundation \
#     model=tirex dataset=cer_bis model.probabilistic=true \
#     dataset.context_length=512 dataset.prediction_length=96 \
#     model.weights_path=/home/d32485/forecasting_benchmark/checkpoints/tirex2

srun python -u -m src.baselines.run_foundation --multirun \
    model=tirex dataset=cer_bis model.probabilistic=true \
    model.weights_path=/home/d32485/forecasting_benchmark/checkpoints/tirex2 \
    dataset.context_length=144,336,672,1440 \
    dataset.prediction_length=96