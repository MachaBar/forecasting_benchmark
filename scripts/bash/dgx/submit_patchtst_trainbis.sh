#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=train
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail
mkdir -p jobs
source .venv/bin/activate


python -m scripts.train.train_patchtst \
    dataset=cer_bis \
    train.max_steps=200000 \
    dataset.context_length=512 \
    dataset.prediction_length=96 \
