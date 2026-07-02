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

# CER — point metrics only
# srun python3 -m scripts.run_chronos dataset=cer

# CER — with probabilistic metrics (WQL/CRPS)
# srun python3 -m scripts.run_chronos dataset=cer model.probabilistic=true

srun python3 -m src.baselines.run_chronos --multirun \
    dataset=cer_bis \
    model.probabilistic=true \
    dataset.context_length=512,336,672,1008 \
    dataset.prediction_length=96,48,336,672