#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=baseline
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail
mkdir -p jobs
source .venv/bin/activate

srun python -u -m src.baselines.run_lgbm \
    dataset=cer_bis \
    dataset.context_length=1440 \
    dataset.prediction_length=96 \
    model.probabilistic=true