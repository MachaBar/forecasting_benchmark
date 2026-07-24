#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
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


srun python -u -m src.baselines.run_foundation --multirun \
    model=chronos dataset=cer_bis model.probabilistic=true \
    model.weights_path=/home/d32485/timetensor/src/timetensor/sota/chronos2/weights \
    dataset.context_length=144,336,672,1440,2500  \
    dataset.prediction_length=336,96


