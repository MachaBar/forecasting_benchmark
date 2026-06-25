#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500G
#SBATCH --job-name=baselines
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail
mkdir -p jobs
source .venv/bin/activate

REPO=/home/d32485/forecasting_benchmark

# # CER — point forecast
# srun python3 -m src.baselines.run_statistical_baselines \
#   dataset=cer \
#   dataset.path=$REPO/data/cer/load_curve.parquet \
#   dataset.path_client_split=$REPO/data/cer/split.pkl

# CER — probabiliste
srun python3 -m src.baselines.run_statistical_baselines \
  dataset=cer \
  model.probabilistic=true \
  model.max_lookback=512 

# # SMACH — saison horaire
# srun python3 -m src.baselines.run_statistical_baselines \
#   dataset=smach \
#   dataset.path=$REPO/data/smach/data.parquet \
#   dataset.path_client_split=$REPO/data/smach/split.pkl \
#   model.season_length=48