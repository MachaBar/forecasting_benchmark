#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
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


# # point
# srun python -u -m src.baselines.run_prophet dataset=cer_bis \
#     dataset.context_length=512 dataset.prediction_length=96

# # probabiliste (WQL/SQL/CRPS via predictive_samples)
# srun python -u -m src.baselines.run_prophet dataset=cer_bis \
#     dataset.context_length=512 dataset.prediction_length=96 \
#     model.probabilistic=true

# # tuning des 2 hyperparamètres clés
# srun python -u -m src.baselines.run_prophet --multirun dataset=cer_bis \
#     model.changepoint_prior_scale=0.01,0.05,0.5 \
#     model.seasonality_prior_scale=5,10,15

srun python -u -m src.baselines.run_prophet --multirun \
    dataset=cer_bis \
    dataset.context_length=512 dataset.prediction_length=96 \
    eval_split=val \
    model.tune_n_clients=50 \
    model.changepoint_prior_scale=0.01,0.05,0.5 \
    model.seasonality_prior_scale=1.0,5.0,10.0

# srun python -u -m src.baselines.run_prophet \
#     dataset=cer_bis \
#     dataset.context_length=1440 dataset.prediction_length=96 \
#     model.probabilistic=true \
#     model.changepoint_prior_scale=<best> \
#     model.seasonality_prior_scale=<best>