#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100-bis
#SBATCH --time=6-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=analysis
#SBATCH -o ./jobs/%j.out
#SBATCH -e ./jobs/%j.err

set -euo pipefail
mkdir -p jobs
source .venv/bin/activate

# uv run python scripts/analysis/dataset_analysis.py
# uv run python scripts/analysis/compare_datasets.py


# IDEAL dataset

# 1. look at the zip first (tells you if homes have multiple mains files -> --agg matters)
# uv run python -m src.dataset.prepare_ideal --scan
# uv run python -m src.dataset.prepare_ideal --inspect

# python -u -m src.dataset.prepare_ideal --pareto

# # 2. build the two artefacts
# python -m src.dataset.prepare_ideal --min-timesteps 8760

# REFIT dataset

python -u -m src.dataset.prepare_refit --inspect     # confirme format CSV + liste des maisons
# python -u -m src.dataset.prepare_refit --periods     # couverture par maison
# python -u -m src.dataset.prepare_refit --scan        # compromis clients/durée
# python -u -m src.dataset.prepare_refit --pareto      # toutes les combinaisons atteignables
# python -u -m src.dataset.prepare_refit --min-timesteps 17520   # génère les 2 fichiers