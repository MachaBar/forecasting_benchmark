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
srun python3 -m src.baselines.run_statistical_baselines
