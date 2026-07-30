#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=a100
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


srun python -u -m scripts.analysis.make_report \
    --csv /home/d32485/forecasting_benchmark/outputs/chronos2/cer_bis/multirun_2026-07-29_13-33-20/summary_all_runs.csv \
    --metrics mae normalized_mae mase wql sql \
    --group-by model \
    --split-by context_length \
    --outdir /home/d32485/forecasting_benchmark/outputs/chronos2/cer_bis/multirun_2026-07-29_13-33-20/report \
    --png