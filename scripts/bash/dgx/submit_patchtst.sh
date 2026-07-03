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


python -m scripts.eval.eval_patchtst \
    dataset=cer_bis \
    dataset.context_length=512 \
    dataset.prediction_length=96 \
    eval.run_dir=outputs/patchtst/cer/ctx512_h96/2026-07-02_15-00-00