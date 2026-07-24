#!/bin/bash
#SBATCH --wckey=p11mh:python
#SBATCH --partition=h100
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

# fine-tune LoRA
# srun python -u -m scripts.finetune.finetune_chronos \
#     dataset.context_length=1440 \
#     dataset=cer_bis \ 
#     dataset.prediction_length=96 \
#     model.finetune_mode=lora

# HYDRA_FULL_ERROR=1 srun python -u -m scripts.finetune.finetune_chronos \
#     dataset=cer_bis \
#     dataset.context_length=1440 \
#     dataset.prediction_length=96 \
#     model.finetune_mode=lora


#  full fine-tuning
# python -m scripts.finetune.finetune_chronos \
#     dataset=cer_bis \
#     model.finetune_mode=full \
#     model.learning_rate=1e-6 \
#     model.num_steps=200

# # 2. éval (poids fine-tunés)
srun python -u -m src.baselines.run_foundation \
    model=chronos dataset=cer_bis model.probabilistic=true \
    dataset.context_length=1440 dataset.prediction_length=96 \
    model.weights_path=outputs/chronos2_finetuned/cer_bis/full/2026-07-23_14-07-03/finetuned_weights

