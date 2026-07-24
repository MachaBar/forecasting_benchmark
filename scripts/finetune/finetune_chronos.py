"""Fine-tune Chronos-2 on the TRAIN clients / TRAIN period, then save the
fine-tuned pipeline so it can be evaluated with the existing run_foundation
pipeline (same test cutoffs / clients / metrics as zero-shot Chronos-2).

Run from the repo root:
    python -m scripts.finetune.finetune_chronos dataset=cer_bis \
        model.finetune_mode=lora dataset.prediction_length=96
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    client_ids_to_indices, load_client_split_pickle, load_dataset, make_cutoffs,
)


def _build_long_df(ts, client_indices, t_start, t_end, id_col, ts_col, target):
    """Long-format df (id, timestamp, target) over [t_start, t_end) for the
    given clients — the input format expected by preprocess.from_data_frame."""
    dates = pd.DatetimeIndex(ts.datetimes[t_start:t_end]).tz_localize(None)
    frames = []
    for u in client_indices:
        frames.append(pd.DataFrame({
            id_col: ts.user_names[u],
            ts_col: dates,
            target: ts.values[t_start:t_end, u],
        }))
    return pd.concat(frames, ignore_index=True)


@hydra.main(config_path="../../configs", config_name="config_finetune_chronos", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    from chronos.chronos2 import Chronos2Pipeline
    from chronos.chronos2 import preprocess

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    split = load_client_split_pickle(split_path)
    train_indices = client_ids_to_indices(ts, split["train"])
    val_indices = client_ids_to_indices(ts, split["val"])

    H = cfg.dataset.prediction_length

    # temporal boundaries (train / val), same 70/15/15 split
    r_train, r_val, _ = (float(x) for x in cfg.dataset.get("ratios", "0.7,0.15,0.15").split(","))
    train_end = int(round(r_train * ts.n_dates))
    val_end = int(round((r_train + r_val) * ts.n_dates))

    print(f"Fine-tuning Chronos-2 on {len(train_indices)} train clients, "
          f"period [0, {train_end}) ; validation on {len(val_indices)} val clients")

    id_col, ts_col, target = "unique_id", "timestamp", "target"

    # ---- Build fine-tuning inputs (train clients / train period) ----
    train_df = _build_long_df(ts, train_indices, 0, train_end, id_col, ts_col, target)
    train_inputs = preprocess.from_data_frame(
        train_df,
        target_columns=[target],
        prediction_length=H,
        id_column=id_col,
        timestamp_column=ts_col,
        # no covariates for univariate load curves
    )

    # ---- Optional validation inputs (val clients / val period) ----
    validation_inputs = None
    if cfg.model.get("use_validation", True):
        val_df = _build_long_df(ts, val_indices, 0, val_end, id_col, ts_col, target)
        validation_inputs = preprocess.from_data_frame(
            val_df, target_columns=[target], prediction_length=H,
            id_column=id_col, timestamp_column=ts_col,
        )

    # ---- Load the pretrained pipeline ----
    pipeline = Chronos2Pipeline.from_pretrained(
        hydra.utils.to_absolute_path(cfg.model.weights_path),
        device_map=cfg.model.get("device_map", "cuda"),
    )

    # ---- Fine-tune ----
    finetune_mode = cfg.model.get("finetune_mode", "lora")     # "full" | "lora"
    default_lr = 1e-4 if finetune_mode == "lora" else 1e-6
    fit_kwargs = dict(
        inputs=train_inputs,
        prediction_length=H,
        finetune_mode=finetune_mode,
        num_steps=cfg.model.get("num_steps", 1000),
        learning_rate=cfg.model.get("learning_rate", default_lr),
        batch_size=cfg.model.get("batch_size", 32),
        logging_steps=cfg.model.get("logging_steps", 100),
    )
    if validation_inputs is not None:
        fit_kwargs["validation_inputs"] = validation_inputs

    print(f"Fine-tuning ({finetune_mode}) — {fit_kwargs['num_steps']} steps, "
          f"lr={fit_kwargs['learning_rate']}, batch={fit_kwargs['batch_size']}")
    t0 = time.perf_counter()
    finetuned = pipeline.fit(**fit_kwargs)
    fit_time = time.perf_counter() - t0
    print(f"Fine-tuning done in {fit_time:.1f}s")

    # ---- Save the fine-tuned pipeline ----
    save_dir = output_dir / "finetuned_weights"
    finetuned.save_pretrained(str(save_dir))     # ← verify method name (see notes)
    print(f"Fine-tuned weights → {save_dir}")

    with open(output_dir / "finetune_info.json", "w") as f:
        json.dump({
            "dataset": cfg.dataset.name,
            "finetune_mode": finetune_mode,
            "n_train_clients": len(train_indices),
            "n_val_clients": len(val_indices),
            "prediction_length": H,
            "num_steps": fit_kwargs["num_steps"],
            "learning_rate": fit_kwargs["learning_rate"],
            "batch_size": fit_kwargs["batch_size"],
            "fit_time_s": round(fit_time, 2),
            "weights_path": str(save_dir),
        }, f, indent=2)


if __name__ == "__main__":
    main()