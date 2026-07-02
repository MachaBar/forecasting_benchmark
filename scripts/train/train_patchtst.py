"""Train PatchTST with the SAME temporal split (70/15/15) training only sees train clients
and train positions; validation uses val cutoffs on val clients.

Run from the repo root:
    python -m scripts.train_patchtst
    python -m scripts.train_patchtst dataset=cer_bis train.max_steps=100000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.dataset.dataset import (
    TrainWindowDataset,
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from src.models.patchtst.patch_tst import PatchTST
from torch.utils.tensorboard import SummaryWriter


def build_model(cfg: DictConfig) -> PatchTST:
    m = cfg.model
    return PatchTST(
        c_in=m.c_in,
        context_window=cfg.dataset.context_length,
        target_window=cfg.dataset.prediction_length,
        patch_len=m.patch_len,
        stride=m.stride,
        n_layers=m.n_layers,
        d_model=m.d_model,
        n_heads=m.n_heads,
        d_ff=m.d_ff,
        dropout=m.dropout,
        fc_dropout=m.fc_dropout,
        head_dropout=m.head_dropout,
        padding_patch=m.padding_patch,
        individual=m.individual,
        res_attention=m.res_attention,
        pre_norm=m.pre_norm,
        store_attn=m.store_attn,
        pe=m.pe,
        learn_pe=m.learn_pe,
        head_type=m.head_type,
        verbose=m.verbose,
    )


def _val_loss(
    model: PatchTST,
    ts,
    cutoffs: np.ndarray,
    cfg: DictConfig,
    device: torch.device,
    val_indices: list[int],
) -> float:
    """MSE on the val cutoffs, restricted to the VAL clients."""
    model.eval()
    criterion = nn.MSELoss()
    total = 0.0
    with torch.no_grad():
        for cutoff in cutoffs:
            batch = eval_batch(
                ts, int(cutoff),
                lags=cfg.dataset.context_length,
                horizon=cfg.dataset.prediction_length,
                users=val_indices,
                device=device,
            )
            y_pred = model(batch["x"])
            total += criterion(y_pred, batch["y"]).item()
    model.train()
    return total / max(len(cutoffs), 1)

def _val_metrics(model, ts, cutoffs, cfg, device, val_indices, season_length):
    """Full metric bundle on val cutoffs / val clients (mean over clients+cutoffs)."""
    from src.utils.metrics import compute_metrics
    model.eval()
    all_rows = []
    with torch.no_grad():
        for cutoff in cutoffs:
            batch = eval_batch(
                ts, int(cutoff),
                lags=cfg.dataset.context_length,
                horizon=cfg.dataset.prediction_length,
                users=val_indices,
                device=device,
            )
            y_pred = model(batch["x"]).cpu().numpy()
            x_cpu = batch["x"].cpu().numpy()
            y_cpu = batch["y"].cpu().numpy()
            for i in range(y_pred.shape[0]):
                client_idx = val_indices[i]
                history_series = ts.values[: int(cutoff) + cfg.dataset.context_length, client_idx]
                m = compute_metrics(
                    y_cpu[i, 0], y_pred[i, 0],
                    context=x_cpu[i, 0],
                    history=history_series,
                    season_length=season_length,
                )
                all_rows.append(m.to_dict())
    model.train()
    df = pd.DataFrame(all_rows)
    return df.mean(numeric_only=True).to_dict()


@hydra.main(config_path="../configs", config_name="config_patchtst", version_base=None)
def main(cfg: DictConfig) -> None:
    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Clients  : {ts.n_users}  |  Timesteps: {ts.n_dates}")
    print(f"Date range: {ts.datetimes[0]} → {ts.datetimes[-1]}")

    # ---- Client split: train on train clients only (no leakage) ----
    split = load_client_split_pickle(split_path)
    train_indices = client_ids_to_indices(ts, split["train"])
    val_indices = client_ids_to_indices(ts, split["val"])
    print(f"Client split: {len(train_indices)} train / {len(val_indices)} val / {len(split['test'])} test")

    # ---- Temporal split: SAME 70/15/15 cutoffs as Chronos / stats ----
    cutoffs = make_cutoffs(
        ts,
        lags=cfg.dataset.context_length,
        horizon=cfg.dataset.prediction_length,
        step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    print(f"Train positions: {len(cutoffs['train_positions'])}")
    print(f"Val cutoffs    : {len(cutoffs['val_cutoffs'])}")
    print(f"Test cutoffs   : {len(cutoffs['test_cutoffs'])} (NOT touched during training)")
    print(f"Context  : {cfg.dataset.context_length} steps | Horizon: {cfg.dataset.prediction_length} steps")

    train_ds = TrainWindowDataset(
        ts,
        lags=cfg.dataset.context_length,
        horizon=cfg.dataset.prediction_length,
        train_positions=cutoffs["train_positions"],
        n_samples=cfg.n_samples,
        client_pool=train_indices,   # <- excludes val AND test clients
        seed=cfg.seed,
    )
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, num_workers=4, pin_memory=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    criterion = nn.MSELoss()

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))

    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs → {tb_dir}")

    # Historiques sauvegardés aussi en CSV (indépendant de TensorBoard)
    train_history: list[dict] = []
    val_history: list[dict] = []

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save training info (same spirit as eval_info.json)
    train_info = {
        "dataset": cfg.dataset.name,
        "context_length": cfg.dataset.context_length,
        "prediction_length": cfg.dataset.prediction_length,
        "stride": cfg.dataset.stride,
        "ratios": cfg.dataset.get("ratios", "0.7,0.15,0.15"),
        "n_train_clients": len(train_indices),
        "n_val_clients": len(val_indices),
        "n_train_positions": int(len(cutoffs["train_positions"])),
        "n_samples": cfg.n_samples,
        "max_steps": cfg.train.max_steps,
        "batch_size": cfg.train.batch_size,
        "learning_rate": cfg.train.learning_rate,
    }
    with open(output_dir / "train_info.json", "w") as f:
        json.dump(train_info, f, indent=2)

        step = 0
    best_val = float("inf")
    patience = cfg.train.get("early_stopping_patience", 10)  # nb de validations sans amélioration
    min_delta = cfg.train.get("early_stopping_min_delta", 1e-5)
    bad_vals = 0
    stop_training = False

    model.train()
    while step < cfg.train.max_steps and not stop_training:
        for batch in loader:
            if step >= cfg.train.max_steps:
                break

            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            if step % cfg.train.log_every == 0:
                print(f"step={step:>7d}  train_loss={loss.item():.6f}")
                writer.add_scalar("train/loss_mse", loss.item(), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                train_history.append({"step": step, "loss_mse": loss.item()})

            if step % cfg.train.val_every == 0 and step > 0:
                val = _val_loss(model, ts, cutoffs["val_cutoffs"], cfg, device, val_indices)
                writer.add_scalar("val/loss_mse", val, step)
                val_history.append({"step": step, "loss_mse": val})
                if val < best_val - min_delta:
                    best_val = val
                    bad_vals = 0
                    torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                    print(f"step={step:>7d}  val_loss={val:.6f}  ✓ new best")
                else:
                    bad_vals += 1
                    print(f"step={step:>7d}  val_loss={val:.6f}  (no improvement {bad_vals}/{patience})")
                    if bad_vals >= patience:
                        print(f"Early stopping at step {step} — no val improvement in {patience} validations.")
                        stop_training = True
                        break

            if step % cfg.train.save_every == 0 and step > 0:
                torch.save(model.state_dict(), ckpt_dir / f"step_{step:07d}.pth")

            if step % cfg.train.get("metrics_every", 10000) == 0 and step > 0:
                vm = _val_metrics(model, ts, cutoffs["val_cutoffs"], cfg, device,
                                  val_indices, cfg.dataset.get("season_length", 48))
                for name, value in vm.items():
                    if value == value:  # skip NaN
                        writer.add_scalar(f"val/{name}", value, step)
                print(f"step={step:>7d}  val_mase={vm.get('mase', float('nan')):.4f}")

            step += 1


if __name__ == "__main__":
    main()