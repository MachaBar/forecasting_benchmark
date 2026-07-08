"""Train PatchTST with the SAME temporal split (70/15/15): training only sees
train clients and train positions; validation uses val cutoffs on val clients.

Two modes (cfg.model.probabilistic):
  - false: point forecast, MSE loss        → model "PatchTST"
  - true : multi-quantile head, pinball    → model "PatchTST-Q" (enables WQL/CRPS at eval)

Run from the repo root:
    python -m scripts.train.train_patchtst
    python -m scripts.train.train_patchtst dataset=cer_bis model.probabilistic=true
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    TrainWindowDataset,
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from src.models.patchtst.patch_tst import PatchTST, QuantilePatchTST


class PinballLoss(nn.Module):
    """Mean pinball (quantile) loss over all levels.
    y_pred: (B, 1, H, Q)   y_true: (B, 1, H)"""

    def __init__(self, quantile_levels):
        super().__init__()
        self.register_buffer("q", torch.tensor(quantile_levels, dtype=torch.float32))

    def forward(self, y_pred, y_true):
        diff = y_true.unsqueeze(-1) - y_pred          # (B, 1, H, Q)
        loss = torch.maximum(self.q * diff, (self.q - 1.0) * diff)
        return loss.mean()


def _backbone_kwargs(cfg: DictConfig) -> dict:
    m = cfg.model
    return dict(
        c_in=m.c_in,
        context_window=cfg.dataset.context_length,
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


def build_model(cfg: DictConfig, quantile_levels: list[float] | None):
    kwargs = _backbone_kwargs(cfg)
    if quantile_levels is not None:
        return QuantilePatchTST(
            quantile_levels=quantile_levels,
            target_window=cfg.dataset.prediction_length,
            **kwargs,
        )
    return PatchTST(target_window=cfg.dataset.prediction_length, **kwargs)


def _val_loss(model, ts, cutoffs, cfg, device, val_indices, criterion) -> float:
    """Same loss as training (MSE or pinball) on val cutoffs / val clients."""
    model.eval()
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
            total += criterion(model(batch["x"]), batch["y"]).item()
    model.train()
    return total / max(len(cutoffs), 1)


def _point_from_output(y_pred: np.ndarray, median_idx: int | None) -> np.ndarray:
    """(n, 1, H) → (n, H) point mode; (n, 1, H, Q) → (n, H) median in quantile mode."""
    if y_pred.ndim == 4:
        return y_pred[:, 0, :, median_idx]
    return y_pred[:, 0, :]


def _val_metrics(model, ts, cutoffs, cfg, device, val_indices, season_length, median_idx):
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
            y_out = model(batch["x"]).cpu().numpy()
            y_point = _point_from_output(y_out, median_idx)
            x_cpu = batch["x"].cpu().numpy()
            y_cpu = batch["y"].cpu().numpy()
            for i in range(y_point.shape[0]):
                client_idx = val_indices[i]
                history_series = ts.values[: int(cutoff) + cfg.dataset.context_length, client_idx]
                m = compute_metrics(
                    y_cpu[i, 0], y_point[i],
                    context=x_cpu[i, 0],
                    history=history_series,
                    season_length=season_length,
                )
                all_rows.append(m.to_dict())
    model.train()
    return pd.DataFrame(all_rows).mean(numeric_only=True).to_dict()


def collate_drop_none(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in batch]
        if vals[0] is None:
            out[k] = None
        elif isinstance(vals[0], str):
            out[k] = vals
        else:
            out[k] = default_collate(vals)
    return out


@hydra.main(config_path="../../configs", config_name="config_patchtst", version_base=None)
def main(cfg: DictConfig) -> None:
    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Clients  : {ts.n_users}  |  Timesteps: {ts.n_dates}")
    print(f"Date range: {ts.datetimes[0]} → {ts.datetimes[-1]}")

    # ---- Probabilistic mode ----
    is_probabilistic = cfg.model.get("probabilistic", False)
    quantile_levels = sorted(cfg.model.get("quantile_levels", [0.1, 0.5, 0.9])) if is_probabilistic else None
    if is_probabilistic:
        median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
        print(f"Mode     : PROBABILISTIC — pinball loss on quantiles {quantile_levels}")
    else:
        median_idx = None
        print(f"Mode     : POINT — MSE loss")

    # ---- Client split ----
    split = load_client_split_pickle(split_path)
    train_indices = client_ids_to_indices(ts, split["train"])
    val_indices = client_ids_to_indices(ts, split["val"])
    print(f"Client split: {len(train_indices)} train / {len(val_indices)} val / {len(split['test'])} test")

    # ---- Temporal split ----
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
        client_pool=train_indices,
        seed=cfg.seed,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.get("num_workers", 0),
        pin_memory=True,
        collate_fn=collate_drop_none,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, quantile_levels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.max_steps, eta_min=cfg.train.get("eta_min", 1e-7),
    )
    criterion = (PinballLoss(quantile_levels) if is_probabilistic else nn.MSELoss()).to(device)

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=30)
    print(f"TensorBoard logs → {tb_dir}")

    train_history: list[dict] = []
    val_history: list[dict] = []

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_info = {
        "dataset": cfg.dataset.name,
        "model": "PatchTST-Q" if is_probabilistic else "PatchTST",
        "probabilistic": bool(is_probabilistic),
        "quantile_levels": quantile_levels,
        "loss": "pinball" if is_probabilistic else "mse",
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

    # ---- Training loop (step-based) ----
    step = 0
    best_val = float("inf")
    best_step = -1
    patience = cfg.train.get("early_stopping_patience", 10)
    min_delta = cfg.train.get("early_stopping_min_delta", 1e-5)
    bad_vals = 0
    stop_training = False
    loss_tag = "loss_pinball" if is_probabilistic else "loss_mse"

    def _save(path, extra=None):
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best_val,
            "best_step": best_step,
            "probabilistic": bool(is_probabilistic),
            "quantile_levels": quantile_levels,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    model.train()
    print("Start training")
    while step < cfg.train.max_steps and not stop_training:
        for batch in loader:
            if step >= cfg.train.max_steps:
                break

            x = batch["x"].to(device)   # (B, 1, ctx)
            y = batch["y"].to(device)   # (B, 1, horizon)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % cfg.train.log_every == 0:
                print(f"step={step:>7d}  train_{loss_tag}={loss.item():.6f}")
                writer.add_scalar(f"train/{loss_tag}", loss.item(), step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                train_history.append({"step": step, loss_tag: loss.item()})

            if step % cfg.train.val_every == 0 and step > 0:
                val = _val_loss(model, ts, cutoffs["val_cutoffs"], cfg, device, val_indices, criterion)
                writer.add_scalar(f"val/{loss_tag}", val, step)
                val_history.append({"step": step, loss_tag: val})
                if val < best_val - min_delta:
                    best_val = val
                    best_step = step
                    bad_vals = 0
                    _save(ckpt_dir / "best_model.pth")
                    with open(ckpt_dir / "best_info.json", "w") as fp:
                        json.dump({"best_step": best_step, "best_val": best_val}, fp, indent=2)
                    print(f"step={step:>7d}  val_{loss_tag}={val:.6f}  new best")
                else:
                    bad_vals += 1
                    print(f"step={step:>7d}  val_{loss_tag}={val:.6f}  (no improvement {bad_vals}/{patience})")
                    if bad_vals >= patience:
                        print(f"Early stopping at step {step} — no val improvement in {patience} validations.")
                        stop_training = True
                        break

            if step % cfg.train.save_every == 0 and step > 0:
                _save(ckpt_dir / f"step_{step:07d}.pth")

            if step % cfg.train.get("metrics_every", 10000) == 0 and step > 0:
                vm = _val_metrics(model, ts, cutoffs["val_cutoffs"], cfg, device,
                                  val_indices, cfg.dataset.get("season_length", 48), median_idx)
                for name, value in vm.items():
                    if value == value:  # skip NaN
                        writer.add_scalar(f"val/{name}", value, step)
                print(f"step={step:>7d}  val_mase={vm.get('mase', float('nan')):.4f}")

            step += 1

    # ---- Finalize ----
    _save(ckpt_dir / "last_model.pth")
    writer.close()
    pd.DataFrame(train_history).to_csv(output_dir / "history_train.csv", index=False)
    pd.DataFrame(val_history).to_csv(output_dir / "history_val.csv", index=False)
    print(f"Done. Best val {loss_tag}: {best_val:.6f} @ step {best_step}")
    print(f"Best checkpoint: {ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()