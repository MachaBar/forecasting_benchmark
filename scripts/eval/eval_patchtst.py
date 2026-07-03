"""Evaluate a trained PatchTST on the SAME test windows / metrics as Chronos
and the statistical baselines.

Loads best_model.pth from a training run and evaluates on the test cutoffs
(70/15/15 split) restricted to the test clients (from the client split pickle).

Run from the repo root:
    python -m scripts.eval.eval_patchtst \
        eval.run_dir=outputs/patchtst/cer/ctx512_h96/2026-07-02_15-00-00
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[2]))   # repo root (scripts/eval/ -> ../../)
from src.dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from src.models.patchtst.patch_tst import PatchTST
from src.utils.metrics import compute_metrics


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


def _load_checkpoint(model: PatchTST, ckpt_path: Path, device: torch.device) -> dict:
    """Handles both the new dict format {model, optimizer, step, ...} and the
    old raw state_dict format."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        return {"best_step": ckpt.get("best_step"), "best_val": ckpt.get("best_val")}
    model.load_state_dict(ckpt)  # old format: raw state_dict
    return {"best_step": None, "best_val": None}


def _save_forecast_plot(
    *, context, y_true, y_pred, uid, cutoff, output_dir, n_context_shown=96,
) -> None:
    n_ctx = min(n_context_shown, len(context))
    ctx_x = np.arange(-n_ctx, 0)
    fc_x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx_x, context[-n_ctx:], color="steelblue", lw=1.5, label="context")
    ax.plot(fc_x, y_true, color="black", lw=1.5, label="ground truth")
    ax.plot(fc_x, y_pred, color="tomato", lw=1.5, label="PatchTST forecast")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"PatchTST forecast — client {uid}, cutoff {cutoff}")
    ax.set_xlabel("timestep relative to forecast origin")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = output_dir / f"forecast_plot_client{uid}_cutoff{cutoff}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved → {path}")


@hydra.main(config_path="../../configs", config_name="config_patchtst", version_base=None)
def main(cfg: DictConfig) -> None:
    # ---- Locate the training run + checkpoint ----
    run_dir = Path(hydra.utils.to_absolute_path(cfg.eval.run_dir))
    ckpt_path = run_dir / "checkpoints" / cfg.eval.get("checkpoint", "best_model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Clients  : {ts.n_users}  |  Timesteps: {ts.n_dates}")
    print(f"Date range: {ts.datetimes[0]} → {ts.datetimes[-1]}")

    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])

    splits = make_cutoffs(
        ts,
        lags=cfg.dataset.context_length,
        horizon=cfg.dataset.prediction_length,
        step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()

    print(f"\n=== Evaluation scope ===")
    print(f"Clients  : {len(test_indices)} test clients")
    print(f"           IDs: {[ts.user_names[i] for i in test_indices[:5]]} ... {[ts.user_names[i] for i in test_indices[-3:]]}")
    print(f"Windows  : {len(cutoffs)} test cutoffs")
    print(f"           first: {cutoffs[0]}  ({ts.datetimes[cutoffs[0]]})")
    print(f"           last : {cutoffs[-1]} ({ts.datetimes[cutoffs[-1]]})")
    print(f"           stride: {cutoffs[1]-cutoffs[0]} steps = {(cutoffs[1]-cutoffs[0])*0.5:.0f}h")
    print(f"Context  : {cfg.dataset.context_length} steps | Horizon: {cfg.dataset.prediction_length} steps")
    print(f"Total evals: {len(test_indices)} clients × {len(cutoffs)} cutoffs = {len(test_indices)*len(cutoffs):,} forecasts")
    print(f"========================\n")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    ckpt_info = _load_checkpoint(model, ckpt_path, device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  best_step={ckpt_info['best_step']}  best_val={ckpt_info['best_val']}")

    # ---- Output goes into the SAME run dir (eval subfolder) ----
    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_info = {
        "dataset": cfg.dataset.name,
        "model": "PatchTST",
        "checkpoint": str(ckpt_path),
        "best_step": ckpt_info["best_step"],
        "best_val": ckpt_info["best_val"],
        "n_test_clients": len(test_indices),
        "test_client_ids": [ts.user_names[i] for i in test_indices],
        "n_cutoffs": len(cutoffs),
        "cutoffs": cutoffs,
        "cutoff_dates": [str(ts.datetimes[c]) for c in cutoffs],
        "stride_steps": int(cutoffs[1] - cutoffs[0]),
        "context_length": cfg.dataset.context_length,
        "prediction_length": cfg.dataset.prediction_length,
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    checkpoint_path = output_dir / "results_checkpoint.csv"
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    plot_saved = False

    rows: list[dict] = []
    with torch.no_grad():
        for cutoff_loop_idx, cutoff in enumerate(cutoffs):
            print(f"cutoff {cutoff_loop_idx+1}/{len(cutoffs)}  idx={cutoff}  ({ts.datetimes[cutoff]}) ...")

            batch = eval_batch(
                ts, int(cutoff),
                lags=cfg.dataset.context_length,
                horizon=cfg.dataset.prediction_length,
                users=test_indices,
                device=device,
            )
            x = batch["x"]                        # (n_clients, 1, ctx)
            y_pred_all = model(x).cpu().numpy()   # (n_clients, 1, horizon)  — RAW (RevIN denorm)
            x_cpu = x.cpu().numpy()
            y_true_all = batch["y"].cpu().numpy()

            cutoff_rows: list[dict] = []
            for i, uid in enumerate(batch["item_ids"]):
                client_idx = test_indices[i]
                y_true = y_true_all[i, 0]
                y_pred = y_pred_all[i, 0]
                context_window = x_cpu[i, 0]
                history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

                # compute_metrics returns raw mae/rmse AND normalized variants
                # (mae_normalized/rmse_normalized) because context is passed.
                metrics = compute_metrics(
                    y_true, y_pred,
                    context=context_window,
                    history=history_series,
                    season_length=season_length,
                    include_mape=cfg.model.get("include_mape", False),
                )
                cutoff_rows.append({
                    "unique_id": uid,
                    "cutoff": cutoff,
                    "model": "PatchTST",
                    "context_length": cfg.dataset.context_length,
                    "prediction_length": cfg.dataset.prediction_length,
                    **metrics.to_dict(),
                })

                if cutoff_loop_idx == 0 and i == 0 and not plot_saved:
                    _save_forecast_plot(
                        context=context_window, y_true=y_true, y_pred=y_pred,
                        uid=uid, cutoff=cutoff, output_dir=output_dir,
                    )
                    plot_saved = True

            rows.extend(cutoff_rows)
            write_header = not checkpoint_path.exists()
            pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
            print(f"  → checkpoint saved ({len(rows)} rows total)")

    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")

    # Accumulated summary across all runs (context/horizon sweep)
    summary_row = summary.copy()
    summary_row["context_length"] = cfg.dataset.context_length
    summary_row["prediction_length"] = cfg.dataset.prediction_length
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()
    summary_all_path = output_dir.parent.parent / "summary_all_runs.csv"  # au niveau du dataset
    if summary_all_path.exists():
        summary_row = pd.concat([pd.read_csv(summary_all_path), summary_row], ignore_index=True)
    summary_row.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")


if __name__ == "__main__":
    main()