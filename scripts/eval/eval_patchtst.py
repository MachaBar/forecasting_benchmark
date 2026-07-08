"""Evaluate a trained PatchTST (point or quantile) on the SAME test windows /
metrics as Chronos2, TS-ICL and the statistical baselines.

The mode (point vs probabilistic) is read from the CHECKPOINT itself — no
need to pass model.probabilistic at eval time.

Run from the repo root:
    python -m scripts.eval.eval_patchtst \
        dataset=cer dataset.context_length=512 dataset.prediction_length=96 \
        eval.run_dir=outputs/patchtst/cer/ctx512_h96/2026-07-03_15-00-00
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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from src.models.patchtst.patch_tst import PatchTST, QuantilePatchTST
from src.utils.metrics import compute_metrics


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


def _save_forecast_plot(
    *, context, y_true, y_pred, uid, cutoff, output_dir,
    quantiles=None, quantile_levels=None, n_context_shown=96,
) -> None:
    """y_pred: (H,). quantiles: (H, Q) or None — adds a q10–q90 band if given."""
    n_ctx = min(n_context_shown, len(context))
    ctx_x = np.arange(-n_ctx, 0)
    fc_x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx_x, context[-n_ctx:], color="steelblue", lw=1.5, label="context")
    ax.plot(fc_x, y_true, color="black", lw=1.5, label="ground truth")
    ax.plot(fc_x, y_pred, color="tomato", lw=1.5, label="forecast (median)")
    if quantiles is not None and quantile_levels is not None:
        q_idx = {q: j for j, q in enumerate(quantile_levels)}
        if 0.1 in q_idx and 0.9 in q_idx:
            ax.fill_between(fc_x, quantiles[:, q_idx[0.1]], quantiles[:, q_idx[0.9]],
                            color="tomato", alpha=0.2, label="q10–q90")
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
    # ---- Locate checkpoint ----
    run_dir = Path(hydra.utils.to_absolute_path(cfg.eval.run_dir))
    ckpt_path = run_dir / "checkpoints" / cfg.eval.get("checkpoint", "best_model.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # ---- Read the mode from the checkpoint itself ----
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        is_probabilistic = ckpt.get("probabilistic", False)
        quantile_levels = ckpt.get("quantile_levels") or [0.1, 0.5, 0.9]
        best_step, best_val = ckpt.get("best_step"), ckpt.get("best_val")
    else:  # old format: raw state_dict, point model
        state_dict = ckpt
        is_probabilistic, quantile_levels = False, None
        best_step = best_val = None

    model_name = "PatchTST-Q" if is_probabilistic else "PatchTST"

    # ---- Build the right architecture and load weights ----
    kwargs = _backbone_kwargs(cfg)
    if is_probabilistic:
        model = QuantilePatchTST(
            quantile_levels=quantile_levels,
            target_window=cfg.dataset.prediction_length,
            **kwargs,
        )
        median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
    else:
        model = PatchTST(target_window=cfg.dataset.prediction_length, **kwargs)
        median_idx = None
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} checkpoint: {ckpt_path}")
    print(f"  best_step={best_step}  best_val={best_val}  quantiles={quantile_levels}")

    # ---- Data + splits (identical to the other model families) ----
    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Clients  : {ts.n_users}  |  Timesteps: {ts.n_dates}")

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
    print(f"Windows  : {len(cutoffs)} test cutoffs "
          f"({ts.datetimes[cutoffs[0]]} → {ts.datetimes[cutoffs[-1]]})")
    print(f"Context  : {cfg.dataset.context_length} steps | Horizon: {cfg.dataset.prediction_length} steps")
    print(f"Total evals: {len(test_indices)} × {len(cutoffs)} = {len(test_indices)*len(cutoffs):,} forecasts")
    print(f"========================\n")

    # ---- Output in the training run dir ----
    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_info = {
        "dataset": cfg.dataset.name,
        "model": model_name,
        "probabilistic": bool(is_probabilistic),
        "quantile_levels": quantile_levels,
        "checkpoint": str(ckpt_path),
        "best_step": best_step,
        "best_val": best_val,
        "n_test_clients": len(test_indices),
        "test_client_ids": [ts.user_names[i] for i in test_indices],
        "n_cutoffs": len(cutoffs),
        "cutoffs": cutoffs,
        "cutoff_dates": [str(ts.datetimes[c]) for c in cutoffs],
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
            x = batch["x"]                          # (n, 1, ctx)
            y_out = model(x).cpu().numpy()          # (n,1,H) point | (n,1,H,Q) quantile — RAW
            x_cpu = x.cpu().numpy()
            y_true_all = batch["y"].cpu().numpy()

            cutoff_rows: list[dict] = []
            for i, uid in enumerate(batch["item_ids"]):
                client_idx = test_indices[i]
                y_true = y_true_all[i, 0]
                context_window = x_cpu[i, 0]
                history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

                if is_probabilistic:
                    quantiles_i = np.sort(y_out[i, 0], axis=-1)   # (H, Q) — sort kills quantile crossing
                    y_pred = quantiles_i[:, median_idx]
                    quantile_preds = {q: quantiles_i[:, j] for j, q in enumerate(quantile_levels)}
                else:
                    quantiles_i = None
                    y_pred = y_out[i, 0]
                    quantile_preds = None

                metrics = compute_metrics(
                    y_true, y_pred,
                    context=context_window,
                    history=history_series,
                    season_length=season_length,
                    include_mape=cfg.model.get("include_mape", False),
                    quantile_preds=quantile_preds,   # → wql + crps in quantile mode
                )
                cutoff_rows.append({
                    "unique_id": uid,
                    "cutoff": cutoff,
                    "model": model_name,
                    "context_length": cfg.dataset.context_length,
                    "prediction_length": cfg.dataset.prediction_length,
                    **metrics.to_dict(),
                })

                if cutoff_loop_idx == 0 and i == 0 and not plot_saved:
                    _save_forecast_plot(
                        context=context_window, y_true=y_true, y_pred=y_pred,
                        uid=uid, cutoff=cutoff, output_dir=output_dir,
                        quantiles=quantiles_i, quantile_levels=quantile_levels,
                    )
                    plot_saved = True

            rows.extend(cutoff_rows)
            write_header = not checkpoint_path.exists()
            pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
            print(f"  → checkpoint saved ({len(rows)} rows total)")

    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)
    results.to_csv(output_dir / f"results_per_client_cutoff_{run_date}.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")

    # Accumulated summary across runs
    summary_row = summary.copy()
    summary_row["context_length"] = cfg.dataset.context_length
    summary_row["prediction_length"] = cfg.dataset.prediction_length
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()
    summary_all_path = run_dir.parent.parent / "summary_all_runs.csv"
    if summary_all_path.exists():
        summary_row = pd.concat([pd.read_csv(summary_all_path), summary_row], ignore_index=True)
    summary_row.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")


if __name__ == "__main__":
    main()