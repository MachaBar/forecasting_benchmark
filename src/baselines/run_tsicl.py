"""Zero-shot inference with TS-ICL on the SAME test windows / metrics as
Chronos2, PatchTST and the statistical baselines.

Run from the repo root:
    python -m src.baselines.run_tsicl
    python -m src.baselines.run_tsicl dataset=cer_bis model.probabilistic=true
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
from src.utils.metrics import compute_metrics


def _save_forecast_plot(
    *, context, y_true, quantiles, quantile_levels, uid, cutoff, output_dir,
    n_context_shown=96,
) -> None:
    """quantiles: (horizon, Q). Median + q10-q90 band, same style as chronos plot."""
    q_idx = {q: j for j, q in enumerate(quantile_levels)}
    median = quantiles[:, q_idx[0.5]]
    q10 = quantiles[:, q_idx.get(0.1, 0)]
    q90 = quantiles[:, q_idx.get(0.9, len(quantile_levels) - 1)]

    n_ctx = min(n_context_shown, len(context))
    ctx_x = np.arange(-n_ctx, 0)
    fc_x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx_x, context[-n_ctx:], color="steelblue", lw=1.5, label="context")
    ax.plot(fc_x, y_true, color="black", lw=1.5, label="ground truth")
    ax.plot(fc_x, median, color="tomato", lw=1.5, label="median (q50)")
    ax.fill_between(fc_x, q10, q90, color="tomato", alpha=0.2, label="q10–q90")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"TS-ICL forecast — client {uid}, cutoff {cutoff}")
    ax.set_xlabel("timestep relative to forecast origin")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = output_dir / f"forecast_plot_client{uid}_cutoff{cutoff}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved → {path}")


@hydra.main(config_path="../../configs", config_name="config_tsicl", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        from tsicl import TSICL
    except ImportError:
        raise ImportError("tsicl is not installed in this environment.")

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
    print(f"Context  : {cfg.dataset.context_length} steps = {cfg.dataset.context_length*0.5:.0f}h")
    print(f"Horizon  : {cfg.dataset.prediction_length} steps = {cfg.dataset.prediction_length*0.5:.0f}h")
    print(f"Total evals: {len(test_indices)} clients × {len(cutoffs)} cutoffs = {len(test_indices)*len(cutoffs):,} inference calls")
    print(f"========================\n")

    prediction_length = cfg.dataset.prediction_length
    context_length = cfg.dataset.context_length
    is_probabilistic = cfg.model.get("probabilistic", False)
    # Always request q10/q50/q90 for the plot; add configured levels when probabilistic
    plot_quantiles = [0.1, 0.5, 0.9]
    quantile_levels = sorted(set(plot_quantiles + (list(cfg.model.get("quantile_levels", [])) if is_probabilistic else [])))
    batch_size = cfg.model.get("batch_size", 32)
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    device = cfg.model.get("device", "cuda")

    model = TSICL(
        model_path=hydra.utils.to_absolute_path(cfg.model.weights_path),
        allow_auto_download=False,
    )
    print("Model loaded.")

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_info = {
        "dataset": cfg.dataset.name,
        "model": "TSICL",
        "n_clients_total": ts.n_users,
        "n_timesteps": ts.n_dates,
        "date_start": str(ts.datetimes[0]),
        "date_end": str(ts.datetimes[-1]),
        "n_test_clients": len(test_indices),
        "test_client_ids": [ts.user_names[i] for i in test_indices],
        "n_cutoffs": len(cutoffs),
        "cutoffs": cutoffs,
        "cutoff_dates": [str(ts.datetimes[c]) for c in cutoffs],
        "stride_steps": int(cutoffs[1] - cutoffs[0]),
        "context_length": context_length,
        "prediction_length": prediction_length,
        "quantile_levels": quantile_levels,
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)
    print(f"Eval info saved → {output_dir / 'eval_info.json'}")

    checkpoint_path = output_dir / "results_checkpoint.csv"
    plot_saved = False

    rows: list[dict] = []
    for cutoff_loop_idx, cutoff in enumerate(cutoffs):
        print(f"cutoff {cutoff_loop_idx+1}/{len(cutoffs)}  idx={cutoff}  ({ts.datetimes[cutoff]}) ...")

        batch = eval_batch(
            ts, int(cutoff),
            lags=context_length,
            horizon=prediction_length,
            users=test_indices,
        )
        x = batch["x"]           # (n_clients, 1, ctx)
        y_true_all = batch["y"]  # (n_clients, 1, horizon)
        n_clients = x.shape[0]

        # TS-ICL accepts [N, L] batched contexts
        all_point = []      # list of (b, horizon)
        all_quantiles = []  # list of (b, horizon, Q)
        for start in range(0, n_clients, batch_size):
            end = min(start + batch_size, n_clients)
            inputs = x[start:end, 0, :].numpy()   # [b, L]

            batch_p, batch_q = model.forecast(
                inputs=inputs,
                prediction_length=prediction_length,
                context_length=context_length,
                batch_size=inputs.shape[0],
                device=device,
                quantile_levels=quantile_levels,
                point_estimator=cfg.model.get("point_estimator", "median"),
                denormalize=True,
            )
            # batch_p: [N, C, H, 1] → (b, horizon);  batch_q: [N, C, H, Q] → (b, horizon, Q)
            p = batch_p.detach().cpu().numpy() if isinstance(batch_p, torch.Tensor) else np.asarray(batch_p)
            q = batch_q.detach().cpu().numpy() if isinstance(batch_q, torch.Tensor) else np.asarray(batch_q)
            # Squeeze channel dim (C=1) and trailing 1 on the point estimate
            p = p.reshape(p.shape[0], prediction_length)                 # (b, H)
            q = q.reshape(q.shape[0], prediction_length, len(quantile_levels))  # (b, H, Q)
            all_point.append(p)
            all_quantiles.append(q)

        point_all = np.concatenate(all_point, axis=0)       # (n_clients, H)
        quantiles_all = np.concatenate(all_quantiles, axis=0)  # (n_clients, H, Q)

        cutoff_rows: list[dict] = []
        for i, uid in enumerate(batch["item_ids"]):
            client_idx = test_indices[i]
            y_true = y_true_all[i, 0].numpy()
            context_window = x[i, 0].numpy()
            history_series = ts.values[: cutoff + context_length, client_idx]

            y_pred = point_all[i]   # (H,) — point estimate (median by default)

            quantile_preds = (
                {q: quantiles_all[i, :, j] for j, q in enumerate(quantile_levels)}
                if is_probabilistic else None
            )

            metrics = compute_metrics(
                y_true, y_pred,
                context=context_window,
                history=history_series,
                season_length=season_length,
                include_mape=cfg.model.get("include_mape", False),
                quantile_preds=quantile_preds,
            )
            cutoff_rows.append({
                "unique_id": uid,
                "cutoff": cutoff,
                "model": "TSICL",
                "context_length": context_length,
                "prediction_length": prediction_length,
                **metrics.to_dict(),
            })

            if cutoff_loop_idx == 0 and i == 0 and not plot_saved:
                _save_forecast_plot(
                    context=context_window, y_true=y_true,
                    quantiles=quantiles_all[i],           # (H, Q)
                    quantile_levels=quantile_levels,
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
    results.to_csv(output_dir / f"results_per_client_cutoff_{run_date}.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")

    # Accumulated summary across runs (ctx/horizon sweep)
    summary_row = summary.copy()
    summary_row["context_length"] = context_length
    summary_row["prediction_length"] = prediction_length
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()
    summary_all_path = output_dir.parent / "summary_all_runs.csv"
    if summary_all_path.exists():
        summary_row = pd.concat([pd.read_csv(summary_all_path), summary_row], ignore_index=True)
    summary_row.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")


if __name__ == "__main__":
    main()