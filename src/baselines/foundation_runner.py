"""Common evaluation engine for zero-shot foundation models.

One generic loop (splits, mini-batching, metrics, timing, saving) shared by
every foundation model — each model only implements a ForecastAdapter
(load + predict). Same test cutoffs / clients / metrics / output format as
the statistical baselines and PatchTST.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset.dataset import (
    client_ids_to_indices, eval_batch, load_client_split_pickle,
    load_dataset, make_cutoffs,
)
from src.utils.metrics import compute_metrics
from .adapters.base import ForecastAdapter


def _cuda_sync():
    """Wait for all GPU work to finish before reading the clock (CUDA is async)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _save_forecast_plot(*, context, y_true, quantiles, quantile_levels, uid,
                        cutoff, output_dir, context_length, prediction_length,
                        model_name="foundation", n_context_shown=None):
    """quantiles: (H, Q). """
    q_idx = {q: j for j, q in enumerate(quantile_levels)}
    median = quantiles[:, q_idx[0.5]]
    q10 = quantiles[:, q_idx.get(0.1, 0)]
    q90 = quantiles[:, q_idx.get(0.9, len(quantile_levels) - 1)]

    n_ctx = len(context) if n_context_shown is None else min(n_context_shown, len(context))
    ctx_x = np.arange(0, n_ctx)
    fc_x = np.arange(n_ctx, n_ctx + len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx_x, context[-n_ctx:], color="steelblue", lw=1.5, label="context")
    ax.plot(fc_x, y_true, color="black", lw=1.5, label="ground truth")
    ax.plot(fc_x, median, color="tomato", lw=1.5, label="median (q50)")
    ax.fill_between(fc_x, q10, q90, color="tomato", alpha=0.2, label="q10–q90")
    ax.axvline(n_ctx, color="gray", lw=0.8, ls="--")
    ax.set_title(f"{model_name} — client {uid}, cutoff {cutoff} "
                 f"(ctx={context_length}, h={prediction_length})")
    ax.set_xlabel("timestep (0 = start of context)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = output_dir / f"forecast_plot_ctx{context_length}_h{prediction_length}_client{uid}_cutoff{cutoff}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved → {path}")


def run_foundation_eval(cfg, adapter: ForecastAdapter, output_dir: Path,
                        save_plot_fn=None) -> pd.DataFrame:
    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])

    # splits = make_cutoffs(
    #     ts, lags=cfg.dataset.context_length, horizon=cfg.dataset.prediction_length,
    #     step_size=cfg.dataset.stride, ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    # )
    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length

    splits = make_cutoffs(
    ts, lags=ctx_len, horizon=H,
    step_size=cfg.dataset.stride,
    ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    cutoff_mode=cfg.dataset.get("cutoff_mode", "fixed"),
    gap_min=cfg.dataset.get("cutoff_gap_min", 24),
    gap_max=cfg.dataset.get("cutoff_gap_max", 96),
    seed=cfg.dataset.get("cutoff_seed", 42),
)
    cutoffs = splits["test_cutoffs"].tolist()

    is_prob = cfg.model.get("probabilistic", False)
    quantile_levels = sorted(set([0.1, 0.5, 0.9] +
        (list(cfg.model.get("quantile_levels", [])) if is_prob else [])))
    batch_size = cfg.model.get("batch_size", 32)
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    # ctx_len = cfg.dataset.context_length
    # H = cfg.dataset.prediction_length

    # ---- Load model (timed) ----
    t0 = time.perf_counter()
    adapter.load()
    _cuda_sync()
    load_time = time.perf_counter() - t0

    # ---- Scope printout ----
    print(f"Dataset  : {cfg.dataset.name}  |  {ts.n_users} clients × {ts.n_dates} steps")
    print(f"Model    : {adapter.name}  (loaded in {load_time:.1f}s)")
    print(f"Clients  : {len(test_indices)} test clients")
    print(f"Windows  : {len(cutoffs)} cutoffs ({ts.datetimes[cutoffs[0]]} → {ts.datetimes[cutoffs[-1]]})")
    print(f"Context  : {ctx_len} | Horizon: {H} | Quantiles: {quantile_levels}")
    print(f"Total evals: {len(test_indices)*len(cutoffs):,} inference calls\n")

    # ---- Warmup: first mini-batch is slow (CUDA kernel compilation) ----
    warm = eval_batch(ts, int(cutoffs[0]), lags=ctx_len, horizon=H,
                      users=test_indices[:min(batch_size, len(test_indices))])
    _ = adapter.predict(warm["x"][:, 0, :].numpy(), H, quantile_levels)
    _cuda_sync()

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows, plot_saved = [], False
    cutoff_times, total_infer_time = [], 0.0

    run_start = time.perf_counter()
    for ci, cutoff in enumerate(cutoffs):
        t_cutoff = time.perf_counter()
        batch = eval_batch(ts, int(cutoff), lags=ctx_len, horizon=H, users=test_indices)
        x = batch["x"]
        n = x.shape[0]

        # ---- Mini-batched inference (pure model time isolated + CUDA-synced) ----
        qparts = []
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            contexts = x[s:e, 0, :].numpy()               # (b, L)
            _cuda_sync()
            t_inf = time.perf_counter()
            q = adapter.predict(contexts, H, quantile_levels)   # (b, H, Q)
            _cuda_sync()
            total_infer_time += time.perf_counter() - t_inf
            qparts.append(q)
        all_q = np.concatenate(qparts, axis=0)            # (n, H, Q)

        median_idx = quantile_levels.index(0.5)
        cutoff_rows = []
        for i, uid in enumerate(batch["item_ids"]):
            y_true = batch["y"][i, 0].numpy()
            ctx = x[i, 0].numpy()
            hist = ts.values[: cutoff + ctx_len, test_indices[i]]
            y_pred = all_q[i, :, median_idx]
            qpreds = {q: all_q[i, :, j] for j, q in enumerate(quantile_levels)} if is_prob else None
            m = compute_metrics(y_true, y_pred, context=ctx, history=hist,
                                season_length=season_length,
                                include_mape=cfg.model.get("include_mape", False),
                                quantile_preds=qpreds)
            cutoff_rows.append({"unique_id": uid, "cutoff": cutoff, "model": adapter.name,
                                "context_length": ctx_len, "prediction_length": H,
                                **m.to_dict()})
            if save_plot_fn and ci == 0 and i == 0 and not plot_saved:
                save_plot_fn(context=ctx, y_true=y_true, quantiles=all_q[i],
                             quantile_levels=quantile_levels, uid=uid,
                             cutoff=cutoff, output_dir=output_dir,
                             context_length=ctx_len, prediction_length=H,
                             model_name=adapter.name)
                plot_saved = True

        rows.extend(cutoff_rows)
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a",
                                         header=not checkpoint_path.exists(), index=False)
        dt = time.perf_counter() - t_cutoff
        cutoff_times.append(dt)
        print(f"cutoff {ci+1}/{len(cutoffs)}  idx={cutoff}  → {len(rows)} rows  ({dt:.1f}s)")

    total_time = time.perf_counter() - run_start
    n_forecasts = len(test_indices) * len(cutoffs)

    timing = {
        "model_load_s": round(load_time, 2),
        "total_eval_s": round(total_time, 2),
        "pure_inference_s": round(total_infer_time, 2),
        "mean_per_cutoff_s": round(float(np.mean(cutoff_times)), 3),
        "per_forecast_ms": round(total_infer_time / n_forecasts * 1000, 3),
    }
    print(f"\nTiming: load={timing['model_load_s']}s  eval={timing['total_eval_s']}s  "
          f"infer={timing['pure_inference_s']}s  ({timing['per_forecast_ms']} ms/forecast)")

    # ---- eval_info.json (with timing) ----
    eval_info = {
        "dataset": cfg.dataset.name, "model": adapter.name,
        "n_test_clients": len(test_indices),
        "test_client_ids": [ts.user_names[i] for i in test_indices],
        "n_cutoffs": len(cutoffs), "cutoffs": cutoffs,
        "cutoff_dates": [str(ts.datetimes[c]) for c in cutoffs],
        "context_length": ctx_len, "prediction_length": H,
        "quantile_levels": quantile_levels,
        "batch_size": batch_size,
        "probabilistic": bool(is_prob),
        "timing": timing,
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    # ---- Final per-client CSVs ----
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)
    results.to_csv(output_dir / f"results_per_client_cutoff_{run_date}.csv", index=False)

    # ---- Enriched summary row (metrics + config + timing) ----
    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)

    summary_row = summary.copy()
    summary_row["context_length"] = ctx_len
    summary_row["prediction_length"] = H
    summary_row["stride"] = cfg.dataset.stride
    summary_row["batch_size"] = batch_size
    summary_row["probabilistic"] = bool(is_prob)
    summary_row["model_load_s"] = timing["model_load_s"]
    summary_row["total_eval_s"] = timing["total_eval_s"]
    summary_row["pure_inference_s"] = timing["pure_inference_s"]
    summary_row["per_forecast_ms"] = timing["per_forecast_ms"]
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()

    # 1. per-run summary — WITH ctx/horizon/timing columns
    summary_row.to_csv(output_dir / "summary_by_model.csv", index=False)

    # 2. accumulated across all runs (progressive: one row appended per run)
    summary_all_path = output_dir.parent / "summary_all_runs.csv"
    acc = summary_row
    if summary_all_path.exists():
        acc = pd.concat([pd.read_csv(summary_all_path), summary_row], ignore_index=True)
    acc.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")

    return results