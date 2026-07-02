"""Zero-shot inference with Chronos-2.

Run from the repo root:
    python -m src.baselines.run_chronos
    python -m src.baselines.run_chronos dataset=cer model.device_map=cuda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from utils.metrics import compute_metrics


def _save_forecast_plot(
    *,
    context: np.ndarray,
    y_true: np.ndarray,
    all_qf: np.ndarray,
    quantile_levels: list[float],
    uid: str,
    cutoff: int,
    output_dir: Path,
    n_context_shown: int = 96,
) -> None:
    q_idx = {q: j for j, q in enumerate(quantile_levels)}
    median = all_qf[q_idx[0.5]]
    q10 = all_qf[q_idx.get(0.1, 0)]
    q90 = all_qf[q_idx.get(0.9, len(quantile_levels) - 1)]

    n_ctx = min(n_context_shown, len(context))
    ctx_x = np.arange(-n_ctx, 0)
    fc_x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx_x, context[-n_ctx:], color="steelblue", lw=1.5, label="context")
    ax.plot(fc_x, y_true, color="black", lw=1.5, label="ground truth")
    ax.plot(fc_x, median, color="tomato", lw=1.5, label="median (q50)")
    ax.fill_between(fc_x, q10, q90, color="tomato", alpha=0.2, label="q10–q90")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Chronos-2 forecast — client {uid}, cutoff {cutoff}")
    ax.set_xlabel("timestep relative to forecast origin")
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = output_dir / f"forecast_plot_client{uid}_cutoff{cutoff}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved → {path}")


@hydra.main(config_path="../../configs", config_name="config_chronos", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        from chronos.chronos2 import Chronos2Pipeline
    except ImportError:
        try:
            from chronos import ChronosPipeline as Chronos2Pipeline
        except ImportError:
            raise ImportError("chronos is not installed. Run: pip install chronos-forecasting")

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
    print(f"Total evals: {len(test_indices)} clients × {len(cutoffs)} cutoffs = {len(test_indices)*len(cutoffs):,} inference calls")
    print(f"Context  : {cfg.dataset.context_length} steps = {cfg.dataset.context_length*0.5 :.0f}h")
    print(f"Horizon  : {cfg.dataset.prediction_length} steps = {cfg.dataset.prediction_length*0.5:.0f}h")
    print(f"========================\n")

    prediction_length = cfg.dataset.prediction_length
    is_probabilistic = cfg.model.get("probabilistic", False)
    num_samples = cfg.model.get("num_samples", 20)
    plot_quantiles = [0.1, 0.5, 0.9]
    quantile_levels = sorted(set(plot_quantiles + (list(cfg.model.get("quantile_levels", [])) if is_probabilistic else [])))
    batch_size = cfg.model.get("batch_size", 32)
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))

    pipeline = Chronos2Pipeline.from_pretrained(
        cfg.model.weights_path,
        device_map=cfg.model.device_map,
        dtype=getattr(torch, cfg.model.torch_dtype),
        local_files_only=True,
    )
    has_predict_quantiles = hasattr(pipeline, "predict_quantiles")

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eval info
    eval_info = {
        "dataset": cfg.dataset.name,
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
        "context_length": cfg.dataset.context_length,
        "prediction_length": cfg.dataset.prediction_length,
        "model": cfg.model.name,
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)
    print(f"Eval info saved → {output_dir / 'eval_info.json'}")

    checkpoint_path = output_dir / "results_checkpoint.csv"
    plot_cutoff_idx, plot_client_idx, plot_saved = 0, 0, False

    rows: list[dict] = []
    for cutoff_loop_idx, cutoff in enumerate(cutoffs):
        print(f"cutoff {cutoff_loop_idx+1}/{len(cutoffs)}  idx={cutoff}  ({ts.datetimes[cutoff]}) ...")

        batch = eval_batch(
            ts, int(cutoff),
            lags=cfg.dataset.context_length,
            horizon=prediction_length,
            users=test_indices,
        )
        x = batch["x"]          # (n_clients, 1, lags)
        y_true_all = batch["y"] # (n_clients, 1, horizon)
        n_clients = x.shape[0]

        quantile_forecasts = []
        raw_samples_list = []
        for start in range(0, n_clients, batch_size):
            end = min(start + batch_size, n_clients)
            context_batch = [x[i, 0] for i in range(start, end)]
            if has_predict_quantiles:
                qf_result = pipeline.predict_quantiles(
                    context_batch,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                )
                # shape: (batch, 1, horizon, n_quantiles) -> (batch, n_quantiles, horizon)
                qf = qf_result[0]
                qf = qf.numpy() if isinstance(qf, torch.Tensor) else np.asarray(qf)
                qf = qf[:, 0, :, :]          # (batch, horizon, n_quantiles)
                qf = qf.transpose(0, 2, 1)   # (batch, n_quantiles, horizon)
                quantile_forecasts.append(qf)
            else:
                samples = pipeline.predict(
                    context_batch,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                )
                raw_samples_list.append(samples.numpy() if isinstance(samples, torch.Tensor) else np.array(samples))

        if has_predict_quantiles:
            all_qf = np.concatenate(quantile_forecasts, axis=0)
        else:
            all_samples = np.concatenate(raw_samples_list, axis=0)
            all_qf = np.quantile(all_samples, quantile_levels, axis=1).transpose(1, 0, 2)

        cutoff_rows: list[dict] = []
        for i, uid in enumerate(batch["item_ids"]):
            client_idx = test_indices[i]
            y_true = y_true_all[i, 0].numpy()
            context_window = x[i, 0].numpy()
            history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

            median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
            y_pred = all_qf[i, median_idx]

            quantile_preds = (
                {q: all_qf[i, j] for j, q in enumerate(quantile_levels)}
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
            "model": "Chronos2",
            "context_length": cfg.dataset.context_length,
            "prediction_length": cfg.dataset.prediction_length,
            **metrics.to_dict(),
        })

            if cutoff_loop_idx == plot_cutoff_idx and i == plot_client_idx and not plot_saved:
                _save_forecast_plot(
                    context=context_window,
                    y_true=y_true,
                    all_qf=all_qf[i],
                    quantile_levels=quantile_levels,
                    uid=uid,
                    cutoff=cutoff,
                    output_dir=output_dir,
                )
                plot_saved = True

        rows.extend(cutoff_rows)

        write_header = not checkpoint_path.exists()
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
        print(f"  → checkpoint saved ({len(rows)} rows total)")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    # metric_cols = [c for c in results.columns if c not in ("unique_id", "cutoff", "model")]
    # summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    # print("\n=== Mean over all TEST clients and evaluation windows ===")
    # print(summary)
    # summary.to_csv(output_dir / "summary_by_model.csv")

    from datetime import datetime
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)
    results.to_csv(output_dir / f"results_per_client_cutoff_{run_date}.csv", index=False)

    metric_cols = [c for c in results.columns if c not in ("unique_id", "cutoff", "model")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")

    # Accumulated summary across all runs
    summary_row = summary.copy()
    summary_row["context_length"] = cfg.dataset.context_length
    summary_row["prediction_length"] = cfg.dataset.prediction_length
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()

    summary_all_path = output_dir / "summary_all_runs.csv"
    if summary_all_path.exists():
        existing = pd.read_csv(summary_all_path)
        summary_row = pd.concat([existing, summary_row], ignore_index=True)
    summary_row.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")


if __name__ == "__main__":
    main()