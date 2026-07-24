"""Zero-shot inference with PatchTST-FM (IBM Time Series Foundation Model).

Run from the repo root:
    python -m scripts.run_patchtst_fm
    python -m scripts.run_patchtst_fm dataset=cer model.batch_size=16
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
)
from src.utils.metrics import compute_metrics


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
    """Saves a PNG showing context + forecast with quantile bands."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    q_idx = {q: j for j, q in enumerate(quantile_levels)}
    median = all_qf[q_idx.get(0.5, len(quantile_levels) // 2)]
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
    ax.set_title(f"PatchTST-FM forecast — client {uid}, cutoff {cutoff}")
    ax.set_xlabel("timestep relative to forecast origin")
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = output_dir / f"forecast_plot_client{uid}_cutoff{cutoff}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved → {path}")


@hydra.main(config_path="../configs", config_name="config_patchtst_fm", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline
    except ImportError:
        raise ImportError(
            "tsfm_public is not installed. "
            "Run: pip install time-series-foundation-models  (or uv add time-series-foundation-models)"
        )

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}")
    print(f"Clients  : {ts.n_users}")
    print(f"Timesteps: {ts.n_dates}")
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
    print(f"Total evals: {len(test_indices)} clients × {len(cutoffs)} cutoffs = {len(test_indices)*len(cutoffs):,} inference calls")
    print(f"========================\n")

    prediction_length = cfg.dataset.prediction_length
    season_length = cfg.model.get("season_length", 48)

    # Load model
    # Resolve local checkpoint paths relative to the original cwd (Hydra changes cwd to the run dir).
    # HF repo ids (e.g. "ibm-research/patchtst-fm-r1") are left untouched.
    model_name = cfg.model.model_name
    if not model_name.startswith(("http://", "https://")) and "/" in model_name:
        candidate = Path(hydra.utils.to_absolute_path(model_name))
        if candidate.exists():
            model_name = str(candidate)

    print(f"Loading PatchTST-FM from {model_name}...")
    t0 = time.time()
    model = PatchTSTFMForPrediction.from_pretrained(model_name)
    model_load_time = time.time() - t0
    print(f"Model loaded in {model_load_time:.1f}s")

    # Create pipeline
    pipe = TimeSeriesForecastingPipeline(
        model=model,
        id_columns=[],
        timestamp_column=cfg.dataset.timestamp_col,
        target_columns=["value"],  # Placeholder; will override per batch
        max_context_length=min(cfg.model.max_context_length, cfg.dataset.context_length + 100),
        context_length=cfg.dataset.context_length,
        prediction_length=prediction_length,
        batch_size=cfg.model.batch_size,
        impute_method=cfg.model.impute_method,
        device=cfg.model.device,
        quantile_levels=cfg.model.quantile_levels,
    )

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "results_checkpoint.csv"

    rows: list[dict] = []
    plot_cutoff_idx = 0
    plot_client_idx = 0
    plot_saved = False

    for cutoff_loop_idx, cutoff in enumerate(cutoffs):
        print(f"cutoff {cutoff_loop_idx+1}/{len(cutoffs)}  idx={cutoff}  ({ts.datetimes[cutoff]}) ...")

        batch = eval_batch(
            ts, int(cutoff),
            lags=cfg.dataset.context_length,
            horizon=prediction_length,
            users=test_indices,
        )
        x = batch["x"]  # (n_clients, 1, lags)
        y_true_all = batch["y"]  # (n_clients, 1, horizon)
        n_clients = x.shape[0]

        # Prepare dataframe for pipeline
        # PatchTST-FM expects: timestamp, target columns, and id columns
        forecast_dfs = []
        cutoff_rows: list[dict] = []
        inference_times = []

        for i, uid in enumerate(batch["item_ids"]):
            client_idx = test_indices[i]
            y_true = y_true_all[i, 0].numpy()
            context_window = x[i, 0].numpy()
            history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

            # Create dataframe for this client's history
            history_dates = pd.DatetimeIndex(ts.datetimes[: cutoff + cfg.dataset.context_length])
            df_hist = pd.DataFrame({
                cfg.dataset.timestamp_col: history_dates,
                "value": history_series,
            })

            # Run inference
            t0 = time.time()
            forecast_result = pipe(df_hist)
            inference_time = time.time() - t0
            inference_times.append(inference_time)

            # Extract forecasts
            # forecast_result is a DataFrame with columns like "value_0.1", "value_0.5", "value_0.9", etc.
            quantile_preds = {}
            for q in cfg.model.quantile_levels:
                col_name = f"value_{q:.2f}"
                if col_name in forecast_result.columns:
                    quantile_preds[q] = forecast_result[col_name].values[-prediction_length:]
                else:
                    # Fallback: use median if column doesn't exist
                    if "value_0.50" in forecast_result.columns:
                        quantile_preds[q] = forecast_result["value_0.50"].values[-prediction_length:]

            # Point forecast: use median (q=0.5)
            if "value_0.50" in forecast_result.columns:
                y_pred = forecast_result["value_0.50"].values[-prediction_length:]
            else:
                # Fallback: average of available quantiles
                if quantile_preds:
                    y_pred = np.mean(list(quantile_preds.values()), axis=0)
                else:
                    y_pred = np.full(prediction_length, np.nan)

            metrics = compute_metrics(
                y_true,
                y_pred,
                context=context_window,
                history=history_series,
                season_length=season_length,
                include_mape=cfg.model.get("include_mape", False),
                quantile_preds=quantile_preds if quantile_preds else None,
            )
            cutoff_rows.append({
                "unique_id": uid,
                "cutoff": cutoff,
                "model": "PatchTST-FM",
                "inference_ms": inference_time * 1000,
                **metrics.to_dict(),
            })

            # Save diagnostic plot for first cutoff, first client
            if cutoff_loop_idx == plot_cutoff_idx and i == plot_client_idx and not plot_saved:
                all_qf = np.array([quantile_preds.get(q, y_pred) for q in cfg.model.quantile_levels])
                _save_forecast_plot(
                    context=context_window,
                    y_true=y_true,
                    all_qf=all_qf,
                    quantile_levels=cfg.model.quantile_levels,
                    uid=uid,
                    cutoff=cutoff,
                    output_dir=output_dir,
                )
                plot_saved = True

        rows.extend(cutoff_rows)

        # Progressive checkpoint
        write_header = not checkpoint_path.exists()
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
        print(f"  → checkpoint saved ({len(rows)} rows total, inference={np.mean(inference_times)*1000:.1f}ms±{np.std(inference_times)*1000:.1f}ms)")

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns if c not in ("unique_id", "cutoff", "model", "inference_ms")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")


if __name__ == "__main__":
    main()