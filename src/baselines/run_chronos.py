"""Zero-shot inference with Chronos-2.

Run from the repo root:
    python -m scripts.run_chronos
    python -m scripts.run_chronos dataset=cer model.device_map=cuda
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
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
    """Saves a PNG showing the last `n_context_shown` context steps + the
    forecast horizon with median and 10/90 quantile band."""
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


@hydra.main(config_path="../configs", config_name="config_chronos", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        from chronos import Chronos2Pipeline
    except ImportError:
        raise ImportError(
            "chronos is not installed. "
            "Run: pip install chronos-forecasting  (or uv add chronos-forecasting)"
        )

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Loaded {ts.n_users} clients × {ts.n_dates} timesteps")

    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])
    print(f"{len(test_indices)} / {len(split['test'])} test client IDs matched.")

    splits = make_cutoffs(
        ts,
        lags=cfg.dataset.context_length,
        horizon=cfg.dataset.prediction_length,
        step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()
    print(f"{len(cutoffs)} test evaluation windows.")

    prediction_length = cfg.dataset.prediction_length

    pipeline = Chronos2Pipeline.from_pretrained(
        cfg.model.weights_path,
        device_map=cfg.model.device_map,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        local_files_only=True,
    )

    is_probabilistic = cfg.model.get("probabilistic", False)
    num_samples = cfg.model.get("num_samples", 20)
    # Always request q10/q50/q90 so we can plot; metrics use all levels when probabilistic=true
    plot_quantiles = [0.1, 0.5, 0.9]
    quantile_levels = sorted(set(plot_quantiles + (list(cfg.model.get("quantile_levels", [])) if is_probabilistic else [])))
    batch_size = cfg.model.get("batch_size", 32)
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))

    # Fixed window index (first test cutoff) and first test client used for the diagnostic plot
    plot_cutoff_idx = 0
    plot_client_idx = 0
    plot_saved = False

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cutoff_loop_idx, cutoff in enumerate(cutoffs):
        print(f"cutoff={cutoff} ({len(cutoffs)} total) ...")

        batch = eval_batch(
            ts, int(cutoff),
            lags=cfg.dataset.context_length,
            horizon=prediction_length,
            users=test_indices,
        )
        x = batch["x"]  # (n_clients, 1, lags)
        y_true_all = batch["y"]  # (n_clients, 1, horizon)
        n_clients = x.shape[0]

        # Run inference in mini-batches to avoid OOM
        quantile_forecasts = []  # list of (batch, n_quantiles, horizon)
        for start in range(0, n_clients, batch_size):
            end = min(start + batch_size, n_clients)
            context_batch = [x[i, 0] for i in range(start, end)]
            qf = pipeline.predict_quantiles(
                context_batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
                num_samples=num_samples,
            )
            # predict_quantiles returns (batch, n_quantiles, horizon) tensor
            quantile_forecasts.append(qf.numpy() if isinstance(qf, torch.Tensor) else np.array(qf))

        all_qf = np.concatenate(quantile_forecasts, axis=0)  # (n_clients, n_quantiles, horizon)

        for i, uid in enumerate(batch["item_ids"]):
            client_idx = test_indices[i]
            y_true = y_true_all[i, 0].numpy()
            context_window = x[i, 0].numpy()
            history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

            # Median forecast (q=0.5) as point prediction
            median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else len(quantile_levels) // 2
            y_pred = all_qf[i, median_idx]

            quantile_preds = (
                {q: all_qf[i, j] for j, q in enumerate(quantile_levels)}
                if is_probabilistic else None
            )

            metrics = compute_metrics(
                y_true,
                y_pred,
                context=context_window,
                history=history_series,
                season_length=season_length,
                include_mape=cfg.model.get("include_mape", False),
                quantile_preds=quantile_preds,
            )
            rows.append({
                "unique_id": uid,
                "cutoff": cutoff,
                "model": "Chronos2",
                **metrics.to_dict(),
            })

            # Save diagnostic plot for the first test window, first client
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

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns if c not in ("unique_id", "cutoff", "model")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")


if __name__ == "__main__":
    main()
