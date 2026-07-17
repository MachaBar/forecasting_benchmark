"""Prophet baseline — fit per client at each cutoff, same test windows /
metrics as every other model family.

Prophet is a LOCAL model (one fit per series): it measures generalization
through time only, not to unseen clients (cf. the statistical baselines).

Run from the repo root:
    python -m src.baselines.run_prophet dataset=cer_bis
    python -m src.baselines.run_prophet dataset=cer_bis model.probabilistic=true
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    client_ids_to_indices, eval_batch, load_client_split_pickle,
    load_dataset, make_cutoffs,
)
from src.utils.metrics import compute_metrics

# silence Prophet / cmdstanpy chatter
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _fit_predict_one(prophet_kwargs, ds_hist, y_hist, ds_future,
                     quantile_levels, is_prob, n_samples):
    """Fit Prophet on one client's history and forecast the future timestamps.
    Returns (point (H,), quantile_dict or None)."""
    from prophet import Prophet
    m = Prophet(**prophet_kwargs)
    m.fit(pd.DataFrame({"ds": ds_hist, "y": y_hist}))
    future = pd.DataFrame({"ds": ds_future})

    if is_prob:
        # posterior predictive samples → empirical quantiles
        samples = m.predictive_samples(future)["yhat"]     # (H, n_samples)
        point = np.median(samples, axis=1)
        qpreds = {q: np.quantile(samples, q, axis=1) for q in quantile_levels}
        return point, qpreds
    else:
        fc = m.predict(future)
        return fc["yhat"].to_numpy(), None


@hydra.main(config_path="../../configs", config_name="config_prophet", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}  |  {ts.n_users} clients × {ts.n_dates} steps")

    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])

    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    is_prob = cfg.model.get("probabilistic", False)
    n_samples = cfg.model.get("uncertainty_samples", 200)
    quantile_levels = sorted(set([0.1, 0.5, 0.9] +
        (list(cfg.model.get("quantile_levels", [])) if is_prob else [])))

    splits = make_cutoffs(
        ts, lags=ctx_len, horizon=H,
        step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    # cutoffs = splits["test_cutoffs"].tolist()

        # ---- Tuning support: evaluate on val cutoffs + subsample clients ----
    eval_split = cfg.get("eval_split", "test")            # "val" pour le tuning
    cutoffs_all = splits["val_cutoffs"] if eval_split == "val" else splits["test_cutoffs"]
    cutoffs = cutoffs_all.tolist()

    tune_n = cfg.model.get("tune_n_clients")              # ex. 50 pour aller vite
    if tune_n is not None and tune_n < len(test_indices):
        rng = np.random.default_rng(cfg.get("seed", 42))
        test_indices = sorted(rng.choice(test_indices, size=tune_n, replace=False).tolist())
        print(f"[tuning] subsampled to {len(test_indices)} clients, split={eval_split}")

    # guard: skip runs with too few windows (cf. our earlier discussion)
    if len(cutoffs) < cfg.dataset.get("min_cutoffs", 3):
        print(f"SKIP: only {len(cutoffs)} cutoffs for ctx={ctx_len}, h={H}.")
        return

    print(f"\n=== Evaluation scope ===")
    print(f"Clients  : {len(test_indices)} test clients")
    print(f"Windows  : {len(cutoffs)} cutoffs")
    print(f"Context  : {ctx_len} | Horizon: {H}")
    print(f"Total fits: {len(test_indices) * len(cutoffs):,} Prophet fits")
    print(f"========================\n")

    prophet_kwargs = dict(
        changepoint_prior_scale=cfg.model.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=cfg.model.get("seasonality_prior_scale", 10.0),
        seasonality_mode=cfg.model.get("seasonality_mode", "additive"),
        daily_seasonality=cfg.model.get("daily_seasonality", True),
        weekly_seasonality=cfg.model.get("weekly_seasonality", True),
        yearly_seasonality=cfg.model.get("yearly_seasonality", False),
        changepoint_range=cfg.model.get("changepoint_range", 0.8),
        uncertainty_samples=n_samples if is_prob else 0,
        interval_width=cfg.model.get("interval_width", 0.8),
    )

    max_lookback = cfg.model.get("max_lookback")   # None = all history
    all_dates = pd.DatetimeIndex(ts.datetimes)

    eval_info = {
        "dataset": cfg.dataset.name, "model": "Prophet",
        "n_test_clients": len(test_indices),
        "n_cutoffs": len(cutoffs), "cutoffs": cutoffs,
        "context_length": ctx_len, "prediction_length": H,
        "probabilistic": bool(is_prob), "quantile_levels": quantile_levels,
        "prophet_kwargs": {k: v for k, v in prophet_kwargs.items()},
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2, default=str)

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows = []
    total_fit = 0.0

    run_start = time.perf_counter()
    for ci, cutoff in enumerate(cutoffs):
        print(f"cutoff {ci+1}/{len(cutoffs)}  idx={cutoff} ...")
        truth = eval_batch(ts, cutoff, lags=ctx_len, horizon=H, users=test_indices)

        # history end = cutoff + ctx ; future = next H timestamps
        hist_end = cutoff + ctx_len
        hist_start = max(0, hist_end - max_lookback) if max_lookback else 0
        ds_hist = all_dates[hist_start:hist_end]
        ds_future = all_dates[hist_end:hist_end + H]

        cutoff_rows = []
        for i, uid in enumerate(truth["item_ids"]):
            client_idx = test_indices[i]
            y_hist = ts.values[hist_start:hist_end, client_idx]
            y_true = truth["y"][i, 0].numpy()
            context_window = truth["x"][i, 0].numpy()
            history_series = ts.values[:hist_end, client_idx]

            t0 = time.perf_counter()
            y_pred, qpreds = _fit_predict_one(
                prophet_kwargs, ds_hist, y_hist, ds_future,
                quantile_levels, is_prob, n_samples,
            )
            total_fit += time.perf_counter() - t0

            metrics = compute_metrics(
                y_true, y_pred,
                context=context_window,
                history=history_series,
                season_length=season_length,
                include_mape=cfg.model.get("include_mape", False),
                include_empq=cfg.model.get("include_empq", True),
                quantile_preds=qpreds,
            )
            cutoff_rows.append({
                "unique_id": uid, "cutoff": cutoff, "model": "Prophet",
                "context_length": ctx_len, "prediction_length": H,
                **metrics.to_dict(),
            })

        rows.extend(cutoff_rows)
        write_header = not checkpoint_path.exists()
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
        print(f"  → checkpoint saved ({len(rows)} rows total)")

    total_time = time.perf_counter() - run_start

    # ---- Save results + summary ----
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)

    n_forecasts = len(test_indices) * len(cutoffs)
    timing = {
        "total_eval_s": round(total_time, 2),
        "fit_infer_s": round(total_fit, 2),
        "per_forecast_ms": round(total_fit / n_forecasts * 1000, 3),
    }
    print(f"Timing: eval={timing['total_eval_s']}s  fit={timing['fit_infer_s']}s  "
          f"({timing['per_forecast_ms']} ms/forecast)")
    eval_info["timing"] = timing
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2, default=str)

    summary_row = summary.copy()
    summary_row["context_length"] = ctx_len
    summary_row["prediction_length"] = H
    summary_row["n_cutoffs"] = len(cutoffs)
    summary_row["total_eval_s"] = timing["total_eval_s"]
    summary_row["per_forecast_ms"] = timing["per_forecast_ms"]
    summary_row["run_date"] = run_date
    summary_row["changepoint_prior_scale"] = prophet_kwargs["changepoint_prior_scale"]
    summary_row["seasonality_prior_scale"] = prophet_kwargs["seasonality_prior_scale"]
    summary_row["eval_split"] = eval_split
    summary_row = summary_row.reset_index()
    summary_row.to_csv(output_dir / "summary_by_model.csv", index=False)

    summary_all = output_dir.parent / "summary_all_runs.csv"
    acc = summary_row
    if summary_all.exists():
        acc = pd.concat([pd.read_csv(summary_all), summary_row], ignore_index=True)
    acc.to_csv(summary_all, index=False)
    print(f"Summary all runs → {summary_all}")


if __name__ == "__main__":
    main()