"""Run the statsforecast baselines on the SAME held-out test clients as your
existing PatchTST pipeline (loaded from path_client_split), sliding windows
across each test client's full timeline with the same context_length /
prediction_length / stride convention as CustomDataset.

Reports MAE/RMSE (raw and instance-normalized), MASE, optionally MAPE, EMPQ,
and — if probabilistic forecasting is enabled — WQL/CRPS, via utils.metrics
(the same module reused for every other model family).

Run from the repo root with:
    python -m src.baselines.run_statistical_baselines
    python -m src.baselines.run_statistical_baselines dataset=cer_bis model.probabilistic=true
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive
from statsforecast.utils import ConformalIntervals

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_cutoffs,
    to_statsforecast_history_df,
)
from src.utils.metrics import compute_metrics

MODEL_REGISTRY = {
    "Naive": Naive,
    "SeasonalNaive": SeasonalNaive,
    "AutoTheta": AutoTheta,
    "AutoETS": AutoETS,
    # "AutoARIMA": AutoARIMA,
}


def build_models(cfg: DictConfig) -> list:
    intervals = None
    if cfg.model.get("probabilistic", False) and cfg.model.get("use_conformal_intervals", True):
        intervals = ConformalIntervals(
            h=cfg.dataset.prediction_length,
            n_windows=cfg.model.get("conformal_n_windows", 2),
        )
    models = []
    for name in cfg.model.models:
        cls = MODEL_REGISTRY[name]
        kwargs: dict = {} if name == "Naive" else {"season_length": cfg.model.season_length}
        if intervals is not None:
            kwargs["prediction_intervals"] = intervals
        models.append(cls(**kwargs))
    return models


def quantile_preds_for_model(fc_rows: pd.DataFrame, model_name: str, levels: list[int]) -> dict[float, np.ndarray]:
    quantile_preds: dict[float, np.ndarray] = {0.5: fc_rows[model_name].to_numpy()}
    for level in levels:
        q_lo, q_hi = 0.5 - level / 200, 0.5 + level / 200
        lo_col, hi_col = f"{model_name}-lo-{level}", f"{model_name}-hi-{level}"
        if lo_col in fc_rows.columns:
            quantile_preds[q_lo] = fc_rows[lo_col].to_numpy()
        if hi_col in fc_rows.columns:
            quantile_preds[q_hi] = fc_rows[hi_col].to_numpy()
    return quantile_preds


@hydra.main(config_path="../../configs", config_name="config_statistical_baselines", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout="wide", date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}  |  {ts.n_users} clients × {ts.n_dates} steps")
    print(f"Date range: {ts.datetimes[0]} → {ts.datetimes[-1]}")

    inferred_freq = pd.infer_freq(pd.DatetimeIndex(ts.datetimes))
    if inferred_freq is None:
        raise ValueError(
            "could not infer a frequency from the data's timestamps — "
            "add an explicit `freq:` field to the dataset config."
        )
    print(f"Inferred frequency: {inferred_freq!r}")

    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])

    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length

    splits = make_cutoffs(
        ts, lags=ctx_len, horizon=H,
        step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()

    print(f"\n=== Evaluation scope ===")
    print(f"Clients  : {len(test_indices)} test clients")
    print(f"Windows  : {len(cutoffs)} test cutoffs "
          f"({ts.datetimes[cutoffs[0]]} → {ts.datetimes[cutoffs[-1]]})")
    print(f"Context  : {ctx_len} | Horizon: {H} | models={cfg.model.models}")
    n_forecasts = len(test_indices) * len(cutoffs)
    print(f"Total evals: {len(test_indices)} × {len(cutoffs)} × {len(cfg.model.models)} models "
          f"= {n_forecasts * len(cfg.model.models):,} fits")
    print(f"========================\n")

    is_probabilistic = cfg.model.get("probabilistic", False)
    levels = list(cfg.model.get("level", [])) if is_probabilistic else []
    quantile_levels = sorted(set([0.5] + [0.5 - l/200 for l in levels] + [0.5 + l/200 for l in levels])) if is_probabilistic else []

    sf = StatsForecast(
        models=build_models(cfg),
        freq=inferred_freq,
        n_jobs=cfg.model.n_jobs,
        fallback_model=SeasonalNaive(season_length=cfg.model.season_length),
    )

    # ---- eval_info.json ----
    eval_info = {
        "dataset": cfg.dataset.name,
        "models": list(cfg.model.models),
        "n_test_clients": len(test_indices),
        "test_client_ids": [ts.user_names[i] for i in test_indices],
        "n_cutoffs": len(cutoffs), "cutoffs": cutoffs,
        "cutoff_dates": [str(ts.datetimes[c]) for c in cutoffs],
        "context_length": ctx_len, "prediction_length": H,
        "probabilistic": bool(is_probabilistic),
        "levels": levels, "quantile_levels": quantile_levels,
        "max_lookback": cfg.model.get("max_lookback"),
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows: list[dict] = []
    total_fit_infer = 0.0

    run_start = time.perf_counter()
    for cutoff_idx, cutoff in enumerate(cutoffs):
        print(f"cutoff {cutoff_idx+1}/{len(cutoffs)}  idx={cutoff}  ({ts.datetimes[cutoff]}) ...")

        df_history = to_statsforecast_history_df(
            ts, cutoff, lags=ctx_len, users=test_indices,
            max_lookback=cfg.model.get("max_lookback"),
        )
        forecast_kwargs = {"level": levels} if is_probabilistic else {}

        t0 = time.perf_counter()
        forecast = sf.forecast(df=df_history, h=H, **forecast_kwargs)
        total_fit_infer += time.perf_counter() - t0

        truth = eval_batch(ts, cutoff, lags=ctx_len, horizon=H, users=test_indices)

        cutoff_rows: list[dict] = []
        for i, uid in enumerate(truth["item_ids"]):
            client_idx = test_indices[i]
            y_true = truth["y"][i, 0].numpy()
            context_window = truth["x"][i, 0].numpy()
            history_series = ts.values[: cutoff + ctx_len, client_idx]

            fc_rows = forecast.loc[forecast["unique_id"] == uid].sort_values("ds")
            for model_name in cfg.model.models:
                if model_name not in fc_rows.columns:
                    continue
                y_pred = fc_rows[model_name].to_numpy()
                quantile_preds = (
                    quantile_preds_for_model(fc_rows, model_name, levels) if is_probabilistic else None
                )
                metrics = compute_metrics(
                    y_true, y_pred,
                    context=context_window,
                    history=history_series,
                    season_length=cfg.model.season_length,
                    include_mape=cfg.model.get("include_mape", False),
                    include_empq=cfg.model.get("include_empq", False),
                    quantile_preds=quantile_preds,
                )
                cutoff_rows.append({
                    "unique_id": uid, "cutoff": cutoff, "model": model_name,
                    "context_length": ctx_len, "prediction_length": H,
                    **metrics.to_dict(),
                })

        rows.extend(cutoff_rows)
        write_header = not checkpoint_path.exists()
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a", header=write_header, index=False)
        print(f"  → checkpoint saved ({len(rows)} rows total)")

    total_time = time.perf_counter() - run_start

    # ---- Final consolidated save ----
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)
    results.to_csv(output_dir / f"results_per_client_cutoff_{run_date}.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)

    # ---- Timing ----
    timing = {
        "total_eval_s": round(total_time, 2),
        "fit_infer_s": round(total_fit_infer, 2),
        "per_forecast_ms": round(total_fit_infer / (n_forecasts * len(cfg.model.models)) * 1000, 3),
    }
    print(f"Timing: eval={timing['total_eval_s']}s  fit+infer={timing['fit_infer_s']}s  "
          f"({timing['per_forecast_ms']} ms/forecast)")
    eval_info["timing"] = timing
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    # ---- Enriched summary: per-run + accumulated across runs ----
    summary_row = summary.copy()
    summary_row["context_length"] = ctx_len
    summary_row["prediction_length"] = H
    summary_row["total_eval_s"] = timing["total_eval_s"]
    summary_row["fit_infer_s"] = timing["fit_infer_s"]
    summary_row["per_forecast_ms"] = timing["per_forecast_ms"]
    summary_row["run_date"] = run_date
    summary_row = summary_row.reset_index()

    summary_row.to_csv(output_dir / "summary_by_model.csv", index=False)

    summary_all_path = output_dir.parent / "summary_all_runs.csv"
    acc = summary_row
    if summary_all_path.exists():
        acc = pd.concat([pd.read_csv(summary_all_path), summary_row], ignore_index=True)
    acc.to_csv(summary_all_path, index=False)
    print(f"Summary all runs → {summary_all_path}")


if __name__ == "__main__":
    main()