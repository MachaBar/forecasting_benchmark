"""Global LightGBM forecaster (skforecast) on the SAME test windows / metrics
as Chronos2, TS-ICL, PatchTST and the statistical baselines.

One LightGBM trained across all TRAIN clients (recursive multi-step via
skforecast's ForecasterRecursiveMultiSeries), then evaluated at the fixed
test cutoffs on the TEST clients.

Run from the repo root:
    python -m src.baselines.run_lgbm dataset=cer_bis
    python -m src.baselines.run_lgbm dataset=cer_bis model.probabilistic=true
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
from lightgbm import LGBMRegressor
from omegaconf import DictConfig
from skforecast.recursive import ForecasterRecursiveMultiSeries

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import (
    client_ids_to_indices, eval_batch, load_client_split_pickle,
    load_dataset, make_cutoffs,
)
from src.utils.metrics import compute_metrics


def _series_dict(ts, indices, start, end):
    """{client_id: pd.Series} over [start, end) with a RangeIndex.
    RangeIndex avoids frequency/gap issues (dropna can create irregular dates)
    and is fully supported by skforecast for lag-based forecasting."""
    n = end - start
    idx = pd.RangeIndex(start=0, stop=n, step=1)
    return {
        ts.user_names[u]: pd.Series(ts.values[start:end, u], index=idx, name=ts.user_names[u])
        for u in indices
    }


def _extract_quantiles(pred_q, uid, quantile_levels, H):
    """Robustly pull the (H, Q) quantile matrix for one client from
    skforecast's long output, whatever the exact quantile column names."""
    sub = pred_q[pred_q["level"] == uid]
    cols = [c for c in sub.columns if c not in ("level",)]
    qpreds = {}
    for q in quantile_levels:
        # try common naming conventions in order
        candidates = [f"q_{q}", str(q), f"quantile_{q}", f"{q:.2f}", f"q_{q:.2f}"]
        col = next((c for c in candidates if c in sub.columns), None)
        if col is None:
            raise KeyError(
                f"quantile {q} not found. Available columns: {sub.columns.tolist()}"
            )
        qpreds[q] = sub[col].to_numpy()[:H]
    return qpreds


@hydra.main(config_path="../../configs", config_name="config_lgbm", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}  |  {ts.n_users} clients × {ts.n_dates} steps")

    split = load_client_split_pickle(split_path)
    train_indices = client_ids_to_indices(ts, split["train"])
    test_indices = client_ids_to_indices(ts, split["test"])

    splits = make_cutoffs(
        ts, lags=cfg.dataset.context_length, horizon=cfg.dataset.prediction_length,
        step_size=cfg.dataset.stride, ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()
    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    is_prob = cfg.model.get("probabilistic", False)
    quantile_levels = sorted(set([0.1, 0.5, 0.9] +
        (list(cfg.model.get("quantile_levels", [])) if is_prob else [])))

    # ---- Temporal boundary: train the forecaster only on the TRAIN period ----
    r_train = float(cfg.dataset.get("ratios", "0.7,0.15,0.15").split(",")[0])
    train_end = int(round(r_train * ts.n_dates))

    print(f"Clients  : {len(train_indices)} train / {len(test_indices)} test")
    print(f"Windows  : {len(cutoffs)} test cutoffs")
    print(f"Context  : {ctx_len} | Horizon: {H} | lags={list(cfg.model.lags)}")

    # ---- Build + fit the global forecaster (once, on train clients/period) ----
    lgbm = LGBMRegressor(
        n_estimators=cfg.model.get("n_estimators", 500),
        learning_rate=cfg.model.get("learning_rate", 0.05),
        max_depth=cfg.model.get("max_depth", -1),
        num_leaves=cfg.model.get("num_leaves", 31),
        n_jobs=cfg.model.get("n_jobs", -1),
        verbose=-1,
    )
    forecaster = ForecasterRecursiveMultiSeries(
        lgbm,
        lags=list(cfg.model.lags),
        encoding=None,           # truly global model → generalizes to unseen clients
    )

    train_series = _series_dict(ts, train_indices, 0, train_end)
    t0 = time.perf_counter()
    # forecaster.fit(series=train_series)
    forecaster.fit(series=train_series, store_in_sample_residuals=True) 
    fit_time = time.perf_counter() - t0
    print(f"Fitted global LGBM on {len(train_series)} train clients in {fit_time:.1f}s")

    # ---- Probabilistic: pool per-level residuals into '_unknown_level' ----
    # skforecast keys residuals by training client; unseen test clients fall
    # back to the '_unknown_level' bucket, which is not auto-populated. Since
    # the model is global (encoding=None) all clients share one error signature,
    # so we pool every training client's residuals into that bucket.
    if is_prob:
        res = forecaster.in_sample_residuals_
        print(f"[debug] residual keys: {list(res.keys())}, "
              f"non-None: {[k for k, v in res.items() if v is not None]}")

        # Si _unknown_level est déjà rempli, rien à faire.
        if res.get("_unknown_level") is None:
            arrays = [v for k, v in res.items() if v is not None and k != "_unknown_level"]
            if not arrays:
                raise RuntimeError(
                    "No in-sample residuals available. Ensure fit(..., "
                    "store_in_sample_residuals=True) was called."
                )
            forecaster.in_sample_residuals_["_unknown_level"] = np.concatenate(arrays)
        print(f"'_unknown_level' residuals: "
              f"{len(forecaster.in_sample_residuals_['_unknown_level'])}")

    # ---- Save eval_info ----
    eval_info = {
        "dataset": cfg.dataset.name, "model": "LGBM",
        "n_train_clients": len(train_indices), "n_test_clients": len(test_indices),
        "n_cutoffs": len(cutoffs), "cutoffs": cutoffs,
        "context_length": ctx_len, "prediction_length": H,
        "lags": list(cfg.model.lags), "probabilistic": bool(is_prob),
        "quantile_levels": quantile_levels, "fit_time_s": round(fit_time, 2),
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows = []
    total_infer = 0.0

    for ci, cutoff in enumerate(cutoffs):
        print(f"cutoff {ci+1}/{len(cutoffs)}  idx={cutoff} ...")
        last_window = _series_dict(ts, test_indices, cutoff, cutoff + ctx_len)
        last_window_df = pd.DataFrame(last_window)

        batch = eval_batch(ts, int(cutoff), lags=ctx_len, horizon=H, users=test_indices)

        t_inf = time.perf_counter()
        if is_prob:
            pred_q = forecaster.predict_quantiles(
                steps=H, quantiles=quantile_levels,
                last_window=last_window_df, n_boot=cfg.model.get("n_boot", 100),
                use_in_sample_residuals=True,
                use_binned_residuals=False,
            )
            point = forecaster.predict(steps=H, last_window=last_window_df)
        else:
            point = forecaster.predict(steps=H, last_window=last_window_df)
        total_infer += time.perf_counter() - t_inf

        # one-time debug of column names on the first probabilistic cutoff
        if is_prob and ci == 0:
            print(f"[debug] pred_q columns: {pred_q.columns.tolist()}")

        cutoff_rows = []
        for i, uid in enumerate(batch["item_ids"]):
            y_true = batch["y"][i, 0].numpy()
            ctx = batch["x"][i, 0].numpy()
            hist = ts.values[: cutoff + ctx_len, test_indices[i]]

            y_pred = point.loc[point["level"] == uid, "pred"].to_numpy()[:H]

            qpreds = None
            if is_prob:
                qpreds = _extract_quantiles(pred_q, uid, quantile_levels, H)

            m = compute_metrics(y_true, y_pred, context=ctx, history=hist,
                                season_length=season_length,
                                include_mape=cfg.model.get("include_mape", False),
                                quantile_preds=qpreds)
            cutoff_rows.append({"unique_id": uid, "cutoff": cutoff, "model": "LGBM",
                                "context_length": ctx_len, "prediction_length": H,
                                **m.to_dict()})

        rows.extend(cutoff_rows)
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a",
                                         header=not checkpoint_path.exists(), index=False)

    # ---- Save results + summary ----
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print(summary)

    # ---- Timing ----
    n_forecasts = len(test_indices) * len(cutoffs)
    timing = {
        "fit_time_s": round(fit_time, 2),
        "total_infer_s": round(total_infer, 2),
        "per_forecast_ms": round(total_infer / n_forecasts * 1000, 3),
    }
    print(f"Timing: fit={timing['fit_time_s']}s  infer={timing['total_infer_s']}s  "
          f"({timing['per_forecast_ms']} ms/forecast)")

    eval_info["timing"] = timing
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    summary_row = summary.copy()
    summary_row["context_length"] = ctx_len
    summary_row["prediction_length"] = H
    summary_row["fit_time_s"] = timing["fit_time_s"]
    summary_row["total_infer_s"] = timing["total_infer_s"]
    summary_row["per_forecast_ms"] = timing["per_forecast_ms"]
    summary_row["run_date"] = run_date
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