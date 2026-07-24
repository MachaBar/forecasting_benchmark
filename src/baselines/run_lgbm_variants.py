"""LGBM baselines with selectable strategy — same test windows / metrics as
every other model family in the benchmark.

Variants (cfg.model.variant):
  global            : one shared model, recursive prediction (baseline, ~ run_lgbm.py)
  global_calendar   : same + cyclical hour/day-of-week/day-of-month features
  per_hour          : 48 models (one per half-hour-of-day slot in the horizon)
  per_hour_calendar : per_hour + cyclical day-of-week/day-of-month features

Run from the repo root:
    python -m src.baselines.run_lgbm_variants dataset=cer_bis model.variant=global
    python -m src.baselines.run_lgbm_variants dataset=cer_bis model.variant=per_hour_calendar
    python -m src.baselines.run_lgbm_variants --multirun \
        dataset=cer_bis model.variant=global,global_calendar,per_hour,per_hour_calendar
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

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.baselines.lgbm_core import build_training_table, fit_global, fit_per_hour, predict_recursive
from src.dataset.dataset import (
    client_ids_to_indices, eval_batch, load_client_split_pickle,
    load_dataset, make_cutoffs,
)
from src.utils.metrics import compute_metrics


@hydra.main(config_path="../../configs", config_name="config_lgbm_variants", version_base=None)
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant = cfg.model.variant
    assert variant in ("global", "global_calendar", "per_hour", "per_hour_calendar"), variant
    model_name = f"LGBM-{variant}"

    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    print(f"Dataset  : {cfg.dataset.name}  |  {ts.n_users} clients × {ts.n_dates} steps")
    print(f"Variant  : {variant}")

    split = load_client_split_pickle(split_path)
    train_indices = client_ids_to_indices(ts, split["train"])
    test_indices = client_ids_to_indices(ts, split["test"])

    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    lags = [l for l in list(cfg.model.lags) if l <= ctx_len]     # filter lags that fit ctx (cf. earlier fix)
    if not lags:
        raise ValueError(f"no usable lag ≤ context_length={ctx_len}")
    print(f"Lags used: {lags}")

    r_train = float(cfg.dataset.get("ratios", "0.7,0.15,0.15").split(",")[0])
    train_end = int(round(r_train * ts.n_dates))

    splits = make_cutoffs(
        ts, lags=ctx_len, horizon=H, step_size=cfg.dataset.stride,
        ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()
    if len(cutoffs) < cfg.dataset.get("min_cutoffs", 3):
        print(f"SKIP: only {len(cutoffs)} cutoffs for ctx={ctx_len}, h={H}.")
        return

    print(f"Clients  : {len(train_indices)} train / {len(test_indices)} test")
    print(f"Windows  : {len(cutoffs)} test cutoffs")

    # ---- Build training table + fit (once) ----
    include_calendar = variant.endswith("calendar")
    per_hour = variant.startswith("per_hour")
    lgbm_kwargs = dict(
        n_estimators=cfg.model.get("n_estimators", 300),
        learning_rate=cfg.model.get("learning_rate", 0.05),
        max_depth=cfg.model.get("max_depth", -1),
        num_leaves=cfg.model.get("num_leaves", 31),
        n_jobs=cfg.model.get("n_jobs", -1),
        verbose=-1,
    )

    t0 = time.perf_counter()
    X, y, slots = build_training_table(
        ts, train_indices, t_min=max(lags), t_max=train_end,
        lags=lags, n_samples=cfg.model.get("n_samples", 200_000),
        seed=cfg.get("seed", 42), include_calendar=include_calendar,
    )
    print(f"Training table: {X.shape[0]} rows × {X.shape[1]} cols")

    if per_hour:
        model = fit_per_hour(X, y, slots, **lgbm_kwargs)
        fallback_model = fit_global(X, y, **lgbm_kwargs)   # used for slots with too few samples
        print(f"Fitted {len(model)}/48 per-hour models (+1 fallback)")
    else:
        model = fit_global(X, y, **lgbm_kwargs)
        fallback_model = None
    fit_time = time.perf_counter() - t0
    print(f"Fit done in {fit_time:.1f}s")

    # ---- Eval info ----
    eval_info = {
        "dataset": cfg.dataset.name, "model": model_name, "variant": variant,
        "n_train_clients": len(train_indices), "n_test_clients": len(test_indices),
        "n_cutoffs": len(cutoffs), "cutoffs": cutoffs,
        "context_length": ctx_len, "prediction_length": H,
        "lags": lags, "fit_time_s": round(fit_time, 2),
    }
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows = []
    total_infer = 0.0

    run_start = time.perf_counter()
    for ci, cutoff in enumerate(cutoffs):
        print(f"cutoff {ci+1}/{len(cutoffs)}  idx={cutoff} ...")
        batch = eval_batch(ts, int(cutoff), lags=ctx_len, horizon=H, users=test_indices)
        context = batch["x"][:, 0, :].numpy()                       # (n_clients, ctx_len)
        future_dates = pd.DatetimeIndex(ts.datetimes[cutoff + ctx_len : cutoff + ctx_len + H])

        t_inf = time.perf_counter()
        preds = predict_recursive(
            context=context, future_dates=future_dates, lags=lags, horizon=H,
            variant=variant, model=model, fallback_model=fallback_model,
        )                                                            # (n_clients, H)
        total_infer += time.perf_counter() - t_inf

        cutoff_rows = []
        for i, uid in enumerate(batch["item_ids"]):
            y_true = batch["y"][i, 0].numpy()
            ctx = context[i]
            hist = ts.values[: cutoff + ctx_len, test_indices[i]]
            y_pred = preds[i]

            m = compute_metrics(
                y_true, y_pred, context=ctx, history=hist,
                season_length=season_length,
                include_mape=cfg.model.get("include_mape", False),
                include_empq=cfg.model.get("include_empq", True),
            )
            cutoff_rows.append({
                "unique_id": uid, "cutoff": cutoff, "model": model_name,
                "context_length": ctx_len, "prediction_length": H,
                **m.to_dict(),
            })

        rows.extend(cutoff_rows)
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a",
                                         header=not checkpoint_path.exists(), index=False)
        print(f"  → checkpoint saved ({len(rows)} rows total)")

    total_time = time.perf_counter() - run_start

    # ---- Save results + summary ----
    run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns
                   if c not in ("unique_id", "cutoff", "model", "context_length", "prediction_length")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print(summary)

    n_forecasts = len(test_indices) * len(cutoffs)
    timing = {
        "fit_time_s": round(fit_time, 2),
        "total_infer_s": round(total_infer, 2),
        "per_forecast_ms": round(total_infer / n_forecasts * 1000, 3),
    }
    eval_info["timing"] = timing
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)

    summary_row = summary.copy()
    summary_row["context_length"] = ctx_len
    summary_row["prediction_length"] = H
    summary_row["variant"] = variant
    summary_row["fit_time_s"] = timing["fit_time_s"]
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