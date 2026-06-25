"""Run the statsforecast baselines on the SAME held-out test clients as your
existing PatchTST pipeline (loaded from path_client_split), sliding windows
across each test client's full timeline with the same context_length /
prediction_length / stride convention as CustomDataset.

Reports MAE/RMSE (raw and instance-normalized), MASE, optionally MAPE, and
— if probabilistic forecasting is enabled — WQL/CRPS, via helpers.metrics
(the same module reused for every other model family).

Place this in baselines/run_statistical.py. Run from the repo root with:

    python -m baselines.run_statistical
    python -m baselines.run_statistical dataset=cer_bis model.season_length=24
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive
from statsforecast.utils import ConformalIntervals

sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset.dataset import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_sliding_cutoffs,
    make_cutoffs,
    to_statsforecast_history_df,
)
from utils.metrics import compute_metrics

MODEL_REGISTRY = {
    "Naive": Naive,
    "SeasonalNaive": SeasonalNaive,
    "AutoTheta": AutoTheta,
    "AutoETS": AutoETS,
    "AutoARIMA": AutoARIMA,
}


def build_models(cfg: DictConfig) -> list:
    intervals = None
    if cfg.model.get("probabilistic", False) and cfg.model.get("use_conformal_intervals", True):
        # Same conformal-prediction method for every model, regardless of
        # whether it has its own native intervals — keeps interval widths
        # comparable across model types (cf. discussion).
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


def quantile_preds_for_model(
    fc_rows: pd.DataFrame, model_name: str, levels: list[int]
) -> dict[float, np.ndarray]:
    """Builds a {quantile_level: predicted_values} dict from statsforecast's
    `lo`/`hi` columns for one model, plus the point forecast as the q=0.5
    proxy (a common practical approximation, not exactly the true median)."""
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
    # Hydra changes the working directory by default — resolve paths
    # against the original launch directory instead.
    data_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    split_path = hydra.utils.to_absolute_path(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout="wide", date_col=cfg.dataset.timestamp_col)

    inferred_freq = pd.infer_freq(pd.DatetimeIndex(ts.datetimes))
    if inferred_freq is None:
        raise ValueError(
            "could not infer a frequency from the data's timestamps — "
            "add an explicit `freq:` field to configs/dataset/smach.yaml "
            "and use cfg.dataset.freq instead of inferred_freq below"
        )
    print(f"Inferred frequency: {inferred_freq!r}")

    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])
    print(f"{len(test_indices)} / {len(split['test'])} test client IDs matched in the loaded data.")

    # cutoffs = make_sliding_cutoffs(
    #     ts,
    #     lags=cfg.dataset.context_length,
    #     horizon=cfg.dataset.prediction_length,
    #     stride=cfg.dataset.stride,
    #     max_windows=cfg.model.get("max_windows"),
    # )
    splits = make_cutoffs(
    ts,
    lags=cfg.dataset.context_length,
    horizon=cfg.dataset.prediction_length,
    step_size=cfg.dataset.stride,
    ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
)
    cutoffs = splits["test_cutoffs"].tolist()
    print(f"{len(cutoffs)} evaluation windows on the test clients' shared timeline.")

    is_probabilistic = cfg.model.get("probabilistic", False)
    levels = list(cfg.model.get("level", [])) if is_probabilistic else []

    sf = StatsForecast(
        models=build_models(cfg),
        freq=inferred_freq,
        n_jobs=cfg.model.n_jobs,
        fallback_model=SeasonalNaive(season_length=cfg.model.season_length),
    )

    rows: list[dict] = []
    for cutoff in cutoffs:
        print(f"Fitting on cutoff={cutoff} ({len(cutoffs)} total)...")

        df_history = to_statsforecast_history_df(
            ts,
            cutoff,
            lags=cfg.dataset.context_length,
            users=test_indices,
            max_lookback=cfg.model.get("max_lookback"),
        )
        forecast_kwargs = {"level": levels} if is_probabilistic else {}
        forecast = sf.forecast(df=df_history, h=cfg.dataset.prediction_length, **forecast_kwargs)

        truth = eval_batch(
            ts, cutoff, lags=cfg.dataset.context_length, horizon=cfg.dataset.prediction_length, users=test_indices
        )

        for i, uid in enumerate(truth["item_ids"]):
            client_idx = test_indices[i]
            y_true = truth["y"][i, 0].numpy()
            context_window = truth["x"][i, 0].numpy()
            history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

            fc_rows = forecast.loc[forecast["unique_id"] == uid].sort_values("ds")
            for model_name in cfg.model.models:
                if model_name not in fc_rows.columns:
                    continue
                y_pred = fc_rows[model_name].to_numpy()

                quantile_preds = (
                    quantile_preds_for_model(fc_rows, model_name, levels) if is_probabilistic else None
                )
                metrics = compute_metrics(
                    y_true,
                    y_pred,
                    context=context_window,
                    history=history_series,
                    season_length=cfg.model.season_length,
                    include_mape=cfg.model.get("include_mape", False),
                    quantile_preds=quantile_preds,
                )
                rows.append({"unique_id": uid, "cutoff": cutoff, "model": model_name, **metrics.to_dict()})

    results = pd.DataFrame(rows)
    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    metric_cols = [c for c in results.columns if c not in ("unique_id", "cutoff", "model")]
    summary = results.groupby("model")[metric_cols].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")


if __name__ == "__main__":
    main()
