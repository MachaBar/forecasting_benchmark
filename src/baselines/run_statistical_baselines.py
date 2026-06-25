"""Run the statsforecast baselines on the SAME held-out test clients as your
existing PatchTST pipeline (loaded from path_client_split), sliding windows
across each test client's full timeline with the same context_length /
prediction_length / stride convention as CustomDataset.

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

sys.path.append(str(Path(__file__).resolve().parents[1]))
from helpers.dataset_general import (
    client_ids_to_indices,
    eval_batch,
    load_client_split_pickle,
    load_dataset,
    make_sliding_cutoffs,
    to_statsforecast_history_df,
)

MODEL_REGISTRY = {
    "Naive": Naive,
    "SeasonalNaive": SeasonalNaive,
    "AutoTheta": AutoTheta,
    "AutoETS": AutoETS,
    "AutoARIMA": AutoARIMA,
}


def build_models(cfg: DictConfig) -> list:
    models = []
    for name in cfg.model.models:
        cls = MODEL_REGISTRY[name]
        models.append(cls() if name == "Naive" else cls(season_length=cfg.model.season_length))
    return models


def mase(y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, season_length: int) -> float:
    """Mean Absolute Scaled Error, scaled by this client's own in-sample
    seasonal-naive error (cf. the GIFT-Eval / fev-bench discussion)."""
    if len(history) <= season_length:
        return float("nan")
    naive_errors = np.abs(history[season_length:] - history[:-season_length])
    scale = max(naive_errors.mean(), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


@hydra.main(config_path="../configs", config_name="config_statistical_baselines", version_base=None)
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

    cutoffs = make_sliding_cutoffs(
        ts,
        lags=cfg.dataset.context_length,
        horizon=cfg.dataset.prediction_length,
        stride=cfg.dataset.stride,
        max_windows=cfg.model.get("max_windows"),
    )
    print(f"{len(cutoffs)} evaluation windows on the test clients' shared timeline.")

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
        forecast = sf.forecast(df=df_history, h=cfg.dataset.prediction_length)
        truth = eval_batch(
            ts, cutoff, lags=cfg.dataset.context_length, horizon=cfg.dataset.prediction_length, users=test_indices
        )

        for i, uid in enumerate(truth["item_ids"]):
            client_idx = test_indices[i]
            y_true = truth["y"][i, 0].numpy()
            history_series = ts.values[: cutoff + cfg.dataset.context_length, client_idx]

            fc_rows = forecast.loc[forecast["unique_id"] == uid].sort_values("ds")
            for model_name in cfg.model.models:
                if model_name not in fc_rows.columns:
                    continue
                y_pred = fc_rows[model_name].to_numpy()
                rows.append(
                    {
                        "unique_id": uid,
                        "cutoff": cutoff,
                        "model": model_name,
                        "mae": float(np.mean(np.abs(y_true - y_pred))),
                        "mase": mase(y_true, y_pred, history_series, cfg.model.season_length),
                    }
                )

    results = pd.DataFrame(rows)
    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "results_per_client_cutoff.csv", index=False)

    summary = results.groupby("model")[["mae", "mase"]].mean().sort_values("mase")
    print("\n=== Mean over all TEST clients and evaluation windows ===")
    print(summary)
    summary.to_csv(output_dir / "summary_by_model.csv")


if __name__ == "__main__":
    main()
