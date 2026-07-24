"""Shared feature engineering, fitting and recursive prediction for the LGBM
variants. Two strategies, selected via `variant`:

  - "global"          : one LGBM shared by all clients/hours (like run_lgbm.py)
  - "global_calendar" : same, + cyclical time-of-day/week/month features
  - "per_hour"        : 48 separate LGBMs, one per half-hour-of-day slot —
                         each occurrence of that slot in the horizon is
                         predicted by ITS OWN model (not shared across hours)
  - "per_hour_calendar": per_hour + cyclical day-of-week/month features

Prediction is recursive (cf. earlier discussion): predicted values are fed
back as lags for the next step. It is BATCHED across all clients at a given
cutoff — one predict() call per horizon step (or per step per slot for
per_hour), not one per client, since all clients share the same timestamp
at a given step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

N_SLOTS_PER_DAY = 48   # 30-min resolution


def half_hour_slot(dt: pd.Timestamp) -> int:
    return dt.hour * 2 + dt.minute // 30


def cyclical_features(dt_index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """sin/cos encoding of hour-of-day (48 slots), day-of-week (7), day-of-month (31)."""
    slot = np.array([half_hour_slot(d) for d in dt_index])
    dow = dt_index.dayofweek.to_numpy()
    dom = (dt_index.day - 1).to_numpy()
    return {
        "hour_sin": np.sin(2 * np.pi * slot / N_SLOTS_PER_DAY),
        "hour_cos": np.cos(2 * np.pi * slot / N_SLOTS_PER_DAY),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "dom_sin": np.sin(2 * np.pi * dom / 31),
        "dom_cos": np.cos(2 * np.pi * dom / 31),
    }


# --------------------------------------------------------------------------- #
# Training table construction
# --------------------------------------------------------------------------- #
def build_training_table(
    ts, client_indices: list[int], t_min: int, t_max: int,
    lags: list[int], n_samples: int, seed: int, include_calendar: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Randomly samples (client, t) pairs with t in [t_min, t_max) and builds
    one row per sample: lag features (+ optional calendar features) -> target
    value at t. Also returns the half-hour slot of each target (for the
    per_hour variant's grouping).
    """
    rng = np.random.default_rng(seed)
    max_lag = max(lags)
    valid_t = np.arange(max(t_min, max_lag), t_max)
    if len(valid_t) == 0:
        raise ValueError(f"no valid training positions in [{t_min}, {t_max}) for lags up to {max_lag}")

    clients = rng.choice(client_indices, size=n_samples, replace=True)
    times = rng.choice(valid_t, size=n_samples, replace=True)

    lag_cols = {f"lag{l}": ts.values[times - l, clients] for l in lags}
    y = ts.values[times, clients]

    dt_index = pd.DatetimeIndex([ts.datetimes[t] for t in times])
    slots = np.array([half_hour_slot(d) for d in dt_index])

    X = pd.DataFrame(lag_cols)
    if include_calendar:
        for name, values in cyclical_features(dt_index).items():
            X[name] = values

    return X, y, slots


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_global(X: pd.DataFrame, y: np.ndarray, **lgbm_kwargs) -> LGBMRegressor:
    model = LGBMRegressor(**lgbm_kwargs)
    model.fit(X, y)
    return model


def fit_per_hour(X: pd.DataFrame, y: np.ndarray, slots: np.ndarray, **lgbm_kwargs) -> dict[int, LGBMRegressor]:
    models: dict[int, LGBMRegressor] = {}
    for s in range(N_SLOTS_PER_DAY):
        mask = slots == s
        if mask.sum() < 20:      # not enough samples for this slot — skip, will fall back at predict time
            continue
        m = LGBMRegressor(**lgbm_kwargs)
        m.fit(X[mask], y[mask])
        models[s] = m
    return models


# --------------------------------------------------------------------------- #
# Recursive prediction, BATCHED across all clients for a given cutoff
# --------------------------------------------------------------------------- #
def predict_recursive(
    *,
    context: np.ndarray,          # (n_clients, ctx_len) — raw history up to the cutoff
    future_dates: pd.DatetimeIndex,  # (H,) — the timestamps to predict, in order
    lags: list[int],
    horizon: int,
    variant: str,                 # "global" | "global_calendar" | "per_hour" | "per_hour_calendar"
    model,                        # LGBMRegressor (global*) or dict[int, LGBMRegressor] (per_hour*)
    fallback_model=None,          # used by per_hour* for slots with no dedicated model
) -> np.ndarray:
    """Returns (n_clients, H) point forecasts."""
    n_clients = context.shape[0]
    max_lag = max(lags)
    include_calendar = variant.endswith("calendar")
    per_hour = variant.startswith("per_hour")

    # rolling buffer: last max_lag known values per client, grown by 1 each step
    buffer = context[:, -max_lag:].copy()          # (n_clients, max_lag)
    preds = np.empty((n_clients, horizon), dtype=np.float32)

    for j in range(horizon):
        lag_feats = {f"lag{l}": buffer[:, -l] for l in lags}
        X = pd.DataFrame(lag_feats)

        if include_calendar:
            dt = pd.DatetimeIndex([future_dates[j]] * n_clients)
            for name, values in cyclical_features(dt).items():
                X[name] = values

        if per_hour:
            slot = half_hour_slot(future_dates[j])
            m = model.get(slot, fallback_model)
        else:
            m = model

        step_pred = m.predict(X)                    # (n_clients,) — ONE call for all clients
        preds[:, j] = step_pred
        buffer = np.concatenate([buffer, step_pred[:, None]], axis=1)   # grow buffer by 1

    return preds