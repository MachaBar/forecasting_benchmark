"""Shared feature engineering, fitting and recursive prediction for the LGBM
variants. Two strategies, selected via `variant`:

  - "global"          : one LGBM shared by all clients/hours (like run_lgbm.py)
  - "global_calendar" : same, + cyclical time-of-day/week/month features
  - "per_hour"        : 48 separate LGBMs, one per half-hour-of-day slot —
                         each occurrence of that slot in the horizon is
                         predicted by ITS OWN model (not shared across hours)
  - "per_hour_calendar": per_hour + cyclical day-of-week/month features
  - "direct"          : one LGBM PER HORIZON POSITION (L+1, L+2, ..., L+H) —
                         each model predicts directly from the real context,
                         never from its own past predictions (no recursion,
                         no error accumulation). Calendar-of-target features
                         (if enabled) let the model distinguish "predicting
                         at 08:00" from "predicting at 20:00" without tying
                         the model identity to a fixed hour-of-day slot —
                         unlike per_hour, this generalizes across cutoffs
                         that start at different times of day.
  - "direct_calendar" : direct + cyclical day-of-week/month features of the
                         TARGET timestamp (the point being predicted, not
                         the cutoff).

Prediction is recursive (cf. earlier discussion): predicted values are fed
back as lags for the next step. It is BATCHED across all clients at a given
cutoff — one predict() call per horizon step (or per step per slot for
per_hour), not one per client, since all clients share the same timestamp
at a given step.
"global"/"per_hour" use RECURSIVE prediction (cf. earlier discussion):
predicted values are fed back as lags for the next step. It is BATCHED
across all clients at a given cutoff — one predict() call per horizon step
(or per step per slot for per_hour), not one per client, since all clients
share the same timestamp at a given step.

"direct" trains H independent models (H = prediction_length) — H times more
total training rows than "global", so fit_time scales roughly linearly with
H. Use with care on long horizons (e.g. H=336 means 336 LGBM fits)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import lightgbm as lgb

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
# def fit_global(X: pd.DataFrame, y: np.ndarray, **lgbm_kwargs) -> LGBMRegressor:
#     model = LGBMRegressor(**lgbm_kwargs)
#     model.fit(X, y)
#     return model


# def fit_per_hour(X: pd.DataFrame, y: np.ndarray, slots: np.ndarray, **lgbm_kwargs) -> dict[int, LGBMRegressor]:
#     models: dict[int, LGBMRegressor] = {}
#     for s in range(N_SLOTS_PER_DAY):
#         mask = slots == s
#         if mask.sum() < 20:      # not enough samples for this slot — skip, will fall back at predict time
#             continue
#         m = LGBMRegressor(**lgbm_kwargs)
#         m.fit(X[mask], y[mask])
#         models[s] = m
#     return models

def fit_global(
    X: pd.DataFrame, y: np.ndarray, *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    **lgbm_kwargs,
) -> LGBMRegressor:
    model = LGBMRegressor(**lgbm_kwargs)
    if early_stopping_rounds and X_val is not None and len(X_val) > 0:
        model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    else:
        model.fit(X, y)
    return model


def fit_per_hour(
    X: pd.DataFrame, y: np.ndarray, slots: np.ndarray, *,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    slots_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
    **lgbm_kwargs,
) -> dict[int, LGBMRegressor]:
    models: dict[int, LGBMRegressor] = {}
    for s in range(N_SLOTS_PER_DAY):
        mask = slots == s
        if mask.sum() < 20:
            continue
        Xv = yv = None
        if early_stopping_rounds and X_val is not None:
            mv = slots_val == s
            if mv.sum() >= 20:          # sinon: pas d'ES pour ce slot, on garde n_estimators
                Xv, yv = X_val[mv], y_val[mv]
        models[s] = fit_global(
            X[mask], y[mask], X_val=Xv, y_val=yv,
            early_stopping_rounds=early_stopping_rounds, **lgbm_kwargs,
        )
    return models


def best_iterations(model) -> dict:
    """Nombre d'arbres réellement retenus (best_iteration_, ou n_estimators sans ES)."""
    if isinstance(model, dict):                       # per_hour
        per_slot = {int(s): int(m.best_iteration_ or m.n_estimators) for s, m in model.items()}
        vals = list(per_slot.values()) or [0]
        return {"per_slot": per_slot, "mean": float(np.mean(vals)),
                "min": int(min(vals)), "max": int(max(vals))}
    it = int(model.best_iteration_ or model.n_estimators)
    return {"per_slot": None, "mean": float(it), "min": it, "max": it}



# Direct multi-horizon: one model per horizon position, no
# recursion — every model sees only the REAL context, never a predicted
# value. Calendar features (if enabled) describe the TARGET timestamp, not
# the cutoff, so the model learns hour-of-day/day-of-week/day-of-month as
# ordinary inputs instead of using them as a model-selection key.
# --------------------------------------------------------------------------- #
def build_training_table_direct(
    ts, client_indices: list[int], t_min: int, t_max: int,
    lags: list[int], horizon: int, n_samples: int, seed: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """Samples (client, r) pairs, r = the cutoff reference position, and
    builds lag features anchored to r (never to the predicted point).
    Returns:
      X          : (n_samples, len(lags)) — lag features, identical
                   formula/semantics as predict_direct's context lags.
      Y          : (n_samples, horizon) — Y[:, i] is the target for
                   model i (i.e. the value i+1 steps after r).
      dt_all     : DatetimeIndex of ALL of ts (for slicing per-horizon
                   target timestamps with dt_all[r + i + 1]).
    """
    rng = np.random.default_rng(seed)
    max_lag = max(lags)
    valid_r = np.arange(max(t_min, max_lag - 1), t_max - horizon)
    if len(valid_r) == 0:
        raise ValueError(
            f"no valid cutoff positions in [{t_min}, {t_max - horizon}) "
            f"for lags up to {max_lag} and horizon={horizon}"
        )

    clients = rng.choice(client_indices, size=n_samples, replace=True)
    r = rng.choice(valid_r, size=n_samples, replace=True)

    lag_cols = {f"lag{l}": ts.values[r - l + 1, clients] for l in lags}
    X = pd.DataFrame(lag_cols)

    Y = np.stack(
        [ts.values[r + i + 1, clients] for i in range(horizon)], axis=1
    ).astype(np.float32)

    dt_all = pd.DatetimeIndex(ts.datetimes)
    return X, Y, dt_all, r


def fit_direct(
    X: pd.DataFrame, Y: np.ndarray, dt_all: pd.DatetimeIndex, r: np.ndarray,
    horizon: int, include_calendar: bool, **lgbm_kwargs,
) -> dict[int, LGBMRegressor]:
    """One LGBM per horizon position. Y[:, i] is the target `horizon`
    position i+1 (1-indexed step count after the cutoff r)."""
    models: dict[int, LGBMRegressor] = {}
    for i in range(horizon):
        Xi = X
        if include_calendar:
            Xi = X.copy()
            target_dt = dt_all[r + i + 1]
            for name, values in cyclical_features(target_dt).items():
                Xi[name] = values
        m = LGBMRegressor(**lgbm_kwargs)
        m.fit(Xi, Y[:, i])
        models[i] = m
    return models


def predict_direct(
    *,
    context: np.ndarray,             # (n_clients, ctx_len) — raw history up to the cutoff
    future_dates: pd.DatetimeIndex,  # (H,) — the timestamps to predict, in order
    lags: list[int],
    horizon: int,
    variant: str,                    # "direct" | "direct_calendar"
    models: dict[int, LGBMRegressor],
) -> np.ndarray:
    """Returns (n_clients, H) point forecasts. No recursion: every model[j]
    predicts directly from the real `context`, using calendar features of
    future_dates[j] (the actual target time) rather than the cutoff time."""
    n_clients = context.shape[0]
    include_calendar = variant.endswith("calendar")

    lag_feats = {f"lag{l}": context[:, -l] for l in lags}
    X = pd.DataFrame(lag_feats)

    preds = np.empty((n_clients, horizon), dtype=np.float32)
    for j in range(horizon):
        Xj = X
        if include_calendar:
            Xj = X.copy()
            dt = pd.DatetimeIndex([future_dates[j]] * n_clients)
            for name, values in cyclical_features(dt).items():
                Xj[name] = values
        preds[:, j] = models[j].predict(Xj)
    return preds


# --------------------------------------------------------------------------- #

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