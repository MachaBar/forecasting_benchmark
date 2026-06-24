"""Data pipeline for multi-client forecasting (csv / parquet / npy).

Principles established through discussion:
  - returned data is ALWAYS raw (never normalized here); normalization,
    if a model needs it, is the responsibility of its own adapter
    (cf. PatchTST + RevIN, which already does it internally);
  - the train/val/test split is computed once, by date, identically for
    every model family (stats, ML, DL, TSFM);
  - training samples random windows (client, position), at two levels,
    so that a client with a long history isn't over-represented;
  - evaluation uses fixed, rolling cutoffs, identical for every model
    being compared.
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# 1. Raw data container + windowing
# --------------------------------------------------------------------------- #
@dataclass
class TimeSeriesFrame:
    """Wide-format data (date x client) + optional global covariates."""

    frame: pd.DataFrame
    past_covariates: pd.DataFrame | None = None
    future_covariates: pd.DataFrame | None = None

    @property
    def values(self) -> np.ndarray:
        return self.frame.to_numpy(dtype=np.float32)

    @property
    def datetimes(self) -> list[Any]:
        return list(self.frame.index)

    @property
    def user_names(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def n_dates(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.frame.shape[1])

    def validate_window(self, start: int, lags: int, horizon: int) -> None:
        stop = int(start) + int(lags) + int(horizon)
        if start < 0 or stop > self.n_dates:
            raise ValueError(
                f"window [{start}, {stop}) is outside the dataset bounds "
                f"({self.n_dates} dates) — refusing rather than silently extrapolating."
            )

    def window_tensor(
        self,
        start: int,
        lags: int,
        horizon: int,
        *,
        users: Sequence[int] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) RAW, shape (n_users, 1, lags) and (n_users, 1, horizon)."""
        self.validate_window(start, lags, horizon)
        values = self.values[start : start + lags + horizon]
        if users is not None:
            values = values[:, list(users)]
        arr = torch.as_tensor(
            rearrange(values, "time user -> user time").copy(),
            dtype=torch.float32,
            device=device,
        )
        return arr[:, None, :lags], arr[:, None, lags:]

    def covariate_tensors(
        self,
        start: int,
        lags: int,
        horizon: int,
        *,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Global covariates (shared across all clients), shape (1, C, time)."""
        self.validate_window(start, lags, horizon)
        past = future = None
        if self.past_covariates is not None:
            vals = self.past_covariates.iloc[start : start + lags].to_numpy(dtype=np.float32)
            past = torch.as_tensor(
                rearrange(vals, "time channel -> 1 channel time").copy(),
                dtype=torch.float32,
                device=device,
            )
        if self.future_covariates is not None:
            vals = self.future_covariates.iloc[
                start + lags : start + lags + horizon
            ].to_numpy(dtype=np.float32)
            future = torch.as_tensor(
                rearrange(vals, "time channel -> 1 channel time").copy(),
                dtype=torch.float32,
                device=device,
            )
        return past, future


# --------------------------------------------------------------------------- #
# 2. Multi-format loading (csv / parquet / npy) -> TimeSeriesFrame
# --------------------------------------------------------------------------- #
def _read_raw_frame(
    path: str | Path,
    *,
    date_col: str | None,
    npy_freq_minutes: int,
    npy_start_date: datetime.datetime | None,
) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        if date_col:
            raw = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)
        else:
            raw = pd.read_csv(path, index_col=0)
            raw.index = pd.to_datetime(raw.index)
        return raw

    if ext == ".parquet":
        raw = pd.read_parquet(path)
        if date_col:
            raw = raw.set_index(date_col)
        else:
            raw.index = pd.to_datetime(raw.index)
        return raw

    if ext == ".npy":
        data = np.load(path)  # (N_ids, T) or (N_ids, D, H)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        if data.ndim != 2:
            raise ValueError(f"Expected shape (N_ids, T), got {data.shape}")
        data = data.T  # -> (T, N)
        start = npy_start_date or datetime.datetime(2021, 1, 1)
        dates = pd.date_range(start=start, periods=data.shape[0], freq=f"{npy_freq_minutes}min")
        cols = [f"user_{i}" for i in range(data.shape[1])]
        return pd.DataFrame(data, index=dates, columns=cols)

    raise ValueError(f"Unsupported format: {ext!r} (expected: .csv, .parquet, .npy)")


def _read_long_frame(
    path: str | Path,
    *,
    id_col: str,
    ds_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Reads a LONG file (unique_id, ds, y, [covariates...]) and pivots it
    to wide (date x client + covariate columns), in exactly the same shape
    that _read_raw_frame returns for an already-wide file — so the rest of
    load_dataset doesn't need to know anything about the original layout.

    Any covariate columns present among the remaining columns must be
    GLOBAL (the same value for every unique_id on a given date): those are
    the only kind TimeSeriesFrame can represent (cf. the earlier discussion
    on per-client covariates, which this design does not support). In case
    of duplicates per date, the first value encountered is kept.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        raw_long = pd.read_csv(path, parse_dates=[ds_col])
    elif ext == ".parquet":
        raw_long = pd.read_parquet(path)
        if not np.issubdtype(raw_long[ds_col].dtype, np.datetime64):
            raw_long[ds_col] = pd.to_datetime(raw_long[ds_col])
    else:
        raise ValueError("layout='long' is only supported for .csv and .parquet")

    target_wide = raw_long.pivot(index=ds_col, columns=id_col, values=value_col)
    target_wide.columns = [str(c) for c in target_wide.columns]
    target_wide = target_wide.sort_index()

    other_cols = [c for c in raw_long.columns if c not in (id_col, ds_col, value_col)]
    if not other_cols:
        return target_wide

    covariates = raw_long[[ds_col] + other_cols].groupby(ds_col).first().sort_index()
    return target_wide.join(covariates, how="left")


def _select_columns(df: pd.DataFrame, cols: Sequence[Any] | str | None) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",") if c.strip()]
    return [str(c) for c in cols if str(c) in df.columns]


def load_dataset(
    path: str | Path,
    *,
    layout: str = "wide",
    id_col: str | None = None,
    value_col: str | None = None,
    target_cols: Sequence[Any] | str | None = None,
    past_covariate_cols: Sequence[Any] | str | None = None,
    future_covariate_cols: Sequence[Any] | str | None = None,
    date_col: str | None = None,
    drop_users: Sequence[Any] | str | None = None,
    resample_freq: str | None = None,
    resample_agg: str = "mean",
    npy_freq_minutes: int = 30,
    npy_start_date: datetime.datetime | None = None,
) -> TimeSeriesFrame:
    """Loads a dataset (date x client) from .csv, .parquet, or .npy.

    layout="wide" (default): one column per client, as before.
    layout="long": columns (id_col, date_col, value_col, [covariates...]),
    one row per (client, date) — pivoted internally and automatically; the
    rest of the processing (resample, dropna, covariate/target selection)
    is identical for both layouts.
    """
    if layout == "long":
        if id_col is None or value_col is None:
            raise ValueError("layout='long' requires id_col and value_col")
        raw = _read_long_frame(path, id_col=id_col, ds_col=date_col or "ds", value_col=value_col)
    elif layout == "wide":
        raw = _read_raw_frame(
            path,
            date_col=date_col,
            npy_freq_minutes=npy_freq_minutes,
            npy_start_date=npy_start_date,
        )
    else:
        raise ValueError(f"layout must be 'wide' or 'long', got {layout!r}")

    if resample_freq:
        raw = raw.resample(resample_freq).agg(resample_agg)

    # Careful: drops the entire DATE row as soon as a single client has a
    # NaN on that day. With many clients, this can be expensive in usable
    # history — consider per-series imputation upstream if this matters.
    raw = raw.dropna(axis=0, how="any")

    past_cols = _select_columns(raw, past_covariate_cols)
    future_cols = _select_columns(raw, future_covariate_cols)
    cov_cols = set(past_cols + future_cols)

    value_cols = (
        [c for c in raw.columns if c not in cov_cols]
        if target_cols is None
        else _select_columns(raw, target_cols)
    )
    if drop_users:
        drop_set = set(_select_columns(raw, drop_users))
        value_cols = [c for c in value_cols if c not in drop_set]

    values = raw[value_cols].copy()
    if values.empty:
        raise ValueError("no target column left after filtering")

    past = raw[past_cols].copy() if past_cols else None
    future = raw[future_cols].copy() if future_cols else None
    return TimeSeriesFrame(values, past_covariates=past, future_covariates=future)


# --------------------------------------------------------------------------- #
# 3. Train cutoffs (random positions) / val+test cutoffs (fixed, rolling)
# --------------------------------------------------------------------------- #
def parse_ratios(value: str | Sequence[float]) -> tuple[float, float, float]:
    """Parses "0.7,0.15,0.15" or [0.7, 0.15, 0.15] -> (r_train, r_val, r_test)."""
    ratios = (
        [float(p) for p in value.split(",")] if isinstance(value, str) else [float(p) for p in value]
    )
    if len(ratios) != 3:
        raise ValueError("exactly 3 ratios are required: train,val,test")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    return ratios[0], ratios[1], ratios[2]


def _rolling_cutoffs_in_range(
    start: int, end: int, *, lags: int, horizon: int, step_size: int
) -> np.ndarray:
    """Cutoffs (context start positions) within [start, end), spaced by
    step_size, such that each context+horizon window stays inside [start, end)."""
    last = end - lags - horizon
    if last < start:
        raise ValueError(
            f"range [{start}, {end}) ({end - start} points) is too small "
            f"for lags={lags} + horizon={horizon}"
        )
    n_windows = (last - start) // step_size + 1
    return np.array([start + i * step_size for i in range(n_windows)])


def make_cutoffs(
    ts: TimeSeriesFrame,
    *,
    lags: int,
    horizon: int,
    n_val_windows: int = 6,
    n_test_windows: int = 6,
    step_size: int | None = None,
    ratios: str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Computes the valid training positions + the val/test cutoffs.

    Two modes, your choice:
      - ratios=None (default): n_val_windows/n_test_windows fixed windows
        counted backward from the end of the series — the val+test size is
        absolute, independent of the total series length;
      - ratios="0.7,0.15,0.15" (or [0.7, 0.15, 0.15]): train/val/test
        boundaries as a % of ts.n_dates (GIFT-Eval style), with as many
        rolling windows as fit inside each val/test block.

    In both cases, the val/test cutoffs are fixed and identical for every
    model being compared (stats, ML, DL, TSFM), and a margin of `horizon`
    separates the last valid training position from the first validation
    cutoff so that no train target overlaps with val.
    """
    step_size = step_size or horizon

    if ratios is not None:
        r_train, r_val, _ = parse_ratios(ratios)
        train_end = int(round(r_train * ts.n_dates))
        val_end = int(round((r_train + r_val) * ts.n_dates))
        test_end = ts.n_dates

        val_cutoffs = _rolling_cutoffs_in_range(
            train_end, val_end, lags=lags, horizon=horizon, step_size=step_size
        )
        test_cutoffs = _rolling_cutoffs_in_range(
            val_end, test_end, lags=lags, horizon=horizon, step_size=step_size
        )
    else:
        last_cutoff = ts.n_dates - lags - horizon  # last valid "start" position
        test_cutoffs = np.array(
            [last_cutoff - i * step_size for i in range(n_test_windows)][::-1]
        )
        first_test = int(test_cutoffs[0])
        val_cutoffs = np.array(
            [first_test - step_size - i * step_size for i in range(n_val_windows)][::-1]
        )

    first_val = int(val_cutoffs[0])
    train_upper_bound = first_val - horizon  # train target must end before the val target
    train_positions = np.arange(lags, train_upper_bound)

    if train_positions.size == 0:
        raise ValueError(
            "not enough data to generate valid training positions with these "
            "parameters (lags/horizon/ratios/n_val_windows/n_test_windows "
            "are incompatible with the length of the series)"
        )

    return {
        "train_positions": train_positions,  # for random sampling during training
        "val_cutoffs": val_cutoffs,           # fixed, same for every model
        "test_cutoffs": test_cutoffs,         # fixed, same for every model
    }


# --------------------------------------------------------------------------- #
# 3bis. Additional split by CLIENT (in addition to the temporal split)
#
# Only relevant for global models (ML, DL, TSFM) that share parameters
# across clients: some clients are entirely excluded from fitting to test
# generalization to a client never seen before. Not applicable to
# statistical models (fit per series, cf. earlier discussion in this
# conversation).
# --------------------------------------------------------------------------- #
def make_client_split(
    ts: TimeSeriesFrame,
    *,
    holdout_fraction: float = 0.2,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Splits the n_users clients into two groups:
      - seen_clients: used for training (via TrainWindowDataset);
      - held_out_clients: never seen during fitting, evaluated separately
        (at the same val_cutoffs/test_cutoffs) to measure generalization to
        an unknown client — distinct from the plain "generalization through
        time" that val_cutoffs/test_cutoffs already measure on seen_clients.
    """
    rng = random.Random(seed)
    all_idx = list(range(ts.n_users))
    rng.shuffle(all_idx)
    n_holdout = int(round(holdout_fraction * ts.n_users))
    held_out_clients = sorted(all_idx[:n_holdout])
    seen_clients = sorted(all_idx[n_holdout:])
    return {"seen_clients": seen_clients, "held_out_clients": held_out_clients}


# --------------------------------------------------------------------------- #
# 4. Training dataset: random sampling (client, position)
# --------------------------------------------------------------------------- #
class TrainWindowDataset(Dataset):
    """One sample = one raw window for ONE client, drawn at random.

    Two-level sampling (client, then position) so that a client with a
    long history isn't over-represented relative to the others.
    """

    def __init__(
        self,
        ts: TimeSeriesFrame,
        *,
        lags: int,
        horizon: int,
        train_positions: np.ndarray,
        n_samples: int,
        client_pool: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> None:
        self.ts = ts
        self.lags = lags
        self.horizon = horizon
        self.train_positions = train_positions
        self.n_samples = n_samples
        # By default, samples among all clients; pass client_pool=
        # seen_clients (cf. make_client_split) to fully exclude held-out
        # clients from training.
        self.client_pool = list(client_pool) if client_pool is not None else list(range(ts.n_users))
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        client_idx = self._rng.choice(self.client_pool)
        start = int(self._rng.choice(self.train_positions))

        x, y = self.ts.window_tensor(start, self.lags, self.horizon, users=[client_idx])
        past_cov, future_cov = self.ts.covariate_tensors(start, self.lags, self.horizon)

        return {
            "x": x[0],  # (1, lags), raw
            "y": y[0],  # (1, horizon), raw
            # NB: if past_covariates/future_covariates are None, PyTorch's
            # default collate_fn will fail when batching None values —
            # provide a custom collate_fn if you use covariates.
            "past_covariates": past_cov[0] if past_cov is not None else None,
            "future_covariates": future_cov[0] if future_cov is not None else None,
            "item_id": self.ts.user_names[client_idx],
            "start_x_date": str(self.ts.datetimes[start]),
            "start_y_date": str(self.ts.datetimes[start + self.lags]),
        }


# --------------------------------------------------------------------------- #
# 5. Extraction for evaluation: fixed cutoff, all requested clients at once
# --------------------------------------------------------------------------- #
def eval_batch(
    ts: TimeSeriesFrame,
    cutoff: int,
    *,
    lags: int,
    horizon: int,
    users: Sequence[int] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """One fixed window (same cutoff) for the requested clients (all by
    default) — pass users=seen_clients / users=held_out_clients (cf.
    make_client_split) to separately fill the two cells of the seen/unseen
    grid, at the same cutoffs."""
    x, y = ts.window_tensor(cutoff, lags, horizon, users=users, device=device)
    past_cov, future_cov = ts.covariate_tensors(cutoff, lags, horizon, device=device)
    item_ids = ts.user_names if users is None else [ts.user_names[u] for u in users]
    return {
        "x": x,  # (len(users) or n_users, 1, lags), raw
        "y": y,
        "past_covariates": past_cov,
        "future_covariates": future_cov,
        "item_ids": item_ids,
        "cutoff": cutoff,
        "start_y_date": str(ts.datetimes[cutoff + lags]),
    }


# --------------------------------------------------------------------------- #
# 6. Optional normalization — never applied by default
# --------------------------------------------------------------------------- #
def normalize_window(
    x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score computed on x only (never on y, to avoid leakage).

    Use only for models that don't already have their own internal
    normalization (unnecessary, even redundant, with PatchTST+RevIN,
    Chronos, TabPFN...). For reporting/comparison across model families,
    always denormalize (pred * scale + mean) before computing a metric —
    cf. the discussion on GIFT-Eval / fev-bench and MASE.
    """
    mean = x.mean(dim=-1, keepdim=True)
    scale = x.std(dim=-1, keepdim=True).clamp_min(eps)
    return (x - mean) / scale, (y - mean) / scale, mean, scale


# --------------------------------------------------------------------------- #
# 7. Adapters per model family
#
# The common format (x/y/covariates as raw tensors, see eval_batch and
# TrainWindowDataset) acts as a neutral representation. Each model family
# has a different native format, so we convert right before calling the
# model rather than complicating the dataset itself.
# --------------------------------------------------------------------------- #
def to_patchtst_input(batch: dict[str, Any]) -> torch.Tensor:
    """PatchTST (or any model consuming (batch, channels, lags) directly)
    — our common format already matches this, no conversion needed."""
    return batch["x"]  # (n_users, 1, lags) — already the expected shape


def to_statsforecast_df(ts: TimeSeriesFrame, batch: dict[str, Any]) -> pd.DataFrame:
    """Converts a window (output of eval_batch) into long format
    (unique_id, ds, y) for statsforecast.forecast()."""
    x = batch["x"]  # (n_users, 1, lags)
    cutoff = batch["cutoff"]
    lags = x.shape[-1]
    dates = ts.datetimes[cutoff : cutoff + lags]

    frames = [
        pd.DataFrame({"unique_id": item_id, "ds": dates, "y": x[i, 0].numpy()})
        for i, item_id in enumerate(batch["item_ids"])
    ]
    return pd.concat(frames, ignore_index=True)


def to_statsforecast_history_df(
    ts: TimeSeriesFrame,
    cutoff: int,
    *,
    lags: int,
    users: Sequence[int] | None = None,
    max_lookback: int | None = None,
) -> pd.DataFrame:
    """Variant for STATISTICAL models (Naive, SeasonalNaive, AutoTheta,
    AutoETS, AutoARIMA): unlike to_statsforecast_df, it does NOT limit
    itself by default to the last `lags` points — these models have no
    notion of a bounded context, they want all available history.

    The decision point is aligned on cutoff + lags (the start of the
    target), exactly the same instant used by PatchTST/Chronos/tsicl for
    this same cutoff — so the comparison stays apples-to-apples on WHEN the
    prediction is made, independently of the question below.

    max_lookback controls HOW MUCH history the model sees before that
    instant:
      - None (default): all history since the beginning — the native mode
        of statistical models (cf. the GIFT-Eval discussion: their
        truncation to 1000 points was a compute-budget constraint, not a
        fairness choice);
      - max_lookback=lags: exactly the same window as the DL/ML/TSFM
        models — answers "given equal information, which model predicts
        best?", a different and equally legitimate question. Use this as a
        complement to the default mode, not a replacement for it.
    """
    history_end = cutoff + lags
    history_start = max(0, history_end - max_lookback) if max_lookback is not None else 0
    values = ts.values[history_start:history_end]
    dates = ts.datetimes[history_start:history_end]
    user_idx = list(users) if users is not None else list(range(ts.n_users))

    frames = [
        pd.DataFrame({"unique_id": ts.user_names[u], "ds": dates, "y": values[:, u]})
        for u in user_idx
    ]
    return pd.concat(frames, ignore_index=True)


def to_chronos2_inputs(
    ts: TimeSeriesFrame, batch: dict[str, Any]
) -> np.ndarray | list[dict[str, Any]]:
    """Converts a window into the format expected by
    Chronos2Pipeline.predict_quantiles: a 3D array if there are no
    covariates, otherwise a list of dicts {target, past_covariates, future_covariates}.
    """
    x = batch["x"]
    if x.ndim == 2:
        x = x[None]  # normalize to (n_users, 1, lags)

    past_cov = batch.get("past_covariates")
    future_cov = batch.get("future_covariates")

    if past_cov is None and future_cov is None:
        # Simple format: (batch_size, num_variates, history_length)
        # — already exactly our common shape, no conversion needed.
        return x.numpy()

    # Format with covariates: a list of dicts, one per series. Our
    # covariates are global (shared across clients) in this representation,
    # so they get duplicated into each item of the list.
    past_names = ts.past_covariates.columns.tolist() if ts.past_covariates is not None else []
    future_names = ts.future_covariates.columns.tolist() if ts.future_covariates is not None else []

    items: list[dict[str, Any]] = []
    for i in range(x.shape[0]):
        item: dict[str, Any] = {"target": x[i, 0].numpy()}  # (lags,), univariate
        if past_cov is not None:
            item["past_covariates"] = {
                name: past_cov[0, c].numpy() for c, name in enumerate(past_names)
            }
        if future_cov is not None:
            item["future_covariates"] = {
                name: future_cov[0, c].numpy() for c, name in enumerate(future_names)
            }
        items.append(item)
    return items


def to_tsicl_inputs(
    ts: TimeSeriesFrame, batch: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Converts a window into the tsicl format: context shape [N, L],
    covariates shape [N, L+H, K] (context AND horizon CONCATENATED, unlike
    Chronos-2 which separates them into past/future).

    Important: tsicl has no notion of a "past-only" covariate — only
    covariates known across both the context AND the horizon can be used
    (i.e. present under the same name in both past_covariates and
    future_covariates). Everything else is silently dropped here rather
    than incorrectly concatenated.
    """
    x = batch["x"]  # (n_users, 1, lags)
    context = x[:, 0, :].numpy()  # -> [N, L]

    past_cov = batch.get("past_covariates")
    future_cov = batch.get("future_covariates")
    if past_cov is None or future_cov is None:
        return context, None  # no complete context+horizon covariate set available

    past_names = ts.past_covariates.columns.tolist()
    future_names = ts.future_covariates.columns.tolist()
    common = [n for n in past_names if n in future_names]
    if not common:
        return context, None

    past_idx = [past_names.index(n) for n in common]
    future_idx = [future_names.index(n) for n in common]

    covars = torch.cat([past_cov[0, past_idx], future_cov[0, future_idx]], dim=-1)  # (K, L+H)
    covars = rearrange(covars, "channel time -> time channel").numpy()  # [L+H, K]

    n_users = context.shape[0]
    covars = np.broadcast_to(covars, (n_users, *covars.shape)).copy()  # shared across clients
    return context, covars
