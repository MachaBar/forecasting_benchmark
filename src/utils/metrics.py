"""Reusable forecasting metrics, model-family agnostic.

Designed to be called identically for statistical, ML, DL, and TSFM models
— each metric takes plain numpy arrays, not anything specific to a given
model's output format (cf. the adapter layer in dataset_general.py, which
handles the OTHER direction: converting a common window into each model's
native input format).

Point metrics (MAE, RMSE, MASE, MAPE) expect a single point forecast.
Probabilistic metrics (CRPS, WQL) expect either a dict of quantile
forecasts or Monte Carlo samples.

All functions operate on 1-D arrays (one forecast horizon at a time, for
one client at one cutoff).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
# def _maybe_normalize(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     context: np.ndarray | None,
#     eps: float = 1e-3,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """If `context` is given, rescales y_true/y_pred by the context's own
#     mean/std (instance normalization) before the metric is computed — the
#     statistics come ONLY from the context, never from y_true, to avoid
#     leakage. Returns the inputs unchanged if context is None."""
#     if context is None:
#         return y_true, y_pred
#     context = np.asarray(context, dtype=float)
#     mean = context.mean()
#     scale = max(float(context.std()), eps)
#     return (y_true - mean) / scale, (y_pred - mean) / scale

def _maybe_normalize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context: np.ndarray | None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Instance-normalization by the context std (→ NMAE / NRMSE when the
    caller then takes MAE / RMSE). If the context is (near-)constant, the
    normalization is undefined — we return NaN so that window is *excluded*
    from the average rather than exploding via an artificial floor."""
    if context is None:
        return y_true, y_pred
    context = np.asarray(context, dtype=float)
    scale = float(context.std())
    if scale < eps:                                    # constant context → undefined
        nan = np.full(np.shape(y_true), np.nan, dtype=float)
        return nan, nan
    mean = context.mean()
    return (y_true - mean) / scale, (y_pred - mean) / scale


def mae(y_true: np.ndarray, y_pred: np.ndarray, *, context: np.ndarray | None = None) -> float:
    """Mean Absolute Error. Pass `context` (the input window, raw) to get
    the instance-normalized variant instead of the raw-scale one — useful
    to compare clients of very different magnitudes (cf. the discussion on
    papers normalizing by mean/std before comparing demand of different
    house sizes)."""
    y_true_, y_pred_ = _maybe_normalize(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), context)
    return float(np.mean(np.abs(y_true_ - y_pred_)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray, *, context: np.ndarray | None = None) -> float:
    """Root Mean Squared Error. Same `context` toggle as mae()."""
    y_true_, y_pred_ = _maybe_normalize(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), context)
    return float(np.sqrt(np.mean((y_true_ - y_pred_) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-3) -> float:
    """Mean Absolute Percentage Error, in %.

    CAUTION: explodes or becomes meaningless when y_true is near zero —
    common for load curves at low-consumption periods (e.g. night-time for
    a residential client). `eps` floors the denominator to avoid a
    division by ~0, but the resulting number can still be dominated by a
    few near-zero points; consider MASE instead if this is a concern for
    your data. No `context` normalization option here (deliberately) — a
    percentage error computed on z-scored values isn't interpretable the
    same way, since "near zero" becomes the typical case post-normalization
    rather than the exception.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, history: np.ndarray, season_length: int) -> float:
    """Mean Absolute Scaled Error: the forecast error scaled by THIS
    series' own in-sample seasonal-naive error (cf. the GIFT-Eval /
    fev-bench / Chronos-2 discussion) — scale-free across clients without
    needing any external normalization. `history` is the raw series up to
    (and not including) the forecast window; no context normalization
    option needed, MASE already is one."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    history = np.asarray(history, dtype=float)
    if len(history) <= season_length:
        return float("nan")
    naive_errors = np.abs(history[season_length:] - history[:-season_length])
    scale = max(float(naive_errors.mean()), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# --------------------------------------------------------------------------- #
# Probabilistic metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, q: float) -> np.ndarray:
    """Pinball (quantile) loss for a single quantile level q in (0, 1).
    Returns the per-point loss array — most callers want wql()/crps_from_quantiles()
    instead of calling this directly."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_quantile = np.asarray(y_pred_quantile, dtype=float)
    diff = y_true - y_pred_quantile
    return np.maximum(q * diff, (q - 1.0) * diff)


def wql(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """Weighted Quantile Loss: pinball loss averaged over the given
    quantile levels, each scaled by sum(|y_true|) — scale-free across
    series, same convention used by GIFT-Eval / fev-bench / the Chronos-2
    paper for probabilistic accuracy. quantile_preds: {quantile_level: predicted_values}.
    """
    y_true = np.asarray(y_true, dtype=float)
    total_abs_y = float(np.sum(np.abs(y_true)))
    if total_abs_y < 1e-8 or not quantile_preds:
        return float("nan")
    per_quantile = [
        2.0 * float(np.sum(pinball_loss(y_true, y_pred_q, q))) / total_abs_y
        for q, y_pred_q in quantile_preds.items()
    ]
    return float(np.mean(per_quantile))


def crps_from_quantiles(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """Approximates CRPS via a Riemann-sum average of pinball losses over
    the provided quantile levels (more levels = better approximation).

    Unlike wql(), this is NOT scaled by sum(|y_true|) — it stays in the
    original units (e.g. Watts), so it's comparable across forecasts for
    the SAME series/scale but not directly across clients of very
    different magnitudes (same raw-vs-scale-free distinction as mae/rmse
    vs mase). Use wql() instead for cross-client comparisons.
    """
    y_true = np.asarray(y_true, dtype=float)
    if not quantile_preds:
        return float("nan")
    per_quantile = [float(np.mean(pinball_loss(y_true, y_pred_q, q))) for q, y_pred_q in quantile_preds.items()]
    return float(2.0 * np.mean(per_quantile))


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Sample-based CRPS (energy-distance form), for models that output
    Monte Carlo samples rather than discrete quantiles.

    samples shape: (n_samples, *y_true.shape). The pairwise term is
    O(n_samples^2) in time/memory — fine for typical sample counts
    (tens to a couple hundred), but avoid calling this with thousands of
    samples without subsampling first.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    term1 = float(np.mean(np.abs(samples - y_true[None, ...])))
    term2 = float(np.mean(np.abs(samples[:, None, ...] - samples[None, :, ...])))
    return term1 - 0.5 * term2

def sql(
    y_true: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
    history: np.ndarray,
    season_length: int,
) -> float:
    """Scaled Quantile Loss — the probabilistic analogue of MASE (SQL, as
    used e.g. in [cite the paper]). Normalizes the quantile (pinball) loss
    by the series' own in-sample seasonal-naive error, exactly like MASE
    does for point forecasts — scale-free PER SERIES, unlike WQL which
    normalizes by the batch's total |y|.
    """
    y_true = np.asarray(y_true, dtype=float)
    history = np.asarray(history, dtype=float)
    if len(history) <= season_length or not quantile_preds:
        return float("nan")
    naive_errors = np.abs(history[season_length:] - history[:-season_length])
    a = max(float(naive_errors.mean()), 1e-8)          # a_{n,d}, comme dans MASE

    per_quantile = [
        2.0 * float(np.mean(pinball_loss(y_true, y_pred_q, q)))
        for q, y_pred_q in quantile_preds.items()
    ]
    return float(np.mean(per_quantile)) / a 

def _kmeans_1d(x: np.ndarray, seeds, n_iter: int = 10):
    """1-D k-means (k=3) seeded by given centroids, min-Euclidean assignment."""
    c = np.asarray(seeds, dtype=float)
    for _ in range(n_iter):
        assign = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        newc = c.copy()
        for k in range(len(c)):
            if np.any(assign == k):
                newc[k] = x[assign == k].mean()
        if np.allclose(newc, c):
            break
        c = newc
    return assign, c


def empq(y_true: np.ndarray, y_pred: np.ndarray, *, zero_frac: float = 0.01, eps: float = 1e-6) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_pred - y_true
    n = len(diff)

    level = float(np.mean(np.abs(y_true)))
    zero_thr = max(zero_frac * level, eps)          # ← plancher absolu ajouté
    out: dict[str, float] = {}

    for sign, tag in ((1, "over"), (-1, "under")):
        mask = diff > 0 if sign == 1 else diff < 0
        grp = diff[mask]
        grp_true = y_true[mask]

        out[f"empq_{tag}_pct"] = float(mask.sum() / n * 100) if n else float("nan")

        valid = np.abs(grp_true) >= zero_thr            # maintenant jamais 0
        if valid.sum() >= 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                pdev = grp[valid] / grp_true[valid] * 100.0
            pdev = pdev[np.isfinite(pdev)]               # filet de sécurité supplémentaire
            if len(pdev) >= 1:
                out[f"empq_{tag}_mindev_pct"] = float(np.percentile(pdev, 25))
                out[f"empq_{tag}_maxdev_pct"] = float(np.percentile(pdev, 75))
            else:
                out[f"empq_{tag}_mindev_pct"] = float("nan")
                out[f"empq_{tag}_maxdev_pct"] = float("nan")
        else:
            out[f"empq_{tag}_mindev_pct"] = float("nan")
            out[f"empq_{tag}_maxdev_pct"] = float("nan")

        if len(grp) >= 3:
            assign, cent = _kmeans_1d(grp, seeds=[grp.min(), grp.mean(), grp.max()])
            order = np.argsort(np.abs(cent))
            remap = np.empty(len(cent), dtype=int)
            remap[order] = np.arange(len(cent))
            lab = remap[assign]
            out[f"empq_{tag}_near_pct"]  = float((lab == 0).mean() * 100)
            out[f"empq_{tag}_inter_pct"] = float((lab == 1).mean() * 100)
            out[f"empq_{tag}_well_pct"]  = float((lab == 2).mean() * 100)
        else:
            out[f"empq_{tag}_near_pct"]  = float("nan")
            out[f"empq_{tag}_inter_pct"] = float("nan")
            out[f"empq_{tag}_well_pct"]  = float("nan")

    return out


# --------------------------------------------------------------------------- #
# Convenience bundle
# --------------------------------------------------------------------------- #
@dataclass
class MetricBundle:
    """Result of compute_metrics() — fields are None when the inputs
    needed for that metric weren't provided."""

    mae: float
    rmse: float
    mae_normalized: float | None = None
    rmse_normalized: float | None = None
    mape: float | None = None
    mase: float | None = None
    wql: float | None = None
    sql: float | None = None
    crps: float | None = None
    empq: dict[str, float] | None = None

    def to_dict(self) -> dict[str, float | None]:
        d = asdict(self)
        empq_dict = d.pop("empq")                  
        if empq_dict:
            d.update(empq_dict)
        return d


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    context: np.ndarray | None = None,
    history: np.ndarray | None = None,
    season_length: int | None = None,
    include_mape: bool = False,
    include_empq=True,
    quantile_preds: dict[float, np.ndarray] | None = None,
    samples: np.ndarray | None = None,
) -> MetricBundle:
    """One call, every metric you have the inputs for. Pass only what's
    available for a given model/window — e.g. a statistical model with no
    probabilistic output simply omits quantile_preds/samples and gets
    None back for wql/crps.
    """
    bundle = MetricBundle(mae=mae(y_true, y_pred), rmse=rmse(y_true, y_pred))

    if context is not None:
        bundle.mae_normalized = mae(y_true, y_pred, context=context)
        bundle.rmse_normalized = rmse(y_true, y_pred, context=context)

    if include_mape:
        bundle.mape = mape(y_true, y_pred)

    if include_empq:
        bundle.empq = empq(y_true, y_pred)

    if history is not None and season_length is not None:
        bundle.mase = mase(y_true, y_pred, history, season_length)

    if quantile_preds:
        bundle.wql = wql(y_true, quantile_preds)
        bundle.crps = crps_from_quantiles(y_true, quantile_preds)
        if history is not None and season_length is not None:
            bundle.sql = sql(y_true, quantile_preds, history, season_length)
    elif samples is not None:
        bundle.crps = crps_from_samples(y_true, samples)

    return bundle
