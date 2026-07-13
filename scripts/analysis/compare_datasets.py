# %% [markdown]
# # Multi-dataset comparison
# Computes stats/profiles for several datasets and produces side-by-side
# comparison tables + overlaid figures in outputs/analysis/_comparison/.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import load_dataset

# ---- CONFIG: list the datasets to compare ----
DATASETS = {
    "cer": {
        "path": "/home/d32485/forecasting_benchmark/data/cer/cer/load_curve.parquet",
        "timestamp_col": "time", "layout": "wide",
    },
    "cer_bis": {
        "path": "/home/d32485/forecasting_benchmark/data/cer_bis/load_curve.parquet",
        "timestamp_col": "time", "layout": "wide",
    },
    "smach": {
        "path": "/home/d32485/forecasting_benchmark/data/smach/load_curve.parquet",
        "timestamp_col": "HORODATAGE", "layout": "wide",
    },
}
SEASON_DAILY = 48
# ----------------------------------------------

out_dir = Path("outputs/analysis/_comparison")
out_dir.mkdir(parents=True, exist_ok=True)


# %%
def analyze(name, cfg):
    """Load one dataset and return a dict of stats + profile series."""
    ts = load_dataset(cfg["path"], layout=cfg["layout"], date_col=cfg["timestamp_col"])
    values = ts.values
    dates = pd.DatetimeIndex(ts.datetimes)
    flat = values[~np.isnan(values)]

    mean_series = pd.Series(np.nanmean(values, axis=1), index=dates)  # avg client

    return {
        "name": name,
        # scalar summary
        "summary": {
            "n_clients": values.shape[1],
            "n_timesteps": values.shape[0],
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "duration_days": (dates[-1] - dates[0]).days,
            "freq": str(pd.infer_freq(dates)),
            "pct_missing": float(np.isnan(values).mean() * 100),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "cv": float(np.std(flat) / (np.mean(flat) + 1e-9)),  # coef. of variation
            "skew": float(pd.Series(flat).skew()),
            "kurtosis": float(pd.Series(flat).kurtosis()),
            **{f"q{int(q*100)}": float(np.quantile(flat, q))
               for q in [0.01, 0.05, 0.5, 0.95, 0.99]},
        },
        # profiles (normalized so datasets of different magnitudes are comparable)
        "daily": mean_series.groupby([dates.hour, dates.minute]).mean(),
        "weekly": mean_series.groupby(dates.dayofweek).mean(),
        "monthly": mean_series.groupby(dates.month).mean(),
        "mean_level": float(np.mean(flat)),
    }


results = {name: analyze(name, cfg) for name, cfg in DATASETS.items()}


# %%
# ---- 1. Side-by-side summary table ----
summary_table = pd.DataFrame({r["name"]: r["summary"] for r in results.values()})
summary_table.to_csv(out_dir / "comparison_summary.csv")
print(summary_table)


# %%
# ---- 2. Overlaid daily profiles (normalized by each dataset's mean) ----
fig, ax = plt.subplots(figsize=(12, 4))
for r in results.values():
    prof = r["daily"].values / r["mean_level"]     # normalize → shape comparison
    ax.plot(range(len(prof)), prof, label=r["name"], lw=1.5)
ax.set_title("Daily profile (normalized by dataset mean)")
ax.set_xlabel("time-of-day step"); ax.set_ylabel("relative consumption")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "comparison_daily_profiles.png", dpi=150)
plt.close(fig)


# %%
# ---- 3. Overlaid weekly profiles ----
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(9, 4))
for r in results.values():
    prof = r["weekly"].values / r["mean_level"]
    ax.plot(dow_labels[:len(prof)], prof, marker="o", label=r["name"])
ax.set_title("Weekly profile (normalized)")
ax.set_ylabel("relative consumption"); ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "comparison_weekly_profiles.png", dpi=150)
plt.close(fig)


# %%
# ---- 4. Overlaid monthly / seasonal profiles ----
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(11, 4))
for r in results.values():
    prof = r["monthly"]
    ax.plot([month_labels[m-1] for m in prof.index], prof.values / r["mean_level"],
            marker="o", label=r["name"])
ax.set_title("Monthly profile (normalized) — seasonality")
ax.set_ylabel("relative consumption"); ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "comparison_monthly_profiles.png", dpi=150)
plt.close(fig)


# %%
# ---- 5. Distribution comparison (per-client mean, normalized) ----
fig, ax = plt.subplots(figsize=(9, 4))
for name, cfg in DATASETS.items():
    ts = load_dataset(cfg["path"], layout=cfg["layout"], date_col=cfg["timestamp_col"])
    cmeans = np.nanmean(ts.values, axis=0)
    cmeans = cmeans / np.nanmean(cmeans)       # normalize for shape comparison
    ax.hist(cmeans, bins=60, alpha=0.5, label=name, density=True)
ax.set_title("Per-client mean distribution (normalized) — client heterogeneity")
ax.set_xlabel("relative client mean"); ax.set_ylabel("density"); ax.legend()
ax.set_xlim(0, 4)
fig.tight_layout()
fig.savefig(out_dir / "comparison_client_heterogeneity.png", dpi=150)
plt.close(fig)


# %%
print(f"\n Comparison complete → {out_dir}")
for f in sorted(out_dir.iterdir()):
    print(" ", f.name)