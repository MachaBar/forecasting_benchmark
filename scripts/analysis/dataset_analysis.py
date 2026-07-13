# %% [markdown]
# # Dataset analysis — load curves
# Period, clients, moments, quantiles, autocorrelation, distributions,
# daily / weekly / annual profiles. Saves everything to outputs/analysis/<dataset>/.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset.dataset import load_dataset

# ---- CONFIG: edit these ----
DATASET_NAME = "cer_bis"
DATA_PATH    = "/home/d32485/forecasting_benchmark/data/cer_bis/load_curve.parquet"
TIMESTAMP_COL = "time"
LAYOUT = "wide"
FREQ_MINUTES = 30           # 30-min steps
SEASON_DAILY = 48           # steps per day
MAX_CLIENTS_PLOT = 200      # subsample for expensive per-client stats
# ----------------------------

out_dir = Path(f"outputs/analysis/{DATASET_NAME}")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving analysis to {out_dir}")

# %%
# ---- Load ----
ts = load_dataset(DATA_PATH, layout=LAYOUT, date_col=TIMESTAMP_COL)
values = ts.values                     # (T, N)
dates = pd.DatetimeIndex(ts.datetimes)
n_dates, n_users = values.shape
print(f"{n_users} clients × {n_dates} timesteps")

# %%
# ---- 1. Basic info ----
freq = pd.infer_freq(dates)
duration = dates[-1] - dates[0]
info = {
    "dataset": DATASET_NAME,
    "n_clients": int(n_users),
    "n_timesteps": int(n_dates),
    "date_start": str(dates[0]),
    "date_end": str(dates[-1]),
    "duration_days": duration.days,
    "inferred_freq": str(freq),
    "n_missing_total": int(np.isnan(values).sum()),
    "pct_missing": float(np.isnan(values).mean() * 100),
}
pd.Series(info).to_csv(out_dir / "01_basic_info.csv")
print(pd.Series(info))

# %%
# ---- 2. Moments & quantiles (global + per client) ----
flat = values[~np.isnan(values)]
global_stats = {
    "mean": np.mean(flat), "std": np.std(flat),
    "skew": pd.Series(flat).skew(), "kurtosis": pd.Series(flat).kurtosis(),
    "min": np.min(flat), "max": np.max(flat),
    **{f"q{int(q*100)}": np.quantile(flat, q)
       for q in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]},
}
pd.Series(global_stats).to_csv(out_dir / "02_global_stats.csv")
print(pd.Series(global_stats))

# per-client stats
per_client = pd.DataFrame({
    "mean": np.nanmean(values, axis=0),
    "std": np.nanstd(values, axis=0),
    "min": np.nanmin(values, axis=0),
    "max": np.nanmax(values, axis=0),
    "median": np.nanmedian(values, axis=0),
}, index=ts.user_names)
per_client.to_csv(out_dir / "02_per_client_stats.csv")
per_client.describe().to_csv(out_dir / "02_per_client_stats_summary.csv")

# %%
# ---- 3. Value distribution ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(flat, bins=100, color="steelblue")
ax[0].set_title(f"{DATASET_NAME} — value distribution")
ax[0].set_xlabel("consumption"); ax[0].set_ylabel("count")
# log scale often clearer for load curves (heavy right tail)
ax[1].hist(np.log1p(flat[flat >= 0]), bins=100, color="teal")
ax[1].set_title("log(1+value) distribution")
ax[1].set_xlabel("log consumption")
fig.tight_layout()
fig.savefig(out_dir / "03_value_distribution.png", dpi=150)
plt.close(fig)

# distribution of per-client means (client heterogeneity)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(per_client["mean"], bins=60, color="darkorange")
ax.set_title(f"{DATASET_NAME} — distribution of per-client mean consumption")
ax.set_xlabel("client mean"); ax.set_ylabel("count")
fig.tight_layout()
fig.savefig(out_dir / "03_client_mean_distribution.png", dpi=150)
plt.close(fig)

# %%
# ---- 4. Autocorrelation (mean ACF across a client sample) ----
from statsmodels.tsa.stattools import acf

n_lags = SEASON_DAILY * 8   # ~1 week of lags
sample = np.random.default_rng(0).choice(n_users, size=min(MAX_CLIENTS_PLOT, n_users), replace=False)
acfs = []
for u in sample:
    s = values[:, u]
    s = s[~np.isnan(s)]
    if len(s) > n_lags + 10:
        acfs.append(acf(s, nlags=n_lags, fft=True))
mean_acf = np.mean(acfs, axis=0)

pd.Series(mean_acf, name="acf").to_csv(out_dir / "04_mean_acf.csv")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(mean_acf, color="steelblue")
for d in range(1, 8):                       # daily peaks
    ax.axvline(d * SEASON_DAILY, color="tomato", ls="--", lw=0.7,
               label="daily cycle" if d == 1 else None)
ax.set_title(f"{DATASET_NAME} — mean autocorrelation ({len(acfs)} clients)")
ax.set_xlabel("lag (steps)"); ax.set_ylabel("ACF")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "04_autocorrelation.png", dpi=150)
plt.close(fig)

# %%
# ---- 5. Daily profile (mean by time-of-day) ----
df = pd.DataFrame(values, index=dates)
mean_series = df.mean(axis=1)               # average client at each timestep

by_tod = mean_series.groupby([dates.hour, dates.minute]).mean()
by_tod.index = [f"{h:02d}:{m:02d}" for h, m in by_tod.index]
by_tod.to_csv(out_dir / "05_daily_profile.csv")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(by_tod)), by_tod.values, color="steelblue")
ax.set_xticks(range(0, len(by_tod), 4))
ax.set_xticklabels(by_tod.index[::4], rotation=45, ha="right", fontsize=8)
ax.set_title(f"{DATASET_NAME} — average daily profile")
ax.set_ylabel("mean consumption")
fig.tight_layout()
fig.savefig(out_dir / "05_daily_profile.png", dpi=150)
plt.close(fig)

# %%
# ---- 6. Weekly profile (mean by day of week) ----
by_dow = mean_series.groupby(dates.dayofweek).mean()
by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_dow.to_csv(out_dir / "06_weekly_profile.csv")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(by_dow.index, by_dow.values, color="teal")
ax.set_title(f"{DATASET_NAME} — average by day of week")
ax.set_ylabel("mean consumption")
fig.tight_layout()
fig.savefig(out_dir / "06_weekly_profile.png", dpi=150)
plt.close(fig)

# heatmap: hour × day-of-week
pivot = mean_series.groupby([dates.dayofweek, dates.hour]).mean().unstack()
pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
ax.set_yticks(range(7)); ax.set_yticklabels(pivot.index)
ax.set_xlabel("hour of day"); ax.set_title(f"{DATASET_NAME} — hour × weekday heatmap")
fig.colorbar(im, ax=ax, label="mean consumption")
fig.tight_layout()
fig.savefig(out_dir / "06_weekly_heatmap.png", dpi=150)
plt.close(fig)

# %%
# ---- 7. Annual / monthly profile ----
by_month = mean_series.groupby(dates.month).mean()
by_month.index = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"][:len(by_month)]
by_month.to_csv(out_dir / "07_monthly_profile.csv")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(by_month.index, by_month.values, marker="o", color="darkorange")
ax.set_title(f"{DATASET_NAME} — average by month (seasonality)")
ax.set_ylabel("mean consumption")
fig.tight_layout()
fig.savefig(out_dir / "07_monthly_profile.png", dpi=150)
plt.close(fig)

# full timeline of the average client (trend + gaps)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates, mean_series.values, lw=0.5, color="steelblue")
ax.set_title(f"{DATASET_NAME} — mean consumption over time (average client)")
fig.tight_layout()
fig.savefig(out_dir / "07_full_timeline.png", dpi=150)
plt.close(fig)

# %%
print(f"\nAnalysis complete. All CSVs + figures saved to: {out_dir}")
print("Files:")
for f in sorted(out_dir.iterdir()):
    print(" ", f.name)