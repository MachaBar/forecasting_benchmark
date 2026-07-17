# scripts/analysis/pick_best_prophet.py
import pandas as pd
from pathlib import Path

# le summary du grid-search val (adapte le chemin daté)
path = sorted(Path("outputs/prophet/cer_bis").glob("multirun_*/summary_all_runs.csv"))[-1]
df = pd.read_csv(path)
df = df[df["eval_split"] == "val"]

best = df.sort_values("mase").iloc[0]     # meilleur = plus bas MASE
print("Best params (min MASE on val):")
print(f"  changepoint_prior_scale = {best['changepoint_prior_scale']}")
print(f"  seasonality_prior_scale = {best['seasonality_prior_scale']}")
print(f"  → val MASE = {best['mase']:.4f}")
print(df[["changepoint_prior_scale","seasonality_prior_scale","mase","wql"]].sort_values("mase").to_string())

# uv run python scripts/analysis/pick_best_prophet.py