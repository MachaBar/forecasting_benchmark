#!/usr/bin/env python3
"""Combine un ou plusieurs summary_all_runs.csv (LGBM,
Prophet, Chronos, naive, ...), sur un ou plusieurs datasets, et produit des
tableaux LaTeX + plots PDF.

Chaque CSV source doit contenir au minimum une colonne "model" et des
colonnes métriques numériques (mae, rmse, mase, mae_normalized, ...). Les colonnes
context_length/prediction_length sont optionnelles mais activent des vues
supplémentaires si présentes.

Usage — un seul run :
    python scripts/analysis/make_report.py \
        --csv outputs/lgbm_variants/cer_bis/multirun_.../summary_all_runs.csv \
        --metrics mae rmse mase \
        --outdir report/cer_bis

Usage — comparaison cross-model (plusieurs familles, même dataset) :
    python scripts/analysis/make_report.py \
        --csv outputs/lgbm_variants/cer_bis/summary_all_runs.csv \
        --csv outputs/prophet/cer_bis/summary_all_runs.csv \
        --csv outputs/chronos/cer_bis/summary_all_runs.csv \
        --metrics mase mae_normalized \
        --outdir report/cer_bis_all_models

Usage — comparaison cross-population (même modèle, plusieurs datasets) :
    python scripts/analysis/make_report.py \
        --csv outputs/chronos/cer_bis/summary_all_runs.csv:cer_bis \
        --csv outputs/chronos/ideal/summary_all_runs.csv:ideal \
        --csv outputs/chronos/refit/summary_all_runs.csv:refit \
        --metrics mase \
        --group-by dataset \
        --outdir report/chronos_cross_population


# Comparer modèles avec timing
python scripts/analysis/make_report.py \
    --csv outputs/lgbm_variants/cer_bis/summary_all_runs.csv \
    --csv outputs/chronos/cer_bis/summary_all_runs.csv \
    --metrics mase mae \
    --timing per_forecast_ms fit_time_s \
    --outdir report/cer_bis_with_timing

# Cross-population + timing
python scripts/analysis/make_report.py \
    --csv outputs/chronos/cer_bis/summary_all_runs.csv:cer_bis \
    --csv outputs/chronos/ideal/summary_all_runs.csv:ideal \
    --csv outputs/chronos/refit/summary_all_runs.csv:refit \
    --metrics mase \
    --timing per_forecast_ms model_load_s \
    --group-by dataset \
    --outdir report/chronos_timing


python scripts/analysis/make_report.py \
    --csv outputs/lgbm_variants/cer_bis/summary_all_runs.csv \
    --csv outputs/lgbm_variants/ideal/summary_all_runs.csv \
    --csv outputs/chronos/cer_bis/summary_all_runs.csv \
    --csv outputs/chronos/ideal/summary_all_runs.csv \
    --metrics mase \
    --split-by dataset \
    --timing per_forecast_ms \
    --outdir report/matrix_models_datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

META_COLS = {
    "model", "variant", "dataset", "context_length", "prediction_length",
    "fit_time_s", "per_forecast_ms", "run_date", "stride", "batch_size",
    "probabilistic", "model_load_s", "total_eval_s", "pure_inference_s",
    "n_estimators_cap", "early_stopping_rounds", "val_mode",
    "best_iteration_mean", "best_iteration_min", "best_iteration_max",
    "cross_learning", "cross_learning_mode", "group_size",
}

TIMING_COLS = {"fit_time_s", "per_forecast_ms", "model_load_s", "total_eval_s",
               "pure_inference_s", "total_infer_s", "mean_per_cutoff_s"}

LOWER_IS_BETTER = True

METRIC_LABELS = {
    "mae_normalized": "NMAE",
    "rmse_normalized": "NRMSE",
    "fit_time_s": "Fit (s)",
    "per_forecast_ms": "Infer (ms)",
    "model_load_s": "Load (s)",
    "total_eval_s": "Total eval (s)",
    "pure_inference_s": "Pure infer (s)",
    "total_infer_s": "Total infer (s)",
    "mean_per_cutoff_s": "Per cutoff (s)",
}


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_one(spec: str) -> pd.DataFrame:
    """spec = "path.csv" ou "path.csv:label" — le label peuple la colonne
    "dataset" si elle est absente du csv (utile pour la comparaison
    cross-population, où chaque csv vient d'un dataset différent)."""
    if ":" in spec and not spec.startswith(("http:", "https:")):
        path, label = spec.rsplit(":", 1)
    else:
        path, label = spec, None

    df = pd.read_csv(path)
    if "variant" not in df.columns and "model" in df.columns:
        pass  # certains csv (foundation_runner) n'ont pas de variant, c'est ok
    if label and "dataset" not in df.columns:
        df["dataset"] = label
    df["_source"] = path
    return df


def load_all(specs: list[str]) -> pd.DataFrame:
    return pd.concat([load_one(s) for s in specs], ignore_index=True)


def available_metrics(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in META_COLS]

def metric_label(m: str) -> str:
    return METRIC_LABELS.get(m, m.upper())

# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def fmt(v, decimals=3):
    return "—" if pd.isna(v) else f"{v:.{decimals}f}"


def latex_summary(df: pd.DataFrame, group_by: str, metrics: list[str], caption: str, label: str,
                  timing_cols: list[str] = None) -> str:
    all_cols = metrics + (timing_cols or [])
    g = df.groupby(group_by)[all_cols].mean()
    best = g[metrics].min() if LOWER_IS_BETTER else g[metrics].max()

    col_spec = "l" + "r" * len(all_cols)
    headers = [rf"\textbf{{{group_by.capitalize()}}}"]
    for m in metrics:
        headers.append(rf"\textbf{{{metric_label(m)}}}")
    for t in (timing_cols or []):
        headers.append(rf"\textbf{{{metric_label(t)}}}")
    header = " & ".join(headers)

    lines = [
        r"\begin{table}[h]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header + r" \\", r"\midrule",
    ]
    for key in sorted(g.index):
        row = [str(key).replace("_", r"\_")]
        for m in metrics:
            v = g.loc[key, m]
            cell = fmt(v)
            if not pd.isna(v) and np.isclose(v, best[m]):
                cell = rf"\textbf{{{cell}}}"
            row.append(cell)
        for t in (timing_cols or []):
            v = g.loc[key, t]
            cell = fmt(v, decimals=2)
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_cross_table(df: pd.DataFrame, rows_col: str, cols_col: str, metric: str, caption: str, label: str) -> str:
    """Tableau croisé, ex: model (lignes) x dataset (colonnes), une métrique."""
    pivot = df.pivot_table(index=rows_col, columns=cols_col, values=metric, aggfunc="mean")
    best_per_col = pivot.min() if LOWER_IS_BETTER else pivot.max()

    col_spec = "l" + "r" * len(pivot.columns)
    header = " & ".join([rf"\textbf{{{rows_col.capitalize()}}}"] +
                         [rf"\textbf{{{str(c).replace('_', r'\_')}}}" for c in pivot.columns])
    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{{caption} ({metric.upper()})}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule", header + r" \\", r"\midrule",
    ]
    for idx in pivot.index:
        row = [str(idx).replace("_", r"\_")]
        for c in pivot.columns:
            v = pivot.loc[idx, c]
            cell = fmt(v)
            if not pd.isna(v) and np.isclose(v, best_per_col[c]):
                cell = rf"\textbf{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots (PDF vectoriel, PNG optionnel en aperçu)
# --------------------------------------------------------------------------- #
def savefig(fig, outpath: Path, also_png: bool):
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".pdf"))
    if also_png:
        fig.savefig(outpath.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, group_by: str, metric: str, outpath: Path, also_png: bool):
    g = df.groupby(group_by)[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(g) + 1.5))
    ax.barh(g.index.astype(str), g.values, color="steelblue")
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{metric.upper()} par {group_by} (plus bas = mieux)")
    ax.grid(axis="x", alpha=0.3)
    savefig(fig, outpath, also_png)


def plot_grouped_bar(df: pd.DataFrame, group_by: str, split_by: str, metric: str, outpath: Path, also_png: bool):
    """Barres groupées : ex. modèle sur l'axe X, une couleur par dataset —
    la vue standard pour illustrer la robustesse cross-population."""
    pivot = df.pivot_table(index=group_by, columns=split_by, values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot)), 5))
    x = np.arange(len(pivot.index))
    w = 0.8 / len(pivot.columns)
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * w, pivot[col].values, width=w, label=str(col))
    ax.set_xticks(x + w * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index.astype(str), rotation=30, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} par {group_by}, ventilé par {split_by}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, outpath, also_png)


def plot_metric_vs_x(df: pd.DataFrame, group_by: str, x_col: str, metric: str, outpath: Path, also_png: bool):
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, sub in df.groupby(group_by):
        g = sub.groupby(x_col)[metric].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=str(key))
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. {x_col}, par {group_by}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    savefig(fig, outpath, also_png)


def plot_accuracy_vs_timing(df: pd.DataFrame, group_by: str, metric: str, timing_col: str,
                            outpath: Path, also_png: bool):
    """Scatter plot: accuracy vs inference time, one dot per model."""
    g = df.groupby(group_by)[[metric, timing_col]].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (name, row) in enumerate(g.iterrows()):
        ax.scatter(row[timing_col], row[metric], s=200, alpha=0.7, label=str(name))
        ax.annotate(str(name), (row[timing_col], row[metric]), fontsize=9, ha="center")
    ax.set_xlabel(timing_col.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Accuracy–Speed Trade-off ({metric.upper()} vs {timing_col.replace('_', ' ')})")
    ax.grid(alpha=0.3)
    savefig(fig, outpath, also_png)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True,
                   help="Chemin(s) vers summary_all_runs.csv. Format 'path.csv:label' "
                        "pour forcer une colonne dataset absente du fichier.")
    p.add_argument("--metrics", nargs="+", default=["mae", "rmse", "mase"])
    p.add_argument("--timing", nargs="+", default=None,
                   help="Colonnes timing à afficher dans tables (ex: per_forecast_ms fit_time_s)")
    p.add_argument("--group-by", default="model", help="Colonne de regroupement principale")
    p.add_argument("--split-by", default=None,
                   help="Colonne secondaire pour les vues croisées/groupées "
                        "(ex: 'dataset' pour comparer la robustesse cross-population)")
    p.add_argument("--outdir", required=True)
    p.add_argument("--png", action="store_true", help="Sauver aussi en PNG (aperçu)")
    p.add_argument("--list-metrics", action="store_true")
    args = p.parse_args()

    df = load_all(args.csv)

    if args.list_metrics:
        print("Métriques disponibles :", available_metrics(df))
        print("Colonnes méta détectées :", [c for c in df.columns if c in META_COLS])
        return

    missing = [m for m in args.metrics if m not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes: {missing}\nDisponibles: {available_metrics(df)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Sources : {args.csv}")
    print(f"Rows    : {len(df)}")
    print(f"Group by: {args.group_by} → {sorted(df[args.group_by].astype(str).unique())}")
    if args.split_by:
        print(f"Split by: {args.split_by} → {sorted(df[args.split_by].astype(str).unique())}")
    print(f"Metrics : {args.metrics}")

    # ---- LaTeX ----
    tex_parts = [
        latex_summary(df, args.group_by, args.metrics,
                      caption=f"Performance summary by {args.group_by}",
                      label=f"tab:summary_{args.group_by}",
                      timing_cols=timing_cols)]
    if args.split_by:
        for m in args.metrics:
            tex_parts.append(
                latex_cross_table(df, args.group_by, args.split_by, m,
                                   caption=f"{args.group_by} vs {args.split_by}",
                                   label=f"tab:cross_{args.group_by}_{args.split_by}_{m}")
            )
    if args.timing:
        print(f"Timing  : {args.timing}")
    # Filter timing cols to those actually present
    timing_cols = [t for t in (args.timing or []) if t in df.columns]

    (outdir / "tables.tex").write_text("\n\n\n".join(tex_parts))
    print(f"✓ LaTeX → {outdir}/tables.tex")

    # ---- Plots ----
    for m in args.metrics:
        plot_bar(df, args.group_by, m, outdir / f"{m}_by_{args.group_by}", args.png)
        if args.split_by:
            plot_grouped_bar(df, args.group_by, args.split_by, m,
                              outdir / f"{m}_by_{args.group_by}_split_{args.split_by}", args.png)
        if "prediction_length" in df.columns:
            plot_metric_vs_x(df, args.group_by, "prediction_length", m,
                              outdir / f"{m}_vs_horizon", args.png)
        if "context_length" in df.columns:
            plot_metric_vs_x(df, args.group_by, "context_length", m,
                              outdir / f"{m}_vs_context", args.png)
        for t in timing_cols:
            plot_accuracy_vs_timing(df, args.group_by, m, t,
                                    outdir / f"{m}_vs_{t}", args.png)
    summary_cols = args.metrics + timing_cols
    print("\n" + df.groupby(args.group_by)[summary_cols].mean().round(3).to_string())

    print("\n" + df.groupby(args.group_by)[args.metrics].mean().round(3).to_string())


if __name__ == "__main__":
    main()
