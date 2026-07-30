#!/usr/bin/env python3
"""Lit summary_all_runs.csv (issu de run_lgbm_variants.py) et produit :
  - des tableaux LaTeX (résumé par variant, par ctx/horizon, détaillé)
  - des plots de comparaison

Usage:
    python scripts/analysis/summarize_lgbm_variants.py \
        --csv outputs/lgbm_variants/cer_bis/multirun_2026-07-28_19-43-10/summary_all_runs.csv \
        --metrics mae rmse mase nmae \
        --outdir outputs/lgbm_variants/cer_bis/report

Édite METRICS ci-dessous pour changer les métriques par défaut sans passer
--metrics à chaque fois.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Métriques par défaut — édite cette liste selon ce que tu veux dans le
# rapport. Doit correspondre à des noms de colonnes présents dans le CSV
# (ex: mae, rmse, mape, mase, wql, sql, crps, ...).
# --------------------------------------------------------------------------- #
METRICS = ["mae", "rmse", "mase", "mape"]

# Colonnes non-métriques présentes dans summary_all_runs.csv (à exclure des
# métriques auto-détectées, et utilisées comme clés de groupement).
META_COLS = {
    "model", "variant", "context_length", "prediction_length",
    "fit_time_s", "per_forecast_ms", "run_date",
    "n_estimators_cap", "early_stopping_rounds", "val_mode",
    "best_iteration_mean", "best_iteration_min", "best_iteration_max",
}

LOWER_IS_BETTER = True   # pour le tri / le bolding du meilleur résultat


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
def load_summary(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "variant" not in df.columns and "model" in df.columns:
        df["variant"] = df["model"].str.replace("LGBM-", "", regex=False)
    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    """Métriques numériques présentes, hors colonnes méta."""
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in META_COLS]


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def fmt(v, decimals=3):
    if pd.isna(v):
        return "—"
    return f"{v:.{decimals}f}"


def latex_summary_by_variant(df: pd.DataFrame, metrics: list[str], caption: str, label: str) -> str:
    """Une ligne par variant, moyenne sur tous les (ctx, horizon)."""
    g = df.groupby("variant")[metrics].mean()
    best = g.min() if LOWER_IS_BETTER else g.max()

    col_spec = "l" + "r" * len(metrics)
    header = " & ".join([r"\textbf{Variant}"] + [rf"\textbf{{{m.upper()}}}" for m in metrics])

    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    for variant in sorted(g.index):
        row = [variant.replace("_", r"\_")]
        for m in metrics:
            v = g.loc[variant, m]
            cell = fmt(v)
            if not pd.isna(v) and np.isclose(v, best[m]):
                cell = rf"\textbf{{{cell}}}"
            row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_by_config(df: pd.DataFrame, metrics: list[str], caption: str, label: str) -> str:
    """Une ligne par (context_length, prediction_length), moyenne sur les variants."""
    g = df.groupby(["context_length", "prediction_length"])[metrics].mean()

    col_spec = "rr" + "r" * len(metrics)
    header = " & ".join(
        [r"\textbf{Context}", r"\textbf{Horizon}"] + [rf"\textbf{{{m.upper()}}}" for m in metrics]
    )

    lines = [
        r"\begin{table}[h]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    for (ctx, h), row_vals in g.iterrows():
        row = [str(int(ctx)), str(int(h))] + [fmt(row_vals[m]) for m in metrics]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_detailed(df: pd.DataFrame, metrics: list[str], caption: str, label: str) -> str:
    """Une ligne par (variant, context_length, prediction_length)."""
    cols = ["variant", "context_length", "prediction_length"] + metrics
    sub = df[cols].sort_values(["variant", "context_length", "prediction_length"])

    col_spec = "lrr" + "r" * len(metrics)
    header = " & ".join(
        [r"\textbf{Variant}", r"\textbf{Ctx}", r"\textbf{H}"]
        + [rf"\textbf{{{m.upper()}}}" for m in metrics]
    )

    lines = [
        r"\begin{table}[h]", r"\small", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        header + r" \\", r"\midrule",
    ]
    for _, row in sub.iterrows():
        cells = [
            str(row["variant"]).replace("_", r"\_"),
            str(int(row["context_length"])),
            str(int(row["prediction_length"])),
        ] + [fmt(row[m]) for m in metrics]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_metric_by_variant(df: pd.DataFrame, metric: str, outpath: Path):
    g = df.groupby("variant")[metric].mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(g) + 1.5))
    ax.barh(g.index, g.values, color="steelblue")
    for i, v in enumerate(g.values):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"{metric.upper()} moyen par variant (plus bas = mieux)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_metric_vs_horizon(df: pd.DataFrame, metric: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, sub in df.groupby("variant"):
        g = sub.groupby("prediction_length")[metric].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=variant)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. horizon, par variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_metric_vs_context(df: pd.DataFrame, metric: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, sub in df.groupby("variant"):
        g = sub.groupby("context_length")[metric].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=variant)
    ax.set_xlabel("Context length")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} vs. context length, par variant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, metric: str, variant: str, outpath: Path):
    sub = df[df["variant"] == variant]
    pivot = sub.pivot_table(index="context_length", columns="prediction_length", values=metric)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Prediction horizon"); ax.set_ylabel("Context length")
    ax.set_title(f"{metric.upper()} — {variant}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_fit_time_vs_horizon(df: pd.DataFrame, outpath: Path):
    if "fit_time_s" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, sub in df.groupby("variant"):
        g = sub.groupby("prediction_length")["fit_time_s"].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", label=variant)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Fit time (s)")
    ax.set_title("Temps d'entraînement vs. horizon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Chemin vers summary_all_runs.csv")
    p.add_argument("--metrics", nargs="+", default=None,
                    help="Métriques à inclure (défaut: METRICS en haut du fichier)")
    p.add_argument("--outdir", default=None, help="Défaut: <dossier du csv>/report")
    p.add_argument("--list-metrics", action="store_true",
                    help="Affiche les métriques numériques disponibles dans le csv et quitte")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = load_summary(csv_path)

    if args.list_metrics:
        print("Métriques disponibles :")
        for m in available_metrics(df):
            print(f"  - {m}")
        return

    metrics = args.metrics or METRICS
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes absentes du CSV: {missing}\n"
            f"Disponibles: {available_metrics(df)}"
        )

    outdir = Path(args.outdir) if args.outdir else csv_path.parent / "report"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"CSV     : {csv_path}")
    print(f"Rows    : {len(df)}")
    print(f"Variants: {sorted(df['variant'].unique())}")
    print(f"Metrics : {metrics}")
    print(f"Outdir  : {outdir}")

    # ---- LaTeX ----
    tex_parts = [
        latex_summary_by_variant(
            df, metrics,
            caption="LGBM variants: performance summary (mean over context/horizon)",
            label="tab:lgbm_variant_summary",
        ),
        latex_by_config(
            df, metrics,
            caption="LGBM variants: performance by context length and horizon (mean over variants)",
            label="tab:lgbm_config_summary",
        ),
        latex_detailed(
            df, metrics,
            caption="LGBM variants: detailed results",
            label="tab:lgbm_detailed",
        ),
    ]
    tex_path = outdir / "tables.tex"
    tex_path.write_text("\n\n\n".join(tex_parts))
    print(f"\n✓ LaTeX  → {tex_path}")

    # ---- Plots ----
    for m in metrics:
        plot_metric_by_variant(df, m, outdir / f"{m}_by_variant.png")
        plot_metric_vs_horizon(df, m, outdir / f"{m}_vs_horizon.png")
        plot_metric_vs_context(df, m, outdir / f"{m}_vs_context.png")
        for variant in df["variant"].unique():
            plot_heatmap(df, m, variant, outdir / f"{m}_heatmap_{variant}.png")
    plot_fit_time_vs_horizon(df, outdir / "fit_time_vs_horizon.png")
    print(f"✓ Plots  → {outdir}/*.png")

    # ---- Aperçu console ----
    print("\n" + "=" * 80)
    print("Résumé par variant")
    print("=" * 80)
    print(df.groupby("variant")[metrics].mean().round(3))


if __name__ == "__main__":
    main()