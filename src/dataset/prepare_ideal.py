#!/usr/bin/env python3
"""
Inspired by https://github.com/mmcux/benchmarking_tsfm_household_load_forecasting/tree/main/src/data_sources

Turn the raw IDEAL household_sensors.zip into the (wide parquet + client-split
pickle) pair that configs/dataset/*.yaml expects.

    python -m scripts.data.prepare_ideal --inspect      # see what's in the zip first
    python -m scripts.data.prepare_ideal --min-timesteps 8760

Why the "dense rectangle" step: load_dataset() does `dropna(axis=0, how="any")`,
so a single NaN kills the whole timestamp row for every client. IDEAL homes have
very different recording periods, so we must first pick a (date-range x clients)
block that is completely dense.
"""
from __future__ import annotations

import argparse
import pickle
import random
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HOME_RE = re.compile(r"(home\d+)_([^_/]+)")


def inspect(zip_path: Path) -> None:
    """Print the electric-mains file layout — how many files per home, etc."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".gz") and "electric-mains" in n]
    print(f"{len(names)} electric-mains files")
    per_home: dict[str, list[str]] = {}
    for n in names:
        m = HOME_RE.search(n)
        per_home.setdefault(m.group(1) if m else "?", []).append(n)
    multi = {h: f for h, f in per_home.items() if len(f) > 1}
    print(f"{len(per_home)} homes; {len(multi)} with MORE THAN ONE file")
    for h, f in list(multi.items())[:5]:
        print(f"  {h}: {[Path(x).name for x in f]}")
    print("\nSample names:")
    for n in names[:5]:
        print("  ", n)



def scan_tradeoff(wide: pd.DataFrame, thresholds: list[int]) -> None:
    """For each candidate min_timesteps, report the best (clients x duration)
    rectangle WITHOUT writing anything — lets you pick a value before committing."""
    mask = wide.isna().to_numpy()
    T, C = mask.shape
    cum = np.vstack([np.zeros(C, dtype=np.int64), np.cumsum(mask, axis=0)])
    valid = ~mask
    has_any = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    last = T - 1 - np.argmax(valid[::-1], axis=0)
    starts = np.unique(first[has_any])
    ends = np.unique(last[has_any])

    print(f"{'min_timesteps':>14} {'~duration':>12} {'n_clients':>10} "
          f"{'n_steps_kept':>13} {'date_start':>12} {'date_end':>12} {'test_cutoffs*':>14}")

    for thr in thresholds:
        best = (0, -1, 0)  # start, end, area
        best_clients = 0
        for s in starts:
            for e in ends:
                if e - s + 1 < thr:
                    continue
                keep = np.where((cum[e + 1] - cum[s]) == 0)[0]
                if keep.size == 0:
                    continue
                area = keep.size * (e - s + 1)
                if area > best[2]:
                    best = (int(s), int(e), area)
                    best_clients = keep.size

        if best[1] < 0:
            print(f"{thr:>14} {'—':>12} {'—':>10} {'no dense block found':>13}")
            continue

        s, e = best[0], best[1]
        n_steps = e - s + 1
        n_days = n_steps / 24
        date_start = wide.index[s].strftime("%Y-%m-%d")
        date_end = wide.index[e].strftime("%Y-%m-%d")
        # *rough test-cutoff estimate assuming ratios=0.7/0.15/0.15, ctx=720, h=48, mean gap=30h
        test_block = 0.15 * n_steps
        approx_cutoffs = max(0, int((test_block - 720 - 48) / 30) + 1) if test_block >= 768 else 0

        print(f"{thr:>14} {f'{n_days:.0f}d':>12} {best_clients:>10} {n_steps:>13} "
              f"{date_start:>12} {date_end:>12} {approx_cutoffs:>14}")

def pareto_rectangles(wide: pd.DataFrame) -> pd.DataFrame:
    """All non-dominated (n_clients, n_steps) dense rectangles: for each
    achievable n_clients, the maximum n_steps obtainable (and vice versa).
    Lets you target an EXACT (clients, steps) pair instead of only the
    max-area one that --scan/best_rectangle would pick."""
    mask = wide.isna().to_numpy()
    T, C = mask.shape
    cum = np.vstack([np.zeros(C, dtype=np.int64), np.cumsum(mask, axis=0)])
    valid = ~mask
    has_any = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    last = T - 1 - np.argmax(valid[::-1], axis=0)
    starts = np.unique(first[has_any])
    ends = np.unique(last[has_any])

    candidates = []
    for s in starts:
        for e in ends:
            n_steps = e - s + 1
            if n_steps <= 0:
                continue
            keep = np.where((cum[e + 1] - cum[s]) == 0)[0]
            if keep.size == 0:
                continue
            candidates.append((int(s), int(e), n_steps, keep.size))

    df = pd.DataFrame(candidates, columns=["start_idx", "end_idx", "n_steps", "n_clients"])
    # keep only the best n_steps for each n_clients value, then prune dominated rows
    df = df.sort_values("n_steps", ascending=False).drop_duplicates("n_clients")
    df = df.sort_values("n_clients").reset_index(drop=True)
    best_so_far = -1
    keep_rows = []
    for i in range(len(df) - 1, -1, -1):  # scan from most clients to fewest
        if df.loc[i, "n_steps"] > best_so_far:
            keep_rows.append(i)
            best_so_far = df.loc[i, "n_steps"]
    df = df.loc[sorted(keep_rows)].reset_index(drop=True)

    df["date_start"] = df["start_idx"].map(lambda i: wide.index[i].strftime("%Y-%m-%d"))
    df["date_end"] = df["end_idx"].map(lambda i: wide.index[i].strftime("%Y-%m-%d"))
    df["duration_days"] = (df["n_steps"] / 24).round(1)
    return df[["n_clients", "n_steps", "duration_days", "date_start", "date_end", "start_idx", "end_idx"]]

def read_zip(zip_path: Path, agg: str) -> pd.DataFrame:
    """Read every electric-mains series, resample to hourly, return long frame."""
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".gz") and "electric-mains" in n]
        for i, name in enumerate(names, 1):
            m = HOME_RE.search(name)
            if m is None:
                print(f"  skip (no home id): {name}")
                continue
            with z.open(name) as fh:
                df = pd.read_csv(fh, compression="gzip", header=None,
                                 names=["timestamp", "consumption"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # hourly mean, W -> kW (same convention as the original loader)
            s = (df.set_index("timestamp")["consumption"]
                   .resample("h").mean() / 1000.0)
            frames.append(pd.DataFrame({
                "timestamp": s.index,
                "consumer_id": m.group(1),
                "consumption": s.to_numpy(),
            }))
            if i % 25 == 0 or i == len(names):
                print(f"  {i}/{len(names)} files")

    long_df = pd.concat(frames, ignore_index=True).dropna(subset=["consumption"])

    # Several files for one home (different circuits) -> one series per home.
    n_before = len(long_df)
    long_df = long_df.groupby(["timestamp", "consumer_id"], as_index=False)["consumption"].agg(agg)
    if len(long_df) < n_before:
        print(f"  merged {n_before - len(long_df):,} same-hour rows with agg='{agg}' "
              f"(homes with multiple electric-mains files)")
    return long_df



def best_rectangle(wide: pd.DataFrame, min_timesteps: int):
    """Largest NaN-free (timesteps x clients) block, maximising their product."""
    mask = wide.isna().to_numpy()
    T, C = mask.shape
    cum = np.vstack([np.zeros(C, dtype=np.int64), np.cumsum(mask, axis=0)])

    valid = ~mask
    has_any = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    last = T - 1 - np.argmax(valid[::-1], axis=0)

    best = (0, -1, [], 0)
    for s in np.unique(first[has_any]):
        for e in np.unique(last[has_any]):
            if e - s + 1 < min_timesteps:
                continue
            keep = np.where((cum[e + 1] - cum[s]) == 0)[0]
            if keep.size == 0:
                continue
            area = keep.size * (e - s + 1)
            if area > best[3]:
                best = (int(s), int(e), wide.columns[keep].tolist(), area)

    if best[1] < 0:
        raise SystemExit(
            f"No NaN-free block of >= {min_timesteps} steps. "
            "Lower --min-timesteps or raise --interpolate-limit."
        )
    return best[0], best[1], best[2]


def load_or_build_wide(args, outdir: Path) -> pd.DataFrame:
    """Read the raw zip ONCE and cache the pivoted matrix. Every later run
    (--scan / --pareto / --periods / --plot / materialization) reads the cache.

    The cache holds the RAW matrix: all clients, full span, NaN untouched, no
    zero-run masking, no interpolation, no per-client trimming — so every
    downstream option can still be explored from it without re-reading the zip.
    """
    cache = outdir / f"{args.name}_raw_wide.parquet"

    if cache.exists() and not args.rebuild:
        wide = pd.read_parquet(cache).set_index("time")
        print(f"Loaded cached raw matrix from {cache}")
    else:
        print("Reading zip (slow — cached afterwards)...")
        long_df = read_zip(Path(args.zip), args.agg)
        print(f"  {len(long_df):,} rows, {long_df['consumer_id'].nunique()} homes")
        wide = to_wide(long_df, args.freq)
        wide.reset_index().to_parquet(cache, index=False)
        print(f"Cached raw matrix -> {cache}")

    print(f"Raw: {wide.shape[0]:,} steps x {wide.shape[1]} clients "
          f"({wide.isna().to_numpy().mean():.1%} NaN)")
    return wide


def plot_landscape(wide: pd.DataFrame, outdir: Path, name: str,
                   highlight: tuple[int, int, list[str]] | None = None) -> None:
    """Missingness heatmap + active-client count + coverage distribution.

    highlight: optional (start_idx, end_idx, kept_clients) to overlay a chosen
    dense rectangle, so you can SEE what a --target-clients choice selects.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    present = wide.notna().to_numpy()                    # (T, C)
    T, C = present.shape

    # sort clients by first valid timestep -> the "staircase" becomes readable
    first_valid = np.where(present.any(axis=0), np.argmax(present, axis=0), T)
    order = np.argsort(first_valid)
    cols_sorted = [wide.columns[i] for i in order]
    mat = present[:, order].T                            # (C, T)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.35)

    # --- Panel 1: missingness heatmap ---
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest",
               extent=[mdates.date2num(wide.index[0]), mdates.date2num(wide.index[-1]), C, 0])
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.set_ylabel(f"clients (n={C}, sorted by start)")
    ax1.set_title(f"{name} — data availability (black = present, white = missing)")

    if highlight is not None:
        s, e, kept = highlight
        ax1.axvspan(mdates.date2num(wide.index[s]), mdates.date2num(wide.index[e]),
                    color="tab:red", alpha=0.15)
        kept_set = set(kept)
        ypos = [i for i, c in enumerate(cols_sorted) if c in kept_set]
        ax1.scatter([mdates.date2num(wide.index[0])] * len(ypos), np.array(ypos) + 0.5,
                    marker="|", s=40, color="tab:red",
                    label=f"selected ({len(kept)} clients)")
        ax1.legend(loc="upper right")

    # --- Panel 2: how many clients are recording at each timestep ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(wide.index, present.sum(axis=1), lw=1.2, color="tab:blue")
    ax2.fill_between(wide.index, present.sum(axis=1), alpha=0.25, color="tab:blue")
    if highlight is not None:
        ax2.axvspan(wide.index[highlight[0]], wide.index[highlight[1]],
                    color="tab:red", alpha=0.15)
    ax2.set_ylabel("active clients")
    ax2.set_title("Number of clients with data at each timestep")
    ax2.grid(alpha=0.3)

    # --- Panel 3: per-client coverage within its own recording span ---
    ax3 = fig.add_subplot(gs[2])
    cov = []
    for i in range(C):
        col_present = present[:, i]
        if not col_present.any():
            continue
        a, b = np.argmax(col_present), T - 1 - np.argmax(col_present[::-1])
        cov.append(col_present[a:b + 1].mean() * 100)
    ax3.hist(cov, bins=40, color="tab:green", edgecolor="white")
    ax3.set_xlabel("coverage within own span (%)")
    ax3.set_ylabel("# clients")
    ax3.set_title("Per-client completeness (100% = no internal gaps)")
    ax3.grid(alpha=0.3)

    path = outdir / f"{name}_landscape.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {path}")

def main():
    print("Debug begin main")
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/raw/Ideal_Dataset/household_sensors.zip")
    p.add_argument("--outdir", default="data/ideal")
    p.add_argument("--name", default="ideal")
    p.add_argument("--inspect", action="store_true", help="describe the zip and exit")
    p.add_argument("--agg", default="sum", choices=["sum", "mean"],
                   help="how to merge multiple electric-mains files of one home")
    p.add_argument("--interpolate-limit", type=int, default=3,
                   help="linearly fill interior gaps up to N hours (0 = off)")
    p.add_argument("--min-timesteps", type=int, default=8760, help="8760 = 1 year hourly")
    p.add_argument("--ratios", default="0.7,0.15,0.15", help="CLIENT split train,val,test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scan", action="store_true",
                   help="print the clients-vs-duration trade-off for several "
                        "min-timesteps values and exit (no files written)")
    p.add_argument("--scan-values", default="2190,4380,6000,8760,13140,17520",
                   help="comma-separated min-timesteps candidates for --scan "
                        "(default: 3mo,6mo,~8mo,1y,1.5y,2y hourly)")
    p.add_argument("--pareto", action="store_true",
                   help="list every non-dominated (n_clients, n_steps) rectangle and exit")
    p.add_argument("--target-clients", type=int, default=None,
                   help="materialize the rectangle with exactly this many clients "
                        "(from --pareto), instead of the max-area heuristic")
    p.add_argument("--rebuild", action="store_true",
                   help="re-read the zip even if the raw cache exists")
    p.add_argument("--plot", action="store_true",
                   help="save an availability/coverage figure and exit")
    p.add_argument("--plot-clients", type=int, default=None,
                   help="with --plot: overlay the --pareto rectangle having this many clients")
    args = p.parse_args()
    print("Debug")
    zip_path = Path(args.zip)
    if args.inspect:
        print("Inspect")
        inspect(zip_path)
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Reading zip (this is the slow part)...")
    # long_df = read_zip(zip_path, args.agg)
    # print(f"  {len(long_df):,} hourly rows, {long_df['consumer_id'].nunique()} homes")

    # wide = long_df.pivot(index="timestamp", columns="consumer_id", values="consumption")
    # wide.columns = [str(c) for c in wide.columns]
    # wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="h"))
    # wide.index.name = "time"
    # print(f"Wide: {wide.shape[0]:,} steps x {wide.shape[1]} clients "
    #       f"({wide.isna().to_numpy().mean():.1%} NaN)")

    wide = load_or_build_wide(args, outdir)

    if args.plot:
        highlight = None
        if args.plot_clients is not None:
            table = pareto_rectangles(wide)
            row = table[table["n_clients"] == args.plot_clients]
            if row.empty:
                raise SystemExit(f"No rectangle with {args.plot_clients} clients. "
                                 f"Available: {table['n_clients'].tolist()}")
            s, e = int(row.iloc[0]["start_idx"]), int(row.iloc[0]["end_idx"])
            kept = wide.columns[wide.isna().to_numpy()[s:e + 1].sum(axis=0) == 0].tolist()
            highlight = (s, e, kept)
        plot_landscape(wide, outdir, args.name, highlight=highlight)
        return


    if args.interpolate_limit > 0:
        before = int(wide.isna().to_numpy().sum())
        wide = wide.interpolate(method="linear", limit=args.interpolate_limit,
                                limit_area="inside")
        print(f"Interpolated <= {args.interpolate_limit}h gaps: "
              f"{before - int(wide.isna().to_numpy().sum()):,} values")

    if args.scan:
        thresholds = [int(x) for x in args.scan_values.split(",")]
        scan_tradeoff(wide, thresholds)
        return

    if args.pareto:
        table = pareto_rectangles(wide)
        pd.set_option("display.max_rows", None)
        print(table.to_string(index=False))
        return

    if args.target_clients is not None:
        table = pareto_rectangles(wide)
        row = table[table["n_clients"] == args.target_clients]
        if row.empty:
            raise SystemExit(
                f"No rectangle with exactly {args.target_clients} clients. "
                f"Closest available: {table['n_clients'].tolist()}"
            )
        s, e = int(row.iloc[0]["start_idx"]), int(row.iloc[0]["end_idx"])
        mask_e = wide.isna().to_numpy()[s:e + 1]
        keep = wide.columns[(mask_e.sum(axis=0) == 0)]
        wide = wide.iloc[s:e + 1][keep]
        assert not wide.isna().to_numpy().any()
        print(f"Materialized target: {len(keep)} clients x {wide.shape[0]} steps "
              f"({wide.index[0]} -> {wide.index[-1]})")
        # skip interpolate + best_rectangle below — fall through to the write step

    s, e, clients = best_rectangle(wide, args.min_timesteps)
    wide = wide.iloc[s:e + 1][clients]
    assert not wide.isna().to_numpy().any()
    print(f"Kept {wide.shape[1]} clients x {wide.shape[0]:,} steps "
          f"({wide.index[0]} -> {wide.index[-1]})")

    out_pq = outdir / f"{args.name}_load_curve.parquet"
    wide.reset_index().to_parquet(out_pq, index=False)   # `time` as a COLUMN
    print(f"\n-> {out_pq}")

    r_train, r_val, _ = (float(x) for x in args.ratios.split(","))
    ids = sorted(wide.columns.tolist())
    random.Random(args.seed).shuffle(ids)
    n_tr, n_va = int(round(r_train * len(ids))), int(round(r_val * len(ids)))
    split = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}

    out_pkl = outdir / f"{args.name}_train_valid_test_id_split.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(split, f)
    print(f"-> {out_pkl}  ({len(split['train'])}/{len(split['val'])}/{len(split['test'])})")


if __name__ == "__main__":
    main()