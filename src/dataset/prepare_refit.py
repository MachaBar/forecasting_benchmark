#!/usr/bin/env python3
"""Turn the raw REFIT archive.zip into the (wide parquet + client-split pickle)
pair expected by configs/dataset/refit.yaml — same conventions as CER.

Only the household Aggregate channel is used (the 9 per-appliance columns are
never loaded). REFIT files are already curated: readings are event-driven,
forward-filled for gaps <= 2 min, and ZERO-filled for gaps > 2 min. There is no
documented sentinel value, so nothing is dropped on value alone — but long runs
of exact zeros are almost certainly outages rather than genuine zero draw, hence
--zero-run-to-nan.

    python -u -m src.dataset.prepare_refit --inspect
    python -u -m src.dataset.prepare_refit --periods
    python -u -m src.dataset.prepare_refit --scan
    python -u -m src.dataset.prepare_refit --pareto
    python -u -m src.dataset.prepare_refit --min-timesteps 17520
    python -u -m src.dataset.prepare_refit --target-clients 18

The zip is extracted once into <outdir>/_extracted/ and reused afterwards.
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

HOUSE_RE = re.compile(r"House(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def ensure_extracted(archive_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(extract_dir.glob("**/*.csv"))
    if csvs:
        print(f"Reusing {len(csvs)} already-extracted CSVs in {extract_dir}")
        return csvs

    if not archive_path.exists():
        raise SystemExit(f"Archive not found: {archive_path}")

    print(f"Extracting {archive_path} -> {extract_dir} (first run only)...")
    with zipfile.ZipFile(archive_path) as z:
        z.extractall(extract_dir)
    csvs = sorted(extract_dir.glob("**/*.csv"))
    print(f"Extracted {len(csvs)} CSV files")
    return csvs


def inspect(archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as z:
        names = z.namelist()
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        print(f"{len(names)} entries, {len(csv_names)} CSV")

        houses = sorted(
            {HOUSE_RE.search(n).group(1) for n in csv_names if HOUSE_RE.search(n)},
            key=int,
        )
        print(f"{len(houses)} houses: {houses}")
        print("\nAll CSV names:")
        for n in csv_names:
            print("  ", n)

        if csv_names:
            print(f"\nFirst 3 lines of {csv_names[0]}:")
            with z.open(csv_names[0]) as fh:
                for _ in range(3):
                    print("  ", fh.readline().decode(errors="replace").rstrip())


def read_refit(csvs: list[Path], freq: str) -> pd.DataFrame:
    """Read each house CSV, keep only Aggregate, resample to `freq`, return a
    long frame (timestamp, consumer_id, consumption) in kW."""
    frames = []
    for i, csv_path in enumerate(csvs, 1):
        m = HOUSE_RE.search(csv_path.stem)
        if m is None:
            print(f"  skip (no house id in name): {csv_path.name}")
            continue
        consumer_id = f"House{m.group(1)}"

        with open(csv_path, "r") as fh:
            first_line = fh.readline()
        has_header = "aggregate" in first_line.lower()

        if has_header:
            df = pd.read_csv(
                csv_path,
                usecols=lambda c: c.strip().lower() in ("unix", "aggregate"),
            )
            df.columns = [c.strip().lower() for c in df.columns]
        else:
            # README column order: DateTime, Unix, Aggregate, Appliance1..9
            df = pd.read_csv(csv_path, header=None, usecols=[1, 2],
                             names=["unix", "aggregate"])

        df["timestamp"] = pd.to_datetime(df["unix"], unit="s", utc=True).dt.tz_localize(None)
        s = (df.set_index("timestamp")["aggregate"]
               .resample(freq).mean() / 1000.0)          # W -> kW

        frames.append(pd.DataFrame({
            "timestamp": s.index,
            "consumer_id": consumer_id,
            "consumption": s.to_numpy(),
        }))
        if i % 5 == 0 or i == len(csvs):
            print(f"  {i}/{len(csvs)} houses processed")

    long_df = pd.concat(frames, ignore_index=True).dropna(subset=["consumption"])

    n_before = len(long_df)
    long_df = long_df.groupby(["timestamp", "consumer_id"], as_index=False)["consumption"].mean()
    if len(long_df) < n_before:
        print(f"  merged {n_before - len(long_df):,} duplicate (timestamp, consumer_id) rows")
    return long_df


def to_wide(long_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    wide = long_df.pivot(index="timestamp", columns="consumer_id", values="consumption")
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq=freq))
    wide.index.name = "time"
    return wide


def mask_zero_runs(wide: pd.DataFrame, min_run: int, eps: float = 1e-9) -> tuple[pd.DataFrame, int]:
    """Turn long runs of exact zeros into NaN.

    REFIT zero-fills gaps > 2 min, so a household drawing exactly 0 W for hours
    is an outage, not real data. Leaving them as zeros would let the
    dense-rectangle search treat fabricated data as valid."""
    out = wide.copy()
    total = 0
    for col in out.columns:
        v = out[col].to_numpy(dtype=float)
        is_zero = np.abs(np.nan_to_num(v, nan=1.0)) <= eps      # NaN is not zero
        if not is_zero.any():
            continue
        change = np.diff(np.concatenate(([0], is_zero.astype(int), [0])))
        starts, ends = np.where(change == 1)[0], np.where(change == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_run:
                v[s:e] = np.nan
                total += e - s
        out[col] = v
    return out, total


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def client_periods(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in wide.columns:
        s = wide[col]
        valid = s.notna()
        if not valid.any():
            continue
        first_idx, last_idx = valid.idxmax(), valid[::-1].idxmax()
        span = s.loc[first_idx:last_idx]
        is_nan = span.isna().to_numpy()
        if is_nan.any():
            change = np.diff(np.concatenate(([0], is_nan.astype(int), [0])))
            starts, ends = np.where(change == 1)[0], np.where(change == -1)[0]
            longest_gap = int((ends - starts).max()) if len(starts) else 0
        else:
            longest_gap = 0
        rows.append({
            "consumer_id": col,
            "start": first_idx, "end": last_idx,
            "duration_days": round((last_idx - first_idx).total_seconds() / 86400, 1),
            "n_valid": int(span.notna().sum()),
            "n_total_in_span": len(span),
            "coverage_pct": round(span.notna().mean() * 100, 1),
            "longest_gap_steps": longest_gap,
        })
    return pd.DataFrame(rows).sort_values("duration_days", ascending=False).reset_index(drop=True)


def _rect_arrays(wide: pd.DataFrame):
    mask = wide.isna().to_numpy()
    T, C = mask.shape
    cum = np.vstack([np.zeros(C, dtype=np.int64), np.cumsum(mask, axis=0)])
    valid = ~mask
    has_any = valid.any(axis=0)
    first = np.argmax(valid, axis=0)
    last = T - 1 - np.argmax(valid[::-1], axis=0)
    return cum, np.unique(first[has_any]), np.unique(last[has_any])


def best_rectangle(wide: pd.DataFrame, min_timesteps: int):
    cum, starts, ends = _rect_arrays(wide)
    best = (0, -1, [], 0)
    for s in starts:
        for e in ends:
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
            "Lower --min-timesteps, or check --pareto for what IS achievable."
        )
    return best[0], best[1], best[2]


def pareto_rectangles(wide: pd.DataFrame) -> pd.DataFrame:
    cum, starts, ends = _rect_arrays(wide)
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

    if not candidates:
        raise SystemExit("No dense rectangle at all — every client has NaN everywhere.")

    df = pd.DataFrame(candidates, columns=["start_idx", "end_idx", "n_steps", "n_clients"])
    df = df.sort_values("n_steps", ascending=False).drop_duplicates("n_clients")
    df = df.sort_values("n_clients").reset_index(drop=True)

    best_so_far, keep_rows = -1, []
    for i in range(len(df) - 1, -1, -1):
        if df.loc[i, "n_steps"] > best_so_far:
            keep_rows.append(i)
            best_so_far = df.loc[i, "n_steps"]
    df = df.loc[sorted(keep_rows)].reset_index(drop=True)

    step = wide.index[1] - wide.index[0]                 # real grid step, not assumed
    df["duration_days"] = (df["n_steps"] * step / pd.Timedelta("1D")).round(2)
    df["date_start"] = df["start_idx"].map(lambda i: wide.index[i].strftime("%Y-%m-%d"))
    df["date_end"] = df["end_idx"].map(lambda i: wide.index[i].strftime("%Y-%m-%d"))
    return df[["n_clients", "n_steps", "duration_days", "date_start", "date_end",
               "start_idx", "end_idx"]]


def scan_tradeoff(wide: pd.DataFrame, thresholds: list[int], ctx: int, horizon: int) -> None:
    cum, starts, ends = _rect_arrays(wide)
    step = wide.index[1] - wide.index[0]
    mean_gap = 60          # mean of cutoff_gap_min=24 / gap_max=96 in refit.yaml

    print(f"{'min_timesteps':>14} {'~days':>8} {'n_clients':>10} {'n_steps':>10} "
          f"{'date_start':>12} {'date_end':>12} {'~test_cutoffs':>14}")
    for thr in thresholds:
        best, best_clients = (0, -1, 0), 0
        for s in starts:
            for e in ends:
                if e - s + 1 < thr:
                    continue
                keep = np.where((cum[e + 1] - cum[s]) == 0)[0]
                if keep.size == 0:
                    continue
                area = keep.size * (e - s + 1)
                if area > best[2]:
                    best, best_clients = (int(s), int(e), area), keep.size
        if best[1] < 0:
            print(f"{thr:>14} {'—':>8} {'—':>10} {'no dense block':>10}")
            continue
        s, e = best[0], best[1]
        n_steps = e - s + 1
        test_block = 0.15 * n_steps
        cutoffs = max(0, int((test_block - ctx - horizon) / mean_gap) + 1) if test_block >= ctx + horizon else 0
        print(f"{thr:>14} {n_steps * step / pd.Timedelta('1D'):>8.0f} {best_clients:>10} "
              f"{n_steps:>10} {wide.index[s].strftime('%Y-%m-%d'):>12} "
              f"{wide.index[e].strftime('%Y-%m-%d'):>12} {cutoffs:>14}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", default="data/raw/Refit_Dataset/archive.zip")
    p.add_argument("--outdir", default="data/refit")
    p.add_argument("--name", default="refit")
    p.add_argument("--freq", default="30min", help="resample target (REFIT native ~6-8s)")

    p.add_argument("--inspect", action="store_true", help="describe the archive and exit")
    p.add_argument("--periods", action="store_true", help="per-client coverage table, then exit")
    p.add_argument("--periods-csv", default=None)
    p.add_argument("--scan", action="store_true", help="clients-vs-duration trade-off, then exit")
    p.add_argument("--scan-values", default="4320,8640,17520,26280,35040",
                   help="min-timesteps candidates at 30min: ~3mo,6mo,1y,1.5y,2y")
    p.add_argument("--pareto", action="store_true", help="all non-dominated rectangles, then exit")

    p.add_argument("--zero-run-to-nan", type=int, default=6,
                   help="turn runs of >=N consecutive zero steps into NaN "
                        "(at 30min, 6 = 3h). REFIT zero-fills outages. 0 disables.")
    p.add_argument("--interpolate-limit", type=int, default=2,
                   help="linearly fill interior gaps up to N steps (30min: 2 = 1h). 0 disables.")

    p.add_argument("--min-timesteps", type=int, default=17520, help="17520 = 1yr @ 30min")
    p.add_argument("--target-clients", type=int, default=None,
                   help="materialize the --pareto row with exactly this many clients")
    p.add_argument("--ratios", default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-split", action="store_true")
    args = p.parse_args()

    archive_path = Path(args.archive)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        inspect(archive_path)
        return

    csvs = ensure_extracted(archive_path, outdir / "_extracted")
    print("Reading + resampling (Aggregate only)...")
    long_df = read_refit(csvs, args.freq)
    print(f"  {len(long_df):,} rows, {long_df['consumer_id'].nunique()} houses")

    wide = to_wide(long_df, args.freq)
    print(f"Wide: {wide.shape[0]:,} steps x {wide.shape[1]} clients "
          f"({wide.isna().to_numpy().mean():.1%} NaN)")

    if args.zero_run_to_nan > 0:
        wide, n_masked = mask_zero_runs(wide, args.zero_run_to_nan)
        print(f"Masked {n_masked:,} values in zero-runs >= {args.zero_run_to_nan} steps "
              f"-> {wide.isna().to_numpy().mean():.1%} NaN")

    if args.interpolate_limit > 0:
        before = int(wide.isna().to_numpy().sum())
        wide = wide.interpolate(method="linear", limit=args.interpolate_limit,
                                limit_area="inside")
        print(f"Interpolated gaps <= {args.interpolate_limit} steps: "
              f"{before - int(wide.isna().to_numpy().sum()):,} values filled")

    if args.periods:
        table = client_periods(wide)
        pd.set_option("display.max_rows", None); pd.set_option("display.width", 220)
        print(table.to_string(index=False))
        if args.periods_csv:
            table.to_csv(args.periods_csv, index=False)
            print(f"\n-> {args.periods_csv}")
        return

    if args.scan:
        scan_tradeoff(wide, [int(x) for x in args.scan_values.split(",")], ctx=1440, horizon=96)
        return

    if args.pareto:
        pd.set_option("display.max_rows", None)
        print(pareto_rectangles(wide).to_string(index=False))
        return

    if args.target_clients is not None:
        table = pareto_rectangles(wide)
        row = table[table["n_clients"] == args.target_clients]
        if row.empty:
            raise SystemExit(f"No rectangle with exactly {args.target_clients} clients. "
                             f"Available: {table['n_clients'].tolist()}")
        s, e = int(row.iloc[0]["start_idx"]), int(row.iloc[0]["end_idx"])
        keep = wide.columns[wide.isna().to_numpy()[s:e + 1].sum(axis=0) == 0]
        wide = wide.iloc[s:e + 1][keep]
    else:
        s, e, clients = best_rectangle(wide, args.min_timesteps)
        wide = wide.iloc[s:e + 1][clients]

    assert not wide.isna().to_numpy().any()
    print(f"Kept {wide.shape[1]} clients x {wide.shape[0]:,} steps "
          f"({wide.index[0]} -> {wide.index[-1]})")

    out_pq = outdir / f"{args.name}_load_curve.parquet"
    wide.reset_index().to_parquet(out_pq, index=False)      # `time` as a column
    print(f"-> {out_pq}")

    out_pkl = outdir / f"{args.name}_train_valid_test_id_split.pkl"
    if out_pkl.exists() and not args.force_split:
        with open(out_pkl, "rb") as f:
            split = pickle.load(f)
        known = set(split["train"]) | set(split["val"]) | set(split["test"])
        if known != set(wide.columns):
            raise SystemExit(
                f"{out_pkl} exists but covers a DIFFERENT client set ({len(known)} ids) "
                f"than this run ({wide.shape[1]} ids). Re-running with different "
                "preparation arguments changes which clients survive, invalidating any "
                "REFIT results already produced. Pass --force-split to regenerate."
            )
        print(f"Reusing existing split -> {out_pkl}")
    else:
        r_train, r_val, _ = (float(x) for x in args.ratios.split(","))
        ids = sorted(wide.columns.tolist())
        random.Random(args.seed).shuffle(ids)
        n_tr, n_va = int(round(r_train * len(ids))), int(round(r_val * len(ids)))
        split = {"train": ids[:n_tr], "val": ids[n_tr:n_tr + n_va], "test": ids[n_tr + n_va:]}
        with open(out_pkl, "wb") as f:
            pickle.dump(split, f)
        print(f"Created split -> {out_pkl}")
    print(f"  {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test")


if __name__ == "__main__":
    main()