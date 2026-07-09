from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.dataset import (
    client_ids_to_indices, eval_batch, load_client_split_pickle,
    load_dataset, make_cutoffs,
)
from src.utils.metrics import compute_metrics
from .adapters.base import ForecastAdapter


def run_foundation_eval(cfg, adapter: ForecastAdapter, output_dir: Path,
                        save_plot_fn=None) -> pd.DataFrame:
    data_path = cfg._resolve(cfg.dataset.path)          # helper de résolution
    split_path = cfg._resolve(cfg.dataset.path_client_split)

    ts = load_dataset(data_path, layout=cfg.dataset.layout, date_col=cfg.dataset.timestamp_col)
    split = load_client_split_pickle(split_path)
    test_indices = client_ids_to_indices(ts, split["test"])

    splits = make_cutoffs(
        ts, lags=cfg.dataset.context_length, horizon=cfg.dataset.prediction_length,
        step_size=cfg.dataset.stride, ratios=cfg.dataset.get("ratios", "0.7,0.15,0.15"),
    )
    cutoffs = splits["test_cutoffs"].tolist()

    is_prob = cfg.model.get("probabilistic", False)
    quantile_levels = sorted(set([0.1, 0.5, 0.9] +
        (list(cfg.model.get("quantile_levels", [])) if is_prob else [])))
    batch_size = cfg.model.get("batch_size", 32)
    season_length = cfg.model.get("season_length", cfg.dataset.get("season_length", 48))
    ctx_len = cfg.dataset.context_length
    H = cfg.dataset.prediction_length

    adapter.load()
    # ... print scope, save eval_info.json (identique à aujourd'hui) ...

    checkpoint_path = output_dir / "results_checkpoint.csv"
    rows, plot_saved = [], False

    for ci, cutoff in enumerate(cutoffs):
        batch = eval_batch(ts, int(cutoff), lags=ctx_len, horizon=H, users=test_indices)
        x = batch["x"]
        n = x.shape[0]

        # mini-batching commun — l'adapter ne voit qu'un bloc de contextes
        qparts = []
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            contexts = x[s:e, 0, :].numpy()               # (b, L)
            qparts.append(adapter.predict(contexts, H, quantile_levels))  # (b, H, Q)
        all_q = np.concatenate(qparts, axis=0)            # (n, H, Q)

        median_idx = quantile_levels.index(0.5)
        cutoff_rows = []
        for i, uid in enumerate(batch["item_ids"]):
            y_true = batch["y"][i, 0].numpy()
            ctx = x[i, 0].numpy()
            hist = ts.values[: cutoff + ctx_len, test_indices[i]]
            y_pred = all_q[i, :, median_idx]
            qpreds = {q: all_q[i, :, j] for j, q in enumerate(quantile_levels)} if is_prob else None
            m = compute_metrics(y_true, y_pred, context=ctx, history=hist,
                                season_length=season_length,
                                include_mape=cfg.model.get("include_mape", False),
                                quantile_preds=qpreds)
            cutoff_rows.append({"unique_id": uid, "cutoff": cutoff, "model": adapter.name,
                                "context_length": ctx_len, "prediction_length": H,
                                **m.to_dict()})
            if save_plot_fn and ci == 0 and i == 0 and not plot_saved:
                save_plot_fn(context=ctx, y_true=y_true, quantiles=all_q[i],
                             quantile_levels=quantile_levels, uid=uid,
                             cutoff=cutoff, output_dir=output_dir)
                plot_saved = True

        rows.extend(cutoff_rows)
        pd.DataFrame(cutoff_rows).to_csv(checkpoint_path, mode="a",
                                         header=not checkpoint_path.exists(), index=False)

    # ... save results_per_client_cutoff.csv, summary, summary_all_runs (identique) ...
    return pd.DataFrame(rows)