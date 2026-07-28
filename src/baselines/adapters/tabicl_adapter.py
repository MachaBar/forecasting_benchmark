from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastAdapter


class TabICLAdapter(ForecastAdapter):
    """TabICL-Forecast — tabular foundation model (TabICLForecaster) applied to
    time series via in-context learning. API mirrors TabPFN-TS (same lab, INRIA-Soda).
    Processes each mini-batch as one multi-series panel (all clients forecast in
    a single predict_df call, distinguished by item_id).
    """

    name = "TabICL"

    def load(self) -> None:
        from tabicl import TabICLForecaster

        self._forecaster = TabICLForecaster(
            max_context_length=self.cfg.get("max_context_length", 4096),
            point_estimate=self.cfg.get("point_estimate", "mean"),
        )

    def predict(self, contexts: np.ndarray, prediction_length: int, quantile_levels: list[float]) -> np.ndarray:
        # contexts: (n, L)
        n, L = contexts.shape

        # Build one long-format panel DataFrame for all series in this mini-batch.
        # Synthetic regular timestamps: only relative spacing matters for the model.
        dummy_dates = pd.date_range("2000-01-01", periods=L, freq="30min")
        frames = [
            pd.DataFrame({
                "item_id": str(i),
                "timestamp": dummy_dates,
                "target": contexts[i],
            })
            for i in range(n)
        ]
        context_df = pd.concat(frames, ignore_index=True)

        pred_df = self._forecaster.predict_df(
            context_df=context_df,
            prediction_length=prediction_length,
            quantiles=quantile_levels,
        )
        # Indexed by (item_id, timestamp); reset for straightforward slicing.
        pred_df = pred_df.reset_index()

        all_q = np.zeros((n, prediction_length, len(quantile_levels)))
        for i in range(n):
            series_pred = pred_df[pred_df["item_id"] == str(i)].sort_values("timestamp")
            for j, q in enumerate(quantile_levels):
                col = q if q in series_pred.columns else str(q)
                if col in series_pred.columns:
                    all_q[i, :, j] = series_pred[col].values[:prediction_length]
                else:
                    # Fallback: point forecast if a specific quantile column is missing
                    all_q[i, :, j] = series_pred["target"].values[:prediction_length]

        return all_q  # (n, H, Q)