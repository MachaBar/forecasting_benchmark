# src/baselines/adapters/patchtst_fm_adapter.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import ForecastAdapter


class PatchTSTFMAdapter(ForecastAdapter):
    name = "PatchTST-FM"

    def load(self):
        from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline

        model_name = self.cfg.model_name
        # Resolve local checkpoint paths relative to repo root; leave HF repo ids untouched.
        if not model_name.startswith(("http://", "https://")) and "/" in model_name:
            import hydra
            candidate = Path(hydra.utils.to_absolute_path(model_name))
            if candidate.exists():
                model_name = str(candidate)

        model = PatchTSTFMForPrediction.from_pretrained(model_name)

        self._pipeline_cls = TimeSeriesForecastingPipeline
        self._model = model
        self._timestamp_col = "timestamp"
        self._max_context_length = self.cfg.get("max_context_length", 8192)
        self._impute_method = self.cfg.get("impute_method", None)
        self._device = self.cfg.get("device", "cuda")

    def predict(self, contexts: np.ndarray, prediction_length: int, quantile_levels: list[float]) -> np.ndarray:
        # contexts: (n, L) — one row per series, evaluated independently (no id_columns
        # sharing), matching PatchTST-FM's example usage (single-series pipeline calls).
        n, L = contexts.shape

        pipe = self._pipeline_cls(
            model=self._model,
            id_columns=[],
            timestamp_column=self._timestamp_col,
            target_columns=["value"],
            max_context_length=min(self._max_context_length, L + 100),
            context_length=L,
            prediction_length=prediction_length,
            batch_size=n,
            impute_method=self._impute_method,
            device=self._device,
            quantile_levels=quantile_levels,
        )

        # Synthetic timestamps (only relative order matters for the pipeline's internal logic)
        dummy_dates = pd.date_range("2000-01-01", periods=L, freq="30min")

        all_q = np.zeros((n, prediction_length, len(quantile_levels)))
        for i in range(n):
            df_hist = pd.DataFrame({self._timestamp_col: dummy_dates, "value": contexts[i]})
            forecast_result = pipe(df_hist)

            for j, q in enumerate(quantile_levels):
                col_name = f"value_{q:.2f}"
                if col_name in forecast_result.columns:
                    all_q[i, :, j] = forecast_result[col_name].values[-prediction_length:]
                elif "value_0.50" in forecast_result.columns:
                    all_q[i, :, j] = forecast_result["value_0.50"].values[-prediction_length:]

        return all_q  # (n, H, Q)