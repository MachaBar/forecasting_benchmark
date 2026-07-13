import numpy as np
import torch
from .base import ForecastAdapter

# TiRex-2 outputs a FIXED grid of 9 quantiles
TIREX_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TiRexAdapter(ForecastAdapter):
    name = "TiRex2"

    def load(self):
        from tirex2 import load_model
        self.model = load_model(
            self.cfg.weights_path,                 # "NX-AI/TiRex-2" or a local path
            device=self.cfg.get("device", "cuda"),
        )

    def predict(self, contexts, prediction_length, quantile_levels):
        from tirex2 import TimeseriesType
        # One TimeseriesType per series (univariate → target shape (1, L))
        items = [
            TimeseriesType(
                target=torch.as_tensor(contexts[i], dtype=torch.float32).unsqueeze(0),
                past_covariates=None, future_covariates=None,
            )
            for i in range(contexts.shape[0])
        ]
        # forecast returns a list; each entry has shape (n_targets=1, 9, H)
        fc = self.model.forecast(
            items, prediction_length=prediction_length, output_type="numpy",
        )
        arr = np.stack([np.asarray(f) for f in fc], axis=0)   # (b, 1, 9, H)
        arr = arr[:, 0, :, :]                                  # (b, 9, H)
        arr = np.transpose(arr, (0, 2, 1))                    # (b, H, 9)

        # Select the requested quantile levels from TiRex's fixed grid
        idx = []
        for q in quantile_levels:
            match = next((j for j, tq in enumerate(TIREX_QUANTILES) if abs(tq - q) < 1e-6), None)
            if match is None:
                raise ValueError(
                    f"TiRex-2 only outputs quantiles {TIREX_QUANTILES}; "
                    f"requested {q} is unavailable. Set model.quantile_levels "
                    f"to a subset of that grid."
                )
            idx.append(match)
        return arr[:, :, idx]                                  # (b, H, Q)