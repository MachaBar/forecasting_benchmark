import numpy as np
import torch
from .base import ForecastAdapter


class TSICLAdapter(ForecastAdapter):
    name = "TSICL"

    def load(self):
        from tsicl import TSICL
        self.model = TSICL(
            model_path=self.cfg.weights_path,
            allow_auto_download=self.cfg.get("allow_auto_download", False),
        )

    def predict(self, contexts, prediction_length, quantile_levels):
        # contexts: (b, L)  →  quantiles: (b, H, Q)
        b = contexts.shape[0]                      # ← la taille de batch CONNUE
        _, batch_q = self.model.forecast(
            inputs=contexts,                       # [N, L] — batché nativement
            prediction_length=prediction_length,
            context_length=contexts.shape[1],
            batch_size=b,
            device=self.cfg.get("device", "cuda"),
            quantile_levels=quantile_levels,
            point_estimator=self.cfg.get("point_estimator", "median"),
            denormalize=True,
        )
        q = batch_q.detach().cpu().numpy() if isinstance(batch_q, torch.Tensor) else np.asarray(batch_q)
        # (N,C,H,Q) ou (N,H,Q) ou (H,Q) si squeezé → on force (b, H, Q)
        return q.reshape(b, prediction_length, len(quantile_levels))