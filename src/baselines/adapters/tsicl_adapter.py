import numpy as np
import torch
from .base import ForecastAdapter


class TSICLAdapter(ForecastAdapter):
    name = "TSICL"

    def load(self):
        from tsicl import TSICL
        self.model = TSICL(model_path=self.cfg.weights_path, allow_auto_download=False)

    def predict(self, contexts, prediction_length, quantile_levels):
        _, batch_q = self.model.forecast(
            inputs=contexts,
            prediction_length=prediction_length,
            context_length=contexts.shape[1],
            batch_size=contexts.shape[0],
            device=self.cfg.get("device", "cuda"),
            quantile_levels=quantile_levels,
            point_estimator=self.cfg.get("point_estimator", "median"),
            denormalize=True,
        )
        q = batch_q.detach().cpu().numpy() if isinstance(batch_q, torch.Tensor) else np.asarray(batch_q)
        # (N, C, H, Q) → (N, H, Q)
        return q.reshape(q.shape[0], prediction_length, len(quantile_levels))