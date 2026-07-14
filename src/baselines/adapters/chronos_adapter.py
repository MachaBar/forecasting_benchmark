import numpy as np
import torch
from .base import ForecastAdapter


class ChronosAdapter(ForecastAdapter):
    name = "Chronos2"

    def load(self):
        try:
            from chronos.chronos2 import Chronos2Pipeline
        except ImportError:
            from chronos import ChronosPipeline as Chronos2Pipeline

        dtype = getattr(torch, self.cfg.torch_dtype)
        base_kwargs = dict(device_map=self.cfg.device_map, local_files_only=True)
        # API changed across versions: dtype kwarg name differs / may be unsupported
        for extra in ({"dtype": dtype}, {"torch_dtype": dtype}, {}):
            try:
                self.pipe = Chronos2Pipeline.from_pretrained(
                    self.cfg.weights_path, **base_kwargs, **extra
                )
                return
            except TypeError:
                continue
        # last resort: no device_map either
        self.pipe = Chronos2Pipeline.from_pretrained(self.cfg.weights_path)

    def predict(self, contexts, prediction_length, quantile_levels):
        ctx = [torch.tensor(contexts[i]) for i in range(contexts.shape[0])]
        qf_result = self.pipe.predict_quantiles(
            ctx, prediction_length=prediction_length, quantile_levels=quantile_levels,
        )
        qf = qf_result[0] if isinstance(qf_result, (tuple, list)) else qf_result
        qf = qf.numpy() if isinstance(qf, torch.Tensor) else np.asarray(qf)
        # (b, 1, H, Q) → (b, H, Q)
        return qf[:, 0, :, :]