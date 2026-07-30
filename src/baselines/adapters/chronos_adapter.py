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
                break
            except TypeError:
                continue
        else:
            # last resort: no device_map either
            self.pipe = Chronos2Pipeline.from_pretrained(self.cfg.weights_path)

        self._setup_cross_learning()

    def _setup_cross_learning(self):
        """Résout le mode de partage entre séries, une fois après le chargement.

        predict_quantiles(inputs, prediction_length=None, quantile_levels=[...],
        **predict_kwargs) -> (quantiles, mean), des LISTES (une entrée par
        tâche d'`inputs`), chaque élément de forme (n_variates, H, Q) / (n_variates, H).

        cross_learning n'apparaît jamais comme paramètre nommé de
        predict_quantiles (absorbé par **predict_kwargs), mais la docstring
        de predict_df — qui délègue via
        self.predict_quantiles(..., cross_learning=cross_learning, ...) —
        confirme que c'est un kwarg réel et documenté 
        """
        self.cross_learning = bool(self.cfg.get("cross_learning", False))
        self.meta = {"cross_learning": self.cross_learning}
        if self.cross_learning:
            self.name = "Chronos2-CL"
        print(f"[{self.name}] cross-learning: {'on' if self.cross_learning else 'off'}")

    def predict(self, contexts, prediction_length, quantile_levels):
        if not hasattr(self, "meta"):
            self._setup_cross_learning()

        n = contexts.shape[0]
        ctx = [torch.tensor(contexts[i]) for i in range(n)]

        extra = {"cross_learning": True} if self.cross_learning else {}
        quantiles, _ = self._call(ctx, prediction_length, quantile_levels, **extra)

        # une tâche = un client, forme (1, H, Q) → (H, Q), puis empile
        return np.stack([q[0] for q in quantiles], axis=0)

    def _call(self, ctx, prediction_length, quantile_levels, **extra):
        quantiles, mean = self.pipe.predict_quantiles(
            ctx, prediction_length=prediction_length,
            quantile_levels=quantile_levels, **extra,
        )
        quantiles = [q.numpy() if isinstance(q, torch.Tensor) else np.asarray(q) for q in quantiles]
        mean = [m.numpy() if isinstance(m, torch.Tensor) else np.asarray(m) for m in mean]
        return quantiles, mean


# import numpy as np
# import torch
# from .base import ForecastAdapter


# class ChronosAdapter(ForecastAdapter):
#     name = "Chronos2"

#     def load(self):
#         try:
#             from chronos.chronos2 import Chronos2Pipeline
#         except ImportError:
#             from chronos import ChronosPipeline as Chronos2Pipeline

#         dtype = getattr(torch, self.cfg.torch_dtype)
#         base_kwargs = dict(device_map=self.cfg.device_map, local_files_only=True)
#         # API changed across versions: dtype kwarg name differs / may be unsupported
#         for extra in ({"dtype": dtype}, {"torch_dtype": dtype}, {}):
#             try:
#                 self.pipe = Chronos2Pipeline.from_pretrained(
#                     self.cfg.weights_path, **base_kwargs, **extra
#                 )
#                 return
#             except TypeError:
#                 continue
#         # last resort: no device_map either
#         self.pipe = Chronos2Pipeline.from_pretrained(self.cfg.weights_path)

#     def predict(self, contexts, prediction_length, quantile_levels):
#         ctx = [torch.tensor(contexts[i]) for i in range(contexts.shape[0])]
#         qf_result = self.pipe.predict_quantiles(
#             ctx, prediction_length=prediction_length, quantile_levels=quantile_levels,predict_batches_jointly = True
#         )
#         qf = qf_result[0] if isinstance(qf_result, (tuple, list)) else qf_result
#         qf = qf.numpy() if isinstance(qf, torch.Tensor) else np.asarray(qf)
#         # (b, 1, H, Q) → (b, H, Q)
#         return qf[:, 0, :, :] # batches, for each horizon step we have 10 quantiles