from __future__ import annotations
import numpy as np
from omegaconf import DictConfig


class ForecastAdapter:
    """Un modèle de fondation, réduit à ce qui le distingue des autres."""

    name: str = "foundation"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def load(self) -> None:
        """Charge les poids. Appelé une fois."""
        raise NotImplementedError

    def predict(
        self, contexts: np.ndarray, prediction_length: int, quantile_levels: list[float]
    ) -> np.ndarray:
        """contexts: (n, L)  →  quantiles: (n, H, Q).."""
        raise NotImplementedError