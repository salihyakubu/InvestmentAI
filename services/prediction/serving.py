"""Model inference serving layer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from services.prediction.models.base import BasePredictor, PredictionOutput
from services.prediction.models.ensemble import EnsemblePredictor
from services.prediction.registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelServer:
    """Loads active models from the registry and serves predictions
    through an :class:`EnsemblePredictor`.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        model_weights: dict[str, float] | None = None,
        min_agreement: int = 3,
    ) -> None:
        self.registry = registry
        self._model_weights = model_weights
        self._min_agreement = min_agreement
        self._models: dict[str, BasePredictor] = {}
        self._ensemble: EnsemblePredictor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_active_models(self) -> dict[str, BasePredictor]:
        """Load all active model versions from the registry.

        Returns:
            Mapping of model_type -> loaded predictor.
        """
        loaded: dict[str, BasePredictor] = {}

        import os
        worker_mode = os.environ.get("WORKER_MODE", "full")
        model_types = ["xgboost", "lightgbm"]
        if worker_mode == "full":
            model_types.extend(["lstm", "transformer"])

        for model_type in model_types:
            try:
                model, meta = self.registry.get_active(model_type)
                loaded[model_type] = model
                logger.info(
                    "Loaded active %s model: %s v%d",
                    model_type,
                    meta.model_name,
                    meta.version,
                )
            except ValueError:
                logger.warning("No active model for type '%s'", model_type)

        self._models = loaded

        if loaded:
            self._ensemble = EnsemblePredictor(
                models=list(loaded.values()),
                weights=self._model_weights,
                min_agreement=self._min_agreement,
            )
        else:
            self._ensemble = None
            logger.warning("No models loaded -- predictions will return flat")

        return loaded

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        symbol: str,
        features_flat: np.ndarray | None = None,
        features_sequence: np.ndarray | None = None,
    ) -> PredictionOutput:
        """Run inference through the ensemble.

        Args:
            symbol: ticker symbol (for logging).
            features_flat: 1-D / 2-D array for tree-based models.
            features_sequence: 2-D / 3-D array for sequence models.

        Returns:
            Combined :class:`PredictionOutput`.
        """
        if self._ensemble is None:
            logger.warning("No ensemble available for %s, returning flat", symbol)
            return PredictionOutput(
                direction="flat",
                confidence=0.0,
                expected_return=0.0,
                probabilities={"long": 0.0, "short": 0.0, "flat": 1.0},
            )

        output = self._ensemble.predict(
            features_flat=features_flat,
            features_sequence=features_sequence,
        )

        logger.debug(
            "Prediction for %s: direction=%s confidence=%.4f return=%.6f",
            symbol,
            output.direction,
            output.confidence,
            output.expected_return,
        )
        return output

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, recent_performance: dict[str, float]) -> None:
        """Propagate recent performance metrics to the ensemble for
        dynamic weight adjustment.
        """
        if self._ensemble is not None:
            self._ensemble.update_weights(recent_performance)

    @property
    def active_model_types(self) -> list[str]:
        return list(self._models.keys())
