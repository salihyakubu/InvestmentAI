"""Ensemble predictor that combines multiple model predictions."""

from __future__ import annotations

import logging

import numpy as np

from services.prediction.models.base import BasePredictor, PredictionOutput

logger = logging.getLogger(__name__)

SEQUENCE_MODELS = {"lstm", "transformer"}
FLAT_MODELS = {"xgboost", "lightgbm"}


class EnsemblePredictor:
    """Weighted ensemble of multiple :class:`BasePredictor` instances.

    This is *not* a :class:`BasePredictor` subclass because it delegates
    to wrapped models rather than implementing a single model lifecycle.
    """

    def __init__(
        self,
        models: list[BasePredictor],
        weights: dict[str, float] | None = None,
        min_agreement: int = 3,
    ) -> None:
        self.models = models
        self._weights = weights or self._default_weights()
        self.min_agreement = min_agreement

    def _default_weights(self) -> dict[str, float]:
        n = len(self.models)
        return {m.model_type: 1.0 / n for m in self.models}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features_flat: np.ndarray | None = None,
        features_sequence: np.ndarray | None = None,
    ) -> PredictionOutput:
        """Run each wrapped model and combine predictions.

        Args:
            features_flat: 1-D or 2-D array for tree-based models.
            features_sequence: 2-D or 3-D array for sequence models.

        Returns:
            Weighted-average :class:`PredictionOutput`.
        """
        individual_outputs: list[tuple[str, PredictionOutput, float]] = []

        for model in self.models:
            model_type = model.model_type
            weight = self._weights.get(model_type, 0.0)
            if weight <= 0:
                continue

            try:
                if model_type in SEQUENCE_MODELS:
                    if features_sequence is None:
                        logger.warning("No sequence features for %s, skipping", model_type)
                        continue
                    output = model.predict(features_sequence)
                else:
                    if features_flat is None:
                        logger.warning("No flat features for %s, skipping", model_type)
                        continue
                    output = model.predict(features_flat)
                individual_outputs.append((model_type, output, weight))
            except Exception:
                logger.exception("Model %s failed during prediction", model_type)

        if not individual_outputs:
            return PredictionOutput(
                direction="flat",
                confidence=0.0,
                expected_return=0.0,
                probabilities={"long": 0.0, "short": 0.0, "flat": 1.0},
            )

        return self._combine(individual_outputs)

    def _combine(
        self,
        outputs: list[tuple[str, PredictionOutput, float]],
    ) -> PredictionOutput:
        """Weighted combination of individual predictions."""
        total_weight = sum(w for _, _, w in outputs)

        # Weighted probability aggregation
        combined_probs = {"long": 0.0, "short": 0.0, "flat": 0.0}
        weighted_return = 0.0

        for _, output, weight in outputs:
            norm_w = weight / total_weight
            for direction, prob in output.probabilities.items():
                combined_probs[direction] += prob * norm_w
            weighted_return += output.expected_return * norm_w

        # Determine direction from combined probabilities
        best_direction = max(combined_probs, key=combined_probs.get)  # type: ignore[arg-type]
        confidence = combined_probs[best_direction]

        # Agreement check: require min_agreement models to agree for a
        # non-flat signal (prevents spurious signals)
        if best_direction != "flat":
            agreeing = sum(
                1 for _, out, _ in outputs if out.direction == best_direction
            )
            if agreeing < min(self.min_agreement, len(outputs)):
                best_direction = "flat"
                confidence = combined_probs["flat"]

        return PredictionOutput(
            direction=best_direction,
            confidence=confidence,
            expected_return=weighted_return,
            probabilities=combined_probs,
        )

    # ------------------------------------------------------------------
    # Dynamic weight updating
    # ------------------------------------------------------------------

    def update_weights(self, recent_performance: dict[str, float]) -> None:
        """Update model weights using softmax over recent accuracy.

        Args:
            recent_performance: mapping of model_type -> recent accuracy (0-1).
        """
        model_types = [m.model_type for m in self.models]
        scores = np.array([recent_performance.get(mt, 0.5) for mt in model_types])

        # Temperature-scaled softmax
        temperature = 0.5
        exp_scores = np.exp((scores - scores.max()) / temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        self._weights = {mt: float(w) for mt, w in zip(model_types, softmax_weights)}
        logger.info("Ensemble weights updated: %s", self._weights)
