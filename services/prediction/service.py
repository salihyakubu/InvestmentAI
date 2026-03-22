"""Main prediction service -- subscribes to feature events and publishes predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.enums import SignalDirection
from core.events.base import EventBus
from core.events.signal_events import FeaturesReadyEvent, PredictionReadyEvent
from services.prediction.models.base import PredictionOutput
from services.prediction.registry import ModelMetadata
from services.prediction.serving import ModelServer

logger = logging.getLogger(__name__)

# Stream / topic constants
_FEATURES_STREAM = "features.ready"
_PREDICTIONS_STREAM = "predictions.ready"
_CONSUMER_GROUP = "prediction-service"


@dataclass
class Prediction:
    """Internal prediction record stored for auditing."""

    symbol: str
    direction: str
    confidence: float
    expected_return: float
    probabilities: dict[str, float]
    model_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PredictionService:
    """Connects the feature pipeline to model inference and publishes results.

    Lifecycle:
        1. ``start()`` -- subscribe to :class:`FeaturesReadyEvent`.
        2. On each event, run :meth:`predict` and publish a
           :class:`PredictionReadyEvent` via the event bus.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Any = None,
        model_server: ModelServer | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._model_server = model_server
        self._prediction_history: list[Prediction] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load active models and begin consuming feature events."""
        if self._model_server is not None:
            self._model_server.load_active_models()

        self._running = True
        logger.info("PredictionService started")

        # Begin consuming from the features stream
        await self._event_bus.subscribe(
            stream=_FEATURES_STREAM,
            group=_CONSUMER_GROUP,
            consumer="prediction-worker",
            handler=self.handle_features_ready,
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("PredictionService stopped")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def handle_features_ready(self, event: FeaturesReadyEvent) -> None:
        """Process a :class:`FeaturesReadyEvent`.

        Runs prediction, records the result, and publishes a
        :class:`PredictionReadyEvent`.
        """
        symbol = event.symbol
        feature_vector = event.feature_vector  # dict[str, float]

        try:
            prediction = self.predict(symbol, feature_vector)

            # Publish downstream
            pred_event = PredictionReadyEvent(
                symbol=symbol,
                direction=prediction.direction,
                confidence=prediction.confidence,
                expected_return=prediction.expected_return,
                model_id=prediction.model_id,
                source_service="prediction-service",
            )
            await self._event_bus.publish(_PREDICTIONS_STREAM, pred_event)

            logger.info(
                "Published prediction for %s: %s (conf=%.4f)",
                symbol,
                prediction.direction,
                prediction.confidence,
            )
        except Exception:
            logger.exception("Failed to produce prediction for %s", symbol)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, symbol: str, features: dict[str, float]) -> Prediction:
        """Synchronous prediction entry-point.

        Args:
            symbol: ticker.
            features: feature name -> value mapping.

        Returns:
            :class:`Prediction` record.
        """
        if self._model_server is None:
            output = PredictionOutput(
                direction=SignalDirection.FLAT,
                confidence=0.0,
                expected_return=0.0,
                probabilities={"long": 0.0, "short": 0.0, "flat": 1.0},
            )
        else:
            # Build flat feature vector (sorted keys for deterministic ordering)
            sorted_keys = sorted(features.keys())
            features_flat = np.array([features[k] for k in sorted_keys], dtype=np.float64)

            # For sequence models we would need buffered history; pass None for now
            output = self._model_server.predict(
                symbol=symbol,
                features_flat=features_flat,
                features_sequence=None,
            )

        active_types = (
            ",".join(self._model_server.active_model_types)
            if self._model_server
            else "none"
        )

        prediction = Prediction(
            symbol=symbol,
            direction=output.direction,
            confidence=output.confidence,
            expected_return=output.expected_return,
            probabilities=output.probabilities,
            model_id=f"ensemble:{active_types}",
        )
        self._prediction_history.append(prediction)
        return prediction

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_models(self) -> list[ModelMetadata]:
        """Return metadata for all currently active model versions."""
        if self._model_server is None:
            return []
        metas: list[ModelMetadata] = []
        for model_type in self._model_server.active_model_types:
            try:
                _, meta = self._model_server.registry.get_active(model_type)
                metas.append(meta)
            except ValueError:
                pass
        return metas

    @property
    def recent_predictions(self) -> list[Prediction]:
        """Return the last 100 predictions."""
        return self._prediction_history[-100:]
