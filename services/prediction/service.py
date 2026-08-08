"""Main prediction service -- subscribes to feature events and publishes predictions."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from core.enums import SignalDirection
from core.events.base import Event, EventBus
from core.events.signal_events import FeaturesReadyEvent, PredictionReadyEvent
from core.events.streams import SYSTEM
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


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
        calibrator: Any = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._model_server = model_server
        # Live p(flat) recalibration layer (GO_LIVE.md 2026-08-09). Applied
        # to the ensemble's output before gating and publishing; identity
        # when absent, disabled, or unfitted.
        self._calibrator = calibrator
        # Served-feature persistence cadence (GO_LIVE.md 2026-08-09 phase 2):
        # attach the ordered feature vector only on minutes divisible by this,
        # matching the de-overlap convention and bounding storage growth.
        self._feature_sample_minutes = 5
        # Bounded audit buffer: the only consumer reads the last 100, so an
        # unbounded list would just leak memory over a long soak.
        self._prediction_history: deque[Prediction] = deque(maxlen=1000)
        self._running = False
        # Minimum confidence to emit a non-flat (tradeable) signal; below this the
        # prediction abstains (flat). Wires up settings.min_prediction_confidence.
        self._min_confidence = float(getattr(settings, "min_prediction_confidence", 0.6))

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
            handler=self.handle_features_ready,  # type: ignore[arg-type]  # bus routes only FeaturesReadyEvent to this stream
        )

        # Hot-reload newly promoted model versions without a worker restart:
        # the continuous-learning service publishes ModelRetrainedEvent on the
        # SYSTEM stream after promoting a new version.
        await self._event_bus.subscribe(
            stream=SYSTEM,
            group=_CONSUMER_GROUP,
            consumer="prediction-reload-1",
            handler=self.handle_model_retrained,
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
            served_features, served_hash = self._sampled_features(feature_vector)

            # Publish downstream
            pred_event = PredictionReadyEvent(
                symbol=symbol,
                direction=prediction.direction,
                confidence=prediction.confidence,
                expected_return=prediction.expected_return,
                model_id=prediction.model_id,
                probabilities=prediction.probabilities,
                features=served_features,
                features_hash=served_hash,
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

    async def handle_model_retrained(self, event: Event) -> None:
        """Hot-reload active models when a new version is promoted.

        The SYSTEM stream also carries other platform events (e.g. drift), so
        anything but a ``ModelRetrainedEvent`` is ignored. Model
        deserialization blocks, so the reload runs in a thread; the serving
        layer swaps in the fully-built ensemble as its final step, so
        predictions keep using the previous ensemble until then.
        """
        if event.event_type != "ModelRetrainedEvent":
            return
        if self._model_server is None:
            return

        await asyncio.to_thread(self._model_server.load_active_models)
        logger.info(
            "Hot-reloaded models after ModelRetrainedEvent: active=%s",
            ",".join(self._model_server.active_model_types) or "none",
        )

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
            # Build the flat feature vector in the SAME column order the models
            # were trained on. Ordering by the persisted feature_names (not by
            # alphabetical sorting) is essential -- otherwise each model column
            # is fed a different feature at inference than during training.
            feature_order = self._model_server.feature_names
            if feature_order:
                missing = [n for n in feature_order if n not in features]
                if missing:
                    logger.warning(
                        "Missing %d feature(s) at inference for %s: %s",
                        len(missing),
                        symbol,
                        missing,
                    )
                features_flat = np.array(
                    [float(features.get(n, 0.0)) for n in feature_order],
                    dtype=np.float64,
                )
            else:
                # Fallback when no trained feature order is available.
                features_flat = np.array(
                    [features[k] for k in sorted(features.keys())], dtype=np.float64
                )

            # For sequence models we would need buffered history; pass None for now
            output = self._model_server.predict(
                symbol=symbol,
                features_flat=features_flat,
                features_sequence=None,
            )

        if self._calibrator is not None:
            output = self._calibrator.recalibrate(output)

        active_types = (
            ",".join(self._model_server.active_model_types)
            if self._model_server
            else "none"
        )

        # Confidence gate: abstain (flat) on low-confidence non-flat signals so
        # uncertain predictions don't drive trades.
        direction = output.direction
        if str(direction).lower() != "flat" and output.confidence < self._min_confidence:
            logger.info(
                "Prediction for %s gated: confidence %.3f < %.3f -> flat",
                symbol,
                output.confidence,
                self._min_confidence,
            )
            direction = "flat"

        prediction = Prediction(
            symbol=symbol,
            direction=direction,
            confidence=output.confidence,
            expected_return=output.expected_return,
            probabilities=output.probabilities,
            model_id=f"ensemble:{active_types}",
        )
        self._prediction_history.append(prediction)
        return prediction

    def _sampled_features(
        self, features: dict[str, float]
    ) -> tuple[list[float] | None, str | None]:
        """The served feature vector, on sampled minutes only.

        Foundation for the registered live-transfer promotion gate: only a
        COMPLETE vector in the trained column order is persisted (a partial
        one could not be replayed through a challenger), rounded to 6
        significant digits, and tagged with a hash of the feature-name
        ordering so replays refuse rows from a different pipeline generation.
        """
        if not self._should_persist_features(datetime.now(UTC)):
            return None, None
        feature_order = (
            self._model_server.feature_names if self._model_server else None
        )
        if not feature_order:
            return None, None
        if any(name not in features for name in feature_order):
            return None, None
        import hashlib

        vector = [float(f"{float(features[n]):.6g}") for n in feature_order]
        digest = hashlib.sha256(",".join(feature_order).encode()).hexdigest()[:16]
        return vector, digest

    def _should_persist_features(self, now: datetime) -> bool:
        return now.minute % self._feature_sample_minutes == 0

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
        return list(self._prediction_history)[-100:]
