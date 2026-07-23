"""Persist model predictions to the database.

The prediction service publishes :class:`PredictionReadyEvent` on the
``predictions.ready`` stream but nothing recorded them, so the ``predictions``
table stayed empty and the dashboard's prediction count / live-outcome metrics
could never be computed. This service subscribes to that stream and inserts a
``Prediction`` row per event. Reporting/audit only; the trading pipeline
(portfolio optimisation, risk, execution) already reacts to the event directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.events.base import Event, EventBus
from core.models.predictions import Prediction

logger = structlog.get_logger(__name__)

# Mirrors _PREDICTIONS_STREAM in services/prediction/service.py -- the stream
# PredictionService publishes PredictionReadyEvent on.
_STREAM = "predictions.ready"


class PredictionPersistenceService:
    """Subscribe to prediction events and persist them to the DB."""

    def __init__(
        self,
        event_bus: EventBus,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._event_bus = event_bus
        self._session_factory = session_factory

    async def start(self) -> None:
        await self._event_bus.subscribe(
            stream=_STREAM,
            group="prediction-persistence",
            consumer="predict-persist-1",
            handler=self.handle_event,
        )
        logger.info("PredictionPersistenceService started.")

    async def handle_event(self, event: Event) -> None:
        """Insert a predictions row for each PredictionReadyEvent; ignore the rest."""
        if event.event_type != "PredictionReadyEvent":
            return

        symbol = str(getattr(event, "symbol", "") or event.payload.get("symbol", ""))
        direction = str(
            getattr(event, "direction", "") or event.payload.get("direction", "")
        )
        confidence = getattr(event, "confidence", None)
        if confidence is None:
            confidence = event.payload.get("confidence", 0.0)
        expected_return = getattr(event, "expected_return", None)
        if expected_return is None:
            expected_return = event.payload.get("expected_return")
        model_id = str(
            getattr(event, "model_id", "") or event.payload.get("model_id", "")
        )
        # The continuous-learning tracker keys predictions by the event id; store
        # it so the outcome resolver can write the realised outcome back to this
        # row and startup rehydration can rebuild drift state from resolved rows.
        event_id = str(getattr(event, "event_id", "") or event.payload.get("event_id", ""))

        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                session.add(
                    Prediction(
                        predicted_at=now,
                        symbol=symbol,
                        model_id=model_id[:100],
                        model_version=1,
                        direction=direction,
                        confidence=float(confidence),
                        expected_return=(
                            None if expected_return is None else float(expected_return)
                        ),
                        # The model predicts the 5-minute forward return, so the
                        # outcome horizon for live-accuracy checks is 5 minutes.
                        horizon_minutes=5,
                        created_at=now,
                        event_id=event_id[:64] or None,
                    )
                )
                await session.commit()
            logger.info("prediction_persisted", symbol=symbol, model_id=model_id)
        except Exception:
            # Persistence must never kill the consumer loop.
            logger.exception("prediction_persist_failed", symbol=symbol)
