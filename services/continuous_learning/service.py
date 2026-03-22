"""Main continuous learning service -- drift detection, evaluation, retraining."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog

from config.settings import Settings
from core.events.base import Event, EventBus
from core.events.system_events import DriftDetectedEvent, ModelRetrainedEvent
from services.continuous_learning.drift_detector import DataDriftDetector, ModelDriftDetector
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.continuous_learning.retrainer import AutoRetrainer

logger = structlog.get_logger(__name__)

# Stream / topic constants.
_ORDERS_STREAM = "orders"
_PREDICTIONS_STREAM = "predictions.ready"
_SYSTEM_STREAM = "system"
_CONSUMER_GROUP = "continuous-learning-service"

# Daily evaluation interval (seconds).
_EVALUATION_INTERVAL = 86_400  # 24 hours


class ContinuousLearningService:
    """Subscribes to trading events and drives the model improvement loop.

    Lifecycle:
        1. ``start()`` -- subscribe to ``OrderFilledEvent`` and
           ``PredictionReadyEvent`` streams, launch periodic evaluation.
        2. Filled orders are fed into the feedback loop.
        3. Predictions are tracked for later evaluation.
        4. Daily evaluation checks drift and triggers retraining when needed.
        5. Publishes ``ModelRetrainedEvent`` and ``DriftDetectedEvent``.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        evaluator: ModelEvaluator,
        retrainer: AutoRetrainer,
        drift_detector: DataDriftDetector,
        feedback_loop: TradingFeedbackLoop,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._evaluator = evaluator
        self._retrainer = retrainer
        self._drift_detector = drift_detector
        self._feedback_loop = feedback_loop
        self._model_drift_detector = ModelDriftDetector()
        self._tasks: list[asyncio.Task[Any]] = []
        self._running = False
        # model_id -> list of prediction records
        self._tracked_predictions: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to event streams and launch periodic evaluation."""
        if self._running:
            logger.warning("continuous_learning.already_running")
            return

        self._running = True
        logger.info("continuous_learning.starting")

        # Subscribe to order fills
        self._tasks.append(
            asyncio.create_task(
                self._event_bus.subscribe(
                    stream=_ORDERS_STREAM,
                    group=_CONSUMER_GROUP,
                    consumer="cl-orders-worker",
                    handler=self.handle_order_filled,
                ),
                name="cl-orders",
            )
        )

        # Subscribe to predictions
        self._tasks.append(
            asyncio.create_task(
                self._event_bus.subscribe(
                    stream=_PREDICTIONS_STREAM,
                    group=_CONSUMER_GROUP,
                    consumer="cl-predictions-worker",
                    handler=self.handle_prediction,
                ),
                name="cl-predictions",
            )
        )

        # Periodic evaluation
        self._tasks.append(
            asyncio.create_task(
                self._periodic_evaluation(),
                name="cl-periodic-eval",
            )
        )

        logger.info("continuous_learning.started")

    async def stop(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("continuous_learning.stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_order_filled(self, event: Event) -> None:
        """Record a filled order outcome in the feedback loop."""
        order_id = getattr(event, "order_id", event.payload.get("order_id", ""))
        fill_price = getattr(event, "fill_price", event.payload.get("fill_price", 0.0))
        fill_quantity = getattr(event, "fill_quantity", event.payload.get("fill_quantity", 0.0))
        commission = getattr(event, "commission", event.payload.get("commission", 0.0))

        # Use order_id as prediction_id linkage (simplified; production would
        # maintain an order->prediction mapping).
        trade_pnl = fill_price * fill_quantity - commission
        self._feedback_loop.record_outcome(
            prediction_id=order_id,
            actual_return=0.0,  # Updated later when position is closed
            trade_pnl=trade_pnl,
        )

        logger.debug(
            "continuous_learning.order_filled",
            order_id=order_id,
            fill_price=fill_price,
        )

    async def handle_prediction(self, event: Event) -> None:
        """Track a prediction for later evaluation."""
        model_id = getattr(event, "model_id", event.payload.get("model_id", ""))
        symbol = getattr(event, "symbol", event.payload.get("symbol", ""))
        direction = getattr(event, "direction", event.payload.get("direction", ""))
        confidence = getattr(event, "confidence", event.payload.get("confidence", 0.0))
        prediction_id = event.event_id

        self._evaluator.record_prediction(
            prediction_id=prediction_id,
            model_id=model_id,
            symbol=symbol,
            predicted=direction,
            confidence=confidence,
        )

        self._feedback_loop.register_prediction(prediction_id, model_id)

        self._tracked_predictions.setdefault(model_id, []).append(
            {
                "prediction_id": prediction_id,
                "predicted": direction,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        logger.debug(
            "continuous_learning.prediction_tracked",
            model_id=model_id,
            symbol=symbol,
        )

    # ------------------------------------------------------------------
    # Periodic evaluation
    # ------------------------------------------------------------------

    async def _periodic_evaluation(self) -> None:
        """Run daily: evaluate models, check drift, trigger retraining."""
        while self._running:
            try:
                await asyncio.sleep(_EVALUATION_INTERVAL)
                await self._run_evaluation_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("continuous_learning.periodic_evaluation.error")

    async def _run_evaluation_cycle(self) -> None:
        """Execute one full evaluation cycle."""
        logger.info("continuous_learning.evaluation_cycle.start")

        for model_id in list(self._tracked_predictions.keys()):
            # 1. Evaluate live performance
            report = self._evaluator.evaluate_live_performance(model_id)
            logger.info(
                "continuous_learning.evaluation",
                model_id=model_id,
                accuracy=report.accuracy,
                sample_size=report.sample_size,
            )

            # 2. Check for model drift
            predictions = self._tracked_predictions.get(model_id, [])
            if len(predictions) >= 100:
                mid = len(predictions) // 2
                older = predictions[:mid]
                recent = predictions[mid:]

                drift_report = self._model_drift_detector.detect_accuracy_drift(
                    recent_predictions=[
                        {"predicted": p["predicted"], "actual": p.get("actual", p["predicted"])}
                        for p in recent
                    ],
                    older_predictions=[
                        {"predicted": p["predicted"], "actual": p.get("actual", p["predicted"])}
                        for p in older
                    ],
                )

                if drift_report.is_drifting:
                    self._retrainer.mark_drift(model_id)

                    drift_event = DriftDetectedEvent(
                        source_service="continuous_learning",
                        drift_type="prediction",
                        score=drift_report.drift_score,
                        threshold=drift_report.threshold,
                    )
                    await self._event_bus.publish(_SYSTEM_STREAM, drift_event)

                    logger.warning(
                        "continuous_learning.drift_detected",
                        model_id=model_id,
                        score=drift_report.drift_score,
                    )

            # 3. Check if retraining is needed
            if self._retrainer.should_retrain(model_id):
                result = await self._retrainer.retrain(model_id)

                if not result.get("skipped"):
                    retrain_event = ModelRetrainedEvent(
                        source_service="continuous_learning",
                        model_id=result["new_model_id"],
                        version=result["version"],
                        metrics=result["metrics"],
                    )
                    await self._event_bus.publish(_SYSTEM_STREAM, retrain_event)

                    logger.info(
                        "continuous_learning.model_retrained",
                        old_model_id=model_id,
                        new_model_id=result["new_model_id"],
                        version=result["version"],
                    )

            # 4. Update ensemble weights
            all_metrics: dict[str, dict[str, Any]] = {}
            for mid in self._tracked_predictions:
                all_metrics[mid] = self._feedback_loop.compute_model_metrics(mid)

            if all_metrics:
                new_weights = self._feedback_loop.update_ensemble_weights(all_metrics)
                logger.info("continuous_learning.weights_updated", weights=new_weights)

        logger.info("continuous_learning.evaluation_cycle.complete")
