"""Main continuous learning service -- drift detection, evaluation, retraining."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.settings import Settings
from core.events.base import Event, EventBus
from core.events.streams import CONTROL, SYSTEM
from core.events.system_events import DriftDetectedEvent, ModelRetrainedEvent
from core.models.market_data import OHLCVRecord
from services.continuous_learning.drift_detector import DataDriftDetector, ModelDriftDetector
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.feedback_loop import TradingFeedbackLoop
from services.continuous_learning.retrainer import AutoRetrainer

logger = structlog.get_logger(__name__)

# Stream / topic constants.
_ORDERS_STREAM = "orders"
_PREDICTIONS_STREAM = "predictions.ready"
_SYSTEM_STREAM = SYSTEM
_CONSUMER_GROUP = "continuous-learning-service"

# Daily evaluation interval (seconds).
_EVALUATION_INTERVAL = 86_400  # 24 hours

# Outcome resolution: a prediction is scored against the realised return over
# the next HORIZON minutes. The resolver runs every
# _OUTCOME_RESOLUTION_INTERVAL seconds and only touches predictions older
# than HORIZON + GRACE so the t+HORIZON ohlcv bar has had time to land.
_OUTCOME_HORIZON = timedelta(minutes=5)
_OUTCOME_GRACE = timedelta(minutes=1)
_OUTCOME_RESOLUTION_INTERVAL = 600  # seconds

# Bounded bar lookups. t0 must fall within a short window before the
# prediction (covers CCXT poll jitter + pipeline latency); t5 comes from the
# BAR GRID (t0 bar time + horizon) with a short slack for late bars. An
# unbounded lookup would score a prediction near session close against the
# NEXT session's first bar (an overnight gap recorded as a 5-minute outcome).
_T0_LOOKBACK = timedelta(minutes=10)
_T5_SLACK = timedelta(minutes=3)

# With bounded lookups, some records (e.g. end-of-session stock predictions)
# can never resolve; drop them after this age instead of re-querying forever.
_UNRESOLVABLE_AFTER = timedelta(minutes=60)

# Per-model cap on RESOLVED records retained for drift detection; oldest
# resolved records beyond the cap are pruned so tracking memory is bounded.
_MAX_RESOLVED_PER_MODEL = 20_000

# Predictions consumed more than 2x the horizon after creation are backlog
# replays -- their outcome window has already passed, so tracking them would
# score stale predictions against unrelated bars.
_STALE_EVENT_CUTOFF = 2 * _OUTCOME_HORIZON

# Drift detection runs over a rolling window of the most recent resolved
# records, split at the midpoint (recent ~1000 vs prior ~1000). At the old
# >=100 gate the fixed 5pp accuracy-drop threshold was a ~0.5-sigma event
# (~31% false fire); ~1000 per half brings the false rate to ~1.3%.
_DRIFT_MIN_RESOLVED = 1000
_DRIFT_WINDOW = 2000

# Realised-direction deadband. The serving path derives the *predicted*
# direction from calibrated class probabilities (the 30/70 logic), which has
# no analogue for a realised price move. We instead classify the realised
# 5-minute return with a simple +/-5bp threshold: > +0.05% -> "long",
# < -0.05% -> "short", inside the band -> "flat". Coarse, but it is a real
# market outcome rather than a fabricated one.
_ACTUAL_RETURN_THRESHOLD = 0.0005


class ContinuousLearningService:
    """Subscribes to trading events and drives the model improvement loop.

    Lifecycle:
        1. ``start()`` -- subscribe to the orders, predictions, and control
           streams; launch periodic evaluation and outcome resolution.
        2. Filled orders are fed into the feedback loop.
        3. Predictions are tracked; a background resolver later scores them
           against realised 1m ohlcv closes (real actuals, not fabricated).
        4. Daily evaluation checks drift and triggers retraining when needed;
           a ``TradingControlEvent(action="retrain")`` triggers it on demand.
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
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._evaluator = evaluator
        self._retrainer = retrainer
        self._drift_detector = drift_detector
        self._feedback_loop = feedback_loop
        self._session_factory = session_factory
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

        # Subscribe to operator control commands (on-demand retrain trigger)
        self._tasks.append(
            asyncio.create_task(
                self._event_bus.subscribe(
                    stream=CONTROL,
                    group=_CONSUMER_GROUP,
                    consumer="cl-control-1",
                    handler=self.handle_control,
                ),
                name="cl-control",
            )
        )

        # Periodic evaluation
        self._tasks.append(
            asyncio.create_task(
                self._periodic_evaluation(),
                name="cl-periodic-eval",
            )
        )

        # Periodic outcome resolution (scores predictions against realised
        # ohlcv closes so evaluation and drift detection use real actuals)
        self._tasks.append(
            asyncio.create_task(
                self._outcome_resolution_loop(),
                name="cl-outcome-resolution",
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
        """Observe a filled order (log only for now).

        The orders stream carries the full order lifecycle (created / filled /
        rejected / cancelled); only fills represent an outcome, so everything
        else is ignored.
        """
        if event.event_type != "OrderFilledEvent":
            return

        order_id = getattr(event, "order_id", event.payload.get("order_id", ""))
        fill_price = getattr(event, "fill_price", event.payload.get("fill_price", 0.0))

        # No feedback-loop record here: fill_price * quantity is NOTIONAL, not
        # P&L, and order_id never matches a registered prediction_id. Real
        # P&L attribution needs an order->prediction map (deliberately not
        # built yet).
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

        # Score against the event's CREATION time, not consumption time --
        # under backlog replay the two differ by minutes and t0 would land on
        # the wrong bars. Events past their outcome window are not tracked.
        event_time = event.timestamp or datetime.now(UTC)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)
        if datetime.now(UTC) - event_time > _STALE_EVENT_CUTOFF:
            logger.debug(
                "continuous_learning.stale_prediction_skipped",
                model_id=model_id,
                symbol=symbol,
                event_timestamp=event_time.isoformat(),
            )
            return

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
                "symbol": symbol,
                "predicted": direction,
                "confidence": confidence,
                "timestamp": event_time,
            }
        )

        logger.debug(
            "continuous_learning.prediction_tracked",
            model_id=model_id,
            symbol=symbol,
        )

    async def handle_control(self, event: Event) -> None:
        """React to operator control commands on the CONTROL stream.

        Only ``retrain`` is handled here (an immediate evaluation / drift /
        retrain cycle). The other actions (halt / resume / flatten) are owned
        by the execution engine and are deliberately ignored.
        """
        if event.event_type != "TradingControlEvent":
            return
        action = getattr(event, "action", event.payload.get("action", ""))
        if action != "retrain":
            return

        logger.info(
            "continuous_learning.retrain_requested",
            reason=getattr(event, "reason", event.payload.get("reason", "")),
        )
        await self._run_evaluation_cycle()

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

    # ------------------------------------------------------------------
    # Outcome resolution
    # ------------------------------------------------------------------

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Return the injected session factory, or the shared one lazily."""
        if self._session_factory is None:
            from core.models.base import get_async_session_factory

            self._session_factory = get_async_session_factory()
        return self._session_factory

    async def _outcome_resolution_loop(self) -> None:
        """Periodically score matured predictions against realised prices."""
        while self._running:
            try:
                await asyncio.sleep(_OUTCOME_RESOLUTION_INTERVAL)
                await self._resolve_outcomes_once()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("continuous_learning.outcome_resolution.error")

    async def _resolve_outcomes_once(self) -> None:
        """Resolve real outcomes for tracked predictions past the horizon.

        For every tracked prediction without an ``actual`` and older than
        HORIZON + GRACE, look up the 1m ohlcv bar at (or just before) the
        prediction time and the bar-grid bar one HORIZON later, compute the
        realised return, classify its direction, and feed the outcome to the
        evaluator and the feedback loop. Afterwards, prune records that keep
        the tracked set bounded: unresolved records past _UNRESOLVABLE_AFTER
        and resolved records beyond _MAX_RESOLVED_PER_MODEL.
        """
        now = datetime.now(UTC)
        matured_cutoff = now - (_OUTCOME_HORIZON + _OUTCOME_GRACE)
        expiry_cutoff = now - _UNRESOLVABLE_AFTER
        resolved = 0
        expired = 0

        factory = self._get_session_factory()
        async with factory() as session:
            # Snapshot the dict: handle_prediction can insert a NEW model_id
            # key during the awaits below (first prediction after a hot
            # reload), which would raise RuntimeError on a live view.
            for records in list(self._tracked_predictions.values()):
                for record in records:
                    if "actual" in record:
                        continue
                    if record["timestamp"] <= expiry_cutoff:
                        continue  # dropped in the prune pass below
                    if not record.get("symbol"):
                        continue  # no symbol; unresolvable, expires above
                    if record["timestamp"] > matured_cutoff:
                        continue  # not matured yet
                    if await self._resolve_record(session, record):
                        resolved += 1

        # Prune pass (no awaits, so atomic w.r.t. the event loop): drop
        # expired unresolved records and cap resolved history per model,
        # keeping arrival order for the drift midpoint split.
        for model_id, records in list(self._tracked_predictions.items()):
            overflow = sum(1 for r in records if "actual" in r) - _MAX_RESOLVED_PER_MODEL
            kept: list[dict[str, Any]] = []
            for record in records:
                if "actual" in record:
                    if overflow > 0:
                        overflow -= 1
                        continue  # oldest resolved beyond the cap
                elif record["timestamp"] <= expiry_cutoff:
                    expired += 1
                    continue
                kept.append(record)
            self._tracked_predictions[model_id] = kept

        if expired:
            logger.debug(
                "continuous_learning.unresolvable_expired", count=expired
            )
        if resolved:
            logger.info("continuous_learning.outcomes_resolved", count=resolved)

    async def _resolve_record(
        self, session: AsyncSession, record: dict[str, Any]
    ) -> bool:
        """Resolve one prediction record; return True if an outcome was set."""
        symbol: str = record["symbol"]
        ts: datetime = record["timestamp"]

        # t0: newest bar in the bounded window before the prediction.
        t0_bar = await self._lookup_bar(
            session, symbol, ts - _T0_LOOKBACK, ts, latest=True
        )
        if t0_bar is None:
            return False  # no bar near the prediction; retry until expiry
        t0_time, close_t0 = t0_bar

        # t5: first bar on the bar grid at t0 + horizon (plus slack). Missing
        # means the session closed / feed halted -- unresolvable, NOT scored
        # against whatever bar comes next.
        target = t0_time + _OUTCOME_HORIZON
        t5_bar = await self._lookup_bar(
            session, symbol, target, target + _T5_SLACK, latest=False
        )
        if t5_bar is None or close_t0 <= 0:
            return False
        _, close_t5 = t5_bar

        actual_return = (close_t5 - close_t0) / close_t0
        # See _ACTUAL_RETURN_THRESHOLD for why realised direction uses a
        # simple +/-5bp deadband instead of the serving-side 30/70 logic.
        if actual_return > _ACTUAL_RETURN_THRESHOLD:
            actual_direction = "long"
        elif actual_return < -_ACTUAL_RETURN_THRESHOLD:
            actual_direction = "short"
        else:
            actual_direction = "flat"

        record["actual"] = actual_direction
        record["actual_return"] = actual_return
        prediction_id: str = record["prediction_id"]
        self._evaluator.record_outcome(prediction_id, actual_direction, actual_return)

        # The feedback loop gets the DIRECTION-SIGNED return (what the model's
        # call would have earned), otherwise its Sharpe measures market drift
        # rather than model skill.
        predicted = record.get("predicted", "")
        if predicted == "long":
            signed_return = actual_return
        elif predicted == "short":
            signed_return = -actual_return
        else:
            signed_return = 0.0
        self._feedback_loop.record_outcome(prediction_id, signed_return, trade_pnl=0.0)

        logger.debug(
            "continuous_learning.outcome_resolved",
            prediction_id=prediction_id,
            symbol=symbol,
            actual=actual_direction,
            actual_return=actual_return,
        )
        return True

    @staticmethod
    async def _lookup_bar(
        session: AsyncSession,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        latest: bool,
    ) -> tuple[datetime, float] | None:
        """Return (time, close) of a 1m bar with time in [start, end].

        ``latest`` picks the newest bar in the window, else the oldest;
        returns ``None`` when the window contains no bar.
        """
        stmt = (
            select(OHLCVRecord.time, OHLCVRecord.close)
            .where(
                OHLCVRecord.symbol == symbol,
                OHLCVRecord.timeframe == "1m",
                OHLCVRecord.time >= start,
                OHLCVRecord.time <= end,
            )
            .order_by(OHLCVRecord.time.desc() if latest else OHLCVRecord.time.asc())
            .limit(1)
        )
        row = (await session.execute(stmt)).first()
        if row is None:
            return None
        bar_time, close = row
        return bar_time, float(close)

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

            # 2. Check for model drift -- only over predictions with a REAL
            # resolved outcome. Records without an "actual" are skipped;
            # substituting the predicted value would fabricate 100% accuracy
            # and mask (or invent) drift.
            predictions = [
                p for p in self._tracked_predictions.get(model_id, []) if "actual" in p
            ]
            if len(predictions) >= _DRIFT_MIN_RESOLVED:
                # Rolling window: recent half vs prior half of the newest
                # _DRIFT_WINDOW records; all-history halves would numb the
                # comparison as history accumulates.
                predictions = predictions[-_DRIFT_WINDOW:]
                midpoint = len(predictions) // 2
                older = predictions[:midpoint]
                recent = predictions[midpoint:]

                drift_report = self._model_drift_detector.detect_accuracy_drift(
                    recent_predictions=[
                        {"predicted": p["predicted"], "actual": p["actual"]}
                        for p in recent
                    ],
                    older_predictions=[
                        {"predicted": p["predicted"], "actual": p["actual"]}
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
                        model_id=model_id,
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
