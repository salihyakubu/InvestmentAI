"""Main monitoring service -- subscribes to events, updates metrics, runs health checks."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import structlog

from config.settings import Settings
from core.events.base import Event, EventBus
from core.events.market_events import BarCloseEvent
from core.events.order_events import (
    OrderCreatedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
)
from core.events.risk_events import CircuitBreakerEvent, RiskBreachedEvent
from core.events.signal_events import PredictionReadyEvent
from core.events.system_events import DriftDetectedEvent
from services.monitoring.alerting import AlertManager, Severity
from services.monitoring.audit import AuditLogger
from services.monitoring.health import HealthChecker, HealthStatus
from services.monitoring.metrics import MetricsCollector

logger = structlog.get_logger(__name__)

# Stream / topic constants (must match other services).
_MARKET_BARS_STREAM = "market.bars"
_PREDICTIONS_STREAM = "predictions.ready"
_ORDERS_STREAM = "orders"
_RISK_STREAM = "risk"
_SYSTEM_STREAM = "system"

_CONSUMER_GROUP = "monitoring-service"
_HEALTH_CHECK_INTERVAL = 60  # seconds


class MonitoringService:
    """Observability hub that ties together metrics, health, alerts, and audit.

    Lifecycle:
        1. ``start()`` -- subscribe to all relevant event streams and launch
           periodic health-check loop.
        2. Incoming events update Prometheus metrics and trigger alert
           conditions when thresholds are breached.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        metrics: MetricsCollector,
        health_checker: HealthChecker,
        alert_manager: AlertManager,
        audit_logger: AuditLogger,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._metrics = metrics
        self._health_checker = health_checker
        self._alert_manager = alert_manager
        self._audit_logger = audit_logger
        self._tasks: list[asyncio.Task[Any]] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to all event streams and start background tasks."""
        if self._running:
            logger.warning("monitoring.already_running")
            return

        self._running = True
        logger.info("monitoring.starting")

        # Subscribe to event streams
        streams_and_handlers = [
            (_MARKET_BARS_STREAM, self._handle_bar_event),
            (_PREDICTIONS_STREAM, self._handle_prediction_event),
            (_ORDERS_STREAM, self._handle_order_event),
            (_RISK_STREAM, self._handle_risk_event),
            (_SYSTEM_STREAM, self._handle_system_event),
        ]

        for stream, handler in streams_and_handlers:
            task = asyncio.create_task(
                self._event_bus.subscribe(
                    stream=stream,
                    group=_CONSUMER_GROUP,
                    consumer="monitoring-worker",
                    handler=handler,
                ),
                name=f"monitoring-{stream}",
            )
            self._tasks.append(task)

        # Periodic health check
        self._tasks.append(
            asyncio.create_task(
                self._periodic_health_check(),
                name="monitoring-health-check",
            )
        )

        logger.info("monitoring.started")

    async def stop(self) -> None:
        """Cancel all background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("monitoring.stopped")

    # ------------------------------------------------------------------
    # Generic event handler
    # ------------------------------------------------------------------

    async def handle_event(self, event: Event) -> None:
        """Route an event to the appropriate handler and record lag."""
        now = datetime.now(timezone.utc)
        lag = (now - event.timestamp).total_seconds()
        self._metrics.record_event_lag(event.event_type, lag)

        event_type = event.event_type
        if event_type == "BarCloseEvent":
            await self._handle_bar_event(event)
        elif event_type == "PredictionReadyEvent":
            await self._handle_prediction_event(event)
        elif event_type in ("OrderCreatedEvent", "OrderFilledEvent", "OrderRejectedEvent"):
            await self._handle_order_event(event)
        elif event_type in ("CircuitBreakerEvent", "RiskBreachedEvent"):
            await self._handle_risk_event(event)
        elif event_type in ("DriftDetectedEvent", "ModelRetrainedEvent"):
            await self._handle_system_event(event)

    # ------------------------------------------------------------------
    # Per-stream handlers
    # ------------------------------------------------------------------

    async def _handle_bar_event(self, event: Event) -> None:
        """Track bar ingestion metrics."""
        source = event.payload.get("source_service", event.source_service or "unknown")
        symbol = getattr(event, "symbol", event.payload.get("symbol", "unknown"))
        self._metrics.record_bar_ingested(source, symbol)

    async def _handle_prediction_event(self, event: Event) -> None:
        """Track prediction metrics."""
        now = datetime.now(timezone.utc)
        lag = (now - event.timestamp).total_seconds()
        self._metrics.record_event_lag("PredictionReadyEvent", lag)

    async def _handle_order_event(self, event: Event) -> None:
        """Track order lifecycle metrics and audit."""
        event_type = event.event_type
        if event_type == "OrderCreatedEvent":
            self._metrics.record_order_submitted()
            await self._audit_logger.log_action(
                service="execution",
                action="order.created",
                entity_type="order",
                entity_id=getattr(event, "order_id", event.payload.get("order_id", "")),
                details=event.payload,
            )
        elif event_type == "OrderFilledEvent":
            self._metrics.record_order_filled()
            await self._audit_logger.log_action(
                service="execution",
                action="order.filled",
                entity_type="order",
                entity_id=getattr(event, "order_id", event.payload.get("order_id", "")),
                details=event.payload,
            )
        elif event_type == "OrderRejectedEvent":
            self._metrics.record_order_rejected()
            reason = getattr(event, "reason", event.payload.get("reason", ""))
            await self._audit_logger.log_action(
                service="execution",
                action="order.rejected",
                entity_type="order",
                entity_id=getattr(event, "order_id", event.payload.get("order_id", "")),
                details={"reason": reason},
            )

    async def _handle_risk_event(self, event: Event) -> None:
        """Process risk events and trigger alerts."""
        event_type = event.event_type

        if event_type == "CircuitBreakerEvent":
            state = getattr(event, "state", event.payload.get("state", ""))
            reason = getattr(event, "reason", event.payload.get("reason", ""))
            if state == "open":
                self._metrics.record_circuit_breaker_trip()
                await self._alert_manager.send_alert(
                    severity=Severity.CRITICAL,
                    title="Circuit Breaker Tripped",
                    message=f"Circuit breaker opened: {reason}",
                    details={"state": state, "reason": reason},
                )

        elif event_type == "RiskBreachedEvent":
            rule = getattr(event, "rule_name", event.payload.get("rule_name", ""))
            current = getattr(event, "current_value", event.payload.get("current_value", 0))
            limit = getattr(event, "limit_value", event.payload.get("limit_value", 0))

            # Alert on large drawdown
            if rule == "max_total_drawdown" and current > self._settings.max_total_drawdown_pct * 0.8:
                await self._alert_manager.send_alert(
                    severity=Severity.CRITICAL,
                    title="Large Drawdown Alert",
                    message=f"Drawdown {current:.2%} approaching limit {limit:.2%}",
                    details={"rule": rule, "current": current, "limit": limit},
                )

    async def _handle_system_event(self, event: Event) -> None:
        """Process system events (drift, retraining)."""
        event_type = event.event_type

        if event_type == "DriftDetectedEvent":
            drift_type = getattr(event, "drift_type", event.payload.get("drift_type", ""))
            score = getattr(event, "score", event.payload.get("score", 0))
            await self._alert_manager.send_alert(
                severity=Severity.WARNING,
                title="Model Drift Detected",
                message=f"{drift_type} drift detected with score {score:.4f}",
                details={"drift_type": drift_type, "score": score},
            )

    # ------------------------------------------------------------------
    # Periodic health check
    # ------------------------------------------------------------------

    async def _periodic_health_check(self) -> None:
        """Run health checks every 60 seconds."""
        while self._running:
            try:
                results = await self._health_checker.check_all()

                # Alert on unhealthy components
                for component, status in results.items():
                    if component == "overall":
                        continue
                    if status == HealthStatus.UNHEALTHY:
                        await self._alert_manager.send_alert(
                            severity=Severity.CRITICAL,
                            title=f"{component} Unhealthy",
                            message=f"Health check failed for {component}",
                            details={"component": component, "status": status.value},
                        )
                    elif status == HealthStatus.DEGRADED:
                        await self._alert_manager.send_alert(
                            severity=Severity.WARNING,
                            title=f"{component} Degraded",
                            message=f"Health check degraded for {component}",
                            details={"component": component, "status": status.value},
                        )

            except Exception:
                logger.exception("monitoring.health_check.error")

            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
