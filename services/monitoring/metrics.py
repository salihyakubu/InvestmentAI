"""Prometheus metrics definitions and collector for the trading platform."""

from __future__ import annotations

import time
from typing import Any

import structlog
from prometheus_client import Counter, Gauge, Histogram

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metric definitions
# ---------------------------------------------------------------------------

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Time taken to produce a prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

order_fill_rate = Counter(
    "order_fill_rate",
    "Order lifecycle counts by outcome",
    labelnames=["outcome"],  # submitted, filled, rejected
)

portfolio_equity = Gauge(
    "portfolio_equity",
    "Current total portfolio equity",
)

portfolio_daily_pnl = Gauge(
    "portfolio_daily_pnl",
    "Portfolio profit and loss for the current day",
)

model_accuracy = Gauge(
    "model_accuracy",
    "Current accuracy of a prediction model",
    labelnames=["model_name"],
)

active_positions_count = Gauge(
    "active_positions_count",
    "Number of currently open positions",
)

risk_var_95 = Gauge(
    "risk_var_95",
    "Portfolio 95th-percentile Value at Risk",
)

drawdown_current = Gauge(
    "drawdown_current",
    "Current portfolio drawdown from peak equity",
)

data_ingestion_bars = Counter(
    "data_ingestion_bars",
    "Number of OHLCV bars ingested",
    labelnames=["source", "symbol"],
)

event_processing_lag_seconds = Histogram(
    "event_processing_lag_seconds",
    "Lag between event timestamp and processing time",
    labelnames=["event_type"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
)

circuit_breaker_trips = Counter(
    "circuit_breaker_trips",
    "Number of circuit breaker activations",
)


class MetricsCollector:
    """Instruments services by updating Prometheus metrics from events."""

    def record_prediction_latency(self, duration_seconds: float) -> None:
        """Record how long a prediction took."""
        prediction_latency_seconds.observe(duration_seconds)

    def record_order_submitted(self) -> None:
        order_fill_rate.labels(outcome="submitted").inc()

    def record_order_filled(self) -> None:
        order_fill_rate.labels(outcome="filled").inc()

    def record_order_rejected(self) -> None:
        order_fill_rate.labels(outcome="rejected").inc()

    def set_portfolio_equity(self, value: float) -> None:
        portfolio_equity.set(value)

    def set_portfolio_daily_pnl(self, value: float) -> None:
        portfolio_daily_pnl.set(value)

    def set_model_accuracy(self, name: str, value: float) -> None:
        model_accuracy.labels(model_name=name).set(value)

    def set_active_positions(self, count: int) -> None:
        active_positions_count.set(count)

    def set_risk_var_95(self, value: float) -> None:
        risk_var_95.set(value)

    def set_drawdown(self, value: float) -> None:
        drawdown_current.set(value)

    def record_bar_ingested(self, source: str, symbol: str) -> None:
        data_ingestion_bars.labels(source=source, symbol=symbol).inc()

    def record_event_lag(self, event_type: str, lag_seconds: float) -> None:
        event_processing_lag_seconds.labels(event_type=event_type).observe(lag_seconds)

    def record_circuit_breaker_trip(self) -> None:
        circuit_breaker_trips.inc()
