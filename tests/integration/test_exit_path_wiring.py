"""The exit path exists and is wired (GO_LIVE.md 2026-11-02).

Three independent reviewers found that no position could ever be closed
automatically: the liquidation manager subscribed to a stream nobody published
on, nothing consumed its liquidation events, and the breaker trip published no
event at all. These tests pin the wiring end to end on the in-process bus.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from config.settings import Settings
from core.events.base import Event, InProcessEventBus
from core.events.risk_events import CircuitBreakerEvent, RiskBreachedEvent
from core.events.streams import CONTROL, LIQUIDATION, ORDERS, RISK_BREACHED
from core.events.system_events import LiquidationTriggeredEvent
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.liquidation import service as liquidation_module
from services.liquidation.service import LiquidationManagerService
from services.monitoring import service as monitoring_module
from services.monitoring.service import MonitoringService
from services.risk.service import RiskManagerService

pytestmark = pytest.mark.asyncio


async def test_consumers_subscribe_to_the_stream_risk_publishes_on() -> None:
    """The constants that were retyped by hand are now imported."""
    assert liquidation_module.STREAM_RISK == RISK_BREACHED
    assert monitoring_module._RISK_STREAM == RISK_BREACHED
    assert liquidation_module.STREAM_LIQUIDATION == LIQUIDATION


async def test_risk_breach_reaches_the_liquidation_manager(mock_settings: Settings) -> None:
    bus = InProcessEventBus()
    liq = LiquidationManagerService(event_bus=bus, settings=mock_settings)
    received: list[Event] = []

    async def spy(event: Event) -> None:
        received.append(event)
        await LiquidationManagerService.handle_risk_breach(liq, event)

    liq.handle_risk_breach = spy  # type: ignore[method-assign]
    await liq.start()
    await bus.publish(
        RISK_BREACHED,
        RiskBreachedEvent(
            rule_name="MaxDrawdown", current_value=0.2, limit_value=0.15,
            action="block", source_service="risk_manager",
        ),
    )
    assert len(received) == 1 and received[0].event_type == "RiskBreachedEvent"


async def test_liquidation_event_closes_full_position_through_tracked_path_while_halted(
    mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("services.execution.service._POLL_INTERVAL", 0.02)
    bus = InProcessEventBus()
    fills: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "OrderFilledEvent":
            fills.append(event)

    await bus.subscribe(ORDERS, "t", "t", capture)
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    ex = ExecutionEngineService(event_bus=bus, settings=mock_settings, brokers={"paper": broker})
    await ex.start()
    try:
        await ex.submit_order(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("5"), reference_price=Decimal("100"),
        )
        ex.halt()  # a halted engine must still let a risk-reducing exit through

        # quantity 0 == "the whole position", resolved against the broker's book
        await bus.publish(
            LIQUIDATION,
            LiquidationTriggeredEvent(
                position_id="p1", symbol="AAPL", reason="stop", quantity=0.0,
                source_service="LiquidationManagerService",
            ),
        )
        assert await broker.get_positions() == []  # closed, synchronously in paper

        # The exit went through the tracked path: its fill is published and tagged.
        for _ in range(200):
            if len(fills) >= 2:
                break
            await asyncio.sleep(0.02)
        closing = [f for f in fills if getattr(f, "side", "") == "sell"]
        assert len(closing) == 1
        assert str(getattr(closing[0], "client_order_id", "")).startswith("liq-")

        # ...and an ordinary order is still refused while halted.
        placed = await ex.submit_order(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("1"), reference_price=Decimal("100"),
        )
        assert placed is None and await broker.get_positions() == []
    finally:
        await ex.stop()


async def test_breaker_transition_publishes_exactly_one_event(mock_settings: Settings) -> None:
    bus = InProcessEventBus()
    events: list[Event] = []

    async def capture(event: Event) -> None:
        if event.event_type == "CircuitBreakerEvent":
            events.append(event)

    await bus.subscribe(RISK_BREACHED, "t", "t", capture)
    risk = RiskManagerService(event_bus=bus, settings=mock_settings)
    risk.update_equity(Decimal("100"))
    risk.reset_daily()  # day-start = 100
    risk.sync_account(Decimal("90"))  # -10% day > 7% breaker threshold
    risk.circuit_breaker.check(risk._daily_pnl_pct)
    assert await risk.publish_breaker_transition() is True
    assert await risk.publish_breaker_transition() is False  # no re-publish
    assert len(events) == 1 and getattr(events[0], "state", "") == "open"


class _AlertSpy:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_alert(self, **kwargs: Any) -> None:
        self.sent.append(kwargs)


class _AuditSpy:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def log_action(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


async def test_monitoring_alerts_and_audits_a_breaker_trip(mock_settings: Settings) -> None:
    alerts, audit = _AlertSpy(), _AuditSpy()
    mon = MonitoringService(
        event_bus=InProcessEventBus(), settings=mock_settings,
        metrics=monitoring_module.MetricsCollector(),
        health_checker=None,  # type: ignore[arg-type]
        alert_manager=alerts,  # type: ignore[arg-type]
        audit_logger=audit,  # type: ignore[arg-type]
    )
    await mon._handle_risk_event(
        CircuitBreakerEvent(state="open", reason="Daily loss -10%", source_service="risk_manager")
    )
    assert len(alerts.sent) == 1 and alerts.sent[0]["title"] == "Circuit Breaker Tripped"
    assert len(audit.rows) == 1 and audit.rows[0]["action"] == "circuit_breaker.open"


async def test_breaker_auto_flatten_is_off_by_default_and_publishes_flatten_when_on(
    mock_settings: Settings,
) -> None:
    for flag, expected in ((False, 0), (True, 1)):
        settings = mock_settings.model_copy(update={"breaker_auto_flatten": flag})
        bus = InProcessEventBus()
        controls: list[Event] = []

        async def capture(event: Event) -> None:
            controls.append(event)

        await bus.subscribe(CONTROL, "t", "t", capture)
        liq = LiquidationManagerService(event_bus=bus, settings=settings)
        await liq.start()
        await bus.publish(
            RISK_BREACHED,
            CircuitBreakerEvent(state="open", reason="Daily loss", source_service="risk_manager"),
        )
        assert len(controls) == expected, f"breaker_auto_flatten={flag}"
        if expected:
            assert getattr(controls[0], "action", "") == "flatten"
