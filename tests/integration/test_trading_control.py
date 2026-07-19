"""The remote kill switch: control events + the paper-broker price feed."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from core.events.base import InProcessEventBus
from core.events.market_events import PriceUpdateEvent
from core.events.order_events import TradingControlEvent
from core.events.streams import CONTROL
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService


def _settings() -> Settings:
    return Settings(
        trading_mode="paper",
        initial_capital=Decimal("100000"),
        jwt_secret="test-secret-not-the-published-default-value",
    )


async def _engine_with_position() -> tuple[ExecutionEngineService, PaperBroker, InProcessEventBus]:
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("BTC/USDT", Decimal("50000"))
    engine = ExecutionEngineService(event_bus=bus, settings=_settings(), brokers={"paper": broker})
    await engine.start()
    await engine.submit_order(
        symbol="BTC/USDT", side="buy", order_type="market", quantity=Decimal("0.01")
    )
    positions = await broker.get_positions()
    assert positions, "drill precondition: an open position"
    return engine, broker, bus


@pytest.mark.asyncio
async def test_flatten_event_closes_positions_and_halts() -> None:
    engine, broker, bus = await _engine_with_position()

    await bus.publish(CONTROL, TradingControlEvent(action="flatten", source_service="test"))

    assert engine.halted is True
    assert await broker.get_positions() == []
    await engine.stop()


@pytest.mark.asyncio
async def test_halt_and_resume_events() -> None:
    engine, _broker, bus = await _engine_with_position()

    await bus.publish(CONTROL, TradingControlEvent(action="halt", source_service="test"))
    assert engine.halted is True

    # New orders must be refused while halted.
    order = await engine.submit_order(
        symbol="BTC/USDT", side="buy", order_type="market", quantity=Decimal("0.01")
    )
    assert order is None or getattr(order, "status", None) != "filled"

    await bus.publish(CONTROL, TradingControlEvent(action="resume", source_service="test"))
    assert engine.halted is False
    await engine.stop()


@pytest.mark.asyncio
async def test_price_updates_feed_the_paper_broker() -> None:
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("1000"))
    engine = ExecutionEngineService(event_bus=bus, settings=_settings(), brokers={"paper": broker})
    await engine.start()

    from datetime import UTC, datetime

    await bus.publish(
        "market.prices",
        PriceUpdateEvent(
            symbol="ETH/USDT", price=3000.5, volume=1.0,
            market_timestamp=datetime.now(UTC), source_service="test",
        ),
    )
    assert broker._last_prices.get("ETH/USDT") == Decimal("3000.5")
    await engine.stop()


# ---------------------------------------------------------------------------
# Admin endpoint
# ---------------------------------------------------------------------------


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    async def publish(self, stream: str, event: Any) -> None:
        self.published.append((stream, event))


def _client_with(role: str, bus: _FakeBus) -> TestClient:
    from uuid import uuid4

    from api.dependencies import get_current_user, get_event_bus
    from api.main import app

    app.dependency_overrides[get_current_user] = lambda: {"user_id": uuid4(), "role": role}
    app.dependency_overrides[get_event_bus] = lambda: bus
    return TestClient(app, raise_server_exceptions=False)


def test_admin_control_publishes_event() -> None:
    bus = _FakeBus()
    client = _client_with("admin", bus)
    try:
        resp = client.post("/api/v1/admin/control", json={"action": "flatten", "reason": "drill"})
        assert resp.status_code == 202
        assert resp.json()["action"] == "flatten"
        assert len(bus.published) == 1
        stream, event = bus.published[0]
        assert stream == CONTROL
        assert event.action == "flatten"
    finally:
        from api.main import app

        app.dependency_overrides.clear()


def test_non_admin_control_is_forbidden() -> None:
    bus = _FakeBus()
    client = _client_with("viewer", bus)
    try:
        resp = client.post("/api/v1/admin/control", json={"action": "halt"})
        assert resp.status_code == 403
        assert bus.published == []
    finally:
        from api.main import app

        app.dependency_overrides.clear()
