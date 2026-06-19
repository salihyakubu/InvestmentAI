"""Manual order intent path: the API publishes an OrderIntentEvent (202) and the
execution worker consumes it and submits the order to the broker.

This is the manual counterpart to the autonomous rebalance->risk->execution path.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.dependencies import get_current_user, get_db, get_event_bus
from api.main import app
from config.settings import Settings
from core.events.base import InProcessEventBus
from core.events.order_events import OrderIntentEvent
from core.events.streams import ORDER_INTENTS
from core.models.base import AsyncBase
from core.models.orders import Fill, Order
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService


@pytest.mark.asyncio
async def test_order_intent_is_executed_by_worker(mock_settings: Settings) -> None:
    bus = InProcessEventBus()
    broker = PaperBroker(initial_cash=Decimal("10000"))
    broker.update_price("AAPL", Decimal("100"))
    ex = ExecutionEngineService(event_bus=bus, settings=mock_settings, brokers={"paper": broker})
    await ex.start()
    try:
        await bus.publish(
            ORDER_INTENTS,
            OrderIntentEvent(
                order_id="db-1", symbol="AAPL", side="buy",
                order_type="market", quantity=3.0, source_service="test",
            ),
        )
        positions = await broker.get_positions()
    finally:
        await ex.stop()

    aapl = next((p for p in positions if p["symbol"] == "AAPL"), None)
    assert aapl is not None
    assert Decimal(aapl["quantity"]) == Decimal("3")


class _CapturingBus:
    def __init__(self) -> None:
        self.published: list = []

    async def publish(self, stream, event):  # noqa: ANN001
        self.published.append((stream, event))
        return "0-0"


def test_create_order_publishes_intent(mock_settings: Settings) -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(
                AsyncBase.metadata.create_all, tables=[Order.__table__, Fill.__table__]
            )

    asyncio.run(_setup())
    bus = _CapturingBus()

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: {"user_id": uuid.uuid4(), "role": "admin"}
    app.dependency_overrides[get_event_bus] = lambda: bus
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/orders",
                json={"symbol": "AAPL", "side": "buy", "order_type": "market", "quantity": "3"},
            )
            assert resp.status_code == 202
    finally:
        app.dependency_overrides.clear()
        asyncio.run(engine.dispose())
        os.unlink(path)

    intents = [e for s, e in bus.published if s == ORDER_INTENTS]
    assert len(intents) == 1
    assert intents[0].symbol == "AAPL" and intents[0].side == "buy"
