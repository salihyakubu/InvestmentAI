"""Halt, breaker and daily baseline survive a restart (GO_LIVE.md 2026-11-02).

Before this change all three lived in process memory: an operator halt was
lifted by the next deploy, a tripped breaker booted CLOSED, and the daily P&L
re-based to initial capital (a $92 account tripped the breaker on every
restart for the rest of the UTC day).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.events.base import InProcessEventBus
from core.events.order_events import TradingControlEvent
from core.models.base import AsyncBase
from core.models.orders import Fill, Order
from core.models.safety import SafetyState
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.service import ExecutionEngineService
from services.persistence.safety_state import SafetyStateStore
from services.risk.circuit_breaker import CircuitBreakerState
from services.risk.service import RiskManagerService

pytestmark = pytest.mark.asyncio


async def _store(trading_mode: str = "paper"):
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all,
            tables=[SafetyState.__table__, Order.__table__, Fill.__table__],
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, factory, SafetyStateStore(session_factory=factory, trading_mode=trading_mode)


def _engine(mock_settings: Settings, factory: Any, store: SafetyStateStore) -> ExecutionEngineService:
    broker = PaperBroker(initial_cash=Decimal("100000"))
    broker.update_price("AAPL", Decimal("100"))
    return ExecutionEngineService(
        event_bus=InProcessEventBus(), settings=mock_settings, brokers={"paper": broker},
        session_factory=factory, state_store=store,
    )


async def test_operator_halt_survives_a_restart_until_resumed(mock_settings: Settings) -> None:
    engine, factory, store = await _store()
    ex_a = _engine(mock_settings, factory, store)
    await ex_a._handle_control(
        TradingControlEvent(action="halt", reason="operator", source_service="api")
    )
    assert ex_a.halted

    ex_b = _engine(mock_settings, factory, store)
    assert not ex_b.halted  # memory alone forgets
    await ex_b.restore_safety_state()
    assert ex_b.halted
    refused = await ex_b.submit_order(
        symbol="AAPL", side="buy", order_type="market", quantity=Decimal("1"),
        reference_price=Decimal("100"),
    )
    assert refused is None

    await ex_b._handle_control(TradingControlEvent(action="resume", source_service="api"))
    ex_c = _engine(mock_settings, factory, store)
    await ex_c.restore_safety_state()
    assert not ex_c.halted
    await engine.dispose()


async def test_emergency_flatten_persists_its_halt(mock_settings: Settings) -> None:
    engine, factory, store = await _store()
    ex = _engine(mock_settings, factory, store)
    await ex.emergency_flatten()
    state = await store.load()
    assert state is not None and state.halted and state.halt_reason == "emergency_flatten"
    await engine.dispose()


def _risk(mock_settings: Settings, store: SafetyStateStore) -> RiskManagerService:
    return RiskManagerService(event_bus=InProcessEventBus(), settings=mock_settings, state_store=store)


async def test_same_day_restart_keeps_baseline_peak_and_open_breaker(
    mock_settings: Settings,
) -> None:
    engine, factory, store = await _store()
    risk_a = _risk(mock_settings, store)
    risk_a.sync_account(Decimal("100"))  # first sync seeds the day's baseline
    risk_a.sync_account(Decimal("120"))  # new peak
    risk_a.sync_account(Decimal("90"))   # -10% on the day -> breaker trips
    risk_a.circuit_breaker.check(risk_a._daily_pnl_pct)
    assert risk_a.circuit_breaker.state == CircuitBreakerState.OPEN
    await risk_a.persist_safety_state()

    risk_b = _risk(mock_settings, store)
    await risk_b.restore_safety_state()
    assert risk_b._day_start_equity == pytest.approx(100.0)
    assert risk_b.drawdown_monitor.peak_equity == pytest.approx(120.0)
    assert risk_b.circuit_breaker.state == CircuitBreakerState.OPEN
    assert await risk_b.publish_breaker_transition() is False  # restored, not a new trip
    risk_b.sync_account(Decimal("92"))
    assert risk_b._daily_pnl_pct == pytest.approx(-0.08)  # against the real baseline
    await engine.dispose()


async def test_previous_day_state_rebases_the_daily_baseline_from_first_sync(
    mock_settings: Settings,
) -> None:
    engine, factory, store = await _store()
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    await store.save_risk(
        breaker_state="open", breaker_tripped_at=datetime.now(UTC) - timedelta(hours=20),
        breaker_reason="yesterday", day_start_equity=Decimal("100"), peak_equity=Decimal("120"),
        session_date=yesterday,
    )
    risk = _risk(mock_settings, store)
    await risk.restore_safety_state()
    assert risk.circuit_breaker.state == CircuitBreakerState.CLOSED  # new day
    assert risk.drawdown_monitor.peak_equity == pytest.approx(120.0)  # all-time peak kept
    risk.sync_account(Decimal("92"))
    assert risk._day_start_equity == pytest.approx(92.0)  # NOT initial_capital, NOT yesterday
    assert risk._daily_pnl_pct == pytest.approx(0.0)
    assert isinstance(risk._session_date, date)
    await engine.dispose()
