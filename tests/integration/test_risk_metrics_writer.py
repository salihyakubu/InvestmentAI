"""The risk-metrics writer: real state persisted, vacuous rules re-armed.

Two defects this pins shut. The ``risk_metrics`` table had readers and no
writer, so the dashboard's breaker card could only ever say UNKNOWN (and
before that, a false green). And the risk engine's return history had no
feeder, so VaR computed to 0.0 structurally and MaxVaRRule/MaxCorrelationRule
approved every order without checking anything -- the quietest possible
failure of a risk system. The re-arming test here is the one that matters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.events.base import InProcessEventBus
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from core.models.risk import RiskMetric
from services.persistence.risk_metrics_writer import RiskMetricsWriter
from services.risk.service import RiskManagerService

pytestmark = pytest.mark.asyncio


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_: JSONB, compiler: Any, **kw: Any) -> str:
    return "JSON"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
    )
    values.update(overrides)
    return Settings(**values)


async def _factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all,
            tables=[RiskMetric.__table__, OHLCVRecord.__table__],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_bars(factory, symbol: str, n: int = 60, wobble: float = 0.01) -> None:
    """n 1m bars with genuine variance, so returns exist to feed the engine."""
    async with factory() as session:
        base = datetime.now(UTC) - timedelta(minutes=n)
        price = 100.0
        for i in range(n):
            price *= 1.0 + (wobble if i % 2 else -wobble)
            session.add(
                OHLCVRecord(
                    time=base + timedelta(minutes=i),
                    symbol=symbol,
                    timeframe="1m",
                    open=Decimal(str(price)),
                    high=Decimal(str(price * 1.001)),
                    low=Decimal(str(price * 0.999)),
                    close=Decimal(str(price)),
                    volume=Decimal("10"),
                    source="test",
                    asset_class="crypto",
                    ingested_at=base + timedelta(minutes=i),
                )
            )
        await session.commit()


def _positions_provider(rows: list[dict[str, Any]]):
    async def provider() -> list[dict[str, Any]]:
        return rows

    return provider


async def _make(factory, positions: list[dict[str, Any]], **settings_overrides: Any):
    risk = RiskManagerService(
        event_bus=InProcessEventBus(),  # type: ignore[arg-type]
        settings=_settings(**settings_overrides),
    )
    risk.update_equity(Decimal("100"))
    writer = RiskMetricsWriter(
        session_factory=factory,
        risk_service=risk,
        positions_provider=_positions_provider(positions),
    )
    return risk, writer


async def _rows(factory) -> list[RiskMetric]:
    async with factory() as session:
        result = await session.execute(select(RiskMetric).order_by(RiskMetric.time))
        return list(result.scalars().all())


async def test_write_once_persists_real_var_and_breaker_state() -> None:
    engine, factory = await _factory()
    await _seed_bars(factory, "BTC/USDT")
    await _seed_bars(factory, "ETH/USDT")
    risk, writer = await _make(
        factory,
        [
            {"symbol": "BTC/USDT", "market_value": "40"},
            {"symbol": "ETH/USDT", "market_value": "30"},
        ],
    )
    await writer.write_once()

    rows = await _rows(factory)
    assert len(rows) == 1
    row = rows[0]
    # VaR is REAL, not the structural 0.0 the empty return history produced.
    assert row.var_95 is not None and row.var_95 > 0
    assert row.var_99 is not None and row.var_99 >= row.var_95
    assert row.cvar_95 is not None and row.cvar_95 > 0
    assert row.circuit_breaker_active is False
    details = row.details or {}
    assert details["circuit_breaker_state"] == "closed"
    assert details["volatility"] is not None and details["volatility"] > 0
    assert details["positions_fed"] == 2
    # Beta has no producer, so it must be NULL -- never 0.00.
    assert row.beta is None
    await engine.dispose()


async def test_tripped_breaker_is_persisted_as_open() -> None:
    engine, factory = await _factory()
    await _seed_bars(factory, "BTC/USDT")
    risk, writer = await _make(factory, [{"symbol": "BTC/USDT", "market_value": "40"}])
    # A daily loss beyond the threshold trips the breaker on next check.
    risk.update_daily_pnl(-0.10)
    await writer.write_once()

    row = (await _rows(factory))[0]
    assert row.circuit_breaker_active is True
    assert (row.details or {})["circuit_breaker_state"] == "open"
    await engine.dispose()


async def test_the_vacuous_rules_are_re_armed() -> None:
    """THE test: before the writer feeds inputs, MaxVaRRule approves anything;
    after one cycle it actually evaluates and can reject."""
    engine, factory = await _factory()
    await _seed_bars(factory, "BTC/USDT", wobble=0.02)
    # A VaR cap so tiny that any real VaR breaches it.
    risk, writer = await _make(
        factory,
        [{"symbol": "BTC/USDT", "market_value": "60"}],
        max_portfolio_var_95=1e-9,
    )

    # BEFORE: empty return history -> VaR 0.0 -> the rule passes vacuously.
    before = risk.check_portfolio_risk()
    assert before.portfolio_var_95 == 0.0
    var_rule_before = next(
        r for r in before.rule_results if "var" in r.rule_name.lower()
    )
    assert var_rule_before.passed is True  # the silent failure

    await writer.write_once()

    # AFTER: real returns -> real VaR -> the same rule now rejects.
    after = risk.check_portfolio_risk()
    assert after.portfolio_var_95 > 0
    var_rule_after = next(
        r for r in after.rule_results if "var" in r.rule_name.lower()
    )
    assert var_rule_after.passed is False
    row = (await _rows(factory))[-1]
    assert "rules_failed" in (row.details or {})
    assert any("var" in name.lower() for name in row.details["rules_failed"])
    await engine.dispose()


async def test_closed_positions_leave_the_engine() -> None:
    """Stale exposure must not inflate concentration and VaR forever."""
    engine, factory = await _factory()
    await _seed_bars(factory, "BTC/USDT")
    risk, writer = await _make(factory, [{"symbol": "BTC/USDT", "market_value": "40"}])
    await writer.write_once()
    assert set(risk.positions) == {"BTC/USDT"}

    writer._positions_provider = _positions_provider([])  # book went flat
    await writer.write_once()
    assert risk.positions == {}
    await engine.dispose()


async def test_metrics_the_engine_cannot_support_stay_null() -> None:
    """No bars, no positions: every derived metric is NULL, never zero."""
    engine, factory = await _factory()
    risk, writer = await _make(factory, [])
    await writer.write_once()
    row = (await _rows(factory))[0]
    assert row.var_95 is None
    assert row.var_99 is None
    assert row.cvar_99 is None
    assert (row.details or {})["volatility"] is None
    # The breaker state is ALWAYS reportable -- it does not need history.
    assert (row.details or {})["circuit_breaker_state"] == "closed"
    await engine.dispose()


async def test_a_failed_iteration_does_not_stop_the_loop() -> None:
    engine, factory = await _factory()

    async def exploding_provider() -> list[dict[str, Any]]:
        raise RuntimeError("venue hiccup")

    risk = RiskManagerService(
        event_bus=InProcessEventBus(),  # type: ignore[arg-type]
        settings=_settings(),
    )
    writer = RiskMetricsWriter(
        session_factory=factory,
        risk_service=risk,
        positions_provider=exploding_provider,
    )
    with pytest.raises(RuntimeError):
        await writer.write_once()  # write_once itself propagates...
    # ...but the run() loop treats it as a logged skip, not a crash. The
    # loop's exception handling is exercised by starting and cancelling it.
    import asyncio

    task = asyncio.create_task(writer.run(interval_seconds=0.01))
    await asyncio.sleep(0.05)
    assert not task.done()  # still looping despite every iteration failing
    task.cancel()
    await engine.dispose()
