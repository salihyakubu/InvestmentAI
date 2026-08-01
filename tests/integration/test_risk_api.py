"""/risk/metrics/latest: absence reads UNKNOWN, presence reads the engine.

Pins the two halves of the false-green fix: with no row the API must say
"not reported" rather than synthesise an all-clear, and with a row it must
carry the engine's own breaker state string plus the fields the UI renders.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import NullPool

from api.dependencies import get_current_user, get_db
from api.main import app
from core.models.base import AsyncBase
from core.models.risk import RiskMetric


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(
                AsyncBase.metadata.create_all, tables=[RiskMetric.__table__]
            )

    asyncio.run(_setup())

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "t", "role": "admin"}
    try:
        with TestClient(app) as c:
            yield c, factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        asyncio.run(engine.dispose())
        os.unlink(path)


def test_no_row_reports_unreported_not_all_clear(client) -> None:
    c, _ = client
    body = c.get("/api/v1/risk/metrics/latest").json()
    assert body["reported"] is False
    # Absent must be null, never a fabricated zero or a green False.
    assert body["circuit_breaker_active"] is None
    assert body["circuit_breaker_state"] is None
    assert body["var_95"] is None


def test_a_written_row_carries_the_engines_state(client) -> None:
    c, factory = client

    async def _seed() -> None:
        async with factory() as session:
            session.add(
                RiskMetric(
                    time=datetime.now(UTC),
                    var_95=1.23,
                    var_99=2.34,
                    cvar_95=1.5,
                    cvar_99=2.9,
                    max_drawdown=0.0115,
                    current_drawdown=0.002,
                    beta=None,
                    correlation_max=0.41,
                    concentration_max=0.28,
                    circuit_breaker_active=True,
                    details={
                        "circuit_breaker_state": "open",
                        "volatility": 0.0123,
                        "daily_pnl_pct": -0.08,
                        "rules_failed": ["MaxDailyDrawdownRule"],
                    },
                )
            )
            await session.commit()

    asyncio.run(_seed())
    body: dict[str, Any] = c.get("/api/v1/risk/metrics/latest").json()
    assert body["reported"] is True
    assert body["circuit_breaker_state"] == "open"
    assert body["circuit_breaker_active"] is True
    # Failing rules are their own field; the reason belongs to the breaker.
    assert body["failing_rules"] == ["MaxDailyDrawdownRule"]
    assert "breached the loss limit" in body["circuit_breaker_reason"]
    assert body["var_99"] == pytest.approx(2.34)
    assert body["cvar_99"] == pytest.approx(2.9)
    assert body["volatility"] == pytest.approx(0.0123)
    # beta stays null on the wire: nothing computes it.
    assert "beta" not in body or body.get("beta") is None


def test_failing_rules_on_a_closed_breaker_do_not_forge_a_reason(client) -> None:
    """The Aug 1 dashboard bug: a green CLOSED card must never carry a
    'Reason: failed rules' line. Rule failures are reported separately."""
    c, factory = client

    async def _seed() -> None:
        async with factory() as session:
            session.add(
                RiskMetric(
                    time=datetime.now(UTC),
                    var_95=0.01,
                    circuit_breaker_active=False,
                    details={
                        "circuit_breaker_state": "closed",
                        "daily_pnl_pct": -0.003,
                        "rules_failed": ["MaxPositionSize", "MaxConcentration"],
                    },
                )
            )
            await session.commit()

    asyncio.run(_seed())
    body = c.get("/api/v1/risk/metrics/latest").json()
    assert body["circuit_breaker_state"] == "closed"
    assert body["circuit_breaker_reason"] is None
    assert body["failing_rules"] == ["MaxPositionSize", "MaxConcentration"]
