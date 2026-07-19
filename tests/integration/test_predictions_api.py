"""GET /api/v1/predictions/* read the predictions table (previously 501 stubs).

The worker has been persisting every PredictionReadyEvent since the soak
started, but the API endpoints in front of that table were never implemented --
the dashboard's prediction views were dead. These tests exercise the three
endpoints against seeded SQLite rows.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
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
from core.models.predictions import Prediction


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    """Render the model's Postgres JSONB columns as JSON on sqlite."""
    return "JSON"


_NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


def _row(symbol: str, direction: str, minutes_ago: int, confidence: float = 0.6) -> Prediction:
    return Prediction(
        predicted_at=_NOW - timedelta(minutes=minutes_ago),
        symbol=symbol,
        model_id="ensemble:xgboost,lightgbm",
        model_version=1,
        direction=direction,
        confidence=confidence,
        expected_return=0.001,
        horizon_minutes=5,
    )


@pytest.fixture
def client_with_predictions():
    # File-based SQLite + NullPool: the setup loop and the TestClient's loop
    # must see the same data (in-memory SQLite is per-connection).
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    seeded: list[Prediction] = [
        _row("BTC/USDT", "flat", minutes_ago=3),
        _row("BTC/USDT", "long", minutes_ago=2, confidence=0.71),
        _row("ETH/USDT", "short", minutes_ago=1),
    ]

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(
                AsyncBase.metadata.create_all, tables=[Prediction.__table__]
            )
        async with factory() as s:
            s.add_all(seeded)
            await s.commit()

    asyncio.run(_setup())

    async def _override_db():
        async with factory() as session:
            yield session

    def _override_user() -> dict[str, Any]:
        return {"user_id": "test-user", "role": "admin"}

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_user
    try:
        with TestClient(app) as client:
            yield client, seeded
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        asyncio.run(engine.dispose())
        os.unlink(path)


def test_latest_returns_rows_newest_first(client_with_predictions) -> None:
    client, _ = client_with_predictions
    resp = client.get("/api/v1/predictions/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert [p["symbol"] for p in body] == ["ETH/USDT", "BTC/USDT", "BTC/USDT"]
    assert body[0]["direction"] == "short"
    assert body[1]["confidence"] == pytest.approx(0.71)
    assert body[0]["timestamp"].startswith("2026-07-19T11:59")


def test_latest_filters_by_symbol_and_limit(client_with_predictions) -> None:
    client, _ = client_with_predictions
    resp = client.get("/api/v1/predictions/latest", params={"symbol": "BTC/USDT", "limit": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["symbol"] == "BTC/USDT"
    assert body[0]["direction"] == "long"  # newest BTC row


def test_history_returns_all_rows(client_with_predictions) -> None:
    client, _ = client_with_predictions
    resp = client.get("/api/v1/predictions/history")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_detail_found_and_missing(client_with_predictions) -> None:
    client, seeded = client_with_predictions
    known_id = seeded[1].id
    resp = client.get(f"/api/v1/predictions/{known_id}")
    assert resp.status_code == 200
    assert resp.json()["direction"] == "long"

    resp = client.get(f"/api/v1/predictions/{uuid.uuid4()}")
    assert resp.status_code == 404
