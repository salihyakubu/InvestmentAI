"""/models/learning-metrics: the wire contract for the learning instrument.

Pins the fetch-time de-overlap (only minute % 5 == 0 predictions are used),
the era segmentation from promotion dates, and honest nulls end to end.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import NullPool

import api.routers.models as models_router
from api.dependencies import get_current_user, get_db
from api.main import app
from core.models.base import AsyncBase
from core.models.ml_models import ModelMetadata
from core.models.predictions import Prediction


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


_TABLES = [Prediction.__table__, ModelMetadata.__table__]


@pytest.fixture
def client():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(AsyncBase.metadata.create_all, tables=_TABLES)

    asyncio.run(_setup())

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "t", "role": "admin"}
    models_router._learning_cache.clear()
    try:
        with TestClient(app) as c:
            yield c, factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        models_router._learning_cache.clear()
        asyncio.run(engine.dispose())
        os.unlink(path)


async def _seed(factory, n_per_day: int = 300, days: int = 2) -> None:
    base = datetime(2026, 7, 20, tzinfo=UTC)
    async with factory() as session:
        session.add(
            ModelMetadata(
                model_name="ensemble", model_type="xgboost", version=1,
                artifact_path="x", is_active=True, hyperparameters={},
                trained_at=base, created_at=base,
            )
        )
        for d in range(days):
            for i in range(n_per_day):
                when = base + timedelta(days=d, minutes=i)
                sign = 1 if i % 2 else -1
                session.add(
                    Prediction(
                        symbol="BTC/USDT",
                        model_id="ensemble:test",
                        model_version=1,
                        direction="flat",
                        confidence=0.5,
                        expected_return=0.001 * sign,
                        horizon_minutes=5,
                        predicted_at=when,
                        created_at=when,
                        actual_return=0.002 * sign,  # perfectly aligned signal
                        actual_direction="long" if sign > 0 else "short",
                        resolved_at=when + timedelta(minutes=6),
                    )
                )
        await session.commit()


def test_contract_and_deoverlap(client) -> None:
    c, factory = client
    asyncio.run(_seed(factory))
    body: dict[str, Any] = c.get("/api/v1/models/learning-metrics").json()

    # Fetch-time de-overlap: 600 seeded rows -> only minute%5==0 used.
    assert body["observations_used"] == 600 // 5
    assert len(body["eras"]) == 1
    era = body["eras"][0]
    # A perfectly aligned signal must show a strongly positive live IC.
    assert era["mean_ic"] is not None and era["mean_ic"] > 0.9
    assert era["sign_agreement"] is not None and era["sign_agreement"] > 0.9
    assert len(body["daily"]) == 2
    assert body["daily"][0]["abstention_rate"] == 1.0
    assert any("PR #59" in note for note in body["notes"])


def test_empty_history_serves_nulls_not_zeros(client) -> None:
    c, factory = client

    async def _promo_only() -> None:
        async with factory() as session:
            session.add(
                ModelMetadata(
                    model_name="ensemble", model_type="xgboost", version=1,
                    artifact_path="x", is_active=True, hyperparameters={},
                    trained_at=datetime(2026, 7, 20, tzinfo=UTC),
                    created_at=datetime(2026, 7, 20, tzinfo=UTC),
                )
            )
            await session.commit()

    asyncio.run(_promo_only())
    body = c.get("/api/v1/models/learning-metrics").json()
    assert body["observations_used"] == 0
    assert body["daily"] == []
    era = body["eras"][0]
    assert era["mean_ic"] is None  # absent, never 0.00
