"""/models/feature-drift: the wire contract for the drift instrument.

Pins the two states the card must render truthfully: enough history gives
per-index PSI with the generation hash; a thin record gives an explicit
non-computable payload with the reason -- never zeros dressed as stability.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
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
from core.models.predictions import Prediction


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
                AsyncBase.metadata.create_all, tables=[Prediction.__table__]
            )

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


async def _seed(
    factory, n: int, start: datetime, shift: float = 0.0, step_minutes: int = 5
) -> None:
    rng = np.random.default_rng(int(start.timestamp()) % 1000)
    async with factory() as session:
        for i in range(n):
            when = start + timedelta(minutes=step_minutes * i)
            vec = [float(x) for x in rng.normal(shift, 1, 4)]
            session.add(
                Prediction(
                    symbol="BTC/USDT", model_id="ensemble:test", model_version=1,
                    direction="flat", confidence=0.5, expected_return=0.001,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    features_used={"v": vec, "h": "deadbeefcafe0000"},
                )
            )
        await session.commit()


def test_drift_contract_with_planted_shift(client) -> None:
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(_seed(factory, 1500, now - timedelta(days=10)))
    asyncio.run(_seed(factory, 700, now - timedelta(days=1), shift=3.0, step_minutes=2))
    body: dict[str, Any] = c.get("/api/v1/models/feature-drift").json()

    assert body["computable"] is True
    assert body["generation_hash"] == "deadbeefcafe0000"
    assert body["n_features"] == 4
    assert body["n_unmeasurable"] == 0
    assert body["n_symbols_measured"] == 1
    assert body["n_reference"] >= 1000
    assert body["n_recent"] >= 500
    # Every feature shifted by 3 sigma: the top of the list must show it.
    assert body["top_drifted"] and body["top_drifted"][0]["psi"] > 0.25
    assert body["share_significant"] == 1.0
    assert body["worst"]["symbol"] == "BTC/USDT"
    # The card dereferences thresholds on the render path: pin it.
    assert body["thresholds"] == {"moderate": 0.1, "significant": 0.25}
    assert any("training distribution" in note for note in body["notes"])
    assert any("PER SYMBOL" in note or "per symbol" in note.lower() for note in body["notes"])


def test_thin_history_is_an_explicit_refusal_not_zeros(client) -> None:
    c, factory = client
    now = datetime.now(UTC)
    asyncio.run(_seed(factory, 200, now - timedelta(days=1)))
    body = c.get("/api/v1/models/feature-drift").json()

    assert body["computable"] is False
    assert "insufficient" in body["reason"]
    assert body["top_drifted"] == []
    assert body["share_significant"] is None
