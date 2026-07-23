"""Backtest API round-trip: run -> poll -> results (was a dead 501 stub).

Uses file-based SQLite with dependency overrides; the runner's network fetch
is monkeypatched to synthetic series so no test touches the network. Also pins
the stale-running-job marking and the 409 on premature result fetches.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import NullPool

import services.backtesting.runner as runner
from api.dependencies import get_current_user, get_db
from api.main import app
from core.models.backtests import BacktestJob
from core.models.base import AsyncBase


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


async def _fake_fetch(symbols: list[str], start: datetime, end: datetime):
    rng = np.random.default_rng(11)
    n = 900
    base = datetime(2020, 1, 1, tzinfo=UTC)
    dates = [base + timedelta(days=i) for i in range(n)]
    return {
        s: (dates, 100.0 * np.cumprod(1 + rng.normal(0.0, 0.012, n))) for s in symbols
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _setup() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(
                AsyncBase.metadata.create_all, tables=[BacktestJob.__table__]
            )

    asyncio.run(_setup())

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_current_user] = lambda: {"user_id": "t", "role": "admin"}
    monkeypatch.setattr(runner, "fetch_series", _fake_fetch)
    # Background task uses the module-level factory getter -- point it at ours.
    monkeypatch.setattr(
        "core.models.base.get_async_session_factory", lambda: factory
    )
    try:
        with TestClient(app) as c:
            yield c, factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        asyncio.run(engine.dispose())
        os.unlink(path)


def _run_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "start_date": "2020-01-01",
        "end_date": "2022-06-01",
        "symbols": ["AAA", "BBB"],
        "strategy": "edge_harness",
        "initial_capital": 10_000,
        "commission": 0.0005,
    }
    cfg.update(overrides)
    return cfg


def test_run_poll_results_round_trip(client) -> None:
    c, _ = client
    resp = c.post("/api/v1/backtest/run", json=_run_config())
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["id"]

    # Poll until the background task completes (training is seconds-scale).
    deadline = time.time() + 120
    status = "queued"
    while time.time() < deadline:
        status = c.get(f"/api/v1/backtest/status/{job_id}").json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert status == "completed", c.get(f"/api/v1/backtest/status/{job_id}").json()

    result = c.get(f"/api/v1/backtest/results/{job_id}").json()
    assert result["id"] == job_id
    assert {"total_return", "sharpe_ratio", "equity_curve", "trades", "verdict"} <= set(result)
    assert result["config"]["symbols"] == ["AAA", "BBB"]

    history = c.get("/api/v1/backtest/history").json()
    assert any(j["id"] == job_id for j in history)


def test_results_before_completion_is_409_and_validation(client) -> None:
    c, factory = client

    async def _seed_running() -> str:
        async with factory() as s:
            job = BacktestJob(status="running", config={}, started_at=datetime.now(UTC))
            s.add(job)
            await s.commit()
            await s.refresh(job)
            return str(job.id)

    running_id = asyncio.run(_seed_running())
    assert c.get(f"/api/v1/backtest/results/{running_id}").status_code == 409
    assert c.get(f"/api/v1/backtest/status/{uuid.uuid4()}").status_code == 404
    assert (
        c.post("/api/v1/backtest/run", json=_run_config(end_date="2019-01-01")).status_code
        == 422
    )
    assert (
        c.post("/api/v1/backtest/run", json=_run_config(symbols=[])).status_code == 422
    )


def test_stale_running_job_marked_failed(client) -> None:
    c, factory = client

    async def _seed_stale() -> str:
        async with factory() as s:
            job = BacktestJob(
                status="running",
                config={},
                started_at=datetime.now(UTC) - timedelta(hours=2),
            )
            s.add(job)
            await s.commit()
            await s.refresh(job)
            return str(job.id)

    stale_id = asyncio.run(_seed_stale())
    body = c.get(f"/api/v1/backtest/status/{stale_id}").json()
    assert body["status"] == "failed"
    assert "orphaned" in body["error"]
