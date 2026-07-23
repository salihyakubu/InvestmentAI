"""Durable learning state: outcomes persist to the predictions table and the
continuous-learning service rehydrates them on startup.

Before this, all learning state was in-memory, so every deploy reset the
evaluator window and the drift detector's >=1000-resolved-outcome gate to zero.
These tests pin: (1) the outcome resolver writes actual_direction / actual_return
/ resolved_at back to the matching predictions row (by event_id); (2) a fresh
service rehydrates recent resolved rows so the evaluator reports the pre-restart
sample size and the tracked-prediction history is repopulated; (3) rehydration
against an absent table is a clean start, never a startup failure.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.events.base import InProcessEventBus
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from core.models.predictions import Prediction
from services.continuous_learning.evaluator import ModelEvaluator
from services.continuous_learning.service import ContinuousLearningService

_MODEL_ID = "ensemble:xgboost,lightgbm,catboost"


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


def _settings() -> Settings:
    return Settings(
        trading_mode="paper",
        jwt_secret="test-secret-not-the-published-default-value",
    )


def _make_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine: AsyncEngine, *, with_predictions: bool = True) -> None:
    tables = [OHLCVRecord.__table__]
    if with_predictions:
        tables.append(Prediction.__table__)
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=tables)


def _make_service(
    session_factory: async_sessionmaker[AsyncSession],
    evaluator: ModelEvaluator | MagicMock,
) -> ContinuousLearningService:
    return ContinuousLearningService(
        event_bus=InProcessEventBus(),  # type: ignore[arg-type]
        settings=_settings(),
        evaluator=evaluator,
        retrainer=MagicMock(),
        drift_detector=MagicMock(),
        feedback_loop=MagicMock(),
        session_factory=session_factory,
    )


def _bar(t: datetime, close: str) -> OHLCVRecord:
    price = Decimal(close)
    return OHLCVRecord(
        time=t, symbol="BTC/USDT", timeframe="1m", asset_class="crypto",
        open=price, high=price, low=price, close=price,
        volume=Decimal("1"), source="test", ingested_at=datetime.now(UTC),
    )


def _prediction_row(event_id: str, predicted_at: datetime, **overrides: Any) -> Prediction:
    fields: dict[str, Any] = dict(
        predicted_at=predicted_at, symbol="BTC/USDT", model_id=_MODEL_ID,
        model_version=1, direction="long", confidence=0.8, expected_return=0.001,
        horizon_minutes=5, created_at=predicted_at, event_id=event_id,
    )
    fields.update(overrides)
    return Prediction(**fields)


@pytest.mark.asyncio
async def test_resolution_writes_outcome_back_to_predictions_row() -> None:
    engine, factory = _make_db()
    await _create_tables(engine)

    now = datetime.now(UTC)
    t0 = now - timedelta(minutes=30)
    async with factory() as session:
        # A predictions row awaiting its outcome, plus the two bars the resolver
        # needs: one at the prediction time and one 5 minutes later (+1% move).
        session.add(_prediction_row("evt-1", t0))
        session.add(_bar(t0, "100"))
        session.add(_bar(t0 + timedelta(minutes=5), "101"))
        await session.commit()

    svc = _make_service(factory, MagicMock(spec=ModelEvaluator))
    svc._tracked_predictions[_MODEL_ID] = [
        {
            "prediction_id": "evt-1", "symbol": "BTC/USDT",
            "predicted": "long", "confidence": 0.8, "timestamp": t0,
        }
    ]

    await svc._resolve_outcomes_once()

    async with factory() as session:
        row = (
            await session.execute(select(Prediction).where(Prediction.event_id == "evt-1"))
        ).scalar_one()
    assert row.actual_direction == "long"
    assert row.actual_return == pytest.approx(0.01, rel=1e-3)
    assert row.resolved_at is not None
    await engine.dispose()


@pytest.mark.asyncio
async def test_rehydrate_seeds_evaluator_and_tracker() -> None:
    engine, factory = _make_db()
    await _create_tables(engine)

    base = datetime.now(UTC) - timedelta(hours=2)
    async with factory() as session:
        for i in range(150):
            session.add(
                _prediction_row(
                    f"evt-{i}", base + timedelta(minutes=i),
                    actual_direction=("long" if i % 2 == 0 else "flat"),
                    actual_return=0.01, resolved_at=base + timedelta(minutes=i + 5),
                )
            )
        # An UNresolved row must not be rehydrated.
        session.add(_prediction_row("evt-open", datetime.now(UTC)))
        await session.commit()

    evaluator = ModelEvaluator()
    svc = _make_service(factory, evaluator)
    await svc._rehydrate_from_db()

    report = evaluator.evaluate_live_performance(_MODEL_ID)
    assert report.sample_size == 150
    assert len(svc._tracked_predictions[_MODEL_ID]) == 150
    assert all("actual" in r for r in svc._tracked_predictions[_MODEL_ID])
    # Oldest-first insertion order preserved (drift midpoint split relies on it).
    ts = [r["timestamp"] for r in svc._tracked_predictions[_MODEL_ID]]
    assert ts == sorted(ts)
    assert "evt-open" not in {r["prediction_id"] for r in svc._tracked_predictions[_MODEL_ID]}
    await engine.dispose()


@pytest.mark.asyncio
async def test_rehydrate_missing_table_is_clean_start() -> None:
    engine, factory = _make_db()
    await _create_tables(engine, with_predictions=False)  # no predictions table

    evaluator = ModelEvaluator()
    svc = _make_service(factory, evaluator)
    await svc._rehydrate_from_db()  # must not raise

    assert svc._tracked_predictions == {}
    assert evaluator.evaluate_live_performance(_MODEL_ID).sample_size == 0
    await engine.dispose()
