"""PredictionPersistenceService: PredictionReadyEvents on the predictions.ready
stream insert rows into the predictions table (dashboard counts / live outcomes).
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core.events.base import Event, InProcessEventBus
from core.events.signal_events import PredictionReadyEvent
from core.models.base import AsyncBase
from core.models.predictions import Prediction
from services.persistence.prediction_sync import PredictionPersistenceService


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    """Render the model's Postgres JSONB columns as JSON on sqlite."""
    return "JSON"


def _make_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=[Prediction.__table__])


async def _fetch_predictions(factory):
    async with factory() as session:
        return (await session.execute(select(Prediction))).scalars().all()


@pytest.mark.asyncio
async def test_prediction_event_persists_row() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    svc = PredictionPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    await svc.handle_event(
        PredictionReadyEvent(
            symbol="AAPL", direction="long", confidence=0.72,
            expected_return=0.0031, model_id="xgb-v3",
            source_service="prediction-service",
        )
    )

    rows = await _fetch_predictions(factory)
    assert len(rows) == 1
    row = rows[0]
    assert row.symbol == "AAPL"
    assert row.direction == "long"
    assert row.confidence == pytest.approx(0.72)
    assert row.expected_return == pytest.approx(0.0031)
    assert row.model_id == "xgb-v3"
    assert row.horizon_minutes == 5
    assert row.predicted_at is not None
    await engine.dispose()


@pytest.mark.asyncio
async def test_non_prediction_event_is_ignored() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    svc = PredictionPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    await svc.handle_event(
        Event(event_type="FeaturesReadyEvent", source_service="feature-pipeline")
    )

    assert await _fetch_predictions(factory) == []
    await engine.dispose()


@pytest.mark.asyncio
async def test_none_expected_return_persists_as_null() -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    svc = PredictionPersistenceService(event_bus=InProcessEventBus(), session_factory=factory)
    await svc.handle_event(
        PredictionReadyEvent(
            symbol="BTCUSDT", direction="flat", confidence=0.5,
            expected_return=None, model_id="lstm-v1",
            source_service="prediction-service",
        )
    )

    rows = await _fetch_predictions(factory)
    assert len(rows) == 1
    assert rows[0].expected_return is None
    assert rows[0].symbol == "BTCUSDT"
    await engine.dispose()
