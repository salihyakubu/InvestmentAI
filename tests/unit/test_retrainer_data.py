"""AutoRetrainer data loading + end-to-end retrain against a sqlite ohlcv table.

Covers:
- ``_load_training_data`` builds real train/val arrays from 1m bars in the DB
  (Decimal columns, timeframe and lookback filtering, per-symbol min-bars skip).
- The all-``None`` skip contract when the table is empty.
- ``_validate_new_model`` enforces the absolute accuracy floor AND the
  champion comparison.
- ``retrain()`` end-to-end: trains, gates, registers + promotes through a real
  filesystem ``ModelRegistry``, and mirrors metadata into ``model_metadata``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from config.settings import Settings
from core.models.base import AsyncBase
from core.models.market_data import OHLCVRecord
from core.models.ml_models import ModelMetadata as DBModelMetadata
from services.continuous_learning.retrainer import AutoRetrainer
from services.prediction.registry import ModelRegistry
from services.prediction.training.dataset_builder import HORIZON, MIN_BARS, WINDOW
from services.prediction.training.trainer import ModelTrainer


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    """Render the model_metadata table's Postgres JSONB columns as JSON on sqlite."""
    return "JSON"


def _make_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine, with_metadata: bool = False) -> None:
    tables = [OHLCVRecord.__table__]
    if with_metadata:
        tables.append(DBModelMetadata.__table__)
    async with engine.begin() as conn:
        await conn.run_sync(AsyncBase.metadata.create_all, tables=tables)


async def _seed_bars(
    factory,
    symbol: str,
    n: int,
    *,
    seed: int,
    timeframe: str = "1m",
    end: datetime | None = None,
) -> None:
    """Insert *n* 1-minute bars for *symbol* ending at *end* (default: now)."""
    end = end or datetime.now(UTC)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=n))
    high = close * (1.0 + rng.uniform(0.0001, 0.004, size=n))
    low = close * (1.0 - rng.uniform(0.0001, 0.004, size=n))
    open_ = np.concatenate(([100.0], close[:-1]))
    volume = rng.uniform(1_000.0, 5_000.0, size=n)

    rows = [
        OHLCVRecord(
            time=end - timedelta(minutes=n - i),
            symbol=symbol,
            timeframe=timeframe,
            asset_class="crypto",
            open=Decimal(f"{open_[i]:.8f}"),
            high=Decimal(f"{high[i]:.8f}"),
            low=Decimal(f"{low[i]:.8f}"),
            close=Decimal(f"{close[i]:.8f}"),
            volume=Decimal(f"{volume[i]:.8f}"),
            source="test",
            ingested_at=end,
        )
        for i in range(n)
    ]
    async with factory() as session:
        session.add_all(rows)
        await session.commit()


def _make_retrainer(factory, tmp_path: Path, settings: Settings) -> AutoRetrainer:
    return AutoRetrainer(
        trainer=ModelTrainer(),
        registry=ModelRegistry(artifact_base=tmp_path / "registry"),
        settings=settings,
        session_factory=factory,
    )


# ---------------------------------------------------------------------------
# _load_training_data
# ---------------------------------------------------------------------------


async def test_load_training_data_builds_arrays_from_db(
    tmp_path: Path, mock_settings: Settings
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    n = 260
    await _seed_bars(factory, "BTC/USDT", n, seed=7)
    # Below the min-bars threshold -> skipped by the dataset builder.
    await _seed_bars(factory, "TINY/USDT", 50, seed=8)
    # Outside the configured lookback window -> excluded by the query.
    await _seed_bars(
        factory, "OLD/USDT", n, seed=9,
        end=datetime.now(UTC) - timedelta(days=mock_settings.retrain_lookback_days + 3),
    )
    # Wrong timeframe -> excluded by the query.
    await _seed_bars(factory, "BTC/USDT", 250, seed=10, timeframe="1d")

    retrainer = _make_retrainer(factory, tmp_path, mock_settings)
    X_train, y_train, r_train, X_val, y_val, r_val, names = (
        await retrainer._load_training_data("xgboost")
    )

    # Exactly the one eligible symbol contributed rows.
    windows = (n - WINDOW) // HORIZON + 1
    labeled = windows - HORIZON
    exp_train = int(labeled * 0.8)
    assert X_train is not None and y_train is not None and r_train is not None
    assert X_val is not None and y_val is not None and r_val is not None
    assert names is not None and names == sorted(names) and len(names) > 0
    assert X_train.shape == (exp_train, len(names))
    assert X_val.shape == (labeled - exp_train, len(names))
    assert len(y_train) == len(r_train) == exp_train
    assert len(y_val) == len(r_val) == labeled - exp_train
    assert X_train.dtype == np.float64
    assert set(np.unique(np.concatenate([y_train, y_val]))) <= {0, 1, 2}

    await engine.dispose()


async def test_load_training_data_empty_table_returns_all_none(
    tmp_path: Path, mock_settings: Settings
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    retrainer = _make_retrainer(factory, tmp_path, mock_settings)
    result = await retrainer._load_training_data("xgboost")

    assert result == (None, None, None, None, None, None, None)
    await engine.dispose()


async def test_load_training_data_only_short_symbols_returns_all_none(
    tmp_path: Path, mock_settings: Settings
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    await _seed_bars(factory, "TINY/USDT", MIN_BARS - 1, seed=11)

    retrainer = _make_retrainer(factory, tmp_path, mock_settings)
    result = await retrainer._load_training_data("xgboost")

    assert result == (None, None, None, None, None, None, None)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Challenger validation gate
# ---------------------------------------------------------------------------


def test_validate_new_model_enforces_floor_and_champion() -> None:
    # Below the absolute floor -> rejected even with no champion.
    assert AutoRetrainer._validate_new_model({"val_accuracy": 0.20}, {}) is False
    assert AutoRetrainer._validate_new_model({"val_accuracy": 0.20}, {"val_accuracy": 0.1}) is False
    # Above the floor with no champion -> accepted.
    assert AutoRetrainer._validate_new_model({"val_accuracy": 0.40}, {}) is True
    # Above the floor but worse than the champion -> rejected.
    assert AutoRetrainer._validate_new_model(
        {"val_accuracy": 0.40}, {"val_accuracy": 0.50}
    ) is False
    # Above the floor and at least as good as the champion -> accepted.
    assert AutoRetrainer._validate_new_model(
        {"val_accuracy": 0.50}, {"val_accuracy": 0.50}
    ) is True
    assert AutoRetrainer._validate_new_model(
        {"val_accuracy": 0.55}, {"val_accuracy": 0.50}
    ) is True


# ---------------------------------------------------------------------------
# retrain() end-to-end (tiny trees, hyperopt off)
# ---------------------------------------------------------------------------


async def test_retrain_end_to_end_promotes_and_mirrors_db(
    tmp_path: Path, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine, with_metadata=True)
    await _seed_bars(factory, "BTC/USDT", 400, seed=21)
    await _seed_bars(factory, "ETH/USDT", 400, seed=22)

    # Keep the test fast + deterministic: tiny trees, no hyperopt, and no
    # dependence on synthetic-data accuracy clearing the absolute floor (the
    # floor itself is unit-tested above).
    captured: dict[str, object] = {}
    orig_train_model = ModelTrainer.train_model

    def fast_train_model(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        kwargs["hyperopt"] = False
        kwargs["extra_params"] = {"n_estimators": 10, "max_depth": 2}
        captured.update(kwargs)
        return orig_train_model(self, *args, **kwargs)

    monkeypatch.setattr(ModelTrainer, "train_model", fast_train_model)
    monkeypatch.setattr("services.continuous_learning.retrainer.MIN_VAL_ACCURACY", 0.0)

    retrainer = _make_retrainer(factory, tmp_path, mock_settings)
    result = await retrainer.retrain("xgboost")

    # Trained + promoted.
    assert "new_model_id" in result, f"retrain skipped: {result}"
    assert result["version"] == 1
    assert "val_accuracy" in result["metrics"]

    # Real forward returns were passed through to the trainer.
    assert isinstance(captured.get("returns_train"), np.ndarray)
    assert isinstance(captured.get("returns_val"), np.ndarray)

    # The filesystem registry serves the new version.
    model, meta = retrainer._registry.get_active("xgboost")
    assert meta.version == 1 and meta.is_active
    assert model.predict(np.zeros(captured["X_train"].shape[1])).direction in {
        "short", "flat", "long",
    }

    # The DB mirror row exists and is active (backs GET /api/v1/models).
    async with factory() as session:
        rows = (await session.execute(select(DBModelMetadata))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.model_name == "xgboost"
    assert row.model_type == "xgboost"
    assert row.version == 1
    assert row.is_active is True
    assert row.hyperparameters == {"retrained": True}
    assert row.validation_metrics is not None
    assert "val_accuracy" in row.validation_metrics
    assert row.artifact_path.endswith("v1")

    await engine.dispose()


async def test_retrain_skips_when_no_data(
    tmp_path: Path, mock_settings: Settings
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    retrainer = _make_retrainer(factory, tmp_path, mock_settings)
    result = await retrainer.retrain("xgboost")

    assert result == {"skipped": True, "reason": "no_training_data"}
    await engine.dispose()
