"""Durable model artifacts: DB blob mirror + startup restore/reconcile.

The deploy target has no volumes, so cloud-promoted artifacts die with the
container. Covers:
- ``AutoRetrainer._persist_artifacts_to_db`` mirrors every artifact file of a
  promoted version into ``model_artifact_blobs``.
- ``retrain()`` invokes the blob persist after a successful promote.
- ``restore_and_reconcile``: promote -> persist -> wipe-dir -> restore
  round-trip on sqlite + a tmp filesystem registry, including metadata
  reconciliation and idempotency.
- A stale ``model_metadata`` row (DB says v2 active, filesystem serves v1) is
  flipped to match the filesystem registry.
- Corrupt blobs and unsafe blob filenames are skipped without disturbing the
  serving champion.
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from core.models.ml_models import ModelArtifactBlob
from core.models.ml_models import ModelMetadata as DBModelMetadata
from services.continuous_learning.retrainer import AutoRetrainer
from services.prediction.models.base import TrainResult
from services.prediction.models.xgboost_model import XGBoostPredictor
from services.prediction.registry import ModelRegistry, restore_and_reconcile
from services.prediction.training.trainer import ModelTrainer


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    """Render the model_metadata table's Postgres JSONB columns as JSON on sqlite."""
    return "JSON"


_N_FEATURES = 4


def _make_factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _create_tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all,
            tables=[ModelArtifactBlob.__table__, DBModelMetadata.__table__],
        )


def _train_tiny_model(seed: int = 5) -> XGBoostPredictor:
    """Train a real (tiny) XGBoost predictor so save/load round-trips are honest."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(90, _N_FEATURES))
    y = np.tile([0, 1, 2], 30)  # all classes present in both splits
    model = XGBoostPredictor(
        classifier_params={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        regressor_params={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        feature_names=[f"f{i}" for i in range(_N_FEATURES)],
    )
    model.train(X[:69], y[:69], X[69:], y[69:])
    return model


def _make_retrainer(factory, registry: ModelRegistry, settings: Settings) -> AutoRetrainer:
    return AutoRetrainer(
        trainer=ModelTrainer(),
        registry=registry,
        settings=settings,
        session_factory=factory,
    )


def _adopt_placeholder_v1(registry: ModelRegistry) -> None:
    """Register + promote a v1 whose artifact dir exists but is never loaded."""
    v1_dir = registry.artifact_base / "xgboost" / "v1"
    v1_dir.mkdir(parents=True)
    (v1_dir / "classifier.joblib").write_bytes(b"placeholder")
    model_id = registry.adopt_version(
        "xgboost", 1, model_type="xgboost", metrics={"val_accuracy": 0.4}
    )
    registry.promote(model_id, 1)


# ---------------------------------------------------------------------------
# Persist -> wipe -> restore round-trip
# ---------------------------------------------------------------------------


async def test_persist_wipe_restore_round_trip(
    tmp_path: Path, mock_settings: Settings
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    # "Cloud" container: retrain promoted a version, blobs were persisted.
    cloud_registry = ModelRegistry(artifact_base=tmp_path / "cloud")
    model = _train_tiny_model()
    model_id, version = cloud_registry.register(model, "xgboost", {"val_accuracy": 0.5})
    cloud_registry.promote(model_id, version)

    retrainer = _make_retrainer(factory, cloud_registry, mock_settings)
    await retrainer._persist_artifacts_to_db("xgboost", version)

    artifact_dir = tmp_path / "cloud" / "xgboost" / "v1"
    on_disk = sorted(p.name for p in artifact_dir.iterdir() if p.is_file())
    assert "classifier.joblib" in on_disk and "regressor.joblib" in on_disk
    async with factory() as session:
        blobs = (await session.execute(select(ModelArtifactBlob))).scalars().all()
    assert sorted(b.filename for b in blobs) == on_disk
    assert all(b.model_name == "xgboost" and b.version == 1 for b in blobs)
    assert all(len(b.content) > 0 for b in blobs)
    for blob in blobs:
        assert blob.content == (artifact_dir / blob.filename).read_bytes()

    # Redeploy: brand-new filesystem, the promoted version's dir is gone.
    fresh_registry = ModelRegistry(artifact_base=tmp_path / "fresh")
    await restore_and_reconcile(fresh_registry, factory)

    restored, meta = fresh_registry.get_active("xgboost")
    assert meta.version == 1 and meta.is_active
    assert restored.predict(np.zeros(_N_FEATURES)).direction in {"short", "flat", "long"}

    # Reconcile inserted the missing metadata row as active (UI sees reality).
    async with factory() as session:
        rows = (await session.execute(select(DBModelMetadata))).scalars().all()
    assert [(r.model_name, r.version, r.is_active) for r in rows] == [("xgboost", 1, True)]

    # Second startup is a no-op: no duplicate registrations.
    await restore_and_reconcile(fresh_registry, factory)
    assert len(fresh_registry.list_versions("xgboost")) == 1

    await engine.dispose()


# ---------------------------------------------------------------------------
# retrain() wiring: a successful promote persists blobs
# ---------------------------------------------------------------------------


async def test_retrain_persists_blobs_after_promote(
    tmp_path: Path, mock_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)
    registry = ModelRegistry(artifact_base=tmp_path / "registry")
    retrainer = _make_retrainer(factory, registry, mock_settings)

    # Stub the heavy pieces (data load + training); registry/promote/mirror/
    # persist run for real.
    tiny = _train_tiny_model(seed=6)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, _N_FEATURES))
    y = np.tile([0, 1, 2], 10)
    returns = rng.normal(size=30) * 0.01
    names = [f"f{i}" for i in range(_N_FEATURES)]

    async def fake_load(self, model_id):  # noqa: ANN001, ANN202, ARG001
        return X[:24], y[:24], returns[:24], X[24:], y[24:], returns[24:], names

    def fake_train_model(self, **kwargs):  # noqa: ANN001, ANN003, ANN202, ARG001
        result = TrainResult(
            train_loss=0.1, val_loss=0.2, train_accuracy=0.9,
            val_accuracy=0.9, epochs_trained=1,
        )
        return result, tiny

    monkeypatch.setattr(AutoRetrainer, "_load_training_data", fake_load)
    monkeypatch.setattr(ModelTrainer, "train_model", fake_train_model)
    monkeypatch.setattr("services.continuous_learning.retrainer.MIN_VAL_ACCURACY", 0.0)
    # These tests pin training/promotion mechanics, not the phase 2
    # live-transfer gate (tested in test_live_replay/test_retrainer_gate);
    # without live rows in their fixtures the gate would fail closed.
    from services.continuous_learning.live_replay import GateDecision

    async def _vacuous_gate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return GateDecision(
            promote=True, reason="test_vacuous_pass", challenger_ic=None,
            champion_ic=None, n_rows=0, span_days=None,
        )

    monkeypatch.setattr(
        "services.continuous_learning.live_replay.live_transfer_gate",
        _vacuous_gate,
    )

    result = await retrainer.retrain("xgboost")
    assert result.get("version") == 1, f"retrain skipped: {result}"

    async with factory() as session:
        blobs = (await session.execute(select(ModelArtifactBlob))).scalars().all()
    assert blobs, "promote did not persist artifact blobs"
    assert all(b.model_name == "xgboost" and b.version == 1 for b in blobs)
    assert {b.filename for b in blobs} >= {"classifier.joblib", "regressor.joblib"}

    await engine.dispose()


# ---------------------------------------------------------------------------
# Reconcile: stale DB rows are flipped to match the filesystem registry
# ---------------------------------------------------------------------------


async def test_reconcile_flips_stale_metadata_row(tmp_path: Path) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    # Filesystem serves v1; the DB still claims a (lost, un-mirrored) v2 is
    # active -- the exact lie observed live after a redeploy.
    registry = ModelRegistry(artifact_base=tmp_path / "registry")
    _adopt_placeholder_v1(registry)

    now = datetime.now(UTC)
    async with factory() as session:
        for version, is_active in ((1, False), (2, True)):
            session.add(
                DBModelMetadata(
                    model_name="xgboost",
                    model_type="xgboost",
                    version=version,
                    hyperparameters={"retrained": True},
                    artifact_path=f"model_artifacts/xgboost/v{version}",
                    trained_at=now,
                    is_active=is_active,
                    created_at=now,
                )
            )
        await session.commit()

    await restore_and_reconcile(registry, factory)

    async with factory() as session:
        rows = (
            await session.execute(
                select(DBModelMetadata).order_by(DBModelMetadata.version)
            )
        ).scalars().all()
    assert [(r.version, r.is_active) for r in rows] == [(1, True), (2, False)]
    # No blobs for v2 -> nothing was resurrected on disk.
    assert registry.latest_version("xgboost") == 1

    await engine.dispose()


# ---------------------------------------------------------------------------
# Corrupt / unsafe blobs are skipped
# ---------------------------------------------------------------------------


async def test_corrupt_blob_is_skipped_and_champion_keeps_serving(
    tmp_path: Path,
) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    registry = ModelRegistry(artifact_base=tmp_path / "registry")
    _adopt_placeholder_v1(registry)

    async with factory() as session:
        session.add(
            ModelArtifactBlob(
                model_name="xgboost",
                version=2,
                filename="classifier.joblib",
                content=b"definitely not a joblib payload",
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    # Must not raise: restore is best-effort.
    await restore_and_reconcile(registry, factory)

    # v2 failed validation -> never registered; v1 stays the active champion.
    assert registry.latest_version("xgboost") == 1
    assert [(e.model_name, e.version) for e in registry.active_entries()] == [("xgboost", 1)]

    # Reconcile reflects the filesystem truth: only v1 active in the DB.
    async with factory() as session:
        rows = (await session.execute(select(DBModelMetadata))).scalars().all()
    assert [(r.version, r.is_active) for r in rows] == [(1, True)]

    await engine.dispose()


async def test_unsafe_blob_filename_is_skipped(tmp_path: Path) -> None:
    engine, factory = _make_factory()
    await _create_tables(engine)

    base = tmp_path / "registry"
    registry = ModelRegistry(artifact_base=base)
    async with factory() as session:
        session.add(
            ModelArtifactBlob(
                model_name="xgboost",
                version=1,
                filename="../../evil.joblib",
                content=b"x",
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    await restore_and_reconcile(registry, factory)

    # Nothing registered and nothing written outside (or inside) the dir.
    assert registry.latest_version("xgboost") == 0
    assert not (base / "xgboost" / "v1").exists()
    assert not (tmp_path / "evil.joblib").exists()

    await engine.dispose()
