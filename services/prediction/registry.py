"""Model registry for versioned storage and promotion of trained models."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.ml_models import ModelArtifactBlob
from core.models.ml_models import ModelMetadata as DBModelMetadata
from services.prediction.models.base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata record for a registered model version."""

    model_id: str
    model_name: str
    model_type: str
    version: int
    is_active: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str = ""
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Filesystem + JSON-based model registry.

    Artefacts are saved under ``artifact_base/<model_name>/v<version>/``.
    A ``registry.json`` file in ``artifact_base`` tracks all metadata.
    If a *db_session* is provided it is reserved for future database
    integration but not used in the current implementation.
    """

    _REGISTRY_FILE = "registry.json"

    def __init__(self, artifact_base: Path, db_session: Any = None) -> None:
        self.artifact_base = artifact_base
        self.artifact_base.mkdir(parents=True, exist_ok=True)
        self._db_session = db_session
        self._registry_path = self.artifact_base / self._REGISTRY_FILE
        self._entries: list[ModelMetadata] = self._load_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: BasePredictor,
        model_name: str,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Register a new model version.

        Returns:
            ``(model_id, version)``
        """
        existing_versions = [e.version for e in self._entries if e.model_name == model_name]
        version = max(existing_versions, default=0) + 1
        model_id = str(uuid.uuid4())

        artifact_dir = self.artifact_base / model_name / f"v{version}"
        model.save(artifact_dir)

        meta = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model.model_type,
            version=version,
            is_active=False,
            metrics=metrics,
            artifact_path=str(artifact_dir),
            created_at=datetime.now(UTC).isoformat(),
            extra=extra or {},
        )
        self._entries.append(meta)
        self._save_registry()

        logger.info("Registered %s v%d (id=%s)", model_name, version, model_id)
        return model_id, version

    def promote(self, model_id: str, version: int | None = None) -> None:
        """Promote a model version to active, deactivating all other
        versions of the same model name.
        """
        target: ModelMetadata | None = None
        for entry in self._entries:
            if entry.model_id == model_id:
                if version is None or entry.version == version:
                    target = entry
                    break

        if target is None:
            raise ValueError(f"Model id={model_id} version={version} not found")

        # Deactivate all versions of this model name
        for entry in self._entries:
            if entry.model_name == target.model_name:
                entry.is_active = False

        target.is_active = True
        self._save_registry()
        logger.info("Promoted %s v%d to active", target.model_name, target.version)

    def get_active(self, model_type: str) -> tuple[BasePredictor, ModelMetadata]:
        """Load and return the active model for the given model type.

        Raises:
            ValueError: if no active model exists.
        """
        for entry in self._entries:
            if entry.model_type == model_type and entry.is_active:
                model = self._load_model(entry)
                return model, entry

        raise ValueError(f"No active model found for type '{model_type}'")

    def list_versions(self, model_name: str) -> list[ModelMetadata]:
        return [e for e in self._entries if e.model_name == model_name]

    def latest_version(self, model_name: str) -> int:
        """Highest registered version for *model_name* (0 when none exist)."""
        return max(
            (e.version for e in self._entries if e.model_name == model_name),
            default=0,
        )

    def active_entries(self) -> list[ModelMetadata]:
        """All entries currently marked active (at most one per model name)."""
        return [e for e in self._entries if e.is_active]

    def adopt_version(
        self,
        model_name: str,
        version: int,
        *,
        model_type: str,
        metrics: dict[str, float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Register an artifact dir that already exists on disk.

        Used when artifacts arrive from outside ``register()`` (e.g. restored
        from the DB blob mirror), so no model object is saved here. Idempotent:
        if ``(model_name, version)`` is already registered the existing
        ``model_id`` is returned unchanged.
        """
        for entry in self._entries:
            if entry.model_name == model_name and entry.version == version:
                return entry.model_id

        meta = ModelMetadata(
            model_id=str(uuid.uuid4()),
            model_name=model_name,
            model_type=model_type,
            version=version,
            is_active=False,
            metrics=metrics or {},
            artifact_path=str(self.artifact_base / model_name / f"v{version}"),
            created_at=datetime.now(UTC).isoformat(),
            extra=extra or {},
        )
        self._entries.append(meta)
        self._save_registry()
        logger.info("Adopted %s v%d (id=%s)", model_name, version, meta.model_id)
        return meta.model_id

    def load_model_from_dir(self, model_type: str, artifact_dir: Path) -> BasePredictor:
        """Instantiate a predictor of *model_type* and load *artifact_dir*.

        Raises on unknown model types and on corrupt/partial artifacts.
        """
        from services.prediction.models.catboost_model import CatBoostPredictor
        from services.prediction.models.lightgbm_model import LightGBMPredictor
        from services.prediction.models.lstm_model import LSTMPredictor
        from services.prediction.models.transformer_model import TransformerPredictor
        from services.prediction.models.xgboost_model import XGBoostPredictor

        model_classes: dict[str, type[BasePredictor]] = {
            "xgboost": XGBoostPredictor,
            "lightgbm": LightGBMPredictor,
            "catboost": CatBoostPredictor,
            "lstm": LSTMPredictor,
            "transformer": TransformerPredictor,
        }
        cls = model_classes.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type '{model_type}'")

        model = cls()
        model.load(artifact_dir)
        return model

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self, meta: ModelMetadata) -> BasePredictor:
        """Instantiate and load a model from its saved artefacts."""
        return self.load_model_from_dir(meta.model_type, Path(meta.artifact_path))

    def _load_registry(self) -> list[ModelMetadata]:
        if not self._registry_path.exists():
            return []
        try:
            data = json.loads(self._registry_path.read_text())
            return [ModelMetadata(**entry) for entry in data]
        except Exception:
            logger.exception("Failed to load registry, starting fresh")
            return []

    def _save_registry(self) -> None:
        data = [asdict(e) for e in self._entries]
        self._registry_path.write_text(json.dumps(data, indent=2))


# ----------------------------------------------------------------------
# Startup restore + reconcile (durable-artifact replay)
# ----------------------------------------------------------------------


async def restore_and_reconcile(
    registry: ModelRegistry,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Replay DB-persisted artifact blobs into the filesystem registry, then
    align ``model_metadata.is_active`` with what actually serves.

    The deploy target has no volumes: a version promoted by the cloud
    retrainer exists only on the container filesystem and vanishes on the next
    deploy, while the ``model_metadata`` mirror still advertises it as active.
    Called at worker startup BEFORE ``ModelServer.load_active_models`` so
    restored versions are what serving picks up.

    Best-effort by design: every failure is logged and skipped -- the
    filesystem registry stays the serving source of truth and startup must
    never be blocked by the durability layer.
    """
    try:
        await _restore_from_blobs(registry, session_factory)
    except Exception:
        logger.exception("Artifact blob restore failed; serving repo champions")
    try:
        await _reconcile_metadata(registry, session_factory)
    except Exception:
        logger.exception("model_metadata reconcile failed")


async def _restore_from_blobs(
    registry: ModelRegistry,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Restore every blob-mirrored version newer than the filesystem registry."""
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(ModelArtifactBlob.model_name, ModelArtifactBlob.version).distinct()
            )
        ).all()

    versions_by_name: dict[str, list[int]] = {}
    for model_name, version in rows:
        versions_by_name.setdefault(model_name, []).append(version)

    for model_name in sorted(versions_by_name):
        latest_on_disk = registry.latest_version(model_name)
        # Ascending order so a promote of each restored version leaves the
        # newest one active, matching what the cloud worker last served.
        for version in sorted(v for v in versions_by_name[model_name] if v > latest_on_disk):
            await _restore_version(registry, session_factory, model_name, version)


async def _restore_version(
    registry: ModelRegistry,
    session_factory: async_sessionmaker[AsyncSession],
    model_name: str,
    version: int,
) -> None:
    """Write one blob-mirrored version to disk, validate it, register+promote.

    A blob set that cannot be written or loaded (corrupt/partial mirror) is
    logged and skipped, leaving the current champion serving.
    """
    async with session_factory() as session:
        blobs = (
            await session.execute(
                select(ModelArtifactBlob).where(
                    ModelArtifactBlob.model_name == model_name,
                    ModelArtifactBlob.version == version,
                )
            )
        ).scalars().all()
        db_meta = (
            await session.execute(
                select(DBModelMetadata).where(
                    DBModelMetadata.model_name == model_name,
                    DBModelMetadata.version == version,
                )
            )
        ).scalar_one_or_none()

    if not blobs:
        return
    for blob in blobs:
        # Filenames were persisted as basenames; anything else means a
        # tampered/corrupt row and must not escape the artifact dir.
        if Path(blob.filename).name != blob.filename:
            logger.warning(
                "Skipping restore of %s v%d: unsafe blob filename %r",
                model_name, version, blob.filename,
            )
            return

    model_type = db_meta.model_type if db_meta is not None else model_name
    metrics: dict[str, float] = {}
    if db_meta is not None and db_meta.validation_metrics:
        metrics = {
            k: float(v)
            for k, v in db_meta.validation_metrics.items()
            if isinstance(v, (int, float))
        }

    artifact_dir = registry.artifact_base / model_name / f"v{version}"

    def _write_files() -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for blob in blobs:
            (artifact_dir / blob.filename).write_bytes(blob.content)

    try:
        await asyncio.to_thread(_write_files)
        # Validate before registering: a partial/corrupt mirror must not
        # become the active model.
        await asyncio.to_thread(registry.load_model_from_dir, model_type, artifact_dir)
    except Exception:
        logger.warning(
            "Skipping restore of %s v%d: blob mirror is corrupt or partial",
            model_name, version, exc_info=True,
        )
        return

    model_id = registry.adopt_version(
        model_name, version, model_type=model_type, metrics=metrics
    )
    registry.promote(model_id, version)
    logger.info(
        "Restored %s v%d from DB blob mirror (%d files) and promoted it",
        model_name, version, len(blobs),
    )


async def _reconcile_metadata(
    registry: ModelRegistry,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Force ``model_metadata.is_active`` to match the filesystem registry.

    The DB mirror can go stale (e.g. a cloud-promoted version lost to a
    redeploy before blobs existed); without this the UI advertises a version
    that is not actually serving.
    """
    active = {(e.model_name, e.version): e for e in registry.active_entries()}
    now = datetime.now(UTC)
    changed = 0

    async with session_factory() as session:
        db_rows = (await session.execute(select(DBModelMetadata))).scalars().all()
        seen: set[tuple[str, int]] = set()
        for row in db_rows:
            key = (row.model_name, row.version)
            seen.add(key)
            should_be_active = key in active
            if row.is_active != should_be_active:
                row.is_active = should_be_active
                changed += 1
        # Actually-serving versions with no DB row at all: insert so the API
        # reflects reality.
        for key, entry in active.items():
            if key in seen:
                continue
            try:
                trained_at = datetime.fromisoformat(entry.created_at)
            except ValueError:
                trained_at = now
            session.add(
                DBModelMetadata(
                    model_name=entry.model_name,
                    model_type=entry.model_type,
                    version=entry.version,
                    hyperparameters={"reconciled": True},
                    validation_metrics=dict(entry.metrics) or None,
                    artifact_path=entry.artifact_path,
                    trained_at=trained_at,
                    is_active=True,
                    created_at=now,
                )
            )
            changed += 1
        if changed:
            await session.commit()
            logger.info(
                "Reconciled model_metadata.is_active with filesystem registry (%d rows)",
                changed,
            )
