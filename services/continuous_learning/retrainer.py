"""Automatic model retraining triggered by drift or schedule."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from config.settings import Settings
from core.models.market_data import OHLCVRecord
from core.models.ml_models import ModelArtifactBlob
from core.models.ml_models import ModelMetadata as DBModelMetadata
from services.prediction.registry import ModelRegistry
from services.prediction.training.dataset_builder import (
    MIN_VAL_ACCURACY,
    bars_matrix,
    build_dataset,
)
from services.prediction.training.trainer import ModelTrainer

logger = structlog.get_logger(__name__)

_TrainingData = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[str] | None,
]

_NO_DATA: _TrainingData = (None, None, None, None, None, None, None)


class AutoRetrainer:
    """Decides when to retrain models and orchestrates the process.

    Retraining is triggered when:
    - The model has not been trained within ``settings.retrain_interval_hours``.
    - A drift detector indicates significant distributional shift.
    """

    def __init__(
        self,
        trainer: ModelTrainer,
        registry: ModelRegistry,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._trainer = trainer
        self._registry = registry
        self._settings = settings
        # Resolved lazily so constructing the retrainer never requires a DB.
        self._session_factory = session_factory
        # model_id -> datetime of last successful retrain
        self._last_trained: dict[str, datetime] = {}
        # model_id -> drift detected flag
        self._drift_flags: dict[str, bool] = {}

    def _get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        if self._session_factory is None:
            from core.models.base import get_async_session_factory

            self._session_factory = get_async_session_factory()
        return self._session_factory

    # ------------------------------------------------------------------
    # Should retrain?
    # ------------------------------------------------------------------

    def should_retrain(self, model_id: str) -> bool:
        """Return ``True`` if the model should be retrained.

        Criteria:
        - Drift has been flagged for this model.
        - Time since last training exceeds ``retrain_interval_hours``.
        """
        if self._drift_flags.get(model_id, False):
            logger.info("retrainer.should_retrain.drift", model_id=model_id)
            return True

        last = self._last_trained.get(model_id)
        if last is None:
            return True

        interval = timedelta(hours=self._settings.retrain_interval_hours)
        if datetime.now(UTC) - last > interval:
            logger.info("retrainer.should_retrain.schedule", model_id=model_id)
            return True

        return False

    def mark_drift(self, model_id: str, drifting: bool = True) -> None:
        """Flag a model as having drifted (or clear the flag)."""
        self._drift_flags[model_id] = drifting

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    async def retrain(self, model_id: str) -> dict[str, Any]:
        """Retrain a model and register the new version if it improves.

        Returns:
            Dictionary with ``new_model_id``, ``version``, and ``metrics``
            on success, or ``{"skipped": True}`` if validation fails.
        """
        logger.info("retrainer.retrain.start", model_id=model_id)

        # Determine model type from registry
        model_type = self._resolve_model_type(model_id)
        if model_type is None:
            logger.warning("retrainer.retrain.unknown_model", model_id=model_id)
            return {"skipped": True, "reason": "unknown_model"}

        # Load training data (recent 1m bars -> serve-time feature replay)
        (
            X_train, y_train, returns_train,
            X_val, y_val, returns_val,
            feature_names,
        ) = await self._load_training_data(model_id)
        if X_train is None or y_train is None or X_val is None or y_val is None:
            return {"skipped": True, "reason": "no_training_data"}

        # Train new model. Training is CPU-bound and can run for minutes; on
        # the event loop it starves every Redis consumer, so push it to a
        # thread (to_thread forwards kwargs).
        result, new_model = await asyncio.to_thread(
            self._trainer.train_model,
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperopt=True,
            n_trials=30,
            feature_names=feature_names,
            returns_train=returns_train,
            returns_val=returns_val,
        )

        new_metrics = result.to_metrics()

        # Validate against old model
        try:
            _, old_meta = self._registry.get_active(model_type)
            old_metrics = old_meta.metrics
        except ValueError:
            old_metrics = {}

        if not self._validate_new_model(new_metrics, old_metrics):
            logger.warning(
                "retrainer.retrain.validation_failed",
                model_id=model_id,
                new_metrics=new_metrics,
                old_metrics=old_metrics,
            )
            return {"skipped": True, "reason": "new_model_not_better"}

        # Register and promote
        new_model_id, version = self._registry.register(
            model=new_model,
            model_name=model_type,
            metrics=new_metrics,
        )
        self._registry.promote(new_model_id, version)

        # Mirror into the DB model_metadata table (backs GET /api/v1/models).
        # Best-effort: a DB hiccup must not undo a successful retrain+promote.
        await self._mirror_metadata_to_db(model_type, version, new_metrics)

        # Durability: the deploy target has no volumes, so without this the
        # promoted artifacts die with the container on the next deploy.
        await self._persist_artifacts_to_db(model_type, version)

        self._last_trained[model_id] = datetime.now(UTC)
        self._drift_flags[model_id] = False

        logger.info(
            "retrainer.retrain.complete",
            old_model_id=model_id,
            new_model_id=new_model_id,
            version=version,
            metrics=new_metrics,
        )
        return {
            "new_model_id": new_model_id,
            "version": version,
            "metrics": new_metrics,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_model_type(self, model_id: str) -> str | None:
        """Resolve the model type string from a model_id."""
        for entry in self._registry._entries:
            if entry.model_id == model_id:
                return entry.model_type
        # Try to infer from the model_id itself (e.g. "ensemble:xgboost,lightgbm")
        for known in ("xgboost", "lightgbm", "lstm", "transformer"):
            if known in model_id:
                return known
        return None

    async def _load_training_data(self, model_id: str) -> _TrainingData:
        """Build a training dataset from recent 1-minute bars in the DB.

        Loads ``settings.retrain_lookback_days`` of 1m OHLCV history, then
        replays the live feature pipeline over it via the shared
        ``dataset_builder`` (the same code path the bootstrap training script
        uses, so retrained models see serve-time feature semantics).

        Returns ``(X_train, y_train, returns_train, X_val, y_val, returns_val,
        feature_names)`` or all ``None`` if data is unavailable.
        """
        try:
            cutoff = datetime.now(UTC) - timedelta(days=self._settings.retrain_lookback_days)
            stmt = (
                select(
                    OHLCVRecord.time,
                    OHLCVRecord.symbol,
                    OHLCVRecord.open,
                    OHLCVRecord.high,
                    OHLCVRecord.low,
                    OHLCVRecord.close,
                    OHLCVRecord.volume,
                )
                .where(OHLCVRecord.timeframe == "1m", OHLCVRecord.time >= cutoff)
                .order_by(OHLCVRecord.symbol, OHLCVRecord.time)
            )
            factory = self._get_session_factory()
            async with factory() as session:
                rows = (await session.execute(stmt)).all()

            if not rows:
                logger.info("retrainer.load_data.empty", model_id=model_id)
                return _NO_DATA

            per_symbol_rows: dict[str, list[Any]] = {}
            for row in rows:
                per_symbol_rows.setdefault(row.symbol, []).append(row)
            per_symbol_cols = {
                sym: bars_matrix(sym_rows) for sym, sym_rows in per_symbol_rows.items()
            }

            # The rolling feature replay is CPU-bound; keep the event loop free.
            return await asyncio.to_thread(build_dataset, per_symbol_cols)
        except ValueError as exc:  # no symbol had enough bars -> skip, not error
            logger.info("retrainer.load_data.insufficient", model_id=model_id, reason=str(exc))
            return _NO_DATA
        except Exception:
            logger.exception("retrainer.load_data.error", model_id=model_id)
            return _NO_DATA

    async def _mirror_metadata_to_db(
        self, model_type: str, version: int, metrics: dict[str, Any]
    ) -> None:
        """Upsert the promoted version into ``model_metadata`` (best-effort).

        GET /api/v1/models reads from the DB table, not the filesystem
        registry, so without this mirror retrained versions would be invisible
        to the API. Failure is logged but never propagated: the filesystem
        registry is the serving source of truth.
        """
        try:
            artifact_path = ""
            for entry in self._registry.list_versions(model_type):
                if entry.version == version:
                    artifact_path = entry.artifact_path
                    break

            now = datetime.now(UTC)
            factory = self._get_session_factory()
            async with factory() as session:
                # Deactivate prior versions of this model name.
                await session.execute(
                    update(DBModelMetadata)
                    .where(DBModelMetadata.model_name == model_type)
                    .values(is_active=False)
                )
                existing = (
                    await session.execute(
                        select(DBModelMetadata).where(
                            DBModelMetadata.model_name == model_type,
                            DBModelMetadata.version == version,
                        )
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    existing.validation_metrics = metrics
                    existing.artifact_path = artifact_path
                    existing.trained_at = now
                    existing.is_active = True
                else:
                    session.add(
                        DBModelMetadata(
                            model_name=model_type,
                            model_type=model_type,
                            version=version,
                            hyperparameters={"retrained": True},
                            validation_metrics=metrics,
                            artifact_path=artifact_path,
                            trained_at=now,
                            is_active=True,
                            created_at=now,
                        )
                    )
                await session.commit()
            logger.info("retrainer.db_mirror.ok", model_name=model_type, version=version)
        except Exception:
            logger.exception(
                "retrainer.db_mirror.failed", model_name=model_type, version=version
            )

    async def _persist_artifacts_to_db(self, model_type: str, version: int) -> None:
        """Copy the promoted version's artifact files into ``model_artifact_blobs``.

        ``restore_and_reconcile`` replays these blobs onto disk at worker
        startup, surviving the volume-less redeploys that otherwise revert
        serving to the repo champions. Best-effort like the metadata mirror:
        failure is logged but never propagated, and the filesystem registry
        remains the serving source of truth.
        """
        try:
            artifact_dir: Path | None = None
            for entry in self._registry.list_versions(model_type):
                if entry.version == version:
                    artifact_dir = Path(entry.artifact_path)
                    break
            if artifact_dir is None or not artifact_dir.is_dir():
                logger.warning(
                    "retrainer.blob_persist.no_artifact_dir",
                    model_name=model_type,
                    version=version,
                )
                return

            # A version dir is a few MB of joblib files; read off-loop anyway.
            def _read_files() -> list[tuple[str, bytes]]:
                return [
                    (path.name, path.read_bytes())
                    for path in sorted(artifact_dir.iterdir())
                    if path.is_file()
                ]

            contents = await asyncio.to_thread(_read_files)
            if not contents:
                logger.warning(
                    "retrainer.blob_persist.empty_dir",
                    model_name=model_type,
                    version=version,
                )
                return

            now = datetime.now(UTC)
            factory = self._get_session_factory()
            async with factory() as session:
                # Replace rows from any earlier partial persist of this version.
                await session.execute(
                    delete(ModelArtifactBlob).where(
                        ModelArtifactBlob.model_name == model_type,
                        ModelArtifactBlob.version == version,
                    )
                )
                for filename, content in contents:
                    session.add(
                        ModelArtifactBlob(
                            model_name=model_type,
                            version=version,
                            filename=filename,
                            content=content,
                            created_at=now,
                        )
                    )
                await session.commit()
            logger.info(
                "retrainer.blob_persist.ok",
                model_name=model_type,
                version=version,
                files=len(contents),
                bytes=sum(len(content) for _, content in contents),
            )
        except Exception:
            logger.exception(
                "retrainer.blob_persist.failed", model_name=model_type, version=version
            )

    @staticmethod
    def _validate_new_model(
        new_metrics: dict[str, Any],
        old_metrics: dict[str, Any],
    ) -> bool:
        """Return ``True`` if the challenger may replace the champion.

        The challenger must beat the absolute out-of-sample floor
        (``MIN_VAL_ACCURACY``, vs the 1/3 random baseline) AND be at least as
        accurate as the current champion (when one exists).
        """
        new_acc = float(new_metrics.get("val_accuracy", new_metrics.get("accuracy", 0.0)))
        if new_acc < MIN_VAL_ACCURACY:
            return False

        if not old_metrics:
            return True

        old_acc = float(old_metrics.get("val_accuracy", old_metrics.get("accuracy", 0.0)))
        return new_acc >= old_acc
