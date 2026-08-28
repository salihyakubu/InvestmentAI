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
from core.models.market_data import AuxMarketState, OHLCVRecord
from core.models.ml_models import ModelArtifactBlob
from core.models.ml_models import ModelMetadata as DBModelMetadata
from services.feature_engineering.aux_features import HistoricalAuxProvider
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

# Prediction events from an ensemble carry ids like "ensemble:xgboost,lightgbm".
_ENSEMBLE_PREFIX = "ensemble:"
_KNOWN_MODEL_TYPES = ("xgboost", "lightgbm", "lstm", "transformer")


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
        """Retrain the model(s) behind *model_id* and promote improvements.

        Plain model ids train a single challenger and return
        ``{"new_model_id", "version", "metrics", "model_type"}`` on success,
        or ``{"skipped": True, "reason": ...}``.

        Ensemble ids (``"ensemble:xgboost,lightgbm"``) name every member type
        they serve: each member is retrained, gated, and promoted
        independently on a shared dataset, and the per-member results are
        returned under ``"members"``. The top-level ``"skipped"`` flag is set
        only when no member was promoted.
        """
        logger.info("retrainer.retrain.start", model_id=model_id)

        member_types = self._resolve_member_types(model_id)
        if not member_types:
            logger.warning("retrainer.retrain.unknown_model", model_id=model_id)
            return {"skipped": True, "reason": "unknown_model"}

        # THE EMBARGO (live-transfer gate amendment, GO_LIVE 2026-08-28):
        # training data ends EMBARGO_DAYS back so the replay window that
        # judges the challenger contains nothing it trained on.
        from services.continuous_learning.live_replay import (
            EMBARGO_DAYS,
            GateDecision,
            prepare_replay,
        )

        training_end = datetime.now(UTC) - timedelta(days=EMBARGO_DAYS)

        # Load training data once (1m bars up to the embargo boundary ->
        # serve-time feature replay); ensemble members share the dataset.
        (
            X_train, y_train, returns_train,
            X_val, y_val, returns_val,
            feature_names,
        ) = await self._load_training_data(model_id, end=training_end)
        if X_train is None or y_train is None or X_val is None or y_val is None:
            return {"skipped": True, "reason": "no_training_data"}

        # Data-side gate checks run BEFORE any training: a deterministic
        # refusal (floors, hash, boundary) must not cost a daily hyperopt
        # cycle that the gate then discards.
        try:
            prepared = await prepare_replay(
                self._get_session_factory(), feature_names, training_end
            )
        except Exception:
            logger.exception("retrainer.retrain.live_transfer_gate_error")
            prepared = GateDecision(
                promote=False, reason="live_transfer_gate_error",
                challenger_ic=None, champion_ic=None, n_rows=0, span_days=None,
            )
        if isinstance(prepared, GateDecision):
            logger.warning(
                "retrainer.retrain.live_transfer_refused_before_training",
                model_id=model_id,
                **prepared.as_dict(),
            )
            results = [
                {
                    "skipped": True,
                    "reason": prepared.reason,
                    "model_type": member,
                    "live_replay": prepared.as_dict(),
                }
                for member in member_types
            ]
            if not model_id.startswith(_ENSEMBLE_PREFIX):
                return results[0]
            return {"members": results, "skipped": True, "reason": prepared.reason}

        results = [
            await self._retrain_member(
                model_id,
                member,
                X_train=X_train,
                y_train=y_train,
                returns_train=returns_train,
                X_val=X_val,
                y_val=y_val,
                returns_val=returns_val,
                feature_names=feature_names,
                replay_data=prepared,
            )
            for member in member_types
        ]

        if any(not r.get("skipped") for r in results):
            # At least one member completed train -> gate -> promote, so the
            # cycle counts as done. A wholly failed cycle leaves the
            # bookkeeping untouched so the next evaluation retries.
            self._last_trained[model_id] = datetime.now(UTC)
            self._drift_flags[model_id] = False

        if not model_id.startswith(_ENSEMBLE_PREFIX):
            return results[0]

        summary: dict[str, Any] = {"members": results}
        if all(r.get("skipped") for r in results):
            summary["skipped"] = True
            summary["reason"] = "no_member_promoted"
        return summary

    async def _retrain_member(
        self,
        model_id: str,
        model_type: str,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        returns_train: np.ndarray | None,
        X_val: np.ndarray,
        y_val: np.ndarray,
        returns_val: np.ndarray | None,
        feature_names: list[str] | None,
        replay_data: Any,
    ) -> dict[str, Any]:
        """Train one model type, gate it against its champion, and promote.

        Training is CPU-bound and can run for minutes; on the event loop it
        starves every Redis consumer, so it runs in a thread.
        """
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
                model_type=model_type,
                new_metrics=new_metrics,
                old_metrics=old_metrics,
            )
            return {
                "skipped": True,
                "reason": "new_model_not_better",
                "model_type": model_type,
            }

        # Phase 2 live-transfer gate (registered 2026-08-28, embargo
        # amendment same day): validation accuracy measurably does not
        # transfer to live data (era 4), so the challenger must ALSO beat
        # the champion replayed over the identical EMBARGOED served rows --
        # data neither model trained on. Conjunctive with the validation
        # gate; the data-side checks already passed upstream in retrain().
        from services.continuous_learning.live_replay import GateDecision, decide

        try:
            champion_model, _ = self._registry.get_active(model_type)
        except ValueError:
            champion_model = None
        try:
            gate = await decide(replay_data, new_model, champion_model)
        except Exception:
            # Fail CLOSED: a gate that cannot judge must refuse, not wave a
            # challenger through -- blind promotion is the measured failure
            # mode this gate exists to end.
            logger.exception(
                "retrainer.retrain.live_transfer_gate_error", model_type=model_type
            )
            gate = GateDecision(
                promote=False, reason="live_transfer_gate_error",
                challenger_ic=None, champion_ic=None, n_rows=0, span_days=None,
            )
        logger.info(
            "retrainer.retrain.live_transfer_gate",
            model_type=model_type,
            **gate.as_dict(),
        )
        if not gate.promote:
            return {
                "skipped": True,
                "reason": gate.reason,
                "model_type": model_type,
                "live_replay": gate.as_dict(),
            }

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
        # (last_trained/drift bookkeeping is handled once at the retrain()
        # loop level, covering all members.)
        await self._persist_artifacts_to_db(model_type, version)

        logger.info(
            "retrainer.retrain.complete",
            old_model_id=model_id,
            model_type=model_type,
            new_model_id=new_model_id,
            version=version,
            metrics=new_metrics,
        )
        return {
            "new_model_id": new_model_id,
            "version": version,
            "metrics": new_metrics,
            "model_type": model_type,
            "live_replay": gate.as_dict(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_member_types(self, model_id: str) -> list[str]:
        """Resolve the model type(s) a model_id refers to.

        Ensemble ids name every member type they serve (the id is the
        comma-joined active types, e.g. ``"ensemble:xgboost,lightgbm"``);
        each member must be retrained individually. Plain ids resolve to a
        single type; unrecognised ids resolve to an empty list.
        """
        if not model_id.startswith(_ENSEMBLE_PREFIX):
            single = self._resolve_model_type(model_id)
            return [] if single is None else [single]

        members: list[str] = []
        for raw in model_id.removeprefix(_ENSEMBLE_PREFIX).split(","):
            member = raw.strip()
            if member in _KNOWN_MODEL_TYPES:
                if member not in members:
                    members.append(member)
            elif member:
                logger.warning(
                    "retrainer.retrain.unknown_member",
                    model_id=model_id,
                    member=member,
                )
        return members

    def _resolve_model_type(self, model_id: str) -> str | None:
        """Resolve the model type string from a non-ensemble model_id."""
        for entry in self._registry._entries:
            if entry.model_id == model_id:
                return entry.model_type
        # Try to infer from the model_id itself (e.g. "xgboost-v3")
        for known in _KNOWN_MODEL_TYPES:
            if known in model_id:
                return known
        return None

    async def _load_training_data(
        self, model_id: str, end: datetime | None = None
    ) -> _TrainingData:
        """Build a training dataset from recent 1-minute bars in the DB.

        Loads ``settings.retrain_lookback_days`` of 1m OHLCV history ending
        at *end* (the live-transfer gate's embargo boundary -- training data
        must stop where the replay window begins, or the gate would score
        the challenger on its own training rows), then replays the live
        feature pipeline over it via the shared ``dataset_builder`` (the
        same code path the bootstrap training script uses, so retrained
        models see serve-time feature semantics).

        Returns ``(X_train, y_train, returns_train, X_val, y_val, returns_val,
        feature_names)`` or all ``None`` if data is unavailable.
        """
        try:
            end = end or datetime.now(UTC)
            cutoff = end - timedelta(days=self._settings.retrain_lookback_days)
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
                .where(
                    OHLCVRecord.timeframe == "1m",
                    OHLCVRecord.time >= cutoff,
                    OHLCVRecord.time < end,
                )
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

            # Aux regime values over the SAME window, replayed as-of each bar
            # time so training sees exactly what live serving would have held
            # (train/serve parity). Best-effort and self-contained: if the aux
            # table is empty/unavailable the provider yields 0.0, matching a
            # live path with no aux data -- training must never fail on aux.
            aux_provider = await self._load_aux_provider(cutoff)

            # The rolling feature replay is CPU-bound; keep the event loop free.
            return await asyncio.to_thread(
                build_dataset, per_symbol_cols, aux_provider=aux_provider
            )
        except ValueError as exc:  # no symbol had enough bars -> skip, not error
            logger.info("retrainer.load_data.insufficient", model_id=model_id, reason=str(exc))
            return _NO_DATA
        except Exception:
            logger.exception("retrainer.load_data.error", model_id=model_id)
            return _NO_DATA

    async def _load_aux_provider(self, cutoff: datetime) -> HistoricalAuxProvider:
        """Build a HistoricalAuxProvider from aux_market_state since *cutoff*.

        Best-effort: any failure (missing table, DB hiccup) yields an empty
        provider that returns 0.0 for every aux feature, so a retrain still
        proceeds on price/volume alone rather than aborting.
        """
        try:
            stmt = (
                select(
                    AuxMarketState.time,
                    AuxMarketState.metric,
                    AuxMarketState.symbol,
                    AuxMarketState.value,
                )
                .where(AuxMarketState.time >= cutoff)
                .order_by(AuxMarketState.time)
            )
            factory = self._get_session_factory()
            async with factory() as session:
                rows = (await session.execute(stmt)).all()
            return HistoricalAuxProvider(
                (r.time, r.metric, r.symbol, r.value) for r in rows
            )
        except Exception:
            logger.warning("retrainer.load_aux.unavailable")
            return HistoricalAuxProvider(())

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
