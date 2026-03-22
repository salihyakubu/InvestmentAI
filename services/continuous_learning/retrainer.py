"""Automatic model retraining triggered by drift or schedule."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import structlog

from config.settings import Settings
from services.prediction.registry import ModelRegistry
from services.prediction.training.trainer import ModelTrainer

logger = structlog.get_logger(__name__)


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
    ) -> None:
        self._trainer = trainer
        self._registry = registry
        self._settings = settings
        # model_id -> datetime of last successful retrain
        self._last_trained: dict[str, datetime] = {}
        # model_id -> drift detected flag
        self._drift_flags: dict[str, bool] = {}

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
        if datetime.now(timezone.utc) - last > interval:
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

        # Load training data
        X_train, y_train, X_val, y_val, feature_names = self._load_training_data(model_id)
        if X_train is None:
            return {"skipped": True, "reason": "no_training_data"}

        # Train new model
        result, new_model = self._trainer.train_model(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperopt=True,
            n_trials=30,
            feature_names=feature_names,
        )

        new_metrics = result.metrics if hasattr(result, "metrics") else {}

        # Validate against old model
        try:
            old_model, old_meta = self._registry.get_active(model_type)
            old_metrics = old_meta.metrics
        except ValueError:
            old_model = None
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

        self._last_trained[model_id] = datetime.now(timezone.utc)
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

    def _load_training_data(
        self, model_id: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str] | None]:
        """Load training data for the given model.

        In production this would pull from the feature store / database.
        Returns ``(X_train, y_train, X_val, y_val, feature_names)`` or
        all ``None`` if data is unavailable.
        """
        try:
            data = self._trainer.data_loader.load_training_data(
                features=np.empty((0, 0)),
                close_prices=np.empty(0),
            )
            if data.X.size == 0:
                return None, None, None, None, None

            split = int(len(data.X) * 0.8)
            return (
                data.X[:split],
                data.y[:split],
                data.X[split:],
                data.y[split:],
                data.feature_names,
            )
        except Exception:
            logger.exception("retrainer.load_data.error", model_id=model_id)
            return None, None, None, None, None

    @staticmethod
    def _validate_new_model(
        new_metrics: dict[str, Any],
        old_metrics: dict[str, Any],
    ) -> bool:
        """Return ``True`` if the new model is at least as good as the old one.

        Compares validation accuracy; if unavailable, accepts the new model.
        """
        if not old_metrics:
            return True

        new_acc = new_metrics.get("val_accuracy", new_metrics.get("accuracy", 0))
        old_acc = old_metrics.get("val_accuracy", old_metrics.get("accuracy", 0))

        return new_acc >= old_acc
