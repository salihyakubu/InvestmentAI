"""LightGBM-based predictor for direction classification and return regression."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, log_loss

from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "short", 1: "flat", 2: "long"}


class LightGBMPredictor(BasePredictor):
    """Three-class direction classifier + return regressor using LightGBM."""

    model_type: str = "lightgbm"
    required_features: list[str] = []

    def __init__(
        self,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        default_cls_params: dict = {
            "n_estimators": 500,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "num_leaves": 63,
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        default_reg_params: dict = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_leaves": 31,
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        if classifier_params:
            default_cls_params.update(classifier_params)
        if regressor_params:
            default_reg_params.update(regressor_params)

        self._classifier = LGBMClassifier(**default_cls_params)
        self._regressor = LGBMRegressor(**default_reg_params)
        self._feature_names = feature_names or []
        self._categorical_features = categorical_features or []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> PredictionOutput:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        probs = self._classifier.predict_proba(features)[0]
        direction_idx = int(np.argmax(probs))
        direction = DIRECTION_MAP[direction_idx]
        confidence = float(probs[direction_idx])
        expected_return = float(self._regressor.predict(features)[0])

        return PredictionOutput(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            probabilities={DIRECTION_MAP[i]: float(p) for i, p in enumerate(probs)},
        )

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        probs = self._classifier.predict_proba(features)
        returns = self._regressor.predict(features)
        outputs: list[PredictionOutput] = []
        for i in range(len(features)):
            direction_idx = int(np.argmax(probs[i]))
            outputs.append(
                PredictionOutput(
                    direction=DIRECTION_MAP[direction_idx],
                    confidence=float(probs[i][direction_idx]),
                    expected_return=float(returns[i]),
                    probabilities={DIRECTION_MAP[j]: float(probs[i][j]) for j in range(3)},
                )
            )
        return outputs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        logger.info("Training LightGBM classifier on %d samples", len(X_train))

        cat_feature_indices = (
            [self._feature_names.index(f) for f in self._categorical_features if f in self._feature_names]
            if self._feature_names and self._categorical_features
            else "auto"
        )

        callbacks = [early_stopping(stopping_rounds=30), log_evaluation(period=0)]

        self._classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_feature_indices,
            callbacks=callbacks,
        )

        # --- Regressor ---
        return_map = {0: -0.01, 1: 0.0, 2: 0.01}
        y_train_returns = np.array([return_map.get(int(y), 0.0) for y in y_train])
        y_val_returns = np.array([return_map.get(int(y), 0.0) for y in y_val])

        self._regressor.fit(
            X_train,
            y_train_returns,
            eval_set=[(X_val, y_val_returns)],
            callbacks=[early_stopping(stopping_rounds=30), log_evaluation(period=0)],
        )

        self._is_fitted = True

        # --- Metrics ---
        train_probs = self._classifier.predict_proba(X_train)
        val_probs = self._classifier.predict_proba(X_val)
        train_preds = np.argmax(train_probs, axis=1)
        val_preds = np.argmax(val_probs, axis=1)

        train_accuracy = float(accuracy_score(y_train, train_preds))
        val_accuracy = float(accuracy_score(y_val, val_preds))
        train_loss = float(log_loss(y_train, train_probs, labels=[0, 1, 2]))
        val_loss = float(log_loss(y_val, val_probs, labels=[0, 1, 2]))

        importance = self.get_feature_importance()

        logger.info(
            "LightGBM training complete  train_acc=%.4f  val_acc=%.4f",
            train_accuracy,
            val_accuracy,
        )

        return TrainResult(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            epochs_trained=int(self._classifier.best_iteration_) if hasattr(self._classifier, "best_iteration_") else self._classifier.n_estimators,
            feature_importance=importance,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._classifier, path / "classifier.joblib")
        joblib.dump(self._regressor, path / "regressor.joblib")
        joblib.dump(self._feature_names, path / "feature_names.joblib")
        joblib.dump(self._categorical_features, path / "categorical_features.joblib")
        logger.info("LightGBM model saved to %s", path)

    def load(self, path: Path) -> None:
        self._classifier = joblib.load(path / "classifier.joblib")
        self._regressor = joblib.load(path / "regressor.joblib")
        feature_path = path / "feature_names.joblib"
        if feature_path.exists():
            self._feature_names = joblib.load(feature_path)
        cat_path = path / "categorical_features.joblib"
        if cat_path.exists():
            self._categorical_features = joblib.load(cat_path)
        self._is_fitted = True
        logger.info("LightGBM model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted:
            return {}
        importances = self._classifier.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return {name: float(imp) for name, imp in zip(names, importances)}
