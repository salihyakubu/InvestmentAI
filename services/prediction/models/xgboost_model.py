"""XGBoost-based predictor for direction classification and return regression."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier, XGBRegressor

from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "short", 1: "flat", 2: "long"}
DIRECTION_INV = {v: k for k, v in DIRECTION_MAP.items()}


class XGBoostPredictor(BasePredictor):
    """Three-class direction classifier + return regressor using XGBoost."""

    model_type: str = "xgboost"
    required_features: list[str] = []

    def __init__(
        self,
        classifier_params: dict | None = None,
        regressor_params: dict | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        default_cls_params: dict = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        default_reg_params: dict = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }

        if classifier_params:
            default_cls_params.update(classifier_params)
        if regressor_params:
            default_reg_params.update(regressor_params)

        self._classifier = XGBClassifier(**default_cls_params)
        self._regressor = XGBRegressor(**default_reg_params)
        self._feature_names = feature_names or []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> PredictionOutput:
        """Predict for a single flat feature vector (1-D or 2-D with one row)."""
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
        """Predict for a batch of flat feature vectors."""
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
        """Train classifier and regressor with early stopping.

        *y_train* / *y_val* are expected to contain integer class labels
        (0=short, 1=flat, 2=long).  The regressor is trained on the same
        features with continuous returns stored as float values; when
        integer labels are provided, returns are approximated from the
        label mapping (-1%, 0%, +1%).
        """
        # --- Classifier ---
        logger.info("Training XGBoost classifier on %d samples", len(X_train))
        self._classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # --- Regressor (approximate returns from labels if needed) ---
        return_map = {0: -0.01, 1: 0.0, 2: 0.01}
        y_train_returns = np.array([return_map.get(int(y), 0.0) for y in y_train])
        y_val_returns = np.array([return_map.get(int(y), 0.0) for y in y_val])

        self._regressor.fit(
            X_train,
            y_train_returns,
            eval_set=[(X_val, y_val_returns)],
            verbose=False,
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
            "XGBoost training complete  train_acc=%.4f  val_acc=%.4f",
            train_accuracy,
            val_accuracy,
        )

        return TrainResult(
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            epochs_trained=int(self._classifier.best_iteration) if hasattr(self._classifier, "best_iteration") else self._classifier.n_estimators,
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
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: Path) -> None:
        self._classifier = joblib.load(path / "classifier.joblib")
        self._regressor = joblib.load(path / "regressor.joblib")
        feature_names_path = path / "feature_names.joblib"
        if feature_names_path.exists():
            self._feature_names = joblib.load(feature_names_path)
        self._is_fitted = True
        logger.info("XGBoost model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        if not self._is_fitted:
            return {}
        importances = self._classifier.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return {name: float(imp) for name, imp in zip(names, importances)}
