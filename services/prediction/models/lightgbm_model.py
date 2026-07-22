"""LightGBM-based predictor for direction classification and return regression."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

from services.prediction.models.base import (
    BasePredictor,
    ConformalState,
    PredictionOutput,
    TrainResult,
    fit_conformal_state,
    probabilities_in_class_order,
    select_calibration_method,
)

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "short", 1: "flat", 2: "long"}

# Minimum validation rows (with all 3 classes present) before we fit a
# probability calibrator; below this the calibration fits are unstable so we
# keep raw probabilities rather than emit a miscalibrated confidence.
_MIN_CALIBRATION_SAMPLES = 30


class LightGBMPredictor(BasePredictor):
    """Three-class direction classifier + return regressor using LightGBM."""

    model_type: str = "lightgbm"
    required_features: list[str] = []

    def __init__(
        self,
        classifier_params: dict[str, Any] | None = None,
        regressor_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[str] | None = None,
        conformal_alpha: float = 0.10,
    ) -> None:
        default_cls_params: dict[str, Any] = {
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
        default_reg_params: dict[str, Any] = {
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
        self._calibrator: CalibratedClassifierCV | None = None
        self._calibration_method: str = "none"  # "isotonic" | "sigmoid" | "none"
        self._brier_isotonic: float | None = None
        self._brier_sigmoid: float | None = None
        self._conformal_alpha = float(conformal_alpha)
        self._conformal: ConformalState | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _proba(self, features: np.ndarray) -> np.ndarray:
        """Return (n, 3) class probabilities in [short, flat, long] order.

        Uses the calibrated probabilities when a calibrator was fitted, else the
        raw classifier softmax. Columns are remapped via the estimator's
        ``classes_`` so a class absent from the fit data yields a zero column.
        """
        estimator = self._calibrator if self._calibrator is not None else self._classifier
        return probabilities_in_class_order(estimator, features)

    def predict(self, features: np.ndarray) -> PredictionOutput:
        if features.ndim == 1:
            features = features.reshape(1, -1)
        probs = self._proba(features)[0]
        direction_idx = int(np.argmax(probs))
        direction = DIRECTION_MAP[direction_idx]
        confidence = float(probs[direction_idx])
        expected_return = float(self._regressor.predict(features)[0])
        prob_map = {DIRECTION_MAP[i]: float(p) for i, p in enumerate(probs)}

        return PredictionOutput(
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            probabilities=prob_map,
            metadata=self._conformal_metadata(prob_map),
        )

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        probs = self._proba(features)
        returns = self._regressor.predict(features)
        outputs: list[PredictionOutput] = []
        for i in range(len(features)):
            direction_idx = int(np.argmax(probs[i]))
            prob_map = {DIRECTION_MAP[j]: float(probs[i][j]) for j in range(3)}
            outputs.append(
                PredictionOutput(
                    direction=DIRECTION_MAP[direction_idx],
                    confidence=float(probs[i][direction_idx]),
                    expected_return=float(returns[i]),
                    probabilities=prob_map,
                    metadata=self._conformal_metadata(prob_map),
                )
            )
        return outputs

    def _fit_calibrator(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> CalibratedClassifierCV | None:
        """Fit isotonic AND sigmoid calibrators; keep the lower-Brier one.

        The reported confidence must approximate a true probability (the live
        confidence gate depends on it). Both candidates are fitted on the
        validation split and the one with the lower multi-class Brier score
        (mean over one-vs-rest) wins; the chosen method and both scores are
        recorded on ``self`` for :class:`TrainResult` and persistence. Skipped
        when the validation split lacks all three classes or is too small for
        a stable fit.
        """
        self._calibration_method = "none"
        self._brier_isotonic = None
        self._brier_sigmoid = None
        try:
            y_val = np.asarray(y_val)
            _, counts = np.unique(y_val, return_counts=True)
            if len(y_val) < _MIN_CALIBRATION_SAMPLES or len(counts) < 3:
                logger.info("Skipping probability calibration: insufficient validation coverage")
                return None
            # Calibrate the already-fitted classifier without refitting it. cv folds
            # the validation rows for the calibration fits; cap folds at the rarest
            # class count so StratifiedKFold always has >=1 row per class per fold.
            n_splits = int(min(5, counts.min()))
            if n_splits < 2:
                logger.info("Skipping probability calibration: a class has too few validation rows")
                return None
            selection = select_calibration_method(
                self._classifier, X_val, y_val, n_splits=n_splits
            )
            self._calibration_method = selection.method
            self._brier_isotonic = selection.brier_isotonic
            self._brier_sigmoid = selection.brier_sigmoid
            logger.info(
                "Selected %s probability calibration on %d validation rows "
                "(brier isotonic=%s sigmoid=%s)",
                selection.method,
                len(y_val),
                selection.brier_isotonic,
                selection.brier_sigmoid,
            )
            return selection.calibrator
        except Exception:
            logger.exception("Probability calibration failed; using raw classifier probabilities")
            return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        returns_train: np.ndarray | None = None,
        returns_val: np.ndarray | None = None,
    ) -> TrainResult:
        logger.info("Training LightGBM classifier on %d samples", len(X_train))

        cat_feature_indices: list[int] | Literal["auto"] = (
            [self._feature_names.index(f) for f in self._categorical_features if f in self._feature_names]
            if self._feature_names and self._categorical_features
            else "auto"
        )

        callbacks: list[Callable[..., Any]] = [
            early_stopping(stopping_rounds=30),
            log_evaluation(period=0),
        ]

        self._classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_feature_indices,
            callbacks=callbacks,
        )

        # --- Regressor ---
        if returns_train is not None and returns_val is not None:
            y_train_returns = np.asarray(returns_train, dtype=np.float64)
            y_val_returns = np.asarray(returns_val, dtype=np.float64)
        else:
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

        # --- Probability calibration (on the held-out validation split) ---
        self._calibrator = self._fit_calibrator(X_val, y_val)

        # --- Split-conformal calibration (validation nonconformity quantiles) ---
        # Fitted AFTER the calibrator so nonconformity uses the same calibrated
        # probabilities that serving emits.
        self._conformal = fit_conformal_state(
            self._proba(X_val), y_val, alpha=self._conformal_alpha
        )

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
            epochs_trained=(
                int(self._classifier.best_iteration_)
                if hasattr(self._classifier, "best_iteration_")
                else self._classifier.n_estimators
            ),
            feature_importance=importance,
            chosen_calibration=self._calibration_method,
            brier_isotonic=self._brier_isotonic,
            brier_sigmoid=self._brier_sigmoid,
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
        if self._calibrator is not None:
            joblib.dump(self._calibrator, path / "calibrator.joblib")
        joblib.dump(
            {
                "chosen_calibration": self._calibration_method,
                "brier_isotonic": self._brier_isotonic,
                "brier_sigmoid": self._brier_sigmoid,
            },
            path / "calibration_meta.joblib",
        )
        if self._conformal is not None:
            joblib.dump(self._conformal.to_dict(), path / "conformal.joblib")
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
        calibrator_path = path / "calibrator.joblib"
        self._calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
        meta_path = path / "calibration_meta.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self._calibration_method = str(meta.get("chosen_calibration", "none"))
            self._brier_isotonic = meta.get("brier_isotonic")
            self._brier_sigmoid = meta.get("brier_sigmoid")
        else:
            # Pre-selection artifact: any persisted calibrator was sigmoid-only.
            self._calibration_method = "sigmoid" if self._calibrator is not None else "none"
            self._brier_isotonic = None
            self._brier_sigmoid = None
        conformal_path = path / "conformal.joblib"
        self._conformal = (
            ConformalState.from_dict(joblib.load(conformal_path))
            if conformal_path.exists()
            else None  # old artifact: no conformal state, non-blocking downstream
        )
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
