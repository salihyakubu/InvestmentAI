"""Abstract base predictor and shared data structures.

This module also hosts the shared *tree-model contract* for probability
calibration and split-conformal prediction so every tabular model (XGBoost,
LightGBM, and future ones such as CatBoost) implements identical semantics:

- ``select_calibration_method``: fit BOTH isotonic and sigmoid calibrators on
  the validation split and keep whichever has the lower multi-class Brier
  score (mean over one-vs-rest).
- ``fit_conformal_state`` / ``ConformalState``: split-conformal calibration on
  the validation split. Nonconformity is ``1 - calibrated probability of the
  true class``; the stored ``(1 - alpha)`` quantiles define per-sample
  prediction SETS (classes whose nonconformity is <= the quantile).

A model that follows the contract stores a ``ConformalState | None`` in
``self._conformal`` and attaches ``BasePredictor._conformal_metadata(probs)``
to every :class:`PredictionOutput` it emits.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Canonical 3-class direction layout shared by all tabular models.
CLASS_NAMES: dict[int, str] = {0: "short", 1: "flat", 2: "long"}

# Minimum validation rows before split-conformal quantiles are estimated;
# below this the model simply carries no conformal state (non-blocking).
_MIN_CONFORMAL_SAMPLES = 30
# Minimum rows of a single class before its class-conditional quantile is
# used; classes below this fall back to the marginal quantile.
_MIN_CONFORMAL_CLASS_SAMPLES = 10


@dataclass
class PredictionOutput:
    """Result of a single prediction."""

    direction: str  # "long", "short", "flat"
    confidence: float  # 0-1
    expected_return: float  # predicted % return
    probabilities: dict[str, float] = field(default_factory=dict)  # class probabilities
    # Optional per-prediction detail (e.g. "conformal_set", "conformal_alpha",
    # "conformal_gated"). Additive and defaulted so existing constructors and
    # consumers are unaffected.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Metrics returned after a training run."""

    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    epochs_trained: int
    feature_importance: dict[str, float] | None = None
    # Calibration-selection record (tree models): which calibrator won and the
    # multi-class Brier score of each candidate on the validation split.
    chosen_calibration: str | None = None  # "isotonic" | "sigmoid" | "none"
    brier_isotonic: float | None = None
    brier_sigmoid: float | None = None

    def to_metrics(self) -> dict[str, Any]:
        """Flatten the core metrics for the registry / promotion gate.

        ``TrainResult`` has no ``metrics`` attribute, so callers must build the
        dict from these fields (a previous bug read a non-existent attribute and
        silently produced an empty dict, breaking champion/challenger).

        Calibration-selection keys (``chosen_calibration``, ``brier_isotonic``,
        ``brier_sigmoid``) are included only when recorded; the promotion gate
        reads ``val_accuracy`` and is unaffected by them.
        """
        metrics: dict[str, Any] = {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }
        if self.chosen_calibration is not None:
            metrics["chosen_calibration"] = self.chosen_calibration
        if self.brier_isotonic is not None:
            metrics["brier_isotonic"] = self.brier_isotonic
        if self.brier_sigmoid is not None:
            metrics["brier_sigmoid"] = self.brier_sigmoid
        return metrics


# ----------------------------------------------------------------------
# Shared probability helpers (tree-model contract)
# ----------------------------------------------------------------------


def probabilities_in_class_order(
    estimator: Any, features: np.ndarray, n_classes: int = 3
) -> np.ndarray:
    """Return ``(n, n_classes)`` probabilities in canonical class order.

    Columns are remapped via the estimator's ``classes_`` so a class absent
    from the fit data yields a zero column rather than a silently shifted
    distribution. Works for raw classifiers and calibrators alike.
    """
    raw = np.asarray(estimator.predict_proba(features))
    out = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(int(c) for c in estimator.classes_):
        if 0 <= cls < n_classes:
            out[:, cls] = raw[:, col]
    return out


def multiclass_brier(
    probabilities: np.ndarray, y_true: np.ndarray, n_classes: int = 3
) -> float:
    """Multi-class Brier score: mean over one-vs-rest squared errors.

    Equivalent to the mean over all (sample, class) cells of
    ``(p - onehot)^2``; lower is better, 0 is perfect.
    """
    y = np.asarray(y_true).astype(int)
    onehot = np.zeros((len(y), n_classes), dtype=np.float64)
    valid = (y >= 0) & (y < n_classes)
    onehot[np.arange(len(y))[valid], y[valid]] = 1.0
    return float(np.mean((np.asarray(probabilities, dtype=np.float64) - onehot) ** 2))


@dataclass
class CalibrationSelection:
    """Outcome of fitting both calibration candidates on the validation split."""

    calibrator: Any | None  # the winning fitted calibrator (None if both failed)
    method: str  # "isotonic" | "sigmoid" | "none"
    brier_isotonic: float | None = None
    brier_sigmoid: float | None = None


def select_calibration_method(
    classifier: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_splits: int,
    n_classes: int = 3,
) -> CalibrationSelection:
    """Fit isotonic AND sigmoid calibrators; keep the lower-Brier one.

    Both candidates calibrate the *already fitted* classifier on the
    validation split (FrozenEstimator when available, ``cv="prefit"`` on older
    sklearn). A candidate that fails to fit is simply excluded; if both fail
    the selection is ``("none", calibrator=None)``.
    """
    from sklearn.calibration import CalibratedClassifierCV

    def _make(method: str) -> Any:
        try:
            from sklearn.frozen import FrozenEstimator

            return CalibratedClassifierCV(
                FrozenEstimator(classifier), method=method, cv=n_splits
            )
        except ImportError:  # sklearn < 1.6
            return CalibratedClassifierCV(classifier, method=method, cv="prefit")

    fitted: dict[str, Any] = {}
    briers: dict[str, float] = {}
    for method in ("isotonic", "sigmoid"):
        try:
            candidate = _make(method)
            candidate.fit(X_val, y_val)
            probs = probabilities_in_class_order(candidate, X_val, n_classes=n_classes)
            briers[method] = multiclass_brier(probs, y_val, n_classes=n_classes)
            fitted[method] = candidate
        except Exception:
            logger.exception("Fitting %s calibrator failed; excluding it from selection", method)

    if not fitted:
        return CalibrationSelection(
            calibrator=None,
            method="none",
            brier_isotonic=briers.get("isotonic"),
            brier_sigmoid=briers.get("sigmoid"),
        )

    chosen = min(fitted, key=lambda m: briers[m])
    return CalibrationSelection(
        calibrator=fitted[chosen],
        method=chosen,
        brier_isotonic=briers.get("isotonic"),
        brier_sigmoid=briers.get("sigmoid"),
    )


# ----------------------------------------------------------------------
# Split-conformal prediction (tree-model contract)
# ----------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected ``(1 - alpha)`` quantile of nonconformity scores.

    Uses the split-conformal rank ``ceil((n + 1) * (1 - alpha))``; when the
    rank exceeds ``n`` (too few calibration rows for the requested coverage)
    the quantile saturates at 1.0, which makes every class conformally
    admissible (maximally conservative, never blocking).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return 1.0
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return 1.0
    return float(np.sort(scores)[rank - 1])


@dataclass
class ConformalState:
    """Split-conformal calibration state persisted with a model artifact."""

    alpha: float  # target miscoverage rate (default 0.10)
    quantiles: dict[str, float]  # direction name -> nonconformity threshold
    n_calibration: int  # validation rows the quantiles were fitted on

    def prediction_set(self, probabilities: dict[str, float]) -> set[str]:
        """Classes whose nonconformity ``1 - p_class`` is within the quantile."""
        return {
            name
            for name, quantile in self.quantiles.items()
            if (1.0 - float(probabilities.get(name, 0.0))) <= quantile + 1e-12
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "quantiles": dict(self.quantiles),
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalState:
        return cls(
            alpha=float(data["alpha"]),
            quantiles={str(k): float(v) for k, v in dict(data["quantiles"]).items()},
            n_calibration=int(data["n_calibration"]),
        )


def fit_conformal_state(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.10,
    class_names: dict[int, str] | None = None,
    min_samples: int = _MIN_CONFORMAL_SAMPLES,
    min_class_samples: int = _MIN_CONFORMAL_CLASS_SAMPLES,
) -> ConformalState | None:
    """Fit split-conformal quantiles on validation-set calibrated probabilities.

    Nonconformity is ``1 - p_true_class``. Quantiles are class-conditional
    (Mondrian) when a class has at least *min_class_samples* rows, else that
    class falls back to the marginal quantile. Returns ``None`` (no conformal
    state, non-blocking) when fewer than *min_samples* usable rows exist.
    """
    names = dict(class_names) if class_names is not None else dict(CLASS_NAMES)
    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y_true).astype(int)
    valid = (y >= 0) & (y < probs.shape[1])
    probs, y = probs[valid], y[valid]
    n = len(y)
    if n < min_samples:
        logger.info("Skipping conformal calibration: %d usable rows < %d", n, min_samples)
        return None

    scores = 1.0 - probs[np.arange(n), y]
    marginal = conformal_quantile(scores, alpha)
    quantiles: dict[str, float] = {}
    for cls, name in names.items():
        cls_scores = scores[y == cls]
        quantiles[name] = (
            conformal_quantile(cls_scores, alpha)
            if len(cls_scores) >= min_class_samples
            else marginal
        )
    return ConformalState(alpha=float(alpha), quantiles=quantiles, n_calibration=n)


class BasePredictor(ABC):
    """Interface that every prediction model must implement.

    Tree models additionally follow the calibration/conformal contract: they
    keep a ``ConformalState | None`` in ``self._conformal`` after training (or
    after loading an artifact that has one) and attach
    :meth:`_conformal_metadata` to every emitted :class:`PredictionOutput`.
    Models without conformal state (old artifacts, torch models) expose
    ``conformal_state is None`` and are treated as non-blocking downstream.
    """

    model_type: str = "base"
    required_features: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        """Ordered training feature names, if the model recorded them.

        Used at inference to build the feature vector in the *same* column order
        the model was trained on, avoiding train/serve feature skew.
        """
        return list(getattr(self, "_feature_names", []) or [])

    @property
    def conformal_state(self) -> ConformalState | None:
        """Split-conformal state, if the model carries one (else ``None``)."""
        state = getattr(self, "_conformal", None)
        return state if isinstance(state, ConformalState) else None

    def conformal_prediction_set(self, probabilities: dict[str, float]) -> set[str] | None:
        """Conformal prediction SET for given class probabilities.

        Returns ``None`` when the model has no conformal state so callers can
        distinguish "abstention layer unavailable" from an empty set.
        """
        state = self.conformal_state
        if state is None:
            return None
        return state.prediction_set(probabilities)

    def _conformal_metadata(self, probabilities: dict[str, float]) -> dict[str, Any]:
        """Metadata dict to attach to a :class:`PredictionOutput`.

        Empty when no conformal state exists, so downstream gating treats the
        model as vote-only (non-blocking).
        """
        state = self.conformal_state
        if state is None:
            return {}
        return {
            "conformal_set": sorted(state.prediction_set(probabilities)),
            "conformal_alpha": state.alpha,
        }

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionOutput:
        """Predict direction and return for a single sample."""
        ...

    @abstractmethod
    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        """Predict direction and return for a batch of samples."""
        ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        """Train the model and return training metrics."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts to *path*."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model artefacts from *path*."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return a mapping of feature name to importance score."""
        ...
