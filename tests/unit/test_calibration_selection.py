"""Calibration method selection: isotonic vs sigmoid chosen by validation Brier.

Tree models must fit BOTH calibrators on the validation split and keep the one
with the lower multi-class Brier score (mean over one-vs-rest), record the
selection in TrainResult, and persist it across save/load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from services.prediction.models.base import (
    multiclass_brier,
    select_calibration_method,
)
from services.prediction.models.xgboost_model import XGBoostPredictor


def _dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=900, n_features=8, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.2, random_state=seed,
    )
    return X[:600], y[:600], X[600:], y[600:]


@pytest.fixture(scope="module")
def trained():
    x_tr, y_tr, x_val, y_val = _dataset()
    fast = {"n_estimators": 60, "max_depth": 4}
    model = XGBoostPredictor(classifier_params=fast, regressor_params=fast)
    result = model.train(x_tr, y_tr, x_val, y_val)
    return model, result, x_val, y_val


def test_multiclass_brier_known_values() -> None:
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert multiclass_brier(perfect, y) == pytest.approx(0.0)

    uniform = np.full((3, 3), 1.0 / 3.0)
    # per row: (1/3 - 1)^2 + 2 * (1/3)^2 = 4/9 + 2/9 = 6/9; mean over 9 cells = 2/9
    assert multiclass_brier(uniform, y) == pytest.approx(2.0 / 9.0)

    # Worse (anti-diagonal) probabilities must score strictly worse than uniform.
    wrong = np.roll(np.eye(3), 1, axis=1)
    assert multiclass_brier(wrong, y) > multiclass_brier(uniform, y)


def test_selection_picks_lower_brier_method(trained) -> None:
    model, _, x_val, y_val = trained
    selection = select_calibration_method(model._classifier, x_val, y_val, n_splits=3)

    assert selection.brier_isotonic is not None
    assert selection.brier_sigmoid is not None
    briers = {"isotonic": selection.brier_isotonic, "sigmoid": selection.brier_sigmoid}
    assert selection.method == min(briers, key=briers.get)
    assert selection.calibrator is not None


def test_train_records_selection_in_result_and_metrics(trained) -> None:
    model, result, _, _ = trained

    assert model._calibration_method in {"isotonic", "sigmoid"}
    assert model._calibrator is not None
    assert result.chosen_calibration == model._calibration_method
    assert result.brier_isotonic is not None
    assert result.brier_sigmoid is not None
    briers = {"isotonic": result.brier_isotonic, "sigmoid": result.brier_sigmoid}
    assert result.chosen_calibration == min(briers, key=briers.get)

    metrics = result.to_metrics()
    assert metrics["chosen_calibration"] == result.chosen_calibration
    assert metrics["brier_isotonic"] == pytest.approx(result.brier_isotonic)
    assert metrics["brier_sigmoid"] == pytest.approx(result.brier_sigmoid)
    # Promotion-gate keys are untouched.
    assert "val_accuracy" in metrics and "val_loss" in metrics


def test_selection_survives_save_load(trained, tmp_path: Path) -> None:
    model, _, x_val, _ = trained
    before = model._proba(x_val)
    model.save(tmp_path)
    assert (tmp_path / "calibration_meta.joblib").exists()

    restored = XGBoostPredictor()
    restored.load(tmp_path)
    assert restored._calibration_method == model._calibration_method
    assert restored._brier_isotonic == pytest.approx(model._brier_isotonic)
    assert restored._brier_sigmoid == pytest.approx(model._brier_sigmoid)
    assert np.allclose(before, restored._proba(x_val))


def test_legacy_artifact_without_meta_infers_method(trained, tmp_path: Path) -> None:
    model, _, x_val, _ = trained
    model.save(tmp_path)

    # Simulate a pre-selection artifact: no calibration_meta, sigmoid-era calibrator.
    (tmp_path / "calibration_meta.joblib").unlink()
    restored = XGBoostPredictor()
    restored.load(tmp_path)
    assert restored._calibration_method == "sigmoid"
    assert restored._brier_isotonic is None and restored._brier_sigmoid is None
    assert restored.predict(x_val[0]).direction in {"short", "flat", "long"}

    # And with no calibrator at all -> "none".
    (tmp_path / "calibrator.joblib").unlink()
    bare = XGBoostPredictor()
    bare.load(tmp_path)
    assert bare._calibration_method == "none"
    assert bare.predict(x_val[0]).direction in {"short", "flat", "long"}
