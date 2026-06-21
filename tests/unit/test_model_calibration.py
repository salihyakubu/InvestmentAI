"""Probability calibration for the tree models.

The live confidence gate (``min_prediction_confidence``) only protects capital
if the reported confidence is a real probability, so ``train()`` must fit and
persist a calibrator and ``predict()`` must actually use it.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

from services.prediction.models.xgboost_model import XGBoostPredictor


def _dataset(seed: int = 0):
    X, y = make_classification(
        n_samples=900, n_features=8, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.2, random_state=seed,
    )
    return X[:600], y[:600], X[600:], y[600:]


def _fast_predictor() -> XGBoostPredictor:
    fast = {"n_estimators": 60, "max_depth": 4}
    return XGBoostPredictor(classifier_params=fast, regressor_params=fast)


def test_train_fits_calibrator_and_predict_uses_it() -> None:
    x_tr, y_tr, x_val, y_val = _dataset()
    model = _fast_predictor()
    model.train(x_tr, y_tr, x_val, y_val)

    assert model._calibrator is not None  # fitted on the validation split

    out = model.predict(x_val[0])
    assert set(out.probabilities) == {"short", "flat", "long"}
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= out.confidence <= 1.0
    assert out.direction in {"short", "flat", "long"}

    # Calibrated probabilities differ from the raw classifier softmax -> proof the
    # calibrator is actually applied, not silently bypassed.
    cal = model._proba(x_val)
    raw_cls = model._classifier.predict_proba(x_val)
    raw = np.zeros_like(cal)
    for col, cls in enumerate(int(c) for c in model._classifier.classes_):
        raw[:, cls] = raw_cls[:, col]
    assert np.abs(cal - raw).max() > 1e-3


def test_predict_is_accurate_on_separable_data() -> None:
    x_tr, y_tr, x_val, y_val = _dataset()
    model = _fast_predictor()
    model.train(x_tr, y_tr, x_val, y_val)

    inv = {"short": 0, "flat": 1, "long": 2}
    preds = [inv[model.predict(x).direction] for x in x_val]
    acc = float(np.mean(np.array(preds) == y_val))
    assert acc > 0.55  # comfortably above 1/3 chance on separable data


def test_calibrator_persists_across_save_load(tmp_path) -> None:
    x_tr, y_tr, x_val, y_val = _dataset()
    model = _fast_predictor()
    model.train(x_tr, y_tr, x_val, y_val)
    before = model._proba(x_val)

    model.save(tmp_path)
    assert (tmp_path / "calibrator.joblib").exists()

    restored = XGBoostPredictor()
    restored.load(tmp_path)
    assert restored._calibrator is not None
    assert np.allclose(before, restored._proba(x_val))


def test_calibration_skipped_when_validation_coverage_is_poor() -> None:
    x_tr, y_tr, x_val, y_val = _dataset()
    model = _fast_predictor()
    model.train(x_tr, y_tr, x_val, y_val)  # establishes a fitted classifier

    assert model._fit_calibrator(x_val[:10], y_val[:10]) is None  # too few rows
    assert model._fit_calibrator(x_val, np.zeros(len(y_val), dtype=int)) is None  # one class
