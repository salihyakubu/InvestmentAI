"""Split-conformal prediction: quantile math, state fitting, model persistence.

Nonconformity is ``1 - calibrated probability of the true class``; the stored
(1 - alpha) quantiles define per-prediction class SETS. Models without
conformal state (old artifacts) must keep loading and predicting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from services.prediction.models.base import (
    ConformalState,
    conformal_quantile,
    fit_conformal_state,
)
from services.prediction.models.xgboost_model import XGBoostPredictor

# ----------------------------------------------------------------------
# Quantile math on known distributions
# ----------------------------------------------------------------------


def test_conformal_quantile_known_distribution() -> None:
    scores = np.arange(1, 11) / 10.0  # 0.1 .. 1.0, n=10

    # alpha=0.5: rank = ceil(11 * 0.5) = 6 -> 6th smallest = 0.6
    assert conformal_quantile(scores, alpha=0.5) == pytest.approx(0.6)
    # alpha=0.1: rank = ceil(11 * 0.9) = 10 -> 10th smallest = 1.0
    assert conformal_quantile(scores, alpha=0.1) == pytest.approx(1.0)

    grid = np.linspace(0.01, 0.99, 99)  # n=99
    # rank = ceil(100 * 0.9) = 90 -> 90th smallest = 0.90
    assert conformal_quantile(grid, alpha=0.1) == pytest.approx(0.90)

    # Too few rows for the requested coverage -> saturates at 1.0 (never blocks).
    assert conformal_quantile(np.array([0.2]), alpha=0.1) == pytest.approx(1.0)
    assert conformal_quantile(np.array([]), alpha=0.1) == pytest.approx(1.0)

    # Order-invariance: quantile works on unsorted scores.
    shuffled = scores.copy()
    np.random.default_rng(0).shuffle(shuffled)
    assert conformal_quantile(shuffled, alpha=0.5) == pytest.approx(0.6)


def test_fit_conformal_state_known_scores() -> None:
    # 120 rows, 40 per class, true-class probability always 0.9 -> scores all 0.1.
    y = np.repeat([0, 1, 2], 40)
    probs = np.full((120, 3), 0.05)
    probs[np.arange(120), y] = 0.9

    state = fit_conformal_state(probs, y, alpha=0.10)
    assert state is not None
    assert state.alpha == pytest.approx(0.10)
    assert state.n_calibration == 120
    assert set(state.quantiles) == {"short", "flat", "long"}
    for q in state.quantiles.values():
        assert q == pytest.approx(0.1)

    # Prediction set: only classes with nonconformity <= 0.1 (prob >= 0.9).
    assert state.prediction_set({"short": 0.95, "flat": 0.5, "long": 0.2}) == {"short"}
    assert state.prediction_set({"short": 0.9, "flat": 0.9, "long": 0.1}) == {"short", "flat"}
    assert state.prediction_set({"short": 0.5, "flat": 0.5, "long": 0.5}) == set()


def test_fit_conformal_state_class_conditional_with_marginal_fallback() -> None:
    # class 0: 60 rows score 0.2 | class 1: 60 rows score 0.05 | class 2: 5 rows score 0.5
    y = np.concatenate([np.zeros(60, int), np.ones(60, int), np.full(5, 2, int)])
    true_prob = np.concatenate([np.full(60, 0.8), np.full(60, 0.95), np.full(5, 0.5)])
    probs = np.full((125, 3), 0.01)
    probs[np.arange(125), y] = true_prob

    state = fit_conformal_state(probs, y, alpha=0.10)
    assert state is not None
    # Marginal quantile: rank ceil(126 * 0.9) = 114 over sorted scores
    # (60 x 0.05, 60 x 0.2, 5 x 0.5) -> 0.2.
    assert state.quantiles["short"] == pytest.approx(0.2)  # class 0, own quantile
    assert state.quantiles["flat"] == pytest.approx(0.05)  # class 1, own quantile
    # class 2 has 5 < 10 rows -> falls back to the marginal quantile.
    assert state.quantiles["long"] == pytest.approx(0.2)


def test_fit_conformal_state_guards() -> None:
    # Too few rows -> no conformal state (non-blocking downstream).
    y = np.repeat([0, 1, 2], 5)
    probs = np.full((15, 3), 1.0 / 3.0)
    assert fit_conformal_state(probs, y, alpha=0.10) is None

    # Out-of-range labels are ignored, not crashed on.
    y_bad = np.concatenate([np.repeat([0, 1, 2], 20), np.full(10, 7, int)])
    probs_ok = np.full((70, 3), 1.0 / 3.0)
    state = fit_conformal_state(probs_ok, y_bad, alpha=0.10)
    assert state is not None
    assert state.n_calibration == 60  # only the valid rows counted


def test_conformal_state_dict_roundtrip() -> None:
    state = ConformalState(
        alpha=0.1, quantiles={"short": 0.2, "flat": 0.3, "long": 0.4}, n_calibration=99
    )
    restored = ConformalState.from_dict(state.to_dict())
    assert restored == state


# ----------------------------------------------------------------------
# Model integration: fit at train time, persist, backward compatible
# ----------------------------------------------------------------------


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
    model.train(x_tr, y_tr, x_val, y_val)
    return model, x_val, y_val


def test_train_fits_conformal_state_with_coverage(trained) -> None:
    model, x_val, y_val = trained
    assert model._conformal is not None
    assert model._conformal.alpha == pytest.approx(0.10)

    # In-sample marginal coverage: the true class must be in the conformal set
    # for >= ~90% of the calibration rows (deterministic by quantile choice).
    names = {0: "short", 1: "flat", 2: "long"}
    covered = 0
    for out, y in zip(model.predict_batch(x_val), y_val):
        cset = out.metadata.get("conformal_set")
        assert cset is not None
        assert set(cset) <= {"short", "flat", "long"}
        assert out.metadata["conformal_alpha"] == pytest.approx(0.10)
        if names[int(y)] in cset:
            covered += 1
    assert covered / len(y_val) >= 0.89


def test_conformal_prediction_set_accessor(trained) -> None:
    model, x_val, _ = trained
    out = model.predict(x_val[0])
    cset = model.conformal_prediction_set(out.probabilities)
    assert cset is not None
    assert sorted(cset) == out.metadata["conformal_set"]


def test_conformal_state_survives_save_load(trained, tmp_path: Path) -> None:
    model, x_val, _ = trained
    model.save(tmp_path)
    assert (tmp_path / "conformal.joblib").exists()

    restored = XGBoostPredictor()
    restored.load(tmp_path)
    assert restored._conformal == model._conformal

    out_before = model.predict(x_val[3])
    out_after = restored.predict(x_val[3])
    assert out_after.metadata["conformal_set"] == out_before.metadata["conformal_set"]


def test_old_artifact_without_conformal_state_still_predicts(trained, tmp_path: Path) -> None:
    model, x_val, _ = trained
    model.save(tmp_path)
    (tmp_path / "conformal.joblib").unlink()  # simulate a pre-conformal artifact

    restored = XGBoostPredictor()
    restored.load(tmp_path)
    assert restored._conformal is None
    assert restored.conformal_state is None

    out = restored.predict(x_val[0])
    assert out.direction in {"short", "flat", "long"}
    assert "conformal_set" not in out.metadata  # non-blocking downstream
    assert restored.conformal_prediction_set(out.probabilities) is None
