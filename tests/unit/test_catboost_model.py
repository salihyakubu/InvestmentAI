"""CatBoostPredictor parity with the tree-model contract.

Mirrors the XGBoost/LightGBM behaviour: 3-class classifier + forward-return
regressor, isotonic-vs-sigmoid calibration selection by Brier score,
split-conformal state, feature names, and a registry-compatible save/load
round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

from services.prediction.models.base import BasePredictor
from services.prediction.models.catboost_model import CatBoostPredictor

FAST = {"iterations": 60, "depth": 4}


def _dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=900, n_features=8, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=1.2, random_state=seed,
    )
    return X[:600], y[:600], X[600:], y[600:]


@pytest.fixture(scope="module")
def trained():
    x_tr, y_tr, x_val, y_val = _dataset()
    rng = np.random.default_rng(7)
    returns_tr = rng.normal(0.0, 0.01, size=len(y_tr))
    returns_val = rng.normal(0.0, 0.01, size=len(y_val))
    model = CatBoostPredictor(classifier_params=FAST, regressor_params=FAST)
    result = model.train(
        x_tr, y_tr, x_val, y_val, returns_train=returns_tr, returns_val=returns_val
    )
    return model, result, x_val, y_val


def test_is_base_predictor_with_type() -> None:
    model = CatBoostPredictor()
    assert isinstance(model, BasePredictor)
    assert model.model_type == "catboost"


def test_train_metrics_and_prediction_contract(trained) -> None:
    model, result, x_val, y_val = trained

    # Separable synthetic data: the classifier must actually learn.
    assert result.val_accuracy > 0.7
    assert result.train_loss > 0 and result.val_loss > 0
    assert result.epochs_trained >= 1
    assert result.feature_importance  # non-empty

    out = model.predict(x_val[0])
    assert out.direction in {"short", "flat", "long"}
    assert 0.0 <= out.confidence <= 1.0
    assert np.isfinite(out.expected_return)
    assert set(out.probabilities) == {"short", "flat", "long"}
    assert sum(out.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert out.confidence == pytest.approx(max(out.probabilities.values()))

    batch = model.predict_batch(x_val[:5])
    assert len(batch) == 5
    assert batch[0].direction == out.direction
    assert batch[0].probabilities == pytest.approx(out.probabilities)


def test_to_metrics_registry_shape(trained) -> None:
    _, result, _, _ = trained
    metrics = result.to_metrics()
    for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        assert isinstance(metrics[key], float)
    # Calibration-selection record present and JSON-safe.
    assert metrics["chosen_calibration"] in {"isotonic", "sigmoid", "none"}
    import json

    json.dumps(metrics)


def test_calibration_selection_recorded(trained) -> None:
    model, result, _, _ = trained
    # 300 validation rows across 3 classes: both calibrators must fit.
    assert model._calibration_method in {"isotonic", "sigmoid"}
    assert model._calibrator is not None
    assert result.chosen_calibration == model._calibration_method
    assert result.brier_isotonic is not None and result.brier_sigmoid is not None
    briers = {"isotonic": result.brier_isotonic, "sigmoid": result.brier_sigmoid}
    assert result.chosen_calibration == min(briers, key=briers.get)


def test_conformal_state_and_metadata(trained) -> None:
    model, _, x_val, y_val = trained
    assert model._conformal is not None
    assert model._conformal.alpha == pytest.approx(0.10)

    names = {0: "short", 1: "flat", 2: "long"}
    covered = 0
    for out, y in zip(model.predict_batch(x_val), y_val):
        cset = out.metadata.get("conformal_set")
        assert cset is not None
        assert set(cset) <= {"short", "flat", "long"}
        assert out.metadata["conformal_alpha"] == pytest.approx(0.10)
        if names[int(y)] in cset:
            covered += 1
    # In-sample marginal coverage tracks 1 - alpha by construction.
    assert covered / len(y_val) >= 0.89


def test_save_load_round_trip(trained, tmp_path: Path) -> None:
    model, _, x_val, _ = trained
    before_probs = model._proba(x_val)
    before_returns = [model.predict(x_val[i]).expected_return for i in range(3)]
    model._feature_names = [f"feat_{i}" for i in range(x_val.shape[1])]
    model.save(tmp_path)
    for artifact in ("classifier.joblib", "regressor.joblib", "feature_names.joblib",
                     "calibrator.joblib", "calibration_meta.joblib", "conformal.joblib"):
        assert (tmp_path / artifact).exists(), artifact

    restored = CatBoostPredictor()
    restored.load(tmp_path)
    assert np.allclose(before_probs, restored._proba(x_val))
    assert restored.feature_names == model._feature_names
    assert restored._calibration_method == model._calibration_method
    assert restored._brier_isotonic == pytest.approx(model._brier_isotonic)
    assert restored._brier_sigmoid == pytest.approx(model._brier_sigmoid)
    assert restored._conformal == model._conformal
    after_returns = [restored.predict(x_val[i]).expected_return for i in range(3)]
    assert after_returns == pytest.approx(before_returns)
    imp = restored.get_feature_importance()
    assert set(imp) == set(model._feature_names)


def test_registry_compatible_artifact_layout(trained, tmp_path: Path) -> None:
    """register() persists via model.save; a fresh predictor reloads the dir."""
    from services.prediction.registry import ModelRegistry

    model, result, x_val, _ = trained
    registry = ModelRegistry(artifact_base=tmp_path)
    model_id, version = registry.register(
        model=model, model_name="catboost", metrics=result.to_metrics()
    )
    registry.promote(model_id, version)

    entry = registry.list_versions("catboost")[0]
    assert entry.model_type == "catboost"
    assert entry.is_active

    restored = CatBoostPredictor()
    restored.load(Path(entry.artifact_path))
    assert np.allclose(model._proba(x_val), restored._proba(x_val))

    # The registry JSON round-trips (metrics include the str calibration key).
    reloaded = ModelRegistry(artifact_base=tmp_path)
    assert reloaded.list_versions("catboost")[0].metrics == entry.metrics


def test_legacy_artifact_without_meta_or_conformal(trained, tmp_path: Path) -> None:
    model, _, x_val, _ = trained
    model.save(tmp_path)
    (tmp_path / "calibration_meta.joblib").unlink()
    (tmp_path / "conformal.joblib").unlink()

    restored = CatBoostPredictor()
    restored.load(tmp_path)
    # Pre-selection artifact with a calibrator present -> inferred sigmoid.
    assert restored._calibration_method == "sigmoid"
    assert restored._brier_isotonic is None and restored._brier_sigmoid is None
    assert restored._conformal is None and restored.conformal_state is None
    out = restored.predict(x_val[0])
    assert out.direction in {"short", "flat", "long"}
    assert "conformal_set" not in out.metadata  # non-blocking downstream

    (tmp_path / "calibrator.joblib").unlink()
    bare = CatBoostPredictor()
    bare.load(tmp_path)
    assert bare._calibration_method == "none"
    assert bare.predict(x_val[0]).direction in {"short", "flat", "long"}


def test_trainer_and_hyperopt_registration() -> None:
    from services.prediction.training.hyperopt import HyperOptimizer
    from services.prediction.training.trainer import _MODEL_CLASSES, ModelTrainer

    assert _MODEL_CLASSES["catboost"] is CatBoostPredictor

    model = ModelTrainer._create_model("catboost", dict(FAST), feature_names=["a", "b"])
    assert isinstance(model, CatBoostPredictor)
    assert model.feature_names == ["a", "b"]

    hyper_model = HyperOptimizer(model_type="catboost", n_trials=1)._create_model(dict(FAST))
    assert isinstance(hyper_model, CatBoostPredictor)


def test_train_without_returns_falls_back_to_label_map() -> None:
    x_tr, y_tr, x_val, y_val = _dataset(seed=3)
    model = CatBoostPredictor(classifier_params=FAST, regressor_params=FAST)
    model.train(x_tr[:200], y_tr[:200], x_val[:100], y_val[:100])
    out = model.predict(x_val[0])
    # Label-map returns are in [-1%, +1%]; regressor output must stay bounded.
    assert -0.05 < out.expected_return < 0.05
