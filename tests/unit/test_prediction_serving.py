"""Prediction serving correctness: feature ordering, no train/serve scaling.

These guard against the train/serve skew where tree models were trained on
StandardScaler-normalised features in a (training-only) column order, but served
raw features ordered alphabetically -- which silently fed each model column the
wrong feature at inference. A stub predictor is used so the tests do not require
xgboost/lightgbm (which need OpenMP at runtime).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.events.base import InProcessEventBus
from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult
from services.prediction.models.ensemble import EnsemblePredictor
from services.prediction.service import PredictionService
from services.prediction.serving import ModelServer
from services.prediction.training.data_loader import TrainingDataLoader


class _StubPredictor(BasePredictor):
    """Records the feature vector it is asked to predict on."""

    model_type = "xgboost"

    def __init__(self, feature_names: list[str]) -> None:
        self._feature_names = feature_names
        self.last_features: np.ndarray | None = None

    def predict(self, features: np.ndarray) -> PredictionOutput:
        self.last_features = np.asarray(features).reshape(-1)
        return PredictionOutput(
            direction="long",
            confidence=0.9,
            expected_return=0.01,
            probabilities={"long": 0.9, "short": 0.05, "flat": 0.05},
        )

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        return [self.predict(features)]

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:  # noqa: ANN001
        raise NotImplementedError

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...

    def get_feature_importance(self) -> dict[str, float]:
        return {}


def _server_with(stub: _StubPredictor) -> ModelServer:
    server = ModelServer(registry=None, min_agreement=1)  # type: ignore[arg-type]
    server._models = {"xgboost": stub}
    server._ensemble = EnsemblePredictor(models=[stub], min_agreement=1)
    return server


def test_inference_orders_features_by_training_order() -> None:
    """Features must be fed in the model's trained column order, not alphabetical."""
    stub = _StubPredictor(feature_names=["c", "a", "b"])  # deliberately non-alphabetical
    svc = PredictionService(event_bus=InProcessEventBus(), model_server=_server_with(stub))

    svc.predict("AAPL", {"a": 1.0, "b": 2.0, "c": 3.0})

    assert stub.last_features is not None
    # Ordered by feature_names [c, a, b] -> [3, 1, 2], NOT sorted [1, 2, 3].
    np.testing.assert_array_equal(stub.last_features, np.array([3.0, 1.0, 2.0]))


def test_model_server_exposes_feature_names() -> None:
    stub = _StubPredictor(feature_names=["x", "y"])
    assert _server_with(stub).feature_names == ["x", "y"]


def test_missing_feature_defaults_to_zero() -> None:
    stub = _StubPredictor(feature_names=["a", "b", "missing"])
    svc = PredictionService(event_bus=InProcessEventBus(), model_server=_server_with(stub))

    svc.predict("AAPL", {"a": 1.0, "b": 2.0})

    assert stub.last_features is not None
    np.testing.assert_array_equal(stub.last_features, np.array([1.0, 2.0, 0.0]))


def test_data_loader_does_not_scale() -> None:
    """Tabular training data must be returned RAW (no scaler) to match serving."""
    loader = TrainingDataLoader(target_horizon_bars=2)
    features = np.array(
        [[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0], [5.0, 500.0]]
    )
    close = np.array([10.0, 11.0, 12.0, 11.0, 10.0])

    ds = loader.load_training_data(features, close, feature_names=["a", "b"])

    assert ds.scaler is None
    np.testing.assert_array_equal(ds.X, features[: len(ds.y)])
    assert ds.feature_names == ["a", "b"]


def test_train_result_to_metrics() -> None:
    """to_metrics must expose real fields (the promotion gate depends on it)."""
    tr = TrainResult(
        train_loss=0.5,
        val_loss=0.6,
        train_accuracy=0.7,
        val_accuracy=0.65,
        epochs_trained=10,
    )
    metrics = tr.to_metrics()

    assert metrics != {}
    assert metrics["val_accuracy"] == 0.65
    assert metrics["val_loss"] == 0.6
    assert metrics["train_accuracy"] == 0.7
