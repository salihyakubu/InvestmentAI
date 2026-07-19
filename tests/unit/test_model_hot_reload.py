"""Model hot-reload: keep-previous guard + event-driven reload.

A newly promoted model version must reach serving WITHOUT a worker restart:
the prediction service subscribes to the SYSTEM stream and re-runs
``ModelServer.load_active_models()`` when a ``ModelRetrainedEvent`` arrives.
A reload that finds *no* active models must keep the previous ensemble
(instead of silently degrading a live serving instance to flat predictions).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.events.base import InProcessEventBus
from core.events.streams import SYSTEM
from core.events.system_events import DriftDetectedEvent, ModelRetrainedEvent
from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult
from services.prediction.service import PredictionService
from services.prediction.serving import ModelServer


class _StubPredictor(BasePredictor):
    """Tiny predictor stub (avoids the xgboost/OpenMP runtime dependency)."""

    model_type = "xgboost"

    def __init__(self) -> None:
        self._feature_names = ["a", "b"]

    def predict(self, features: np.ndarray) -> PredictionOutput:
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


class _FakeRegistry:
    """Minimal registry double: model_type -> (predictor, metadata)."""

    def __init__(self) -> None:
        self.active: dict[str, tuple[BasePredictor, SimpleNamespace]] = {}

    def get_active(self, model_type: str) -> tuple[BasePredictor, SimpleNamespace]:
        try:
            return self.active[model_type]
        except KeyError:
            raise ValueError(f"No active model for type '{model_type}'") from None


def _server(registry: _FakeRegistry) -> ModelServer:
    return ModelServer(registry=registry, min_agreement=1)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# (a) load_active_models keep-previous guard
# ----------------------------------------------------------------------


def test_reload_finding_no_models_keeps_previous_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-call that loads ZERO models must keep the existing ensemble."""
    monkeypatch.setenv("WORKER_MODE", "lite")  # only tree model types probed
    registry = _FakeRegistry()
    stub = _StubPredictor()
    registry.active["xgboost"] = (stub, SimpleNamespace(model_name="stub", version=1))

    server = _server(registry)
    loaded = server.load_active_models()
    assert set(loaded) == {"xgboost"}
    first_ensemble = server._ensemble
    assert first_ensemble is not None

    # Registry loses all active versions (e.g. transient failure during a
    # reload): get_active now raises ValueError for every type.
    registry.active.clear()
    result = server.load_active_models()

    # Previous ensemble and models are UNCHANGED -- not degraded to flat.
    assert server._ensemble is first_ensemble
    assert server.active_model_types == ["xgboost"]
    assert result == {"xgboost": stub}
    assert server.predict("AAPL", features_flat=np.zeros(2)).direction == "long"


def test_first_call_with_empty_registry_is_flat(monkeypatch: pytest.MonkeyPatch) -> None:
    """First-call behavior unchanged: no previous ensemble -> flat predictions."""
    monkeypatch.setenv("WORKER_MODE", "lite")
    server = _server(_FakeRegistry())

    loaded = server.load_active_models()

    assert loaded == {}
    assert server._ensemble is None
    assert server.active_model_types == []
    assert server.predict("AAPL", features_flat=np.zeros(2)).direction == "flat"


def test_reload_swaps_in_newly_promoted_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful reload replaces both the models and the ensemble."""
    monkeypatch.setenv("WORKER_MODE", "lite")
    registry = _FakeRegistry()
    registry.active["xgboost"] = (_StubPredictor(), SimpleNamespace(model_name="stub", version=1))

    server = _server(registry)
    server.load_active_models()
    first_ensemble = server._ensemble

    new_stub = _StubPredictor()
    registry.active["xgboost"] = (new_stub, SimpleNamespace(model_name="stub", version=2))
    server.load_active_models()

    assert server._models["xgboost"] is new_stub
    assert server._ensemble is not None
    assert server._ensemble is not first_ensemble


# ----------------------------------------------------------------------
# (b) PredictionService: ModelRetrainedEvent on SYSTEM triggers a reload
# ----------------------------------------------------------------------


class _SpyModelServer:
    """Stands in for ModelServer; counts load_active_models calls."""

    def __init__(self) -> None:
        self.load_calls = 0

    def load_active_models(self) -> dict[str, BasePredictor]:
        self.load_calls += 1
        return {}

    @property
    def active_model_types(self) -> list[str]:
        return []


@pytest.mark.asyncio
async def test_model_retrained_event_triggers_reload() -> None:
    bus = InProcessEventBus()
    spy = _SpyModelServer()
    svc = PredictionService(event_bus=bus, model_server=spy)  # type: ignore[arg-type]

    await svc.start()
    assert spy.load_calls == 1  # initial load at startup

    await bus.publish(
        SYSTEM,
        ModelRetrainedEvent(
            source_service="continuous_learning",
            model_id="xgboost_v2",
            version=2,
            metrics={"val_accuracy": 0.7},
        ),
    )

    assert spy.load_calls == 2  # hot reload -- no restart required
    await svc.stop()


@pytest.mark.asyncio
async def test_non_retrain_system_event_is_ignored() -> None:
    bus = InProcessEventBus()
    spy = _SpyModelServer()
    svc = PredictionService(event_bus=bus, model_server=spy)  # type: ignore[arg-type]

    await svc.start()
    await bus.publish(
        SYSTEM,
        DriftDetectedEvent(
            source_service="continuous_learning",
            drift_type="prediction",
            score=0.5,
            threshold=0.3,
        ),
    )

    assert spy.load_calls == 1  # only the startup load
    await svc.stop()


@pytest.mark.asyncio
async def test_retrain_event_without_model_server_is_noop() -> None:
    bus = InProcessEventBus()
    svc = PredictionService(event_bus=bus, model_server=None)

    await svc.start()
    await bus.publish(
        SYSTEM,
        ModelRetrainedEvent(
            source_service="continuous_learning",
            model_id="xgboost_v2",
            version=2,
            metrics={},
        ),
    )  # must not raise
    await svc.stop()
