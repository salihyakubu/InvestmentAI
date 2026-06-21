"""The prediction service must gate low-confidence signals to flat (abstain),
wiring up settings.min_prediction_confidence (previously defined but unused), so
uncertain predictions don't drive trades.
"""

from __future__ import annotations

from core.events.base import InProcessEventBus
from services.prediction.models.base import PredictionOutput
from services.prediction.service import PredictionService


class _FakeServer:
    """Minimal model server returning a fixed output (bypasses the ensemble)."""

    def __init__(self, direction: str, confidence: float) -> None:
        self._out = PredictionOutput(
            direction=direction,
            confidence=confidence,
            expected_return=0.01,
            probabilities={direction: confidence},
        )

    @property
    def feature_names(self) -> list[str]:
        return []

    @property
    def active_model_types(self) -> list[str]:
        return ["xgboost"]

    def predict(self, symbol, features_flat=None, features_sequence=None):
        return self._out


def _service(direction: str, confidence: float) -> PredictionService:
    # settings=None -> default minimum confidence of 0.6.
    return PredictionService(
        event_bus=InProcessEventBus(), model_server=_FakeServer(direction, confidence)
    )


def test_low_confidence_signal_abstains_to_flat() -> None:
    pred = _service("long", 0.4).predict("AAPL", {"a": 1.0})  # 0.4 < 0.6
    assert pred.direction == "flat"
    assert pred.confidence == 0.4  # original confidence retained on the record


def test_high_confidence_signal_passes() -> None:
    pred = _service("long", 0.9).predict("AAPL", {"a": 1.0})
    assert pred.direction == "long"


def test_low_confidence_flat_stays_flat() -> None:
    pred = _service("flat", 0.3).predict("AAPL", {"a": 1.0})
    assert pred.direction == "flat"
