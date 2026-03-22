"""Signal / ML pipeline events."""

from __future__ import annotations

from core.events.base import Event
from core.types import FeatureVector


class FeaturesReadyEvent(Event):
    """Emitted when the feature pipeline has computed features for a symbol."""

    symbol: str
    feature_vector: FeatureVector


class PredictionReadyEvent(Event):
    """Emitted when the ML model produces a prediction for a symbol."""

    symbol: str
    direction: str  # SignalDirection value
    confidence: float
    expected_return: float | None = None
    model_id: str
