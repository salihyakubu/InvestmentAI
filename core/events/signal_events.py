"""Signal / ML pipeline events."""

from __future__ import annotations

from pydantic import Field

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
    # Calibrated class probabilities keyed by SignalDirection value
    # ("long"/"flat"/"short"). The portfolio optimizer's edge-margin gate
    # needs the full map, not just max-class confidence; an empty map (e.g.
    # an event from a pre-upgrade publisher) fails that gate closed.
    probabilities: dict[str, float] = Field(default_factory=dict)
