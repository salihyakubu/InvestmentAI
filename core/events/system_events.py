"""Platform-level system events."""

from __future__ import annotations

from typing import Any

from core.events.base import Event


class ModelRetrainedEvent(Event):
    """Emitted after a model is retrained and a new version is persisted."""

    model_id: str
    version: int
    metrics: dict[str, Any]


class DriftDetectedEvent(Event):
    """Emitted when data or concept drift is detected."""

    drift_type: str  # 'data', 'concept', 'prediction'
    score: float
    threshold: float


class LiquidationTriggeredEvent(Event):
    """Emitted when a forced liquidation is initiated."""

    position_id: str
    symbol: str
    reason: str
    quantity: float
