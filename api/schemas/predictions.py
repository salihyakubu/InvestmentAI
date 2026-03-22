"""Prediction response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel

from core.enums import SignalDirection


class PredictionSchema(BaseModel):
    """A single model prediction."""

    id: uuid.UUID | None = None
    symbol: str
    direction: SignalDirection
    confidence: float
    expected_return: float | None = None
    model_id: str | None = None
    timestamp: datetime

    model_config = {"from_attributes": True}
