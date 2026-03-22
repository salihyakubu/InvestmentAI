"""Shared type aliases used across the InvestAI platform."""

from __future__ import annotations

from decimal import Decimal
from typing import NewType

# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

Symbol = str
"""Ticker symbol, e.g. 'AAPL' or 'BTC-USD'."""

ModelID = NewType("ModelID", str)
OrderID = NewType("OrderID", str)
PositionID = NewType("PositionID", str)
PredictionID = NewType("PredictionID", str)

# ---------------------------------------------------------------------------
# Numeric types wrapping Decimal for financial precision
# ---------------------------------------------------------------------------

Price = NewType("Price", Decimal)
Quantity = NewType("Quantity", Decimal)
Money = NewType("Money", Decimal)

# ---------------------------------------------------------------------------
# ML feature representations
# ---------------------------------------------------------------------------

FeatureVector = dict[str, float]
"""A mapping of feature names to their computed float values."""
