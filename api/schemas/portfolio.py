"""Portfolio-related response schemas."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class PortfolioSummary(BaseModel):
    """Current portfolio state summary."""

    total_equity: Decimal
    cash: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_return_pct: float | None = None


class AllocationSchema(BaseModel):
    """Single allocation entry."""

    symbol: str
    weight: float
    value: Decimal


class PositionSchema(BaseModel):
    """Position summary within the portfolio context."""

    symbol: str
    side: str
    quantity: Decimal
    avg_entry: Decimal
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None

    model_config = {"from_attributes": True}
