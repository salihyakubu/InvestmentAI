"""Portfolio-related response schemas."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel


class PortfolioSummary(BaseModel):
    """Current portfolio state and performance.

    The performance fields are ``None`` when the stored history cannot support
    them (for example a Sharpe ratio over a handful of days). Clients must
    render that as "insufficient history" rather than as zero -- a fabricated
    0.00 is indistinguishable from a real one.
    """

    total_equity: Decimal
    cash: Decimal
    positions_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_return_pct: float | None = None

    daily_pnl: Decimal = Decimal("0")
    daily_pnl_pct: float | None = None
    total_return: Decimal = Decimal("0")
    total_return_pct: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    closed_trades: int = 0
    position_count: int = 0


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
