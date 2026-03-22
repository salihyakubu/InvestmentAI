"""Risk-related response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RiskMetricsSchema(BaseModel):
    """Current risk metrics snapshot."""

    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    max_drawdown: float | None = None
    current_drawdown: float | None = None
    circuit_breaker_active: bool = False


class RiskRulesSchema(BaseModel):
    """Configurable risk rule parameters."""

    max_position_pct: float
    max_sector_pct: float
    max_asset_class_pct: float
    max_single_order_pct: float
    max_daily_drawdown_pct: float
    max_total_drawdown_pct: float
    max_pairwise_correlation: float
    max_portfolio_positions: int
    max_portfolio_var_95: float
    circuit_breaker_loss_pct: float
    circuit_breaker_cooldown_minutes: int
