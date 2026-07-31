"""Risk-related response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RiskMetricsSchema(BaseModel):
    """Current risk metrics snapshot.

    ``circuit_breaker_active`` is deliberately nullable: nothing writes the
    ``risk_metrics`` table yet, and defaulting an unreported breaker to
    ``False`` renders a green "all systems normal" shield that is a pure
    artifact of missing data. Absent must read as unknown, not as safe.
    """

    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    cvar_99: float | None = None
    max_drawdown: float | None = None
    current_drawdown: float | None = None
    volatility: float | None = None
    correlation_max: float | None = None
    concentration_max: float | None = None
    daily_pnl_pct: float | None = None
    circuit_breaker_active: bool | None = None
    circuit_breaker_state: str | None = None
    circuit_breaker_reason: str | None = None
    reported: bool = False


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
