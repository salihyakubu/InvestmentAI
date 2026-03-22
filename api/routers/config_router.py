"""Configuration management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from config.settings import get_settings as _get_settings
from api.dependencies import get_current_user
from api.schemas.risk import RiskRulesSchema

router = APIRouter(prefix="/config")


@router.get(
    "/risk-rules",
    response_model=RiskRulesSchema,
    summary="Get current risk rules",
)
async def get_risk_rules(
    _user: dict = Depends(get_current_user),
) -> RiskRulesSchema:
    """Return the current risk rule configuration."""
    settings = _get_settings()
    return RiskRulesSchema(
        max_position_pct=settings.max_position_pct,
        max_sector_pct=settings.max_sector_pct,
        max_asset_class_pct=settings.max_asset_class_pct,
        max_single_order_pct=settings.max_single_order_pct,
        max_daily_drawdown_pct=settings.max_daily_drawdown_pct,
        max_total_drawdown_pct=settings.max_total_drawdown_pct,
        max_pairwise_correlation=settings.max_pairwise_correlation,
        max_portfolio_positions=settings.max_portfolio_positions,
        max_portfolio_var_95=settings.max_portfolio_var_95,
        circuit_breaker_loss_pct=settings.circuit_breaker_loss_pct,
        circuit_breaker_cooldown_minutes=settings.circuit_breaker_cooldown_minutes,
    )


@router.put(
    "/risk-rules",
    response_model=RiskRulesSchema,
    summary="Update risk rules",
)
async def update_risk_rules(
    rules: RiskRulesSchema,
    _user: dict = Depends(get_current_user),
) -> RiskRulesSchema:
    """Update risk rule configuration.

    Changes are applied in-memory for the current session.
    Persistent storage will be added when the config service is built.
    """
    # TODO: Persist to database and notify risk service via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Risk rules persistence not yet implemented",
    )


@router.get(
    "/trading-mode",
    summary="Get current trading mode",
)
async def get_trading_mode(
    _user: dict = Depends(get_current_user),
) -> dict:
    """Return the current trading execution mode."""
    settings = _get_settings()
    return {"trading_mode": settings.trading_mode}


@router.put(
    "/trading-mode",
    summary="Switch trading mode",
)
async def update_trading_mode(
    mode: dict,
    _user: dict = Depends(get_current_user),
) -> dict:
    """Switch between backtest, paper, and live trading modes.

    Requires admin role. Mode changes require service restart.
    """
    # TODO: Validate mode value against TradingMode enum, persist, and restart
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Trading mode switching not yet implemented",
    )
