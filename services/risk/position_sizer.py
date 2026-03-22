"""Position sizing strategies for risk-aware order construction."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from decimal import Decimal

import numpy as np

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """Base class for position sizing strategies."""

    @abstractmethod
    def size(self, **kwargs) -> Decimal:
        """Return the dollar amount to allocate to a position."""


class FixedFractionalSizer(PositionSizer):
    """Allocate a fixed fraction of equity to each position.

    Parameters
    ----------
    fraction:
        Fraction of equity to risk per position (e.g. 0.02 for 2 %).
    """

    def __init__(self, fraction: float = 0.02) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = Decimal(str(fraction))

    def size(self, *, equity: Decimal, **_kwargs) -> Decimal:
        """Return ``equity * fraction``."""
        return equity * self.fraction


class KellyCriterionSizer(PositionSizer):
    """Kelly Criterion position sizing with a fractional multiplier.

    Uses quarter-Kelly by default to reduce variance while still
    capturing the majority of the theoretical growth rate.

    Parameters
    ----------
    kelly_fraction:
        Fraction of the full Kelly bet to use (default 0.25 = quarter-Kelly).
    """

    def __init__(self, kelly_fraction: float = 0.25) -> None:
        if not 0.0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        self.kelly_fraction = kelly_fraction

    def size(
        self,
        *,
        win_rate: float,
        win_loss_ratio: float,
        equity: Decimal,
        max_pct: float = 0.10,
        **_kwargs,
    ) -> Decimal:
        """Compute the Kelly-optimal position size.

        Parameters
        ----------
        win_rate:
            Historical win probability (0, 1).
        win_loss_ratio:
            Ratio of average win to average loss (> 0).
        equity:
            Current account equity.
        max_pct:
            Hard cap on allocation as a fraction of equity.

        Returns
        -------
        Decimal
            Dollar amount to allocate.
        """
        if win_rate <= 0.0 or win_rate >= 1.0:
            logger.warning("win_rate %.4f out of range; returning zero size.", win_rate)
            return Decimal("0")

        if win_loss_ratio <= 0.0:
            logger.warning("win_loss_ratio %.4f <= 0; returning zero size.", win_loss_ratio)
            return Decimal("0")

        # Kelly percentage: W - (1-W) / R
        kelly_pct = win_rate - ((1.0 - win_rate) / win_loss_ratio)

        # Apply fractional Kelly
        adjusted = kelly_pct * self.kelly_fraction

        # Cap at max_pct
        capped = min(adjusted, max_pct)

        # Floor at zero (negative Kelly means the edge is negative)
        allocation_pct = max(capped, 0.0)

        return equity * Decimal(str(allocation_pct))


class VolatilityTargetSizer(PositionSizer):
    """Size positions so each contributes a target amount of volatility.

    Parameters
    ----------
    target_annual_vol:
        Desired annualised volatility contribution per position (e.g. 0.02).
    trading_days:
        Number of trading days per year (used for annualisation).
    """

    def __init__(
        self,
        target_annual_vol: float = 0.02,
        trading_days: int = 252,
    ) -> None:
        if target_annual_vol <= 0.0:
            raise ValueError(f"target_annual_vol must be > 0, got {target_annual_vol}")
        self.target_annual_vol = target_annual_vol
        self.trading_days = trading_days

    def size(
        self,
        *,
        equity: Decimal,
        asset_daily_vol: float,
        max_pct: float = 0.10,
        **_kwargs,
    ) -> Decimal:
        """Return dollar size so position contributes *target_annual_vol*.

        Parameters
        ----------
        equity:
            Current account equity.
        asset_daily_vol:
            Annualised or daily volatility of the asset (daily std of returns).
            If zero or negative, returns zero.
        max_pct:
            Hard cap on allocation as a fraction of equity.

        Returns
        -------
        Decimal
            Dollar amount to allocate.
        """
        if asset_daily_vol <= 0.0:
            logger.warning("asset_daily_vol %.6f <= 0; returning zero size.", asset_daily_vol)
            return Decimal("0")

        # Annualise the daily vol
        annualised_vol = asset_daily_vol * np.sqrt(self.trading_days)

        if annualised_vol <= 0.0:
            return Decimal("0")

        # Weight = target_vol / asset_vol
        weight = self.target_annual_vol / annualised_vol

        # Cap
        weight = min(weight, max_pct)

        return equity * Decimal(str(weight))
