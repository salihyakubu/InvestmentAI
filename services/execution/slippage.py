"""Slippage estimation and adjustment utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SlippageEstimate:
    """Estimated slippage for a proposed trade."""

    expected_bps: Decimal
    cost: Decimal


class SlippageEstimator:
    """Estimates market-impact slippage using a square-root model.

    The model is:
        impact_bps = spread_bps / 2 + 10 * sqrt(participation_rate)

    where participation_rate = quantity / avg_daily_volume.
    """

    def estimate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        avg_daily_volume: Decimal,
        spread_bps: Decimal,
        price: Decimal | None = None,
    ) -> SlippageEstimate:
        """Estimate expected slippage in basis points and dollar cost."""
        if avg_daily_volume <= 0:
            participation_rate = Decimal("0")
        else:
            participation_rate = quantity / avg_daily_volume

        sqrt_participation = Decimal(str(math.sqrt(float(participation_rate))))
        impact_bps = spread_bps / Decimal("2") + Decimal("10") * sqrt_participation

        # Compute dollar cost if price is available.
        if price is not None and price > 0:
            impact_fraction = impact_bps / Decimal("10000")
            cost = impact_fraction * price * quantity
        else:
            cost = Decimal("0")

        logger.debug(
            "slippage_estimate",
            symbol=symbol,
            side=side,
            participation_rate=str(participation_rate),
            impact_bps=str(impact_bps),
            cost=str(cost),
        )

        return SlippageEstimate(expected_bps=impact_bps, cost=cost)

    @staticmethod
    def apply_slippage(
        price: Decimal,
        side: str,
        slippage_bps: Decimal,
    ) -> Decimal:
        """Adjust *price* for expected slippage.

        Buys are adjusted upward; sells are adjusted downward.
        """
        adjustment = price * slippage_bps / Decimal("10000")
        if side == "buy":
            return price + adjustment
        else:
            return price - adjustment
