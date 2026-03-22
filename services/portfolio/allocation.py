"""Target allocation and rebalancing logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from core.enums import OrderSide, OrderType

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Lightweight order descriptor emitted by the rebalancer."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    notional: float  # dollar amount to trade
    weight_delta: float  # signed change in portfolio weight


@dataclass
class TargetAllocation:
    """Maps symbols to their target portfolio weights."""

    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Drop zero-weight entries.
        self.weights = {s: w for s, w in self.weights.items() if w > 1e-9}

    @property
    def symbols(self) -> list[str]:
        return list(self.weights.keys())

    def total_weight(self) -> float:
        return sum(self.weights.values())


class Rebalancer:
    """Computes trades required to move from current positions to a target
    allocation, respecting turnover limits and minimum trade sizes.
    """

    def __init__(self, min_trade_notional: float = 1.0) -> None:
        """
        Args:
            min_trade_notional: Minimum dollar value for a trade to be
                emitted.  Defaults to $1 for micro-accounts.
        """
        self.min_trade_notional = min_trade_notional

    def compute_trades(
        self,
        current_positions: dict[str, float],
        target_allocation: TargetAllocation,
        total_equity: float,
    ) -> list[OrderRequest]:
        """Determine trades needed to reach *target_allocation*.

        Args:
            current_positions: symbol -> current dollar value.
            target_allocation: desired weights.
            total_equity: total portfolio value (cash + positions).

        Returns:
            List of :class:`OrderRequest` objects.
        """
        if total_equity <= 0:
            logger.warning("Total equity is zero or negative; no trades.")
            return []

        orders: list[OrderRequest] = []

        # Current weights.
        current_weights: dict[str, float] = {
            sym: val / total_equity for sym, val in current_positions.items()
        }

        # Union of all symbols involved.
        all_symbols = set(current_weights.keys()) | set(target_allocation.symbols)

        for symbol in sorted(all_symbols):
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_allocation.weights.get(symbol, 0.0)
            delta_w = target_w - current_w

            notional = abs(delta_w) * total_equity

            if notional < self.min_trade_notional:
                continue

            side = OrderSide.BUY if delta_w > 0 else OrderSide.SELL

            orders.append(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    notional=round(notional, 2),
                    weight_delta=round(delta_w, 6),
                )
            )

        return orders

    @staticmethod
    def apply_turnover_limit(
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        limit: float,
    ) -> dict[str, float]:
        """Scale target towards current so that total turnover stays
        within *limit* (expressed as fraction of portfolio).

        Turnover = 0.5 * sum(|target_i - current_i|).

        Args:
            current_weights: symbol -> current weight.
            target_weights: symbol -> desired weight.
            limit: maximum one-way turnover fraction (e.g. 0.20 = 20%).

        Returns:
            Adjusted target weights dict.
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        turnover = 0.5 * sum(
            abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            for s in all_symbols
        )

        if turnover <= limit or turnover < 1e-12:
            return dict(target_weights)

        # Blend: adjusted = current + scale_factor * (target - current)
        scale = limit / turnover
        adjusted: dict[str, float] = {}

        for s in all_symbols:
            cur = current_weights.get(s, 0.0)
            tgt = target_weights.get(s, 0.0)
            adjusted[s] = cur + scale * (tgt - cur)

        # Remove near-zero entries.
        adjusted = {s: w for s, w in adjusted.items() if w > 1e-9}

        return adjusted
