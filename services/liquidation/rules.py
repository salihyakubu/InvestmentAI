"""Liquidation rule definitions using the Strategy pattern."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Minimal position data consumed by liquidation rules."""

    position_id: str
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: Decimal
    quantity: Decimal
    stop_price: Decimal | None = None
    trailing_pct: Decimal | None = None
    highest_price: Decimal | None = None
    entry_time: datetime | None = None
    max_loss_usd: Decimal | None = None
    max_hold_seconds: int | None = None


@dataclass
class LiquidationDecision:
    """Result of a rule evaluation."""

    should_liquidate: bool
    rule_name: str
    reason: str
    position_id: str
    symbol: str


class LiquidationRule(ABC):
    """Abstract base for all liquidation rules."""

    @abstractmethod
    def check(
        self, position: PositionInfo, current_price: Decimal
    ) -> LiquidationDecision:
        """Evaluate the rule against *position* at *current_price*.

        Returns a :class:`LiquidationDecision` indicating whether
        liquidation should be triggered.
        """


class StopLossRule(LiquidationRule):
    """Triggers when the current price falls below the fixed stop price."""

    def check(
        self, position: PositionInfo, current_price: Decimal
    ) -> LiquidationDecision:
        if position.stop_price is None:
            return LiquidationDecision(
                should_liquidate=False,
                rule_name="StopLossRule",
                reason="No stop price set",
                position_id=position.position_id,
                symbol=position.symbol,
            )

        if position.side == "buy":
            triggered = current_price <= position.stop_price
        else:
            triggered = current_price >= position.stop_price

        reason = (
            f"Price {current_price} breached stop {position.stop_price}"
            if triggered
            else f"Price {current_price} within stop {position.stop_price}"
        )
        return LiquidationDecision(
            should_liquidate=triggered,
            rule_name="StopLossRule",
            reason=reason,
            position_id=position.position_id,
            symbol=position.symbol,
        )


class TrailingStopRule(LiquidationRule):
    """Triggers when price retraces beyond trailing_pct from the extreme."""

    def check(
        self, position: PositionInfo, current_price: Decimal
    ) -> LiquidationDecision:
        if position.trailing_pct is None or position.highest_price is None:
            return LiquidationDecision(
                should_liquidate=False,
                rule_name="TrailingStopRule",
                reason="Trailing stop not configured",
                position_id=position.position_id,
                symbol=position.symbol,
            )

        if position.side == "buy":
            stop_level = position.highest_price * (
                Decimal("1") - position.trailing_pct
            )
            triggered = current_price <= stop_level
        else:
            stop_level = position.highest_price * (
                Decimal("1") + position.trailing_pct
            )
            triggered = current_price >= stop_level

        reason = (
            f"Trailing stop triggered at {current_price} "
            f"(stop_level={stop_level}, extreme={position.highest_price})"
            if triggered
            else f"Price {current_price} within trailing range"
        )
        return LiquidationDecision(
            should_liquidate=triggered,
            rule_name="TrailingStopRule",
            reason=reason,
            position_id=position.position_id,
            symbol=position.symbol,
        )


class MaxLossRule(LiquidationRule):
    """Triggers when the absolute dollar loss exceeds a limit."""

    def check(
        self, position: PositionInfo, current_price: Decimal
    ) -> LiquidationDecision:
        if position.max_loss_usd is None:
            return LiquidationDecision(
                should_liquidate=False,
                rule_name="MaxLossRule",
                reason="No max loss limit set",
                position_id=position.position_id,
                symbol=position.symbol,
            )

        if position.side == "buy":
            pnl = (current_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - current_price) * position.quantity

        loss = -pnl if pnl < Decimal("0") else Decimal("0")
        triggered = loss >= position.max_loss_usd

        reason = (
            f"Loss ${loss} exceeds max ${position.max_loss_usd}"
            if triggered
            else f"Loss ${loss} within limit ${position.max_loss_usd}"
        )
        return LiquidationDecision(
            should_liquidate=triggered,
            rule_name="MaxLossRule",
            reason=reason,
            position_id=position.position_id,
            symbol=position.symbol,
        )


class TimeBasedRule(LiquidationRule):
    """Triggers when a position has been held longer than max_hold_seconds."""

    def check(
        self, position: PositionInfo, current_price: Decimal
    ) -> LiquidationDecision:
        if position.entry_time is None or position.max_hold_seconds is None:
            return LiquidationDecision(
                should_liquidate=False,
                rule_name="TimeBasedRule",
                reason="No time limit configured",
                position_id=position.position_id,
                symbol=position.symbol,
            )

        now = datetime.now(timezone.utc)
        held_seconds = (now - position.entry_time).total_seconds()
        triggered = held_seconds >= position.max_hold_seconds

        reason = (
            f"Held {held_seconds:.0f}s exceeds max {position.max_hold_seconds}s"
            if triggered
            else f"Held {held_seconds:.0f}s within {position.max_hold_seconds}s limit"
        )
        return LiquidationDecision(
            should_liquidate=triggered,
            rule_name="TimeBasedRule",
            reason=reason,
            position_id=position.position_id,
            symbol=position.symbol,
        )
