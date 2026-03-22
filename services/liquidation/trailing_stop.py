"""Trailing stop management for open positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from core.enums import OrderSide

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopRecord:
    """Internal record for a tracked trailing stop."""

    position_id: str
    symbol: str
    side: OrderSide  # LONG or SHORT (using BUY/SELL)
    trailing_pct: Decimal
    extreme_price: Decimal  # highest for long, lowest for short


@dataclass
class TriggeredTrailingStop:
    """Returned when a trailing stop is triggered."""

    position_id: str
    symbol: str
    trigger_price: Decimal
    stop_level: Decimal
    reason: str


class TrailingStopManager:
    """Manages trailing stop-loss levels that ratchet with price movement.

    For **long** positions the stop trails below the highest observed price::

        stop = highest_price * (1 - trailing_pct)

    For **short** positions the stop trails above the lowest observed price::

        stop = lowest_price * (1 + trailing_pct)
    """

    def __init__(self) -> None:
        self._stops: dict[str, TrailingStopRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_trailing_stop(
        self,
        position_id: str,
        symbol: str,
        side: OrderSide,
        trailing_pct: Decimal,
        highest_price: Decimal,
    ) -> None:
        """Register a trailing stop for *position_id*.

        *highest_price* seeds the extreme price tracker.  For short
        positions this is interpreted as the initial lowest price.
        """
        record = TrailingStopRecord(
            position_id=position_id,
            symbol=symbol,
            side=side,
            trailing_pct=trailing_pct,
            extreme_price=highest_price,
        )
        self._stops[position_id] = record
        logger.info(
            "Trailing stop set for %s (%s): side=%s pct=%s extreme=%s",
            position_id,
            symbol,
            side,
            trailing_pct,
            highest_price,
        )

    def update_prices(
        self, current_prices: dict[str, Decimal]
    ) -> list[TriggeredTrailingStop]:
        """Update extreme prices and check for triggered trailing stops.

        Returns a list of :class:`TriggeredTrailingStop` for every stop
        that has been breached.  Triggered stops are automatically removed.
        """
        triggered: list[TriggeredTrailingStop] = []
        to_remove: list[str] = []

        for position_id, record in self._stops.items():
            price = current_prices.get(record.symbol)
            if price is None:
                continue

            if record.side == OrderSide.BUY:
                # Long position: track highest price
                if price > record.extreme_price:
                    record.extreme_price = price

                stop_level = record.extreme_price * (
                    Decimal("1") - record.trailing_pct
                )

                if price <= stop_level:
                    triggered.append(
                        TriggeredTrailingStop(
                            position_id=position_id,
                            symbol=record.symbol,
                            trigger_price=price,
                            stop_level=stop_level,
                            reason=(
                                f"Trailing stop (long) triggered: price {price} "
                                f"<= stop {stop_level} "
                                f"(high={record.extreme_price}, "
                                f"trail={record.trailing_pct})"
                            ),
                        )
                    )
                    to_remove.append(position_id)

            else:
                # Short position: track lowest price
                if price < record.extreme_price:
                    record.extreme_price = price

                stop_level = record.extreme_price * (
                    Decimal("1") + record.trailing_pct
                )

                if price >= stop_level:
                    triggered.append(
                        TriggeredTrailingStop(
                            position_id=position_id,
                            symbol=record.symbol,
                            trigger_price=price,
                            stop_level=stop_level,
                            reason=(
                                f"Trailing stop (short) triggered: price {price} "
                                f">= stop {stop_level} "
                                f"(low={record.extreme_price}, "
                                f"trail={record.trailing_pct})"
                            ),
                        )
                    )
                    to_remove.append(position_id)

        for pid in to_remove:
            self._stops.pop(pid, None)
            logger.warning("Trailing stop removed after trigger: %s", pid)

        return triggered

    def remove_stop(self, position_id: str) -> None:
        """Remove trailing stop for a closed position."""
        self._stops.pop(position_id, None)

    @property
    def active_stops(self) -> dict[str, TrailingStopRecord]:
        """Return a snapshot of active trailing stop records."""
        return dict(self._stops)
