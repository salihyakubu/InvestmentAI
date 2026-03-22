"""Stop loss management for open positions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class StopLossRecord:
    """Internal record for a tracked stop loss."""

    position_id: str
    symbol: str
    entry_price: Decimal
    stop_price: Decimal


@dataclass
class TriggeredStop:
    """Returned when a stop loss is triggered."""

    position_id: str
    symbol: str
    trigger_price: Decimal
    reason: str


class StopLossManager:
    """Manages fixed stop-loss levels for open positions.

    Maintains a registry of (position_id -> stop record) and checks
    current market prices against each stop on every tick.
    """

    def __init__(self) -> None:
        self._stops: dict[str, StopLossRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_stop_loss(
        self,
        position_id: str,
        symbol: str,
        entry_price: Decimal,
        stop_price: Decimal,
    ) -> None:
        """Register a stop-loss level for *position_id*.

        If a stop already exists for this position it is overwritten.
        """
        record = StopLossRecord(
            position_id=position_id,
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
        )
        self._stops[position_id] = record
        logger.info(
            "Stop loss set for %s (%s): entry=%s stop=%s",
            position_id,
            symbol,
            entry_price,
            stop_price,
        )

    def check_stops(
        self, current_prices: dict[str, Decimal]
    ) -> list[TriggeredStop]:
        """Check all registered stops against *current_prices*.

        Returns a list of :class:`TriggeredStop` for every stop that has
        been breached.  Triggered stops are automatically removed from the
        internal registry.
        """
        triggered: list[TriggeredStop] = []
        to_remove: list[str] = []

        for position_id, record in self._stops.items():
            price = current_prices.get(record.symbol)
            if price is None:
                continue

            if price <= record.stop_price:
                triggered.append(
                    TriggeredStop(
                        position_id=position_id,
                        symbol=record.symbol,
                        trigger_price=price,
                        reason=(
                            f"Stop loss triggered: price {price} <= "
                            f"stop {record.stop_price}"
                        ),
                    )
                )
                to_remove.append(position_id)
                logger.warning(
                    "Stop loss triggered for %s (%s) at %s",
                    position_id,
                    record.symbol,
                    price,
                )

        for pid in to_remove:
            self._stops.pop(pid, None)

        return triggered

    def remove_stop(self, position_id: str) -> None:
        """Remove the stop for a closed position."""
        removed = self._stops.pop(position_id, None)
        if removed:
            logger.info("Removed stop loss for %s (%s)", position_id, removed.symbol)

    @property
    def active_stops(self) -> dict[str, StopLossRecord]:
        """Return a read-only snapshot of active stop records."""
        return dict(self._stops)
