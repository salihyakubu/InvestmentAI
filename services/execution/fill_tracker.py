"""Fill tracking, position reconciliation, and slippage analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionRecord:
    """Aggregated position built from fills."""

    symbol: str
    quantity: Decimal = Decimal("0")
    avg_price: Decimal = Decimal("0")
    total_cost: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")


@dataclass
class Discrepancy:
    """A mismatch between broker and local position records."""

    symbol: str
    broker_quantity: Decimal
    local_quantity: Decimal
    difference: Decimal


class FillTracker:
    """Tracks fills, builds positions, and reconciles with broker state."""

    def __init__(self) -> None:
        # symbol -> PositionRecord
        self._positions: dict[str, PositionRecord] = defaultdict(
            lambda: PositionRecord(symbol="")
        )
        # order_id -> list of fill dicts
        self._fills_by_order: dict[str, list[dict]] = defaultdict(list)

    def record_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> PositionRecord:
        """Record a fill and update the local position for *symbol*."""
        self._fills_by_order[order_id].append({
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "commission": commission,
        })

        pos = self._positions[symbol]
        pos.symbol = symbol

        signed_qty = quantity if side == "buy" else -quantity
        cost = price * quantity

        if side == "buy":
            pos.total_cost += cost
        else:
            pos.total_cost -= cost

        pos.quantity += signed_qty
        pos.total_commission += commission

        # Recompute average price.
        if pos.quantity != Decimal("0"):
            pos.avg_price = abs(pos.total_cost / pos.quantity)
        else:
            pos.avg_price = Decimal("0")
            pos.total_cost = Decimal("0")

        logger.info(
            "fill_tracked",
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=str(price),
            quantity=str(quantity),
            position_qty=str(pos.quantity),
        )
        return pos

    def reconcile(
        self,
        broker_positions: list[dict],
        db_positions: list[dict] | None = None,
    ) -> list[Discrepancy]:
        """Compare broker positions against local records and find mismatches.

        *broker_positions*: list of dicts with 'symbol' and 'quantity' keys.
        *db_positions*: optional list from database; falls back to local tracking.
        """
        local_map: dict[str, Decimal] = {}
        if db_positions is not None:
            for p in db_positions:
                local_map[p["symbol"]] = Decimal(str(p["quantity"]))
        else:
            for sym, pos in self._positions.items():
                if pos.quantity != Decimal("0"):
                    local_map[sym] = pos.quantity

        broker_map: dict[str, Decimal] = {}
        for p in broker_positions:
            broker_map[p["symbol"]] = Decimal(str(p["quantity"]))

        all_symbols = set(local_map.keys()) | set(broker_map.keys())
        discrepancies: list[Discrepancy] = []

        for symbol in all_symbols:
            broker_qty = broker_map.get(symbol, Decimal("0"))
            local_qty = local_map.get(symbol, Decimal("0"))
            if broker_qty != local_qty:
                disc = Discrepancy(
                    symbol=symbol,
                    broker_quantity=broker_qty,
                    local_quantity=local_qty,
                    difference=broker_qty - local_qty,
                )
                discrepancies.append(disc)
                logger.warning(
                    "position_discrepancy",
                    symbol=symbol,
                    broker_qty=str(broker_qty),
                    local_qty=str(local_qty),
                    difference=str(disc.difference),
                )

        if not discrepancies:
            logger.info("reconciliation_clean", symbols_checked=len(all_symbols))

        return discrepancies

    @staticmethod
    def compute_slippage(
        expected_price: Decimal,
        actual_price: Decimal,
    ) -> Decimal:
        """Compute realised slippage in basis points.

        Positive means worse-than-expected execution.
        """
        if expected_price <= 0:
            return Decimal("0")
        slippage_bps = abs(actual_price - expected_price) / expected_price * Decimal("10000")
        return slippage_bps.quantize(Decimal("0.01"))

    def get_position(self, symbol: str) -> PositionRecord | None:
        """Return the tracked position for *symbol*."""
        pos = self._positions.get(symbol)
        if pos and pos.quantity != Decimal("0"):
            return pos
        return None

    def get_all_positions(self) -> list[PositionRecord]:
        """Return all non-zero tracked positions."""
        return [p for p in self._positions.values() if p.quantity != Decimal("0")]
