"""Paper trading broker that simulates realistic order fills in-memory."""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from services.execution.brokers.base import BaseBroker, BrokerFill, BrokerOrder

logger = structlog.get_logger(__name__)

# Slippage range in basis points for market orders.
_MIN_SLIPPAGE_BPS = Decimal("0")
_MAX_SLIPPAGE_BPS = Decimal("5")

# Simulated latency bounds (seconds).
_MIN_LATENCY = 0.010
_MAX_LATENCY = 0.050


class PaperBroker(BaseBroker):
    """In-memory broker that simulates fills with configurable slippage.

    Designed for paper trading and backtesting without hitting real venues.
    """

    name = "paper"
    asset_class = "all"
    supports_live = False

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self._cash = initial_cash
        self._initial_cash = initial_cash
        # symbol -> signed quantity (positive = long)
        self._positions: dict[str, Decimal] = defaultdict(Decimal)
        # external_id -> BrokerOrder
        self._orders: dict[str, BrokerOrder] = {}
        # external_id -> status string
        self._order_statuses: dict[str, str] = {}
        # All fills for audit trail
        self._fills: list[BrokerFill] = []
        # Pending limit / stop orders awaiting trigger
        self._pending_orders: dict[str, BrokerOrder] = {}
        # Last known prices per symbol (used to evaluate pending orders)
        self._last_prices: dict[str, Decimal] = {}

    # ------------------------------------------------------------------
    # Price feed (call externally to update simulated market)
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update last known price and check pending orders."""
        self._last_prices[symbol] = price
        self._check_pending_orders(symbol, price)

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order; market orders fill immediately."""
        await asyncio.sleep(random.uniform(_MIN_LATENCY, _MAX_LATENCY))

        external_id = order.external_id or str(uuid.uuid4())
        order.external_id = external_id
        self._orders[external_id] = order

        if order.order_type == "market":
            await self._fill_market_order(order)
        elif order.order_type == "limit":
            self._pending_orders[external_id] = order
            self._order_statuses[external_id] = "submitted"
            logger.info(
                "paper_limit_order_submitted",
                external_id=external_id,
                symbol=order.symbol,
                limit_price=str(order.limit_price),
            )
        elif order.order_type in ("stop", "stop_limit"):
            self._pending_orders[external_id] = order
            self._order_statuses[external_id] = "submitted"
            logger.info(
                "paper_stop_order_submitted",
                external_id=external_id,
                symbol=order.symbol,
                stop_price=str(order.stop_price),
            )
        else:
            self._order_statuses[external_id] = "rejected"
            logger.warning(
                "paper_unsupported_order_type",
                order_type=order.order_type,
            )

        return external_id

    async def cancel_order(self, external_id: str) -> bool:
        """Cancel a pending order."""
        if external_id in self._pending_orders:
            del self._pending_orders[external_id]
            self._order_statuses[external_id] = "cancelled"
            logger.info("paper_order_cancelled", external_id=external_id)
            return True
        return False

    async def get_order_status(self, external_id: str) -> dict:
        """Return the current status of an order."""
        status = self._order_statuses.get(external_id, "unknown")
        order = self._orders.get(external_id)
        return {
            "external_id": external_id,
            "status": status,
            "symbol": order.symbol if order else None,
            "side": order.side if order else None,
            "quantity": str(order.quantity) if order else None,
        }

    async def get_positions(self) -> list[dict]:
        """Return all non-zero positions."""
        positions = []
        for symbol, qty in self._positions.items():
            if qty != Decimal("0"):
                last_price = self._last_prices.get(symbol, Decimal("0"))
                positions.append({
                    "symbol": symbol,
                    "quantity": str(qty),
                    "market_value": str(qty * last_price),
                    "avg_entry_price": "0",  # simplified
                    "unrealized_pnl": "0",
                })
        return positions

    async def get_account(self) -> dict:
        """Return simulated account summary."""
        positions_value = sum(
            qty * self._last_prices.get(sym, Decimal("0"))
            for sym, qty in self._positions.items()
        )
        equity = self._cash + positions_value
        return {
            "equity": str(equity),
            "cash": str(self._cash),
            "positions_value": str(positions_value),
            "buying_power": str(self._cash),
            "initial_capital": str(self._initial_cash),
        }

    async def health_check(self) -> bool:
        """Paper broker is always healthy."""
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fill_market_order(self, order: BrokerOrder) -> None:
        """Fill a market order immediately with random slippage."""
        base_price = self._last_prices.get(order.symbol)
        if base_price is None:
            # No price available; use limit_price as fallback or reject.
            if order.limit_price is not None:
                base_price = order.limit_price
            else:
                self._order_statuses[order.external_id] = "rejected"
                logger.warning(
                    "paper_no_price_for_market_order",
                    symbol=order.symbol,
                )
                return

        slippage_bps = _MIN_SLIPPAGE_BPS + (
            _MAX_SLIPPAGE_BPS - _MIN_SLIPPAGE_BPS
        ) * Decimal(str(random.random()))

        if order.side == "buy":
            fill_price = base_price * (Decimal("1") + slippage_bps / Decimal("10000"))
        else:
            fill_price = base_price * (Decimal("1") - slippage_bps / Decimal("10000"))

        fill_price = fill_price.quantize(Decimal("0.01"))

        fill = BrokerFill(
            external_id=str(uuid.uuid4()),
            order_external_id=order.external_id,
            price=fill_price,
            quantity=order.quantity,
            commission=Decimal("0"),  # paper trading: no commissions
            filled_at=datetime.now(timezone.utc),
        )
        self._fills.append(fill)

        # Update positions and cash
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        self._positions[order.symbol] += signed_qty
        self._cash -= signed_qty * fill_price

        self._order_statuses[order.external_id] = "filled"
        logger.info(
            "paper_market_order_filled",
            external_id=order.external_id,
            symbol=order.symbol,
            side=order.side,
            fill_price=str(fill_price),
            quantity=str(order.quantity),
            slippage_bps=str(slippage_bps),
        )

    def _check_pending_orders(self, symbol: str, price: Decimal) -> None:
        """Check if any pending limit/stop orders should be triggered."""
        to_fill: list[str] = []

        for ext_id, order in self._pending_orders.items():
            if order.symbol != symbol:
                continue

            if order.order_type == "limit":
                # Buy limit fills at or below limit price
                if order.side == "buy" and order.limit_price is not None and price <= order.limit_price:
                    to_fill.append(ext_id)
                # Sell limit fills at or above limit price
                elif order.side == "sell" and order.limit_price is not None and price >= order.limit_price:
                    to_fill.append(ext_id)

            elif order.order_type in ("stop", "stop_limit"):
                # Buy stop triggers when price rises to stop price
                if order.side == "buy" and order.stop_price is not None and price >= order.stop_price:
                    to_fill.append(ext_id)
                # Sell stop triggers when price falls to stop price
                elif order.side == "sell" and order.stop_price is not None and price <= order.stop_price:
                    to_fill.append(ext_id)

        for ext_id in to_fill:
            order = self._pending_orders.pop(ext_id)
            fill_price = price.quantize(Decimal("0.01"))
            fill = BrokerFill(
                external_id=str(uuid.uuid4()),
                order_external_id=ext_id,
                price=fill_price,
                quantity=order.quantity,
                commission=Decimal("0"),
                filled_at=datetime.now(timezone.utc),
            )
            self._fills.append(fill)

            signed_qty = order.quantity if order.side == "buy" else -order.quantity
            self._positions[order.symbol] += signed_qty
            self._cash -= signed_qty * fill_price
            self._order_statuses[ext_id] = "filled"

            logger.info(
                "paper_pending_order_filled",
                external_id=ext_id,
                order_type=order.order_type,
                symbol=order.symbol,
                fill_price=str(fill_price),
            )

    # ------------------------------------------------------------------
    # Audit helpers
    # ------------------------------------------------------------------

    @property
    def fills(self) -> list[BrokerFill]:
        """Return all recorded fills."""
        return list(self._fills)

    @property
    def open_orders(self) -> dict[str, BrokerOrder]:
        """Return pending orders."""
        return dict(self._pending_orders)
