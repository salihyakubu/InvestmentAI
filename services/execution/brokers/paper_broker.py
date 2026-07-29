"""Paper trading broker that simulates realistic order fills in-memory."""

from __future__ import annotations

import asyncio
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog

from services.execution.brokers.base import BaseBroker, BrokerFill, BrokerOrder

logger = structlog.get_logger(__name__)

# Slippage range in basis points for market orders.
_MIN_SLIPPAGE_BPS = Decimal("0")
_MAX_SLIPPAGE_BPS = Decimal("5")

# Simulated latency bounds (seconds).
_MIN_LATENCY = 0.010
_MAX_LATENCY = 0.050

# Terminal orders and their fills older than this are dropped from memory;
# without a cap the broker's order/fill maps grow for the process lifetime.
_TERMINAL_RETENTION = timedelta(hours=48)


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
        # symbol -> volume-weighted cost basis of the open position; without it
        # positions report a zero entry price and P&L cannot be attributed.
        self._avg_entry: dict[str, Decimal] = {}
        # Cumulative realised P&L from closed quantity, net of commission.
        self._realized_pnl = Decimal("0")
        # external_id -> BrokerOrder
        self._orders: dict[str, BrokerOrder] = {}
        # external_id -> status string
        self._order_statuses: dict[str, str] = {}
        # order external_id -> fills, so get_order_status stays O(fills for
        # that order) instead of scanning every fill ever recorded
        self._fills_by_order: dict[str, list[BrokerFill]] = {}
        # external_id -> when the order reached a terminal status (prune key)
        self._terminal_at: dict[str, datetime] = {}
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
        # Opportunistic prune: a broker object must not run background tasks,
        # so expired terminal orders are dropped on the next submission.
        self._prune()
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
            self._mark_terminal(external_id, "rejected")
            logger.warning(
                "paper_unsupported_order_type",
                order_type=order.order_type,
            )

        return external_id

    async def cancel_order(self, external_id: str) -> bool:
        """Cancel a pending order."""
        if external_id in self._pending_orders:
            del self._pending_orders[external_id]
            self._mark_terminal(external_id, "cancelled")
            logger.info("paper_order_cancelled", external_id=external_id)
            return True
        return False

    async def get_order_status(self, external_id: str) -> dict[str, Any]:
        """Return the current status of an order, including fill details.

        ``filled_qty`` / ``filled_avg_price`` are what the execution engine's
        monitor reads to record a fill, so they must be reported here.
        """
        status = self._order_statuses.get(external_id, "unknown")
        order = self._orders.get(external_id)

        order_fills = self._fills_by_order.get(external_id, [])
        filled_qty = sum((f.quantity for f in order_fills), Decimal("0"))
        if filled_qty > 0:
            filled_avg_price = (
                sum((f.price * f.quantity for f in order_fills), Decimal("0")) / filled_qty
            )
        else:
            filled_avg_price = Decimal("0")

        return {
            "external_id": external_id,
            "status": status,
            "symbol": order.symbol if order else None,
            "side": order.side if order else None,
            "quantity": str(order.quantity) if order else None,
            "filled_qty": str(filled_qty),
            "filled_avg_price": str(filled_avg_price),
        }

    async def get_positions(self) -> list[dict[str, Any]]:
        """Return all non-zero positions, marked at the last seen price."""
        positions = []
        for symbol, qty in self._positions.items():
            if qty != Decimal("0"):
                avg_entry = self._avg_entry.get(symbol, Decimal("0"))
                last_price = self._last_prices.get(symbol) or avg_entry
                positions.append({
                    "symbol": symbol,
                    "quantity": str(qty),
                    "market_value": str(qty * last_price),
                    "avg_entry_price": str(avg_entry),
                    "current_price": str(last_price),
                    "unrealized_pnl": str(qty * (last_price - avg_entry)),
                })
        return positions

    async def get_account(self) -> dict[str, Any]:
        """Return simulated account summary."""
        positions_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        for sym, qty in self._positions.items():
            if qty == Decimal("0"):
                continue
            avg_entry = self._avg_entry.get(sym, Decimal("0"))
            # Fall back to cost basis rather than zero: an unpriced symbol
            # would otherwise read as a total loss of the position's value.
            last_price = self._last_prices.get(sym) or avg_entry
            positions_value += qty * last_price
            unrealized_pnl += qty * (last_price - avg_entry)
        equity = self._cash + positions_value
        return {
            "equity": str(equity),
            "cash": str(self._cash),
            "positions_value": str(positions_value),
            "unrealized_pnl": str(unrealized_pnl),
            "realized_pnl": str(self._realized_pnl),
            "buying_power": str(self._cash),
            "initial_capital": str(self._initial_cash),
        }

    async def health_check(self) -> bool:
        """Paper broker is always healthy."""
        return True

    # ------------------------------------------------------------------
    # Durable state
    # ------------------------------------------------------------------

    def restore_state(
        self,
        *,
        cash: Decimal,
        positions: dict[str, Decimal] | None = None,
        avg_entry: dict[str, Decimal] | None = None,
        last_prices: dict[str, Decimal] | None = None,
        realized_pnl: Decimal | None = None,
    ) -> None:
        """Rebuild account state from a durable checkpoint.

        Account state lives in this process, so without a restore every
        restart silently rebases equity to ``initial_cash`` and orphans the
        open book.

        *avg_entry* and *last_prices* must both be seeded alongside
        *positions*. Cost basis is what makes P&L attributable: restored
        without it, every position reports its whole market value as
        unrealised profit. And ``get_account`` marks at the last seen price,
        so an empty price map would report a position-less equity until the
        next tick and register as a false drawdown.
        """
        self._cash = cash
        self._positions = defaultdict(Decimal)
        self._avg_entry = {}
        for symbol, quantity in (positions or {}).items():
            if quantity != Decimal("0"):
                self._positions[symbol] = quantity
        for symbol, entry in (avg_entry or {}).items():
            if entry:
                self._avg_entry[symbol] = entry
        for symbol, price in (last_prices or {}).items():
            self._last_prices.setdefault(symbol, price)
        if realized_pnl is not None:
            self._realized_pnl = realized_pnl
        logger.info(
            "paper_state_restored",
            cash=str(cash),
            positions=len(self._positions),
        )

    def apply_external_fill(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> None:
        """Replay a persisted fill onto restored state (checkpoint catch-up)."""
        self._apply_to_book(symbol, side, price, quantity, Decimal("0"))

    def _apply_to_book(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
        commission: Decimal,
    ) -> None:
        """Apply one fill to cash, position, cost basis and realised P&L.

        Quantity closed against an opposing position realises P&L at the
        difference to the cost basis; quantity added in the same direction
        re-averages it. A fill that flips the sign does both.
        """
        signed_qty = quantity if side == "buy" else -quantity
        old_qty = self._positions[symbol]
        old_avg = self._avg_entry.get(symbol, Decimal("0"))
        new_qty = old_qty + signed_qty

        if old_qty == Decimal("0") or (old_qty > 0) == (signed_qty > 0):
            total = abs(old_qty) + abs(signed_qty)
            new_avg = (
                (abs(old_qty) * old_avg + abs(signed_qty) * price) / total
                if total
                else Decimal("0")
            )
        else:
            closed = min(abs(old_qty), abs(signed_qty))
            direction = Decimal("1") if old_qty > 0 else Decimal("-1")
            self._realized_pnl += direction * closed * (price - old_avg)
            # A flip leaves a fresh position opened at this fill's price.
            new_avg = price if new_qty != 0 and (old_qty > 0) != (new_qty > 0) else old_avg

        self._positions[symbol] = new_qty
        if new_qty == Decimal("0"):
            self._avg_entry.pop(symbol, None)
        else:
            self._avg_entry[symbol] = new_avg

        self._cash -= signed_qty * price
        self._cash -= commission
        self._realized_pnl -= commission

    # ------------------------------------------------------------------
    # Memory retention
    # ------------------------------------------------------------------

    def _mark_terminal(self, external_id: str, status: str) -> None:
        """Set a terminal status and stamp it for later pruning."""
        self._order_statuses[external_id] = status
        self._terminal_at[external_id] = datetime.now(UTC)

    def _prune(self) -> None:
        """Drop terminal orders (and their fills) past the retention window.

        Account state (_cash/_positions/_last_prices) already reflects every
        fill, so removing fill records cannot change balances or positions.
        Open and pending orders are never pruned.
        """
        cutoff = datetime.now(UTC) - _TERMINAL_RETENTION
        expired = [
            ext_id for ext_id, at in self._terminal_at.items() if at < cutoff
        ]
        for ext_id in expired:
            del self._terminal_at[ext_id]
            self._orders.pop(ext_id, None)
            self._order_statuses.pop(ext_id, None)
            self._fills_by_order.pop(ext_id, None)
        if expired:
            logger.info("paper_terminal_orders_pruned", count=len(expired))

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
                self._mark_terminal(order.external_id, "rejected")
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

        # Buying-power backstop: a buy cannot spend more cash than is available.
        if order.side == "buy":
            cost = order.quantity * fill_price
            if cost > self._cash:
                self._mark_terminal(order.external_id, "rejected")
                logger.warning(
                    "paper_insufficient_cash",
                    symbol=order.symbol,
                    cost=str(cost),
                    cash=str(self._cash),
                )
                return

        fill = BrokerFill(
            external_id=str(uuid.uuid4()),
            order_external_id=order.external_id,
            price=fill_price,
            quantity=order.quantity,
            commission=Decimal("0"),  # paper trading: no commissions
            filled_at=datetime.now(UTC),
        )
        self._fills_by_order.setdefault(order.external_id, []).append(fill)

        self._apply_to_book(
            order.symbol, order.side, fill_price, order.quantity, fill.commission
        )

        self._mark_terminal(order.external_id, "filled")
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
                filled_at=datetime.now(UTC),
            )
            self._fills_by_order.setdefault(ext_id, []).append(fill)

            self._apply_to_book(
                order.symbol, order.side, fill_price, order.quantity, fill.commission
            )
            self._mark_terminal(ext_id, "filled")

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
        """Return all retained fills in chronological order."""
        all_fills = [f for fills in self._fills_by_order.values() for f in fills]
        all_fills.sort(key=lambda f: f.filled_at)
        return all_fills

    @property
    def open_orders(self) -> dict[str, BrokerOrder]:
        """Return pending orders."""
        return dict(self._pending_orders)
