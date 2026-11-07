"""Main liquidation manager service coordinating all stop/liquidation logic."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from config.settings import Settings
from core.enums import AssetClass, OrderSide
from core.events.base import Event, EventBus
from core.events.market_events import PriceUpdateEvent
from core.events.order_events import TradingControlEvent
from core.events.risk_events import RiskBreachedEvent
from core.events.streams import CONTROL, LIQUIDATION, ORDERS, RISK_BREACHED
from core.events.system_events import LiquidationTriggeredEvent
from services.liquidation.emergency import EmergencyLiquidator, PositionSnapshot
from services.liquidation.stop_loss import StopLossManager
from services.liquidation.trailing_stop import TrailingStopManager

logger = logging.getLogger(__name__)

# Stream names used by the event bus.
STREAM_MARKET = "market.prices"
# Imported, not retyped: this constant read "risk.breaches" while the risk
# service published on "risk.breached", so no breach ever arrived here
# (found by the 2026-11-02 fleet review).
STREAM_RISK = RISK_BREACHED
STREAM_LIQUIDATION = LIQUIDATION
# Fills: a buy registers the position's stop, a flattening sell removes it.
STREAM_ORDERS = ORDERS

PositionProvider = Callable[[], Awaitable[list[dict[str, Any]]]]


class LiquidationManagerService:
    """Orchestrates stop-loss, trailing-stop, and emergency liquidation.

    Subscribes to :class:`PriceUpdateEvent` and :class:`RiskBreachedEvent`
    from the event bus, evaluates all registered stops on each tick, and
    publishes :class:`LiquidationTriggeredEvent` when a position must be
    closed.

    Parameters
    ----------
    event_bus:
        The platform event bus (Redis-backed or in-process).
    settings:
        Application settings (used for risk thresholds).
    execution_service_ref:
        An optional reference to the execution service for forwarding
        liquidation orders.  Kept as ``Any`` to avoid circular imports.
    position_provider:
        Async callable returning the broker's open positions (the shape of
        ``BaseBroker.get_positions``). With it, a fill is judged against the
        venue's actual book: a buy registers the stop at the position's
        cost basis and a sell that flattens the symbol removes it. Without
        it the fill's own price and quantity are used and a sell never
        removes a stop (the execution engine resolves a stale trigger
        against the real book and finds nothing to close).
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        execution_service_ref: Any = None,
        position_provider: PositionProvider | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._execution_service = execution_service_ref
        self._position_provider = position_provider

        self.stop_loss_manager = StopLossManager()
        self.trailing_stop_manager = TrailingStopManager()
        self.emergency_liquidator = EmergencyLiquidator()

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant event streams and begin processing."""
        if self._running:
            return

        self._running = True
        logger.info("LiquidationManagerService starting")

        # Register handlers -- for InProcessEventBus these return
        # immediately; for Redis EventBus they would be run as tasks.
        await self._event_bus.subscribe(
            stream=STREAM_MARKET,
            group="liquidation-svc",
            consumer="liquidation-1",
            handler=self.handle_price_update,
        )
        await self._event_bus.subscribe(
            stream=STREAM_RISK,
            group="liquidation-svc",
            consumer="liquidation-1",
            handler=self.handle_risk_breach,
        )
        await self._event_bus.subscribe(
            stream=STREAM_ORDERS,
            group="liquidation-svc",
            consumer="liquidation-1",
            handler=self.handle_order_fill,
        )
        logger.info("LiquidationManagerService subscribed to events")

    async def stop(self) -> None:
        """Mark the service as stopped."""
        self._running = False
        logger.info("LiquidationManagerService stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def handle_price_update(self, event: Event) -> None:
        """Handle a :class:`PriceUpdateEvent` -- check all stops."""
        if not isinstance(event, PriceUpdateEvent):
            # Deserialise from generic Event if needed.
            symbol = event.payload.get("symbol", "")
            price = Decimal(str(event.payload.get("price", 0)))
        else:
            symbol = event.symbol
            price = Decimal(str(event.price))

        if not symbol:
            return

        current_prices: dict[str, Decimal] = {symbol: price}

        # 1. Fixed stop losses
        for stop in self.stop_loss_manager.check_stops(current_prices):
            await self._trigger_liquidation(
                position_id=stop.position_id,
                symbol=stop.symbol,
                reason=stop.reason,
                quantity=0.0,  # full close implied
            )
            self.emergency_liquidator.unregister_position(stop.position_id)

        # 2. Trailing stops
        for tstop in self.trailing_stop_manager.update_prices(current_prices):
            await self._trigger_liquidation(
                position_id=tstop.position_id,
                symbol=tstop.symbol,
                reason=tstop.reason,
                quantity=0.0,
            )
            self.emergency_liquidator.unregister_position(tstop.position_id)

    async def handle_order_fill(self, event: Event) -> None:
        """Own every position the moment it exists.

        A buy fill registers (or refreshes) the symbol's stop at the
        position's cost basis; a sell that leaves nothing held removes it.
        Until 2026-11-07 nothing called ``set_stop_loss`` anywhere, so no
        position ever had an exit of its own.
        """
        if event.event_type != "OrderFilledEvent":
            return
        symbol = str(getattr(event, "symbol", "") or event.payload.get("symbol", ""))
        side = str(getattr(event, "side", "") or event.payload.get("side", "")).lower()
        price_raw = getattr(event, "fill_price", None) or event.payload.get("fill_price")
        qty_raw = getattr(event, "fill_quantity", None) or event.payload.get("fill_quantity")
        if not symbol or price_raw is None:
            return
        try:
            fill_price = Decimal(str(price_raw))
            fill_qty = Decimal(str(qty_raw or "0"))
        except InvalidOperation:
            return

        held = await self._held_position(symbol)
        held_qty = held["quantity"] if held is not None else None

        if side == "buy":
            if held is not None and held["quantity"] > 0:
                entry = held["entry"] if held["entry"] > 0 else fill_price
                self._register_stop(symbol, entry=entry, quantity=held["quantity"])
            elif held is not None and held["quantity"] < 0:
                return  # the buy covered part of a short: nothing long to protect
            else:
                # No provider, or a provider that has not yet seen this fill
                # (a venue balance can lag): protect the position from the
                # fill itself. A stop on a symbol that turns out flat is
                # harmless; a new long with no stop is not (risk sign-off
                # 2026-11-07, finding 2).
                self._register_stop(symbol, entry=fill_price, quantity=fill_qty)
            return

        if held_qty is not None and held_qty <= 0:
            self._forget(symbol)
        # A partial sell keeps the stop; so does a sell with no provider.

    async def seed_stops(self, positions: list[dict[str, Any]]) -> None:
        """Register a stop for every long found at boot.

        A restored position is otherwise ownerless: no fill will ever arrive
        for it. Entry is the broker's cost basis when it carries one, else
        the current mark. A restored short (a paper artifact: spot cannot
        short) gets no stop; it is closed through the held-book exit path.
        """
        for position in positions:
            symbol = str(position.get("symbol", ""))
            try:
                quantity = Decimal(str(position.get("quantity", "0")))
                entry = Decimal(str(position.get("avg_entry_price", "0") or "0"))
                mark = Decimal(str(position.get("current_price", "0") or "0"))
            except InvalidOperation:
                continue
            if not symbol or quantity == 0:
                continue
            if quantity < 0:
                logger.warning(
                    "Restored short %s (%s) has no stop; it exits via the held book",
                    symbol,
                    quantity,
                )
                continue
            self._register_stop(
                symbol, entry=entry if entry > 0 else mark, quantity=quantity
            )

    # ------------------------------------------------------------------
    # Stop registry helpers
    # ------------------------------------------------------------------

    def _register_stop(self, symbol: str, *, entry: Decimal, quantity: Decimal) -> None:
        pct = Decimal(str(self._settings.stop_loss_pct))
        if pct <= 0:
            logger.info("Stop registration disabled (stop_loss_pct=0): %s has no stop", symbol)
            return
        if entry <= 0:
            # A venue adapter that reports no cost basis and no mark leaves
            # the position without a stop; say so loudly rather than silently
            # (risk sign-off 2026-11-07, finding 3).
            logger.warning(
                "No stop registered for %s: no positive entry price available", symbol
            )
            return
        stop_price = entry * (Decimal("1") - pct)
        self.stop_loss_manager.set_stop_loss(
            position_id=symbol, symbol=symbol, entry_price=entry, stop_price=stop_price
        )
        self.emergency_liquidator.register_position(
            PositionSnapshot(
                position_id=symbol,
                symbol=symbol,
                asset_class=AssetClass.CRYPTO if "/" in symbol else AssetClass.STOCK,
                side=OrderSide.BUY,
                quantity=abs(quantity),
            )
        )

    def _forget(self, symbol: str) -> None:
        self.stop_loss_manager.remove_stop(symbol)
        self.trailing_stop_manager.remove_stop(symbol)
        self.emergency_liquidator.unregister_position(symbol)

    async def _held_position(self, symbol: str) -> dict[str, Decimal] | None:
        """The broker's signed quantity and cost basis for *symbol*, or None
        when no provider is wired or it fails (never raises into a handler)."""
        if self._position_provider is None:
            return None
        try:
            positions = await self._position_provider()
        except Exception:
            logger.exception("liquidation.position_provider_failed")
            return None
        quantity = Decimal("0")
        entry = Decimal("0")
        for position in positions:
            if str(position.get("symbol", "")) != symbol:
                continue
            try:
                quantity += Decimal(str(position.get("quantity", "0")))
                entry = Decimal(str(position.get("avg_entry_price", "0") or "0"))
            except InvalidOperation:
                continue
        return {"quantity": quantity, "entry": entry}

    async def handle_risk_breach(self, event: Event) -> None:
        """Handle risk-stream events: breaches and circuit-breaker transitions.

        A breaker trip liquidates only when ``breaker_auto_flatten`` is set;
        it then reuses the kill switch's flatten control (cancel, close every
        position through the tracked order path, halt) rather than a private
        liquidation path, so the record of the exit is the same one the
        operator's flatten would leave.
        """
        if event.event_type == "CircuitBreakerEvent":
            await self._handle_breaker_transition(event)
            return
        if isinstance(event, RiskBreachedEvent):
            rule_name = event.rule_name
            action = event.action
            current_value = event.current_value
            limit_value = event.limit_value
        else:
            rule_name = event.payload.get("rule_name", "unknown")
            action = event.payload.get("action", "warn")
            current_value = event.payload.get("current_value", 0)
            limit_value = event.payload.get("limit_value", 0)

        logger.warning(
            "Risk breach received: rule=%s action=%s value=%s limit=%s",
            rule_name,
            action,
            current_value,
            limit_value,
        )

        if action in ("reduce", "liquidate"):
            reason = (
                f"Risk breach [{rule_name}]: "
                f"value {current_value} > limit {limit_value}"
            )
            orders = self.emergency_liquidator.liquidate_all(reason)
            for order in orders:
                await self._trigger_liquidation(
                    position_id=order.position_id,
                    symbol=order.symbol,
                    reason=order.reason,
                    quantity=float(order.quantity),
                )

    async def _handle_breaker_transition(self, event: Event) -> None:
        state = str(getattr(event, "state", "") or event.payload.get("state", ""))
        reason = str(getattr(event, "reason", "") or event.payload.get("reason", ""))
        if state != "open":
            return
        if not self._settings.breaker_auto_flatten:
            logger.warning(
                "Circuit breaker OPEN (%s); breaker_auto_flatten is off, book left open",
                reason,
            )
            return
        await self._event_bus.publish(
            CONTROL,
            TradingControlEvent(
                action="flatten",
                reason=f"circuit breaker open: {reason}",
                source_service="LiquidationManagerService",
            ),
        )
        logger.critical("Circuit breaker OPEN -> flatten control published (%s)", reason)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _trigger_liquidation(
        self,
        position_id: str,
        symbol: str,
        reason: str,
        quantity: float,
    ) -> None:
        """Publish a :class:`LiquidationTriggeredEvent` on the event bus."""
        event = LiquidationTriggeredEvent(
            position_id=position_id,
            symbol=symbol,
            reason=reason,
            quantity=quantity,
            source_service="LiquidationManagerService",
        )
        await self._event_bus.publish(STREAM_LIQUIDATION, event)
        logger.warning(
            "Liquidation triggered: position=%s symbol=%s reason=%s",
            position_id,
            symbol,
            reason,
        )
