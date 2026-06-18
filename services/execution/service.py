"""Main execution engine service that orchestrates order routing and fills."""

from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation

import structlog

from config.settings import Settings
from core.enums import OrderStatus
from core.events import (
    EventBus,
    OrderCancelledEvent,
    OrderCreatedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
)
from core.events.base import Event
from core.events.streams import ORDERS, RISK_APPROVED
from services.execution.brokers.base import BaseBroker, BrokerOrder
from services.execution.fill_tracker import FillTracker
from services.execution.order_manager import OrderError, OrderManager
from services.execution.smart_router import SmartOrderRouter

logger = structlog.get_logger(__name__)

# Stream / group names for the event bus (see core.events.streams).
_RISK_STREAM = RISK_APPROVED
_ORDER_STREAM = ORDERS
_CONSUMER_GROUP = "execution-engine"

# Polling interval for pending order monitoring (seconds).
_POLL_INTERVAL = 2.0


class ExecutionEngineService:
    """Top-level execution service.

    Responsibilities:
    - Listens for RiskApprovedEvent from the event bus.
    - Routes orders through SmartOrderRouter.
    - Submits orders to the appropriate broker.
    - Monitors pending orders and publishes fill / reject / cancel events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        brokers: dict[str, BaseBroker],
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._brokers = brokers
        self._order_manager = OrderManager()
        self._router = SmartOrderRouter(brokers, settings)
        self._fill_tracker = FillTracker()
        self._monitor_task: asyncio.Task | None = None
        # order_id -> (broker_name, external_id) for pending tracking.
        self._in_flight: dict[str, tuple[str, str]] = {}
        # Correlation ids of RiskApprovedEvents already handled -- idempotency
        # under at-least-once event delivery (a redelivered event is skipped).
        self._processed_correlation_ids: set[str] = set()
        # Kill switch: when True, no new orders are submitted.
        self._halted: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to risk-approved events and start background monitoring."""
        await self._event_bus.subscribe(
            stream=_RISK_STREAM,
            group=_CONSUMER_GROUP,
            consumer="exec-worker-1",
            handler=self._handle_risk_approved,
        )
        self._monitor_task = asyncio.create_task(self._monitor_pending_orders())
        logger.info("execution_engine_started")

    async def stop(self) -> None:
        """Gracefully stop background tasks."""
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("execution_engine_stopped")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_risk_approved(self, event: Event) -> None:
        """Create and submit the order described by a RiskApprovedEvent.

        The event carries the full, sized order intent, so the order is created
        fresh in this engine's OrderManager (the risk service runs in a separate
        process with its own state, so its order id is not resolvable here).
        """
        symbol = getattr(event, "symbol", "") or event.payload.get("symbol", "")
        side = getattr(event, "side", "") or event.payload.get("side", "")
        order_type = (
            getattr(event, "order_type", "")
            or event.payload.get("order_type", "")
            or "market"
        )
        quantity_raw = getattr(event, "quantity", 0.0) or event.payload.get("quantity", 0.0)
        correlation_id = getattr(event, "order_id", "") or event.payload.get("order_id", "")

        if correlation_id and correlation_id in self._processed_correlation_ids:
            logger.info("risk_approved_duplicate_skipped", correlation_id=correlation_id)
            return

        if not symbol or not side:
            logger.warning(
                "risk_approved_missing_fields",
                event_id=event.event_id,
                symbol=symbol,
                side=side,
            )
            return

        try:
            quantity = Decimal(str(quantity_raw))
        except (InvalidOperation, ValueError):
            logger.warning("risk_approved_bad_quantity", quantity=quantity_raw)
            return
        if quantity <= 0:
            logger.warning("risk_approved_nonpositive_quantity", quantity=str(quantity))
            return

        limit_price = self._coerce_decimal(getattr(event, "limit_price", None))
        stop_price = self._coerce_decimal(getattr(event, "stop_price", None))
        reference_price = self._coerce_decimal(getattr(event, "reference_price", None))

        if correlation_id:
            self._processed_correlation_ids.add(correlation_id)

        try:
            await self.submit_order(
                order_id=None,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                reference_price=reference_price,
            )
            logger.info(
                "risk_approved_order_submitted",
                correlation_id=correlation_id,
                symbol=symbol,
                side=side,
                quantity=str(quantity),
            )
        except Exception as exc:
            logger.error(
                "risk_approved_submit_failed",
                correlation_id=correlation_id,
                error=str(exc),
            )

    @staticmethod
    def _coerce_decimal(value: object) -> Decimal | None:
        """Best-effort convert an optional numeric value to Decimal."""
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None

    async def _get_buying_power(self, broker: BaseBroker) -> Decimal | None:
        """Return the broker's available buying power, or None if unavailable."""
        try:
            account = await broker.get_account()
            return Decimal(str(account.get("buying_power", "0")))
        except Exception as exc:
            logger.warning("buying_power_check_failed", error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit_order(
        self,
        order_id: str | None = None,
        symbol: str = "",
        side: str = "",
        order_type: str = "",
        quantity: Decimal = Decimal("0"),
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
        reference_price: Decimal | None = None,
    ):
        """Create (or reuse) an order, route it, and submit to the broker."""
        if self._halted:
            logger.warning("order_rejected_trading_halted", symbol=symbol)
            return None
        # Create order if no existing order_id supplied.
        if order_id is None:
            order = self._order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
            )
            order_id = order.order_id
        else:
            order = self._order_manager.get_order(order_id)
            if order is None:
                raise OrderError(f"Order not found: {order_id}")

        # Publish OrderCreatedEvent.
        await self._event_bus.publish(
            _ORDER_STREAM,
            OrderCreatedEvent(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=float(order.quantity),
                source_service="execution-engine",
            ),
        )

        # Route the order.
        decision = self._router.route(
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
        )

        broker = self._brokers.get(decision.broker_name)
        if broker is None:
            reason = f"Broker not found: {decision.broker_name}"
            await self._publish_rejected(order_id, reason)
            return order

        # Pre-trade buying-power check for buys (defence on the money path).
        est_price = order.limit_price if order.limit_price is not None else reference_price
        if order.side == "buy" and est_price is not None and est_price > 0:
            buying_power = await self._get_buying_power(broker)
            required = order.quantity * est_price
            if buying_power is not None and required > buying_power:
                await self._publish_rejected(
                    order_id,
                    f"insufficient_buying_power required={required} available={buying_power}",
                )
                return order

        # Build broker order. external_id doubles as the client order id, so a
        # broker-side retry after a lost response is deduped rather than
        # placing a second live order.
        broker_order = BrokerOrder(
            external_id=order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_price=order.stop_price,
        )

        try:
            external_id = await broker.submit_order(broker_order)
            self._order_manager.update_status(order_id, OrderStatus.SUBMITTED)
            order.external_id = external_id
            order.broker_name = decision.broker_name
            self._in_flight[order_id] = (decision.broker_name, external_id)

            logger.info(
                "order_submitted_to_broker",
                order_id=order_id,
                broker=decision.broker_name,
                external_id=external_id,
            )
        except Exception as exc:
            logger.error(
                "broker_submit_failed",
                order_id=order_id,
                broker=decision.broker_name,
                error=str(exc),
            )
            await self._publish_rejected(order_id, str(exc))

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an in-flight order."""
        flight_info = self._in_flight.get(order_id)
        if not flight_info:
            logger.warning("cancel_order_not_in_flight", order_id=order_id)
            return False

        broker_name, external_id = flight_info
        broker = self._brokers.get(broker_name)
        if broker is None:
            return False

        cancelled = await broker.cancel_order(external_id)
        if cancelled:
            try:
                self._order_manager.update_status(order_id, OrderStatus.CANCELLED)
            except OrderError:
                pass
            del self._in_flight[order_id]
            await self._event_bus.publish(
                _ORDER_STREAM,
                OrderCancelledEvent(
                    order_id=order_id,
                    reason="user_requested",
                    source_service="execution-engine",
                ),
            )
            logger.info("order_cancelled", order_id=order_id)
        return cancelled

    # ------------------------------------------------------------------
    # Kill switch & reconciliation
    # ------------------------------------------------------------------

    @property
    def halted(self) -> bool:
        """Whether the engine is currently refusing new orders."""
        return self._halted

    def halt(self) -> None:
        """Stop accepting new orders."""
        self._halted = True
        logger.warning("trading_halted")

    def resume(self) -> None:
        """Resume accepting new orders."""
        self._halted = False
        logger.info("trading_resumed")

    async def emergency_flatten(self) -> dict:
        """Kill switch: halt new orders, cancel open orders, and flatten every
        broker position with market orders. Returns a summary."""
        self._halted = True
        cancelled = 0
        for order_id in list(self._in_flight):
            try:
                if await self.cancel_order(order_id):
                    cancelled += 1
            except Exception as exc:
                logger.error("flatten_cancel_failed", order_id=order_id, error=str(exc))

        flattened: list[dict] = []
        for broker_name, broker in self._brokers.items():
            try:
                positions = await broker.get_positions()
            except Exception as exc:
                logger.error("flatten_positions_failed", broker=broker_name, error=str(exc))
                continue
            for pos in positions:
                qty = Decimal(str(pos.get("quantity", "0")))
                if qty == 0:
                    continue
                side = "sell" if qty > 0 else "buy"
                close = BrokerOrder(
                    external_id="",
                    symbol=pos["symbol"],
                    side=side,
                    order_type="market",
                    quantity=abs(qty),
                )
                try:
                    await broker.submit_order(close)
                    flattened.append(
                        {"symbol": pos["symbol"], "side": side, "quantity": str(abs(qty))}
                    )
                except Exception as exc:
                    logger.error("flatten_submit_failed", symbol=pos["symbol"], error=str(exc))

        logger.warning(
            "emergency_flatten_complete", cancelled=cancelled, flattened=len(flattened)
        )
        return {"halted": True, "cancelled": cancelled, "flattened": flattened}

    async def reconcile_positions(self, expected: dict[str, Decimal]) -> list[dict]:
        """Compare *expected* positions (symbol -> signed qty) against the
        broker's actual positions; return discrepancies. The broker is the
        source of truth, so a non-empty result means our books have drifted."""
        actual: dict[str, Decimal] = {}
        for broker in self._brokers.values():
            try:
                for pos in await broker.get_positions():
                    sym = pos["symbol"]
                    actual[sym] = actual.get(sym, Decimal("0")) + Decimal(
                        str(pos.get("quantity", "0"))
                    )
            except Exception as exc:
                logger.error("reconcile_positions_failed", error=str(exc))

        discrepancies: list[dict] = []
        for sym in sorted(set(expected) | set(actual)):
            exp = Decimal(str(expected.get(sym, 0)))
            act = actual.get(sym, Decimal("0"))
            if exp != act:
                discrepancies.append(
                    {
                        "symbol": sym,
                        "expected": str(exp),
                        "actual": str(act),
                        "diff": str(act - exp),
                    }
                )
        return discrepancies

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def _monitor_pending_orders(self) -> None:
        """Periodically poll brokers for fill updates on in-flight orders."""
        while True:
            try:
                await asyncio.sleep(_POLL_INTERVAL)

                for order_id, (broker_name, external_id) in list(self._in_flight.items()):
                    broker = self._brokers.get(broker_name)
                    if broker is None:
                        continue

                    try:
                        status_info = await broker.get_order_status(external_id)
                    except Exception as exc:
                        logger.warning(
                            "poll_status_failed",
                            order_id=order_id,
                            error=str(exc),
                        )
                        continue

                    raw_status = status_info.get("status", "")

                    if raw_status in (OrderStatus.FILLED, "filled"):
                        filled_qty = Decimal(str(status_info.get("filled_qty", 0)))
                        filled_price = Decimal(str(status_info.get("filled_avg_price", 0)))

                        if filled_qty > 0 and filled_price > 0:
                            order = self._order_manager.get_order(order_id)
                            symbol = order.symbol if order else status_info.get("symbol", "")
                            side = order.side if order else status_info.get("side", "")

                            self._order_manager.record_fill(
                                order_id=order_id,
                                price=filled_price,
                                quantity=filled_qty,
                            )
                            self._fill_tracker.record_fill(
                                order_id=order_id,
                                symbol=symbol,
                                side=side,
                                price=filled_price,
                                quantity=filled_qty,
                            )

                            await self._event_bus.publish(
                                _ORDER_STREAM,
                                OrderFilledEvent(
                                    order_id=order_id,
                                    symbol=symbol,
                                    side=side,
                                    fill_price=float(filled_price),
                                    fill_quantity=float(filled_qty),
                                    source_service="execution-engine",
                                ),
                            )
                        del self._in_flight[order_id]

                    elif raw_status in (OrderStatus.REJECTED, "rejected"):
                        reason = status_info.get("reason", "broker_rejected")
                        await self._publish_rejected(order_id, reason)
                        del self._in_flight[order_id]

                    elif raw_status in (OrderStatus.CANCELLED, "cancelled", "canceled"):
                        try:
                            self._order_manager.update_status(order_id, OrderStatus.CANCELLED)
                        except OrderError:
                            pass
                        await self._event_bus.publish(
                            _ORDER_STREAM,
                            OrderCancelledEvent(
                                order_id=order_id,
                                reason="broker_cancelled",
                                source_service="execution-engine",
                            ),
                        )
                        del self._in_flight[order_id]

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("monitor_pending_orders_error")
                await asyncio.sleep(_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _publish_rejected(self, order_id: str, reason: str) -> None:
        """Mark an order as rejected and publish the event."""
        try:
            self._order_manager.update_status(order_id, OrderStatus.REJECTED)
        except OrderError:
            pass
        await self._event_bus.publish(
            _ORDER_STREAM,
            OrderRejectedEvent(
                order_id=order_id,
                reason=reason,
                source_service="execution-engine",
            ),
        )
        if order_id in self._in_flight:
            del self._in_flight[order_id]
        logger.warning("order_rejected", order_id=order_id, reason=reason)
