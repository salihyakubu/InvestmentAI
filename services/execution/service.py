"""Main execution engine service that orchestrates order routing and fills."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import structlog

from config.settings import Settings
from core.enums import OrderStatus, TradingMode
from core.events import (
    EventBus,
    OrderCancelledEvent,
    OrderCreatedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
    RiskApprovedEvent,
)
from core.events.base import Event
from services.execution.brokers.base import BaseBroker, BrokerFill, BrokerOrder
from services.execution.fill_tracker import FillTracker
from services.execution.order_manager import OrderError, OrderManager
from services.execution.smart_router import SmartOrderRouter

logger = structlog.get_logger(__name__)

# Stream / group names for the event bus.
_RISK_STREAM = "risk.approved"
_ORDER_STREAM = "orders"
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
        """Handle a RiskApprovedEvent by submitting the order to a broker."""
        order_id = event.payload.get("order_id") or getattr(event, "order_id", None)
        if not order_id:
            logger.warning("risk_approved_missing_order_id", event_id=event.event_id)
            return

        order = self._order_manager.get_order(order_id)
        if order is None:
            logger.warning(
                "risk_approved_order_not_found",
                order_id=order_id,
            )
            return

        try:
            await self.submit_order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        except Exception as exc:
            logger.error(
                "risk_approved_submit_failed",
                order_id=order_id,
                error=str(exc),
            )
            await self._publish_rejected(order_id, str(exc))

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
    ):
        """Create (or reuse) an order, route it, and submit to the broker."""
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

        # Build broker order.
        broker_order = BrokerOrder(
            external_id="",
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
