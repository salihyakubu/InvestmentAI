"""Main execution engine service that orchestrates order routing and fills."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

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
from core.events.streams import CONTROL, LIQUIDATION, ORDER_INTENTS, ORDERS, RISK_APPROVED
from services.execution.brokers.base import BaseBroker, BrokerOrder
from services.execution.fill_tracker import FillTracker
from services.execution.order_manager import Order, OrderError, OrderManager
from services.execution.smart_router import SmartOrderRouter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from services.persistence.safety_state import SafetyStateStore

logger = structlog.get_logger(__name__)

# Durable-row statuses that mean "the venue may still act on this order".
_NON_TERMINAL_ROW_STATUSES = (
    OrderStatus.PENDING.value,
    OrderStatus.SUBMITTED.value,
    OrderStatus.PARTIAL_FILL.value,
)

# Stream / group names for the event bus (see core.events.streams).
_RISK_STREAM = RISK_APPROVED
_ORDER_STREAM = ORDERS
_CONSUMER_GROUP = "execution-engine"

# Polling interval for pending order monitoring (seconds).
_POLL_INTERVAL = 2.0

# CONTROL-stream actions owned by other services (each consumer group sees
# every control event); logged at debug so a valid request is not misread as
# a failure. "retrain" belongs to the continuous-learning service.
_FOREIGN_CONTROL_ACTIONS = frozenset({"retrain"})


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
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        state_store: SafetyStateStore | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._settings = settings
        self._brokers = brokers
        # Durable order rows (written BEFORE the broker is called) and the
        # safety-state row (halt flag). None = in-memory only, for tests.
        self._session_factory = session_factory
        self._state_store = state_store
        self._halt_reason: str = ""
        self._order_manager = OrderManager()
        self._router = SmartOrderRouter(brokers, settings)
        self._fill_tracker = FillTracker()
        self._monitor_task: asyncio.Task[None] | None = None
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
        await self._event_bus.subscribe(
            stream=ORDER_INTENTS,
            group=_CONSUMER_GROUP,
            consumer="exec-intents-1",
            handler=self._handle_order_intent,
        )
        # Remote kill switch: operator halt / resume / flatten commands.
        await self._event_bus.subscribe(
            stream=CONTROL,
            group=_CONSUMER_GROUP,
            consumer="exec-control-1",
            handler=self._handle_control,
        )
        # Forced exits (stops, emergency liquidation). Until 2026-11-02 this
        # stream had no consumer at all: the liquidation manager published and
        # nothing closed a position.
        await self._event_bus.subscribe(
            stream=LIQUIDATION,
            group=_CONSUMER_GROUP,
            consumer="exec-liquidation-1",
            handler=self._handle_liquidation,
        )
        # Feed live prices into simulated brokers so paper fills happen at real
        # market prices and pending limit/stop orders can actually trigger.
        # ("market.prices" literal is owned by data ingestion; kept in sync.)
        await self._event_bus.subscribe(
            stream="market.prices",
            group=_CONSUMER_GROUP,
            consumer="exec-prices-1",
            handler=self._handle_price_update,
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
        # Restart-safe dedupe: the in-memory set is empty after a restart, but
        # the Order row written before submit carries the correlation id.
        if correlation_id and await self._correlation_already_ordered(correlation_id):
            self._processed_correlation_ids.add(correlation_id)
            logger.warning(
                "risk_approved_duplicate_skipped_durable", correlation_id=correlation_id
            )
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
                # The risk correlation id ("rebal-.."/"explore-..") rides in
                # client_order_id so OrderFilledEvent carries the linkage.
                client_order_id=correlation_id or None,
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

    async def _handle_control(self, event: Event) -> None:
        """Operator control commands (the remote kill switch)."""
        if event.event_type != "TradingControlEvent":
            return
        action = getattr(event, "action", "") or event.payload.get("action", "")
        reason = getattr(event, "reason", "") or event.payload.get("reason", "")
        # Foreign actions (owned by other services, e.g. retrain) arrive on the
        # shared CONTROL stream too — debug, so runbook readers only see
        # WARNING for genuinely operator-critical halt/resume/flatten.
        control_log = (
            logger.debug if action in _FOREIGN_CONTROL_ACTIONS else logger.warning
        )
        control_log("control_command_received", action=action, reason=reason)
        if action == "halt":
            self.halt(reason or "operator halt")
            await self._persist_halt(reason or "operator halt")
        elif action == "resume":
            self.resume()
            await self._persist_halt(reason or "operator resume")
        elif action == "flatten":
            result = await self.emergency_flatten()
            logger.warning("control_flatten_result", **result)
        elif action in _FOREIGN_CONTROL_ACTIONS:
            logger.debug("control_foreign_action", action=action)
        else:
            logger.warning("control_unknown_action", action=action)

    async def _handle_liquidation(self, event: Event) -> None:
        """Close a position the liquidation manager says must go.

        ``quantity <= 0`` means "the whole position": it is resolved against
        the broker's actual signed book, never against a stale local figure.
        Liquidation orders are risk-reducing, so like the kill switch's own
        flatten legs they bypass a halt; they never bypass the live-venue
        gate or the tracked path, and they carry a ``liq-`` correlation id.
        """
        if event.event_type != "LiquidationTriggeredEvent":
            return
        if event.event_id in self._processed_correlation_ids:
            return
        self._processed_correlation_ids.add(event.event_id)
        symbol = str(getattr(event, "symbol", "") or event.payload.get("symbol", ""))
        reason = str(getattr(event, "reason", "") or event.payload.get("reason", ""))
        requested = self._coerce_decimal(getattr(event, "quantity", None)) or Decimal("0")
        if not symbol:
            logger.warning("liquidation_missing_symbol", event_id=event.event_id)
            return

        signed = await self._broker_signed_quantity(symbol)
        if signed == 0:
            logger.warning("liquidation_nothing_held", symbol=symbol, reason=reason)
            return
        quantity = min(abs(signed), requested) if requested > 0 else abs(signed)
        side = "sell" if signed > 0 else "buy"
        try:
            placed = await self.submit_order(
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=quantity,
                client_order_id=f"liq-{event.event_id}",
                bypass_halt=True,
            )
        except Exception as exc:
            logger.error("liquidation_submit_failed", symbol=symbol, error=str(exc))
            return
        logger.warning(
            "liquidation_order_submitted",
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            reason=reason,
            order_id=placed.order_id if placed else None,
        )

    async def _broker_signed_quantity(self, symbol: str) -> Decimal:
        """Net signed quantity of *symbol* across the configured brokers."""
        total = Decimal("0")
        for broker in self._brokers.values():
            try:
                for pos in await broker.get_positions():
                    if pos.get("symbol") == symbol:
                        total += Decimal(str(pos.get("quantity", "0")))
            except Exception as exc:
                logger.error("liquidation_positions_failed", error=str(exc))
        return total

    async def _handle_price_update(self, event: Event) -> None:
        """Feed PriceUpdateEvents into simulated brokers.

        The paper broker fills market orders at its last known price and
        triggers pending limit/stop orders on price updates -- without this
        feed, paper fills would execute against missing prices and resting
        orders would never trigger.
        """
        if event.event_type != "PriceUpdateEvent":
            return
        symbol = getattr(event, "symbol", "") or event.payload.get("symbol", "")
        price_raw = getattr(event, "price", None) or event.payload.get("price")
        if not symbol or price_raw is None:
            return
        try:
            price = Decimal(str(price_raw))
        except InvalidOperation:
            return
        for broker in self._brokers.values():
            update = getattr(broker, "update_price", None)
            if update is not None and not broker.supports_live:
                update(symbol, price)

    async def _handle_order_intent(self, event: Event) -> None:
        """Handle a manual OrderIntentEvent: create + submit the order.

        Manual orders rely on the execution-level gates (buying-power, halt)
        rather than the weight-based pre_trade_check used by the rebalance path.
        """
        symbol = getattr(event, "symbol", "") or event.payload.get("symbol", "")
        side = getattr(event, "side", "") or event.payload.get("side", "")
        order_type = (
            getattr(event, "order_type", "")
            or event.payload.get("order_type", "")
            or "market"
        )
        quantity_raw = getattr(event, "quantity", 0.0) or event.payload.get("quantity", 0.0)
        if not symbol or not side:
            logger.warning("order_intent_missing_fields", event_id=event.event_id)
            return
        try:
            quantity = Decimal(str(quantity_raw))
        except (InvalidOperation, ValueError):
            logger.warning("order_intent_bad_quantity", quantity=quantity_raw)
            return
        if quantity <= 0:
            return
        # The intent's order_id is the originating DB order id. It becomes the
        # engine's order id too (one durable row == one engine order), and it
        # rides in client_order_id so fills carry the linkage.
        db_order_id = str(
            getattr(event, "order_id", "") or event.payload.get("order_id", "")
        )
        preferred = db_order_id if self._is_uuid(db_order_id) else None
        if preferred is not None and self._order_manager.get_order(preferred) is not None:
            logger.info("order_intent_duplicate_skipped", order_id=preferred)
            return
        try:
            await self.submit_order(
                order_id=None,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=self._coerce_decimal(getattr(event, "limit_price", None)),
                stop_price=self._coerce_decimal(getattr(event, "stop_price", None)),
                reference_price=self._coerce_decimal(getattr(event, "reference_price", None)),
                client_order_id=db_order_id or None,
                preferred_order_id=preferred,
            )
        except Exception as exc:
            logger.error("order_intent_submit_failed", error=str(exc))

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
        client_order_id: str | None = None,
        bypass_halt: bool = False,
        preferred_order_id: str | None = None,
    ) -> Order | None:
        """Create (or reuse) an order, route it, and submit to the broker.

        When a session factory is configured the Order row is written BEFORE
        the broker is called and the submit is refused if that write fails:
        an order the venue knows about but the ledger does not is the exact
        failure the 2026-09-12 drift investigation found. *preferred_order_id*
        makes the engine's order id equal to an existing row id (manual API
        orders) so the row and the engine object are one.

        *bypass_halt* is for the kill switch's OWN remediation orders only.
        emergency_flatten sets ``_halted`` before flattening, so its closing
        orders would otherwise be refused here -- which is why they used to
        go straight to the broker, changing the book while the durable
        record learned nothing (no Order row, no fill event, no ledger
        entry). Routing them through this path keeps the record faithful at
        exactly the moment it matters most.
        """
        if self._halted and not bypass_halt:
            logger.warning("order_rejected_trading_halted", symbol=symbol)
            return None
        # Create order if no existing order_id supplied.
        order: Order | None
        if order_id is None:
            order = self._order_manager.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                order_id=preferred_order_id,
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

        # The router returns None when no broker is safe to route to (e.g. only
        # live brokers wired in a non-live mode). Refuse rather than dispatch.
        broker_name = decision.broker_name
        if broker_name is None:
            await self._publish_rejected(
                order_id,
                f"no_safe_broker symbol={order.symbol} "
                f"trading_mode={self._settings.trading_mode}",
            )
            return order

        broker = self._brokers.get(broker_name)
        if broker is None:
            await self._publish_rejected(order_id, f"Broker not found: {broker_name}")
            return order

        # Hard safety guard (defence in depth): never dispatch to a venue that
        # can place real-money orders unless we are explicitly in live mode.
        # This is independent of the router's selection and of broker-dict
        # ordering, so the paper-first posture holds even if a live broker is
        # somehow wired or selected outside live mode.
        if broker.supports_live and self._settings.trading_mode != "live":
            await self._publish_rejected(
                order_id,
                f"live_broker_blocked broker={broker_name} "
                f"trading_mode={self._settings.trading_mode}",
            )
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

        # Durable row BEFORE the venue is touched. Fail closed: no row, no order.
        if not await self._persist_order_row(order, broker_name):
            await self._publish_rejected(order_id, "order_persistence_failed")
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
            order.broker_name = broker_name
            self._in_flight[order_id] = (broker_name, external_id)
            await self._mark_row(
                order_id,
                status=OrderStatus.SUBMITTED,
                external_id=external_id,
                submitted_at=datetime.now(UTC),
            )

            logger.info(
                "order_submitted_to_broker",
                order_id=order_id,
                broker=broker_name,
                external_id=external_id,
            )
            # Simulated venues fill market orders inside submit_order; polling
            # now instead of in 2 s closes the restart window between the
            # broker's book changing and the fill reaching the durable record.
            if not broker.supports_live:
                try:
                    await self._poll_order(order_id, broker_name, external_id)
                except Exception:
                    logger.exception("post_submit_poll_failed", order_id=order_id)
        except Exception as exc:
            logger.error(
                "broker_submit_failed",
                order_id=order_id,
                broker=broker_name,
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
            await self._mark_row(
                order_id, status=OrderStatus.CANCELLED, cancelled_at=datetime.now(UTC)
            )
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

    @property
    def brokers(self) -> dict[str, BaseBroker]:
        """Configured brokers (name -> adapter)."""
        return self._brokers

    def halt(self, reason: str = "") -> None:
        """Stop accepting new orders (in memory; callers persist via _persist_halt)."""
        self._halted = True
        self._halt_reason = reason
        logger.warning("trading_halted", reason=reason)

    def resume(self) -> None:
        """Resume accepting new orders."""
        self._halted = False
        self._halt_reason = ""
        logger.info("trading_resumed")

    async def restore_safety_state(self) -> None:
        """Re-apply a persisted halt before any order can be accepted.

        Until 2026-11-02 ``_halted`` was process memory, so an operator halt
        was silently lifted by the next deploy or crash-restart.
        """
        if self._state_store is None:
            return
        try:
            state = await self._state_store.load()
        except Exception:
            logger.exception("safety_state_load_failed")
            return
        if state is not None and state.halted:
            self._halted = True
            self._halt_reason = state.halt_reason or "restored from safety_state"
            logger.warning("trading_halt_restored", reason=self._halt_reason)

    async def _persist_halt(self, reason: str) -> None:
        if self._state_store is None:
            return
        try:
            await self._state_store.save_halt(self._halted, reason)
        except Exception:
            logger.exception("safety_state_save_failed")

    async def emergency_flatten(self) -> dict[str, Any]:
        """Kill switch: halt new orders, cancel open orders, and flatten every
        broker position with market orders. Returns a summary."""
        self._halted = True
        self._halt_reason = "emergency_flatten"
        await self._persist_halt("emergency_flatten")
        cancelled = 0
        for order_id in list(self._in_flight):
            try:
                if await self.cancel_order(order_id):
                    cancelled += 1
            except Exception as exc:
                logger.error("flatten_cancel_failed", order_id=order_id, error=str(exc))

        flattened: list[dict[str, Any]] = []
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
                try:
                    # Through the tracked path, not straight to the broker:
                    # the monitor loop only publishes fills for orders the
                    # order manager knows about, so a direct call would
                    # flatten the book and leave no durable trace of it.
                    placed = await self.submit_order(
                        symbol=pos["symbol"],
                        side=side,
                        order_type="market",
                        quantity=abs(qty),
                        client_order_id=f"flatten-{uuid.uuid4()}",
                        bypass_halt=True,
                    )
                    if placed is not None:
                        flattened.append(
                            {"symbol": pos["symbol"], "side": side, "quantity": str(abs(qty))}
                        )
                except Exception as exc:
                    logger.error("flatten_submit_failed", symbol=pos["symbol"], error=str(exc))

        logger.warning(
            "emergency_flatten_complete", cancelled=cancelled, flattened=len(flattened)
        )
        return {"halted": True, "cancelled": cancelled, "flattened": flattened}

    async def reconcile_positions(self, expected: dict[str, Decimal]) -> list[dict[str, Any]]:
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

        discrepancies: list[dict[str, Any]] = []
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
    # Durable order rows + startup reconciliation
    # ------------------------------------------------------------------

    async def reconcile_on_start(self) -> dict[str, int]:
        """Pick up every non-terminal Order row and ask the venue what happened.

        Call before ``start()``. A fill the venue reports is recorded and
        published exactly as the monitor would have; an order the venue still
        holds is re-seeded into ``_in_flight``; an order the venue has no
        memory of (the paper broker after a restart) is marked ``expired`` with
        reason ``lost_on_restart`` and logged at error level -- never silently
        dropped, which is what happened to every in-flight order before
        2026-11-02.
        """
        summary = {"rows": 0, "filled": 0, "open": 0, "closed": 0, "lost": 0}
        if self._session_factory is None:
            return summary
        from core.models.orders import Order as OrderRow

        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(OrderRow).where(
                        OrderRow.trading_mode == self._settings.trading_mode,
                        OrderRow.status.in_(_NON_TERMINAL_ROW_STATUSES),
                    )
                )
            ).scalars().all()

        for row in rows:
            if row.broker_name is None:
                # Written by the API but never claimed by an engine: its
                # OrderIntentEvent is still in the stream (or never came) and
                # will create the order normally. Not ours to expire.
                continue
            summary["rows"] += 1
            order_id = str(row.id)
            order = Order(
                order_id=order_id,
                symbol=row.symbol,
                side=row.side,
                order_type=row.order_type,
                quantity=Decimal(str(row.quantity)),
                status=(
                    OrderStatus.PENDING
                    if row.status == OrderStatus.PENDING.value
                    else OrderStatus.SUBMITTED
                ),
                limit_price=Decimal(str(row.limit_price)) if row.limit_price is not None else None,
                stop_price=Decimal(str(row.stop_price)) if row.stop_price is not None else None,
                external_id=row.external_id,
                broker_name=row.broker_name,
                client_order_id=row.correlation_id,
            )
            if self._order_manager.get_order(order_id) is None:
                self._order_manager.restore_order(order)
            if order.client_order_id:
                self._processed_correlation_ids.add(order.client_order_id)

            broker = self._brokers.get(row.broker_name or "")
            external_id = row.external_id or order_id
            status_info: dict[str, Any] = {}
            if broker is not None and row.status != OrderStatus.PENDING.value:
                try:
                    status_info = await broker.get_order_status(external_id)
                except Exception as exc:
                    logger.warning("reconcile_status_failed", order_id=order_id, error=str(exc))
            raw = str(status_info.get("status", "unknown") or "unknown")

            if raw in (OrderStatus.FILLED, "filled"):
                filled_qty = Decimal(str(status_info.get("filled_qty", 0)))
                filled_price = Decimal(str(status_info.get("filled_avg_price", 0)))
                if filled_qty > 0 and filled_price > 0:
                    await self._record_and_publish_fill(
                        order_id, status_info, filled_qty, filled_price
                    )
                    summary["filled"] += 1
                else:
                    summary["lost"] += 1
                    await self._expire_lost(order_id, "filled_without_quantity")
            elif raw in ("submitted", "new", "accepted", "open", "pending", "partially_filled",
                         OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL):
                self._in_flight[order_id] = (row.broker_name or "", external_id)
                summary["open"] += 1
            elif raw in (OrderStatus.REJECTED, "rejected"):
                await self._publish_rejected(order_id, "broker_rejected_while_down")
                summary["closed"] += 1
            elif raw in (OrderStatus.CANCELLED, "cancelled", "canceled", "expired",
                         OrderStatus.EXPIRED):
                try:
                    self._order_manager.update_status(order_id, OrderStatus.CANCELLED)
                except OrderError:
                    pass
                await self._mark_row(
                    order_id, status=OrderStatus.CANCELLED, cancelled_at=datetime.now(UTC)
                )
                await self._event_bus.publish(
                    _ORDER_STREAM,
                    OrderCancelledEvent(
                        order_id=order_id,
                        reason="broker_cancelled_while_down",
                        source_service="execution-engine",
                    ),
                )
                summary["closed"] += 1
            else:
                summary["lost"] += 1
                await self._expire_lost(order_id, "lost_on_restart")

        logger.warning("order_reconciliation_complete", **summary)
        return summary

    async def _expire_lost(self, order_id: str, reason: str) -> None:
        """Mark an order nobody can account for: the one outcome never allowed
        to be silent. The broker's book may or may not include it; the ledger
        now says so explicitly and the metrics layer keeps withholding until
        it reconciles."""
        try:
            self._order_manager.update_status(order_id, OrderStatus.EXPIRED)
        except OrderError:
            pass
        await self._mark_row(order_id, status=OrderStatus.EXPIRED)
        await self._event_bus.publish(
            _ORDER_STREAM,
            OrderCancelledEvent(order_id=order_id, reason=reason, source_service="execution-engine"),
        )
        logger.error("order_lost_on_restart", order_id=order_id, reason=reason)

    @staticmethod
    def _is_uuid(value: str | None) -> bool:
        if not value:
            return False
        try:
            uuid.UUID(str(value))
            return True
        except (ValueError, TypeError):
            return False

    async def _correlation_already_ordered(self, correlation_id: str) -> bool:
        if self._session_factory is None:
            return False
        from core.models.orders import Order as OrderRow

        try:
            async with self._session_factory() as session:
                found = await session.execute(
                    select(OrderRow.id).where(OrderRow.correlation_id == correlation_id).limit(1)
                )
                return found.first() is not None
        except Exception:
            logger.exception("correlation_lookup_failed", correlation_id=correlation_id)
            return False

    async def _persist_order_row(self, order: Order, broker_name: str) -> bool:
        """Write (or claim) the Order row for *order* before the venue is called.

        Returns True when the row exists afterwards. Without a session factory
        the engine runs in-memory only and this is a no-op that returns True.
        """
        if self._session_factory is None:
            return True
        from core.models.orders import Order as OrderRow

        now = datetime.now(UTC)
        row_id = uuid.UUID(order.order_id)
        correlation = order.client_order_id or None
        try:
            async with self._session_factory() as session:
                row = await session.get(OrderRow, row_id)
                if row is None:
                    row = OrderRow(
                        id=row_id,
                        symbol=order.symbol,
                        asset_class="crypto" if "/" in order.symbol else "stock",
                        side=order.side,
                        order_type=order.order_type,
                        quantity=order.quantity,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        status=OrderStatus.PENDING.value,
                        trading_mode=self._settings.trading_mode,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(row)
                row.broker_name = broker_name
                if correlation and not self._is_uuid(correlation):
                    row.correlation_id = correlation[:100]
                row.updated_at = now
                await session.commit()
            return True
        except Exception:
            logger.exception("order_row_persist_failed", order_id=order.order_id)
            return False

    async def _mark_row(self, order_id: str, *, status: OrderStatus, **fields: Any) -> None:
        """Best-effort status update on the durable row (the event is the record of truth)."""
        if self._session_factory is None or not self._is_uuid(order_id):
            return
        from core.models.orders import Order as OrderRow

        try:
            async with self._session_factory() as session:
                row = await session.get(OrderRow, uuid.UUID(order_id))
                if row is None:
                    return
                row.status = status.value
                for key, value in fields.items():
                    setattr(row, key, value)
                row.updated_at = datetime.now(UTC)
                await session.commit()
        except Exception:
            logger.exception("order_row_update_failed", order_id=order_id)

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------

    async def _monitor_pending_orders(self) -> None:
        """Periodically poll brokers for fill updates on in-flight orders.

        Each in-flight order is polled under its own exception guard. Before
        this, one order whose fill had been recorded but whose event failed to
        publish (a Redis blip) raised OrderError on every later poll
        (FILLED -> FILLED is not a legal transition) and aborted the whole
        loop body, so every order behind it in ``_in_flight`` never had its
        fill published for the life of the process -- the "closes missing
        from the ledger" signature (GO_LIVE.md 2026-09-12) with no restart
        required. Recording is now idempotent and the in-flight entry is
        released only after the fill event has been published.
        """
        while True:
            try:
                await asyncio.sleep(_POLL_INTERVAL)
                for order_id, (broker_name, external_id) in list(self._in_flight.items()):
                    try:
                        await self._poll_order(order_id, broker_name, external_id)
                    except Exception:
                        logger.exception("monitor_order_poll_error", order_id=order_id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("monitor_pending_orders_error")
                await asyncio.sleep(_POLL_INTERVAL)

    async def _poll_order(self, order_id: str, broker_name: str, external_id: str) -> None:
        """Poll one in-flight order and publish whatever terminal state it reached."""
        broker = self._brokers.get(broker_name)
        if broker is None:
            return
        try:
            status_info = await broker.get_order_status(external_id)
        except Exception as exc:
            logger.warning("poll_status_failed", order_id=order_id, error=str(exc))
            return

        raw_status = status_info.get("status", "")

        if raw_status in (OrderStatus.FILLED, "filled"):
            filled_qty = Decimal(str(status_info.get("filled_qty", 0)))
            filled_price = Decimal(str(status_info.get("filled_avg_price", 0)))
            if filled_qty > 0 and filled_price > 0:
                await self._record_and_publish_fill(
                    order_id, status_info, filled_qty, filled_price
                )
            # Released only once the event is out: a publish failure raises
            # past this line and the order is polled again next tick.
            self._in_flight.pop(order_id, None)

        elif raw_status in (OrderStatus.REJECTED, "rejected"):
            reason = status_info.get("reason", "broker_rejected")
            await self._publish_rejected(order_id, reason)
            self._in_flight.pop(order_id, None)

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
            self._in_flight.pop(order_id, None)

    async def _record_and_publish_fill(
        self,
        order_id: str,
        status_info: dict[str, Any],
        filled_qty: Decimal,
        filled_price: Decimal,
    ) -> None:
        """Record a fill exactly once and publish it.

        Safe to call again after a failed publish: only the quantity not yet
        recorded on the order is recorded, so a retry neither double-counts
        the book nor attempts an illegal FILLED -> FILLED transition.
        """
        order = self._order_manager.get_order(order_id)
        symbol = order.symbol if order else str(status_info.get("symbol") or "")
        side = order.side if order else str(status_info.get("side") or "")
        already = order.filled_quantity if order else Decimal("0")
        delta = filled_qty - already
        if delta > 0:
            self._order_manager.record_fill(
                order_id=order_id, price=filled_price, quantity=delta
            )
            self._fill_tracker.record_fill(
                order_id=order_id,
                symbol=symbol,
                side=side,
                price=filled_price,
                quantity=delta,
            )
        # Dedupe key for the persistence layer: the venue's own fill id when
        # the adapter reports one, else a stand-in that is stable across
        # retries of the same cumulative fill.
        fills = status_info.get("fills") or []
        fill_id = str(fills[-1].get("id")) if fills and fills[-1].get("id") else (
            f"{status_info.get('external_id') or order_id}:{filled_qty}"
        )
        await self._event_bus.publish(
            _ORDER_STREAM,
            OrderFilledEvent(
                order_id=order_id,
                symbol=symbol,
                side=side,
                fill_price=float(filled_price),
                fill_quantity=float(filled_qty),
                client_order_id=(order.client_order_id or "") if order else "",
                fill_id=fill_id,
                source_service="execution-engine",
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _publish_rejected(self, order_id: str, reason: str) -> None:
        """Mark an order as rejected and publish the event."""
        order = self._order_manager.get_order(order_id)
        try:
            self._order_manager.update_status(order_id, OrderStatus.REJECTED)
        except OrderError:
            pass
        await self._mark_row(order_id, status=OrderStatus.REJECTED)
        await self._event_bus.publish(
            _ORDER_STREAM,
            OrderRejectedEvent(
                order_id=order_id,
                reason=reason,
                client_order_id=(order.client_order_id or "") if order else "",
                source_service="execution-engine",
            ),
        )
        if order_id in self._in_flight:
            del self._in_flight[order_id]
        logger.warning("order_rejected", order_id=order_id, reason=reason)
