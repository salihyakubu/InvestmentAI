"""Persist execution outcomes back to the database.

The execution engine runs on in-memory order state and emits lifecycle events
on the ``orders`` stream. This service subscribes to those events and persists
them. Since 2026-11-02 the execution engine writes an Order row BEFORE the
broker is called, with the row id equal to its own order id, so a fill event
resolves to its row directly; fills from engines without a session factory (or
events from before that date) still INSERT a completed row. Fills are deduped
on the venue fill id and write errors are left to the bus to redeliver.
Reporting/audit only; the trade itself and risk/position tracking already
react to fills directly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.enums import OrderStatus
from core.events.base import Event, EventBus
from core.events.streams import ORDERS
from core.models.orders import Fill, Order

logger = structlog.get_logger(__name__)


class OrderPersistenceService:
    """Subscribe to order lifecycle events and persist them to the DB."""

    def __init__(
        self,
        event_bus: EventBus,
        session_factory: async_sessionmaker[AsyncSession],
        trading_mode: str = "paper",
    ) -> None:
        self._event_bus = event_bus
        self._session_factory = session_factory
        self._trading_mode = trading_mode

    async def start(self) -> None:
        await self._event_bus.subscribe(
            stream=ORDERS,
            group="order-persistence",
            consumer="persist-1",
            handler=self.handle_event,
        )
        logger.info("OrderPersistenceService started.")

    async def handle_event(self, event: Event) -> None:
        """Route fills/rejections to the DB; ignore other order events."""
        if event.event_type == "OrderFilledEvent":
            await self._apply_fill(event)
        elif event.event_type == "OrderRejectedEvent":
            await self._apply_status(event, OrderStatus.REJECTED)

    @staticmethod
    def _db_order_id(event: Event) -> uuid.UUID | None:
        raw = getattr(event, "client_order_id", "") or event.payload.get("client_order_id", "")
        if not raw:
            return None
        try:
            return uuid.UUID(str(raw))
        except (ValueError, TypeError):
            return None  # not a DB-originated order (e.g. a rebalance correlation id)

    @staticmethod
    def _to_decimal(value: Any, default: str = "0") -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return Decimal(default)

    @staticmethod
    def _engine_order_id(event: Event) -> uuid.UUID | None:
        """The execution engine's order id, which since 2026-11-02 IS the DB row id."""
        raw = getattr(event, "order_id", "") or event.payload.get("order_id", "")
        try:
            return uuid.UUID(str(raw)) if raw else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _fill_id(event: Event) -> str | None:
        raw = getattr(event, "fill_id", "") or event.payload.get("fill_id", "")
        return str(raw)[:100] if raw else None

    async def _resolve_order(self, session: AsyncSession, event: Event) -> Order | None:
        """Find the durable row for a fill: by engine order id, then by client id."""
        for candidate in (self._engine_order_id(event), self._db_order_id(event)):
            if candidate is None:
                continue
            order = await session.get(Order, candidate)
            if order is not None:
                return order
        return None

    async def _apply_fill(self, event: Event) -> None:
        """Persist a fill exactly once.

        Delivery is at-least-once (Redis consumer groups redeliver anything
        whose handler crashed), so the venue fill id is the dedupe key. Errors
        are NOT swallowed: a failed write leaves the message pending so the
        bus redelivers it, and the dedupe makes that retry safe. The previous
        version caught everything and acked, which turned a transient DB error
        into a close permanently missing from the ledger.
        """
        fill_id = self._fill_id(event)
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            if fill_id is not None:
                dup = await session.execute(
                    select(Fill.id).where(Fill.external_fill_id == fill_id)
                )
                if dup.first() is not None:
                    logger.info("fill_duplicate_skipped", fill_id=fill_id)
                    return
            order = await self._resolve_order(session, event)
            if order is None:
                # No row written before submit (an engine without a session
                # factory, or an event from before 2026-11-02): record it now.
                await self._insert_autonomous_fill(event, session, fill_id)
                return
            order.status = OrderStatus.FILLED.value
            order.filled_at = now
            order.updated_at = now
            session.add(
                Fill(
                    order_id=order.id,
                    external_fill_id=fill_id,
                    price=self._to_decimal(getattr(event, "fill_price", 0.0)),
                    quantity=self._to_decimal(getattr(event, "fill_quantity", 0.0)),
                    commission=self._to_decimal(getattr(event, "commission", 0.0)),
                    filled_at=now,
                    created_at=now,
                )
            )
            await session.commit()
            logger.info("order_persisted_filled", order_id=str(order.id), fill_id=fill_id)

    async def _apply_status(self, event: Event, status: OrderStatus) -> None:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            order = await self._resolve_order(session, event)
            if order is None:
                return
            order.status = status.value
            order.updated_at = now
            await session.commit()
            logger.info("order_persisted_status", order_id=str(order.id), status=status.value)

    async def _insert_autonomous_fill(
        self, event: Event, session: AsyncSession, fill_id: str | None
    ) -> None:
        """Insert a filled order row for a fill that has no durable row yet.

        ``correlation_id`` stores the risk correlation id ("rebal-.."/
        "explore-.."/"liq-..") so learning trades stay identifiable;
        ``external_id`` keeps carrying it too for rows read by older code.
        """
        symbol = str(getattr(event, "symbol", "") or event.payload.get("symbol", ""))
        side = str(getattr(event, "side", "") or event.payload.get("side", ""))
        quantity = self._to_decimal(getattr(event, "fill_quantity", 0.0))
        price = self._to_decimal(getattr(event, "fill_price", 0.0))
        if not symbol or not side or quantity <= 0:
            return
        correlation = str(
            getattr(event, "client_order_id", "")
            or event.payload.get("client_order_id", "")
        )
        now = datetime.now(UTC)
        order = Order(
            external_id=correlation[:100] or None,
            correlation_id=correlation[:100] or None,
            symbol=symbol,
            asset_class="crypto" if "/" in symbol else "stock",
            side=side,
            order_type="market",
            quantity=quantity,
            status=OrderStatus.FILLED.value,
            trading_mode=self._trading_mode,
            filled_at=now,
            created_at=now,
            updated_at=now,
        )
        session.add(order)
        await session.flush()  # assign order.id for the fill row
        session.add(
            Fill(
                order_id=order.id,
                external_fill_id=fill_id,
                price=price,
                quantity=quantity,
                commission=self._to_decimal(getattr(event, "commission", 0.0)),
                filled_at=now,
                created_at=now,
            )
        )
        await session.commit()
        logger.info(
            "order_persisted_autonomous",
            symbol=symbol,
            side=side,
            correlation_id=correlation,
        )
