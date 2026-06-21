"""Persist execution outcomes back to the database.

The execution engine runs on in-memory order state and emits lifecycle events
on the ``orders`` stream. This service subscribes to those events and, keyed by
the ``client_order_id`` (the originating DB order id), updates the DB order row's
status and records fills -- so a manually-submitted order's row reflects what
actually happened. Reporting/audit only; the trade itself and risk/position
tracking already react to fills directly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog
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
    ) -> None:
        self._event_bus = event_bus
        self._session_factory = session_factory

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

    async def _apply_fill(self, event: Event) -> None:
        order_id = self._db_order_id(event)
        if order_id is None:
            return
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            order = await session.get(Order, order_id)
            if order is None:
                return
            order.status = OrderStatus.FILLED.value
            order.filled_at = now
            order.updated_at = now
            session.add(
                Fill(
                    order_id=order.id,
                    price=self._to_decimal(getattr(event, "fill_price", 0.0)),
                    quantity=self._to_decimal(getattr(event, "fill_quantity", 0.0)),
                    commission=self._to_decimal(getattr(event, "commission", 0.0)),
                    filled_at=now,
                    created_at=now,
                )
            )
            await session.commit()
            logger.info("order_persisted_filled", order_id=str(order_id))

    async def _apply_status(self, event: Event, status: OrderStatus) -> None:
        order_id = self._db_order_id(event)
        if order_id is None:
            return
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            order = await session.get(Order, order_id)
            if order is None:
                return
            order.status = status.value
            order.updated_at = now
            await session.commit()
            logger.info("order_persisted_status", order_id=str(order_id), status=status.value)
