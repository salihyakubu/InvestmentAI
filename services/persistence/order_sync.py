"""Persist execution outcomes back to the database.

The execution engine runs on in-memory order state and emits lifecycle events
on the ``orders`` stream. This service subscribes to those events and persists
them: fills whose ``client_order_id`` is a DB order id (manually-submitted
orders, which pre-create their row via the API) UPDATE that row; fills carrying
a rebalance/exploration correlation id (autonomous orders, which have no
pre-created row) INSERT a completed row so the Trading page and audit trail see
every trade the platform makes. Reporting/audit only; the trade itself and
risk/position tracking already react to fills directly.
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

    async def _apply_fill(self, event: Event) -> None:
        order_id = self._db_order_id(event)
        if order_id is None:
            # Autonomous order (rebalance/exploration correlation id, not a DB
            # row id): insert a completed row so it exists in the audit trail.
            await self._insert_autonomous_fill(event)
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

    async def _insert_autonomous_fill(self, event: Event) -> None:
        """Insert a filled order row for an autonomous (non-API) order.

        ``external_id`` stores the risk correlation id ("rebal-.."/"explore-..")
        so exploration learning trades stay identifiable in the orders table.
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
        try:
            async with self._session_factory() as session:
                order = Order(
                    external_id=correlation[:100] or None,
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
        except Exception:
            # Persistence must never kill the consumer loop.
            logger.exception("order_persist_autonomous_failed", symbol=symbol)
