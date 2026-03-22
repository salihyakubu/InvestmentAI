"""Order and Fill ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.models.base import AsyncBase, UUIDMixin


class Order(UUIDMixin, AsyncBase):
    """Maps to the ``orders`` table."""

    __tablename__ = "orders"

    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    stop_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    trading_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("predictions.id"),
        nullable=True,
    )
    parent_order_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id"),
        nullable=True,
    )
    algo_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    filled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Relationships
    fills: Mapped[list["Fill"]] = relationship(
        "Fill", back_populates="order", lazy="selectin"
    )
    child_orders: Mapped[list["Order"]] = relationship(
        "Order", back_populates="parent_order", remote_side="Order.id"
    )
    parent_order: Mapped[Optional["Order"]] = relationship(
        "Order", back_populates="child_orders", remote_side="Order.parent_order_id"
    )

    def __repr__(self) -> str:
        return (
            f"<Order id={self.id} symbol={self.symbol!r} "
            f"side={self.side!r} status={self.status!r}>"
        )


class Fill(UUIDMixin, AsyncBase):
    """Maps to the ``fills`` table."""

    __tablename__ = "fills"

    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("orders.id"), nullable=False
    )
    external_fill_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    commission: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), nullable=False, server_default="0"
    )
    slippage_bps: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Relationships
    order: Mapped["Order"] = relationship("Order", back_populates="fills")

    def __repr__(self) -> str:
        return (
            f"<Fill id={self.id} order={self.order_id} "
            f"price={self.price} qty={self.quantity}>"
        )
