"""Order request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from core.enums import OrderSide, OrderStatus, OrderType

# Sanity / overflow guard on order size. The real economic guard is the
# buying-power check in the execution engine; this just rejects absurd values.
_MAX_ORDER_QUANTITY = Decimal("1_000_000_000")


class OrderCreate(BaseModel):
    """Payload to submit a new order."""

    symbol: str = Field(min_length=1, max_length=20)
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0, le=_MAX_ORDER_QUANTITY)
    limit_price: Decimal | None = Field(default=None, gt=0)
    stop_price: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_prices_for_type(self) -> OrderCreate:
        """Limit/stop orders must carry the price that defines them."""
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError("limit_price is required for limit and stop_limit orders")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("stop_price is required for stop and stop_limit orders")
        return self


class OrderResponse(BaseModel):
    """Full order representation returned by the API."""

    id: uuid.UUID
    external_id: str | None = None
    symbol: str
    asset_class: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus
    trading_mode: str
    prediction_id: uuid.UUID | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    cancelled_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class OrderListResponse(BaseModel):
    """Paginated list of orders."""

    items: list[OrderResponse]
    total: int
    page: int
    page_size: int
