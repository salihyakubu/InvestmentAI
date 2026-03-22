"""Order request/response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field

from core.enums import OrderSide, OrderStatus, OrderType


class OrderCreate(BaseModel):
    """Payload to submit a new order."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None


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
