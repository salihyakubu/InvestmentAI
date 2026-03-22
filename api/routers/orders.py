"""Order management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.enums import OrderStatus
from core.models.orders import Order
from api.dependencies import get_current_user, get_db
from api.schemas.orders import OrderCreate, OrderListResponse, OrderResponse

router = APIRouter(prefix="/orders")


@router.get(
    "",
    response_model=OrderListResponse,
    summary="List orders with filters",
)
async def list_orders(
    symbol: str | None = Query(default=None),
    status_filter: OrderStatus | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> OrderListResponse:
    """Return a paginated list of orders, optionally filtered."""
    stmt = select(Order).order_by(Order.created_at.desc())

    if symbol is not None:
        stmt = stmt.where(Order.symbol == symbol)
    if status_filter is not None:
        stmt = stmt.where(Order.status == status_filter.value)

    # Count
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Page
    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return OrderListResponse(
        items=[OrderResponse.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{order_id}",
    response_model=OrderResponse,
    summary="Get order details",
)
async def get_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> OrderResponse:
    """Return a single order by its ID."""
    stmt = select(Order).where(Order.id == order_id)
    result = await db.execute(stmt)
    order = result.scalar_one_or_none()

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found",
        )

    return OrderResponse.model_validate(order)


@router.post(
    "",
    response_model=OrderResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a manual order",
)
async def create_order(
    payload: OrderCreate,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> OrderResponse:
    """Submit a new manual order.

    The order will be validated against risk rules before execution.
    """
    # TODO: Integrate with execution service and risk checks
    now = datetime.now(timezone.utc)
    order = Order(
        symbol=payload.symbol,
        asset_class="stock",  # TODO: infer from symbol
        side=payload.side.value,
        order_type=payload.order_type.value,
        quantity=payload.quantity,
        limit_price=payload.limit_price,
        stop_price=payload.stop_price,
        status=OrderStatus.PENDING.value,
        trading_mode="paper",  # TODO: from settings
        created_at=now,
        updated_at=now,
    )
    db.add(order)
    await db.flush()
    await db.refresh(order)

    return OrderResponse.model_validate(order)


@router.delete(
    "/{order_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel an order",
)
async def cancel_order(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> None:
    """Cancel a pending or submitted order."""
    stmt = select(Order).where(Order.id == order_id)
    result = await db.execute(stmt)
    order = result.scalar_one_or_none()

    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found",
        )

    if order.status not in (OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel order with status: {order.status}",
        )

    order.status = OrderStatus.CANCELLED.value
    order.cancelled_at = datetime.now(timezone.utc)
    order.updated_at = datetime.now(timezone.utc)
    await db.flush()
