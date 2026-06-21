"""Validation tests for the OrderCreate request schema (input bounds)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from api.schemas.orders import OrderCreate
from core.enums import OrderSide, OrderType


def test_valid_market_order() -> None:
    order = OrderCreate(
        symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("1")
    )
    assert order.quantity == Decimal("1")


def test_quantity_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        OrderCreate(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=Decimal("0")
        )


def test_quantity_capped() -> None:
    with pytest.raises(ValidationError):
        OrderCreate(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1e12"),
        )


def test_limit_order_requires_limit_price() -> None:
    with pytest.raises(ValidationError):
        OrderCreate(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=Decimal("1")
        )


def test_stop_order_requires_stop_price() -> None:
    with pytest.raises(ValidationError):
        OrderCreate(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.STOP, quantity=Decimal("1")
        )


def test_negative_price_rejected() -> None:
    with pytest.raises(ValidationError):
        OrderCreate(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            limit_price=Decimal("-5"),
        )


def test_valid_limit_order() -> None:
    order = OrderCreate(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("2"),
        limit_price=Decimal("150.25"),
    )
    assert order.limit_price == Decimal("150.25")
