"""Factory Boy factories for generating test model instances.

These factories produce plain dataclass / ORM-like objects suitable for
unit and integration tests without requiring a live database connection.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import factory

from core.enums import AssetClass, OrderSide, OrderStatus, OrderType, TradingMode


# ---------------------------------------------------------------------------
# OHLCV Factory
# ---------------------------------------------------------------------------


class OHLCVFactory(factory.Factory):
    """Generates OHLCV bar dictionaries for market data tests."""

    class Meta:
        model = dict

    time = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    symbol = "AAPL"
    timeframe = "1d"
    asset_class = AssetClass.STOCK
    open = factory.LazyFunction(lambda: Decimal("150.00"))
    high = factory.LazyFunction(lambda: Decimal("152.50"))
    low = factory.LazyFunction(lambda: Decimal("149.00"))
    close = factory.LazyFunction(lambda: Decimal("151.75"))
    volume = factory.LazyFunction(lambda: Decimal("50000000"))
    vwap = factory.LazyFunction(lambda: Decimal("151.00"))
    trade_count = 125000
    source = "test"
    ingested_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Order Factory
# ---------------------------------------------------------------------------


class OrderFactory(factory.Factory):
    """Generates Order-like dictionaries matching the ``orders`` table schema."""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: uuid.uuid4())
    external_id = None
    symbol = "AAPL"
    asset_class = AssetClass.STOCK
    side = OrderSide.BUY
    order_type = OrderType.MARKET
    quantity = factory.LazyFunction(lambda: Decimal("10"))
    limit_price = None
    stop_price = None
    status = OrderStatus.PENDING
    trading_mode = TradingMode.PAPER
    prediction_id = None
    parent_order_id = None
    algo_type = None
    submitted_at = None
    filled_at = None
    cancelled_at = None
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Fill Factory
# ---------------------------------------------------------------------------


class FillFactory(factory.Factory):
    """Generates Fill-like dictionaries matching the ``fills`` table schema."""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: uuid.uuid4())
    order_id = factory.LazyFunction(lambda: uuid.uuid4())
    external_fill_id = None
    price = factory.LazyFunction(lambda: Decimal("151.50"))
    quantity = factory.LazyFunction(lambda: Decimal("10"))
    commission = factory.LazyFunction(lambda: Decimal("0.00"))
    slippage_bps = 1.5
    filled_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    created_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Position Factory
# ---------------------------------------------------------------------------


class PositionFactory(factory.Factory):
    """Generates Position-like dictionaries matching the ``positions`` table."""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: uuid.uuid4())
    symbol = "AAPL"
    asset_class = AssetClass.STOCK
    side = OrderSide.BUY
    quantity = factory.LazyFunction(lambda: Decimal("10"))
    avg_entry_price = factory.LazyFunction(lambda: Decimal("150.00"))
    current_price = factory.LazyFunction(lambda: Decimal("152.00"))
    unrealized_pnl = factory.LazyFunction(lambda: Decimal("20.00"))
    realized_pnl = factory.LazyFunction(lambda: Decimal("0.00"))
    stop_loss_price = None
    trailing_stop_pct = None
    highest_price = None
    opened_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    closed_at = None
    updated_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Prediction Factory
# ---------------------------------------------------------------------------


class PredictionFactory(factory.Factory):
    """Generates prediction result dictionaries for ML pipeline tests."""

    class Meta:
        model = dict

    prediction_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    symbol = "AAPL"
    direction = "long"
    confidence = 0.75
    expected_return = 0.02
    model_id = "xgboost_v1"
    timestamp = factory.LazyFunction(lambda: datetime.now(timezone.utc))
