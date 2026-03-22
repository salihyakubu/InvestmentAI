"""Enumerations used across the InvestAI platform."""

from enum import StrEnum


class AssetClass(StrEnum):
    """Supported asset classes."""

    STOCK = "stock"
    CRYPTO = "crypto"


class OrderSide(StrEnum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"
    VWAP = "vwap"


class OrderStatus(StrEnum):
    """Order lifecycle states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradingMode(StrEnum):
    """Trading execution modes."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class SignalDirection(StrEnum):
    """Prediction signal directions."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TimeFrame(StrEnum):
    """Supported OHLCV timeframes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
