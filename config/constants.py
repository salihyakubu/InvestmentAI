"""Trading constants for the AI investment platform."""

from dataclasses import dataclass
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Exchange hours
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExchangeHours:
    name: str
    timezone: ZoneInfo
    open_hour: int
    open_minute: int
    close_hour: int
    close_minute: int
    pre_market_open_hour: int = 4
    pre_market_open_minute: int = 0
    after_hours_close_hour: int = 20
    after_hours_close_minute: int = 0


STOCK_EXCHANGES: dict[str, ExchangeHours] = {
    "NYSE": ExchangeHours(
        name="New York Stock Exchange",
        timezone=ZoneInfo("America/New_York"),
        open_hour=9,
        open_minute=30,
        close_hour=16,
        close_minute=0,
    ),
    "NASDAQ": ExchangeHours(
        name="NASDAQ",
        timezone=ZoneInfo("America/New_York"),
        open_hour=9,
        open_minute=30,
        close_hour=16,
        close_minute=0,
    ),
}

CRYPTO_24_7: bool = True

# ---------------------------------------------------------------------------
# Fee structures
# ---------------------------------------------------------------------------

FEES = {
    "alpaca": {
        "commission": 0.0,          # commission-free equity trades
        "sec_fee_per_dollar": 8e-6, # SEC fee ~$8 per million
        "taf_fee_per_share": 0.000119,
        "min_fee": 0.0,
    },
    "binance": {
        "maker_fee": 0.001,         # 0.10 %
        "taker_fee": 0.001,         # 0.10 %
        "bnb_discount": 0.25,       # 25 % discount when paying fees in BNB
    },
}

# ---------------------------------------------------------------------------
# Default timeframes
# ---------------------------------------------------------------------------

TIMEFRAMES = {
    "scalp": "1m",
    "intraday": "5m",
    "swing": "1h",
    "position": "1d",
}

DEFAULT_LOOKBACK_BARS: int = 500
DEFAULT_WARMUP_BARS: int = 50

# ---------------------------------------------------------------------------
# ML feature names (canonical ordering)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    # Price-derived
    "return_1",
    "return_5",
    "return_10",
    "return_20",
    "log_return_1",
    # Volatility
    "volatility_10",
    "volatility_20",
    "atr_14",
    # Momentum
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "adx_14",
    "cci_20",
    "williams_r_14",
    # Trend
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_10",
    "ema_20",
    "ema_50",
    "price_to_sma_20",
    "price_to_sma_50",
    # Volume
    "volume_ratio_20",
    "obv",
    "vwap",
    # Microstructure
    "bid_ask_spread",
    "order_imbalance",
    # Sentiment
    "news_sentiment",
    "social_sentiment",
]

# ---------------------------------------------------------------------------
# Order / position constants
# ---------------------------------------------------------------------------

ORDER_TYPES = ("market", "limit", "stop", "stop_limit")
POSITION_SIDES = ("long", "short")
ASSET_CLASSES = ("stock", "crypto")
