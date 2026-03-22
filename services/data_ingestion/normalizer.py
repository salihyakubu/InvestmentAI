"""Normalise raw provider bars into a canonical format for DB storage."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from core.enums import AssetClass
from services.data_ingestion.providers.base import RawBar

logger = structlog.get_logger(__name__)


class NormalizationError(Exception):
    """Raised when a bar cannot be normalised."""


def _standardize_symbol(symbol: str, asset_class: AssetClass) -> str:
    """Convert provider-specific symbol formats to a canonical form.

    Stocks  : uppercase ticker, no change  (``AAPL`` -> ``AAPL``)
    Crypto  : slash-separated pair          (``BTC/USDT``, ``BTC-USDT`` -> ``BTC/USDT``)
    """
    symbol = symbol.strip().upper()

    if asset_class == AssetClass.CRYPTO:
        # Normalise all separators to "/".
        symbol = re.sub(r"[-_.]", "/", symbol)
        # Remove duplicate slashes.
        symbol = re.sub(r"/+", "/", symbol)
    return symbol


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone.  Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_bar(
    raw_bar: RawBar,
    asset_class: AssetClass,
    source: str,
    timeframe: str,
) -> dict:
    """Transform a :class:`RawBar` into a dict suitable for DB insertion.

    Performs the following:
    - Symbol format standardisation.
    - Timezone enforcement (UTC).
    - Basic sanity validation (positive prices, non-negative volume).

    Returns a ``dict`` whose keys match :class:`core.models.market_data.OHLCVRecord`.

    Raises
    ------
    NormalizationError
        If a bar contains fundamentally invalid data (e.g. negative price).
    """
    symbol = _standardize_symbol(raw_bar.symbol, asset_class)
    bar_time = _ensure_utc(raw_bar.time)
    now_utc = datetime.now(timezone.utc)

    # ----- Sanity checks -----
    for field_name, value in [
        ("open", raw_bar.open),
        ("high", raw_bar.high),
        ("low", raw_bar.low),
        ("close", raw_bar.close),
    ]:
        if value <= 0:
            raise NormalizationError(
                f"Non-positive {field_name} price ({value}) for {symbol} at {bar_time}"
            )

    if raw_bar.volume < 0:
        raise NormalizationError(
            f"Negative volume ({raw_bar.volume}) for {symbol} at {bar_time}"
        )

    return {
        "time": bar_time,
        "symbol": symbol,
        "asset_class": str(asset_class),
        "timeframe": timeframe,
        "open": Decimal(str(raw_bar.open)),
        "high": Decimal(str(raw_bar.high)),
        "low": Decimal(str(raw_bar.low)),
        "close": Decimal(str(raw_bar.close)),
        "volume": Decimal(str(raw_bar.volume)),
        "vwap": Decimal(str(raw_bar.vwap)) if raw_bar.vwap is not None else None,
        "trade_count": raw_bar.trade_count,
        "source": source,
        "ingested_at": now_utc,
    }
