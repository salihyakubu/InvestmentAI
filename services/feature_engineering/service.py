"""Feature Engineering Service.

Subscribes to ``BarCloseEvent`` on the event bus, computes a full
feature vector for the symbol/timeframe, caches it in Redis, and
publishes a ``FeaturesReadyEvent`` so downstream consumers (ML models,
signal generators) can react.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from config.settings import Settings, get_settings
from core.events.base import Event, EventBus
from core.events.market_events import BarCloseEvent
from core.events.signal_events import FeaturesReadyEvent
from services.feature_engineering.feature_store import FeatureStore
from services.feature_engineering.feature_registry import FeatureRegistry
from services.feature_engineering.technical import indicators as ti

logger = logging.getLogger(__name__)

# Redis Stream / consumer-group names.
_STREAM_BARS = "market.bars"
_STREAM_FEATURES = "features.ready"
_CONSUMER_GROUP = "feature_engineering"
_CONSUMER_NAME = "fe_worker_1"


class FeatureEngineeringService:
    """Orchestrates feature computation in response to bar-close events.

    Parameters
    ----------
    event_bus : EventBus (Redis-backed or in-process)
    settings  : application settings
    feature_store : optional pre-built FeatureStore (created internally if
                    not provided)
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
        feature_store: FeatureStore | None = None,
    ) -> None:
        self._bus = event_bus
        self._settings = settings or get_settings()
        self._store = feature_store or FeatureStore()
        self._registry = FeatureRegistry()
        self._bar_buffers: dict[str, list[dict[str, Any]]] = {}
        self._register_default_features()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to ``BarCloseEvent`` and begin processing."""
        logger.info("FeatureEngineeringService starting")
        await self._bus.subscribe(
            stream=_STREAM_BARS,
            group=_CONSUMER_GROUP,
            consumer=_CONSUMER_NAME,
            handler=self._on_event,
        )

    async def _on_event(self, event: Event) -> None:
        """Route raw events to typed handlers."""
        if event.event_type == "BarCloseEvent":
            bar = BarCloseEvent.model_validate(event.model_dump())
            await self.handle_bar_close(bar)

    # ------------------------------------------------------------------
    # Bar-close handler
    # ------------------------------------------------------------------

    async def handle_bar_close(self, event: BarCloseEvent) -> None:
        """Ingest a new bar, recompute features, and emit downstream.

        Steps:
        1. Append the bar to the in-memory look-back buffer.
        2. Convert the buffer to a polars DataFrame.
        3. Compute the full feature vector via ``FeatureStore``.
        4. Cache to Redis.
        5. Publish ``FeaturesReadyEvent``.
        """
        key = f"{event.symbol}:{event.timeframe}"

        buf = self._bar_buffers.setdefault(key, [])
        buf.append(
            {
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "volume": event.volume,
                "vwap": event.vwap,
            }
        )

        # Keep only the most recent look-back bars.
        max_lookback = 200
        if len(buf) > max_lookback:
            self._bar_buffers[key] = buf[-max_lookback:]
            buf = self._bar_buffers[key]

        df = self._bars_to_df(buf)
        features = self._store.compute_all_features(event.symbol, df)

        if not features:
            logger.warning("No features computed for %s", key)
            return

        # Cache and publish.
        await self._store._cache_features(event.symbol, features)

        ready = FeaturesReadyEvent(
            symbol=event.symbol,
            feature_vector=features,
            source_service="feature_engineering",
        )
        await self._bus.publish(_STREAM_FEATURES, ready)
        logger.debug(
            "Published FeaturesReadyEvent for %s (%d features)",
            event.symbol,
            len(features),
        )

    # ------------------------------------------------------------------
    # Direct computation (non-event-driven)
    # ------------------------------------------------------------------

    def compute_features(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 200,
    ) -> dict[str, float]:
        """Compute features from the internal bar buffer.

        This method is useful for on-demand queries outside the
        event-driven pipeline.
        """
        key = f"{symbol}:{timeframe}"
        buf = self._bar_buffers.get(key)
        if not buf:
            logger.warning("No buffered bars for %s", key)
            return {}
        df = self._bars_to_df(buf[-lookback:])
        return self._store.compute_all_features(symbol, df)

    async def _load_recent_bars(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 200,
    ) -> pl.DataFrame:
        """Load recent bars from the database when the buffer is cold.

        Falls back to the in-memory buffer if no DB session exists.
        """
        if self._store._db is not None:
            try:
                from datetime import datetime, timedelta, timezone
                from sqlalchemy import text

                end = datetime.now(timezone.utc)
                start = end - timedelta(days=lookback)
                query = (
                    "SELECT open, high, low, close, volume, vwap "
                    "FROM ohlcv_bars "
                    "WHERE symbol = :symbol AND timeframe = :tf "
                    "  AND bar_time >= :start "
                    "ORDER BY bar_time "
                    "LIMIT :limit"
                )
                result = await self._store._db.execute(
                    text(query),
                    {
                        "symbol": symbol,
                        "tf": timeframe,
                        "start": start,
                        "limit": lookback,
                    },
                )
                rows = result.mappings().all()
                if rows:
                    return pl.DataFrame([dict(r) for r in rows])
            except Exception:
                logger.exception("Failed to load bars from DB for %s", symbol)

        # Fallback to in-memory buffer.
        key = f"{symbol}:{timeframe}"
        buf = self._bar_buffers.get(key, [])
        return self._bars_to_df(buf[-lookback:])

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _register_default_features(self) -> None:
        """Populate the feature registry with the standard indicator set."""
        _tech = "technical"
        for period in (5, 10, 20, 50, 200):
            self._registry.register(
                f"sma_{period}",
                lambda c, p=period: ti.sma(c, p),
                _tech,
                f"Simple Moving Average ({period})",
            )
            self._registry.register(
                f"ema_{period}",
                lambda c, p=period: ti.ema(c, p),
                _tech,
                f"Exponential Moving Average ({period})",
            )

        self._registry.register("rsi_14", lambda c: ti.rsi(c, 14), _tech, "RSI (14)")
        self._registry.register("macd_line", lambda c: ti.macd(c)[0], _tech, "MACD line")
        self._registry.register("macd_signal", lambda c: ti.macd(c)[1], _tech, "MACD signal")
        self._registry.register("macd_histogram", lambda c: ti.macd(c)[2], _tech, "MACD histogram")
        self._registry.register("bb_width", lambda c: ti.bollinger(c)[3], _tech, "Bollinger width")
        self._registry.register("bb_pct_b", lambda c: ti.bollinger(c)[4], _tech, "Bollinger %B")
        self._registry.register(
            "atr_14",
            lambda h, l, c: ti.atr(h, l, c, 14),
            _tech,
            "Average True Range (14)",
        )
        self._registry.register(
            "adx_14",
            lambda h, l, c: ti.adx(h, l, c, 14),
            _tech,
            "Average Directional Index (14)",
        )
        self._registry.register(
            "cci_20",
            lambda h, l, c: ti.cci(h, l, c, 20),
            _tech,
            "Commodity Channel Index (20)",
        )
        self._registry.register("roc_12", lambda c: ti.roc(c, 12), _tech, "Rate of Change (12)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_df(bars: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert a list of bar dicts to a polars DataFrame."""
        if not bars:
            return pl.DataFrame()

        columns: dict[str, list[Any]] = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        has_vwap = any(b.get("vwap") is not None for b in bars)
        if has_vwap:
            columns["vwap"] = []

        for b in bars:
            columns["open"].append(b["open"])
            columns["high"].append(b["high"])
            columns["low"].append(b["low"])
            columns["close"].append(b["close"])
            columns["volume"].append(b["volume"])
            if has_vwap:
                columns["vwap"].append(b.get("vwap") or 0.0)

        return pl.DataFrame(columns)
