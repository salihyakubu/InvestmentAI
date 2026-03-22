"""Feature caching and retrieval layer.

Combines Redis for hot (real-time) features and a database for
historical training data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from services.feature_engineering.technical import indicators as ti
from services.feature_engineering.technical.patterns import detect_patterns
from services.feature_engineering.technical.microstructure import (
    price_momentum,
    volatility_regime,
    volume_profile,
)

logger = logging.getLogger(__name__)

_FEATURE_CACHE_TTL_SECONDS = 300  # 5 minutes


class FeatureStore:
    """Compute, cache, and retrieve feature vectors.

    Parameters
    ----------
    db_session : async SQLAlchemy session (or None during tests)
    redis      : aioredis connection (or None during tests)
    """

    def __init__(self, db_session: Any = None, redis: Any = None) -> None:
        self._db = db_session
        self._redis = redis

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        symbol: str,
        ohlcv_df: pl.DataFrame,
    ) -> dict[str, float]:
        """Compute the full feature vector from an OHLCV polars DataFrame.

        Expected columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
        An optional ``vwap`` column is used when available.

        Returns a flat ``{feature_name: value}`` dict.
        """
        if ohlcv_df.is_empty():
            logger.warning("Empty OHLCV data for %s – returning empty features", symbol)
            return {}

        o = ohlcv_df["open"].to_numpy().astype(np.float64)
        h = ohlcv_df["high"].to_numpy().astype(np.float64)
        l = ohlcv_df["low"].to_numpy().astype(np.float64)  # noqa: E741
        c = ohlcv_df["close"].to_numpy().astype(np.float64)
        v = ohlcv_df["volume"].to_numpy().astype(np.float64)

        features: dict[str, float] = {}

        # ---- Moving averages ------------------------------------------------
        for period in (5, 10, 20, 50, 200):
            _put_last(features, f"sma_{period}", ti.sma(c, period))
            _put_last(features, f"ema_{period}", ti.ema(c, period))

        # ---- MACD ------------------------------------------------------------
        macd_line, signal_line, histogram = ti.macd(c)
        _put_last(features, "macd_line", macd_line)
        _put_last(features, "macd_signal", signal_line)
        _put_last(features, "macd_histogram", histogram)

        # ---- RSI -------------------------------------------------------------
        _put_last(features, "rsi_14", ti.rsi(c, 14))

        # ---- Bollinger Bands -------------------------------------------------
        bb_upper, bb_mid, bb_lower, bb_width, bb_pctb = ti.bollinger(c)
        _put_last(features, "bb_upper", bb_upper)
        _put_last(features, "bb_middle", bb_mid)
        _put_last(features, "bb_lower", bb_lower)
        _put_last(features, "bb_width", bb_width)
        _put_last(features, "bb_pct_b", bb_pctb)

        # ---- ATR -------------------------------------------------------------
        _put_last(features, "atr_14", ti.atr(h, l, c, 14))

        # ---- Stochastic -----------------------------------------------------
        stoch_k, stoch_d = ti.stochastic(h, l, c)
        _put_last(features, "stoch_k", stoch_k)
        _put_last(features, "stoch_d", stoch_d)

        # ---- Williams %R ----------------------------------------------------
        _put_last(features, "williams_r_14", ti.williams_r(h, l, c, 14))

        # ---- OBV -------------------------------------------------------------
        _put_last(features, "obv", ti.obv(c, v))

        # ---- VWAP deviation --------------------------------------------------
        if "vwap" in ohlcv_df.columns:
            vwap = ohlcv_df["vwap"].to_numpy().astype(np.float64)
            _put_last(features, "vwap_deviation", ti.vwap_deviation(c, v, vwap))

        # ---- Volume / SMA ratio ----------------------------------------------
        _put_last(features, "volume_sma_ratio_20", ti.volume_sma_ratio(v, 20))

        # ---- ADX -------------------------------------------------------------
        _put_last(features, "adx_14", ti.adx(h, l, c, 14))

        # ---- CCI -------------------------------------------------------------
        _put_last(features, "cci_20", ti.cci(h, l, c, 20))

        # ---- MFI -------------------------------------------------------------
        _put_last(features, "mfi_14", ti.mfi(h, l, c, v, 14))

        # ---- ROC -------------------------------------------------------------
        _put_last(features, "roc_12", ti.roc(c, 12))

        # ---- Historical volatility -------------------------------------------
        for period in (10, 30, 60):
            _put_last(features, f"hist_vol_{period}", ti.hist_volatility(c, period))

        # ---- Candlestick patterns --------------------------------------------
        patterns = detect_patterns(o, h, l, c)
        for name, score in patterns.items():
            features[f"pattern_{name}"] = score

        # ---- Microstructure --------------------------------------------------
        momentum = price_momentum(c)
        features.update(momentum)

        vol_regime = volatility_regime(c)
        features["volatility_regime_encoded"] = {"low": 0.0, "medium": 0.5, "high": 1.0}.get(
            vol_regime, 0.5
        )

        vp = volume_profile(c, v)
        features["vp_value_area_high"] = vp["value_area_high"]
        features["vp_value_area_low"] = vp["value_area_low"]
        features["vp_poc"] = vp["poc"]

        # ---- Price-derived helpers -------------------------------------------
        if len(c) > 0:
            features["close"] = float(c[-1])
            features["daily_return"] = float((c[-1] / c[-2] - 1.0) if len(c) > 1 else 0.0)

        # Replace any remaining NaN with 0 for downstream consumers.
        for k, val in features.items():
            if isinstance(val, float) and np.isnan(val):
                features[k] = 0.0

        return features

    # ------------------------------------------------------------------
    # Cache (Redis)
    # ------------------------------------------------------------------

    async def get_latest(
        self,
        symbol: str,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Retrieve the latest cached feature vector from Redis.

        Returns an empty dict if nothing is cached or Redis is unavailable.
        """
        if self._redis is None:
            return {}

        key = f"features:{symbol}:latest"
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return {}
            data: dict[str, float] = json.loads(raw)
            if feature_names is not None:
                return {k: data[k] for k in feature_names if k in data}
            return data
        except Exception:
            logger.exception("Redis get_latest failed for %s", symbol)
            return {}

    async def _cache_features(self, symbol: str, features: dict[str, float]) -> None:
        """Write a feature vector to Redis with a TTL."""
        if self._redis is None:
            return

        key = f"features:{symbol}:latest"
        try:
            payload = json.dumps(features)
            await self._redis.set(key, payload, ex=_FEATURE_CACHE_TTL_SECONDS)
            logger.debug("Cached %d features for %s", len(features), symbol)
        except Exception:
            logger.exception("Redis cache_features failed for %s", symbol)

    # ------------------------------------------------------------------
    # Historical / training data (DB)
    # ------------------------------------------------------------------

    async def get_training_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Load historical feature rows from the database.

        Returns a polars DataFrame with one row per timestamp.
        Falls back to an empty frame when no DB session is configured.
        """
        if self._db is None:
            logger.warning(
                "No DB session – returning empty training data for %s", symbol
            )
            return pl.DataFrame()

        try:
            query = (
                "SELECT * FROM feature_snapshots "
                "WHERE symbol = :symbol AND timestamp >= :start AND timestamp <= :end "
                "ORDER BY timestamp"
            )
            from sqlalchemy import text

            result = await self._db.execute(
                text(query),
                {"symbol": symbol, "start": start, "end": end},
            )
            rows = result.mappings().all()
            if not rows:
                return pl.DataFrame()
            return pl.DataFrame([dict(r) for r in rows])
        except Exception:
            logger.exception("DB get_training_data failed for %s", symbol)
            return pl.DataFrame()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _put_last(target: dict[str, float], key: str, arr: np.ndarray) -> None:
    """Store the last element of *arr* into *target[key]*, or NaN."""
    if len(arr) > 0:
        target[key] = float(arr[-1])
    else:
        target[key] = float("nan")
