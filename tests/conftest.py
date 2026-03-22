"""Shared test fixtures for the InvestAI test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pytest

from config.settings import Settings
from core.events.base import InProcessEventBus


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> InProcessEventBus:
    """Return an in-memory event bus that does not require Redis."""
    return InProcessEventBus()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Return a ``Settings`` instance with test-friendly defaults.

    No ``.env`` file is loaded; values are explicit so tests are
    deterministic and independent of the developer's environment.
    """
    return Settings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_investai",
        redis_url="redis://localhost:6379/1",
        alpaca_api_key="test-key",
        alpaca_secret_key="test-secret",
        alpaca_base_url="https://paper-api.alpaca.markets",
        binance_api_key="test-binance-key",
        binance_secret_key="test-binance-secret",
        trading_mode="paper",
        initial_capital=Decimal("10000.00"),
        active_symbols_stocks=["AAPL", "MSFT", "GOOGL"],
        active_symbols_crypto=["BTC/USDT", "ETH/USDT"],
        max_position_pct=0.10,
        max_sector_pct=0.30,
        max_asset_class_pct=0.70,
        max_single_order_pct=0.05,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.15,
        max_pairwise_correlation=0.85,
        max_portfolio_positions=10,
        max_portfolio_var_95=0.03,
        circuit_breaker_loss_pct=0.07,
        circuit_breaker_cooldown_minutes=30,
        model_artifact_path="model_artifacts",
        retrain_interval_hours=168,
        min_prediction_confidence=0.6,
        ensemble_min_agreement=3,
        jwt_secret="test-jwt-secret",
        jwt_algorithm="HS256",
        jwt_expire_minutes=60,
        api_rate_limit=100,
        prometheus_enabled=False,
    )


# ---------------------------------------------------------------------------
# Sample OHLCV data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_arrays() -> dict[str, np.ndarray]:
    """Generate 100 days of realistic OHLCV data for AAPL.

    Prices start around $150 and drift upward with daily noise.
    Volume is randomised around 50 million shares.
    """
    rng = np.random.default_rng(seed=42)
    n = 100

    # Random walk for close prices starting at $150.
    daily_returns = rng.normal(loc=0.0005, scale=0.015, size=n)
    close = 150.0 * np.cumprod(1.0 + daily_returns)

    # Derive OHLV from close.
    high = close * (1.0 + rng.uniform(0.001, 0.02, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.02, size=n))
    open_ = close * (1.0 + rng.uniform(-0.01, 0.01, size=n))
    volume = rng.uniform(30e6, 70e6, size=n)
    vwap = (high + low + close) / 3.0

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
    }


@pytest.fixture
def sample_ohlcv_bars(sample_ohlcv_arrays: dict[str, np.ndarray]) -> list[dict]:
    """Return OHLCV data as a list of bar dicts suitable for FeatureStore."""
    n = len(sample_ohlcv_arrays["close"])
    bars: list[dict] = []
    for i in range(n):
        bars.append(
            {
                "open": float(sample_ohlcv_arrays["open"][i]),
                "high": float(sample_ohlcv_arrays["high"][i]),
                "low": float(sample_ohlcv_arrays["low"][i]),
                "close": float(sample_ohlcv_arrays["close"][i]),
                "volume": float(sample_ohlcv_arrays["volume"][i]),
                "vwap": float(sample_ohlcv_arrays["vwap"][i]),
            }
        )
    return bars


# ---------------------------------------------------------------------------
# Sample features
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features(sample_ohlcv_arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute a realistic feature vector from the sample OHLCV data."""
    import polars as pl

    from services.feature_engineering.feature_store import FeatureStore

    df = pl.DataFrame(
        {
            "open": sample_ohlcv_arrays["open"].tolist(),
            "high": sample_ohlcv_arrays["high"].tolist(),
            "low": sample_ohlcv_arrays["low"].tolist(),
            "close": sample_ohlcv_arrays["close"].tolist(),
            "volume": sample_ohlcv_arrays["volume"].tolist(),
            "vwap": sample_ohlcv_arrays["vwap"].tolist(),
        }
    )

    store = FeatureStore()
    return store.compute_all_features("AAPL", df)
