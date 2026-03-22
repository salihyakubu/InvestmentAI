"""Unit tests for technical analysis indicators."""

from __future__ import annotations

import numpy as np
import pytest

from services.feature_engineering.technical.indicators import (
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
)


class TestSMA:
    """Tests for the simple moving average indicator."""

    def test_sma_basic(self, sample_ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """SMA of period *p* should equal the mean of the last *p* values."""
        close = sample_ohlcv_arrays["close"]
        period = 10
        result = sma(close, period)

        # First (period - 1) values must be NaN.
        assert np.all(np.isnan(result[: period - 1]))

        # Value at index (period - 1) should equal the mean of the first
        # *period* elements.
        expected = np.mean(close[:period])
        assert pytest.approx(result[period - 1], rel=1e-10) == expected

        # Spot-check the last value.
        expected_last = np.mean(close[-period:])
        assert pytest.approx(result[-1], rel=1e-10) == expected_last


class TestEMA:
    """Tests for the exponential moving average indicator."""

    def test_ema_basic(self, sample_ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """EMA seed should equal SMA of the first *period* values,
        and subsequent values should follow the recursive formula."""
        close = sample_ohlcv_arrays["close"]
        period = 10
        result = ema(close, period)

        # First (period - 1) values must be NaN.
        assert np.all(np.isnan(result[: period - 1]))

        # Seed value should equal SMA.
        seed = np.mean(close[:period])
        assert pytest.approx(result[period - 1], rel=1e-10) == seed

        # Check recursive formula for the next value.
        alpha = 2.0 / (period + 1)
        expected_next = alpha * close[period] + (1 - alpha) * seed
        assert pytest.approx(result[period], rel=1e-10) == expected_next


class TestRSI:
    """Tests for the Relative Strength Index."""

    def test_rsi_range(self, sample_ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """RSI values must always be between 0 and 100 (inclusive)."""
        close = sample_ohlcv_arrays["close"]
        result = rsi(close, 14)

        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_rsi_known_values(self) -> None:
        """Verify RSI against a manually computed example.

        A monotonically increasing series should produce RSI = 100
        (all gains, no losses).
        """
        close = np.arange(1.0, 30.0, dtype=np.float64)
        result = rsi(close, 14)
        valid = result[~np.isnan(result)]

        # All gains, zero losses => RSI = 100
        assert np.all(valid == 100.0)


class TestMACD:
    """Tests for the MACD indicator."""

    def test_macd_signal_cross(self, sample_ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """Verify that MACD line and signal line cross at least once
        over 100 bars of noisy data, and the histogram reflects the
        difference correctly."""
        close = sample_ohlcv_arrays["close"]
        macd_line, signal_line, histogram = macd(close)

        # The histogram should always equal macd_line - signal_line
        # where both are non-NaN.
        valid_mask = ~np.isnan(macd_line) & ~np.isnan(signal_line)
        expected_hist = macd_line[valid_mask] - signal_line[valid_mask]
        np.testing.assert_allclose(
            histogram[valid_mask], expected_hist, atol=1e-12
        )

        # There should be at least one sign change (crossover) in the
        # histogram.
        valid_hist = histogram[valid_mask]
        sign_changes = np.diff(np.sign(valid_hist))
        assert np.any(sign_changes != 0), "Expected at least one MACD crossover"


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_bands_contain_price(
        self, sample_ohlcv_arrays: dict[str, np.ndarray]
    ) -> None:
        """Price should fall within the bands the vast majority of
        the time (at least 85% for 2-std bands)."""
        close = sample_ohlcv_arrays["close"]
        upper, middle, lower, width, pct_b = bollinger(close, period=20, num_std=2.0)

        # Only check where bands are defined (after warm-up).
        valid_mask = ~np.isnan(upper) & ~np.isnan(lower)
        within = (close[valid_mask] >= lower[valid_mask]) & (
            close[valid_mask] <= upper[valid_mask]
        )
        containment_pct = np.sum(within) / np.sum(valid_mask)
        assert containment_pct >= 0.85, (
            f"Expected >=85% containment, got {containment_pct:.1%}"
        )


class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self, sample_ohlcv_arrays: dict[str, np.ndarray]) -> None:
        """ATR values should always be positive (volatility cannot be
        negative)."""
        high = sample_ohlcv_arrays["high"]
        low = sample_ohlcv_arrays["low"]
        close = sample_ohlcv_arrays["close"]

        result = atr(high, low, close, period=14)
        valid = result[~np.isnan(result)]

        assert len(valid) > 0
        assert np.all(valid > 0.0)
