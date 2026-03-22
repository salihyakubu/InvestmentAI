"""Unit tests for feature engineering computation."""

from __future__ import annotations

import numpy as np
import pytest


class TestFeatureComputation:
    """Tests for the FeatureStore.compute_all_features method."""

    def test_compute_all_features_returns_expected_keys(
        self, sample_features: dict[str, float]
    ) -> None:
        """The feature vector should contain standard technical
        indicator keys."""
        expected_keys = [
            "sma_5",
            "sma_20",
            "ema_10",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_lower",
            "bb_width",
            "atr_14",
            "close",
            "daily_return",
        ]

        for key in expected_keys:
            assert key in sample_features, f"Missing expected feature: {key}"

    def test_features_not_nan(
        self, sample_features: dict[str, float]
    ) -> None:
        """No feature values should be NaN after compute_all_features
        replaces them with 0.0."""
        for key, value in sample_features.items():
            assert not (isinstance(value, float) and np.isnan(value)), (
                f"Feature {key} is NaN"
            )

    def test_feature_values_reasonable(
        self, sample_features: dict[str, float]
    ) -> None:
        """Spot-check that feature values are within reasonable ranges."""
        # RSI should be between 0 and 100.
        rsi = sample_features.get("rsi_14", 0.0)
        assert 0.0 <= rsi <= 100.0, f"RSI out of range: {rsi}"

        # Close price should be around $150 (our sample data starts there).
        close = sample_features.get("close", 0.0)
        assert 100.0 < close < 300.0, f"Close price out of range: {close}"

        # ATR should be positive.
        atr = sample_features.get("atr_14", 0.0)
        assert atr >= 0.0, f"ATR is negative: {atr}"

        # Daily return should be small (within +/-10%).
        daily_ret = sample_features.get("daily_return", 0.0)
        assert -0.10 < daily_ret < 0.10, (
            f"Daily return out of range: {daily_ret}"
        )
