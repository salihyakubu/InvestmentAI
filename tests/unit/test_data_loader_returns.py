"""The training data loader must expose the real forward returns so the return
regressor trains on actual magnitudes -- not the fabricated {-0.01, 0, +0.01}
step the models fell back to when only class labels were available.
"""

from __future__ import annotations

import numpy as np

from services.prediction.training.data_loader import TrainingDataLoader


def test_load_training_data_exposes_real_forward_returns() -> None:
    loader = TrainingDataLoader(
        target_horizon_bars=2, up_threshold=0.005, down_threshold=-0.005
    )
    close = np.array([100.0, 101.0, 104.0, 103.0, 100.0, 99.0])
    features = np.arange(len(close) * 2, dtype=float).reshape(len(close), 2)

    ds = loader.load_training_data(features, close, feature_names=["a", "b"])

    assert ds.returns is not None
    # Aligned with X / y; length == len(close) - horizon.
    assert len(ds.returns) == len(ds.y) == len(ds.X) == len(close) - 2

    # The continuous forward returns: (close[i+2] - close[i]) / close[i].
    expected = (close[2:] - close[:-2]) / close[:-2]
    np.testing.assert_allclose(ds.returns, expected)

    # And NOT the fabricated 3-valued step.
    assert not set(np.round(ds.returns, 8)).issubset({-0.01, 0.0, 0.01})


def test_returns_align_with_labels() -> None:
    loader = TrainingDataLoader(
        target_horizon_bars=1, up_threshold=0.01, down_threshold=-0.01
    )
    close = np.array([100.0, 102.0, 101.0, 100.0])  # +2%, -0.98%, -0.99%
    features = np.zeros((len(close), 1))

    ds = loader.load_training_data(features, close, feature_names=["a"])

    # First forward return is +2% -> label long (2); the stored return is the
    # real magnitude, not a rounded step.
    assert ds.y[0] == 2
    assert abs(ds.returns[0] - 0.02) < 1e-9
