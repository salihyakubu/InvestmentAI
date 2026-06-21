"""Sequence training data: features must be returned RAW (so the LSTM/Transformer
own normalisation end-to-end with no double-scaling), and window/label alignment
must not leak the future into the model inputs.
"""

from __future__ import annotations

import numpy as np

from services.prediction.training.data_loader import TrainingDataLoader


def test_sequence_loader_returns_raw_unscaled_features() -> None:
    loader = TrainingDataLoader(target_horizon_bars=2)
    rng = np.random.default_rng(0)
    n, n_features = 120, 3
    # Features centred far from zero: had they been z-scored the mean would be ~0.
    features = rng.normal(loc=50.0, scale=4.0, size=(n, n_features))
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, size=n))

    ds = loader.load_sequence_data(features, close, seq_length=10)

    assert ds.scaler is None  # no scaler handed out -> model owns normalisation
    assert ds.X.ndim == 3
    assert ds.X.shape[1] == 10
    assert ds.X.shape[2] == n_features
    assert ds.X.mean() > 25.0  # raw (~50), not standardised (~0)


def test_sequence_alignment_has_no_lookahead() -> None:
    horizon, seq_len = 2, 5
    loader = TrainingDataLoader(target_horizon_bars=horizon)
    n = 60
    features = np.arange(n, dtype=float).reshape(n, 1)  # feature value == bar index
    close = np.linspace(100.0, 160.0, n)

    labels = loader._create_labels(close, horizon)
    ds = loader.load_sequence_data(features, close, seq_length=seq_len)

    for j in range(min(6, len(ds.X))):
        # window j ends at bar (j + seq_len - 1): its last feature is that bar
        assert ds.X[j, -1, 0] == float(j + seq_len - 1)
        # and its label is the forward return that BEGINS at that same last bar,
        # whose future component is never inside the window -> no leakage.
        assert ds.y[j] == labels[j + seq_len - 1]
