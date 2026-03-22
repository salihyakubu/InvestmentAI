"""Training data pipeline for loading, labelling, and normalising feature data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataset:
    """Container for a prepared training dataset."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None


class TrainingDataLoader:
    """Builds training datasets from raw feature / price data.

    The loader is deliberately decoupled from a specific data store so
    that callers provide feature matrices and price arrays directly.
    """

    def __init__(
        self,
        target_horizon_bars: int = 5,
        up_threshold: float = 0.005,
        down_threshold: float = -0.005,
    ) -> None:
        self.target_horizon_bars = target_horizon_bars
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_training_data(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> TrainingDataset:
        """Prepare flat (tabular) training data.

        Args:
            features: (n_samples, n_features)
            close_prices: (n_samples,) closing prices aligned with features
            feature_names: optional list of feature names

        Returns:
            :class:`TrainingDataset` with normalised X and labels y.
        """
        labels = self._create_labels(close_prices, self.target_horizon_bars)

        # Trim features to match labels length (labels are shorter by horizon)
        valid_len = len(labels)
        X = features[:valid_len]

        X_norm, scaler = self._normalize(X)

        logger.info(
            "Prepared %d samples  class distribution: short=%d flat=%d long=%d",
            valid_len,
            int((labels == 0).sum()),
            int((labels == 1).sum()),
            int((labels == 2).sum()),
        )

        return TrainingDataset(
            X=X_norm,
            y=labels,
            feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
            scaler=scaler,
        )

    def load_sequence_data(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        seq_length: int = 60,
        feature_names: list[str] | None = None,
    ) -> TrainingDataset:
        """Prepare sequential training data for LSTM / Transformer models.

        Returns X of shape (n_samples, seq_length, n_features) and aligned labels.
        """
        labels = self._create_labels(close_prices, self.target_horizon_bars)
        valid_len = len(labels)
        feat = features[:valid_len]

        feat_norm, scaler = self._normalize(feat)

        sequences, seq_labels = self._create_sequences(feat_norm, labels, seq_length)

        logger.info(
            "Prepared %d sequences (seq_len=%d)  class distribution: short=%d flat=%d long=%d",
            len(sequences),
            seq_length,
            int((seq_labels == 0).sum()),
            int((seq_labels == 1).sum()),
            int((seq_labels == 2).sum()),
        )

        return TrainingDataset(
            X=sequences,
            y=seq_labels,
            feature_names=feature_names or [f"f{i}" for i in range(features.shape[1])],
            scaler=scaler,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_labels(self, close_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Create 3-class direction labels based on forward returns.

        Returns:
            Array of length ``len(close_prices) - horizon`` with values
            0 (short / down), 1 (flat), 2 (long / up).
        """
        forward_returns = (close_prices[horizon:] - close_prices[:-horizon]) / close_prices[:-horizon]
        labels = np.ones(len(forward_returns), dtype=np.int64)  # default flat
        labels[forward_returns <= self.down_threshold] = 0  # short
        labels[forward_returns >= self.up_threshold] = 2  # long
        return labels

    def _create_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window over features to produce (n, seq_length, n_features)."""
        n_samples = len(features) - seq_length + 1
        if n_samples <= 0:
            raise ValueError(f"Not enough data ({len(features)}) for sequence length {seq_length}")

        sequences = np.lib.stride_tricks.sliding_window_view(features, (seq_length, features.shape[1]))
        sequences = sequences.squeeze(axis=1)  # (n_samples, seq_length, n_features)

        # Align labels with the last timestep of each window
        seq_labels = labels[seq_length - 1 :]
        min_len = min(len(sequences), len(seq_labels))
        return sequences[:min_len], seq_labels[:min_len]

    @staticmethod
    def _normalize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        return X_norm, scaler
