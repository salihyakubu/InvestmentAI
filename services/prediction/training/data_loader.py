"""Training data pipeline for loading, labelling, and normalising feature data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    returns: np.ndarray | None = None  # real forward returns aligned with y


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
        forward_returns = self._forward_returns(close_prices, self.target_horizon_bars)
        labels = self._labels_from_returns(forward_returns)

        # Trim features to match labels length (labels are shorter by horizon)
        valid_len = len(labels)
        X = features[:valid_len]

        # Do NOT scale the tabular features. The tree models that consume this
        # path are invariant to monotonic feature scaling, and the live
        # inference path feeds RAW features through a scaler that was never
        # persisted or applied at serving -- that mismatch was the source of
        # severe train/serve skew. Keeping X raw makes training and serving match.

        logger.info(
            "Prepared %d samples  class distribution: short=%d flat=%d long=%d",
            valid_len,
            int((labels == 0).sum()),
            int((labels == 1).sum()),
            int((labels == 2).sum()),
        )

        return TrainingDataset(
            X=X,
            y=labels,
            feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
            scaler=None,
            returns=forward_returns,
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

        Features are returned RAW (unscaled), with ``scaler=None``. The sequence
        models own their normalisation: each computes mean/std on the raw training
        tensor, persists them, and re-applies the *same* statistics at inference.
        Scaling here as well caused double normalisation -- the model recomputed
        stats on already z-scored data, then at serve time applied near-identity
        stats to raw features, so training and serving saw different
        distributions. Keeping X raw makes a single, model-owned normalisation the
        source of truth (mirrors ``load_training_data`` for the tree path).
        """
        labels = self._create_labels(close_prices, self.target_horizon_bars)
        valid_len = len(labels)
        feat = features[:valid_len]

        sequences, seq_labels = self._create_sequences(feat, labels, seq_length)

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
            scaler=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_returns(self, close_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Forward simple returns over *horizon* bars (len = len(close) - horizon)."""
        returns: np.ndarray = (close_prices[horizon:] - close_prices[:-horizon]) / close_prices[:-horizon]
        return returns

    def _labels_from_returns(self, forward_returns: np.ndarray) -> np.ndarray:
        """Map forward returns to 3-class labels: 0 short, 1 flat, 2 long."""
        labels = np.ones(len(forward_returns), dtype=np.int64)  # default flat
        labels[forward_returns <= self.down_threshold] = 0  # short
        labels[forward_returns >= self.up_threshold] = 2  # long
        return labels

    def _create_labels(self, close_prices: np.ndarray, horizon: int) -> np.ndarray:
        """Create 3-class direction labels based on forward returns."""
        return self._labels_from_returns(self._forward_returns(close_prices, horizon))

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

        # Align each window's label with the forward return that begins at the
        # window's LAST bar: window j covers bars [j, j+seq_length-1] and its
        # label is the return realised AFTER bar (j+seq_length-1). The future
        # part of that return is never inside the window, so there is no
        # look-ahead leakage.
        seq_labels = labels[seq_length - 1 :]
        min_len = min(len(sequences), len(seq_labels))
        return sequences[:min_len], seq_labels[:min_len]
