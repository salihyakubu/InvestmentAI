"""Walk-forward validation for time-series models."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from services.prediction.models.base import BasePredictor, TrainResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward evaluation."""

    per_fold_metrics: list[TrainResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def avg_accuracy(self) -> float:
        if not self.per_fold_metrics:
            return 0.0
        return float(np.mean([m.val_accuracy for m in self.per_fold_metrics]))

    @property
    def avg_loss(self) -> float:
        if not self.per_fold_metrics:
            return 0.0
        return float(np.mean([m.val_loss for m in self.per_fold_metrics]))


class WalkForwardValidator:
    """Walk-forward (expanding or rolling window) cross-validation.

    Designed for time-series data where future information must never
    leak into training folds.
    """

    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        step_size: int = 21,
        expanding: bool = True,
    ) -> None:
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.expanding = expanding

    def split(self, data_length: int) -> Iterator[tuple[range, range]]:
        """Yield (train_indices, test_indices) folds.

        Args:
            data_length: total number of samples in the dataset.

        Yields:
            Tuples of ``(range(train_start, train_end), range(test_start, test_end))``.
        """
        start = 0
        train_end = self.train_window

        while train_end + self.test_window <= data_length:
            if self.expanding:
                train_range = range(start, train_end)
            else:
                train_start = max(0, train_end - self.train_window)
                train_range = range(train_start, train_end)

            test_range = range(train_end, train_end + self.test_window)
            yield train_range, test_range

            train_end += self.step_size

    def evaluate(
        self,
        model: BasePredictor,
        X: np.ndarray,
        y: np.ndarray,
    ) -> WalkForwardResults:
        """Run walk-forward evaluation, retraining the model at each fold.

        Args:
            model: a *fresh* predictor instance (will be mutated).
            X: features array (2-D for flat models, 3-D for sequence).
            y: label array.

        Returns:
            :class:`WalkForwardResults` with per-fold and aggregate metrics.
        """
        results = WalkForwardResults()
        folds = list(self.split(len(X)))

        if not folds:
            logger.warning("No folds produced for data_length=%d", len(X))
            return results

        logger.info("Walk-forward: %d folds", len(folds))

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx.start : train_idx.stop]
            y_train = y[train_idx.start : train_idx.stop]
            X_test = X[test_idx.start : test_idx.stop]
            y_test = y[test_idx.start : test_idx.stop]

            try:
                fold_result = model.train(X_train, y_train, X_test, y_test)
                results.per_fold_metrics.append(fold_result)
                logger.info(
                    "Fold %d/%d  val_acc=%.4f  val_loss=%.4f",
                    fold_idx + 1,
                    len(folds),
                    fold_result.val_accuracy,
                    fold_result.val_loss,
                )
            except Exception:
                logger.exception("Fold %d failed", fold_idx + 1)

        if results.per_fold_metrics:
            results.aggregate_metrics = {
                "avg_val_accuracy": results.avg_accuracy,
                "avg_val_loss": results.avg_loss,
                "avg_train_accuracy": float(np.mean([m.train_accuracy for m in results.per_fold_metrics])),
                "avg_train_loss": float(np.mean([m.train_loss for m in results.per_fold_metrics])),
                "n_folds": len(results.per_fold_metrics),
                "std_val_accuracy": float(np.std([m.val_accuracy for m in results.per_fold_metrics])),
            }

        return results
