"""Walk-forward validation for time-series models.

Contains :class:`WalkForwardValidator` (expanding / rolling walk-forward
evaluation that retrains a model per fold) and :func:`purged_chrono_folds`
(pure index-generation for purged, embargoed chronological K-fold CV — the
split primitive consumed by hyperparameter search).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from services.prediction.models.base import BasePredictor, TrainResult
from services.prediction.training.dataset_builder import HORIZON

logger = logging.getLogger(__name__)


def default_embargo(stride: int = HORIZON) -> int:
    """Default embargo (in replay steps) for :func:`purged_chrono_folds`.

    ``ceil(HORIZON / stride) + 1``: each sample is a replay step advancing
    ``stride`` bars, and its label is realized over the ``HORIZON``-bar
    forward horizon, i.e. it overlaps ``ceil(HORIZON / stride)`` subsequent
    replay steps; the +1 covers the boundary step. At the dataset builder's
    default ``stride == HORIZON`` this evaluates to 2.

    Caveat: ``build_dataset``'s returns/labels are measured over ``HORIZON``
    *replay steps* (``forward_returns`` on the replay-step closes; the
    triple-barrier time barrier sits ``HORIZON`` replay steps ahead), which is
    ``HORIZON`` steps in sample units regardless of stride. This formula
    matches that span exactly at ``stride == 1``; for coarser strides pass
    ``embargo=HORIZON + 1`` explicitly to fully cover the label span.
    """
    return math.ceil(HORIZON / stride) + 1


def purged_chrono_folds(
    n_samples: int, n_folds: int, embargo: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged, embargoed chronological K-fold split (Lopez de Prado).

    Samples ``0..n_samples-1`` are assumed chronologically ordered replay
    steps whose labels look up to the label horizon *forward* in time. Plain
    K-fold leaks: a training sample just before the validation block has a
    label window reaching into the block (it is partly "graded" on validation
    data), and one just after starts inside the span over which the last
    validation labels are still being realized (serial correlation flows the
    other way). Both are removed by excluding every training sample within
    ``embargo`` steps of the validation block on either side — purging before
    it, embargo after it.

    Args:
        n_samples: total number of chronologically ordered samples.
        n_folds: number of contiguous validation blocks (>= 2). Blocks are
            ``np.array_split(np.arange(n_samples), n_folds)`` — chronological
            and near-equal sized.
        embargo: exclusion radius in samples (replay steps). ``None`` uses
            :func:`default_embargo` — see its docstring for the formula and
            the stride caveat.

    Returns:
        List of ``(train_idx, val_idx)`` int64 index-array pairs, one per
        fold, in chronological fold order. ``train_idx`` is sorted ascending
        and never overlaps ``[val_idx[0] - embargo, val_idx[-1] + embargo]``.

    Raises:
        ValueError: on invalid arguments, or if the embargo is so wide that a
            fold would have an empty training set.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_samples < n_folds:
        raise ValueError(f"n_samples={n_samples} < n_folds={n_folds}")
    emb = default_embargo() if embargo is None else embargo
    if emb < 0:
        raise ValueError(f"embargo must be >= 0, got {emb}")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in np.array_split(np.arange(n_samples, dtype=np.int64), n_folds):
        lo, hi = int(block[0]), int(block[-1])
        before = np.arange(0, max(lo - emb, 0), dtype=np.int64)
        after = np.arange(min(hi + emb + 1, n_samples), n_samples, dtype=np.int64)
        train_idx = np.concatenate([before, after])
        if train_idx.size == 0:
            raise ValueError(
                f"embargo={emb} leaves no training samples for the fold spanning "
                f"[{lo}, {hi}] (n_samples={n_samples}, n_folds={n_folds})"
            )
        folds.append((train_idx, block))
    return folds


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
