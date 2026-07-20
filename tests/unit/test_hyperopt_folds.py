"""Hyperopt: catboost search space + purged-CV fold objective.

The fold objective must actually iterate the purged chronological folds
(asserted via a counting fake model) and maximise MEAN fold accuracy; without
folds the original single-split val-loss objective is preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from services.prediction.models.base import TrainResult
from services.prediction.training.hyperopt import HyperOptimizer
from services.prediction.training.walk_forward import purged_chrono_folds


def _tiny_data(n: int = 150, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5).astype(np.int64)  # 0/1/2
    return X, y


class _CountingFakeModel:
    """Records every train() call; accuracy is a function of one param."""

    calls: list[dict[str, Any]] = []

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def train(self, X_train, y_train, X_val, y_val) -> TrainResult:
        type(self).calls.append(
            {
                "params": dict(self.params),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "val_is": X_val,
            }
        )
        acc = float(self.params.get("subsample", 0.0))  # higher subsample "wins"
        return TrainResult(
            train_loss=1.0, val_loss=1.0 - acc, train_accuracy=acc,
            val_accuracy=acc, epochs_trained=1,
        )


@pytest.fixture()
def fake_optimizer(monkeypatch: pytest.MonkeyPatch) -> HyperOptimizer:
    _CountingFakeModel.calls = []
    optimizer = HyperOptimizer(model_type="xgboost", n_trials=3)
    monkeypatch.setattr(
        HyperOptimizer, "_create_model", lambda self, params: _CountingFakeModel(params)
    )
    return optimizer


def test_unknown_model_type_rejected() -> None:
    with pytest.raises(ValueError):
        HyperOptimizer(model_type="nonsense")


def test_catboost_space_smoke_two_trials() -> None:
    """2 real catboost trials on tiny data return the dictated dimensions."""
    X, y = _tiny_data()
    optimizer = HyperOptimizer(model_type="catboost", n_trials=2)
    best = optimizer.optimize(X[:100], y[:100], X[100:], y[100:])
    assert set(best) == {
        "iterations", "depth", "learning_rate", "l2_leaf_reg", "bagging_temperature",
    }
    assert 100 <= best["iterations"] <= 500
    assert 4 <= best["depth"] <= 10


def test_fold_objective_iterates_every_fold(fake_optimizer: HyperOptimizer) -> None:
    X, y = _tiny_data(n=90)
    x_val, y_val = np.zeros((10, 5)), np.zeros(10, dtype=np.int64)
    folds = purged_chrono_folds(len(X), 3, embargo=2)

    fake_optimizer.optimize(X, y, x_val, y_val, folds=folds)

    # Every trial trains one fresh model per fold.
    assert len(_CountingFakeModel.calls) == fake_optimizer.n_trials * len(folds)

    # Each fold call sees exactly the purged index selection, not the outer val.
    expected_sizes = [(len(tr), len(va)) for tr, va in folds]
    for trial_start in range(0, len(_CountingFakeModel.calls), len(folds)):
        trial_calls = _CountingFakeModel.calls[trial_start : trial_start + len(folds)]
        assert [(c["n_train"], c["n_val"]) for c in trial_calls] == expected_sizes
        for call, (_, va_idx) in zip(trial_calls, folds):
            np.testing.assert_array_equal(call["val_is"], X[va_idx])


def test_fold_objective_maximises_mean_accuracy(fake_optimizer: HyperOptimizer) -> None:
    X, y = _tiny_data(n=60)
    folds = purged_chrono_folds(len(X), 2, embargo=1)
    best = fake_optimizer.optimize(X, y, X[:5], y[:5], folds=folds)

    # The fake's accuracy == its subsample: the winning params must carry the
    # highest subsample seen across trials (maximise direction).
    seen = [c["params"]["subsample"] for c in _CountingFakeModel.calls]
    assert best["subsample"] == pytest.approx(max(seen))


def test_single_split_behaviour_unchanged(fake_optimizer: HyperOptimizer) -> None:
    X, y = _tiny_data(n=40)
    x_val, y_val = X[:12], y[:12]
    best = fake_optimizer.optimize(X, y, x_val, y_val)

    # One train() per trial, on the outer split.
    assert len(_CountingFakeModel.calls) == fake_optimizer.n_trials
    for call in _CountingFakeModel.calls:
        assert call["n_train"] == len(X)
        np.testing.assert_array_equal(call["val_is"], x_val)

    # Minimise val_loss == 1 - subsample -> the winner is still the max subsample.
    seen = [c["params"]["subsample"] for c in _CountingFakeModel.calls]
    assert best["subsample"] == pytest.approx(max(seen))
