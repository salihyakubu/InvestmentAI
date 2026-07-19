"""Shared dataset builder: replay/label/split semantics used by both the
bootstrap training script and the AutoRetrainer.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pytest

from services.feature_engineering.feature_store import FeatureStore
from services.prediction.training.dataset_builder import (
    HORIZON,
    MIN_BARS,
    WINDOW,
    bars_matrix,
    build_dataset,
    forward_returns,
    replay_features,
)


def _make_cols(n: int, seed: int, start: float = 100.0) -> dict[str, np.ndarray]:
    """Synthetic random-walk OHLCV column arrays."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.002, size=n)
    close = start * np.cumprod(1.0 + rets)
    high = close * (1.0 + rng.uniform(0.0001, 0.004, size=n))
    low = close * (1.0 - rng.uniform(0.0001, 0.004, size=n))
    open_ = np.concatenate(([start], close[:-1]))
    volume = rng.uniform(1_000.0, 5_000.0, size=n)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


# ---------------------------------------------------------------------------
# bars_matrix / forward_returns
# ---------------------------------------------------------------------------


def test_bars_matrix_coerces_decimal_to_float64() -> None:
    bars = [
        SimpleNamespace(
            open=Decimal("1.5"), high=Decimal("2.25"), low=Decimal("1.25"),
            close=Decimal("2.0"), volume=Decimal("10"),
        ),
        SimpleNamespace(
            open=Decimal("2.0"), high=Decimal("2.5"), low=Decimal("1.75"),
            close=Decimal("2.25"), volume=Decimal("12.5"),
        ),
    ]
    cols = bars_matrix(bars)
    assert set(cols) == {"open", "high", "low", "close", "volume"}
    for arr in cols.values():
        assert arr.dtype == np.float64
        assert arr.shape == (2,)
    np.testing.assert_allclose(cols["close"], [2.0, 2.25])
    np.testing.assert_allclose(cols["volume"], [10.0, 12.5])


def test_forward_returns_known_values() -> None:
    closes = np.array([100.0, 110.0, 121.0])
    np.testing.assert_allclose(forward_returns(closes, 1), [0.1, 0.1])
    np.testing.assert_allclose(forward_returns(closes, 2), [0.21])


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_shapes_labels_and_split() -> None:
    per = {"AAA": _make_cols(350, seed=1), "BBB": _make_cols(350, seed=2)}
    tr_X, tr_y, tr_r, va_X, va_y, va_r, names = build_dataset(per)

    # Feature names come from the live pipeline, sorted, and size the matrix.
    assert names == sorted(names)
    assert len(names) > 0
    assert tr_X.shape[1] == va_X.shape[1] == len(names)

    # Recompute the expected pipeline independently from the public helpers.
    store = FeatureStore(db_session=None, redis=None)
    replays = {sym: replay_features(sym, cols, HORIZON, store) for sym, cols in per.items()}
    pooled = np.concatenate([forward_returns(closes, HORIZON) for _, closes, _ in replays.values()])
    lo, hi = np.percentile(pooled, [30, 70])

    exp_tr_X, exp_tr_y, exp_tr_r, exp_va_X, exp_va_y, exp_va_r = [], [], [], [], [], []
    for _sym, (X, closes, _n) in replays.items():
        r = forward_returns(closes, HORIZON)
        Xs = X[: len(r)]
        y = np.ones(len(r), dtype=np.int64)
        y[r <= lo] = 0
        y[r >= hi] = 2
        split = int(len(Xs) * 0.8)  # chronological 80/20 within each symbol
        exp_tr_X.append(Xs[:split])
        exp_tr_y.append(y[:split])
        exp_tr_r.append(r[:split])
        exp_va_X.append(Xs[split:])
        exp_va_y.append(y[split:])
        exp_va_r.append(r[split:])

    np.testing.assert_array_equal(tr_X, np.concatenate(exp_tr_X))
    np.testing.assert_array_equal(tr_y, np.concatenate(exp_tr_y))
    np.testing.assert_array_equal(tr_r, np.concatenate(exp_tr_r))
    np.testing.assert_array_equal(va_X, np.concatenate(exp_va_X))
    np.testing.assert_array_equal(va_y, np.concatenate(exp_va_y))
    np.testing.assert_array_equal(va_r, np.concatenate(exp_va_r))

    # Structural invariants: per-symbol 80/20 split, valid 3-class labels,
    # ~30% of pooled rows below / above the percentile thresholds.
    windows_per_symbol = (350 - WINDOW) // HORIZON + 1
    labeled_per_symbol = windows_per_symbol - HORIZON
    exp_train = 2 * int(labeled_per_symbol * 0.8)
    exp_val = 2 * labeled_per_symbol - exp_train
    assert tr_X.shape[0] == len(tr_y) == len(tr_r) == exp_train
    assert va_X.shape[0] == len(va_y) == len(va_r) == exp_val
    all_y = np.concatenate([tr_y, va_y])
    assert set(np.unique(all_y)) <= {0, 1, 2}
    total = len(all_y)
    assert np.isclose((all_y == 0).sum() / total, 0.3, atol=0.05)
    assert np.isclose((all_y == 2).sum() / total, 0.3, atol=0.05)


def test_symbol_below_min_bars_is_skipped() -> None:
    big = _make_cols(300, seed=3)
    small = _make_cols(MIN_BARS - 1, seed=4)

    with_small = build_dataset({"BIG": big, "SMALL": small})
    without_small = build_dataset({"BIG": big})

    for got, expected in zip(with_small[:6], without_small[:6]):
        np.testing.assert_array_equal(got, expected)
    assert with_small[6] == without_small[6]


def test_all_symbols_too_short_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_dataset({"SMALL": _make_cols(100, seed=5)})
    with pytest.raises(ValueError):
        build_dataset({})
