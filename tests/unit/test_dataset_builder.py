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
    triple_barrier_labels,
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
    tr_X, tr_y, tr_r, va_X, va_y, va_r, names = build_dataset(per, labeling="percentile")

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


def test_build_dataset_rejects_unknown_labeling() -> None:
    with pytest.raises(ValueError, match="unknown labeling"):
        build_dataset({"AAA": _make_cols(350, seed=1)}, labeling="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# triple-barrier labeling
# ---------------------------------------------------------------------------

# Hand-built geometry: window=5, stride=1, horizon=2. The trailing window has
# 1-bar returns [+1%, -1%, +1%, -1%], so the population std is exactly 0.01.
# With k_up = k_down = 2 the barrier half-width is 2%: entry price
# P0 = 100 * (1.01 * 0.99)^2 = 99.980001, upper ~= 101.9796, lower ~= 97.9804.
# Exactly one replay step gets labeled (n = window + horizon bars), and its
# walk is the final `horizon` bar closes.
_TB_WINDOW = 5
_TB_HORIZON = 2
_TB_K = 2.0
_TB_P0 = 100.0 * (1.01 * 0.99) ** 2  # 99.980001
_TB_UPPER = _TB_P0 * 1.02            # ~101.9796
_TB_LOWER = _TB_P0 * 0.98            # ~97.9804


def _tb_case(future_closes: list[float]) -> np.ndarray:
    """Label the hand-built window followed by *future_closes* (len == horizon)."""
    assert len(future_closes) == _TB_HORIZON
    window_rets = np.array([0.01, -0.01, 0.01, -0.01])
    window_closes = 100.0 * np.cumprod(np.concatenate(([1.0], 1.0 + window_rets)))
    closes = np.concatenate((window_closes, np.asarray(future_closes, dtype=np.float64)))
    return triple_barrier_labels(
        closes, stride=1, horizon=_TB_HORIZON, window=_TB_WINDOW, k_up=_TB_K, k_down=_TB_K
    )


def test_triple_barrier_upper_touch_labels_long() -> None:
    np.testing.assert_array_equal(_tb_case([102.5, 100.0]), [2])


def test_triple_barrier_lower_touch_labels_short() -> None:
    np.testing.assert_array_equal(_tb_case([97.5, 100.0]), [0])


def test_triple_barrier_time_expiry_labels_flat() -> None:
    # Both future closes strictly inside (lower, upper): time barrier expires.
    assert _TB_LOWER < 100.5 < _TB_UPPER and _TB_LOWER < 101.0 < _TB_UPPER
    np.testing.assert_array_equal(_tb_case([100.5, 101.0]), [1])


def test_triple_barrier_touch_on_time_barrier_bar_counts() -> None:
    # Inside on the first bar, beyond the upper barrier exactly on the
    # time-barrier bar: still a touch, not an expiry.
    np.testing.assert_array_equal(_tb_case([100.5, 102.5]), [2])


def test_triple_barrier_first_touch_wins() -> None:
    # Both barriers are crossed within the horizon; the FIRST touch decides.
    np.testing.assert_array_equal(_tb_case([102.5, 97.0]), [2])
    np.testing.assert_array_equal(_tb_case([97.5, 103.0]), [0])


def test_triple_barrier_row_count_matches_forward_returns() -> None:
    closes = _make_cols(350, seed=7)["close"]
    y = triple_barrier_labels(closes, stride=HORIZON)
    n_steps = (len(closes) - WINDOW) // HORIZON + 1
    assert len(y) == n_steps - HORIZON
    # Too few bars for any replay window -> empty labels, not an error.
    assert len(triple_barrier_labels(closes[: WINDOW - 1], stride=HORIZON)) == 0


def test_triple_barrier_label_balance_on_random_walks() -> None:
    # Default k on driftless random walks must produce all three classes in
    # sane proportions (no class collapses, flat does not dominate).
    for seed in (11, 12):
        rng = np.random.default_rng(seed)
        closes = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=6000))
        y = triple_barrier_labels(closes, stride=HORIZON)
        fracs = [(y == c).mean() for c in (0, 1, 2)]
        assert all(0.15 <= f <= 0.60 for f in fracs), fracs


def test_build_dataset_triple_barrier_default_wiring() -> None:
    per = {"AAA": _make_cols(350, seed=1)}
    default_out = build_dataset(per)
    tb_out = build_dataset(per, labeling="triple_barrier")
    pct_out = build_dataset(per, labeling="percentile")

    # The default is triple-barrier.
    for got, expected in zip(default_out[:6], tb_out[:6]):
        np.testing.assert_array_equal(got, expected)
    assert default_out[6] == tb_out[6]

    # Features, forward returns, and names are labeling-invariant; only y differs.
    tr_X, tr_y, tr_r, va_X, va_y, va_r, names = tb_out
    np.testing.assert_array_equal(tr_X, pct_out[0])
    np.testing.assert_array_equal(tr_r, pct_out[2])
    np.testing.assert_array_equal(va_X, pct_out[3])
    np.testing.assert_array_equal(va_r, pct_out[5])
    assert names == pct_out[6]
    assert len(tr_y) == len(pct_out[1]) and len(va_y) == len(pct_out[4])
    assert set(np.unique(np.concatenate([tr_y, va_y]))) <= {0, 1, 2}

    # y is exactly triple_barrier_labels on the bar closes, split 80/20.
    expected_y = triple_barrier_labels(per["AAA"]["close"], stride=HORIZON)
    split = int(len(expected_y) * 0.8)
    np.testing.assert_array_equal(tr_y, expected_y[:split])
    np.testing.assert_array_equal(va_y, expected_y[split:])
