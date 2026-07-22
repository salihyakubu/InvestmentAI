"""Shared training-dataset construction for bootstrap training and auto-retraining.

Both ``scripts/train_and_promote.py`` (initial bootstrap) and the
``AutoRetrainer`` (scheduled / drift-triggered retraining) must feed the models
the EXACT numbers the worker computes at serve time. This module holds the one
implementation of that replay + labeling pipeline so the two paths can never
drift apart:

- Rolling ``WINDOW``-bar replay of ``FeatureStore.compute_all_features`` (the
  live feature computation; obv / volume-profile depend on the window itself
  and ema/macd/rsi/atr/adx are path-dependent, so only a windowed replay
  reproduces serve-time values).
- Triple-barrier class labels (Lopez de Prado) by default: per replay step,
  volatility-scaled upper/lower price barriers with a time barrier ``HORIZON``
  replay steps ahead; the first barrier touched decides the class. The legacy
  pooled 30th/70th percentile forward-return labeling remains available via
  ``labeling="percentile"``.
- The returns array (consumed by the return regressors) is always the actual
  ``HORIZON``-replay-step forward return, independent of the labeling mode.
- Chronological per-symbol 80/20 train/validation split.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl

from services.feature_engineering.feature_store import FeatureStore

logger = logging.getLogger(__name__)

WINDOW = 200          # live feature buffer size (feature_engineering/service.py)
HORIZON = 5           # label horizon in bars == prediction service semantics
MIN_VAL_ACCURACY = 0.34  # must beat the 1/3 random baseline out of sample

# Default triple-barrier width multiplier: barrier offset = k * (1-bar close
# return volatility over the trailing WINDOW). Tuned on synthetic random walks
# so the default replay geometry (stride = HORIZON -> a HORIZON*stride = 25 bar
# path to the time barrier) yields a sane class balance: ~31% short / 38% flat /
# 31% long on Gaussian walks, and every class stays well populated (>18%) under
# drift or fat-tailed returns. If you change stride/horizon, scale k roughly
# with sqrt(horizon * stride); 4.5 ~= 0.9 * sqrt(25).
TB_K = 4.5

Labeling = Literal["triple_barrier", "percentile"]

# A symbol needs at least this many bars to produce any usable labeled rows.
MIN_BARS = WINDOW + HORIZON * 3

DatasetSplits = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]


def bars_matrix(bars: Sequence[Any]) -> dict[str, np.ndarray]:
    """Turn a sequence of bar-like records into float64 OHLCV column arrays.

    Each element must expose ``open`` / ``high`` / ``low`` / ``close`` /
    ``volume`` attributes whose values are float-coercible (float, int, or
    ``Decimal`` as returned by the ``ohlcv`` table's ``Numeric`` columns).
    """
    return {
        "open": np.array([b.open for b in bars], dtype=np.float64),
        "high": np.array([b.high for b in bars], dtype=np.float64),
        "low": np.array([b.low for b in bars], dtype=np.float64),
        "close": np.array([b.close for b in bars], dtype=np.float64),
        "volume": np.array([b.volume for b in bars], dtype=np.float64),
    }


def replay_features(
    symbol: str, cols: dict[str, np.ndarray], stride: int, store: FeatureStore
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rolling ``WINDOW``-bar replay of the live feature computation.

    Returns (X, close_at_window_end, feature_names). Stride = HORIZON keeps the
    forward-return labels non-overlapping. Windowed replay (not full-history
    indicators) is mandatory: obv / volume-profile depend on the window itself
    and ema/macd/rsi/atr/adx are path-dependent, so only this reproduces the
    numbers the worker computes live.
    """
    n = len(cols["close"])
    names: list[str] | None = None
    rows: list[list[float]] = []
    closes: list[float] = []
    for end_i in range(WINDOW, n + 1, stride):
        window = {k: v[end_i - WINDOW : end_i] for k, v in cols.items()}
        feats = store.compute_all_features(symbol, pl.DataFrame(window))
        if names is None:
            names = sorted(feats)
        rows.append([float(feats.get(k, 0.0)) for k in names])
        closes.append(float(window["close"][-1]))
    if names is None:
        return np.empty((0, 0)), np.empty(0), []
    return np.asarray(rows, dtype=np.float64), np.asarray(closes), names


def forward_returns(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Forward percentage return over *horizon* replay steps."""
    returns: np.ndarray = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
    return returns


def triple_barrier_labels(
    bar_closes: np.ndarray,
    *,
    stride: int,
    horizon: int = HORIZON,
    window: int = WINDOW,
    k_up: float = TB_K,
    k_down: float = TB_K,
) -> np.ndarray:
    """Triple-barrier class labels (Lopez de Prado) over bar-level closes.

    Replay step ``j`` corresponds to the feature window ending at bar index
    ``e_j = window - 1 + j * stride`` (matching ``replay_features``). For each
    labeled step:

    - entry price = ``bar_closes[e_j]``;
    - barrier width = ``k * sigma`` where ``sigma`` is the population std of
      the 1-bar close returns over the trailing ``window`` bars (upper barrier
      ``P0 * (1 + k_up * sigma)``, lower ``P0 * (1 - k_down * sigma)``);
    - time barrier = ``horizon`` replay steps ahead, i.e. bar index
      ``e_j + horizon * stride`` -- exactly the close the ``horizon``-step
      forward return is measured against.

    The bar closes strictly after ``e_j`` up to and including the time-barrier
    bar are walked in order: label 2 (long) if the upper barrier is touched
    first, 0 (short) if the lower barrier is touched first, 1 (flat) if the
    time barrier expires with neither touched. Touches are ``>=`` / ``<=`` on
    the bar close. The width is floored at a tiny epsilon so a zero-volatility
    window cannot produce coincident barriers (which would make a single close
    "touch" both sides).

    Returns an int64 array of length ``max(n_steps - horizon, 0)`` where
    ``n_steps`` is the number of replay windows -- the same row count as
    ``forward_returns`` on the replay-step closes.
    """
    n = len(bar_closes)
    n_steps = (n - window) // stride + 1 if n >= window else 0
    n_labeled = max(n_steps - horizon, 0)
    labels = np.ones(n_labeled, dtype=np.int64)
    if n_labeled == 0:
        return labels

    span = horizon * stride
    one_bar_rets = np.diff(bar_closes) / bar_closes[:-1]
    untouched = n  # sentinel index: larger than any real touch index
    for j in range(n_labeled):
        e = window - 1 + j * stride
        sigma = float(np.std(one_bar_rets[e - window + 1 : e]))
        p0 = float(bar_closes[e])
        width_up = max(k_up * sigma, 1e-12)
        width_down = max(k_down * sigma, 1e-12)
        path = bar_closes[e + 1 : e + span + 1]
        up_hits = path >= p0 * (1.0 + width_up)
        down_hits = path <= p0 * (1.0 - width_down)
        first_up = int(np.argmax(up_hits)) if up_hits.any() else untouched
        first_down = int(np.argmax(down_hits)) if down_hits.any() else untouched
        if first_up == first_down:  # only possible as untouched == untouched
            labels[j] = 1
        elif first_up < first_down:
            labels[j] = 2
        else:
            labels[j] = 0
    return labels


def build_dataset(
    per_symbol_cols: dict[str, dict[str, np.ndarray]],
    *,
    stride: int = HORIZON,
    store: FeatureStore | None = None,
    labeling: Labeling = "triple_barrier",
    tb_k_up: float = TB_K,
    tb_k_down: float = TB_K,
) -> DatasetSplits:
    """Replay features, label, and split -> pooled train/val arrays.

    Args:
        per_symbol_cols: symbol -> float64 OHLCV column arrays (``bars_matrix``
            output). Symbols with fewer than ``MIN_BARS`` bars are skipped.
        stride: replay window stride in bars (default = label horizon, which
            keeps forward-return labels non-overlapping).
        store: optional ``FeatureStore`` (a fresh offline one is built if None).
        labeling: ``"triple_barrier"`` (default) labels each replay step per
            symbol by first touch of volatility-scaled price barriers (see
            :func:`triple_barrier_labels`); ``"percentile"`` restores the
            legacy pooled 30th/70th percentile forward-return thresholds.
        tb_k_up: upper-barrier width multiplier (triple-barrier mode only).
        tb_k_down: lower-barrier width multiplier (triple-barrier mode only).

    Returns:
        ``(X_train, y_train, returns_train, X_val, y_val, returns_val,
        feature_names)`` pooled across symbols, split 80/20 chronologically
        within each symbol. ``returns_*`` are always the actual
        ``HORIZON``-replay-step forward returns regardless of labeling mode.

    Raises:
        ValueError: if no symbol yields usable labeled data, or *labeling* is
            not a recognized mode.
    """
    store = store if store is not None else FeatureStore(db_session=None, redis=None)

    per_symbol: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    names: list[str] = []
    for sym, cols in per_symbol_cols.items():
        if len(cols["close"]) < MIN_BARS:
            continue
        X, closes, n = replay_features(sym, cols, stride, store)
        names = names or n
        per_symbol.append((sym, X, closes, cols["close"]))

    if not per_symbol:
        raise ValueError("no symbol produced usable training data")

    labels: list[np.ndarray] = []
    if labeling == "percentile":
        # Pooled, data-driven label thresholds for the HORIZON-step fwd return.
        pooled = np.concatenate([forward_returns(c, HORIZON) for _, _, c, _ in per_symbol])
        if pooled.size == 0:
            raise ValueError("no symbol produced enough replay windows for labels")
        lo, hi = np.percentile(pooled, [30, 70])
        logger.info(
            "label thresholds (%d-bar fwd return): short<=%.5f  long>=%.5f", HORIZON, lo, hi
        )
        for _sym, _X, closes, _bar_closes in per_symbol:
            r = forward_returns(closes, HORIZON)
            y = np.ones(len(r), dtype=np.int64)
            y[r <= lo] = 0
            y[r >= hi] = 2
            labels.append(y)
    elif labeling == "triple_barrier":
        for _sym, _X, _closes, bar_closes in per_symbol:
            labels.append(
                triple_barrier_labels(bar_closes, stride=stride, k_up=tb_k_up, k_down=tb_k_down)
            )
        all_y = np.concatenate(labels)
        if all_y.size == 0:
            raise ValueError("no symbol produced enough replay windows for labels")
        counts = np.bincount(all_y, minlength=3)
        logger.info(
            "triple-barrier labels (k_up=%.2f k_down=%.2f): short=%.1f%%  flat=%.1f%%  long=%.1f%%",
            tb_k_up, tb_k_down,
            100.0 * counts[0] / all_y.size, 100.0 * counts[1] / all_y.size,
            100.0 * counts[2] / all_y.size,
        )
    else:
        raise ValueError(f"unknown labeling mode: {labeling!r}")

    tr_X: list[np.ndarray] = []
    tr_y: list[np.ndarray] = []
    tr_r: list[np.ndarray] = []
    va_X: list[np.ndarray] = []
    va_y: list[np.ndarray] = []
    va_r: list[np.ndarray] = []
    for (_sym, X, closes, _bar_closes), y in zip(per_symbol, labels):
        r = forward_returns(closes, HORIZON)
        Xs = X[: len(r)]
        split = int(len(Xs) * 0.8)  # chronological within each symbol
        tr_X.append(Xs[:split])
        tr_y.append(y[:split])
        tr_r.append(r[:split])
        va_X.append(Xs[split:])
        va_y.append(y[split:])
        va_r.append(r[split:])

    return (
        np.concatenate(tr_X), np.concatenate(tr_y), np.concatenate(tr_r),
        np.concatenate(va_X), np.concatenate(va_y), np.concatenate(va_r),
        names,
    )
