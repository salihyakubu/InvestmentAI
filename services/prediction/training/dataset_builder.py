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
- ``HORIZON``-bar forward-return labels with pooled, data-driven 30th/70th
  percentile thresholds (fixed daily-scale thresholds would label nearly
  everything flat at 1-minute cadence).
- Chronological per-symbol 80/20 train/validation split.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from services.feature_engineering.feature_store import FeatureStore

logger = logging.getLogger(__name__)

WINDOW = 200          # live feature buffer size (feature_engineering/service.py)
HORIZON = 5           # label horizon in bars == prediction service semantics
MIN_VAL_ACCURACY = 0.34  # must beat the 1/3 random baseline out of sample

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


def build_dataset(
    per_symbol_cols: dict[str, dict[str, np.ndarray]],
    *,
    stride: int = HORIZON,
    store: FeatureStore | None = None,
) -> DatasetSplits:
    """Replay features, label, and split -> pooled train/val arrays.

    Args:
        per_symbol_cols: symbol -> float64 OHLCV column arrays (``bars_matrix``
            output). Symbols with fewer than ``MIN_BARS`` bars are skipped.
        stride: replay window stride in bars (default = label horizon, which
            keeps forward-return labels non-overlapping).
        store: optional ``FeatureStore`` (a fresh offline one is built if None).

    Returns:
        ``(X_train, y_train, returns_train, X_val, y_val, returns_val,
        feature_names)`` pooled across symbols, split 80/20 chronologically
        within each symbol, with class labels from pooled 30th/70th percentile
        forward-return thresholds.

    Raises:
        ValueError: if no symbol yields usable labeled data.
    """
    store = store if store is not None else FeatureStore(db_session=None, redis=None)

    per_symbol: list[tuple[str, np.ndarray, np.ndarray]] = []
    names: list[str] = []
    for sym, cols in per_symbol_cols.items():
        if len(cols["close"]) < MIN_BARS:
            continue
        X, closes, n = replay_features(sym, cols, stride, store)
        names = names or n
        per_symbol.append((sym, X, closes))

    if not per_symbol:
        raise ValueError("no symbol produced usable training data")

    # Pooled, data-driven label thresholds for the HORIZON-bar forward return.
    pooled = np.concatenate([forward_returns(c, HORIZON) for _, _, c in per_symbol])
    if pooled.size == 0:
        raise ValueError("no symbol produced enough replay windows for labels")
    lo, hi = np.percentile(pooled, [30, 70])
    logger.info(
        "label thresholds (%d-bar fwd return): short<=%.5f  long>=%.5f", HORIZON, lo, hi
    )

    tr_X: list[np.ndarray] = []
    tr_y: list[np.ndarray] = []
    tr_r: list[np.ndarray] = []
    va_X: list[np.ndarray] = []
    va_y: list[np.ndarray] = []
    va_r: list[np.ndarray] = []
    for _sym, X, closes in per_symbol:
        r = forward_returns(closes, HORIZON)
        Xs = X[: len(r)]
        y = np.ones(len(r), dtype=np.int64)
        y[r <= lo] = 0
        y[r >= hi] = 2
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
