"""Candlestick pattern detection using pure numpy.

Each helper returns a score in [-1, 1]:
  -1 = strong bearish signal
   0 = neutral / no pattern
   1 = strong bullish signal

The main entry point, ``detect_patterns``, evaluates all patterns on the
**last** bar of the provided OHLC arrays and returns a dictionary of
pattern names to scores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _body(open_: NDArray, close: NDArray) -> NDArray:
    """Absolute body size."""
    return np.abs(close - open_)


def _upper_shadow(open_: NDArray, high: NDArray, close: NDArray) -> NDArray:
    return high - np.maximum(open_, close)


def _lower_shadow(open_: NDArray, low: NDArray, close: NDArray) -> NDArray:
    return np.minimum(open_, close) - low


def _range(high: NDArray, low: NDArray) -> NDArray:
    return high - low


def _is_bullish(open_: NDArray, close: NDArray) -> NDArray:
    return close > open_


def _is_bearish(open_: NDArray, close: NDArray) -> NDArray:
    return close < open_


# ---------------------------------------------------------------------------
# Individual pattern detectors
# ---------------------------------------------------------------------------


def _doji(open_: NDArray, high: NDArray, low: NDArray, close: NDArray) -> float:
    """Doji: body is very small relative to the bar range."""
    body = _body(open_, close)
    rng = _range(high, low)
    if len(rng) == 0:
        return 0.0
    idx = -1
    if rng[idx] == 0:
        return 0.0
    ratio = body[idx] / rng[idx]
    if ratio < 0.05:
        return 0.8  # near-perfect doji (neutral-bullish bias at support)
    if ratio < 0.10:
        return 0.4
    return 0.0


def _hammer(open_: NDArray, high: NDArray, low: NDArray, close: NDArray) -> float:
    """Hammer / hanging man.

    Bullish when the lower shadow is >= 2x the body and the upper shadow
    is very small.
    """
    body = _body(open_, close)
    ls = _lower_shadow(open_, low, close)
    us = _upper_shadow(open_, high, close)
    idx = -1
    if body[idx] == 0:
        return 0.0
    if ls[idx] >= 2.0 * body[idx] and us[idx] <= 0.3 * body[idx]:
        return 1.0 if _is_bullish(open_, close)[idx] else 0.6
    return 0.0


def _engulfing(open_: NDArray, _high: NDArray, _low: NDArray, close: NDArray) -> float:
    """Bullish or bearish engulfing (two-bar pattern)."""
    if len(open_) < 2:
        return 0.0

    prev_body = _body(open_, close)[-2]
    curr_body = _body(open_, close)[-1]

    if curr_body <= prev_body:
        return 0.0

    prev_bull = _is_bullish(open_, close)[-2]
    curr_bull = _is_bullish(open_, close)[-1]

    # Bullish engulfing: previous bearish bar engulfed by current bullish bar.
    if not prev_bull and curr_bull:
        if close[-1] > open_[-2] and open_[-1] < close[-2]:
            return 1.0

    # Bearish engulfing: previous bullish bar engulfed by current bearish bar.
    if prev_bull and not curr_bull:
        if close[-1] < open_[-2] and open_[-1] > close[-2]:
            return -1.0

    return 0.0


def _morning_star(open_: NDArray, high: NDArray, low: NDArray, close: NDArray) -> float:
    """Morning star (three-bar bullish reversal)."""
    if len(open_) < 3:
        return 0.0

    body = _body(open_, close)
    rng = _range(high, low)

    # Bar 1: large bearish bar
    bar1_bearish = _is_bearish(open_, close)[-3]
    bar1_large = body[-3] > 0.6 * rng[-3] if rng[-3] != 0 else False

    # Bar 2: small body (star)
    bar2_small = body[-2] < 0.3 * rng[-2] if rng[-2] != 0 else False

    # Bar 3: large bullish bar closing above bar-1 midpoint
    bar3_bullish = _is_bullish(open_, close)[-1]
    bar3_large = body[-1] > 0.6 * rng[-1] if rng[-1] != 0 else False
    midpoint = (open_[-3] + close[-3]) / 2.0
    bar3_above_mid = close[-1] > midpoint

    if bar1_bearish and bar1_large and bar2_small and bar3_bullish and bar3_large and bar3_above_mid:
        return 1.0
    return 0.0


def _evening_star(open_: NDArray, high: NDArray, low: NDArray, close: NDArray) -> float:
    """Evening star (three-bar bearish reversal)."""
    if len(open_) < 3:
        return 0.0

    body = _body(open_, close)
    rng = _range(high, low)

    bar1_bullish = _is_bullish(open_, close)[-3]
    bar1_large = body[-3] > 0.6 * rng[-3] if rng[-3] != 0 else False

    bar2_small = body[-2] < 0.3 * rng[-2] if rng[-2] != 0 else False

    bar3_bearish = _is_bearish(open_, close)[-1]
    bar3_large = body[-1] > 0.6 * rng[-1] if rng[-1] != 0 else False
    midpoint = (open_[-3] + close[-3]) / 2.0
    bar3_below_mid = close[-1] < midpoint

    if bar1_bullish and bar1_large and bar2_small and bar3_bearish and bar3_large and bar3_below_mid:
        return -1.0
    return 0.0


def _three_white_soldiers(
    open_: NDArray, high: NDArray, low: NDArray, close: NDArray
) -> float:
    """Three white soldiers (three consecutive bullish bars with higher closes)."""
    if len(open_) < 3:
        return 0.0

    body = _body(open_, close)
    rng = _range(high, low)

    for offset in (-3, -2, -1):
        if not _is_bullish(open_, close)[offset]:
            return 0.0
        if rng[offset] == 0:
            return 0.0
        if body[offset] < 0.5 * rng[offset]:
            return 0.0

    if close[-2] > close[-3] and close[-1] > close[-2]:
        if open_[-2] > open_[-3] and open_[-1] > open_[-2]:
            return 1.0
    return 0.0


def _three_black_crows(
    open_: NDArray, high: NDArray, low: NDArray, close: NDArray
) -> float:
    """Three black crows (three consecutive bearish bars with lower closes)."""
    if len(open_) < 3:
        return 0.0

    body = _body(open_, close)
    rng = _range(high, low)

    for offset in (-3, -2, -1):
        if not _is_bearish(open_, close)[offset]:
            return 0.0
        if rng[offset] == 0:
            return 0.0
        if body[offset] < 0.5 * rng[offset]:
            return 0.0

    if close[-2] < close[-3] and close[-1] < close[-2]:
        if open_[-2] < open_[-3] and open_[-1] < open_[-2]:
            return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PATTERN_FUNCS: dict[str, callable] = {
    "doji": _doji,
    "hammer": _hammer,
    "engulfing": _engulfing,
    "morning_star": _morning_star,
    "evening_star": _evening_star,
    "three_white_soldiers": _three_white_soldiers,
    "three_black_crows": _three_black_crows,
}


def detect_patterns(
    open_: NDArray,
    high: NDArray,
    low: NDArray,
    close: NDArray,
) -> dict[str, float]:
    """Evaluate all candlestick patterns on the latest bar(s).

    Parameters
    ----------
    open_, high, low, close : NDArray
        OHLC numpy arrays of at least 3 elements for multi-bar patterns.

    Returns
    -------
    dict[str, float]
        Mapping of pattern name to score in [-1, 1].
    """
    results: dict[str, float] = {}
    for name, fn in _PATTERN_FUNCS.items():
        try:
            results[name] = float(fn(open_, high, low, close))
        except (IndexError, ValueError):
            results[name] = 0.0
    return results
