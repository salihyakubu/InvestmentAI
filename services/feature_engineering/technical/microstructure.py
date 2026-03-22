"""Market microstructure analytics.

Provides volume profile construction, multi-period momentum, and
volatility regime classification.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def volume_profile(
    close: NDArray,
    volume: NDArray,
    bins: int = 20,
) -> dict[str, float]:
    """Build a volume-at-price profile and extract key levels.

    Parameters
    ----------
    close  : price series
    volume : corresponding volume series
    bins   : number of price buckets

    Returns
    -------
    dict with keys:
        value_area_high  - upper boundary of the value area (70 % of volume)
        value_area_low   - lower boundary of the value area
        poc              - point of control (price level with highest volume)
    """
    if len(close) == 0 or len(volume) == 0:
        return {"value_area_high": 0.0, "value_area_low": 0.0, "poc": 0.0}

    price_min, price_max = float(np.min(close)), float(np.max(close))
    if price_min == price_max:
        return {
            "value_area_high": price_max,
            "value_area_low": price_min,
            "poc": price_min,
        }

    edges = np.linspace(price_min, price_max, bins + 1)
    vol_per_bin = np.zeros(bins, dtype=np.float64)

    # Assign each bar's volume to its price bucket.
    indices = np.digitize(close, edges) - 1
    indices = np.clip(indices, 0, bins - 1)
    for i, idx in enumerate(indices):
        vol_per_bin[idx] += volume[i]

    # Point of control: bin with the most volume.
    poc_bin = int(np.argmax(vol_per_bin))
    poc = float((edges[poc_bin] + edges[poc_bin + 1]) / 2.0)

    # Value area: expand outward from POC until 70 % of total volume is covered.
    total_vol = float(np.sum(vol_per_bin))
    target = 0.70 * total_vol
    accumulated = float(vol_per_bin[poc_bin])
    lo_idx, hi_idx = poc_bin, poc_bin

    while accumulated < target and (lo_idx > 0 or hi_idx < bins - 1):
        expand_lo = vol_per_bin[lo_idx - 1] if lo_idx > 0 else -1.0
        expand_hi = vol_per_bin[hi_idx + 1] if hi_idx < bins - 1 else -1.0
        if expand_lo >= expand_hi:
            lo_idx -= 1
            accumulated += vol_per_bin[lo_idx]
        else:
            hi_idx += 1
            accumulated += vol_per_bin[hi_idx]

    value_area_low = float(edges[lo_idx])
    value_area_high = float(edges[hi_idx + 1])

    return {
        "value_area_high": value_area_high,
        "value_area_low": value_area_low,
        "poc": poc,
    }


def price_momentum(
    close: NDArray,
    periods: list[int] | None = None,
) -> dict[str, float]:
    """Compute simple price momentum (percentage return) over multiple windows.

    Returns
    -------
    dict mapping ``"momentum_{period}"`` to the percent return over that
    look-back window.
    """
    if periods is None:
        periods = [5, 10, 20]

    result: dict[str, float] = {}
    n = len(close)
    for p in periods:
        key = f"momentum_{p}"
        if n <= p or close[-p - 1] == 0:
            result[key] = 0.0
        else:
            result[key] = float((close[-1] - close[-p - 1]) / close[-p - 1] * 100.0)
    return result


def volatility_regime(
    close: NDArray,
    short_period: int = 10,
    long_period: int = 60,
) -> str:
    """Classify the current volatility regime.

    Compares the short-term realised volatility to the long-term realised
    volatility to bucket the regime into ``"low"``, ``"medium"``, or
    ``"high"``.

    Returns
    -------
    str
        One of ``"low"``, ``"medium"``, ``"high"``.
    """
    n = len(close)
    if n < long_period + 1:
        return "medium"  # not enough data to classify

    log_ret = np.log(close[1:] / close[:-1])

    short_vol = float(np.std(log_ret[-short_period:], ddof=1))
    long_vol = float(np.std(log_ret[-long_period:], ddof=1))

    if long_vol == 0:
        return "medium"

    ratio = short_vol / long_vol

    if ratio < 0.75:
        return "low"
    if ratio > 1.25:
        return "high"
    return "medium"
