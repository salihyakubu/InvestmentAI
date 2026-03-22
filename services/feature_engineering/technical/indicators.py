"""Technical analysis indicators implemented with pure numpy.

Every function accepts numpy arrays and returns numpy arrays.
NaN values are used for the warm-up period where an indicator
cannot yet be computed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(series: NDArray, period: int) -> NDArray:
    """Simple moving average using a cumulative-sum trick for O(n) speed."""
    out = np.full_like(series, np.nan, dtype=np.float64)
    if len(series) < period:
        return out
    cumsum = np.cumsum(series, dtype=np.float64)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    out[period - 1 :] = cumsum[period - 1 :] / period
    return out


def ema(series: NDArray, period: int) -> NDArray:
    """Exponential moving average (Wilder smoothing multiplier)."""
    out = np.empty_like(series, dtype=np.float64)
    out[:] = np.nan
    if len(series) < period:
        return out
    alpha = 2.0 / (period + 1)
    # Seed with the SMA of the first *period* values.
    out[period - 1] = np.mean(series[:period])
    for i in range(period, len(series)):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


# Convenience wrappers for commonly-used SMA/EMA periods.

def sma_5(series: NDArray) -> NDArray:
    return sma(series, 5)

def sma_10(series: NDArray) -> NDArray:
    return sma(series, 10)

def sma_20(series: NDArray) -> NDArray:
    return sma(series, 20)

def sma_50(series: NDArray) -> NDArray:
    return sma(series, 50)

def sma_200(series: NDArray) -> NDArray:
    return sma(series, 200)

def ema_5(series: NDArray) -> NDArray:
    return ema(series, 5)

def ema_10(series: NDArray) -> NDArray:
    return ema(series, 10)

def ema_20(series: NDArray) -> NDArray:
    return ema(series, 20)

def ema_50(series: NDArray) -> NDArray:
    return ema(series, 50)

def ema_200(series: NDArray) -> NDArray:
    return ema(series, 200)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    close: NDArray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[NDArray, NDArray, NDArray]:
    """MACD indicator.

    Returns
    -------
    macd_line, signal_line, histogram
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(close: NDArray, period: int = 14) -> NDArray:
    """Relative Strength Index (Wilder smoothing)."""
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period + 1:
        return out

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def bollinger(
    close: NDArray, period: int = 20, num_std: float = 2.0
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Bollinger Bands.

    Returns
    -------
    upper, middle, lower, width, pct_b
    """
    middle = sma(close, period)

    # Rolling standard deviation via cumulative sums.
    std = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period - 1, len(close)):
        std[i] = np.std(close[i - period + 1 : i + 1], ddof=0)

    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / np.where(middle != 0, middle, np.nan)
    band_range = upper - lower
    pct_b = np.where(band_range != 0, (close - lower) / band_range, np.nan)

    return upper, middle, lower, width, pct_b


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------


def atr(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Average True Range using Wilder smoothing."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2:
        return out

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ),
    )

    if n < period:
        return out

    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period

    return out


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------


def stochastic(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[NDArray, NDArray]:
    """Stochastic %K and %D."""
    n = len(close)
    k = np.full(n, np.nan, dtype=np.float64)

    for i in range(k_period - 1, n):
        hh = np.max(high[i - k_period + 1 : i + 1])
        ll = np.min(low[i - k_period + 1 : i + 1])
        rng = hh - ll
        k[i] = 100.0 * (close[i] - ll) / rng if rng != 0 else 50.0

    d = sma(k, d_period)
    return k, d


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------


def williams_r(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Williams %R oscillator (-100 to 0)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1 : i + 1])
        ll = np.min(low[i - period + 1 : i + 1])
        rng = hh - ll
        out[i] = -100.0 * (hh - close[i]) / rng if rng != 0 else -50.0
    return out


# ---------------------------------------------------------------------------
# On-Balance Volume
# ---------------------------------------------------------------------------


def obv(close: NDArray, volume: NDArray) -> NDArray:
    """On-Balance Volume."""
    out = np.empty_like(close, dtype=np.float64)
    out[0] = volume[0]
    direction = np.sign(np.diff(close))
    out[1:] = volume[1:] * direction
    return np.cumsum(out)


# ---------------------------------------------------------------------------
# VWAP deviation
# ---------------------------------------------------------------------------


def vwap_deviation(close: NDArray, volume: NDArray, vwap: NDArray) -> NDArray:
    """Percentage deviation of close from VWAP."""
    return np.where(vwap != 0, (close - vwap) / vwap, 0.0)


# ---------------------------------------------------------------------------
# Volume / SMA ratio
# ---------------------------------------------------------------------------


def volume_sma_ratio(volume: NDArray, period: int = 20) -> NDArray:
    """Ratio of current volume to its SMA."""
    vol_sma = sma(volume, period)
    return np.where(vol_sma != 0, volume / vol_sma, np.nan)


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------


def adx(high: NDArray, low: NDArray, close: NDArray, period: int = 14) -> NDArray:
    """Average Directional Index using Wilder smoothing."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period * 2:
        return out

    # True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )

    # Directional movement
    up_move = np.diff(high)
    down_move = -np.diff(low)  # previous low - current low simplified
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing for TR, +DM, -DM
    atr_val = np.mean(tr[1 : period + 1])
    plus_dm_smooth = np.mean(plus_dm[:period])
    minus_dm_smooth = np.mean(minus_dm[:period])

    plus_di_arr = np.full(n, np.nan, dtype=np.float64)
    minus_di_arr = np.full(n, np.nan, dtype=np.float64)
    dx_arr = np.full(n, np.nan, dtype=np.float64)

    if atr_val != 0:
        plus_di_arr[period] = 100.0 * plus_dm_smooth / atr_val
        minus_di_arr[period] = 100.0 * minus_dm_smooth / atr_val
    else:
        plus_di_arr[period] = 0.0
        minus_di_arr[period] = 0.0

    di_sum = plus_di_arr[period] + minus_di_arr[period]
    dx_arr[period] = (
        100.0 * abs(plus_di_arr[period] - minus_di_arr[period]) / di_sum
        if di_sum != 0
        else 0.0
    )

    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        plus_dm_smooth = (plus_dm_smooth * (period - 1) + plus_dm[i - 1]) / period
        minus_dm_smooth = (minus_dm_smooth * (period - 1) + minus_dm[i - 1]) / period

        if atr_val != 0:
            plus_di_arr[i] = 100.0 * plus_dm_smooth / atr_val
            minus_di_arr[i] = 100.0 * minus_dm_smooth / atr_val
        else:
            plus_di_arr[i] = 0.0
            minus_di_arr[i] = 0.0

        di_sum = plus_di_arr[i] + minus_di_arr[i]
        dx_arr[i] = (
            100.0 * abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum
            if di_sum != 0
            else 0.0
        )

    # Smooth the DX to get ADX
    first_adx_idx = period * 2
    if first_adx_idx < n:
        out[first_adx_idx] = np.nanmean(dx_arr[period : first_adx_idx + 1])
        for i in range(first_adx_idx + 1, n):
            out[i] = (out[i - 1] * (period - 1) + dx_arr[i]) / period

    return out


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------


def cci(high: NDArray, low: NDArray, close: NDArray, period: int = 20) -> NDArray:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, period)

    # Mean absolute deviation
    mad = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period - 1, len(close)):
        window = tp[i - period + 1 : i + 1]
        mad[i] = np.mean(np.abs(window - tp_sma[i]))

    denom = 0.015 * mad
    return np.where(denom != 0, (tp - tp_sma) / denom, 0.0)


# ---------------------------------------------------------------------------
# MFI (Money Flow Index)
# ---------------------------------------------------------------------------


def mfi(
    high: NDArray,
    low: NDArray,
    close: NDArray,
    volume: NDArray,
    period: int = 14,
) -> NDArray:
    """Money Flow Index (volume-weighted RSI variant)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    tp = (high + low + close) / 3.0
    raw_mf = tp * volume

    for i in range(period, n):
        pos_flow = 0.0
        neg_flow = 0.0
        for j in range(i - period + 1, i + 1):
            if tp[j] > tp[j - 1]:
                pos_flow += raw_mf[j]
            elif tp[j] < tp[j - 1]:
                neg_flow += raw_mf[j]

        if neg_flow == 0:
            out[i] = 100.0
        else:
            mf_ratio = pos_flow / neg_flow
            out[i] = 100.0 - 100.0 / (1.0 + mf_ratio)

    return out


# ---------------------------------------------------------------------------
# ROC (Rate of Change)
# ---------------------------------------------------------------------------


def roc(close: NDArray, period: int = 12) -> NDArray:
    """Rate of Change (percent)."""
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) <= period:
        return out
    prev = close[:-period]
    out[period:] = np.where(prev != 0, (close[period:] - prev) / prev * 100.0, 0.0)
    return out


# ---------------------------------------------------------------------------
# Historical Volatility
# ---------------------------------------------------------------------------


def hist_volatility(close: NDArray, period: int) -> NDArray:
    """Annualised historical volatility from log returns.

    Annualisation factor assumes 252 trading days.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    log_ret = np.log(close[1:] / close[:-1])
    for i in range(period, len(log_ret) + 1):
        window = log_ret[i - period : i]
        out[i] = np.std(window, ddof=1) * np.sqrt(252)

    return out


def hist_volatility_10(close: NDArray) -> NDArray:
    return hist_volatility(close, 10)

def hist_volatility_30(close: NDArray) -> NDArray:
    return hist_volatility(close, 30)

def hist_volatility_60(close: NDArray) -> NDArray:
    return hist_volatility(close, 60)
