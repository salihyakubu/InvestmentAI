"""Cross-sectional factors built from perpetual funding rates.

Everything this platform has measured so far was computed from prices, and
none of it was tradeable: the live 5-minute signal has no edge, cross-
sectional reversal is real but costs ~11x what it pays, and no turnover
setting rescues it.

Funding is a different input. It is the periodic payment between perpetual
longs and shorts, so it measures POSITIONING -- how crowded a side is and how
much it is paying to stay there. Two properties make it worth testing after
three price-based nulls:

* It is not derivable from the price series, so it is not the same
  information every other participant is already ranking.
* It is stamped every 8 hours, so factors built on it move slowly BY
  CONSTRUCTION. Turnover was the binding constraint on everything found so
  far, and damping a fast signal after the fact has already been tested and
  failed. This attacks the constraint at the source.

The economic prior, stated before looking: persistently positive funding
means longs are paying to hold, which marks crowded long positioning. Crowded
positioning is associated with subsequent underperformance, so the signal is
NEGATED funding. That direction is declared here rather than discovered by
trying both, because trying both doubles the search and halves the meaning of
whatever comes out.

Every factor is z-scored cross-sectionally and evaluated by the same harness
as the price factors, so the monotonicity, tail-dominance, holdout and
deflated-Sharpe guards all apply unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np

from services.backtesting.cross_section import cross_sectional_zscore

# Funding settles every 8 hours on Binance.
FUNDING_INTERVAL_HOURS = 8

# A constant series does not have a standard deviation of exactly zero after
# floating-point accumulation -- np.nanstd of a constant 0.001 returns ~6.5e-19.
# Dividing by that produces a meaningless z-score of order 1, and because these
# scores feed a cross-sectional RANKING, one numerically degenerate contract
# can land at the top or bottom of the book on pure noise. Anything below this
# floor is treated as constant. Funding rates are ratios of order 1e-4, so a
# genuine standard deviation is many orders of magnitude above it.
_MIN_STD = 1e-9


def trailing_mean(values: np.ndarray, lookback: int) -> np.ndarray:
    """Mean of the past *lookback* rows, inclusive of the current one.

    Uses only rows at or before t. NaN-tolerant so contracts that list
    part-way through the sample contribute as soon as they have history.
    """
    out = np.full_like(values, np.nan, dtype=float)
    for t in range(values.shape[0]):
        lo = max(0, t - lookback + 1)
        window = values[lo : t + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[t] = np.nanmean(window, axis=0)
    return out


def trailing_std(values: np.ndarray, lookback: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    for t in range(values.shape[0]):
        lo = max(0, t - lookback + 1)
        window = values[lo : t + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[t] = np.nanstd(window, axis=0)
    return out


def funding_zscore(funding: np.ndarray, lookback: int) -> np.ndarray:
    """Funding relative to the contract's OWN recent history.

    A raw cross-sectional rank of funding is dominated by which contracts are
    structurally expensive to hold. Normalising per contract first isolates
    the thing of interest: this asset's positioning is unusual FOR IT.
    """
    mean = trailing_mean(funding, lookback)
    std = trailing_std(funding, lookback)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(std > _MIN_STD, (funding - mean) / std, np.nan)
    return np.asarray(z, dtype=float)


def build_funding_factors(funding: np.ndarray) -> dict[str, np.ndarray]:
    """The pre-declared funding battery, z-scored per timestamp.

    Four factors, not forty. Each additional one inflates the multiple-testing
    penalty charged against whatever survives, and the point of declaring them
    up front is that the penalty is honest.

    Sign convention: every factor is oriented so that POSITIVE means "expected
    to outperform". Crowded longs (high funding) are expected to
    underperform, hence the negations.
    """
    # 24h and 72h are 3 and 9 funding settlements -- enough to smooth a single
    # noisy stamp without blurring away a genuine positioning shift.
    factors: dict[str, np.ndarray] = {
        # Crowded longs pay to stay long; fade them.
        "funding_level": -funding,
        # Sustained crowding rather than a one-stamp spike.
        "funding_carry_24h": -trailing_mean(funding, 24),
        "funding_carry_72h": -trailing_mean(funding, 72),
        # Crowding that is unusual FOR THIS CONTRACT, not merely expensive.
        "funding_zscore_7d": -funding_zscore(funding, 24 * 7),
    }
    return {name: cross_sectional_zscore(v) for name, v in factors.items()}


def combined_factors(
    close: np.ndarray, volume: np.ndarray, funding: np.ndarray
) -> dict[str, np.ndarray]:
    """Funding battery plus the price battery, evaluated side by side.

    Keeping the price factors in the same run is deliberate: they are the
    control. If funding factors score no better than reversal did on the same
    contracts over the same window, the new data source has added nothing and
    the honest conclusion is that it added nothing.
    """
    from services.backtesting.cross_section import build_factors

    combined = build_factors(close, volume)
    combined.update(build_funding_factors(funding))
    return combined
