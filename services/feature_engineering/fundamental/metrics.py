"""Fundamental analysis metrics derived from price data.

In production these would integrate with financial data providers
(SEC filings, earnings APIs, etc.).  For now the functions use simple
price-derived proxies that are computed once per day.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimated_market_cap(
    close: NDArray,
    shares_outstanding: float = 1.0,
) -> float:
    """Estimate market capitalisation from the latest close price.

    Parameters
    ----------
    close : price series
    shares_outstanding : number of shares (default 1.0 as placeholder)
    """
    if len(close) == 0:
        return 0.0
    return float(close[-1]) * shares_outstanding


def price_to_earnings_proxy(
    close: NDArray,
    trailing_eps: float | None = None,
) -> float:
    """Return a P/E ratio estimate.

    If ``trailing_eps`` is not supplied the function returns NaN,
    signalling the caller that fundamental data is missing.
    """
    if trailing_eps is None or trailing_eps == 0 or len(close) == 0:
        return float("nan")
    return float(close[-1]) / trailing_eps


def price_to_book_proxy(
    close: NDArray,
    book_value_per_share: float | None = None,
) -> float:
    """Return a P/B ratio estimate."""
    if book_value_per_share is None or book_value_per_share == 0 or len(close) == 0:
        return float("nan")
    return float(close[-1]) / book_value_per_share


def dividend_yield_proxy(
    close: NDArray,
    annual_dividend: float | None = None,
) -> float:
    """Return estimated dividend yield as a decimal."""
    if annual_dividend is None or len(close) == 0 or close[-1] == 0:
        return 0.0
    return annual_dividend / float(close[-1])


def earnings_yield(
    close: NDArray,
    trailing_eps: float | None = None,
) -> float:
    """Inverse of P/E (E/P), useful for cross-asset comparison."""
    pe = price_to_earnings_proxy(close, trailing_eps)
    if np.isnan(pe) or pe == 0:
        return 0.0
    return 1.0 / pe


def price_to_sales_proxy(
    close: NDArray,
    revenue_per_share: float | None = None,
) -> float:
    """Price-to-sales ratio estimate."""
    if revenue_per_share is None or revenue_per_share == 0 or len(close) == 0:
        return float("nan")
    return float(close[-1]) / revenue_per_share


def compute_fundamental_features(
    close: NDArray,
    *,
    shares_outstanding: float = 1.0,
    trailing_eps: float | None = None,
    book_value_per_share: float | None = None,
    annual_dividend: float | None = None,
    revenue_per_share: float | None = None,
) -> dict[str, float]:
    """Compute all available fundamental features and return as a flat dict.

    Missing or incalculable metrics are returned as NaN.
    """
    return {
        "fundamental_market_cap": estimated_market_cap(close, shares_outstanding),
        "fundamental_pe_ratio": price_to_earnings_proxy(close, trailing_eps),
        "fundamental_pb_ratio": price_to_book_proxy(close, book_value_per_share),
        "fundamental_dividend_yield": dividend_yield_proxy(close, annual_dividend),
        "fundamental_earnings_yield": earnings_yield(close, trailing_eps),
        "fundamental_ps_ratio": price_to_sales_proxy(close, revenue_per_share),
    }
