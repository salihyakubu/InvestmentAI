"""Market-regime (auxiliary) features with strict train/serve parity.

The core platform invariant is that a feature must carry the SAME numeric value
whether it is computed live (as a bar closes) or replayed during training. The
technical indicators achieve this by recomputing from the identical OHLCV window
both times. Slow-moving *market-regime* metrics -- crypto funding rate, the
Crypto Fear & Greed index, the VIX close, and the SPY daily return -- are not
derivable from a symbol's own price window, so they are supplied through an
:class:`AuxFeatureProvider`:

- **Live serving** uses :class:`LiveAuxProvider`, an in-memory snapshot of the
  latest known metrics kept fresh by the ingestion service. ``features_asof``
  ignores the timestamp and returns the current snapshot (live == latest).
- **Training replay** uses :class:`HistoricalAuxProvider`, built from the
  ``aux_market_state`` table, which does a real AS-OF lookup: the latest value
  recorded at-or-before the window-end bar's timestamp. This reproduces what the
  live snapshot *would have held* at that historical moment (no look-ahead),
  which is exactly what parity requires.

The feature KEY SET (:data:`AUX_FEATURE_NAMES`) is FIXED and STABLE regardless of
data availability or whether any provider is wired: missing data maps to ``0.0``,
never to a missing key. This keeps the emitted feature vector's schema constant
so adding aux features stays backward-compatible with already-trained models
(they map features by name and ignore unknown keys).
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Fixed feature-key set
# ---------------------------------------------------------------------------

# The aux feature keys, in SORTED order (the dataset builder and the
# continuous-learning feature-drift path both assume sorted names). This set is
# STABLE: it never changes with data availability, provider wiring, or asset
# class. Every provider returns EXACTLY these keys.
AUX_FEATURE_NAMES: list[str] = [
    "aux_fear_greed",
    "aux_funding_rate",
    "aux_spy_daily_return",
    "aux_vix_close",
]

# ---------------------------------------------------------------------------
# Metric identifiers (as stored in the aux_market_state.metric column)
# ---------------------------------------------------------------------------

METRIC_FEAR_GREED = "fear_greed"
METRIC_FUNDING_RATE = "funding_rate"
METRIC_SPY_RETURN = "spy_daily_return"
METRIC_VIX_CLOSE = "vix_close"

# Global (non-per-symbol) metrics are stored with this empty-symbol sentinel.
GLOBAL_SYMBOL = ""

# Which aux feature key each stored metric feeds.
_FEAR_GREED_FEATURE = "aux_fear_greed"
_FUNDING_FEATURE = "aux_funding_rate"
_SPY_FEATURE = "aux_spy_daily_return"
_VIX_FEATURE = "aux_vix_close"


def zero_aux() -> dict[str, float]:
    """Return every aux feature keyed to ``0.0``.

    Used whenever no provider is wired or no timestamp is available, so the
    emitted feature vector always carries the full, stable aux key set.
    """
    return dict.fromkeys(AUX_FEATURE_NAMES, 0.0)


@runtime_checkable
class AuxFeatureProvider(Protocol):
    """Supplies the aux feature values for a symbol as of a point in time."""

    def features_asof(self, symbol: str, ts: datetime) -> dict[str, float]:
        """Return EXACTLY :data:`AUX_FEATURE_NAMES` (``0.0`` for anything
        missing) for *symbol* as of timestamp *ts*."""
        ...


# ---------------------------------------------------------------------------
# Live provider (latest-value snapshot)
# ---------------------------------------------------------------------------


class LiveAuxProvider:
    """In-memory snapshot of the latest global + per-symbol regime metrics.

    The ingestion service (:class:`~services.data_ingestion.aux_market.
    AuxMarketService`) keeps this fresh via the ``update_*`` methods. Because
    live serving always wants the newest known value, :meth:`features_asof`
    ignores its ``ts`` argument.
    """

    def __init__(self) -> None:
        self._fear_greed: float = 0.0
        self._vix_close: float = 0.0
        self._spy_daily_return: float = 0.0
        self._funding: dict[str, float] = {}

    def update_fear_greed(self, value: float) -> None:
        self._fear_greed = float(value)

    def update_vix_close(self, value: float) -> None:
        self._vix_close = float(value)

    def update_spy_daily_return(self, value: float) -> None:
        self._spy_daily_return = float(value)

    def update_funding(self, symbol: str, value: float) -> None:
        self._funding[symbol] = float(value)

    def features_asof(self, symbol: str, ts: datetime) -> dict[str, float]:
        # ts is ignored: live serving uses the latest snapshot. Unknown-symbol
        # funding (e.g. stocks) resolves to 0.0.
        return {
            _FEAR_GREED_FEATURE: self._fear_greed,
            _FUNDING_FEATURE: self._funding.get(symbol, 0.0),
            _SPY_FEATURE: self._spy_daily_return,
            _VIX_FEATURE: self._vix_close,
        }


# ---------------------------------------------------------------------------
# Historical provider (as-of lookup)
# ---------------------------------------------------------------------------


class HistoricalAuxProvider:
    """AS-OF aux lookup over historical ``(time, metric, symbol, value)`` rows.

    Rows are pre-sorted per ``(metric, symbol)`` by time so each lookup is a
    single :func:`bisect` for the latest value recorded at-or-before the query
    timestamp. Global metrics are looked up under :data:`GLOBAL_SYMBOL`; the
    funding rate is looked up per symbol. Anything missing resolves to ``0.0``.
    """

    def __init__(self, rows: Iterable[tuple[datetime, str, str, float]]) -> None:
        grouped: dict[tuple[str, str], list[tuple[datetime, float]]] = {}
        for ts, metric, symbol, value in rows:
            grouped.setdefault((metric, symbol), []).append((ts, float(value)))

        # Store parallel (times, values) lists sorted by time for bisect.
        self._series: dict[tuple[str, str], tuple[list[datetime], list[float]]] = {}
        for key, pairs in grouped.items():
            pairs.sort(key=lambda p: p[0])
            self._series[key] = ([p[0] for p in pairs], [p[1] for p in pairs])

    def _asof(self, metric: str, symbol: str, ts: datetime) -> float:
        entry = self._series.get((metric, symbol))
        if entry is None:
            return 0.0
        times, values = entry
        # Latest value at-or-before ts. bisect_right - 1 gives the rightmost
        # index whose time is <= ts.
        idx = bisect.bisect_right(times, ts) - 1
        if idx < 0:
            return 0.0
        return values[idx]

    def features_asof(self, symbol: str, ts: datetime) -> dict[str, float]:
        return {
            _FEAR_GREED_FEATURE: self._asof(METRIC_FEAR_GREED, GLOBAL_SYMBOL, ts),
            _FUNDING_FEATURE: self._asof(METRIC_FUNDING_RATE, symbol, ts),
            _SPY_FEATURE: self._asof(METRIC_SPY_RETURN, GLOBAL_SYMBOL, ts),
            _VIX_FEATURE: self._asof(METRIC_VIX_CLOSE, GLOBAL_SYMBOL, ts),
        }
