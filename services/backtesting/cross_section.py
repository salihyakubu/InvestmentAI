"""Cross-sectional edge research: rank symbols against each other.

The platform predicts each symbol independently and asks "will this go up?".
Measured across its own 60,732 live predictions, that has no edge at the
horizon it trades. This module tests a different question, the one most
retail-reachable edge actually lives in: **at this moment, which symbols are
attractive RELATIVE TO THE OTHERS?**

Why that changes the odds:

* Absolute direction is dominated by market beta, which no 5-minute model is
  going to forecast. Ranking is beta-neutral by construction, so a weak
  relative signal is not drowned by it.
* Every timestamp contributes one observation per symbol rather than one
  total. A 300-symbol universe buys ~300x the cross-sectional sample of the
  5-symbol book, which is the only way to get statistical power in weeks
  instead of years.

The evaluation deliberately reuses the same standards as ``live_signal``:
de-overlapped periods, cost ladders, deflated Sharpe against a declared trial
count, and a verdict that is as able to say NO EDGE as yes.

    TWO STANDING CAVEATS, both fatal if forgotten:

    1. SURVIVORSHIP. The universe cache is built from pairs listed today, so
       delisted losers are missing. Long/short is far less exposed than
       long-only, but every number here is still an upper bound.
    2. COSTS ARE BRUTAL AT THIS HORIZON. Binance spot charges ~10bps per side
       at base tier, so a full rebalance round trip is ~20bps. An hourly
       strategy must clear that EVERY HOUR. The cost ladder runs to 20bps for
       exactly this reason, and the headline verdict uses a realistic tier,
       not a flattering one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from services.backtesting.live_signal import deflated_sharpe

# Binance spot at VIP0 is ~10bps per side. A dollar-neutral rebalance trades
# both legs, so the ladder is expressed per unit of turnover.
COST_LADDER_BPS = (0.0, 2.0, 5.0, 10.0, 20.0)
DEFAULT_COST_BPS = 10.0

# Hours per year, for annualising an hourly-period Sharpe.
_HOURS_PER_YEAR = 365.25 * 24

# A cross-section thinner than this is not a cross-section.
MIN_SYMBOLS_PER_PERIOD = 20

# Mean |IC| considered meaningful in cross-sectional equity research is
# 0.02-0.05. Below 0.01 is not separable from noise at any realistic T.
MIN_MEANINGFUL_IC = 0.01


@dataclass
class FactorResult:
    """One feature's cross-sectional performance."""

    name: str
    periods: int
    mean_ic: float
    ic_std: float
    ic_t_stat: float
    information_ratio: float
    net_bps: dict[float, float]
    gross_bps: float
    turnover: float
    breakeven_bps: float
    monotonicity: float
    tail_dominance: float
    sharpe: float
    deflated_sharpe: float
    has_edge: bool


@dataclass
class CrossSectionReport:
    verdict: str
    has_edge: bool
    periods: int
    mean_universe_size: float
    factors: list[FactorResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Standardise each row (timestamp) across symbols, ignoring NaN.

    Ranking is done within a timestamp and never across time: a signal must
    say "this symbol vs its peers RIGHT NOW", and any cross-time comparison
    would smuggle in market direction.
    """
    out = np.full_like(values, np.nan, dtype=float)
    for t in range(values.shape[0]):
        row = values[t]
        finite = np.isfinite(row)
        if finite.sum() < MIN_SYMBOLS_PER_PERIOD:
            continue
        vals = row[finite]
        sd = vals.std()
        if sd == 0:
            continue
        out[t, finite] = (vals - vals.mean()) / sd
    return out


def momentum(close: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing return over *lookback* bars, using only past information."""
    out = np.full_like(close, np.nan, dtype=float)
    if lookback >= close.shape[0]:
        return out
    prior = close[:-lookback]
    out[lookback:] = close[lookback:] / prior - 1.0
    return out


def realised_volatility(close: np.ndarray, lookback: int) -> np.ndarray:
    """Standard deviation of past bar returns."""
    out = np.full_like(close, np.nan, dtype=float)
    rets = np.full_like(close, np.nan, dtype=float)
    rets[1:] = close[1:] / close[:-1] - 1.0
    for t in range(lookback, close.shape[0]):
        window = rets[t - lookback + 1 : t + 1]
        with np.errstate(invalid="ignore"):
            out[t] = np.nanstd(window, axis=0)
    return out


def turnover_ratio(volume: np.ndarray, lookback: int) -> np.ndarray:
    """Recent volume relative to its own trailing average."""
    out = np.full_like(volume, np.nan, dtype=float)
    for t in range(lookback, volume.shape[0]):
        window = volume[t - lookback + 1 : t + 1]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(window, axis=0)
        out[t] = np.where(mean > 0, volume[t] / mean, np.nan)
    return out


def build_factors(close: np.ndarray, volume: np.ndarray) -> dict[str, np.ndarray]:
    """The standard cross-sectional battery, each z-scored per timestamp.

    Deliberately a small, pre-declared set of well-known effects rather than a
    wide search: every extra factor tested inflates the multiple-testing
    penalty applied to whatever comes out on top.
    """
    factors: dict[str, np.ndarray] = {
        "reversal_1h": -momentum(close, 1),
        "reversal_6h": -momentum(close, 6),
        "momentum_24h": momentum(close, 24),
        "momentum_168h": momentum(close, 168),
        "low_volatility_24h": -realised_volatility(close, 24),
        "volume_surge_24h": turnover_ratio(volume, 24),
    }
    return {name: cross_sectional_zscore(v) for name, v in factors.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Return from t to t+horizon, market-neutral (cross-sectional demean).

    Demeaning is what makes this a test of RELATIVE skill: a signal cannot
    score by being long a rising market.
    """
    out = np.full_like(close, np.nan, dtype=float)
    if horizon >= close.shape[0]:
        return out
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    for t in range(out.shape[0]):
        row = out[t]
        finite = np.isfinite(row)
        if finite.sum() >= MIN_SYMBOLS_PER_PERIOD:
            out[t, finite] = row[finite] - row[finite].mean()
        else:
            out[t] = np.nan
    return out


def period_ic(signal: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Spearman IC per timestamp; NaN where the cross-section is too thin."""
    ics = np.full(signal.shape[0], np.nan)
    for t in range(signal.shape[0]):
        usable = np.isfinite(signal[t]) & np.isfinite(forward[t])
        if usable.sum() < MIN_SYMBOLS_PER_PERIOD:
            continue
        s, f = signal[t][usable], forward[t][usable]
        if s.std() == 0 or f.std() == 0:
            continue
        result = stats.spearmanr(s, f)
        if not np.isnan(result.statistic):
            ics[t] = float(result.statistic)
    return ics


def smooth_signal(signal: np.ndarray, span: int) -> np.ndarray:
    """Exponentially-weighted average of the signal over past periods only.

    Churn is the enemy here: a signal that re-ranks completely every hour pays
    the fee every hour. Smoothing trades a little responsiveness for a lot of
    turnover. Strictly causal -- period t uses t and earlier, never later.
    """
    if span <= 1:
        return signal
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(signal, np.nan, dtype=float)
    state = np.full(signal.shape[1], np.nan)
    for t in range(signal.shape[0]):
        row = signal[t]
        fresh = np.isfinite(row)
        seeded = fresh & ~np.isfinite(state)
        state[seeded] = row[seeded]
        update = fresh & np.isfinite(state) & ~seeded
        state[update] = alpha * row[update] + (1 - alpha) * state[update]
        out[t] = state
    return out


def long_short_returns(
    signal: np.ndarray,
    forward: np.ndarray,
    quantile: float = 0.2,
    exit_quantile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dollar-neutral top-vs-bottom portfolio returns and per-period turnover.

    Weights are equal within each leg and sum to +1 long / -1 short, so the
    result is a spread, not a market bet. Turnover is the L1 change in weights
    between consecutive rebalances -- the thing that actually pays the fee.

    *exit_quantile* adds hysteresis (the standard index-construction buffer):
    a symbol ENTERS a leg from the top *quantile* but is only dropped once it
    falls outside the wider *exit_quantile*. Without it, symbols oscillating
    across the boundary are bought and sold repeatedly for no reason -- pure
    fee. With a real signal the alpha cost is small and the turnover saving is
    large, which is exactly the trade this strategy needs.
    """
    n_periods, n_symbols = signal.shape
    returns = np.full(n_periods, np.nan)
    turnover = np.full(n_periods, np.nan)
    previous = np.zeros(n_symbols)
    prev_long: list[int] = []
    prev_short: list[int] = []

    for t in range(n_periods):
        usable = np.isfinite(signal[t]) & np.isfinite(forward[t])
        count = int(usable.sum())
        if count < MIN_SYMBOLS_PER_PERIOD:
            previous = np.zeros(n_symbols)
            prev_long, prev_short = [], []
            continue
        idx = np.flatnonzero(usable)
        order = idx[np.argsort(signal[t][idx])]
        k = max(1, int(round(count * quantile)))

        if exit_quantile is None:
            longs = list(order[-k:])
            shorts = list(order[:k])
        else:
            keep = max(k, int(round(count * exit_quantile)))
            long_zone = set(order[-keep:].tolist())
            short_zone = set(order[:keep].tolist())
            # Incumbents that are still inside the wider zone stay put.
            longs = [s for s in prev_long if s in long_zone][:k]
            shorts = [s for s in prev_short if s in short_zone][:k]
            for candidate in reversed(order.tolist()):  # strongest first
                if len(longs) >= k:
                    break
                if candidate not in longs and candidate not in shorts:
                    longs.append(candidate)
            for candidate in order.tolist():  # weakest first
                if len(shorts) >= k:
                    break
                if candidate not in shorts and candidate not in longs:
                    shorts.append(candidate)

        weights = np.zeros(n_symbols)
        if longs:
            weights[longs] = 1.0 / len(longs)
        if shorts:
            weights[shorts] = -1.0 / len(shorts)
        returns[t] = float(np.dot(weights[usable], forward[t][usable]))
        turnover[t] = float(np.abs(weights - previous).sum())
        previous = weights
        prev_long, prev_short = longs, shorts
    return returns, turnover


def quantile_profile(
    signal: np.ndarray, forward: np.ndarray, n_buckets: int = 5
) -> list[float]:
    """Mean relative return per signal bucket, low signal to high.

    The most bug-resistant evidence available: a real factor produces a
    monotone ladder. A spread driven by one extreme bucket while the middle
    is flat is a tail artifact, and it is invisible in the headline spread.
    """
    sums: list[list[float]] = [[] for _ in range(n_buckets)]
    for t in range(signal.shape[0]):
        usable = np.isfinite(signal[t]) & np.isfinite(forward[t])
        if usable.sum() < MIN_SYMBOLS_PER_PERIOD:
            continue
        idx = np.flatnonzero(usable)
        order = idx[np.argsort(signal[t][idx])]
        for bucket, members in enumerate(np.array_split(order, n_buckets)):
            if members.size:
                sums[bucket].append(float(forward[t][members].mean()))
    return [float(np.mean(v) * 1e4) if v else 0.0 for v in sums]


def profile_monotonicity(profile: list[float]) -> float:
    """Rank correlation of bucket index against its mean return."""
    if len(profile) < 3:
        return 0.0
    values = np.array(profile)
    if values.std() == 0:
        return 0.0
    result = stats.spearmanr(np.arange(values.size), values)
    return 0.0 if np.isnan(result.statistic) else float(result.statistic)


def tail_dominance(profile: list[float]) -> float:
    """Share of the whole spread carried by the single largest bucket step.

    Rank correlation alone is too blunt to catch a tail artifact: on five
    buckets one out-of-order pair still scores 0.9, which is exactly what
    momentum_168h scored while 80% of its spread came from one jump into the
    top bucket. This measures that directly. Near 1/(n-1) the ladder rises
    evenly; near 1.0 a single bucket IS the strategy.
    """
    if len(profile) < 3:
        return 1.0
    values = np.array(profile)
    spread = values[-1] - values[0]
    if spread == 0:
        return 1.0
    steps = np.abs(np.diff(values))
    return float(steps.max() / abs(spread))


def evaluate_factor(
    name: str,
    signal: np.ndarray,
    forward: np.ndarray,
    horizon: int,
    cost_bps: float,
    n_trials: int,
) -> FactorResult:
    """Score one factor on IC, net-of-cost spread and a deflated Sharpe."""
    # Subsample to non-overlapping rebalances FIRST, then build the
    # portfolio. Consecutive hourly timestamps share the same forward window
    # and would multiply-count one bet; and turnover has to be the weight
    # change between successive REBALANCES, which is what actually pays the
    # fee -- measuring it bar-to-bar and subsampling afterwards understates
    # the cost of a slow strategy.
    sampled_signal = signal[::horizon]
    sampled_forward = forward[::horizon]
    ics = period_ic(sampled_signal, sampled_forward)
    ics = ics[np.isfinite(ics)]
    gross, turn = long_short_returns(sampled_signal, sampled_forward)
    keep = np.isfinite(gross)
    gross, turn = gross[keep], turn[keep]

    mean_ic = float(ics.mean()) if ics.size else 0.0
    ic_std = float(ics.std(ddof=1)) if ics.size > 1 else 0.0
    ic_t = float(mean_ic / ic_std * np.sqrt(ics.size)) if ic_std > 0 else 0.0
    ir = float(mean_ic / ic_std) if ic_std > 0 else 0.0

    mean_turnover = float(turn.mean()) if turn.size else 0.0
    net_by_cost = {
        bps: float((gross - turn * (bps / 1e4)).mean() * 1e4) if gross.size else 0.0
        for bps in COST_LADDER_BPS
    }
    priced = gross - turn * (cost_bps / 1e4) if gross.size else np.array([])
    sd = float(priced.std(ddof=1)) if priced.size > 1 else 0.0
    per_period = float(priced.mean() / sd) if sd > 0 else 0.0
    sharpe = per_period * np.sqrt(_HOURS_PER_YEAR / horizon) if sd > 0 else 0.0
    dsr = deflated_sharpe(per_period, priced.size, n_trials)

    # The single most decision-relevant number: the per-unit-turnover cost at
    # which this factor stops making money. Compare it to what the venue
    # actually charges. A factor with a huge IC and a huge turnover can have a
    # breakeven below one basis point, which means it is unharvestable by
    # anyone crossing the spread -- the alpha IS the liquidity premium.
    gross_bps = net_by_cost.get(0.0, 0.0)
    breakeven = (gross_bps / mean_turnover) if mean_turnover > 0 else 0.0
    profile = quantile_profile(sampled_signal, sampled_forward, n_buckets=10)
    mono = profile_monotonicity(profile)
    tail = tail_dominance(profile)

    has_edge = (
        abs(mean_ic) > MIN_MEANINGFUL_IC
        and abs(ic_t) > 3.0
        and net_by_cost.get(cost_bps, 0.0) > 0
        and dsr > 0.95
        # A spread produced by one extreme bucket is a tail artifact, not a
        # factor. Both guards are needed: momentum_168h scored 0.9 on
        # 5-bucket monotonicity while one jump carried 80% of its spread,
        # and it showed the best breakeven cost in the whole horizon sweep.
        and abs(mono) >= 0.8
        and tail < 0.5
    )
    return FactorResult(
        name=name,
        periods=int(ics.size),
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_t_stat=ic_t,
        information_ratio=ir,
        net_bps=net_by_cost,
        gross_bps=net_by_cost.get(0.0, 0.0),
        turnover=mean_turnover,
        breakeven_bps=float(breakeven),
        monotonicity=mono,
        tail_dominance=tail,
        sharpe=float(sharpe),
        deflated_sharpe=dsr,
        has_edge=has_edge,
    )


def evaluate(
    close: np.ndarray,
    volume: np.ndarray,
    horizon: int = 1,
    cost_bps: float = DEFAULT_COST_BPS,
    factors: dict[str, np.ndarray] | None = None,
) -> CrossSectionReport:
    """Test the factor battery cross-sectionally. May well return NO EDGE."""
    signals = factors if factors is not None else build_factors(close, volume)
    forward = forward_returns(close, horizon)
    # Every factor tested is a trial; the deflated Sharpe pays for the whole
    # battery, not just the one that happened to win.
    n_trials = max(1, len(signals))

    results = [
        evaluate_factor(name, sig, forward, horizon, cost_bps, n_trials)
        for name, sig in sorted(signals.items())
    ]
    results.sort(key=lambda r: abs(r.ic_t_stat), reverse=True)

    per_period_universe = np.isfinite(close).sum(axis=1)
    mean_universe = float(per_period_universe[per_period_universe > 0].mean())
    periods = max((r.periods for r in results), default=0)

    passed = [r for r in results if r.has_edge]
    gross_positive = [r for r in results if r.gross_bps > 0 and abs(r.ic_t_stat) > 3.0]

    notes = [
        f"universe averages {mean_universe:.0f} symbols per period; "
        f"{periods} non-overlapping rebalances at a {horizon}-bar horizon",
        f"deflated Sharpe charged for {n_trials} factors tested; bar is 0.95",
        "SURVIVORSHIP: universe is today's survivors -- every figure is an "
        "upper bound",
        f"headline cost {cost_bps:.0f}bps per unit turnover (Binance spot VIP0 "
        "is ~10bps per side)",
    ]

    if not results or periods < 100:
        verdict = "insufficient history to judge the cross-section"
        has_edge = False
    elif passed:
        verdict = (
            f"CROSS-SECTIONAL EDGE in {len(passed)}/{len(results)} factors "
            f"(best: {passed[0].name}, IC {passed[0].mean_ic:+.4f}, "
            f"t={passed[0].ic_t_stat:+.1f}) -> confirm out-of-sample before sizing"
        )
        has_edge = True
    elif gross_positive:
        verdict = (
            f"SIGNAL BUT NOT PROFITABLE: {len(gross_positive)} factor(s) have a "
            f"real gross IC (best {gross_positive[0].name}, t="
            f"{gross_positive[0].ic_t_stat:+.1f}) but none survives "
            f"{cost_bps:.0f}bps of cost"
        )
        has_edge = False
    else:
        verdict = f"NO CROSS-SECTIONAL EDGE across {len(results)} factors tested"
        has_edge = False

    return CrossSectionReport(
        verdict=verdict,
        has_edge=has_edge,
        periods=periods,
        mean_universe_size=mean_universe,
        factors=results,
        notes=notes,
    )


def format_report(report: CrossSectionReport, cost_bps: float = DEFAULT_COST_BPS) -> str:
    lines = ["=" * 88, "CROSS-SECTIONAL EDGE -- BROAD UNIVERSE", "=" * 88, ""]
    lines.append(f"VERDICT: {report.verdict}")
    lines.append("")
    lines.append(
        f"  {'factor':<20} {'periods':>8} {'mean IC':>9} {'t-stat':>8} "
        f"{'gross':>8} {'net':>8} {'turn':>6} {'breakeven':>10} {'mono':>6} {'tail':>6}"
    )
    for r in report.factors:
        lines.append(
            f"  {r.name:<20} {r.periods:>8} {r.mean_ic:>+9.4f} {r.ic_t_stat:>+8.2f} "
            f"{r.gross_bps:>+8.2f} {r.net_bps.get(cost_bps, 0.0):>+8.2f} "
            f"{r.turnover:>6.2f} {r.breakeven_bps:>9.2f}bps {r.monotonicity:>+6.2f} "
            f"{r.tail_dominance:>6.2f}"
        )
    lines.append("")
    if report.factors:
        best = report.factors[0]
        lines.append(f"  cost sensitivity for the strongest factor ({best.name}):")
        for bps, net in sorted(best.net_bps.items()):
            tag = "  <- gross" if bps == 0 else ""
            lines.append(f"      {bps:5.1f}bps -> {net:+8.3f} bps per rebalance{tag}")
        lines.append("")
    lines.append("  notes:")
    for note in report.notes:
        lines.append(f"    - {note}")
    lines.append("=" * 88)
    return "\n".join(lines)


@dataclass(frozen=True)
class FrontierPoint:
    """One turnover-reduction configuration, scored in and out of sample."""

    smoothing: int
    buffer: float | None
    is_gross_bps: float
    is_turnover: float
    is_net_bps: float
    oos_gross_bps: float
    oos_turnover: float
    oos_net_bps: float
    breakeven_bps: float


def turnover_frontier(
    signal: np.ndarray,
    forward: np.ndarray,
    split: int,
    cost_bps: float,
    spans: tuple[int, ...] = (1, 3, 6, 12, 24),
    buffers: tuple[float | None, ...] = (None, 0.3, 0.4, 0.5),
) -> list[FrontierPoint]:
    """Sweep turnover-reduction settings, scoring EVERY point in and out of sample.

    Turnover is the binding constraint on a high-churn factor, so the obvious
    move is to damp it: smooth the signal, and buffer leg membership so
    symbols oscillating across the boundary are not churned. Both reliably
    cut turnover.

    The reason both halves are reported for every configuration, rather than
    the best in-sample one being carried forward, is that this sweep is a
    parameter search over ~20 settings. Picking the winner on the first half
    and quoting its number is precisely how a backtest is overfitted. If a
    configuration is real, it is profitable on data that had no say in
    choosing it.
    """
    points: list[FrontierPoint] = []
    for span in spans:
        smoothed = smooth_signal(signal, span) if span > 1 else signal
        for buffer in buffers:
            row: list[float] = []
            for lo, hi in ((0, split), (split, signal.shape[0])):
                returns, turns = long_short_returns(
                    smoothed[lo:hi], forward[lo:hi], exit_quantile=buffer
                )
                keep = np.isfinite(returns)
                returns, turns = returns[keep], turns[keep]
                if returns.size == 0:
                    row.extend([0.0, 0.0, 0.0])
                    continue
                gross = float(returns.mean() * 1e4)
                turnover = float(turns.mean())
                net = float((returns - turns * (cost_bps / 1e4)).mean() * 1e4)
                row.extend([gross, turnover, net])
            breakeven = row[0] / row[1] if row[1] > 0 else 0.0
            points.append(
                FrontierPoint(
                    smoothing=span,
                    buffer=buffer,
                    is_gross_bps=row[0],
                    is_turnover=row[1],
                    is_net_bps=row[2],
                    oos_gross_bps=row[3],
                    oos_turnover=row[4],
                    oos_net_bps=row[5],
                    breakeven_bps=breakeven,
                )
            )
    return points


def format_frontier(points: list[FrontierPoint], cost_bps: float) -> str:
    """Render the frontier, with the in-sample/holdout disagreement visible."""
    lines = [
        f"  turnover reduction frontier at {cost_bps:.0f}bps per unit turnover",
        "  (a configuration is only real if the HOLDOUT column is positive)",
        "",
        f"  {'smooth':>7} {'buffer':>7} {'IS gross':>9} {'IS turn':>8} {'IS net':>8}"
        f" | {'OOS gross':>10} {'OOS turn':>9} {'OOS net':>8}",
    ]
    for p in points:
        tag = "none" if p.buffer is None else f"{p.buffer:.1f}"
        lines.append(
            f"  {p.smoothing:>7d} {tag:>7} {p.is_gross_bps:>9.3f} "
            f"{p.is_turnover:>8.2f} {p.is_net_bps:>8.3f} | "
            f"{p.oos_gross_bps:>10.3f} {p.oos_turnover:>9.2f} {p.oos_net_bps:>8.3f}"
        )
    is_winners = [p for p in points if p.is_net_bps > 0]
    oos_winners = [p for p in points if p.oos_net_bps > 0]
    lines.append("")
    lines.append(
        f"  {len(is_winners)}/{len(points)} configurations are profitable IN SAMPLE; "
        f"{len(oos_winners)}/{len(points)} survive on the HOLDOUT."
    )
    if is_winners and not oos_winners:
        lines.append(
            "  NONE survives. The in-sample winners are a parameter search "
            "fitting noise -- selecting the best of them would have produced a "
            "profitable-looking strategy that loses money."
        )
    return "\n".join(lines)


def load_universe(path: str) -> dict[str, Any]:
    """Load the local research cache written by scripts/build_universe.py."""
    data = np.load(path, allow_pickle=False)
    return {
        "times": data["times"],
        "symbols": data["symbols"],
        "close": data["close"],
        "volume": data["volume"],
    }
