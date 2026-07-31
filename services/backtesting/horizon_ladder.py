"""Span every horizon, and test whether blending them beats any one of them.

Four research passes all lived between 1 hour and 48 hours -- precisely the
band where transaction costs dominate, because cost is paid per rebalance and
a fast strategy rebalances constantly. Nothing has ever been measured at the
daily-to-multi-week horizons where the classic factor premia actually live
and where cost stops being the binding constraint.

Two things are needed to close that gap honestly.

**Comparability.** Per-rebalance figures cannot be compared across horizons:
2.7 bps every hour and 2.7 bps every week differ by three orders of magnitude
per year. Everything here is annualised, and the bar size is passed
explicitly rather than assumed -- using the hourly constant on daily bars
would inflate every Sharpe by sqrt(24).

**Blending.** A book of weak, weakly-correlated signals can carry a far
better Sharpe than its best component, because the noise diversifies faster
than the signal does. That is how multi-strategy books earn their keep, and
it is the one structural idea this platform has not tested. If the blend does
not beat its best component, the components are simply too correlated -- and
that is a finding, not a failure.

The usual standards apply unchanged: chronological holdout, deflated Sharpe
against the full trial count, monotonicity and tail guards, and a verdict
that can come back negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from services.backtesting.cross_section import (
    HOURS_PER_YEAR,
    MIN_SYMBOLS_PER_PERIOD,
    FactorResult,
    evaluate_factor,
    forward_returns,
    long_short_returns,
)
from services.backtesting.live_signal import deflated_sharpe

# Below this many non-overlapping rebalances a horizon cannot be judged: a
# 30-day hold over two years of history is roughly 24 observations, and no
# statistic computed on 24 points should be allowed near a capital decision.
MIN_REBALANCES = 60


@dataclass
class LadderRung:
    """One (factor, horizon) cell, scored in and out of sample."""

    factor: str
    horizon: int
    in_sample: FactorResult
    holdout: FactorResult
    sign_held: bool
    judged: bool


@dataclass
class BlendResult:
    """A multi-horizon combination and how it compares to its parts."""

    components: list[str]
    annual_net_pct: float
    sharpe: float
    deflated_sharpe: float
    turnover: float
    best_component_sharpe: float
    beats_best_component: bool
    mean_correlation: float


@dataclass
class LadderReport:
    verdict: str
    has_edge: bool
    rungs: list[LadderRung] = field(default_factory=list)
    blend: BlendResult | None = None
    notes: list[str] = field(default_factory=list)


def evaluate_ladder(
    close: np.ndarray,
    factors: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    cost_bps: float,
    bars_per_year: float = HOURS_PER_YEAR,
    holdout_fraction: float = 0.4,
) -> list[LadderRung]:
    """Score every (factor, horizon) cell on both halves of the sample.

    Both halves for every cell, deliberately. A ladder is a large search --
    factors times horizons -- and reporting only the winning cell's in-sample
    number is the single most reliable way to produce a strategy that works
    until it is funded.
    """
    split = int(close.shape[0] * (1 - holdout_fraction))
    n_trials = max(1, len(factors) * len(horizons))
    rungs: list[LadderRung] = []

    for horizon in horizons:
        forward = forward_returns(close, horizon)
        for name, signal in sorted(factors.items()):
            in_sample = evaluate_factor(
                name, signal[:split], forward[:split], horizon,
                cost_bps, n_trials, bars_per_year,
            )
            holdout = evaluate_factor(
                name, signal[split:], forward[split:], horizon,
                cost_bps, n_trials, bars_per_year,
            )
            judged = (
                in_sample.periods >= MIN_REBALANCES
                and holdout.periods >= MIN_REBALANCES
            )
            sign_held = (
                in_sample.mean_ic * holdout.mean_ic > 0
                and abs(in_sample.mean_ic) > 0.005
            )
            rungs.append(
                LadderRung(
                    factor=name,
                    horizon=horizon,
                    in_sample=in_sample,
                    holdout=holdout,
                    sign_held=sign_held,
                    judged=judged,
                )
            )
    return rungs


def blend_returns(
    close: np.ndarray,
    factors: dict[str, np.ndarray],
    horizon: int,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-risk blend of several factors traded at one horizon.

    Signals are averaged BEFORE portfolio construction rather than after.
    Combining at the signal level lets offsetting views net out inside a
    single book, so the blend trades the disagreement once instead of paying
    to hold both sides of it -- which matters enormously when turnover is the
    binding constraint.
    """
    names = sorted(factors)
    stack = np.stack([factors[n] for n in names])
    if weights is not None:
        w = np.array([weights.get(n, 0.0) for n in names]).reshape(-1, 1, 1)
    else:
        w = np.full((len(names), 1, 1), 1.0 / len(names))
    with np.errstate(invalid="ignore"):
        combined = np.nansum(stack * w, axis=0)
    # A cell where every component is absent must stay absent, not become 0.
    combined[np.all(~np.isfinite(stack), axis=0)] = np.nan
    forward = forward_returns(close, horizon)
    return long_short_returns(combined[::horizon], forward[::horizon])


def factor_correlations(
    close: np.ndarray, factors: dict[str, np.ndarray], horizon: int
) -> float:
    """Mean pairwise correlation of the components' return streams.

    The number that decides whether blending can help at all. Near 1.0 the
    components are the same bet wearing different names and diversification
    is an illusion; near 0 the blend's Sharpe rises roughly as sqrt(n).
    """
    forward = forward_returns(close, horizon)
    streams: list[np.ndarray] = []
    for name in sorted(factors):
        returns, _ = long_short_returns(
            factors[name][::horizon], forward[::horizon]
        )
        streams.append(returns)
    if len(streams) < 2:
        return 0.0
    corrs: list[float] = []
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            both = np.isfinite(streams[i]) & np.isfinite(streams[j])
            if both.sum() < MIN_SYMBOLS_PER_PERIOD:
                continue
            a, b = streams[i][both], streams[j][both]
            if a.std() == 0 or b.std() == 0:
                continue
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs)) if corrs else 0.0


def evaluate_blend(
    close: np.ndarray,
    factors: dict[str, np.ndarray],
    horizon: int,
    cost_bps: float,
    bars_per_year: float = HOURS_PER_YEAR,
    n_trials: int = 1,
    holdout_start: int = 0,
) -> BlendResult:
    """Score the blend on the holdout and compare it to its best component."""
    window = slice(holdout_start, close.shape[0])
    sub_close = close[window]
    sub_factors = {n: f[window] for n, f in factors.items()}

    returns, turns = blend_returns(sub_close, sub_factors, horizon)
    keep = np.isfinite(returns)
    returns, turns = returns[keep], turns[keep]
    periods_per_year = bars_per_year / horizon

    if returns.size < 2:
        return BlendResult(
            components=sorted(factors), annual_net_pct=0.0, sharpe=0.0,
            deflated_sharpe=0.0, turnover=0.0, best_component_sharpe=0.0,
            beats_best_component=False, mean_correlation=0.0,
        )

    net = returns - turns * (cost_bps / 1e4)
    sd = float(net.std(ddof=1))
    per_period = float(net.mean() / sd) if sd > 0 else 0.0
    sharpe = per_period * np.sqrt(periods_per_year) if sd > 0 else 0.0

    forward = forward_returns(sub_close, horizon)
    component_sharpes = [
        evaluate_factor(
            n, f, forward, horizon, cost_bps, n_trials, bars_per_year
        ).sharpe
        for n, f in sub_factors.items()
    ]
    best = max(component_sharpes) if component_sharpes else 0.0

    return BlendResult(
        components=sorted(factors),
        annual_net_pct=float(net.mean() * 1e4 * periods_per_year / 100.0),
        sharpe=float(sharpe),
        deflated_sharpe=deflated_sharpe(per_period, net.size, n_trials),
        turnover=float(turns.mean()),
        best_component_sharpe=float(best),
        beats_best_component=bool(sharpe > best),
        mean_correlation=factor_correlations(sub_close, sub_factors, horizon),
    )


def format_ladder(rungs: list[LadderRung], blend: BlendResult | None = None) -> str:
    """Render the ladder with in-sample and holdout side by side."""
    lines = [
        "  horizon ladder -- annualised, so bands are comparable",
        "  (a rung is only real if the OOS column is positive AND the sign held)",
        "",
        f"  {'factor':<20} {'horiz':>6} {'IS ann%':>9} {'OOS ann%':>9} "
        f"{'OOS Shrp':>9} {'turn':>6} {'sign':>5} {'n':>6}",
    ]
    for r in sorted(rungs, key=lambda x: (-x.holdout.annual_net_pct)):
        flag = "-" if not r.judged else ("yes" if r.sign_held else "no")
        lines.append(
            f"  {r.factor:<20} {r.horizon:>6d} {r.in_sample.annual_net_pct:>+9.1f} "
            f"{r.holdout.annual_net_pct:>+9.1f} {r.holdout.sharpe:>+9.2f} "
            f"{r.holdout.turnover:>6.2f} {flag:>5} {r.holdout.periods:>6d}"
        )
    judged = [r for r in rungs if r.judged]
    survivors = [
        r for r in judged
        if r.holdout.annual_net_pct > 0 and r.in_sample.annual_net_pct > 0 and r.sign_held
    ]
    lines.append("")
    lines.append(
        f"  {len(judged)}/{len(rungs)} rungs had enough non-overlapping periods "
        f"to judge (>= {MIN_REBALANCES}); {len(survivors)} are positive in BOTH "
        f"halves with a stable sign."
    )
    if blend is not None:
        lines.append("")
        lines.append("  BLEND (holdout only):")
        lines.append(f"    components         : {', '.join(blend.components)}")
        lines.append(f"    mean pairwise corr : {blend.mean_correlation:+.3f}")
        lines.append(f"    annual net         : {blend.annual_net_pct:+.2f}%")
        lines.append(f"    Sharpe             : {blend.sharpe:+.2f}")
        lines.append(f"    best component     : {blend.best_component_sharpe:+.2f}")
        lines.append(f"    deflated Sharpe    : {blend.deflated_sharpe:.2f}")
        lines.append(
            "    verdict            : "
            + (
                "blend beats its best component"
                if blend.beats_best_component
                else "blend does NOT beat its best component"
            )
        )
    return "\n".join(lines)
