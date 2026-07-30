"""Does the LIVE serving signal have edge net of costs?

``edge.py`` validates the daily-horizon harness strategy. It has never been
pointed at the ensemble that actually trades, so the platform's one published
edge verdict says nothing about the models running in production.

This module answers that question from the strongest evidence available: the
platform's own recorded predictions. Every serving prediction is written to
``predictions`` at emission time and its outcome resolved later by the
continuous-learning loop, which makes the set genuinely point-in-time and
out-of-sample -- no feature-parity risk, no look-ahead, no re-simulation. It
is a live forward test that has already run.

Three properties of that data drive the design:

1. **The direction label is uninformative.** Conformal abstention flattens
   essentially every prediction to "flat", so accuracy-on-labels measures the
   abstention rate, not skill. The signal under test is ``expected_return``.
2. **Observations overlap 5x.** Predictions arrive every minute over a
   5-minute horizon, so naive statistics overstate significance by ~sqrt(5).
   Every headline statistic here is computed on a de-overlapped subsample.
3. **The search was not free.** Champion selection tried many configurations,
   so an unadjusted Sharpe is biased upward. The verdict applies a deflated
   Sharpe ratio against a declared trial count.

Honesty policy: this module must be as capable of returning NO EDGE as it is
of returning edge, and the null must be as loud as the alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

# The paper broker slips 0-5bps uniformly per side, so a round trip costs
# ~5bps on average. Live venues add spread; the report shows a sensitivity
# ladder rather than pretending one number is authoritative.
DEFAULT_COST_BPS = 5.0
COST_LADDER_BPS = (0.0, 2.0, 5.0, 10.0)

# Predictions are emitted per minute against a 5-minute horizon.
DEFAULT_OVERLAP = 5

# Minutes per year, for annualising a 5-minute-period Sharpe.
_PERIODS_PER_YEAR = 365.25 * 24 * 60 / 5

# Champion selection searched model families x calibration x thresholds. This
# is the trial count the deflated Sharpe is charged for; it is deliberately a
# declared assumption rather than a hidden constant.
DEFAULT_N_TRIALS = 24

# An information coefficient below this is not distinguishable from noise at
# realistic sample sizes; published equity-factor ICs of 0.02-0.05 are
# considered good, so this is a floor, not a target.
MIN_MEANINGFUL_IC = 0.01


@dataclass(frozen=True)
class Prediction:
    """One resolved, point-in-time serving prediction."""

    symbol: str
    predicted_at: datetime
    expected_return: float
    confidence: float
    actual_return: float


@dataclass
class SymbolEdge:
    """Per-symbol verdict on the live signal."""

    symbol: str
    n: int
    n_effective: int
    ic: float
    ic_p_value: float
    ic_ci: tuple[float, float]
    net_mean_bps: float
    sharpe: float
    deflated_sharpe: float
    hit_rate: float
    has_edge: bool
    insufficient: bool = False


@dataclass
class EdgeReport:
    """The full verdict."""

    verdict: str
    has_edge: bool
    n_total: int
    n_effective: int
    ic: float
    ic_p_value: float
    ic_ci: tuple[float, float]
    hit_rate: float
    gross_mean_bps: float
    cost_ladder: dict[float, float] = field(default_factory=dict)
    deciles: list[dict[str, float]] = field(default_factory=list)
    per_symbol: list[SymbolEdge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def deoverlap(values: np.ndarray, overlap: int = DEFAULT_OVERLAP) -> np.ndarray:
    """Take every *overlap*-th observation so returns no longer share bars.

    Overlapping forward returns are mechanically autocorrelated; testing them
    as if independent inflates every t-statistic by roughly sqrt(overlap).
    """
    if overlap <= 1:
        return values
    return values[::overlap]


def information_coefficient(
    signal: np.ndarray, outcome: np.ndarray
) -> tuple[float, float]:
    """Spearman rank correlation between signal and realised return.

    Rank correlation rather than Pearson: the question is whether the ordering
    is right, and it is robust to the fat tails of minute-bar returns.
    """
    if signal.size < 3 or np.all(signal == signal[0]):
        return 0.0, 1.0
    result = stats.spearmanr(signal, outcome)
    ic = float(result.statistic)
    p = float(result.pvalue)
    return (0.0 if np.isnan(ic) else ic, 1.0 if np.isnan(p) else p)


def block_bootstrap_ic(
    signal: np.ndarray,
    outcome: np.ndarray,
    block: int = 60,
    n_boot: int = 1000,
    seed: int = 7,
) -> tuple[float, float]:
    """95% CI for the IC via a moving-block bootstrap.

    Blocks preserve the local autocorrelation that an i.i.d. bootstrap would
    destroy (and thereby understate the true uncertainty).
    """
    n = signal.size
    if n < block * 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = n // block
    ics = np.empty(n_boot)
    starts_max = n - block
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        ic, _ = information_coefficient(signal[idx], outcome[idx])
        ics[i] = ic
    return (float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5)))


def deflated_sharpe(
    sharpe: float, n_obs: int, n_trials: int = DEFAULT_N_TRIALS, skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability the observed Sharpe survives the selection that produced it.

    Bailey & Lopez de Prado's deflated Sharpe ratio: searching N strategies
    produces a maximum Sharpe well above zero even when every strategy is
    worthless, so the benchmark to beat rises with the size of the search.
    Returns a probability in [0, 1]; below ~0.95 the result is not
    distinguishable from the best of N coin flips.

    *sharpe* is the PER-OBSERVATION Sharpe, not an annualised one, and the
    expected-maximum benchmark is scaled by the sampling standard deviation
    of a Sharpe estimate under the null (~1/sqrt(n-1)). Omitting that scaling
    compares a per-observation Sharpe against a benchmark of ~2.0 and reports
    every strategy as worthless, however strong.
    """
    if n_obs < 3 or sharpe == 0.0:
        return 0.0
    # Expected maximum Sharpe from n_trials independent worthless strategies,
    # in units of the Sharpe estimator's own standard error.
    euler = 0.5772156649
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        expected_max_z = (1 - euler) * z1 + euler * z2
    else:
        expected_max_z = 0.0
    sr_std_error = 1.0 / np.sqrt(n_obs - 1)
    sr0 = expected_max_z * sr_std_error
    denom = np.sqrt(1 - skew * sharpe + (kurtosis - 1) / 4.0 * sharpe**2)
    if denom <= 0:
        return 0.0
    z = (sharpe - sr0) * np.sqrt(n_obs - 1) / denom
    return float(stats.norm.cdf(z))


def strategy_returns(
    signal: np.ndarray, outcome: np.ndarray, cost_bps: float, threshold: float = 0.0
) -> np.ndarray:
    """Per-period net returns of trading the sign of the (centred) signal.

    The signal is centred on its own median first: ``expected_return`` is
    almost always positive, so its raw sign is a constant long bet, which
    measures market drift rather than skill.
    """
    centred = signal - np.median(signal)
    position = np.where(
        centred > threshold, 1.0, np.where(centred < -threshold, -1.0, 0.0)
    )
    gross = position * outcome
    cost = (position != 0).astype(float) * (cost_bps / 1e4)
    net: np.ndarray = gross - cost
    return net


@dataclass(frozen=True)
class HorizonResult:
    """The signal's behaviour at one holding period."""

    horizon: int
    n_effective: int
    ic: float
    p_value: float
    gross_bps: float
    net_bps: dict[float, float]
    deflated_sharpe: float


def scan_horizons(
    signal: np.ndarray,
    close: np.ndarray,
    bar_index: np.ndarray,
    horizons: tuple[int, ...] = (5, 15, 30, 60, 240),
    cost_bps: float = DEFAULT_COST_BPS,
    n_trials: int = DEFAULT_N_TRIALS,
) -> list[HorizonResult]:
    """Measure the signal against forward returns at several holding periods.

    A signal can be real and still unprofitable at the horizon it is traded
    on: transaction cost is paid per round trip, so edge per unit time must
    exceed cost per unit time. Scanning horizons separates "no signal" from
    "signal too small for this execution path".

    IMPORTANT: scanning is itself a search. Every result here carries the
    multiple-testing penalty of the whole scan via *n_trials*, and a horizon
    picked from this scan is a HYPOTHESIS, not a validated finding -- it must
    be confirmed on data this scan never saw.
    """
    results: list[HorizonResult] = []
    for horizon in horizons:
        usable = bar_index + horizon < close.size
        if usable.sum() < horizon * 20:
            continue
        idx = bar_index[usable]
        forward = close[idx + horizon] / close[idx] - 1.0
        centred = signal[usable] - np.median(signal[usable])
        sig_d = deoverlap(centred, horizon)
        fwd_d = deoverlap(forward, horizon)
        if sig_d.size < 20:
            continue
        ic, p = information_coefficient(sig_d, fwd_d)
        net = {
            bps: float(strategy_returns(sig_d, fwd_d, bps).mean() * 1e4)
            for bps in COST_LADDER_BPS
        }
        priced = strategy_returns(sig_d, fwd_d, cost_bps)
        sd = float(priced.std(ddof=1)) if priced.size > 1 else 0.0
        per_period = float(priced.mean() / sd) if sd > 0 else 0.0
        results.append(
            HorizonResult(
                horizon=horizon,
                n_effective=int(sig_d.size),
                ic=ic,
                p_value=p,
                gross_bps=net.get(0.0, 0.0),
                net_bps=net,
                deflated_sharpe=deflated_sharpe(per_period, sig_d.size, n_trials),
            )
        )
    return results


def confidence_strata(
    confidence: np.ndarray,
    signal: np.ndarray,
    outcome: np.ndarray,
    overlap: int = DEFAULT_OVERLAP,
    n_strata: int = 5,
) -> list[dict[str, float]]:
    """Does the model's own confidence identify where it is right?

    WHAT ``confidence`` ACTUALLY IS on this platform: ``EnsemblePredictor``
    sets ``confidence = combined_probs[best_direction]``, and whenever the
    agreement vote or the conformal gate flattens a signal it is overwritten
    with ``combined_probs["flat"]``. Since ~100% of stored predictions are
    flattened, the recorded value is **p(flat)** -- the model's belief that
    price will NOT move -- not its confidence in a direction. Read it as an
    abstention score.

    ZERO-RETURN GUARD (this cost a retracted finding): low-priced assets with
    coarse ticks produce genuinely unchanged bars -- 32.8% of DOT's and 18.0%
    of ADA's 5-minute windows. ``np.sign(0)`` is 0 and matches no position, so
    including them drags agreement below 50% in exact proportion to how often
    the price stood still. That correlates with p(flat) BY CONSTRUCTION, and
    manufactures a textbook-looking "confidence is inverted" result out of a
    model that was simply right about flatness. Agreement is therefore
    computed only where the price actually moved, and the zero fraction is
    reported so the reader can see the effect that was removed.
    """
    if confidence.size < n_strata * 100:
        return []
    edges = np.percentile(confidence, np.linspace(0, 100, n_strata + 1))
    rows: list[dict[str, float]] = []
    for i in range(n_strata):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (
            confidence <= hi if i == n_strata - 1 else confidence < hi
        )
        if mask.sum() < 100:
            continue
        centred = deoverlap(signal[mask] - np.median(signal[mask]), overlap)
        out = deoverlap(outcome[mask], overlap)
        moved = out != 0.0
        ic, p = information_coefficient(centred[moved], out[moved])
        agreement = (
            float((np.sign(centred[moved]) == np.sign(out[moved])).mean())
            if moved.any()
            else float("nan")
        )
        rows.append(
            {
                "stratum": float(i + 1),
                "conf_lo": float(lo),
                "conf_hi": float(hi),
                "n": float(mask.sum()),
                "zero_return_fraction": float((~moved).mean()),
                "sign_agreement": agreement,
                "ic": ic,
                "p_value": p,
            }
        )
    return rows


def confidence_is_inverted(strata: list[dict[str, float]]) -> bool:
    """True when directional agreement falls as confidence rises."""
    if len(strata) < 3:
        return False
    idx = np.array([s["stratum"] for s in strata])
    agree = np.array([s["sign_agreement"] for s in strata])
    corr, _ = information_coefficient(idx, agree)
    return bool(corr < -0.5)


def decile_table(signal: np.ndarray, outcome: np.ndarray) -> list[dict[str, float]]:
    """Mean realised return per signal decile.

    A real signal produces a monotonic ladder; noise produces a jumble. This
    is the most reader-proof evidence in the report -- it cannot be rescued
    by a favourable summary statistic.
    """
    if signal.size < 100:
        return []
    edges = np.percentile(signal, np.arange(0, 101, 10))
    rows: list[dict[str, float]] = []
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (signal >= lo) & (signal <= hi if i == 9 else signal < hi)
        if not mask.any():
            continue
        rows.append(
            {
                "decile": float(i + 1),
                "n": float(mask.sum()),
                "signal_mean_bps": float(signal[mask].mean() * 1e4),
                "actual_mean_bps": float(outcome[mask].mean() * 1e4),
            }
        )
    return rows


def monotonicity(deciles: list[dict[str, float]]) -> float:
    """Rank correlation between decile index and its mean realised return."""
    if len(deciles) < 3:
        return 0.0
    idx = np.array([d["decile"] for d in deciles])
    val = np.array([d["actual_mean_bps"] for d in deciles])
    ic, _ = information_coefficient(idx, val)
    return ic


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_arrays(
    signal: np.ndarray,
    outcome: np.ndarray,
    cost_bps: float,
    overlap: int,
    n_trials: int,
) -> dict[str, Any]:
    sig = deoverlap(signal, overlap)
    out = deoverlap(outcome, overlap)
    ic, p = information_coefficient(sig, out)
    net = strategy_returns(sig, out, cost_bps)
    traded = net[net != 0.0]
    sd = float(net.std(ddof=1)) if net.size > 1 else 0.0
    sharpe = (
        float(net.mean() / sd * np.sqrt(_PERIODS_PER_YEAR)) if sd > 0 else 0.0
    )
    per_period_sharpe = float(net.mean() / sd) if sd > 0 else 0.0
    return {
        "n": int(signal.size),
        "n_effective": int(sig.size),
        "ic": ic,
        "ic_p_value": p,
        "net_mean_bps": float(net.mean() * 1e4),
        "sharpe": sharpe,
        "deflated_sharpe": deflated_sharpe(per_period_sharpe, sig.size, n_trials),
        "hit_rate": float((traded > 0).mean()) if traded.size else 0.0,
    }


def evaluate(
    predictions: list[Prediction],
    cost_bps: float = DEFAULT_COST_BPS,
    overlap: int = DEFAULT_OVERLAP,
    n_trials: int = DEFAULT_N_TRIALS,
    min_per_symbol: int = 500,
) -> EdgeReport:
    """Judge the live signal. Returns a verdict that may well be NO EDGE."""
    notes: list[str] = []
    if not predictions:
        return EdgeReport(
            verdict="NO resolved predictions -> cannot assess the live signal",
            has_edge=False, n_total=0, n_effective=0, ic=0.0, ic_p_value=1.0,
            ic_ci=(float("nan"), float("nan")), hit_rate=0.0, gross_mean_bps=0.0,
            notes=["The predictions table has no resolved outcomes."],
        )

    ordered = sorted(predictions, key=lambda p: (p.symbol, p.predicted_at))
    signal = np.array([p.expected_return for p in ordered], dtype=float)
    outcome = np.array([p.actual_return for p in ordered], dtype=float)
    symbols = np.array([p.symbol for p in ordered])

    overall = _evaluate_arrays(signal, outcome, cost_bps, overlap, n_trials)
    sig_d = deoverlap(signal, overlap)
    out_d = deoverlap(outcome, overlap)
    ic_ci = block_bootstrap_ic(sig_d, out_d)
    deciles = decile_table(sig_d, out_d)

    ladder = {
        bps: float(strategy_returns(sig_d, out_d, bps).mean() * 1e4)
        for bps in COST_LADDER_BPS
    }

    per_symbol: list[SymbolEdge] = []
    for sym in sorted(set(symbols)):
        mask = symbols == sym
        if int(mask.sum()) < min_per_symbol:
            per_symbol.append(
                SymbolEdge(
                    symbol=sym, n=int(mask.sum()), n_effective=0, ic=0.0,
                    ic_p_value=1.0, ic_ci=(float("nan"), float("nan")),
                    net_mean_bps=0.0, sharpe=0.0, deflated_sharpe=0.0,
                    hit_rate=0.0, has_edge=False, insufficient=True,
                )
            )
            continue
        r = _evaluate_arrays(signal[mask], outcome[mask], cost_bps, overlap, n_trials)
        s_d = deoverlap(signal[mask], overlap)
        o_d = deoverlap(outcome[mask], overlap)
        ci = block_bootstrap_ic(s_d, o_d)
        # A symbol passes only on all three: a positive IC that clears noise,
        # positive net-of-cost expectancy, and a Sharpe that survives the
        # search that produced it.
        has_edge = (
            r["ic"] > MIN_MEANINGFUL_IC
            and r["ic_p_value"] < 0.05
            and r["net_mean_bps"] > 0
            and r["deflated_sharpe"] > 0.95
        )
        per_symbol.append(
            SymbolEdge(
                symbol=sym, n=r["n"], n_effective=r["n_effective"], ic=r["ic"],
                ic_p_value=r["ic_p_value"], ic_ci=ci,
                net_mean_bps=r["net_mean_bps"], sharpe=r["sharpe"],
                deflated_sharpe=r["deflated_sharpe"], hit_rate=r["hit_rate"],
                has_edge=has_edge,
            )
        )

    judged = [s for s in per_symbol if not s.insufficient]
    passed = [s for s in judged if s.has_edge]
    mono = monotonicity(deciles)

    notes.append(
        "Direction labels are ~100% 'flat' (conformal abstention), so the "
        "signal under test is expected_return, not the label."
    )
    notes.append(
        f"Overlap {overlap}x removed by subsampling: {overall['n']} raw "
        f"observations -> {overall['n_effective']} independent."
    )
    notes.append(
        f"Deflated Sharpe charged for {n_trials} search trials; > 0.95 is the bar."
    )
    notes.append(f"Decile monotonicity (rank corr of decile vs realised): {mono:+.3f}")
    if ladder.get(0.0, 0.0) <= 0:
        notes.append(
            "Gross expectancy is <= 0 BEFORE costs: the signal is not merely "
            "being eaten by slippage."
        )

    if not judged:
        verdict = "NO symbol had enough resolved predictions -> cannot assess edge"
        has_edge = False
    elif not passed:
        verdict = (
            f"NO EDGE in the live signal: 0/{len(judged)} symbols passed "
            f"(IC {overall['ic']:+.4f}, p={overall['ic_p_value']:.3f}, "
            f"net {overall['net_mean_bps']:+.3f} bps/trade at {cost_bps:.0f}bps cost)"
        )
        has_edge = False
    elif len(passed) / len(judged) >= 0.6 and len(judged) >= 3:
        verdict = (
            f"EDGE in {len(passed)}/{len(judged)} symbols -> candidate, but "
            "confirm on a fresh out-of-sample window before sizing up"
        )
        has_edge = True
    else:
        verdict = (
            f"INCONCLUSIVE: {len(passed)}/{len(judged)} symbols passed -- "
            "not a stable cross-sectional result"
        )
        has_edge = False

    return EdgeReport(
        verdict=verdict,
        has_edge=has_edge,
        n_total=overall["n"],
        n_effective=overall["n_effective"],
        ic=overall["ic"],
        ic_p_value=overall["ic_p_value"],
        ic_ci=ic_ci,
        hit_rate=overall["hit_rate"],
        gross_mean_bps=ladder.get(0.0, 0.0),
        cost_ladder=ladder,
        deciles=deciles,
        per_symbol=per_symbol,
        notes=notes,
    )


def format_report(report: EdgeReport) -> str:
    """Render the verdict as a plain-text report."""
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("LIVE SERVING SIGNAL -- EDGE EVALUATION")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"VERDICT: {report.verdict}")
    lines.append("")
    lines.append(
        f"  observations   : {report.n_total:,} resolved "
        f"({report.n_effective:,} independent after de-overlapping)"
    )
    lo, hi = report.ic_ci
    ci = "n/a" if np.isnan(lo) else f"[{lo:+.4f}, {hi:+.4f}]"
    lines.append(
        f"  information coef: {report.ic:+.4f}  p={report.ic_p_value:.4f}  95% CI {ci}"
    )
    lines.append(f"  hit rate       : {report.hit_rate * 100:.1f}%")
    lines.append("")
    lines.append("  expectancy vs cost (bps per period):")
    for bps, mean in sorted(report.cost_ladder.items()):
        tag = "  <- gross" if bps == 0 else ""
        lines.append(f"      cost {bps:5.1f}bps -> {mean:+8.4f} bps{tag}")
    lines.append("")

    if report.deciles:
        lines.append("  signal decile -> realised return (a real signal is monotonic):")
        lines.append("      decile      n   signal(bps)   realised(bps)")
        for d in report.deciles:
            lines.append(
                f"      {int(d['decile']):>6} {int(d['n']):>6} "
                f"{d['signal_mean_bps']:>12.3f} {d['actual_mean_bps']:>15.3f}"
            )
        lines.append("")

    judged = [s for s in report.per_symbol if not s.insufficient]
    if judged:
        lines.append("  per symbol:")
        lines.append(
            "      symbol         n      IC      p     net(bps)  DSR   verdict"
        )
        for s in judged:
            mark = "EDGE" if s.has_edge else "none"
            lines.append(
                f"      {s.symbol:<10} {s.n:>7} {s.ic:+7.4f} {s.ic_p_value:6.3f} "
                f"{s.net_mean_bps:+9.4f} {s.deflated_sharpe:5.2f}  {mark}"
            )
    skipped = [s for s in report.per_symbol if s.insufficient]
    if skipped:
        lines.append(
            f"      (skipped for too few resolved predictions: "
            f"{', '.join(s.symbol for s in skipped)})"
        )
    lines.append("")
    lines.append("  notes:")
    for note in report.notes:
        lines.append(f"    - {note}")
    lines.append("=" * 74)
    return "\n".join(lines)
