"""Audit the "EDGE STABLE, Sharpe 1.17" daily-harness claim. ONE run.

    python scripts/gauntlet_daily_harness.py

Executes the pre-registration committed to main (GO_LIVE.md, 2026-08-02)
BEFORE this script ran. The claim under audit was graded 2026-07-23, before
the deflated Sharpe, beta controls, survivorship measurement or
pre-registration existed, on three hand-picked mega-cap survivors whose OOS
window was a strong bull era.

Five registered legs, all must pass for the claim to survive:
  R1  reproduction of the original 3-symbol pass/fail pattern
  H1  breadth: pass fraction >= 0.60 on the fixed 64-name universe
  H2  beta control: strategy beats buy-and-hold for a majority of symbols
  H3  unseen era: pooled net return positive on post-2025-12-31 OOS periods
  DSR portfolio deflated Sharpe > 0.95 at n_trials = 24

A failed leg is a successful audit. Exit code is 0 either way.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting.edge import (  # noqa: E402
    HORIZON,
    MIN_SYMBOL_PASS_FRAC,
    annualized_sharpe,
    evaluate_symbol,
)
from services.backtesting.live_signal import deflated_sharpe  # noqa: E402

# The registered universe -- fixed in GO_LIVE.md before this script existed.
UNIVERSE = (
    "AAPL MSFT SPY QQQ AMZN GOOGL META NVDA TSLA JPM BAC WFC GS XOM CVX COP "
    "JNJ PFE MRK UNH ABBV LLY PG KO PEP WMT COST HD MCD NKE DIS NFLX CRM ORCL "
    "ADBE INTC AMD QCOM CSCO IBM T VZ CMCSA BA CAT GE MMM HON UPS FDX LMT RTX "
    "DE F GM V MA PYPL AXP BRK-B IWM DIA XLF XLE GLD"
).split()

ORIGINAL_SYMBOLS = ["AAPL", "MSFT", "SPY"]
ORIGINAL_END = datetime(2025, 12, 31, tzinfo=UTC)
COST_BPS = 5.0
N_TRIALS = 24  # declared in the registration
PPY = 252 / HORIZON


def buy_and_hold_sharpe(period_prices: np.ndarray) -> float:
    """Annualised Sharpe of holding the symbol over the same OOS periods.

    Costless by construction (no trading), which biases this control IN
    FAVOUR of buy-and-hold -- the strategy must beat a benchmark that pays
    nothing. If it cannot, its returns are beta, not skill.
    """
    if period_prices.size < 3:
        return 0.0
    returns = np.diff(period_prices) / period_prices[:-1]
    return annualized_sharpe(returns, PPY)


def pool_by_date(
    per_symbol: dict[str, tuple[list[datetime], np.ndarray]],
) -> tuple[list[datetime], np.ndarray]:
    """Equal-weight portfolio period returns, aligned on actual period dates.

    Symbols have different listing lengths so their 70/30 splits land on
    different dates; averaging arrays positionally would blend different
    calendar periods into one number. Bucketing by date keeps each portfolio
    period a real, simultaneous cross-section.
    """
    buckets: dict[datetime, list[float]] = {}
    for _, (dates, returns) in per_symbol.items():
        for when, ret in zip(dates, returns, strict=False):
            key = when.replace(hour=0, minute=0, second=0, microsecond=0)
            buckets.setdefault(key, []).append(float(ret))
    ordered = sorted(buckets)
    return ordered, np.array([float(np.mean(buckets[d])) for d in ordered])


def era_slice(
    dates: list[datetime], returns: np.ndarray, after: datetime
) -> np.ndarray:
    """Portfolio periods strictly after *after* (the unseen era)."""
    mask = np.array([d > after for d in dates])
    return returns[mask]


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    import yfinance as yf

    print(f"fetching {len(UNIVERSE)} symbols, 2018-01-01 -> today (daily)...", flush=True)
    raw = yf.download(
        UNIVERSE, start="2018-01-01", auto_adjust=True, progress=False
    )["Close"]

    series: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for symbol in UNIVERSE:
        if symbol not in raw.columns:
            continue
        column = raw[symbol].dropna()
        if len(column) < 400:
            continue
        dates = [
            d.to_pydatetime().replace(tzinfo=UTC) if d.tzinfo is None else d.to_pydatetime()
            for d in column.index
        ]
        series[symbol] = (dates, column.to_numpy(dtype=float))
    print(f"  usable series: {len(series)}/{len(UNIVERSE)}", flush=True)

    # ------------------------------------------------------------------
    # R1 + H1 + H2 + DSR: the original window (2018 .. 2025-12-31)
    # ------------------------------------------------------------------
    judged: dict[str, dict] = {}
    pooled_inputs: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for symbol, (dates, closes) in series.items():
        keep = [i for i, d in enumerate(dates) if d <= ORIGINAL_END]
        if len(keep) < 400:
            continue
        window_dates = [dates[i] for i in keep]
        evaluation = evaluate_symbol(
            symbol, closes[: len(keep)], COST_BPS, dates=window_dates
        )
        summary = evaluation.summary
        if summary.get("insufficient"):
            continue
        bh = buy_and_hold_sharpe(np.asarray(evaluation.period_prices, dtype=float))
        judged[symbol] = {
            "edge": bool(summary["edge"]),
            "sharpe": float(summary["sharpe"]),
            "hit_rate": float(summary["hit_rate"]),
            "stability": float(summary["stability"]),
            "bh_sharpe": bh,
            "beats_bh": float(summary["sharpe"]) > bh,
        }
        pooled_inputs[symbol] = (evaluation.period_dates, evaluation.net_returns)
        print(
            f"  {symbol:6s} edge={'Y' if summary['edge'] else 'n'} "
            f"sharpe={summary['sharpe']:+.2f} vs B&H {bh:+.2f} "
            f"hit={summary['hit_rate']:.3f} stab={summary['stability']:.2f}",
            flush=True,
        )

    # R1 -- reproduction
    r1_pattern = {s: judged.get(s, {}).get("edge") for s in ORIGINAL_SYMBOLS}
    r1_pass = r1_pattern == {"AAPL": True, "MSFT": False, "SPY": True}

    # H1 -- breadth
    passing = [s for s, v in judged.items() if v["edge"]]
    pass_fraction = len(passing) / len(judged) if judged else 0.0
    h1_pass = pass_fraction >= MIN_SYMBOL_PASS_FRAC

    # H2 -- beta control
    beats = [s for s, v in judged.items() if v["beats_bh"]]
    beat_fraction = len(beats) / len(judged) if judged else 0.0
    h2_pass = beat_fraction > 0.5

    # DSR -- portfolio deflated Sharpe on the broad universe
    portfolio_dates, portfolio_returns = pool_by_date(pooled_inputs)
    sd = float(portfolio_returns.std(ddof=1)) if portfolio_returns.size > 1 else 0.0
    per_period = float(portfolio_returns.mean() / sd) if sd > 0 else 0.0
    portfolio_sharpe = per_period * np.sqrt(PPY)
    dsr = deflated_sharpe(per_period, portfolio_returns.size, N_TRIALS)
    dsr_pass = dsr > 0.95

    # ------------------------------------------------------------------
    # H3 -- unseen era: full series, OOS periods after the original end
    # ------------------------------------------------------------------
    unseen_inputs: dict[str, tuple[list[datetime], np.ndarray]] = {}
    for symbol, (dates, closes) in series.items():
        evaluation = evaluate_symbol(symbol, closes, COST_BPS, dates=dates)
        if evaluation.summary.get("insufficient"):
            continue
        unseen_inputs[symbol] = (evaluation.period_dates, evaluation.net_returns)
    unseen_dates, unseen_portfolio = pool_by_date(unseen_inputs)
    unseen = era_slice(unseen_dates, unseen_portfolio, ORIGINAL_END)
    unseen_mean_bps = float(unseen.mean() * 1e4) if unseen.size else 0.0
    unseen_t = (
        float(unseen.mean() / unseen.std(ddof=1) * np.sqrt(unseen.size))
        if unseen.size > 2 and unseen.std(ddof=1) > 0
        else 0.0
    )
    h3_pass = unseen.size >= 10 and float(unseen.mean()) > 0

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("GAUNTLET VERDICT -- daily-harness claim (registered 2026-08-02)")
    print("=" * 76)
    print(f"  R1 reproduction   : {r1_pattern}  -> {'PASS' if r1_pass else 'FAIL'}")
    print(
        f"  H1 breadth        : {len(passing)}/{len(judged)} pass "
        f"({pass_fraction:.1%} vs required {MIN_SYMBOL_PASS_FRAC:.0%})"
        f"  -> {'PASS' if h1_pass else 'FAIL'}"
    )
    print(
        f"  H2 beta control   : {len(beats)}/{len(judged)} beat buy-and-hold "
        f"({beat_fraction:.1%})  -> {'PASS' if h2_pass else 'FAIL'}"
    )
    print(
        f"  H3 unseen era     : {unseen.size} portfolio periods after "
        f"{ORIGINAL_END.date()}, mean {unseen_mean_bps:+.1f} bps (t={unseen_t:+.2f})"
        f"  -> {'PASS' if h3_pass else 'FAIL'}"
    )
    print(
        f"  DSR               : portfolio Sharpe {portfolio_sharpe:+.2f}, "
        f"deflated {dsr:.3f} (n={portfolio_returns.size}, trials={N_TRIALS})"
        f"  -> {'PASS' if dsr_pass else 'FAIL'}"
    )
    survives = all([r1_pass, h1_pass, h2_pass, h3_pass, dsr_pass])
    print("-" * 76)
    if survives:
        print(
            "  CLAIM SURVIVES ALL FIVE LEGS -> graduate to a registered "
            "walk-forward watch before any capital discussion."
        )
    else:
        failed = [
            name
            for name, ok in [
                ("R1", r1_pass), ("H1", h1_pass), ("H2", h2_pass),
                ("H3", h3_pass), ("DSR", dsr_pass),
            ]
            if not ok
        ]
        print(f"  CLAIM DOES NOT SURVIVE: failed {', '.join(failed)}.")
    print(
        "\n  SURVIVORSHIP: universe is today's liquid names -- every pass "
        "statistic above is an UPPER BOUND."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
