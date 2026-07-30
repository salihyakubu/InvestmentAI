"""Judge the LIVE serving signal against its own recorded outcomes.

    export DATABASE_URL=...            # the Railway Postgres URL
    python scripts/evaluate_live_edge.py --days 120 --cost-bps 5

The platform's published edge verdict ("EDGE STABLE across 2/3 symbols")
came from the daily-horizon harness in ``services/backtesting/edge.py`` --
it says nothing about the 5-minute ensemble that actually trades. This
points the same standard of proof at the production models, using the
predictions they emitted live and the outcomes the learning loop resolved.

The result is whatever it is. A NO EDGE verdict here is a successful run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from services.backtesting.live_signal import (  # noqa: E402
    DEFAULT_COST_BPS,
    DEFAULT_N_TRIALS,
    DEFAULT_OVERLAP,
    Prediction,
    confidence_is_inverted,
    confidence_strata,
    evaluate,
    format_report,
    scan_horizons,
)

_SCAN_HORIZONS = (5, 15, 30, 60, 240)

_QUERY = """
    SELECT symbol, predicted_at, expected_return, confidence, actual_return
    FROM predictions
    WHERE actual_return IS NOT NULL
      AND expected_return IS NOT NULL
      AND predicted_at >= $1
    ORDER BY symbol, predicted_at
"""

_BARS = """
    SELECT time, close FROM ohlcv
    WHERE symbol = $1 AND timeframe = '1m'
    ORDER BY time
"""


async def load(url: str, days: int) -> list[Prediction]:
    import asyncpg

    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        rows = await conn.fetch(_QUERY, datetime.now(UTC) - timedelta(days=days))
    finally:
        await conn.close()
    return [
        Prediction(
            symbol=r["symbol"],
            predicted_at=r["predicted_at"],
            expected_return=float(r["expected_return"]),
            confidence=float(r["confidence"] or 0.0),
            actual_return=float(r["actual_return"]),
        )
        for r in rows
    ]


async def horizon_report(
    url: str, predictions: list[Prediction], cost_bps: float, min_bars: int = 5_000
) -> str:
    """Scan holding periods per symbol, using the stored 1-minute bars.

    Separates "no signal" from "signal too small to pay for this execution
    path" -- the two have completely different remedies.
    """
    import asyncpg

    by_symbol: dict[str, list[Prediction]] = {}
    for p in predictions:
        by_symbol.setdefault(p.symbol, []).append(p)

    # Every (symbol, horizon) pair is a trial; the deflated Sharpe is charged
    # for the whole scan, not for one lucky cell of it.
    n_trials = max(1, len(by_symbol) * len(_SCAN_HORIZONS))

    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    lines: list[str] = []
    lines.append("  IC by holding period (a signal can be real but too slow to pay):")
    header = "      symbol     " + "".join(f"{h:>6d}m" for h in _SCAN_HORIZONS)
    lines.append(header)
    pooled: dict[int, list[float]] = {h: [] for h in _SCAN_HORIZONS}
    net_at: dict[int, list[float]] = {h: [] for h in _SCAN_HORIZONS}
    try:
        for symbol in sorted(by_symbol):
            bars = await conn.fetch(_BARS, symbol)
            if len(bars) < min_bars:
                continue
            index_of = {
                b["time"].replace(second=0, microsecond=0): i for i, b in enumerate(bars)
            }
            close = np.array([float(b["close"]) for b in bars])
            idx, sig = [], []
            for p in by_symbol[symbol]:
                k = index_of.get(p.predicted_at.replace(second=0, microsecond=0))
                if k is not None:
                    idx.append(k)
                    sig.append(p.expected_return)
            if len(idx) < 3_000:
                continue
            results = {
                r.horizon: r
                for r in scan_horizons(
                    np.array(sig),
                    close,
                    np.array(idx),
                    horizons=_SCAN_HORIZONS,
                    cost_bps=cost_bps,
                    n_trials=n_trials,
                )
            }
            row = f"      {symbol:<10} "
            for h in _SCAN_HORIZONS:
                r = results.get(h)
                if r is None:
                    row += "   n/a "
                    continue
                pooled[h].append(r.ic)
                net_at[h].append(r.net_bps.get(cost_bps, 0.0))
                row += f"{r.ic:+6.3f}{'*' if r.p_value < 0.05 else ' '}"
            lines.append(row)
    finally:
        await conn.close()

    lines.append(
        "      MEAN IC    "
        + "".join(
            f"{np.mean(pooled[h]):+6.3f} " if pooled[h] else "   n/a " for h in _SCAN_HORIZONS
        )
    )
    lines.append(
        f"      net@{cost_bps:.0f}bps  "
        + "".join(
            f"{np.mean(net_at[h]):+6.2f} " if net_at[h] else "   n/a " for h in _SCAN_HORIZONS
        )
        + "  (bps per hold)"
    )
    lines.append("      * p < 0.05, BEFORE any multiple-testing correction.")
    lines.append(
        "      A horizon chosen from this scan is a HYPOTHESIS. Confirm it on "
        "data the scan never saw before trading it."
    )
    return "\n".join(lines)


def calibration_report(predictions: list[Prediction], overlap: int) -> str:
    """Is the model most accurate where it is most confident?"""
    conf = np.array([p.confidence for p in predictions])
    sig = np.array([p.expected_return for p in predictions])
    out = np.array([p.actual_return for p in predictions])
    strata = confidence_strata(conf, sig, out, overlap=overlap)
    if not strata:
        return "  confidence calibration: too few predictions to assess"
    lines = ["  confidence quintile -> directional agreement:"]
    lines.append("      quintile   range            n   agree      IC       p")
    for s in strata:
        lines.append(
            f"      Q{int(s['stratum'])}       "
            f"[{s['conf_lo']:.3f},{s['conf_hi']:.3f}] {int(s['n']):>7} "
            f"{s['sign_agreement'] * 100:6.1f}% {s['ic']:+8.4f} {s['p_value']:7.3f}"
        )
    if confidence_is_inverted(strata):
        lines.append(
            "      INVERTED: accuracy FALLS as confidence rises. The abstention "
            "gate is keeping the wrong predictions."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="predictions per horizon; subsampled away before testing",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--database-url-env", default="DATABASE_URL")
    args = parser.parse_args()

    url = os.environ.get(args.database_url_env)
    if not url:
        print(f"error: ${args.database_url_env} is not set", file=sys.stderr)
        return 2

    predictions = asyncio.run(load(url, args.days))
    print(f"loaded {len(predictions):,} resolved predictions\n")
    report = evaluate(
        predictions,
        cost_bps=args.cost_bps,
        overlap=args.overlap,
        n_trials=args.n_trials,
    )
    print(format_report(report))
    print()
    print(calibration_report(predictions, args.overlap))
    print()
    print(asyncio.run(horizon_report(url, predictions, args.cost_bps)))
    # Exit 0 regardless of the verdict: "no edge" is a valid, successful
    # answer, and a non-zero exit would tempt someone to re-run until green.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
