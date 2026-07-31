"""Does funding-rate positioning predict relative perpetual returns?

    python scripts/build_funding_universe.py --top 150 --years 1.5   # once
    python scripts/evaluate_funding.py --horizon 8 --cost-bps 2

Three price-based research passes have come back empty or uneconomic. This
tests an input that is not in the price series at all -- what perpetual longs
and shorts are paying each other to hold their positions -- against the same
standards.

The price factors are evaluated alongside as a CONTROL. If funding scores no
better than reversal did on the same contracts over the same window, the new
data source added nothing, and that is the finding.

Exit code is 0 whatever the verdict.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting.cross_section import (  # noqa: E402
    evaluate,
    format_frontier,
    format_report,
    forward_returns,
    turnover_frontier,
)
from services.backtesting.funding_factors import combined_factors  # noqa: E402

_FUNDING_FACTORS = {
    "funding_level",
    "funding_carry_24h",
    "funding_carry_72h",
    "funding_zscore_7d",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/perp_funding_1h.npz")
    parser.add_argument(
        "--horizon", type=int, default=8, help="holding period in hours (8 = one funding period)"
    )
    parser.add_argument(
        "--cost-bps", type=float, default=2.0, help="Binance futures maker is ~2bps"
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.4)
    args = parser.parse_args()

    path = Path(args.universe)
    if not path.exists():
        print(
            f"error: {path} not found -- run scripts/build_funding_universe.py first",
            file=sys.stderr,
        )
        return 2

    data = np.load(path, allow_pickle=False)
    close, volume, funding = data["close"], data["volume"], data["funding"]
    print(
        f"universe: {close.shape[0]:,} hourly bars x {close.shape[1]} perpetuals; "
        f"funding coverage {np.isfinite(funding).mean() * 100:.1f}%"
    )
    print(
        f"holding period {args.horizon}h, cost {args.cost_bps:.0f}bps per unit "
        f"turnover\n"
    )

    factors = combined_factors(close, volume, funding)
    split = int(close.shape[0] * (1 - args.holdout_fraction))

    for label, lo, hi in (
        ("IN-SAMPLE (selection permitted here)", 0, split),
        ("HOLDOUT (never used for selection)", split, close.shape[0]),
    ):
        print(f"\n{label}")
        report = evaluate(
            close[lo:hi],
            volume[lo:hi],
            horizon=args.horizon,
            cost_bps=args.cost_bps,
            factors={name: f[lo:hi] for name, f in factors.items()},
        )
        print(format_report(report, args.cost_bps))
        funding_rows = [f for f in report.factors if f.name in _FUNDING_FACTORS]
        price_rows = [f for f in report.factors if f.name not in _FUNDING_FACTORS]
        if funding_rows and price_rows:
            best_funding = max(abs(f.ic_t_stat) for f in funding_rows)
            best_price = max(abs(f.ic_t_stat) for f in price_rows)
            verdict = (
                "funding beats the price control"
                if best_funding > best_price
                else "funding does NOT beat the price control"
            )
            print(
                f"  CONTROL: best |t| funding {best_funding:.1f} vs "
                f"price {best_price:.1f} -> {verdict}"
            )

    # Turnover frontier for the strongest funding factor, both halves.
    forward = forward_returns(close, args.horizon)
    in_sample = evaluate(
        close[:split],
        volume[:split],
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        factors={name: f[:split] for name, f in factors.items()},
    )
    ranked = [f for f in in_sample.factors if f.name in _FUNDING_FACTORS]
    if ranked:
        target = max(ranked, key=lambda f: abs(f.ic_t_stat)).name
        print(f"\n\nTURNOVER REDUCTION -- {target}")
        points = turnover_frontier(
            factors[target], forward, split, args.cost_bps
        )
        print(format_frontier(points, args.cost_bps))

    print(
        "\n  SURVIVORSHIP: perpetuals listed today; delisted contracts absent. "
        "Every figure is an upper bound."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
