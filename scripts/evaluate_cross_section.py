"""Is there cross-sectional edge in a broad crypto universe?

    python scripts/build_universe.py --top 300 --years 2   # once
    python scripts/evaluate_cross_section.py --horizon 1 --cost-bps 10

The live per-symbol signal has no edge at the horizon it trades, measured
across 60,732 of its own predictions. This asks the different question that
breadth makes answerable: at each hour, does ranking symbols against each
other predict their RELATIVE returns?

A NO EDGE verdict here is a successful run, and the exit code is 0 either way
so nobody is tempted to re-run until it turns green.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting.cross_section import (  # noqa: E402
    COST_LADDER_BPS,
    DEFAULT_COST_BPS,
    build_factors,
    evaluate,
    format_frontier,
    format_report,
    forward_returns,
    load_universe,
    turnover_frontier,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/universe_1h.npz")
    parser.add_argument(
        "--horizon", type=int, default=1, help="holding period in bars (hours)"
    )
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument(
        "--frontier",
        default="",
        help="also sweep turnover-reduction settings for this factor",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.4,
        help="final fraction of history reserved as an untouched holdout",
    )
    args = parser.parse_args()

    path = Path(args.universe)
    if not path.exists():
        print(
            f"error: {path} not found -- run scripts/build_universe.py first",
            file=sys.stderr,
        )
        return 2

    data = load_universe(str(path))
    close, volume = data["close"], data["volume"]
    print(
        f"universe: {close.shape[0]:,} bars x {close.shape[1]} symbols "
        f"({args.horizon}-hour holding period)\n"
    )

    # Split BEFORE looking at anything. The in-sample half is where a factor
    # may be chosen; the holdout is the only number that means anything about
    # a factor selected on the strength of the first.
    split = int(close.shape[0] * (1 - args.holdout_fraction))
    factors = build_factors(close, volume)

    print("IN-SAMPLE (factor selection permitted here)")
    in_sample = evaluate(
        close[:split],
        volume[:split],
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        factors={name: f[:split] for name, f in factors.items()},
    )
    print(format_report(in_sample, args.cost_bps))

    print("\n\nHOLDOUT (never used for selection)")
    holdout = evaluate(
        close[split:],
        volume[split:],
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        factors={name: f[split:] for name, f in factors.items()},
    )
    print(format_report(holdout, args.cost_bps))

    print("\n\nCONSISTENCY (the only comparison that matters)")
    in_by_name = {f.name: f for f in in_sample.factors}
    print(f"  {'factor':<20} {'IS mean IC':>12} {'OOS mean IC':>13} {'sign held':>11}")
    agreed = 0
    for out in holdout.factors:
        ins = in_by_name.get(out.name)
        if ins is None:
            continue
        held = (
            ins.mean_ic * out.mean_ic > 0 and abs(ins.mean_ic) > 0.005
        )
        agreed += bool(held)
        print(
            f"  {out.name:<20} {ins.mean_ic:>+12.4f} {out.mean_ic:>+13.4f} "
            f"{'yes' if held else 'no':>11}"
        )
    print(
        f"\n  {agreed}/{len(holdout.factors)} factors kept their sign out of sample."
    )
    print(
        "  A factor that flips sign on the holdout was noise, however good its "
        "in-sample t-statistic looked."
    )
    if args.frontier:
        if args.frontier not in factors:
            print(f"\nunknown factor {args.frontier!r}; have: {sorted(factors)}")
        else:
            print(f"\n\nTURNOVER REDUCTION -- {args.frontier}")
            points = turnover_frontier(
                factors[args.frontier],
                forward_returns(close, args.horizon),
                split,
                args.cost_bps,
            )
            print(format_frontier(points, args.cost_bps))

    print(
        f"\n  Costs assumed: {args.cost_bps:.0f}bps per unit turnover. Binance spot "
        "is ~10bps PER SIDE at base tier; futures maker is ~2bps."
    )
    print(f"  Ladder shown: {', '.join(f'{c:.0f}' for c in COST_LADDER_BPS)} bps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
