"""Every horizon level, one report: intraday, daily, weekly, multi-week.

    python scripts/evaluate_horizons.py

All previous research lived between 1 hour and 48 hours -- the band where
cost per rebalance dominates. This spans the whole ladder using both caches:

  hourly cache (2y x 117 symbols): horizons 1h, 4h, 12h, 24h, 72h
  daily cache  (5y x  53 symbols): horizons 1d, 3d, 7d, 14d, 30d

Everything is annualised so the bands are directly comparable, every cell is
scored on both a selection half and a holdout, and a multi-factor blend is
tested at the horizons that have any surviving structure. The verdict can be
-- and given four prior nulls, may well be -- that no level survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting.cross_section import (  # noqa: E402
    DAYS_PER_YEAR,
    HOURS_PER_YEAR,
    build_factors,
    load_universe,
)
from services.backtesting.horizon_ladder import (  # noqa: E402
    evaluate_blend,
    evaluate_ladder,
    format_ladder,
)

# Futures maker fee -- the cheapest venue actually available.
COST_BPS = 2.0
HOLDOUT_FRACTION = 0.4


def run_band(
    label: str,
    universe_path: str,
    horizons: tuple[int, ...],
    bars_per_year: float,
    horizon_unit: str,
) -> None:
    path = Path(universe_path)
    if not path.exists():
        print(f"{label}: {path} missing -- run the corresponding build script")
        return
    data = load_universe(str(path))
    close, volume = data["close"], data["volume"]
    factors = build_factors(close, volume)
    print(f"\n{'=' * 88}\n{label}: {close.shape[0]:,} bars x {close.shape[1]} symbols "
          f"(horizons in {horizon_unit}, cost {COST_BPS:.0f}bps)\n{'=' * 88}")
    rungs = evaluate_ladder(
        close, factors, horizons, COST_BPS, bars_per_year, HOLDOUT_FRACTION
    )

    # Blend at the slowest judged horizon: that is where a multi-factor book
    # is structurally cheapest to hold, and the blend is evaluated on the
    # holdout only.
    judged_horizons = sorted({r.horizon for r in rungs if r.judged})
    blend = None
    if judged_horizons:
        slowest = judged_horizons[-1]
        split = int(close.shape[0] * (1 - HOLDOUT_FRACTION))
        blend = evaluate_blend(
            close,
            factors,
            slowest,
            COST_BPS,
            bars_per_year,
            n_trials=len(factors) * len(horizons),
            holdout_start=split,
        )
        print(f"\n  (blend tested at the slowest judged horizon: {slowest} {horizon_unit})")
    print(format_ladder(rungs, blend))


def main() -> int:
    run_band(
        "INTRADAY-TO-MULTIDAY (hourly bars)",
        "data/universe_1h.npz",
        (1, 4, 12, 24, 72),
        HOURS_PER_YEAR,
        "hours",
    )
    run_band(
        "DAILY-TO-MULTIWEEK (daily bars)",
        "data/universe_1d.npz",
        (1, 3, 7, 14, 30),
        DAYS_PER_YEAR,
        "days",
    )
    print(
        "\n  SURVIVORSHIP: both universes are today's survivors, and the daily "
        "cache DOUBLY so\n  (5 years of history is only possessed by coins that "
        "lasted 5 years). Upper bounds throughout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
