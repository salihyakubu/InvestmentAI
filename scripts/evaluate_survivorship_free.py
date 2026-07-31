"""Execute the pre-registered survivorship-free funding test. ONE run.

    python scripts/build_survivorship_free_funding.py --months 24   # once
    python scripts/evaluate_survivorship_free.py

The registration (GO_LIVE.md, committed 2026-07-31 BEFORE the data was
fetched) fixes everything that could otherwise be bent after seeing results:

  H1 (primary): +funding_carry_24h -- HIGH funding predicts relative
      OUTPERFORMANCE (the reverse of PR #62's rejected prior) -- on the FULL
      universe including delisted contracts. 24h horizon on the 8h grid,
      2 bps cost, 40% chronological holdout, standard gate on BOTH halves.
  H2: the same factor on the survivors-only subset of the same period must
      show a MORE positive IC; the difference is the survivorship bias,
      measured directly.
  Prediction: if PR #62's reversal was survivorship, H1 attenuates toward
      zero and H2 shows a material gap.

Everything outside H1/H2 is labelled EXPLORATORY. The verdict prints against
the registered prediction, whichever way it lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.backtesting.cross_section import (  # noqa: E402
    cross_sectional_zscore,
    evaluate_factor,
    forward_returns,
)
from services.backtesting.funding_factors import trailing_mean  # noqa: E402

# 8h grid: three stamps per day.
BARS_PER_YEAR = 365.25 * 3
HORIZON = 3  # 3 stamps = 24 hours, as registered
COST_BPS = 2.0
HOLDOUT_FRACTION = 0.4
# Registered primary + two labelled-exploratory variants = the trial count
# the deflated Sharpe is charged for.
N_TRIALS = 3


def gate_line(name: str, r) -> str:
    return (
        f"  {name:<28} IC {r.mean_ic:+.4f}  t {r.ic_t_stat:+6.2f}  "
        f"net {r.net_bps.get(COST_BPS, 0.0):+7.2f}bps  ann {r.annual_net_pct:+7.1f}%  "
        f"mono {r.monotonicity:+.2f}  tail {r.tail_dominance:.2f}  "
        f"DSR {r.deflated_sharpe:.2f}  -> {'PASS' if r.has_edge else 'fail'}"
    )


def main() -> int:
    path = Path("data/perp_funding_8h_full.npz")
    if not path.exists():
        print("error: run scripts/build_survivorship_free_funding.py first", file=sys.stderr)
        return 2
    data = np.load(path, allow_pickle=False)
    close, funding = data["close"], data["funding"]
    active = data["active_at_end"]
    split = int(close.shape[0] * (1 - HOLDOUT_FRACTION))

    print(
        f"universe: {close.shape[0]:,} 8h stamps x {close.shape[1]} contracts "
        f"({int(active.sum())} survivors, {int((~active).sum())} delisted -- "
        f"the contracts every prior universe was missing)\n"
    )

    # The registered primary: POSITIVE funding carry. Sign fixed by the
    # registration; not adjustable here.
    primary = cross_sectional_zscore(trailing_mean(funding, HORIZON))
    forward = forward_returns(close, HORIZON)

    print("H1 -- registered primary: +funding_carry_24h on the FULL universe")
    for label, lo, hi in (
        ("in-sample ", 0, split),
        ("HOLDOUT   ", split, close.shape[0]),
    ):
        r = evaluate_factor(
            "h1", primary[lo:hi], forward[lo:hi], HORIZON, COST_BPS,
            N_TRIALS, BARS_PER_YEAR,
        )
        print(gate_line(label, r))

    # H2: the survivorship measurement. Same factor, same period, survivors
    # only -- the delta in IC IS the bias.
    surv_close = close[:, active]
    surv_funding = funding[:, active]
    surv_primary = cross_sectional_zscore(trailing_mean(surv_funding, HORIZON))
    surv_forward = forward_returns(surv_close, HORIZON)

    print("\nH2 -- same factor, SURVIVORS-ONLY subset (the biased universe)")
    full_ics: list[float] = []
    surv_ics: list[float] = []
    for label, lo, hi in (
        ("in-sample ", 0, split),
        ("HOLDOUT   ", split, close.shape[0]),
    ):
        full = evaluate_factor(
            "full", primary[lo:hi], forward[lo:hi], HORIZON, COST_BPS,
            N_TRIALS, BARS_PER_YEAR,
        )
        surv = evaluate_factor(
            "surv", surv_primary[lo:hi], surv_forward[lo:hi], HORIZON, COST_BPS,
            N_TRIALS, BARS_PER_YEAR,
        )
        full_ics.append(full.mean_ic)
        surv_ics.append(surv.mean_ic)
        print(
            f"  {label:<12} IC full {full.mean_ic:+.4f}  vs survivors-only "
            f"{surv.mean_ic:+.4f}   bias {surv.mean_ic - full.mean_ic:+.4f}"
        )

    print("\nEXPLORATORY (labelled as such; no claims):")
    for name, lookback in (("+funding_level", 1), ("+funding_carry_72h", 9)):
        factor = cross_sectional_zscore(trailing_mean(funding, lookback))
        r = evaluate_factor(
            name, factor[split:], forward[split:], HORIZON, COST_BPS,
            N_TRIALS, BARS_PER_YEAR,
        )
        print(gate_line(f"{name} (holdout)", r))

    print("\nAGAINST THE REGISTERED PREDICTION:")
    mean_bias = float(np.mean(np.array(surv_ics) - np.array(full_ics)))
    print(
        f"  survivorship bias (survivors-only IC minus full IC, mean of both "
        f"halves): {mean_bias:+.4f}"
    )
    print(
        "  interpretation: a positive gap means the survivors-only universe "
        "flattered the\n  reversed-funding story, exactly as the registration "
        "predicted it would."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
