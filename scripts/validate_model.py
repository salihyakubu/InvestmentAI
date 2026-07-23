"""Stage-2 gate: train the tree model and measure OUT-OF-SAMPLE edge.

    # null self-test (zero-drift random walk -> must report NO edge):
    PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python scripts/validate_model.py
    # one symbol (necessary, NOT sufficient):
    PYTHONPATH=. python scripts/validate_model.py aapl.csv
    # a universe (the real gate -- edge must hold across symbols):
    PYTHONPATH=. python scripts/validate_model.py aapl.csv msft.csv spy.csv --cost-bps 5

Each CSV needs a ``close`` column (see scripts/fetch_history.py). The gate is
deliberately conservative:
  * returns are NET of a round-trip transaction cost (commission + slippage);
  * holding periods are NON-overlapping (no inflated Sharpe);
  * the edge must be STABLE across out-of-sample sub-periods, not one lucky run;
  * and it must hold across MULTIPLE symbols -- a single symbol is flagged as
    necessary-but-not-sufficient even when it passes.
No edge -> do not trade, regardless of code quality.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Canonical methodology now lives in services.backtesting.edge; this script is
# the CLI front-end. Names are re-exported so existing imports keep working.
from services.backtesting.edge import (
    DEFAULT_COST_BPS,
    HORIZON,  # noqa: F401  # re-exported: part of this script's documented contract
    MIN_HIT_RATE,
    MIN_SHARPE,
    MIN_STABLE_FRAC,
    MIN_SYMBOLS_FOR_CONFIRM,  # noqa: F401  # re-exported
    STABILITY_FOLDS,
    aggregate_verdict,
    annualized_sharpe,
    net_period_returns,
    stability_fraction,
    symbol_has_edge,
)
from services.backtesting.edge import evaluate_symbol as _evaluate_symbol_dated

__all__ = [
    "aggregate_verdict",
    "annualized_sharpe",
    "net_period_returns",
    "stability_fraction",
    "symbol_has_edge",
]


def _synthetic_close(n: int = 4000, seed: int = 7) -> np.ndarray:
    # A TRUE zero-drift random walk: the harness must report NO edge on it.
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(0.0, 0.012, n))


def _load_csv_close(path: str) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["close"]) for r in rows], dtype=float)


def evaluate_symbol(name: str, close: np.ndarray, cost_bps: float) -> dict[str, Any]:
    """CLI-compatible wrapper returning the summary dict."""
    return _evaluate_symbol_dated(name, close, cost_bps).summary


def _print_report(results: list[dict[str, Any]], cost_bps: float) -> None:
    print("\n=== Out-of-sample edge validation (net of costs) ===")
    print(f"cost model         : {cost_bps:.1f} bps round-trip per holding period")
    print(f"gate (per symbol)  : hit-rate > {MIN_HIT_RATE:.2f}, Sharpe > {MIN_SHARPE:.2f}, "
          f"stability >= {MIN_STABLE_FRAC:.2f} of {STABILITY_FOLDS} sub-periods")
    header = f"{'symbol':<22}{'bars':>7}{'hit':>7}{'Sharpe':>8}{'stab':>7}{'return':>10}{'maxDD':>9}  edge"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        if r.get("insufficient"):
            print(f"{r['name'][:21]:<22}{r['bars']:>7}{'':>7}{'':>8}{'':>7}{'':>10}{'':>9}  n/a (too little data)")
            continue
        print(
            f"{r['name'][:21]:<22}{r['bars']:>7}{r['hit_rate']:>7.3f}{r['sharpe']:>8.2f}"
            f"{r['stability']:>7.2f}{r['total_return']:>9.1%}{r['max_drawdown']:>9.1%}"
            f"  {'YES' if r['edge'] else 'no'}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Measure out-of-sample edge (net of costs) across symbols.")
    parser.add_argument("csvs", nargs="*", help="CSV files with a 'close' column (none -> synthetic null self-test)")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help=f"round-trip cost per holding period in bps (default {DEFAULT_COST_BPS})")
    args = parser.parse_args(argv[1:])

    if args.csvs:
        results = [
            evaluate_symbol(Path(path).name, _load_csv_close(path), args.cost_bps)
            for path in args.csvs
        ]
    else:
        results = [evaluate_symbol("synthetic (null)", _synthetic_close(), args.cost_bps)]

    _print_report(results, args.cost_bps)
    verdict, _green = aggregate_verdict(results)
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
