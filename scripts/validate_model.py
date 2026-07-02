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

import numpy as np

from backtesting.performance import PerformanceAnalyzer
from services.prediction.training.data_loader import TrainingDataLoader

# Forward-return horizon for labels AND the strategy holding period (they must
# match: the model predicts the H-bar-ahead direction, so a position is held H bars).
HORIZON = 5
# Default round-trip cost per non-overlapping holding period, in basis points
# (commission + slippage). Conservative for liquid US equities; raise for crypto
# or illiquid names via --cost-bps.
DEFAULT_COST_BPS = 5.0

# Per-symbol gate thresholds.
MIN_HIT_RATE = 0.52
MIN_SHARPE = 0.5
STABILITY_FOLDS = 4
MIN_STABLE_FRAC = 0.75          # >= this fraction of OOS sub-periods must be net-positive
# Cross-symbol gate.
MIN_SYMBOL_PASS_FRAC = 0.60     # >= this fraction of symbols must individually pass
MIN_SYMBOLS_FOR_CONFIRM = 3     # need at least this many symbols to CONFIRM an edge
MIN_TRAIN_ROWS = 200            # below this, a symbol has too little data to judge


# ---------------------------------------------------------------------------
# Pure, unit-tested evaluation logic (no ML here)
# ---------------------------------------------------------------------------


def net_period_returns(
    dirs: np.ndarray, fwd_ret: np.ndarray, horizon: int, cost_bps: float
) -> np.ndarray:
    """Net-of-cost, NON-overlapping per-period strategy returns.

    Positions are held ``horizon`` bars, so we sample every ``horizon``-th bar
    (no overlap -- overlapping windows count each move ~horizon times and inflate
    Sharpe). Each period holding a nonzero position pays a round-trip cost
    (enter + exit) of ``cost_bps`` basis points, which erodes marginal signals the
    way real trading does.
    """
    gross = (dirs * fwd_ret)[::horizon].astype(float)
    pos = dirs[::horizon]
    cost = (pos != 0).astype(float) * (cost_bps / 1e4)
    return gross - cost


def annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def stability_fraction(returns: np.ndarray, n_folds: int = STABILITY_FOLDS) -> float:
    """Fraction of contiguous OOS sub-periods whose net return is positive.

    Guards against an edge that comes from a single lucky stretch: we want it
    present across time. Returns 0.0 when there is too little data to split.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < n_folds:
        return 0.0
    folds = np.array_split(returns, n_folds)
    positive = sum(1 for f in folds if f.size and float(f.sum()) > 0.0)
    return positive / n_folds


def symbol_has_edge(hit_rate: float, sharpe: float, stability: float) -> bool:
    return hit_rate > MIN_HIT_RATE and sharpe > MIN_SHARPE and stability >= MIN_STABLE_FRAC


def aggregate_verdict(results: list[dict]) -> tuple[str, bool]:
    """Combine per-symbol results into an overall go/no-go.

    Returns (verdict_text, is_green_light). A single symbol is never a green
    light; confirming an edge needs a majority of at least MIN_SYMBOLS_FOR_CONFIRM
    symbols passing.
    """
    judged = [r for r in results if not r.get("insufficient")]
    if not judged:
        return "NO symbol had enough data to judge -> cannot assess edge", False

    passed = [r for r in judged if r["edge"]]
    n, k = len(judged), len(passed)
    frac = k / n

    if n == 1:
        if passed:
            return (
                "SINGLE-SYMBOL PASS -- necessary but NOT sufficient. Re-run across "
                f">= {MIN_SYMBOLS_FOR_CONFIRM} symbols before risking capital.",
                False,
            )
        return "NO demonstrable edge -> DO NOT trade", False

    if frac >= MIN_SYMBOL_PASS_FRAC and n >= MIN_SYMBOLS_FOR_CONFIRM:
        return f"EDGE STABLE across {k}/{n} symbols -> candidate for paper soak", True
    if frac >= MIN_SYMBOL_PASS_FRAC:
        return (
            f"edge in {k}/{n} symbols but fewer than {MIN_SYMBOLS_FOR_CONFIRM} tested "
            "-- add more symbols before trusting it",
            False,
        )
    return f"NO stable edge ({k}/{n} symbols passed) -> DO NOT trade", False


# ---------------------------------------------------------------------------
# Data + features
# ---------------------------------------------------------------------------


def _synthetic_close(n: int = 4000, seed: int = 7) -> np.ndarray:
    # A TRUE zero-drift random walk: no signal and no drift to ride, so the harness
    # must report NO edge on it (a non-zero mean would give a long-biased model a
    # free ride and falsely trip the gate). Larger n keeps the non-overlapping OOS
    # sample big enough that noise can't fluke past the gate.
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(0.0, 0.012, n))


def _load_csv_close(path: str) -> np.ndarray:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["close"]) for r in rows], dtype=float)


def _build_features(close: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = len(close)
    ret1 = np.zeros(n)
    ret1[1:] = np.diff(close) / close[:-1]

    def roll(a: np.ndarray, w: int, fn) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = fn(a[i - w + 1 : i + 1])
        return out

    def lag_ret(w: int) -> np.ndarray:
        out = np.full(n, np.nan)
        out[w:] = (close[w:] - close[:-w]) / close[:-w]
        return out

    sma5, sma20 = roll(close, 5, np.mean), roll(close, 20, np.mean)
    feats = {
        "ret1": ret1,
        "ret5": lag_ret(5),
        "mom10": lag_ret(10),
        "sma_ratio_5_20": sma5 / sma20,
        "px_to_sma20": close / sma20,
        "vol20": roll(ret1, 20, np.std),
    }
    names = sorted(feats)
    return np.column_stack([feats[k] for k in names]), names


# ---------------------------------------------------------------------------
# Per-symbol evaluation (trains the model)
# ---------------------------------------------------------------------------


def evaluate_symbol(name: str, close: np.ndarray, cost_bps: float) -> dict:
    X, names = _build_features(close)
    valid = ~np.isnan(X).any(axis=1)
    X, close_v = X[valid], close[valid]

    loader = TrainingDataLoader(target_horizon_bars=HORIZON, up_threshold=0.004, down_threshold=-0.004)
    ds = loader.load_training_data(X, close_v, feature_names=names)

    split = int(len(ds.X) * 0.7)
    if split < MIN_TRAIN_ROWS or len(ds.X) - split < STABILITY_FOLDS * HORIZON:
        return {"name": name, "bars": len(close), "insufficient": True, "edge": False}

    # Local import: keeps xgboost/OpenMP out of the import path for the pure
    # evaluation helpers (and their unit tests).
    from services.prediction.models.xgboost_model import XGBoostPredictor

    model = XGBoostPredictor(feature_names=names)
    model.train(
        ds.X[:split], ds.y[:split], ds.X[split:], ds.y[split:],
        returns_train=ds.returns[:split], returns_val=ds.returns[split:],
    )

    X_test, r_test = ds.X[split:], ds.returns[split:]
    preds = model.predict_batch(X_test)
    dirs = np.array(
        [1 if p.direction == "long" else -1 if p.direction == "short" else 0 for p in preds]
    )

    traded = dirs != 0
    hit_rate = float((np.sign(dirs[traded]) == np.sign(r_test[traded])).mean()) if traded.any() else 0.0

    net = net_period_returns(dirs, r_test, HORIZON, cost_bps)
    ppy = 252 / HORIZON
    sharpe = annualized_sharpe(net, ppy)
    stability = stability_fraction(net)
    equity = np.concatenate([[10_000.0], 10_000.0 * np.cumprod(1 + net)])
    m = PerformanceAnalyzer.compute_metrics(equity, [], periods_per_year=ppy)

    return {
        "name": name,
        "bars": len(close),
        "test_periods": int(net.size),
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "stability": stability,
        "total_return": float(m.get("total_return", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "insufficient": False,
        "edge": symbol_has_edge(hit_rate, sharpe, stability),
    }


def _print_report(results: list[dict], cost_bps: float) -> None:
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
