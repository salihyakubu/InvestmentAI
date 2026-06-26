"""Stage-2 gate: train the tree model and measure OUT-OF-SAMPLE edge.

    PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib \
        python scripts/validate_model.py            # synthetic random-walk (smoke)
    PYTHONPATH=. python scripts/validate_model.py prices.csv   # CSV with a 'close' column

On synthetic random-walk data there is (correctly) NO edge -- the point of that
run is to prove the train -> predict -> backtest -> report pipeline executes. Run
it on REAL history: a positive, stable out-of-sample Sharpe + hit-rate > 50% is
the gate for risking real money. No edge -> do not trade, regardless of code
quality.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from backtesting.performance import PerformanceAnalyzer
from services.prediction.models.xgboost_model import XGBoostPredictor
from services.prediction.training.data_loader import TrainingDataLoader

# Forward-return horizon for labels AND for the strategy holding period. The two
# must match: the model predicts the H-bar-ahead direction, so a position is held
# for H bars.
HORIZON = 5


def _synthetic_close(n: int = 4000, seed: int = 7) -> np.ndarray:
    # A TRUE zero-drift random walk: there is no signal and no drift to ride, so
    # the harness must report NO edge on it. (A non-zero mean here would give a
    # long-biased model a free ride and falsely trip the edge gate -- defeating
    # the point of the null self-test.) Larger n keeps the non-overlapping
    # out-of-sample sample big enough that noise can't fluke past the gate.
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(0.0, 0.012, n))


def _load_close(argv: list[str]) -> tuple[np.ndarray, str]:
    if len(argv) > 1:
        with open(argv[1]) as f:
            rows = list(csv.DictReader(f))
        close = np.array([float(r["close"]) for r in rows], dtype=float)
        return close, f"{argv[1]} ({len(close)} bars)"
    return _synthetic_close(), "synthetic random walk (2000 bars)"


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


def main(argv: list[str]) -> int:
    close, source = _load_close(argv)
    X, names = _build_features(close)

    valid = ~np.isnan(X).any(axis=1)
    X, close_v = X[valid], close[valid]

    loader = TrainingDataLoader(target_horizon_bars=HORIZON, up_threshold=0.004, down_threshold=-0.004)
    ds = loader.load_training_data(X, close_v, feature_names=names)

    split = int(len(ds.X) * 0.7)
    model = XGBoostPredictor(feature_names=names)
    model.train(
        ds.X[:split], ds.y[:split], ds.X[split:], ds.y[split:],
        returns_train=ds.returns[:split], returns_val=ds.returns[split:],
    )

    X_test, y_test, r_test = ds.X[split:], ds.y[split:], ds.returns[split:]
    preds = model.predict_batch(X_test)
    dirs = np.array(
        [1 if p.direction == "long" else -1 if p.direction == "short" else 0 for p in preds]
    )

    # Out-of-sample edge metrics.
    accuracy = float((np.argmax(model._classifier.predict_proba(X_test), axis=1) == y_test).mean())
    traded = dirs != 0
    hit_rate = float((np.sign(dirs[traded]) == np.sign(r_test[traded])).mean()) if traded.any() else 0.0

    # Non-overlapping holding periods: each prediction holds for HORIZON bars, so
    # we step by HORIZON instead of compounding the H-bar forward return at EVERY
    # bar. Overlapping windows (the naive `dirs * r_test` compounded daily) count
    # each move ~H times, grossly inflating both total return and Sharpe -- which
    # would make this go-live gate dangerously optimistic.
    strat_ret = (dirs * r_test)[::HORIZON]
    equity = np.concatenate([[10_000.0], 10_000.0 * np.cumprod(1 + strat_ret)])
    # ~252/HORIZON independent holding periods per year.
    m = PerformanceAnalyzer.compute_metrics(equity, [], periods_per_year=252 / HORIZON)

    print("\n=== Out-of-sample model validation ===")
    print(f"data source        : {source}")
    print(f"train / test bars  : {split} / {len(X_test)}")
    print(f"class accuracy     : {accuracy:.3f}  (baseline ~0.33 for 3 classes)")
    print(f"hit rate (traded)  : {hit_rate:.3f}  (need > 0.50 for edge)")
    print(f"strategy Sharpe    : {m.get('sharpe_ratio', 0):.3f}")
    print(f"total return       : {m.get('total_return', 0):.2%}")
    print(f"max drawdown       : {m.get('max_drawdown', 0):.2%}")

    has_edge = hit_rate > 0.52 and m.get("sharpe_ratio", 0) > 0.5
    verdict = (
        "EDGE DETECTED -> candidate for paper soak"
        if has_edge
        else "NO demonstrable edge -> DO NOT trade"
    )
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
