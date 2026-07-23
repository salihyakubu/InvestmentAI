"""Edge-harness backtesting as a library (canonical home of the Stage-2 logic).

``scripts/validate_model.py`` pioneered this methodology and now imports from
here; the API's backtest endpoints call :func:`run_backtest`. The rules are
deliberately conservative and are the same ones the Stage-2 gate enforces:

* returns are NET of a round-trip transaction cost per holding period;
* holding periods are NON-overlapping (overlapping windows count each move
  ~horizon times and inflate Sharpe);
* the edge must be STABLE across out-of-sample sub-periods; and
* a single passing symbol is necessary but NOT sufficient.

The engine is pure and synchronous: callers fetch price series (async,
provider-based) and hand them in, so tests inject data and never touch the
network, and the API can push the CPU-bound work to a thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from backtesting.performance import PerformanceAnalyzer
from services.prediction.training.data_loader import TrainingDataLoader

# Forward-return horizon for labels AND the strategy holding period (they must
# match: the model predicts the H-bar-ahead direction, so a position is held H bars).
HORIZON = 5
DEFAULT_COST_BPS = 5.0

# Per-symbol gate thresholds (Stage-2 gate values -- keep in sync with GO_LIVE).
MIN_HIT_RATE = 0.52
MIN_SHARPE = 0.5
STABILITY_FOLDS = 4
MIN_STABLE_FRAC = 0.75
MIN_SYMBOL_PASS_FRAC = 0.60
MIN_SYMBOLS_FOR_CONFIRM = 3
MIN_TRAIN_ROWS = 200


# ---------------------------------------------------------------------------
# Pure, unit-tested evaluation logic (no ML here)
# ---------------------------------------------------------------------------


def net_period_returns(
    dirs: np.ndarray, fwd_ret: np.ndarray, horizon: int, cost_bps: float
) -> np.ndarray:
    """Net-of-cost, NON-overlapping per-period strategy returns."""
    gross = (dirs * fwd_ret)[::horizon].astype(float)
    pos = dirs[::horizon]
    cost = (pos != 0).astype(float) * (cost_bps / 1e4)
    net: np.ndarray = gross - cost
    return net


def annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    sd = float(returns.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def stability_fraction(returns: np.ndarray, n_folds: int = STABILITY_FOLDS) -> float:
    """Fraction of contiguous OOS sub-periods whose net return is positive."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < n_folds:
        return 0.0
    folds = np.array_split(returns, n_folds)
    positive = sum(1 for f in folds if f.size and float(f.sum()) > 0.0)
    return positive / n_folds


def symbol_has_edge(hit_rate: float, sharpe: float, stability: float) -> bool:
    return hit_rate > MIN_HIT_RATE and sharpe > MIN_SHARPE and stability >= MIN_STABLE_FRAC


def aggregate_verdict(results: list[dict[str, Any]]) -> tuple[str, bool]:
    """Combine per-symbol results into an overall go/no-go."""
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
# Features (harness-local, simple by design -- see validate_model.py history)
# ---------------------------------------------------------------------------


def build_features(close: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = len(close)
    ret1 = np.zeros(n)
    ret1[1:] = np.diff(close) / close[:-1]

    def roll(a: np.ndarray, w: int, fn: Any) -> np.ndarray:
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


@dataclass
class SymbolEvaluation:
    """Per-symbol harness outcome plus the dated period series for assembly."""

    summary: dict[str, Any]
    # Aligned per non-overlapping OOS period (empty when insufficient):
    net_returns: np.ndarray = field(default_factory=lambda: np.empty(0))
    directions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    period_dates: list[datetime] = field(default_factory=list)
    period_prices: np.ndarray = field(default_factory=lambda: np.empty(0))


def evaluate_symbol(
    name: str,
    close: np.ndarray,
    cost_bps: float,
    dates: list[datetime] | None = None,
) -> SymbolEvaluation:
    """Train, predict out-of-sample, and score one symbol (dated when possible)."""
    X, names = build_features(close)
    valid = ~np.isnan(X).any(axis=1)
    X, close_v = X[valid], close[valid]
    dates_v = [d for d, ok in zip(dates, valid, strict=True) if ok] if dates else None

    loader = TrainingDataLoader(
        target_horizon_bars=HORIZON, up_threshold=0.004, down_threshold=-0.004
    )
    ds = loader.load_training_data(X, close_v, feature_names=names)
    if ds.returns is None:  # tree-path datasets always carry returns
        raise ValueError("training dataset missing forward returns")

    split = int(len(ds.X) * 0.7)
    if split < MIN_TRAIN_ROWS or len(ds.X) - split < STABILITY_FOLDS * HORIZON:
        return SymbolEvaluation(
            summary={"name": name, "bars": len(close), "insufficient": True, "edge": False}
        )

    # Local import keeps xgboost/OpenMP out of the pure helpers' import path.
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
    hit_rate = (
        float((np.sign(dirs[traded]) == np.sign(r_test[traded])).mean())
        if traded.any()
        else 0.0
    )

    net = net_period_returns(dirs, r_test, HORIZON, cost_bps)
    ppy = 252 / HORIZON
    sharpe = annualized_sharpe(net, ppy)
    stability = stability_fraction(net)
    equity = np.concatenate([[10_000.0], 10_000.0 * np.cumprod(1 + net)])
    m = PerformanceAnalyzer.compute_metrics(equity, [], periods_per_year=ppy)

    # Dated period series: the dataset drops the last HORIZON rows for labels,
    # so test row i maps to valid-row index split + i; periods sample every
    # HORIZON-th test row (matching net_period_returns' stride).
    period_idx = np.arange(0, len(dirs), HORIZON)
    if dates_v is not None:
        period_dates = [dates_v[split + int(i)] for i in period_idx]
    else:
        period_dates = []
    period_prices = close_v[split : split + len(dirs)][period_idx]

    return SymbolEvaluation(
        summary={
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
        },
        net_returns=net,
        directions=dirs[period_idx],
        period_dates=period_dates,
        period_prices=period_prices,
    )


# ---------------------------------------------------------------------------
# Portfolio-level backtest (the API entry point)
# ---------------------------------------------------------------------------


def run_backtest(
    series: dict[str, tuple[list[datetime], np.ndarray]],
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    initial_capital: float = 10_000.0,
) -> dict[str, Any]:
    """Backtest an equal-weight portfolio of per-symbol harness strategies.

    Args:
        series: symbol -> (dates, closes) daily series, oldest first.
        cost_bps: round-trip cost per holding period, in basis points.
        initial_capital: starting equity for the reported curve.

    Returns a JSON-safe dict matching the dashboard's ``BacktestResult`` shape
    (equity_curve, trades, headline metrics) plus ``verdict`` and
    ``per_symbol`` detail. Portfolio period return = equal-weight mean of the
    symbols active in that period; capital is split across active symbols, so
    per-trade P&L uses the per-symbol slice of capital.
    """
    evals: dict[str, SymbolEvaluation] = {
        sym: evaluate_symbol(sym, closes, cost_bps, dates)
        for sym, (dates, closes) in series.items()
    }
    summaries = [e.summary for e in evals.values()]
    verdict, green = aggregate_verdict(summaries)

    judged = {s: e for s, e in evals.items() if not e.summary.get("insufficient")}
    n_periods = max((e.net_returns.size for e in judged.values()), default=0)

    equity_curve: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    portfolio_returns: list[float] = []

    if n_periods:
        # Portfolio return per period index = mean over symbols active then.
        for i in range(n_periods):
            active = [e.net_returns[i] for e in judged.values() if i < e.net_returns.size]
            portfolio_returns.append(float(np.mean(active)) if active else 0.0)

        # Date each period from the earliest available per-symbol date.
        period_dates: list[datetime | None] = []
        for i in range(n_periods):
            candidates = [
                e.period_dates[i]
                for e in judged.values()
                if i < len(e.period_dates)
            ]
            period_dates.append(min(candidates) if candidates else None)

        equity = initial_capital
        prev_date = None
        first_date = period_dates[0] if period_dates and period_dates[0] else None
        equity_curve.append(
            {
                "date": first_date.isoformat() if first_date else "",
                "equity": round(equity, 2),
            }
        )
        for i, r in enumerate(portfolio_returns):
            equity *= 1.0 + r
            d = period_dates[i] or prev_date
            prev_date = d
            equity_curve.append(
                {"date": d.isoformat() if d else "", "equity": round(equity, 2)}
            )

        # Trades: one entry per symbol per traded period (the simulated
        # non-overlapping hold), P&L on that symbol's equal-weight slice.
        for sym, e in judged.items():
            slice_capital = initial_capital / max(len(judged), 1)
            for i in range(e.net_returns.size):
                direction = int(e.directions[i]) if i < e.directions.size else 0
                if direction == 0:
                    continue
                entry_date = e.period_dates[i] if i < len(e.period_dates) else None
                exit_date = (
                    e.period_dates[i + 1] if i + 1 < len(e.period_dates) else entry_date
                )
                entry_price = float(e.period_prices[i]) if i < e.period_prices.size else 0.0
                net_r = float(e.net_returns[i])
                trades.append(
                    {
                        "symbol": sym,
                        "side": "buy" if direction > 0 else "sell",
                        "entry_date": entry_date.isoformat() if entry_date else "",
                        "exit_date": exit_date.isoformat() if exit_date else "",
                        "entry_price": round(entry_price, 6),
                        "exit_price": round(entry_price * (1.0 + direction * net_r), 6),
                        "quantity": round(slice_capital / entry_price, 6) if entry_price else 0.0,
                        "pnl": round(slice_capital * net_r, 2),
                        "return_pct": round(net_r * 100.0, 4),
                    }
                )
        trades.sort(key=lambda t: t["entry_date"])

    pr = np.asarray(portfolio_returns, dtype=float)
    traded_periods = pr[pr != 0.0]
    wins = float(traded_periods[traded_periods > 0].sum()) if traded_periods.size else 0.0
    losses = float(-traded_periods[traded_periods < 0].sum()) if traded_periods.size else 0.0
    final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    curve = np.array([p["equity"] for p in equity_curve]) if equity_curve else np.array([initial_capital])
    running_max = np.maximum.accumulate(curve)
    max_dd = float(((curve - running_max) / running_max).min()) if curve.size else 0.0

    return {
        "total_return": float(final_equity / initial_capital - 1.0),
        "sharpe_ratio": annualized_sharpe(pr, 252 / HORIZON),
        "max_drawdown": abs(max_dd),
        "win_rate": (
            float((traded_periods > 0).mean()) if traded_periods.size else 0.0
        ),
        "total_trades": len(trades),
        "profit_factor": (wins / losses) if losses > 0 else (wins and float("inf") or 0.0),
        "equity_curve": equity_curve,
        "trades": trades,
        "verdict": verdict,
        "edge_confirmed": green,
        "per_symbol": summaries,
        "cost_bps": cost_bps,
    }
