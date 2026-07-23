/**
 * Normalizers between raw API responses and the UI's typed models.
 *
 * The API serializes Decimals as strings, omits fields the UI expects, uses
 * null for not-yet-available metrics, and names some fields differently
 * (order_type vs type, daily_return_pct vs daily_pnl_pct). Components render
 * `x.toFixed(...)` and friends, so every numeric field must arrive as a real
 * number — a single undefined crashed the whole app before the error boundary
 * existed. All coercion lives here, at the data boundary, so components stay
 * simple.
 */

import type {
  BacktestResult,
  BacktestTrade,
  EquityPoint,
  ModelCalibration,
  ModelInfo,
  Order,
  PortfolioSummary,
  Position,
  Prediction,
  RiskMetrics,
} from '../types';

type Raw = Record<string, unknown>;

/** Coerce API numerics (number | string Decimal | null | missing) to number. */
export function num(v: unknown, fallback = 0): number {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

const str = (v: unknown, fallback = ''): string =>
  typeof v === 'string' ? v : fallback;

/** Like num(), but absent/garbage stays undefined so the UI can hide it. */
function numOpt(v: unknown): number | undefined {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

/**
 * validation_metrics carries calibration-selection keys only when the training
 * run recorded them (chosen_calibration: 'isotonic'|'sigmoid'|'none', plus
 * brier_isotonic / brier_sigmoid). Missing method -> undefined (line hidden);
 * missing Brier -> method shown alone. Never NaN.
 */
function normalizeCalibration(metrics: Raw): ModelCalibration | undefined {
  const method = str(metrics.chosen_calibration);
  if (method === '') return undefined;
  const brier =
    method === 'isotonic'
      ? numOpt(metrics.brier_isotonic)
      : method === 'sigmoid'
        ? numOpt(metrics.brier_sigmoid)
        : undefined;
  return brier === undefined ? { method } : { method, brier };
}

export function normalizePortfolioSummary(raw: Raw): PortfolioSummary {
  const totalEquity = num(raw.total_equity);
  const cash = num(raw.cash);
  return {
    total_equity: totalEquity,
    cash,
    market_value: num(raw.positions_value ?? raw.market_value),
    daily_pnl: num(raw.daily_pnl),
    daily_pnl_pct: num(raw.daily_pnl_pct ?? raw.daily_return_pct),
    total_return: num(raw.total_return ?? raw.realized_pnl),
    total_return_pct: num(raw.total_return_pct),
    sharpe_ratio: num(raw.sharpe_ratio),
    max_drawdown: num(raw.max_drawdown),
    win_rate: num(raw.win_rate),
    positions_count: num(raw.positions_count ?? raw.position_count),
    open_orders_count: num(raw.open_orders_count),
  };
}

export function normalizePosition(raw: Raw): Position {
  const quantity = num(raw.quantity);
  const entry = num(raw.avg_entry ?? raw.entry_price ?? raw.avg_entry_price);
  const current = num(raw.current_price, entry);
  const marketValue = num(raw.market_value, quantity * current);
  const unrealized = num(raw.unrealized_pnl);
  const costBasis = quantity * entry;
  return {
    id: str(raw.id, str(raw.symbol)),
    symbol: str(raw.symbol),
    side: raw.side === 'short' ? 'short' : 'long',
    quantity,
    entry_price: entry,
    current_price: current,
    unrealized_pnl: unrealized,
    unrealized_pnl_pct: num(
      raw.unrealized_pnl_pct,
      costBasis !== 0 ? (unrealized / Math.abs(costBasis)) * 100 : 0,
    ),
    market_value: marketValue,
    opened_at: str(raw.opened_at),
  };
}

export function normalizeOrder(raw: Raw): Order {
  return {
    id: str(raw.id),
    symbol: str(raw.symbol),
    side: raw.side === 'sell' ? 'sell' : 'buy',
    type: (str(raw.order_type ?? raw.type, 'market') as Order['type']),
    quantity: num(raw.quantity),
    limit_price: raw.limit_price == null ? undefined : num(raw.limit_price),
    stop_price: raw.stop_price == null ? undefined : num(raw.stop_price),
    status: (str(raw.status, 'pending') as Order['status']),
    filled_quantity: num(raw.filled_quantity),
    filled_avg_price:
      raw.filled_avg_price == null ? undefined : num(raw.filled_avg_price),
    created_at: str(raw.created_at),
    updated_at: str(raw.updated_at),
  };
}

export function normalizeRiskMetrics(raw: Raw): RiskMetrics {
  return {
    var_95: num(raw.var_95),
    var_99: num(raw.var_99),
    cvar_95: num(raw.cvar_95),
    cvar_99: num(raw.cvar_99),
    max_drawdown: num(raw.max_drawdown),
    current_drawdown: num(raw.current_drawdown),
    beta: num(raw.beta),
    volatility: num(raw.volatility),
    correlation_matrix:
      (raw.correlation_matrix as RiskMetrics['correlation_matrix']) ?? {},
    circuit_breaker_status: raw.circuit_breaker_active ? 'OPEN' : 'CLOSED',
    circuit_breaker_reason:
      raw.circuit_breaker_reason == null
        ? undefined
        : str(raw.circuit_breaker_reason),
    position_concentration:
      (raw.position_concentration as RiskMetrics['position_concentration']) ??
      {},
  };
}

export function normalizeModelInfo(raw: Raw): ModelInfo {
  // The API exposes registry metadata (model_name, validation_metrics{...});
  // the UI's richer fields (precision/recall/f1/sharpe) default to 0 until the
  // backend computes them from live prediction outcomes.
  const metrics = (raw.validation_metrics ?? {}) as Raw;
  const calibration = normalizeCalibration(metrics);
  return {
    id: str(raw.id, str(raw.model_name)),
    name: str(raw.model_name, 'model'),
    type: str(raw.model_type, str(raw.model_name)),
    version: String(raw.version ?? ''),
    status: raw.is_active ? 'active' : 'inactive',
    accuracy: num(metrics.val_accuracy ?? metrics.accuracy),
    precision: num(metrics.precision),
    recall: num(metrics.recall),
    f1_score: num(metrics.f1_score),
    sharpe_ratio: num(metrics.sharpe_ratio),
    last_trained: str(raw.trained_at),
    feature_importance:
      (raw.feature_importance as ModelInfo['feature_importance']) ?? {},
    prediction_count: num(raw.prediction_count),
    ...(calibration ? { calibration } : {}),
  };
}

/** API direction vocabulary (long/short/flat) -> UI signal (buy/sell/hold). */
const DIRECTION_TO_SIGNAL: Record<string, Prediction['signal']> = {
  long: 'buy',
  short: 'sell',
  flat: 'hold',
};

export function normalizePrediction(raw: Raw): Prediction {
  // The API speaks the trading vocabulary (direction/expected_return); the UI
  // components were written against signal/predicted_return. Anything
  // unrecognized degrades to 'hold' -- an unknown direction must never crash
  // the dashboard (undefined.icon did exactly that when this endpoint first
  // returned real rows).
  return {
    id: str(raw.id, `${str(raw.symbol)}-${str(raw.timestamp)}`),
    symbol: str(raw.symbol),
    model_id: str(raw.model_id),
    model_name: str(raw.model_id, 'model'),
    signal: DIRECTION_TO_SIGNAL[str(raw.direction)] ?? 'hold',
    confidence: num(raw.confidence),
    predicted_return: num(raw.expected_return),
    horizon: '5m',
    features: (raw.features as Record<string, number>) ?? {},
    timestamp: str(raw.timestamp),
  };
}

/** Portfolio snapshots -> equity-curve points ({time,total_equity} -> {date,equity}). */
export function normalizeEquityPoint(raw: Raw): EquityPoint {
  return {
    date: str(raw.time ?? raw.date),
    equity: num(raw.total_equity ?? raw.equity),
    drawdown:
      raw.max_drawdown_pct == null ? undefined : num(raw.max_drawdown_pct),
  };
}

/** Backtest job result -> UI shape. Every numeric coerced; arrays defensive. */
export function normalizeBacktestResult(raw: Raw): BacktestResult {
  const curve = Array.isArray(raw.equity_curve) ? (raw.equity_curve as Raw[]) : [];
  const trades = Array.isArray(raw.trades) ? (raw.trades as Raw[]) : [];
  return {
    id: str(raw.id),
    config: (raw.config as BacktestResult['config']) ?? {
      start_date: '', end_date: '', symbols: [], strategy: '',
      initial_capital: 0, commission: 0,
    },
    total_return: num(raw.total_return),
    sharpe_ratio: num(raw.sharpe_ratio),
    max_drawdown: num(raw.max_drawdown),
    win_rate: num(raw.win_rate),
    total_trades: num(raw.total_trades),
    profit_factor: Number.isFinite(Number(raw.profit_factor)) ? num(raw.profit_factor) : 0,
    equity_curve: curve.map((p) => ({ date: str(p.date), equity: num(p.equity) })),
    trades: trades.map((t): BacktestTrade => ({
      symbol: str(t.symbol),
      side: t.side === 'sell' ? 'sell' : 'buy',
      entry_date: str(t.entry_date),
      exit_date: str(t.exit_date),
      entry_price: num(t.entry_price),
      exit_price: num(t.exit_price),
      quantity: num(t.quantity),
      pnl: num(t.pnl),
      return_pct: num(t.return_pct),
    })),
    ...(typeof raw.verdict === 'string' ? { verdict: raw.verdict } : {}),
  };
}

