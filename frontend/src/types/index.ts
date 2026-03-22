export interface Bar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  symbol: string;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  market_value: number;
  opened_at: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  limit_price?: number;
  stop_price?: number;
  status: 'pending' | 'submitted' | 'partial' | 'filled' | 'cancelled' | 'rejected';
  filled_quantity: number;
  filled_avg_price?: number;
  created_at: string;
  updated_at: string;
}

export interface Fill {
  id: string;
  order_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  commission: number;
  timestamp: string;
}

export interface Prediction {
  id: string;
  symbol: string;
  model_id: string;
  model_name: string;
  signal: 'buy' | 'sell' | 'hold';
  confidence: number;
  predicted_return: number;
  horizon: string;
  features: Record<string, number>;
  timestamp: string;
}

export interface PortfolioSummary {
  total_equity: number;
  cash: number;
  market_value: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  positions_count: number;
  open_orders_count: number;
}

export interface RiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  max_drawdown: number;
  current_drawdown: number;
  beta: number;
  volatility: number;
  correlation_matrix: Record<string, Record<string, number>>;
  circuit_breaker_status: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
  circuit_breaker_reason?: string;
  position_concentration: Record<string, number>;
}

export interface ModelInfo {
  id: string;
  name: string;
  type: string;
  version: string;
  status: 'active' | 'inactive' | 'training' | 'failed';
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  last_trained: string;
  feature_importance: Record<string, number>;
  prediction_count: number;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  event_type: string;
  component: string;
  action: string;
  details: Record<string, unknown>;
  user?: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}

export interface BacktestConfig {
  start_date: string;
  end_date: string;
  symbols: string[];
  strategy: string;
  initial_capital: number;
  commission: number;
}

export interface BacktestResult {
  id: string;
  config: BacktestConfig;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  equity_curve: { date: string; equity: number }[];
  trades: BacktestTrade[];
}

export interface BacktestTrade {
  symbol: string;
  side: 'buy' | 'sell';
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl: number;
  return_pct: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
  drawdown?: number;
}
