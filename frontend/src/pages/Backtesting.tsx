import { useState } from 'react';
import clsx from 'clsx';
import { format } from 'date-fns';
import EquityCurve from '../components/charts/EquityCurve';
import {
  useBacktestResults,
  useBacktestStatus,
  useRunBacktest,
} from '../api/hooks';
import type { BacktestConfig } from '../types';

export default function Backtesting() {
  const [config, setConfig] = useState<BacktestConfig>({
    // The harness needs enough daily bars to train (>=200 rows post-split),
    // so the default range is deliberately multi-year.
    start_date: '2018-01-01',
    end_date: '2025-12-31',
    symbols: ['AAPL', 'GOOGL', 'MSFT'],
    strategy: 'edge_harness',
    initial_capital: 100000,
    commission: 0.0005,
  });
  const [symbolInput, setSymbolInput] = useState(config.symbols.join(', '));
  const [jobId, setJobId] = useState<string | null>(null);

  const runBacktest = useRunBacktest();
  const status = useBacktestStatus(jobId);
  const jobState = status.data?.status;
  const results = useBacktestResults(jobId, jobState === 'completed');
  const result = results.data ?? null;
  const isRunning =
    runBacktest.isPending || jobState === 'queued' || jobState === 'running';
  const errorText =
    runBacktest.error?.message ??
    (jobState === 'failed' ? (status.data?.error ?? 'backtest failed') : null);

  const handleRun = () => {
    const symbols = symbolInput
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean);
    const updatedConfig = { ...config, symbols };
    setConfig(updatedConfig);
    setJobId(null);
    runBacktest.mutate(updatedConfig, {
      onSuccess: (data) => setJobId(data.id),
    });
  };

  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(v);

  return (
    <div className="space-y-6">
      <h1 className="page-title">Backtesting</h1>

      <div className="card">
        <div className="card-header">Configuration</div>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Start Date
            </label>
            <input
              type="date"
              value={config.start_date}
              onChange={(e) =>
                setConfig({ ...config, start_date: e.target.value })
              }
              className="input-field w-full text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              End Date
            </label>
            <input
              type="date"
              value={config.end_date}
              onChange={(e) =>
                setConfig({ ...config, end_date: e.target.value })
              }
              className="input-field w-full text-sm"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Symbols
            </label>
            <input
              type="text"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value)}
              placeholder="AAPL, GOOGL, MSFT"
              className="input-field w-full text-sm font-mono"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Strategy
            </label>
            <select
              value={config.strategy}
              onChange={(e) =>
                setConfig({ ...config, strategy: e.target.value })
              }
              className="input-field w-full text-sm"
            >
              <option value="edge_harness">
                ML Edge Harness (walk-forward, net of costs)
              </option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Initial Capital
            </label>
            <input
              type="number"
              value={config.initial_capital}
              onChange={(e) =>
                setConfig({
                  ...config,
                  initial_capital: parseFloat(e.target.value),
                })
              }
              className="input-field w-full text-sm font-mono"
            />
          </div>
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="btn-primary w-full disabled:opacity-50"
            >
              {isRunning ? 'Running...' : 'Run Backtest'}
            </button>
          </div>
        </div>
      </div>

      {errorText && (
        <div className="card border border-red-500/40">
          <div className="text-red-400 text-sm font-mono">
            Backtest failed: {errorText}
          </div>
        </div>
      )}

      {isRunning && !result && (
        <div className="card">
          <div className="text-sm text-gray-400">
            Running backtest (fetching history, training per symbol, scoring
            out-of-sample)... this takes up to a minute.
          </div>
        </div>
      )}

      {result && (
        <>
          {result.verdict && (
            <div className="card">
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
                Edge Verdict (net of costs, non-overlapping, stability-gated)
              </div>
              <div className="text-sm font-mono text-white">{result.verdict}</div>
            </div>
          )}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              {
                label: 'Total Return',
                value: `${(result.total_return * 100).toFixed(2)}%`,
                positive: result.total_return >= 0,
              },
              {
                label: 'Sharpe Ratio',
                value: result.sharpe_ratio.toFixed(2),
                positive: result.sharpe_ratio >= 1,
              },
              {
                label: 'Max Drawdown',
                value: `${(result.max_drawdown * 100).toFixed(2)}%`,
                positive: false,
              },
              {
                label: 'Win Rate',
                value: `${(result.win_rate * 100).toFixed(1)}%`,
                positive: result.win_rate >= 0.5,
              },
              {
                label: 'Total Trades',
                value: result.total_trades.toString(),
                positive: true,
              },
              {
                label: 'Profit Factor',
                value: result.profit_factor.toFixed(2),
                positive: result.profit_factor >= 1,
              },
            ].map((metric) => (
              <div key={metric.label} className="card">
                <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
                  {metric.label}
                </div>
                <div
                  className={clsx(
                    'text-lg font-bold font-mono',
                    metric.positive ? 'text-green-500' : 'text-red-500',
                  )}
                >
                  {metric.value}
                </div>
              </div>
            ))}
          </div>

          <EquityCurve data={result.equity_curve} />

          <div className="card">
            <div className="card-header">
              Trades ({result.trades.length})
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700">
                    {[
                      'Symbol',
                      'Side',
                      'Entry Date',
                      'Exit Date',
                      'Entry',
                      'Exit',
                      'Qty',
                      'P&L',
                      'Return',
                    ].map((h) => (
                      <th
                        key={h}
                        className="text-left text-xs text-gray-500 font-medium uppercase tracking-wider py-2 px-3"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.trades.slice(0, 50).map((trade, i) => (
                    <tr
                      key={i}
                      className="border-b border-gray-700/50 hover:bg-gray-800/50"
                    >
                      <td className="py-2 px-3 font-mono font-semibold text-white text-sm">
                        {trade.symbol}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={clsx(
                            'text-xs font-bold uppercase px-2 py-0.5 rounded',
                            trade.side === 'buy'
                              ? 'text-green-400 bg-green-500/10'
                              : 'text-red-400 bg-red-500/10',
                          )}
                        >
                          {trade.side}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-400 font-mono">
                        {format(new Date(trade.entry_date), 'MM/dd/yy')}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-400 font-mono">
                        {format(new Date(trade.exit_date), 'MM/dd/yy')}
                      </td>
                      <td className="py-2 px-3 font-mono text-sm text-gray-300">
                        {formatCurrency(trade.entry_price)}
                      </td>
                      <td className="py-2 px-3 font-mono text-sm text-gray-300">
                        {formatCurrency(trade.exit_price)}
                      </td>
                      <td className="py-2 px-3 font-mono text-sm text-gray-300">
                        {trade.quantity}
                      </td>
                      <td
                        className={clsx(
                          'py-2 px-3 font-mono text-sm font-semibold',
                          trade.pnl >= 0
                            ? 'text-green-500'
                            : 'text-red-500',
                        )}
                      >
                        {trade.pnl >= 0 ? '+' : ''}
                        {formatCurrency(trade.pnl)}
                      </td>
                      <td
                        className={clsx(
                          'py-2 px-3 font-mono text-sm font-semibold',
                          trade.return_pct >= 0
                            ? 'text-green-500'
                            : 'text-red-500',
                        )}
                      >
                        {trade.return_pct >= 0 ? '+' : ''}
                        {trade.return_pct.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
