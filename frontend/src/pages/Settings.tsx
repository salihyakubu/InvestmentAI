import clsx from 'clsx';
import { useSettings, useTradingMode } from '../api/hooks';
import { num } from '../api/normalize';
import GoLiveReadiness from '../components/settings/GoLiveReadiness';

/**
 * Settings is intentionally READ-ONLY.
 *
 * Trading mode, risk limits, symbols, and credentials are deployment
 * configuration (environment variables on the server), not browser state: a
 * compromised session must never be able to flip a paper system to live, relax
 * a risk limit, or exfiltrate a key. This page shows the platform's REAL
 * configuration as reported by the backend.
 */
export default function Settings() {
  const { data: tradingMode = 'paper' } = useTradingMode();
  const { data: rules } = useSettings();

  const pct = (v: unknown) => `${(num(v) * 100).toFixed(1)}%`;

  const riskRows: { label: string; value: string }[] = rules
    ? [
        { label: 'Max position size', value: pct(rules.max_position_pct) },
        { label: 'Max sector exposure', value: pct(rules.max_sector_pct) },
        { label: 'Max asset-class exposure', value: pct(rules.max_asset_class_pct) },
        { label: 'Max single order', value: pct(rules.max_single_order_pct) },
        { label: 'Max daily drawdown', value: pct(rules.max_daily_drawdown_pct) },
        { label: 'Max total drawdown', value: pct(rules.max_total_drawdown_pct) },
        { label: 'Max pairwise correlation', value: num(rules.max_pairwise_correlation).toFixed(2) },
        { label: 'Max open positions', value: String(num(rules.max_portfolio_positions)) },
        { label: 'Max portfolio VaR (95%)', value: pct(rules.max_portfolio_var_95) },
        { label: 'Circuit breaker daily loss', value: pct(rules.circuit_breaker_loss_pct) },
        {
          label: 'Circuit breaker cooldown',
          value: `${num(rules.circuit_breaker_cooldown_minutes)} min`,
        },
      ]
    : [];

  return (
    <div className="space-y-6 max-w-4xl">
      <h1 className="page-title">Settings</h1>

      {/* Trading Mode -- read-only, reported by the backend */}
      <div className="card">
        <div className="card-header">Trading Mode</div>
        <div className="flex items-center gap-4">
          <span
            className={clsx(
              'px-4 py-2 rounded-full text-sm font-bold uppercase tracking-wider',
              tradingMode === 'live'
                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
            )}
          >
            {tradingMode}
          </span>
          <p className="text-sm text-gray-400">
            Set by the deployment&rsquo;s <code className="font-mono">TRADING_MODE</code>{' '}
            environment variable. By design it cannot be changed from the
            browser &mdash; switching to live requires a deliberate redeploy.
          </p>
        </div>
      </div>

      {/* Go-live: preconditions + operator procedure; never a toggle */}
      <GoLiveReadiness />

      {/* Risk Parameters -- the risk engine's real limits */}
      <div className="card">
        <div className="card-header">Risk Parameters (enforced by the risk engine)</div>
        {rules ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
            {riskRows.map((row) => (
              <div
                key={row.label}
                className="flex items-center justify-between py-2 border-b border-gray-800"
              >
                <span className="text-sm text-gray-400">{row.label}</span>
                <span className="text-sm font-mono text-white">{row.value}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500">Loading risk configuration…</p>
        )}
        <p className="text-xs text-gray-500 mt-3">
          Limits are deployment configuration; every order is checked against
          them server-side before it can reach a broker.
        </p>
      </div>

      {/* Credentials -- never handled by the UI */}
      <div className="card">
        <div className="card-header">Broker &amp; Data Credentials</div>
        <p className="text-sm text-gray-400">
          API keys (Alpaca, Binance) are configured as environment variables on
          the deployment and are never entered, stored, or displayed in this
          interface. To rotate a key: revoke it at the broker, update the
          deployment&rsquo;s variables, and redeploy.
        </p>
      </div>
    </div>
  );
}
