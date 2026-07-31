import { ShieldAlert, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';
import type { RiskMetrics } from '../../types';
import VaRDisplay from './VaRDisplay';
import CircuitBreakerStatus from './CircuitBreakerStatus';

interface RiskDashboardProps {
  metrics: RiskMetrics;
}

export default function RiskDashboard({ metrics }: RiskDashboardProps) {
  // An unreported metric renders as a grey em-dash. A 0.00 VaR or 0.00%
  // volatility reads as a MEASUREMENT of safety, which is the exact lie
  // this page used to tell. Beta has no tile at all: nothing computes it.
  const UNAVAILABLE = '\u2014';
  const muted = { color: 'text-gray-500', bgColor: 'bg-gray-500/10' };
  const formatPct = (v: number | undefined) =>
    v === undefined ? UNAVAILABLE : `${(v * 100).toFixed(2)}%`;
  const formatCurrency = (v: number | undefined) =>
    v === undefined
      ? UNAVAILABLE
      : new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 2,
        }).format(v);

  const drawdownHot =
    metrics.current_drawdown !== undefined &&
    metrics.max_drawdown !== undefined &&
    metrics.current_drawdown > metrics.max_drawdown * 0.5;

  const cards = [
    {
      label: 'VaR (95%)',
      value: formatCurrency(metrics.var_95),
      icon: ShieldAlert,
      ...(metrics.var_95 === undefined
        ? muted
        : { color: 'text-yellow-500', bgColor: 'bg-yellow-500/10' }),
    },
    {
      label: 'VaR (99%)',
      value: formatCurrency(metrics.var_99),
      icon: AlertTriangle,
      ...(metrics.var_99 === undefined
        ? muted
        : { color: 'text-orange-500', bgColor: 'bg-orange-500/10' }),
    },
    {
      label: 'CVaR (95%)',
      value: formatCurrency(metrics.cvar_95),
      icon: AlertTriangle,
      ...(metrics.cvar_95 === undefined
        ? muted
        : { color: 'text-orange-400', bgColor: 'bg-orange-500/10' }),
    },
    {
      label: 'Max Drawdown',
      value: formatPct(metrics.max_drawdown),
      icon: TrendingDown,
      ...(metrics.max_drawdown === undefined
        ? muted
        : { color: 'text-red-500', bgColor: 'bg-red-500/10' }),
    },
    {
      label: 'Current Drawdown',
      value: formatPct(metrics.current_drawdown),
      icon: Activity,
      ...(metrics.current_drawdown === undefined
        ? muted
        : drawdownHot
          ? { color: 'text-red-500', bgColor: 'bg-red-500/10' }
          : { color: 'text-yellow-500', bgColor: 'bg-yellow-500/10' }),
    },
    {
      label: 'Volatility',
      value: formatPct(metrics.volatility),
      icon: Activity,
      ...(metrics.volatility === undefined
        ? muted
        : { color: 'text-purple-400', bgColor: 'bg-purple-500/10' }),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {cards.map((card) => (
          <div key={card.label} className="card">
            <div className="flex items-center gap-2 mb-2">
              <div className={clsx('p-1.5 rounded-lg', card.bgColor)}>
                <card.icon className={clsx('w-4 h-4', card.color)} />
              </div>
              <span className="text-xs text-gray-500 uppercase tracking-wider">
                {card.label}
              </span>
            </div>
            <div className={clsx('text-lg font-bold font-mono', card.color)}>
              {card.value}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {metrics.var_95 !== undefined &&
        metrics.var_99 !== undefined &&
        metrics.cvar_95 !== undefined ? (
          <VaRDisplay
            var95={metrics.var_95}
            var99={metrics.var_99}
            cvar95={metrics.cvar_95}
          />
        ) : (
          <div className="card h-full flex items-center justify-center text-gray-500 text-sm">
            Value-at-risk not reported yet
          </div>
        )}
        <CircuitBreakerStatus
          status={metrics.circuit_breaker_status}
          reason={metrics.circuit_breaker_reason}
        />
      </div>
    </div>
  );
}
