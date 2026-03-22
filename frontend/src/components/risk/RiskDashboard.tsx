import { ShieldAlert, TrendingDown, Activity, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';
import type { RiskMetrics } from '../../types';
import VaRDisplay from './VaRDisplay';
import CircuitBreakerStatus from './CircuitBreakerStatus';

interface RiskDashboardProps {
  metrics: RiskMetrics;
}

export default function RiskDashboard({ metrics }: RiskDashboardProps) {
  const formatPct = (v: number) => `${(v * 100).toFixed(2)}%`;
  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(v);

  const cards = [
    {
      label: 'VaR (95%)',
      value: formatCurrency(metrics.var_95),
      icon: ShieldAlert,
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
    },
    {
      label: 'VaR (99%)',
      value: formatCurrency(metrics.var_99),
      icon: AlertTriangle,
      color: 'text-orange-500',
      bgColor: 'bg-orange-500/10',
    },
    {
      label: 'Max Drawdown',
      value: formatPct(metrics.max_drawdown),
      icon: TrendingDown,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
    },
    {
      label: 'Current Drawdown',
      value: formatPct(metrics.current_drawdown),
      icon: Activity,
      color:
        metrics.current_drawdown > metrics.max_drawdown * 0.5
          ? 'text-red-500'
          : 'text-yellow-500',
      bgColor:
        metrics.current_drawdown > metrics.max_drawdown * 0.5
          ? 'bg-red-500/10'
          : 'bg-yellow-500/10',
    },
    {
      label: 'Beta',
      value: metrics.beta.toFixed(2),
      icon: Activity,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Volatility',
      value: formatPct(metrics.volatility),
      icon: Activity,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
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
        <VaRDisplay var95={metrics.var_95} var99={metrics.var_99} cvar95={metrics.cvar_95} />
        <CircuitBreakerStatus
          status={metrics.circuit_breaker_status}
          reason={metrics.circuit_breaker_reason}
        />
      </div>
    </div>
  );
}
