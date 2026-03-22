import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  BarChart3,
  Target,
  Activity,
} from 'lucide-react';
import clsx from 'clsx';
import type { PortfolioSummary as PortfolioSummaryType } from '../../types';

interface PortfolioSummaryProps {
  data: PortfolioSummaryType;
}

export default function PortfolioSummary({ data }: PortfolioSummaryProps) {
  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(v);

  const cards = [
    {
      label: 'Total Equity',
      value: formatCurrency(data.total_equity),
      icon: DollarSign,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Daily P&L',
      value: `${data.daily_pnl >= 0 ? '+' : ''}${formatCurrency(data.daily_pnl)}`,
      sub: `${data.daily_pnl_pct >= 0 ? '+' : ''}${data.daily_pnl_pct.toFixed(2)}%`,
      icon: data.daily_pnl >= 0 ? TrendingUp : TrendingDown,
      color: data.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500',
      bgColor:
        data.daily_pnl >= 0 ? 'bg-green-500/10' : 'bg-red-500/10',
    },
    {
      label: 'Total Return',
      value: `${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct.toFixed(2)}%`,
      sub: formatCurrency(data.total_return),
      icon: BarChart3,
      color: data.total_return >= 0 ? 'text-green-500' : 'text-red-500',
      bgColor:
        data.total_return >= 0 ? 'bg-green-500/10' : 'bg-red-500/10',
    },
    {
      label: 'Sharpe Ratio',
      value: data.sharpe_ratio.toFixed(2),
      icon: Target,
      color: data.sharpe_ratio >= 1 ? 'text-green-500' : 'text-yellow-500',
      bgColor:
        data.sharpe_ratio >= 1 ? 'bg-green-500/10' : 'bg-yellow-500/10',
    },
    {
      label: 'Win Rate',
      value: `${(data.win_rate * 100).toFixed(1)}%`,
      icon: Activity,
      color: data.win_rate >= 0.5 ? 'text-green-500' : 'text-red-500',
      bgColor:
        data.win_rate >= 0.5 ? 'bg-green-500/10' : 'bg-red-500/10',
    },
    {
      label: 'Max Drawdown',
      value: `${(data.max_drawdown * 100).toFixed(2)}%`,
      icon: TrendingDown,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
    },
  ];

  return (
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
          {card.sub && (
            <div className="text-xs text-gray-500 font-mono mt-0.5">
              {card.sub}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
