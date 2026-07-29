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

  // A metric the backend could not compute arrives as undefined. It renders
  // as a grey em-dash: a coloured 0.00 reads as a measurement, and this
  // platform never shows a performance figure the data does not support.
  const UNAVAILABLE = '—';
  const muted = { color: 'text-gray-500', bgColor: 'bg-gray-500/10' };
  const signed = (v: number) => (v >= 0 ? 'text-green-500' : 'text-red-500');
  const signedBg = (v: number) => (v >= 0 ? 'bg-green-500/10' : 'bg-red-500/10');

  interface Card {
    label: string;
    value: string;
    sub?: string;
    icon: typeof DollarSign;
    color: string;
    bgColor: string;
  }

  const cards: Card[] = [
    {
      label: 'Total Equity',
      value: formatCurrency(data.total_equity),
      icon: DollarSign,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Daily P&L',
      value:
        data.daily_pnl === undefined
          ? UNAVAILABLE
          : `${data.daily_pnl >= 0 ? '+' : ''}${formatCurrency(data.daily_pnl)}`,
      sub:
        data.daily_pnl_pct === undefined
          ? undefined
          : `${data.daily_pnl_pct >= 0 ? '+' : ''}${data.daily_pnl_pct.toFixed(2)}%`,
      icon: (data.daily_pnl ?? 0) >= 0 ? TrendingUp : TrendingDown,
      ...(data.daily_pnl === undefined
        ? muted
        : { color: signed(data.daily_pnl), bgColor: signedBg(data.daily_pnl) }),
    },
    {
      label: 'Total Return',
      value:
        data.total_return_pct === undefined
          ? UNAVAILABLE
          : `${data.total_return_pct >= 0 ? '+' : ''}${data.total_return_pct.toFixed(2)}%`,
      sub: data.total_return === undefined ? undefined : formatCurrency(data.total_return),
      icon: BarChart3,
      ...(data.total_return === undefined
        ? muted
        : { color: signed(data.total_return), bgColor: signedBg(data.total_return) }),
    },
    {
      label: 'Sharpe Ratio',
      value: data.sharpe_ratio === undefined ? UNAVAILABLE : data.sharpe_ratio.toFixed(2),
      sub: data.sharpe_ratio === undefined ? 'needs 20+ days' : undefined,
      icon: Target,
      ...(data.sharpe_ratio === undefined
        ? muted
        : {
            color: data.sharpe_ratio >= 1 ? 'text-green-500' : 'text-yellow-500',
            bgColor: data.sharpe_ratio >= 1 ? 'bg-green-500/10' : 'bg-yellow-500/10',
          }),
    },
    {
      label: 'Win Rate',
      value:
        data.win_rate === undefined
          ? UNAVAILABLE
          : `${(data.win_rate * 100).toFixed(1)}%`,
      sub:
        data.win_rate === undefined
          ? 'no closed trades'
          : `${data.closed_trades} closed`,
      icon: Activity,
      ...(data.win_rate === undefined
        ? muted
        : {
            color: data.win_rate >= 0.5 ? 'text-green-500' : 'text-red-500',
            bgColor: data.win_rate >= 0.5 ? 'bg-green-500/10' : 'bg-red-500/10',
          }),
    },
    {
      label: 'Max Drawdown',
      value:
        data.max_drawdown === undefined
          ? UNAVAILABLE
          : `-${(data.max_drawdown * 100).toFixed(2)}%`,
      icon: TrendingDown,
      ...(data.max_drawdown === undefined
        ? muted
        : { color: 'text-red-500', bgColor: 'bg-red-500/10' }),
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
