import { Bell, Wifi, WifiOff } from 'lucide-react';
import clsx from 'clsx';
import { useAppStore } from '../../store';
import { usePortfolioSummary } from '../../api/hooks';

export default function Header() {
  const { tradingMode, connected, alerts } = useAppStore();
  const { data: portfolio } = usePortfolioSummary();
  const unreadCount = alerts.filter((a) => !a.read).length;

  const formatCurrency = (value: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);

  return (
    <header className="h-16 bg-gray-950 border-b border-gray-800 flex items-center justify-between px-6 fixed top-0 left-60 right-0 z-20">
      <div className="flex items-center gap-4">
        <div
          className={clsx(
            'px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider',
            tradingMode === 'live'
              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
              : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
          )}
        >
          {tradingMode === 'live' ? 'LIVE' : 'PAPER'} TRADING
        </div>

        <div className="flex items-center gap-2">
          {connected ? (
            <Wifi className="w-4 h-4 text-green-500" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-500" />
          )}
          <span
            className={clsx(
              'text-xs font-medium',
              connected ? 'text-green-500' : 'text-red-500',
            )}
          >
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        {portfolio && (
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="text-xs text-gray-500 uppercase">Equity</div>
              <div className="text-sm font-bold text-white font-mono">
                {formatCurrency(portfolio.total_equity)}
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500 uppercase">Daily P&L</div>
              <div
                className={clsx(
                  'text-sm font-bold font-mono',
                  portfolio.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500',
                )}
              >
                {portfolio.daily_pnl >= 0 ? '+' : ''}
                {formatCurrency(portfolio.daily_pnl)}
                <span className="text-xs ml-1">
                  ({portfolio.daily_pnl_pct >= 0 ? '+' : ''}
                  {portfolio.daily_pnl_pct.toFixed(2)}%)
                </span>
              </div>
            </div>
          </div>
        )}

        <button className="relative p-2 text-gray-400 hover:text-gray-200 transition-colors">
          <Bell className="w-5 h-5" />
          {unreadCount > 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-[10px] font-bold text-white flex items-center justify-center">
              {unreadCount}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
