import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Briefcase,
  CandlestickChart,
  ShieldAlert,
  Brain,
  FlaskConical,
  ScrollText,
  Settings,
} from 'lucide-react';
import clsx from 'clsx';

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { to: '/trading', label: 'Trading', icon: CandlestickChart },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/models', label: 'ML Models', icon: Brain },
  { to: '/backtesting', label: 'Backtesting', icon: FlaskConical },
  { to: '/audit', label: 'Audit Log', icon: ScrollText },
  { to: '/settings', label: 'Settings', icon: Settings },
];

export default function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 glass-panel border-r border-ink-hairline flex flex-col z-30">
      <div className="h-16 flex items-center gap-3 px-5">
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-accent-dark flex items-center justify-center shadow-[0_6px_16px_rgba(10,132,255,0.35)]">
          <CandlestickChart className="w-5 h-5 text-white" />
        </div>
        <span className="text-[17px] font-semibold text-white tracking-tight">
          InvestAI
        </span>
      </div>

      <nav className="flex-1 py-3 px-3 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              clsx(
                'group flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-[13.5px] font-medium',
                'transition-all duration-150',
                isActive
                  ? 'bg-white/10 text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]'
                  : 'text-gray-400 hover:text-gray-100 hover:bg-white/5',
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon
                  className={clsx(
                    'w-[18px] h-[18px] flex-shrink-0 transition-colors',
                    isActive
                      ? 'text-accent-light'
                      : 'text-gray-500 group-hover:text-gray-300',
                  )}
                />
                {item.label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="px-5 py-4 border-t border-ink-hairline">
        <div className="text-[11px] text-gray-600 tracking-wide">
          Investment AI Platform v1.0
        </div>
      </div>
    </aside>
  );
}
