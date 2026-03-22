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
    <aside className="fixed left-0 top-0 h-screen w-60 bg-gray-950 border-r border-gray-800 flex flex-col z-30">
      <div className="h-16 flex items-center px-5 border-b border-gray-800">
        <CandlestickChart className="w-7 h-7 text-blue-500 mr-2" />
        <span className="text-lg font-bold text-white tracking-tight">
          InvestAI
        </span>
      </div>

      <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-blue-600/20 text-blue-400'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800',
              )
            }
          >
            <item.icon className="w-5 h-5 flex-shrink-0" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-gray-800">
        <div className="text-xs text-gray-500 text-center">
          Investment AI Platform v1.0
        </div>
      </div>
    </aside>
  );
}
