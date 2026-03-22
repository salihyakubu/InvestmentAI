import { useState } from 'react';
import clsx from 'clsx';
import { format } from 'date-fns';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useAuditLogs } from '../api/hooks';

export default function AuditLog() {
  const [componentFilter, setComponentFilter] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: logs, isLoading } = useAuditLogs({
    component: componentFilter || undefined,
    risk_level: riskFilter || undefined,
    limit: 100,
  });

  const components = [
    '',
    'order_manager',
    'risk_engine',
    'ml_pipeline',
    'data_feed',
    'portfolio',
    'circuit_breaker',
  ];
  const riskLevels = ['', 'low', 'medium', 'high', 'critical'];

  const riskColor = (level: string) => {
    switch (level) {
      case 'critical':
        return 'text-red-400 bg-red-500/10';
      case 'high':
        return 'text-orange-400 bg-orange-500/10';
      case 'medium':
        return 'text-yellow-400 bg-yellow-500/10';
      default:
        return 'text-green-400 bg-green-500/10';
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Audit Log</h1>

      <div className="card">
        <div className="flex items-center gap-4 mb-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Component
            </label>
            <select
              value={componentFilter}
              onChange={(e) => setComponentFilter(e.target.value)}
              className="input-field text-sm"
            >
              {components.map((c) => (
                <option key={c} value={c}>
                  {c || 'All Components'}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Risk Level
            </label>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
              className="input-field text-sm"
            >
              {riskLevels.map((r) => (
                <option key={r} value={r}>
                  {r ? r.charAt(0).toUpperCase() + r.slice(1) : 'All Levels'}
                </option>
              ))}
            </select>
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : !logs || logs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No audit log entries found
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  {['', 'Timestamp', 'Component', 'Event', 'Action', 'Risk', 'User'].map(
                    (h) => (
                      <th
                        key={h}
                        className="text-left text-xs text-gray-500 font-medium uppercase tracking-wider py-2 px-3"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {logs.map((entry) => (
                  <>
                    <tr
                      key={entry.id}
                      className="border-b border-gray-700/50 hover:bg-gray-800/50 cursor-pointer"
                      onClick={() =>
                        setExpandedId(
                          expandedId === entry.id ? null : entry.id,
                        )
                      }
                    >
                      <td className="py-2 px-3">
                        {expandedId === entry.id ? (
                          <ChevronDown className="w-4 h-4 text-gray-500" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-500" />
                        )}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-400 font-mono">
                        {format(
                          new Date(entry.timestamp),
                          'MM/dd HH:mm:ss.SSS',
                        )}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-300 font-mono">
                        {entry.component}
                      </td>
                      <td className="py-2 px-3 text-sm text-gray-300">
                        {entry.event_type}
                      </td>
                      <td className="py-2 px-3 text-sm text-gray-300">
                        {entry.action}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={clsx(
                            'text-xs font-medium px-2 py-0.5 rounded capitalize',
                            riskColor(entry.risk_level),
                          )}
                        >
                          {entry.risk_level}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-500">
                        {entry.user ?? 'system'}
                      </td>
                    </tr>
                    {expandedId === entry.id && (
                      <tr key={`${entry.id}-details`}>
                        <td colSpan={7} className="px-3 py-3 bg-gray-900">
                          <div className="text-xs font-mono text-gray-400 whitespace-pre-wrap bg-gray-950 rounded-lg p-3 border border-gray-800">
                            {JSON.stringify(entry.details, null, 2)}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
