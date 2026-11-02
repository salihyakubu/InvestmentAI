import { useState } from 'react';
import { format } from 'date-fns';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useAuditLogs } from '../api/hooks';

// Services and actions the backend actually writes (services/monitoring/audit.py
// callers). The old page filtered on component/risk_level, fields the API never
// had, so its filters were silently ignored and three columns rendered blank.
const SERVICES = ['', 'api', 'execution', 'risk', 'trading_pipeline'];
const ACTIONS = [
  '',
  'order.created',
  'order.filled',
  'order.rejected',
  'order.manual_create',
  'order.manual_cancel',
  'control.halt',
  'control.resume',
  'control.flatten',
  'circuit_breaker.open',
  'circuit_breaker.half_open',
  'circuit_breaker.closed',
  'trade.decision',
];

// Severity is DERIVED from the action for colouring only; the API stores none.
const actionTone = (action: string) => {
  if (action.startsWith('control.') || action === 'circuit_breaker.open') {
    return 'text-red-400 bg-red-500/10';
  }
  if (action === 'order.rejected' || action.startsWith('circuit_breaker.')) {
    return 'text-yellow-400 bg-yellow-500/10';
  }
  return 'text-gray-300 bg-gray-500/10';
};

export default function AuditLog() {
  const [serviceFilter, setServiceFilter] = useState('');
  const [actionFilter, setActionFilter] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const { data: logs, isLoading, isError } = useAuditLogs({
    service: serviceFilter || undefined,
    action: actionFilter || undefined,
    limit: 200,
  });

  return (
    <div className="space-y-6">
      <h1 className="page-title">Audit Log</h1>

      <div className="card">
        <div className="flex items-center gap-4 mb-4">
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Service
            </label>
            <select
              value={serviceFilter}
              onChange={(e) => setServiceFilter(e.target.value)}
              className="input-field text-sm"
            >
              {SERVICES.map((s) => (
                <option key={s} value={s}>
                  {s || 'All services'}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Action
            </label>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value)}
              className="input-field text-sm"
            >
              {ACTIONS.map((a) => (
                <option key={a} value={a}>
                  {a || 'All actions'}
                </option>
              ))}
            </select>
          </div>
          <div className="ml-auto text-xs text-gray-500">
            Newest 200 rows. Times are UTC.
          </div>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : isError ? (
          <div className="text-center py-8 text-red-400">
            Audit log unavailable (request failed)
          </div>
        ) : !logs || logs.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            No audit log entries found
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  {['', 'Timestamp (UTC)', 'Service', 'Action', 'Entity', 'Actor'].map(
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
                {logs.map((entry) => {
                  const expanded = expandedId === entry.id;
                  const entity = entry.entity_id
                    ? `${entry.entity_type ?? ''} ${entry.entity_id}`.trim()
                    : (entry.entity_type ?? '—');
                  return [
                    <tr
                      key={entry.id}
                      className="border-b border-gray-700/50 hover:bg-gray-800/50 cursor-pointer"
                      onClick={() => setExpandedId(expanded ? null : entry.id)}
                    >
                      <td className="py-2 px-3">
                        {expanded ? (
                          <ChevronDown className="w-4 h-4 text-gray-500" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-500" />
                        )}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-400 font-mono">
                        {format(new Date(entry.timestamp), 'MM/dd HH:mm:ss.SSS')}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-300 font-mono">
                        {entry.service}
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded font-mono ${actionTone(entry.action)}`}
                        >
                          {entry.action}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-400 font-mono truncate max-w-xs">
                        {entity}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-500 font-mono">
                        {entry.user_id ?? 'system'}
                      </td>
                    </tr>,
                    expanded ? (
                      <tr key={`${entry.id}-details`}>
                        <td colSpan={6} className="px-3 py-3 bg-gray-900">
                          <div className="text-xs font-mono text-gray-400 whitespace-pre-wrap bg-gray-950 rounded-lg p-3 border border-gray-800">
                            {JSON.stringify(entry.details, null, 2)}
                          </div>
                        </td>
                      </tr>
                    ) : null,
                  ];
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
