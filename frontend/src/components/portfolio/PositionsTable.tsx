import { useState } from 'react';
import { ArrowUpDown, X } from 'lucide-react';
import clsx from 'clsx';
import type { Position } from '../../types';

interface PositionsTableProps {
  positions: Position[];
  onClose?: (positionId: string) => void;
}

type SortField = keyof Position;

export default function PositionsTable({
  positions,
  onClose,
}: PositionsTableProps) {
  const [sortField, setSortField] = useState<SortField>('unrealized_pnl');
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sorted = [...positions].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortAsc ? aVal - bVal : bVal - aVal;
    }
    return sortAsc
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
  });

  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(v);

  const columns: { key: SortField; label: string }[] = [
    { key: 'symbol', label: 'Symbol' },
    { key: 'side', label: 'Side' },
    { key: 'quantity', label: 'Qty' },
    { key: 'entry_price', label: 'Entry' },
    { key: 'current_price', label: 'Current' },
    { key: 'unrealized_pnl', label: 'P&L' },
    { key: 'unrealized_pnl_pct', label: 'P&L %' },
  ];

  if (positions.length === 0) {
    return (
      <div className="card">
        <div className="card-header">Positions</div>
        <div className="text-center py-8 text-gray-500">No open positions</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Positions ({positions.length})</div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className="text-left text-xs text-gray-500 font-medium uppercase tracking-wider py-2 px-3 cursor-pointer hover:text-gray-300"
                  onClick={() => handleSort(col.key)}
                >
                  <div className="flex items-center gap-1">
                    {col.label}
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
              ))}
              {onClose && (
                <th className="text-right text-xs text-gray-500 font-medium uppercase tracking-wider py-2 px-3">
                  Action
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {sorted.map((pos) => (
              <tr
                key={pos.id}
                className="border-b border-gray-700/50 hover:bg-gray-800/50"
              >
                <td className="py-2.5 px-3 font-mono font-semibold text-white text-sm">
                  {pos.symbol}
                </td>
                <td className="py-2.5 px-3">
                  <span
                    className={clsx(
                      'text-xs font-bold uppercase px-2 py-0.5 rounded',
                      pos.side === 'long'
                        ? 'text-green-400 bg-green-500/10'
                        : 'text-red-400 bg-red-500/10',
                    )}
                  >
                    {pos.side}
                  </span>
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {pos.quantity}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {formatCurrency(pos.entry_price)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {formatCurrency(pos.current_price)}
                </td>
                <td
                  className={clsx(
                    'py-2.5 px-3 font-mono text-sm font-semibold',
                    pos.unrealized_pnl >= 0
                      ? 'text-green-500'
                      : 'text-red-500',
                  )}
                >
                  {pos.unrealized_pnl >= 0 ? '+' : ''}
                  {formatCurrency(pos.unrealized_pnl)}
                </td>
                <td
                  className={clsx(
                    'py-2.5 px-3 font-mono text-sm font-semibold',
                    pos.unrealized_pnl_pct >= 0
                      ? 'text-green-500'
                      : 'text-red-500',
                  )}
                >
                  {pos.unrealized_pnl_pct >= 0 ? '+' : ''}
                  {pos.unrealized_pnl_pct.toFixed(2)}%
                </td>
                {onClose && (
                  <td className="py-2.5 px-3 text-right">
                    <button
                      onClick={() => onClose(pos.id)}
                      className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                      title="Close position"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
