import { useState } from 'react';
import clsx from 'clsx';
import { format } from 'date-fns';
import type { Order } from '../../types';

interface OrderHistoryProps {
  orders: Order[];
}

export default function OrderHistory({ orders }: OrderHistoryProps) {
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(v);

  const filtered =
    statusFilter === 'all'
      ? orders
      : orders.filter((o) => o.status === statusFilter);

  const statuses = ['all', 'filled', 'cancelled', 'rejected', 'pending', 'partial'];

  const statusColor = (status: string) => {
    switch (status) {
      case 'filled':
        return 'text-green-400 bg-green-500/10';
      case 'cancelled':
        return 'text-gray-400 bg-gray-500/10';
      case 'rejected':
        return 'text-red-400 bg-red-500/10';
      case 'partial':
        return 'text-yellow-400 bg-yellow-500/10';
      default:
        return 'text-blue-400 bg-blue-500/10';
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="card-header mb-0">Order History</div>
        <div className="flex gap-1">
          {statuses.map((s) => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={clsx(
                'px-2 py-1 rounded text-xs font-medium transition-colors capitalize',
                statusFilter === s
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700',
              )}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-8 text-gray-500">No orders found</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-700">
                {['Time', 'Symbol', 'Side', 'Type', 'Qty', 'Price', 'Avg Fill', 'Status'].map(
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
              {filtered.map((order) => (
                <tr
                  key={order.id}
                  className="border-b border-gray-700/50 hover:bg-gray-800/50"
                >
                  <td className="py-2 px-3 text-xs text-gray-500 font-mono">
                    {format(new Date(order.created_at), 'MM/dd HH:mm:ss')}
                  </td>
                  <td className="py-2 px-3 font-mono font-semibold text-white text-sm">
                    {order.symbol}
                  </td>
                  <td className="py-2 px-3">
                    <span
                      className={clsx(
                        'text-xs font-bold uppercase px-2 py-0.5 rounded',
                        order.side === 'buy'
                          ? 'text-green-400 bg-green-500/10'
                          : 'text-red-400 bg-red-500/10',
                      )}
                    >
                      {order.side}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-xs text-gray-400 uppercase">
                    {order.type}
                  </td>
                  <td className="py-2 px-3 font-mono text-sm text-gray-300">
                    {order.quantity}
                  </td>
                  <td className="py-2 px-3 font-mono text-sm text-gray-300">
                    {order.limit_price
                      ? formatCurrency(order.limit_price)
                      : 'MKT'}
                  </td>
                  <td className="py-2 px-3 font-mono text-sm text-gray-300">
                    {order.filled_avg_price
                      ? formatCurrency(order.filled_avg_price)
                      : '-'}
                  </td>
                  <td className="py-2 px-3">
                    <span
                      className={clsx(
                        'text-xs font-medium px-2 py-0.5 rounded',
                        statusColor(order.status),
                      )}
                    >
                      {order.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
