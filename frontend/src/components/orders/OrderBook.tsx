import { X } from 'lucide-react';
import clsx from 'clsx';
import { format } from 'date-fns';
import type { Order } from '../../types';

interface OrderBookProps {
  orders: Order[];
  onCancel?: (orderId: string) => void;
}

export default function OrderBook({ orders, onCancel }: OrderBookProps) {
  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(v);

  const openOrders = orders.filter(
    (o) => o.status === 'pending' || o.status === 'submitted' || o.status === 'partial',
  );

  if (openOrders.length === 0) {
    return (
      <div className="card">
        <div className="card-header">Open Orders</div>
        <div className="text-center py-8 text-gray-500">No open orders</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Open Orders ({openOrders.length})</div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              {['Symbol', 'Side', 'Type', 'Qty', 'Price', 'Filled', 'Status', 'Time', ''].map(
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
            {openOrders.map((order) => (
              <tr
                key={order.id}
                className="border-b border-gray-700/50 hover:bg-gray-800/50"
              >
                <td className="py-2.5 px-3 font-mono font-semibold text-white text-sm">
                  {order.symbol}
                </td>
                <td className="py-2.5 px-3">
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
                <td className="py-2.5 px-3 text-xs text-gray-400 uppercase">
                  {order.type}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {order.quantity}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {order.limit_price ? formatCurrency(order.limit_price) : 'MKT'}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {order.filled_quantity}/{order.quantity}
                </td>
                <td className="py-2.5 px-3">
                  <span
                    className={clsx(
                      'text-xs font-medium px-2 py-0.5 rounded',
                      order.status === 'partial'
                        ? 'text-yellow-400 bg-yellow-500/10'
                        : 'text-blue-400 bg-blue-500/10',
                    )}
                  >
                    {order.status}
                  </span>
                </td>
                <td className="py-2.5 px-3 text-xs text-gray-500">
                  {format(new Date(order.created_at), 'HH:mm:ss')}
                </td>
                <td className="py-2.5 px-3 text-right">
                  {onCancel && (
                    <button
                      onClick={() => onCancel(order.id)}
                      className="p-1 text-gray-500 hover:text-red-400 transition-colors"
                      title="Cancel order"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
