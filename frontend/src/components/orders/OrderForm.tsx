import { useState } from 'react';
import clsx from 'clsx';
import { useSubmitOrder } from '../../api/hooks';

export default function OrderForm() {
  const [symbol, setSymbol] = useState('');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState('market');
  const [quantity, setQuantity] = useState('');
  const [limitPrice, setLimitPrice] = useState('');

  const submitOrder = useSubmitOrder();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol || !quantity) return;

    submitOrder.mutate(
      {
        symbol: symbol.toUpperCase(),
        side,
        type: orderType,
        quantity: parseFloat(quantity),
        ...(orderType === 'limit' && limitPrice
          ? { limit_price: parseFloat(limitPrice) }
          : {}),
      },
      {
        onSuccess: () => {
          setSymbol('');
          setQuantity('');
          setLimitPrice('');
        },
      },
    );
  };

  return (
    <div className="card">
      <div className="card-header">Place Order</div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs text-gray-500 mb-1 uppercase">
            Symbol
          </label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="AAPL"
            className="input-field w-full font-mono"
          />
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1 uppercase">
            Side
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setSide('buy')}
              className={clsx(
                'py-2 rounded-lg font-bold text-sm transition-colors',
                side === 'buy'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
              )}
            >
              BUY
            </button>
            <button
              type="button"
              onClick={() => setSide('sell')}
              className={clsx(
                'py-2 rounded-lg font-bold text-sm transition-colors',
                side === 'sell'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
              )}
            >
              SELL
            </button>
          </div>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1 uppercase">
            Order Type
          </label>
          <select
            value={orderType}
            onChange={(e) => setOrderType(e.target.value)}
            className="input-field w-full"
          >
            <option value="market">Market</option>
            <option value="limit">Limit</option>
            <option value="stop">Stop</option>
            <option value="stop_limit">Stop Limit</option>
          </select>
        </div>

        <div>
          <label className="block text-xs text-gray-500 mb-1 uppercase">
            Quantity
          </label>
          <input
            type="number"
            value={quantity}
            onChange={(e) => setQuantity(e.target.value)}
            placeholder="100"
            min="0"
            step="1"
            className="input-field w-full font-mono"
          />
        </div>

        {(orderType === 'limit' || orderType === 'stop_limit') && (
          <div>
            <label className="block text-xs text-gray-500 mb-1 uppercase">
              Limit Price
            </label>
            <input
              type="number"
              value={limitPrice}
              onChange={(e) => setLimitPrice(e.target.value)}
              placeholder="0.00"
              min="0"
              step="0.01"
              className="input-field w-full font-mono"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={!symbol || !quantity || submitOrder.isPending}
          className={clsx(
            'w-full py-3 rounded-lg font-bold text-sm transition-colors disabled:opacity-50',
            side === 'buy'
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-red-600 hover:bg-red-700 text-white',
          )}
        >
          {submitOrder.isPending
            ? 'Submitting...'
            : `${side.toUpperCase()} ${symbol || 'SYMBOL'}`}
        </button>

        {submitOrder.isError && (
          <div className="text-xs text-red-400 bg-red-500/10 rounded-lg p-2">
            Order failed. Please try again.
          </div>
        )}
      </form>
    </div>
  );
}
