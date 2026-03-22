import OrderForm from '../components/orders/OrderForm';
import OrderBook from '../components/orders/OrderBook';
import OrderHistory from '../components/orders/OrderHistory';
import PriceChart from '../components/charts/PriceChart';
import { useOrders, useCancelOrder, useMarketData } from '../api/hooks';
import { useAppStore } from '../store';

export default function Trading() {
  const { selectedSymbol, setSelectedSymbol } = useAppStore();
  const { data: allOrders } = useOrders();
  const { data: marketData } = useMarketData(selectedSymbol);
  const cancelOrder = useCancelOrder();

  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Trading</h1>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Symbol:</span>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="input-field text-sm font-mono"
          >
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          {marketData && (
            <PriceChart data={marketData} symbol={selectedSymbol} />
          )}
        </div>
        <div>
          <OrderForm />
        </div>
      </div>

      {allOrders && (
        <OrderBook
          orders={allOrders}
          onCancel={(id) => cancelOrder.mutate(id)}
        />
      )}

      {allOrders && <OrderHistory orders={allOrders} />}
    </div>
  );
}
