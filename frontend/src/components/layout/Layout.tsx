import { useEffect, type ReactNode } from 'react';
import Sidebar from './Sidebar';
import Header from './Header';
import { wsManager } from '../../api/websocket';
import { num } from '../../api/normalize';
import { useAppStore } from '../../store';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const setConnected = useAppStore((s) => s.setConnected);
  const updatePrice = useAppStore((s) => s.updatePrice);

  // Live-updates socket: the header chip reflects its state, and price ticks
  // flow into the store. Auto-reconnect lives inside wsManager.
  useEffect(() => {
    const offStatus = wsManager.onMessage('_connection', (data) => {
      const status = (data as { status?: string }).status;
      setConnected(status === 'connected');
    });
    const offPrices = wsManager.onMessage('prices', (data) => {
      const msg = data as { symbol?: string; price?: unknown };
      if (msg.symbol) updatePrice(msg.symbol, num(msg.price));
    });
    wsManager.connect('prices');

    return () => {
      offStatus();
      offPrices();
      wsManager.disconnect();
      setConnected(false);
    };
  }, [setConnected, updatePrice]);

  return (
    <div className="min-h-screen bg-gray-900">
      <Sidebar />
      <Header />
      <main className="ml-60 mt-16 p-6">{children}</main>
    </div>
  );
}
