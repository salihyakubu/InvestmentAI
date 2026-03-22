import { create } from 'zustand';
import type { Alert } from '../types';

interface AppState {
  // Live prices
  prices: Record<string, number>;
  updatePrice: (symbol: string, price: number) => void;

  // Alerts
  alerts: Alert[];
  addAlert: (alert: Omit<Alert, 'id' | 'timestamp' | 'read'>) => void;
  markAlertRead: (id: string) => void;
  clearAlerts: () => void;

  // Trading mode
  tradingMode: 'paper' | 'live';
  setTradingMode: (mode: 'paper' | 'live') => void;

  // Connection status
  connected: boolean;
  setConnected: (status: boolean) => void;

  // Selected symbol
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  prices: {},
  updatePrice: (symbol, price) =>
    set((state) => ({
      prices: { ...state.prices, [symbol]: price },
    })),

  alerts: [],
  addAlert: (alert) =>
    set((state) => ({
      alerts: [
        {
          ...alert,
          id: crypto.randomUUID(),
          timestamp: new Date().toISOString(),
          read: false,
        },
        ...state.alerts,
      ].slice(0, 100),
    })),
  markAlertRead: (id) =>
    set((state) => ({
      alerts: state.alerts.map((a) =>
        a.id === id ? { ...a, read: true } : a,
      ),
    })),
  clearAlerts: () => set({ alerts: [] }),

  tradingMode: 'paper',
  setTradingMode: (mode) => set({ tradingMode: mode }),

  connected: false,
  setConnected: (status) => set({ connected: status }),

  selectedSymbol: 'AAPL',
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
}));
