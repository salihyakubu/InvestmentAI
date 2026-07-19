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

  // Connection status
  connected: boolean;
  setConnected: (status: boolean) => void;

  // Selected symbol
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;

  // Auth (token mirrors localStorage, which the axios client reads)
  token: string | null;
  setToken: (token: string) => void;
  logout: () => void;
}

const AUTH_TOKEN_KEY = 'auth_token';

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

  connected: false,
  setConnected: (status) => set({ connected: status }),

  selectedSymbol: 'AAPL',
  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),

  token: localStorage.getItem(AUTH_TOKEN_KEY),
  setToken: (token) => {
    localStorage.setItem(AUTH_TOKEN_KEY, token);
    set({ token });
  },
  logout: () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    set({ token: null });
  },
}));
