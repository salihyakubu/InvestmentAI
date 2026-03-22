import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from './client';
import type {
  PortfolioSummary,
  Position,
  Order,
  RiskMetrics,
  Prediction,
  ModelInfo,
  Bar,
  AuditLogEntry,
  BacktestConfig,
  BacktestResult,
} from '../types';

// Portfolio
export function usePortfolioSummary() {
  return useQuery<PortfolioSummary>({
    queryKey: ['portfolio', 'summary'],
    queryFn: () => apiClient.get('/portfolio/summary').then((r) => r.data),
    refetchInterval: 5_000,
  });
}

export function usePositions() {
  return useQuery<Position[]>({
    queryKey: ['portfolio', 'positions'],
    queryFn: () => apiClient.get('/portfolio/positions').then((r) => r.data),
    refetchInterval: 5_000,
  });
}

export function useEquityCurve() {
  return useQuery<{ date: string; equity: number }[]>({
    queryKey: ['portfolio', 'equity-curve'],
    queryFn: () => apiClient.get('/portfolio/equity-curve').then((r) => r.data),
    refetchInterval: 30_000,
  });
}

// Orders
export function useOrders(status?: string) {
  return useQuery<Order[]>({
    queryKey: ['orders', status],
    queryFn: () =>
      apiClient
        .get('/orders', { params: status ? { status } : undefined })
        .then((r) => r.data),
    refetchInterval: 3_000,
  });
}

export function useSubmitOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (order: {
      symbol: string;
      side: 'buy' | 'sell';
      type: string;
      quantity: number;
      limit_price?: number;
    }) => apiClient.post('/orders', order).then((r) => r.data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['orders'] });
      void queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });
}

export function useCancelOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (orderId: string) =>
      apiClient.delete(`/orders/${orderId}`).then((r) => r.data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}

// Risk
export function useRiskMetrics() {
  return useQuery<RiskMetrics>({
    queryKey: ['risk', 'metrics'],
    queryFn: () => apiClient.get('/risk/metrics').then((r) => r.data),
    refetchInterval: 10_000,
  });
}

// Predictions
export function usePredictions(symbol?: string) {
  return useQuery<Prediction[]>({
    queryKey: ['predictions', symbol],
    queryFn: () =>
      apiClient
        .get('/predictions', { params: symbol ? { symbol } : undefined })
        .then((r) => r.data),
    refetchInterval: 15_000,
  });
}

// Models
export function useModels() {
  return useQuery<ModelInfo[]>({
    queryKey: ['models'],
    queryFn: () => apiClient.get('/models').then((r) => r.data),
  });
}

export function useRetrainModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) =>
      apiClient.post(`/models/${modelId}/retrain`).then((r) => r.data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['models'] });
    },
  });
}

// Market Data
export function useMarketData(symbol: string, timeframe = '1D', limit = 100) {
  return useQuery<Bar[]>({
    queryKey: ['market-data', symbol, timeframe, limit],
    queryFn: () =>
      apiClient
        .get(`/market-data/${symbol}/bars`, { params: { timeframe, limit } })
        .then((r) => r.data),
    enabled: !!symbol,
    refetchInterval: 60_000,
  });
}

// Audit Logs
export function useAuditLogs(params?: {
  component?: string;
  risk_level?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery<AuditLogEntry[]>({
    queryKey: ['audit-logs', params],
    queryFn: () =>
      apiClient.get('/audit/logs', { params }).then((r) => r.data),
  });
}

// Backtesting
export function useRunBacktest() {
  return useMutation<BacktestResult, Error, BacktestConfig>({
    mutationFn: (config) =>
      apiClient.post('/backtesting/run', config).then((r) => r.data),
  });
}

// Settings
export function useSettings() {
  return useQuery<Record<string, unknown>>({
    queryKey: ['settings'],
    queryFn: () => apiClient.get('/settings').then((r) => r.data),
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (settings: Record<string, unknown>) =>
      apiClient.put('/settings', settings).then((r) => r.data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });
}
