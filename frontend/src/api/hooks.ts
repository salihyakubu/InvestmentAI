import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiClient from './client';
import {
  normalizeEquityPoint,
  normalizeModelInfo,
  normalizeOrder,
  normalizePortfolioSummary,
  normalizePosition,
  normalizeBacktestResult,
  normalizePrediction,
  normalizeRiskMetrics,
} from './normalize';
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
    queryFn: () =>
      apiClient
        .get('/portfolio/summary')
        .then((r) => normalizePortfolioSummary(r.data)),
    refetchInterval: 5_000,
  });
}

export function usePositions() {
  return useQuery<Position[]>({
    queryKey: ['positions'],
    queryFn: () =>
      apiClient
        .get('/positions')
        .then((r) => (Array.isArray(r.data) ? r.data : []).map(normalizePosition)),
    refetchInterval: 5_000,
  });
}

/**
 * Equity history. The window is expressed in days: snapshots land every five
 * minutes, so the old row-capped default covered barely eight hours and hid
 * the whole soak behind a single date label. Sorted defensively -- a
 * descending payload would render the curve mirrored in time.
 */
export function useEquityCurve(days = 30) {
  return useQuery<{ date: string; equity: number }[]>({
    queryKey: ['portfolio', 'snapshots', days],
    queryFn: () =>
      apiClient
        .get('/portfolio/snapshots', { params: { days } })
        .then((r) =>
          (Array.isArray(r.data) ? r.data : [])
            .map(normalizeEquityPoint)
            .sort((a, b) => +new Date(a.date) - +new Date(b.date)),
        ),
    refetchInterval: 30_000,
  });
}

// Do-nothing benchmark: the live bar every strategy must beat.
export function useBenchmark(days = 30) {
  return useQuery<{ time: string; benchmark_equity: number }[]>({
    queryKey: ['portfolio', 'benchmark', days],
    queryFn: () =>
      apiClient
        .get('/portfolio/benchmark', { params: { days } })
        .then((r) => (Array.isArray(r.data) ? r.data : [])),
    refetchInterval: 300_000,
  });
}

// Research: walk-forward adjudication record (read-only)
export function useFundingWatch() {
  return useQuery<import('../components/ml/ResearchWatch').FundingWatch>({
    queryKey: ['research', 'funding-watch'],
    queryFn: () => apiClient.get('/research/funding-watch').then((r) => r.data),
    refetchInterval: 300_000, // ICs resolve every 8h; 5min is plenty
  });
}

// Orders
export function useOrders(status?: string) {
  return useQuery<Order[]>({
    queryKey: ['orders', status],
    queryFn: () =>
      apiClient
        .get('/orders', { params: status ? { status } : undefined })
        // The API returns a paginated envelope {items, total, page, page_size}.
        .then((r) => (r.data?.items ?? []).map(normalizeOrder)),
    refetchInterval: 3_000,
  });
}

export function useSubmitOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (order: {
      symbol: string;
      side: 'buy' | 'sell';
      order_type: string;
      quantity: number;
      limit_price?: number;
      stop_price?: number;
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
    queryFn: () =>
      apiClient
        .get('/risk/metrics/latest')
        .then((r) => normalizeRiskMetrics(r.data)),
    refetchInterval: 10_000,
  });
}

// Predictions
export function usePredictions(symbol?: string) {
  return useQuery<Prediction[]>({
    queryKey: ['predictions', symbol],
    queryFn: () =>
      apiClient
        .get('/predictions/latest', { params: symbol ? { symbol } : undefined })
        .then((r) => (r.data as unknown[]).map((p) => normalizePrediction(p as Record<string, unknown>))),
    refetchInterval: 15_000,
  });
}

// Models
export function useModels() {
  return useQuery<ModelInfo[]>({
    queryKey: ['models'],
    queryFn: () =>
      apiClient
        .get('/models')
        .then((r) => (Array.isArray(r.data) ? r.data : []).map(normalizeModelInfo)),
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

// Backtesting -- job-based: run returns an id, the UI polls status and
// fetches the normalized result once completed (visible failure states).
export interface BacktestJobStatus {
  id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  error: string | null;
}

export function useRunBacktest() {
  return useMutation<BacktestJobStatus, Error, BacktestConfig>({
    mutationFn: (config) =>
      apiClient.post('/backtest/run', config).then((r) => r.data),
  });
}

export function useBacktestStatus(jobId: string | null) {
  return useQuery<BacktestJobStatus>({
    queryKey: ['backtest', 'status', jobId],
    queryFn: () =>
      apiClient.get(`/backtest/status/${jobId}`).then((r) => r.data),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === 'queued' || s === 'running' ? 2000 : false;
    },
  });
}

export function useBacktestResults(jobId: string | null, ready: boolean) {
  return useQuery<BacktestResult>({
    queryKey: ['backtest', 'results', jobId],
    queryFn: () =>
      apiClient
        .get(`/backtest/results/${jobId}`)
        .then((r) => normalizeBacktestResult(r.data as Record<string, unknown>)),
    enabled: jobId != null && ready,
    staleTime: Infinity,
  });
}

// Trading mode -- the BACKEND's real mode (set by the deployment's
// TRADING_MODE env var). The UI can only display it, never change it: a
// compromised browser session must not be able to flip a paper system live.
export function useTradingMode() {
  return useQuery<'paper' | 'live' | 'backtest'>({
    queryKey: ['config', 'trading-mode'],
    queryFn: () =>
      apiClient
        .get('/config/trading-mode')
        .then((r) => r.data?.trading_mode ?? 'paper'),
    refetchInterval: 30_000,
  });
}

// Settings (backed by the config router on the API)
export function useSettings() {
  return useQuery<Record<string, unknown>>({
    queryKey: ['settings'],
    queryFn: () => apiClient.get('/config/risk-rules').then((r) => r.data),
  });
}

export function useUpdateSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (settings: Record<string, unknown>) =>
      apiClient.put('/config/risk-rules', settings).then((r) => r.data),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });
}
