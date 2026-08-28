/**
 * DEV-ONLY visual harness: every infographic component rendered with
 * representative mock data, INCLUDING the honest edge states (unavailable
 * metrics, in-progress quarters, non-computable drift). Routed only when
 * import.meta.env.DEV -- never part of a production bundle, never fed by
 * real endpoints, and the mock numbers are deliberately implausible-looking
 * period pieces, not fabricated performance claims.
 */
import PortfolioSummary from '../components/portfolio/PortfolioSummary';
import ResearchWatch from '../components/ml/ResearchWatch';
import LearningMetrics from '../components/ml/LearningMetrics';
import FeatureDrift from '../components/ml/FeatureDrift';
import EquityCurve from '../components/charts/EquityCurve';
import type { PortfolioSummary as PortfolioSummaryType } from '../types';

const equity = Array.from({ length: 60 }, (_, i) => ({
  date: new Date(Date.UTC(2026, 7, 1, i * 4)).toISOString(),
  equity: 100 + Math.sin(i / 6) * 1.4 + i * 0.02,
}));
const benchmark = equity.map((d, i) => ({
  time: d.date,
  benchmark_equity: 100 + i * 0.045,
}));

const portfolio = {
  total_equity: 99.99,
  daily_pnl: -0.42,
  daily_pnl_pct: -0.42,
  total_return: -0.01,
  total_return_pct: -0.01,
  sharpe_ratio: undefined,
  win_rate: 0.48,
  closed_trades: 210,
  max_drawdown: 0.031,
} as unknown as PortfolioSummaryType;

const watch = {
  horizon_hours: 24,
  required_consecutive_positive_quarters: 4,
  verdict:
    'watching -- no trading proposal may cite any factor until its registered bar is met on complete quarters of unseen data',
  factors: [
    {
      factor: 'funding_carry_24h_plus',
      hypothesis: 'Since mid-2025, perpetuals with HIGH funding outperform',
      observations_recorded: 172,
      current_positive_streak: 1,
      quarters: [
        { quarter: '2026-Q3', n: 95, mean_ic: 0.0124, t_stat: 1.52, positive: true },
      ],
    },
    {
      factor: 'reversal_8h_minus',
      hypothesis: 'Fading the last 8h move predicts relative return',
      observations_recorded: 140,
      current_positive_streak: 0,
      quarters: [
        { quarter: '2026-Q3', n: 88, mean_ic: -0.0031, t_stat: -0.4, positive: false },
      ],
    },
    {
      factor: 'near_high_fade_minus',
      hypothesis: 'Contracts at their own 7-day high underperform (external claim)',
      observations_recorded: 0,
      current_positive_streak: 0,
      quarters: [],
    },
  ],
};

const learning = {
  feature_rows_persisted: 24_812,
  observations_used: 61_400,
  predictions_last_7d: 46_212,
  notes: ['PR #59 zero-return guard applies'],
  eras: [
    {
      label: 'era 3 (v3)', start: '2026-07-24', end: '2026-07-28', days: 5,
      n: 4100, mean_ic: 0.0034, ic_t_stat: 0.26, mean_brier: 0.2712,
      sign_agreement: 0.502,
    },
    {
      label: 'era 4 (v4)', start: '2026-07-28', end: null, days: 12,
      n: 12100, mean_ic: -0.0255, ic_t_stat: -5.07, mean_brier: 0.2415,
      sign_agreement: null,
    },
  ],
  daily: [],
  calibration: [
    { bin_low: 0.0, bin_high: 0.4, n: 2131, mean_forecast: 0.358, realized_flat_rate: 0.425, gap: 0.066 },
    { bin_low: 0.4, bin_high: 0.5, n: 7389, mean_forecast: 0.456, realized_flat_rate: 0.445, gap: -0.011 },
    { bin_low: 0.5, bin_high: 0.6, n: 10868, mean_forecast: 0.55, realized_flat_rate: 0.348, gap: -0.202 },
    { bin_low: 0.6, bin_high: 1.0, n: 3815, mean_forecast: 0.629, realized_flat_rate: 0.268, gap: -0.361 },
  ],
};

const drift = {
  computable: true,
  reason: null,
  generation_hash: 'b9104ca584379a86',
  n_reference: 12_431,
  n_recent: 3_901,
  n_symbols_measured: 5,
  reference_start: '2026-08-09', reference_end: '2026-08-16', recent_start: '2026-08-26',
  n_features: 52,
  n_unmeasurable: 6,
  share_significant: 0.76,
  top_drifted: [
    { index: 49, psi: 8.267, psi_max: 8.29, n_symbols: 5 },
    { index: 12, psi: 0.61, psi_max: 1.4, n_symbols: 5 },
    { index: 7, psi: 0.18, psi_max: 0.24, n_symbols: 5 },
    { index: 31, psi: 0.06, psi_max: 0.09, n_symbols: 4 },
  ],
  worst: { symbol: 'ETH/USDT', index: 49, psi: 8.2897 },
  thresholds: { moderate: 0.1, significant: 0.25 },
  notes: ['Reference is the platform’s own early serving history, not the training distribution'],
};

const driftRefusal = {
  ...drift,
  computable: false,
  reason: 'insufficient history: 0.6 days persisted for the current generation; needs >= 5',
  top_drifted: [],
  share_significant: null,
  worst: null,
};

export default function DevPreview() {
  // #hero | #watch | #learning -- render one section above the fold so the
  // pane can screenshot it without scrolling; no hash renders everything.
  const section = window.location.hash.replace('#', '');
  return (
    <div className="min-h-screen bg-ink p-8 space-y-6 max-w-[1100px] mx-auto">
      <h1 className="page-title">Dev Preview — infographic states</h1>
      {(!section || section === 'hero') && (
        <>
          <PortfolioSummary
            data={portfolio}
            equitySeries={equity.map((d) => d.equity)}
          />
          <EquityCurve data={equity} benchmark={benchmark} />
        </>
      )}
      {(!section || section === 'watch') && <ResearchWatch data={watch} />}
      {(!section || section === 'learning') && (
        <>
          <LearningMetrics data={learning} />
          <FeatureDrift data={drift} />
          <FeatureDrift data={driftRefusal} />
        </>
      )}
    </div>
  );
}
