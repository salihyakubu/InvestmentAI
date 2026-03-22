import PortfolioSummaryComponent from '../components/portfolio/PortfolioSummary';
import PositionsTable from '../components/portfolio/PositionsTable';
import EquityCurve from '../components/charts/EquityCurve';
import PredictionView from '../components/ml/PredictionView';
import CircuitBreakerStatus from '../components/risk/CircuitBreakerStatus';
import {
  usePortfolioSummary,
  usePositions,
  useEquityCurve,
  usePredictions,
  useRiskMetrics,
} from '../api/hooks';

export default function Dashboard() {
  const { data: portfolio, isLoading: portfolioLoading } = usePortfolioSummary();
  const { data: positions } = usePositions();
  const { data: equityData } = useEquityCurve();
  const { data: predictions } = usePredictions();
  const { data: riskMetrics } = useRiskMetrics();

  if (portfolioLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading dashboard...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <div className="text-sm text-gray-500">
          {new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </div>
      </div>

      {portfolio && <PortfolioSummaryComponent data={portfolio} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {equityData && <EquityCurve data={equityData} />}
        </div>
        <div>
          {riskMetrics && (
            <CircuitBreakerStatus
              status={riskMetrics.circuit_breaker_status}
              reason={riskMetrics.circuit_breaker_reason}
            />
          )}
        </div>
      </div>

      {positions && positions.length > 0 && (
        <PositionsTable positions={positions} />
      )}

      {predictions && predictions.length > 0 && (
        <PredictionView predictions={predictions.slice(0, 6)} />
      )}
    </div>
  );
}
