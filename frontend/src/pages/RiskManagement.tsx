import RiskDashboard from '../components/risk/RiskDashboard';
import DrawdownChart from '../components/charts/DrawdownChart';
import CorrelationHeatmap from '../components/charts/CorrelationHeatmap';
import { useRiskMetrics, useEquityCurve } from '../api/hooks';

export default function RiskManagement() {
  const { data: metrics, isLoading } = useRiskMetrics();
  const { data: equityData } = useEquityCurve();

  // Derive drawdown data from equity curve
  const drawdownData =
    equityData?.map((point, i) => {
      const peak = Math.max(
        ...equityData.slice(0, i + 1).map((p) => p.equity),
      );
      const drawdown = peak > 0 ? (point.equity - peak) / peak : 0;
      return { date: point.date, drawdown };
    }) ?? [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading risk metrics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Risk Management</h1>

      {metrics && <RiskDashboard metrics={metrics} />}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DrawdownChart data={drawdownData} />
        {metrics && (
          <CorrelationHeatmap data={metrics.correlation_matrix} />
        )}
      </div>
    </div>
  );
}
