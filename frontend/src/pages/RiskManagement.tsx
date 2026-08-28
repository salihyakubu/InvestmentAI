import RiskDashboard from '../components/risk/RiskDashboard';
import DrawdownChart from '../components/charts/DrawdownChart';
import CorrelationHeatmap from '../components/charts/CorrelationHeatmap';
import { useRiskMetrics, useEquityCurve } from '../api/hooks';

export default function RiskManagement() {
  const { data: metrics, isLoading } = useRiskMetrics();
  const { data: equityData } = useEquityCurve();

  // Drawdown against the running peak, in one forward pass. The series
  // arrives chronologically, so the peak is genuinely the prior maximum --
  // the previous slice-and-Math.max was both O(n^2) and, on the older
  // newest-first payload, measured against future equity.
  let peak = -Infinity;
  const drawdownData = (equityData ?? []).map((point) => {
    peak = Math.max(peak, point.equity);
    return {
      date: point.date,
      drawdown: peak > 0 ? (point.equity - peak) / peak : 0,
    };
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading risk metrics...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="page-title">Risk Management</h1>

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
