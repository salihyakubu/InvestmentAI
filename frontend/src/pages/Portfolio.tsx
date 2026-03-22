import PortfolioSummaryComponent from '../components/portfolio/PortfolioSummary';
import PositionsTable from '../components/portfolio/PositionsTable';
import AllocationPie from '../components/portfolio/AllocationPie';
import EquityCurve from '../components/charts/EquityCurve';
import {
  usePortfolioSummary,
  usePositions,
  useEquityCurve,
} from '../api/hooks';

export default function Portfolio() {
  const { data: portfolio, isLoading } = usePortfolioSummary();
  const { data: positions } = usePositions();
  const { data: equityData } = useEquityCurve();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading portfolio...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Portfolio</h1>

      {portfolio && <PortfolioSummaryComponent data={portfolio} />}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {equityData && <EquityCurve data={equityData} />}
        </div>
        <div>{positions && <AllocationPie positions={positions} />}</div>
      </div>

      {positions && <PositionsTable positions={positions} />}
    </div>
  );
}
