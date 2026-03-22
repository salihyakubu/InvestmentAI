import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from 'recharts';
import type { Position } from '../../types';

interface AllocationPieProps {
  positions: Position[];
}

const COLORS = [
  '#3b82f6',
  '#22c55e',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
  '#ec4899',
  '#f97316',
  '#14b8a6',
  '#6366f1',
];

export default function AllocationPie({ positions }: AllocationPieProps) {
  const totalValue = positions.reduce(
    (sum, p) => sum + Math.abs(p.market_value),
    0,
  );

  const data = positions.map((p) => ({
    name: p.symbol,
    value: Math.abs(p.market_value),
    pct: totalValue > 0 ? (Math.abs(p.market_value) / totalValue) * 100 : 0,
  }));

  if (data.length === 0) {
    return (
      <div className="card h-72 flex items-center justify-center text-gray-500">
        No positions to display
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Portfolio Allocation</div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={90}
              dataKey="value"
              nameKey="name"
              paddingAngle={2}
            >
              {data.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              formatter={(value: number, name: string) => [
                `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
                name,
              ]}
            />
            <Legend
              formatter={(value: string) => (
                <span className="text-xs text-gray-400">{value}</span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
