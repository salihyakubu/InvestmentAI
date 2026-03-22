import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { format } from 'date-fns';

interface EquityCurveProps {
  data: { date: string; equity: number }[];
}

export default function EquityCurve({ data }: EquityCurveProps) {
  const chartData = data.map((d) => ({
    ...d,
    dateLabel: format(new Date(d.date), 'MMM dd'),
  }));

  const startEquity = chartData[0]?.equity ?? 0;
  const endEquity = chartData[chartData.length - 1]?.equity ?? 0;
  const isPositive = endEquity >= startEquity;

  if (chartData.length === 0) {
    return (
      <div className="card h-64 flex items-center justify-center text-gray-500">
        No equity data available
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Equity Curve</div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="dateLabel"
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v: number) =>
                `$${(v / 1000).toFixed(0)}k`
              }
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              formatter={(value: number) => [
                `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
                'Equity',
              ]}
            />
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  stopColor={isPositive ? '#22c55e' : '#ef4444'}
                  stopOpacity={0.3}
                />
                <stop
                  offset="100%"
                  stopColor={isPositive ? '#22c55e' : '#ef4444'}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="equity"
              stroke={isPositive ? '#22c55e' : '#ef4444'}
              fill="url(#equityGrad)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
