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

/**
 * Scale-aware axis label: sub-$10k values render as plain dollars ("$100",
 * "$9,999") -- a flat /1000 divisor showed a $100 paper account as "$0k" on
 * every tick. At $10k+ switch to thousands with one significant decimal
 * ("$10.2k"), dropping a trailing ".0" ("$250k").
 */
export function formatEquityTick(v: number): string {
  if (Math.abs(v) < 10_000) {
    return `$${Math.round(v).toLocaleString('en-US')}`;
  }
  return `$${(v / 1000).toFixed(1).replace(/\.0$/, '')}k`;
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
              tickFormatter={formatEquityTick}
              domain={['dataMin - 0.5', 'dataMax + 0.5']}
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
