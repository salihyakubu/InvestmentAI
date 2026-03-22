import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

interface DrawdownChartProps {
  data: { date: string; drawdown: number }[];
}

export default function DrawdownChart({ data }: DrawdownChartProps) {
  const chartData = data.map((d) => ({
    ...d,
    dateLabel: format(new Date(d.date), 'MMM dd'),
    drawdownPct: d.drawdown * 100,
  }));

  if (chartData.length === 0) {
    return (
      <div className="card h-64 flex items-center justify-center text-gray-500">
        No drawdown data available
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Drawdown</div>
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
              tickFormatter={(v: number) => `${v.toFixed(1)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              formatter={(value: number) => [
                `${value.toFixed(2)}%`,
                'Drawdown',
              ]}
            />
            <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="3 3" />
            <defs>
              <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.4} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="drawdownPct"
              stroke="#ef4444"
              fill="url(#ddGrad)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
