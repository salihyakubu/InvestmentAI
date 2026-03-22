import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Bar as RechartsBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { format } from 'date-fns';
import type { Bar } from '../../types';

interface PriceChartProps {
  data: Bar[];
  symbol: string;
}

export default function PriceChart({ data, symbol }: PriceChartProps) {
  const chartData = data.map((bar) => ({
    ...bar,
    time: format(new Date(bar.timestamp), 'MM/dd HH:mm'),
    color: bar.close >= bar.open ? '#22c55e' : '#ef4444',
  }));

  if (chartData.length === 0) {
    return (
      <div className="card h-80 flex items-center justify-center text-gray-500">
        No price data available for {symbol}
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Price Chart - {symbol}</div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="75%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="time"
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
              domain={['auto', 'auto']}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              labelStyle={{ color: '#9ca3af' }}
              formatter={(value: number) => [`$${value.toFixed(2)}`]}
            />
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="close"
              stroke="#3b82f6"
              fill="url(#priceGradient)"
              strokeWidth={2}
              name="Close"
            />
          </ComposedChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height="25%">
          <ComposedChart data={chartData}>
            <XAxis dataKey="time" hide />
            <YAxis stroke="#6b7280" fontSize={10} tickLine={false} hide />
            <RechartsBar
              dataKey="volume"
              fill="#4b5563"
              opacity={0.6}
              name="Volume"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
