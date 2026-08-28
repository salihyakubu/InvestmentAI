import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { format } from 'date-fns';

interface EquityCurveProps {
  data: { date: string; equity: number }[];
  /**
   * The do-nothing benchmark (equal-weight buy-and-hold since inception),
   * sampled at the same snapshot times. Drawn as a dashed reference line so
   * the account's decisions are always judged against doing nothing.
   */
  benchmark?: { time: string; benchmark_equity: number }[];
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

/**
 * Tick precision has to follow the *visible* range, not the account size: a
 * $100 account moving $1 needs cents on the axis, or every tick rounds to the
 * same "$100" and the curve reads as a flat line. Built as a factory because
 * recharts calls `tickFormatter(value, index)` -- an optional second
 * parameter would silently receive the tick index.
 */
export function makeEquityTickFormatter(valueSpan: number) {
  return (v: number): string => {
    if (Math.abs(v) >= 10_000) {
      return `$${(v / 1000).toFixed(1).replace(/\.0$/, '')}k`;
    }
    const digits = valueSpan >= 100 ? 0 : valueSpan >= 10 ? 1 : 2;
    return `$${v.toLocaleString('en-US', {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })}`;
  };
}

export default function EquityCurve({ data, benchmark }: EquityCurveProps) {
  const benchmarkByTs = new Map(
    (benchmark ?? []).map((b) => [new Date(b.time).getTime(), b.benchmark_equity]),
  );
  const chartData = data.map((d) => {
    const ts = new Date(d.date).getTime();
    return { ...d, ts, benchmark: benchmarkByTs.get(ts) };
  });
  const hasBenchmark = chartData.some((d) => d.benchmark !== undefined);

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

  // Below the empty guard. Math.abs keeps the label choice correct even if a
  // caller hands us a descending series.
  const timeSpan = Math.abs(
    (chartData[chartData.length - 1]?.ts ?? 0) - (chartData[0]?.ts ?? 0),
  );
  const equities = chartData.map((d) => d.equity);
  const valueSpan = Math.max(...equities) - Math.min(...equities);
  const formatTick = makeEquityTickFormatter(valueSpan);
  const formatTime = (ts: number) =>
    format(
      new Date(ts),
      timeSpan < 36 * 3600e3
        ? 'HH:mm'
        : timeSpan < 14 * 864e5
          ? 'MMM d HH:mm'
          : 'MMM d',
    );

  return (
    <div className="card">
      <div className="card-header">Equity Curve</div>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={['dataMin', 'dataMax']}
              stroke="rgba(255,255,255,0.28)"
              fontSize={11}
              tickLine={false}
              tickFormatter={formatTime}
            />
            <YAxis
              stroke="rgba(255,255,255,0.28)"
              fontSize={11}
              tickLine={false}
              tickFormatter={formatTick}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              labelFormatter={(ts: number) =>
                format(new Date(ts), 'MMM d, HH:mm')
              }
              formatter={(value: number, name: string) => [
                `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
                name === 'benchmark' ? 'Do-nothing benchmark' : 'Equity',
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
            {hasBenchmark && (
              <Line
                type="monotone"
                dataKey="benchmark"
                stroke="#64748b"
                strokeDasharray="6 4"
                strokeWidth={1.5}
                dot={false}
                connectNulls
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
