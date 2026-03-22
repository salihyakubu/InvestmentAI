import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';

interface FeatureImportanceProps {
  data: Record<string, number>;
  modelName?: string;
}

export default function FeatureImportance({
  data,
  modelName,
}: FeatureImportanceProps) {
  const chartData = Object.entries(data)
    .map(([feature, importance]) => ({
      feature,
      importance: Math.round(importance * 10000) / 100,
    }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15);

  if (chartData.length === 0) {
    return (
      <div className="card h-80 flex items-center justify-center text-gray-500">
        No feature importance data available
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        Feature Importance{modelName ? ` - ${modelName}` : ''}
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              type="number"
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v: number) => `${v}%`}
            />
            <YAxis
              type="category"
              dataKey="feature"
              stroke="#6b7280"
              fontSize={11}
              tickLine={false}
              width={120}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1f2937',
                border: '1px solid #374151',
                borderRadius: '8px',
                color: '#f3f4f6',
              }}
              formatter={(value: number) => [`${value.toFixed(2)}%`, 'Importance']}
            />
            <Bar
              dataKey="importance"
              fill="#3b82f6"
              radius={[0, 4, 4, 0]}
              barSize={18}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
