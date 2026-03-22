import clsx from 'clsx';

interface CorrelationHeatmapProps {
  data: Record<string, Record<string, number>>;
}

export default function CorrelationHeatmap({ data }: CorrelationHeatmapProps) {
  const symbols = Object.keys(data);

  if (symbols.length === 0) {
    return (
      <div className="card h-64 flex items-center justify-center text-gray-500">
        No correlation data available
      </div>
    );
  }

  const getColor = (value: number): string => {
    if (value >= 0.8) return 'bg-red-600';
    if (value >= 0.6) return 'bg-red-500/70';
    if (value >= 0.4) return 'bg-orange-500/60';
    if (value >= 0.2) return 'bg-yellow-500/50';
    if (value >= 0) return 'bg-gray-600/50';
    if (value >= -0.2) return 'bg-cyan-500/30';
    if (value >= -0.4) return 'bg-cyan-500/50';
    if (value >= -0.6) return 'bg-blue-500/60';
    return 'bg-blue-600';
  };

  return (
    <div className="card">
      <div className="card-header">Correlation Matrix</div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="p-2 text-xs text-gray-500" />
              {symbols.map((s) => (
                <th
                  key={s}
                  className="p-2 text-xs text-gray-400 font-mono font-medium"
                >
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {symbols.map((row) => (
              <tr key={row}>
                <td className="p-2 text-xs text-gray-400 font-mono font-medium">
                  {row}
                </td>
                {symbols.map((col) => {
                  const value = data[row]?.[col] ?? 0;
                  return (
                    <td key={col} className="p-1">
                      <div
                        className={clsx(
                          'w-full aspect-square flex items-center justify-center rounded text-xs font-mono',
                          getColor(value),
                          row === col
                            ? 'text-white font-bold'
                            : 'text-gray-200',
                        )}
                        title={`${row} / ${col}: ${value.toFixed(3)}`}
                      >
                        {value.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
