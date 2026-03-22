import clsx from 'clsx';

interface VaRDisplayProps {
  var95: number;
  var99: number;
  cvar95: number;
}

export default function VaRDisplay({ var95, var99, cvar95 }: VaRDisplayProps) {
  const formatCurrency = (v: number) =>
    new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(v);

  const maxVar = Math.max(Math.abs(var99), Math.abs(cvar95));

  const bars = [
    { label: 'VaR 95%', value: var95, color: 'bg-yellow-500' },
    { label: 'VaR 99%', value: var99, color: 'bg-orange-500' },
    { label: 'CVaR 95%', value: cvar95, color: 'bg-red-500' },
  ];

  return (
    <div className="card">
      <div className="card-header">Value at Risk</div>
      <div className="space-y-4">
        {bars.map((bar) => {
          const pct = maxVar > 0 ? (Math.abs(bar.value) / maxVar) * 100 : 0;
          return (
            <div key={bar.label}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-sm text-gray-400">{bar.label}</span>
                <span className="text-sm font-mono font-semibold text-gray-200">
                  {formatCurrency(Math.abs(bar.value))}
                </span>
              </div>
              <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={clsx('h-full rounded-full transition-all', bar.color)}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-4 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500">
          VaR represents the maximum expected loss over a 1-day period at the given
          confidence level. CVaR (Expected Shortfall) measures the average loss
          beyond the VaR threshold.
        </p>
      </div>
    </div>
  );
}
