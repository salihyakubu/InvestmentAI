/**
 * A tiny inline trend line for stat tiles: 2px stroke, gradient fade fill,
 * no axes, no labels -- the hero number beside it carries the value, the
 * sparkline carries only the shape. Color follows the series' NET direction
 * (the same polarity convention as the equity chart), or a caller-supplied
 * neutral for series where direction is not a claim.
 */
export default function Sparkline({
  values,
  width = 120,
  height = 34,
  stroke,
}: {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pad = 2;
  const points = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * (width - pad * 2);
    const y = pad + (1 - (v - min) / span) * (height - pad * 2);
    return [x, y] as const;
  });
  const first = points[0]!;
  const last = points[points.length - 1]!;
  const path = points
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ');
  const area = `${path} L${last[0].toFixed(1)},${height} L${first[0].toFixed(1)},${height} Z`;
  const color =
    stroke ??
    ((values[values.length - 1] ?? 0) >= (values[0] ?? 0)
      ? 'var(--color-profit)'
      : 'var(--color-loss)');
  const gradId = `spark-${color.replace(/[^a-z0-9]/gi, '')}`;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      className="overflow-visible"
    >
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gradId})`} />
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx={last[0]} cy={last[1]} r={2.5} fill={color} />
    </svg>
  );
}
