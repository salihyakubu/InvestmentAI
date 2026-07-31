import { FlaskConical } from 'lucide-react';
import clsx from 'clsx';

export interface WatchQuarter {
  quarter: string;
  n: number;
  mean_ic: number;
  t_stat: number;
  positive: boolean;
}

export interface FundingWatch {
  factor: string;
  hypothesis: string;
  horizon_hours: number;
  required_consecutive_positive_quarters: number;
  observations_recorded: number;
  quarters: WatchQuarter[];
  current_positive_streak: number;
  verdict: string;
}

interface ResearchWatchProps {
  data: FundingWatch;
}

/**
 * The walk-forward adjudication record. Deliberately undramatic: this card
 * reports a registered hypothesis being judged by unseen data, and it must
 * never look like a performance claim. No green until the registered bar is
 * met, and the verdict line comes verbatim from the API.
 */
export default function ResearchWatch({ data }: ResearchWatchProps) {
  const required = data.required_consecutive_positive_quarters;
  const streak = Math.min(data.current_positive_streak, required);

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <FlaskConical className="w-4 h-4 text-purple-400" />
        Walk-Forward Watch
        <span className="text-xs text-gray-500 normal-case font-normal">
          registered hypothesis — not a strategy
        </span>
      </div>

      <p className="text-xs text-gray-400 mb-3">{data.hypothesis}</p>

      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-gray-500 uppercase tracking-wider">
          Progress
        </span>
        {Array.from({ length: required }, (_, i) => (
          <span
            key={i}
            className={clsx(
              'w-3 h-3 rounded-full border',
              i < streak
                ? 'bg-purple-500/70 border-purple-400'
                : 'bg-gray-800 border-gray-600',
            )}
          />
        ))}
        <span className="text-xs text-gray-500">
          {streak}/{required} consecutive positive quarters on unseen data
        </span>
      </div>

      {data.quarters.length > 0 ? (
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500">
              <th className="text-left font-normal pb-1">quarter</th>
              <th className="text-right font-normal pb-1">obs</th>
              <th className="text-right font-normal pb-1">mean IC</th>
              <th className="text-right font-normal pb-1">t</th>
            </tr>
          </thead>
          <tbody>
            {data.quarters.map((q) => (
              <tr key={q.quarter} className="text-gray-300">
                <td className="py-0.5">{q.quarter}</td>
                <td className="text-right">{q.n}</td>
                <td
                  className={clsx(
                    'text-right',
                    q.positive ? 'text-purple-300' : 'text-gray-400',
                  )}
                >
                  {q.mean_ic >= 0 ? '+' : ''}
                  {q.mean_ic.toFixed(4)}
                </td>
                <td className="text-right">{q.t_stat.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-xs text-gray-500">
          No resolved observations yet — the watch records its first ICs after
          the next worker cycle.
        </p>
      )}

      <p className="text-[11px] text-gray-600 mt-3">{data.verdict}</p>
    </div>
  );
}
