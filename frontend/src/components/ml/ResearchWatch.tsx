import { FlaskConical } from 'lucide-react';
import clsx from 'clsx';

export interface WatchQuarter {
  quarter: string;
  n: number;
  mean_ic: number;
  t_stat: number;
  positive: boolean;
}

export interface WatchedFactor {
  factor: string;
  hypothesis: string;
  observations_recorded: number;
  quarters: WatchQuarter[];
  current_positive_streak: number;
}

export interface FundingWatch {
  horizon_hours: number;
  required_consecutive_positive_quarters: number;
  factors: WatchedFactor[];
  verdict: string;
}

interface ResearchWatchProps {
  data: FundingWatch;
}

/**
 * The walk-forward adjudication record, one row of dots per registered
 * hypothesis. Deliberately undramatic: nothing here is a performance claim,
 * no green until a registered bar is met, and the verdict line comes
 * verbatim from the API.
 */
export default function ResearchWatch({ data }: ResearchWatchProps) {
  const required = data.required_consecutive_positive_quarters;

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <FlaskConical className="w-4 h-4 text-purple-400" />
        Walk-Forward Watch
        <span className="text-xs text-gray-500 normal-case font-normal">
          {data.factors.length} registered hypotheses — not strategies
        </span>
      </div>

      <div className="space-y-4">
        {data.factors.map((f) => {
          const streak = Math.min(f.current_positive_streak, required);
          return (
            <div key={f.factor} className="border-t border-gray-800 pt-3 first:border-t-0 first:pt-0">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <span className="text-sm font-mono text-gray-200">{f.factor}</span>
                <span className="flex items-center gap-1.5">
                  {Array.from({ length: required }, (_, i) => (
                    <span
                      key={i}
                      className={clsx(
                        'w-2.5 h-2.5 rounded-full border',
                        i < streak
                          ? 'bg-purple-500/70 border-purple-400'
                          : 'bg-gray-800 border-gray-600',
                      )}
                    />
                  ))}
                  <span className="text-[11px] text-gray-500 ml-1">
                    {streak}/{required}
                  </span>
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-0.5">{f.hypothesis}</p>
              {f.quarters.length > 0 ? (
                <div className="flex flex-wrap gap-x-4 gap-y-0.5 mt-1 text-xs font-mono">
                  {f.quarters.map((q) => (
                    <span
                      key={q.quarter}
                      className={q.positive ? 'text-purple-300' : 'text-gray-500'}
                    >
                      {q.quarter}: {q.mean_ic >= 0 ? '+' : ''}
                      {q.mean_ic.toFixed(4)} (n={q.n})
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-600 mt-1">
                  no resolved observations yet
                </p>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-gray-600 mt-3">{data.verdict}</p>
    </div>
  );
}
