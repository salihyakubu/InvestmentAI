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
/** The calendar quarter we are currently inside: its cell is provisional. */
function currentQuarter(now = new Date()): string {
  return `${now.getUTCFullYear()}-Q${Math.floor(now.getUTCMonth() / 3) + 1}`;
}

export default function ResearchWatch({ data }: ResearchWatchProps) {
  const required = data.required_consecutive_positive_quarters;
  const inProgress = currentQuarter();

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
            <div
              key={f.factor}
              className="border-t border-ink-hairline pt-3.5 first:border-t-0 first:pt-0"
            >
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <span className="text-[13px] font-mono text-gray-200">
                  {f.factor}
                </span>
                <span
                  className="flex items-center gap-1.5"
                  title={`${streak} of ${required} consecutive positive quarters`}
                >
                  {Array.from({ length: required }, (_, i) => (
                    <span
                      key={i}
                      className={clsx(
                        'h-1.5 rounded-full transition-all',
                        i < streak
                          ? 'w-5 bg-purple-400'
                          : 'w-2.5 bg-white/10',
                      )}
                    />
                  ))}
                  <span className="text-[11px] text-gray-500 ml-1 tabular-nums">
                    {streak}/{required}
                  </span>
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">{f.hypothesis}</p>
              {f.quarters.length > 0 ? (
                <div className="flex flex-wrap gap-2 mt-2.5">
                  {f.quarters.map((q) => {
                    const provisional = q.quarter === inProgress;
                    return (
                      <div
                        key={q.quarter}
                        title={`${q.quarter}: mean IC ${q.mean_ic >= 0 ? '+' : ''}${q.mean_ic.toFixed(4)}, t=${q.t_stat.toFixed(2)}, n=${q.n}${provisional ? ' (quarter in progress)' : ''}`}
                        className={clsx(
                          'rounded-lg px-2.5 py-1.5 min-w-[86px]',
                          q.positive ? 'bg-purple-500/15' : 'bg-white/[0.04]',
                          provisional
                            ? 'border border-dashed border-white/20'
                            : 'border border-transparent',
                        )}
                      >
                        <div className="flex items-center gap-1.5">
                          <span
                            className={clsx(
                              'w-1.5 h-1.5 rounded-full',
                              q.positive ? 'bg-purple-400' : 'bg-gray-600',
                            )}
                          />
                          <span className="text-[10px] text-gray-500 tracking-wide">
                            {q.quarter}
                          </span>
                          {provisional && (
                            <span className="text-[9px] text-gray-600 italic">
                              in progress
                            </span>
                          )}
                        </div>
                        <div
                          className={clsx(
                            'text-[12px] font-mono tabular-nums mt-0.5',
                            q.positive ? 'text-purple-200' : 'text-gray-400',
                          )}
                        >
                          {q.mean_ic >= 0 ? '+' : ''}
                          {q.mean_ic.toFixed(4)}
                        </div>
                        <div className="text-[10px] text-gray-600 font-mono">
                          n={q.n}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-gray-600 mt-1.5">
                  no resolved observations yet
                </p>
              )}
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-gray-600 mt-4">{data.verdict}</p>
    </div>
  );
}
