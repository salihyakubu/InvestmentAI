import { Brain } from 'lucide-react';
import clsx from 'clsx';

interface Era {
  label: string;
  start: string;
  end: string | null;
  days: number;
  n: number;
  mean_ic: number | null;
  ic_t_stat: number | null;
  mean_brier: number | null;
  sign_agreement: number | null;
}

interface CalibrationBin {
  bin_low: number;
  bin_high: number;
  n: number;
  mean_forecast: number;
  realized_flat_rate: number;
  gap: number;
}

export interface LearningMetricsData {
  feature_rows_persisted: number;
  eras: Era[];
  daily: {
    day: string;
    n: number;
    ic: number | null;
    abstention_rate: number;
  }[];
  calibration: CalibrationBin[];
  predictions_last_7d: number;
  observations_used: number;
  notes: string[];
}

interface LearningMetricsProps {
  data: LearningMetricsData;
}

const DASH = '—';

function fmt(v: number | null, digits = 4, signed = true): string {
  if (v === null) return DASH;
  const s = v.toFixed(digits);
  return signed && v >= 0 ? `+${s}` : s;
}

/**
 * The "is it learning?" instrument. Eras are promotion-dated segments of
 * LIVE out-of-sample signal quality -- if later eras are not better than
 * earlier ones, retraining is churning, not learning, and this card will
 * say so. Neutral styling on purpose: an IC is a measurement, not a win.
 */
export default function LearningMetrics({ data }: LearningMetricsProps) {
  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Brain className="w-4 h-4 text-sky-400" />
        Learning Metrics
        <span className="text-xs text-gray-500 normal-case font-normal">
          live out-of-sample, {data.observations_used.toLocaleString()} resolved
          observations
        </span>
      </div>

      <div className="mb-3">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
          Signal quality by model era (does retraining help?)
        </div>
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500">
              <th className="text-left font-normal pb-1">era</th>
              <th className="text-right font-normal pb-1">days</th>
              <th className="text-right font-normal pb-1">live IC</th>
              <th className="text-right font-normal pb-1">t</th>
              <th className="text-right font-normal pb-1">agree</th>
              <th className="text-right font-normal pb-1">Brier</th>
            </tr>
          </thead>
          <tbody>
            {data.eras.map((era) => (
              <tr key={era.label} className="text-gray-300">
                <td className="py-0.5">{era.label}</td>
                <td className="text-right">{era.days}</td>
                <td
                  className={clsx(
                    'text-right',
                    era.mean_ic === null
                      ? 'text-gray-600'
                      : era.mean_ic > 0
                        ? 'text-sky-300'
                        : 'text-gray-400',
                  )}
                >
                  {fmt(era.mean_ic)}
                </td>
                <td className="text-right">{fmt(era.ic_t_stat, 2)}</td>
                <td className="text-right">
                  {era.sign_agreement === null
                    ? DASH
                    : `${(era.sign_agreement * 100).toFixed(1)}%`}
                </td>
                <td className="text-right">{fmt(era.mean_brier, 4, false)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.calibration.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">
            Abstention calibration — stated p(flat) vs realized
          </div>
          {/* Each bin as a dumbbell on a 0-100% track: hollow marker = what
              the gate STATED, filled marker = what reality DELIVERED, the
              connector is the miscalibration. Amber only when the gap is
              material (>=5pp) -- a measurement, not a decoration. */}
          <div className="space-y-1.5">
            {data.calibration.map((bin) => {
              const stated = bin.mean_forecast * 100;
              const realized = bin.realized_flat_rate * 100;
              const material = Math.abs(bin.gap) >= 0.05;
              const lo = Math.min(stated, realized);
              const width = Math.abs(stated - realized);
              return (
                <div
                  key={bin.bin_low}
                  className="flex items-center gap-3"
                  title={`bin [${bin.bin_low.toFixed(1)}–${bin.bin_high.toFixed(1)}): stated ${stated.toFixed(1)}% → realized ${realized.toFixed(1)}% (gap ${(bin.gap * 100).toFixed(1)}pp, n=${bin.n})`}
                >
                  <span className="text-[11px] font-mono text-gray-500 w-16 shrink-0">
                    [{bin.bin_low.toFixed(1)}–{bin.bin_high.toFixed(1)})
                  </span>
                  <div className="relative flex-1 h-4">
                    <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-px bg-white/[0.07]" />
                    <div
                      className={clsx(
                        'absolute top-1/2 -translate-y-1/2 h-[3px] rounded-full',
                        material ? 'bg-amber-400/50' : 'bg-white/15',
                      )}
                      style={{ left: `${lo}%`, width: `${width}%` }}
                    />
                    <span
                      title="stated"
                      className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full border-2 border-gray-400 bg-ink-raised"
                      style={{ left: `${stated}%` }}
                    />
                    <span
                      title="realized"
                      className={clsx(
                        'absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full',
                        material ? 'bg-amber-400' : 'bg-gray-300',
                      )}
                      style={{ left: `${realized}%` }}
                    />
                  </div>
                  <span
                    className={clsx(
                      'text-[11px] font-mono tabular-nums w-24 shrink-0 text-right',
                      material ? 'text-amber-400' : 'text-gray-400',
                    )}
                  >
                    {stated.toFixed(0)}%→{realized.toFixed(0)}%
                  </span>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-4 mt-2 text-[10px] text-gray-600">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full border-2 border-gray-400 bg-ink-raised inline-block" />
              stated
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-gray-300 inline-block" />
              realized
            </span>
          </div>
        </div>
      )}

      <p className="text-[11px] text-gray-600">
        {data.predictions_last_7d.toLocaleString()} predictions in the last 7
        days; {data.feature_rows_persisted.toLocaleString()} feature-bearing
        rows accumulated toward the live-transfer promotion gate. IC
        de-overlapped; sign agreement excludes unchanged bars; a dash means
        the data cannot support the metric.
      </p>
    </div>
  );
}
