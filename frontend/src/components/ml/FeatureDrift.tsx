import { Activity } from 'lucide-react';
import clsx from 'clsx';

export interface FeatureDriftData {
  computable: boolean;
  reason: string | null;
  generation_hash: string | null;
  n_reference: number;
  n_recent: number;
  n_symbols_measured: number;
  reference_start: string | null;
  reference_end: string | null;
  recent_start: string | null;
  n_features: number;
  n_unmeasurable: number;
  share_significant: number | null;
  top_drifted: {
    index: number;
    psi: number;
    psi_max: number | null;
    n_symbols: number;
  }[];
  worst: { symbol: string; index: number; psi: number } | null;
  thresholds: { moderate: number; significant: number };
  notes: string[];
}

interface FeatureDriftProps {
  data: FeatureDriftData;
}

/**
 * The "did the world move under the models?" instrument. PSI of each served
 * feature's recent distribution against the platform's own earlier serving
 * history. Context for the era table: input drift is evidence for the
 * regime-shift reading of a degrading era; its absence points at the models.
 */
export default function FeatureDrift({ data }: FeatureDriftProps) {
  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <Activity className="w-4 h-4 text-sky-400" />
        Feature Drift
        <span className="text-xs text-gray-500 normal-case font-normal">
          served inputs, recent vs own history
        </span>
      </div>

      {!data.computable ? (
        <p className="text-xs text-gray-500 font-mono">— {data.reason}</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono mb-2">
            <span className="text-gray-400">
              {data.n_features} features · {data.n_symbols_measured} symbols ·{' '}
              {data.n_reference.toLocaleString()} ref /{' '}
              {data.n_recent.toLocaleString()} recent vectors
            </span>
            {data.n_unmeasurable > 0 && (
              <span className="text-gray-500">
                {data.n_unmeasurable} unmeasurable
              </span>
            )}
            {data.share_significant !== null && (
              <span
                className={
                  data.share_significant > 0.1
                    ? 'text-amber-400'
                    : 'text-gray-300'
                }
              >
                {(data.share_significant * 100).toFixed(0)}% significantly
                drifted
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs font-mono">
            {data.top_drifted.length === 0 && (
              <span className="text-gray-600">
                — no feature was measurable
              </span>
            )}
            {data.top_drifted.map((f) => (
              <span
                key={f.index}
                className={clsx(
                  f.psi > data.thresholds.significant
                    ? 'text-amber-400'
                    : f.psi > data.thresholds.moderate
                      ? 'text-yellow-200'
                      : 'text-gray-300',
                )}
              >
                f{f.index}: {f.psi.toFixed(3)}
              </span>
            ))}
            {data.worst && data.worst.psi > data.thresholds.moderate && (
              <span className="text-gray-500">
                worst: f{data.worst.index}@{data.worst.symbol}{' '}
                {data.worst.psi.toFixed(3)}
              </span>
            )}
          </div>
        </>
      )}

      <p className="text-[11px] text-gray-600 mt-2">
        Reference is the platform's own early serving history, not the
        training distribution — drift brackets train-serve skew. Features
        are indexed within pipeline generation{' '}
        {data.generation_hash ?? '—'}; a dash means the data cannot support
        the metric.
      </p>
    </div>
  );
}
