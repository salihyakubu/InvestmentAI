import clsx from 'clsx';
import { format } from 'date-fns';
import { RefreshCw } from 'lucide-react';
import type { ModelInfo } from '../../types';

interface ModelPerformanceProps {
  models: ModelInfo[];
  onRetrain?: (modelId: string) => void;
  retraining?: boolean;
}

// A metric the backend does not compute is shown as "not measured", never as 0.
const NOT_MEASURED = '\u2014';
const pct = (v: number | undefined) =>
  v === undefined ? NOT_MEASURED : `${(v * 100).toFixed(1)}%`;
const fixed = (v: number | undefined, digits: number) =>
  v === undefined ? NOT_MEASURED : v.toFixed(digits);
// Three-class chance level is 1/3; the gate requires strictly more.
const skillColor = (balanced: number | undefined) =>
  balanced === undefined
    ? 'text-gray-500'
    : balanced > 1 / 3
      ? 'text-green-500'
      : 'text-red-400';

export default function ModelPerformance({
  models,
  onRetrain,
  retraining,
}: ModelPerformanceProps) {
  const statusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-400 bg-green-500/10';
      case 'training':
        return 'text-yellow-400 bg-yellow-500/10';
      case 'failed':
        return 'text-red-400 bg-red-500/10';
      default:
        return 'text-gray-400 bg-gray-500/10';
    }
  };

  if (models.length === 0) {
    return (
      <div className="card">
        <div className="card-header">Model Performance</div>
        <div className="text-center py-8 text-gray-500">No models available</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Model Performance</div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              {[
                'Model',
                'Type',
                'Status',
                'Accuracy',
                'Base rate',
                'Balanced acc',
                'Precision',
                'Recall',
                'F1',
                'Sharpe',
                'Predictions',
                'Last Trained',
                '',
              ].map((h) => (
                <th
                  key={h}
                  className="text-left text-xs text-gray-500 font-medium uppercase tracking-wider py-2 px-3"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr
                key={model.id}
                className="border-b border-gray-700/50 hover:bg-gray-800/50"
              >
                <td className="py-2.5 px-3 text-sm">
                  <div className="font-semibold text-white">{model.name}</div>
                  {model.calibration && (
                    <div className="text-xs text-gray-500 font-mono">
                      cal: {model.calibration.method}
                      {model.calibration.brier !== undefined &&
                        ` · Brier ${model.calibration.brier.toFixed(4)}`}
                    </div>
                  )}
                </td>
                <td className="py-2.5 px-3 text-xs text-gray-400 uppercase font-mono">
                  {model.type}
                </td>
                <td className="py-2.5 px-3">
                  <span
                    className={clsx(
                      'text-xs font-medium px-2 py-0.5 rounded capitalize',
                      statusColor(model.status),
                    )}
                  >
                    {model.status}
                  </span>
                </td>
                <td
                  className="py-2.5 px-3 font-mono text-sm text-gray-300"
                  title="Raw validation accuracy. Equal to the base rate for a model that always predicts the majority class."
                >
                  {pct(model.accuracy)}
                </td>
                <td
                  className="py-2.5 px-3 font-mono text-sm text-gray-500"
                  title="Majority-class rate of the validation labels: the accuracy a constant predictor scores."
                >
                  {pct(model.majority_class_rate)}
                </td>
                <td
                  className={clsx(
                    'py-2.5 px-3 font-mono text-sm font-semibold',
                    skillColor(model.balanced_accuracy),
                  )}
                  title="Balanced accuracy on the served pipeline (chance = 33.3% for three classes). The promotion gate's metric."
                >
                  {pct(model.balanced_accuracy)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {pct(model.precision)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {pct(model.recall)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {fixed(model.f1_score, 3)}
                </td>
                <td
                  className={clsx(
                    'py-2.5 px-3 font-mono text-sm font-semibold',
                    model.sharpe_ratio === undefined
                      ? 'text-gray-500'
                      : model.sharpe_ratio >= 1
                        ? 'text-green-500'
                        : 'text-yellow-500',
                  )}
                >
                  {fixed(model.sharpe_ratio, 2)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {model.prediction_count === undefined
                    ? NOT_MEASURED
                    : model.prediction_count.toLocaleString()}
                </td>
                <td className="py-2.5 px-3 text-xs text-gray-500">
                  {format(new Date(model.last_trained), 'MMM dd, HH:mm')}
                </td>
                <td className="py-2.5 px-3 text-right">
                  {onRetrain && (
                    <button
                      onClick={() => onRetrain(model.id)}
                      disabled={retraining || model.status === 'training'}
                      className="p-1.5 text-gray-500 hover:text-blue-400 transition-colors disabled:opacity-50"
                      title="Retrain model"
                    >
                      <RefreshCw
                        className={clsx(
                          'w-4 h-4',
                          model.status === 'training' && 'animate-spin',
                        )}
                      />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
