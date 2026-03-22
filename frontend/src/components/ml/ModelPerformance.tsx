import clsx from 'clsx';
import { format } from 'date-fns';
import { RefreshCw } from 'lucide-react';
import type { ModelInfo } from '../../types';

interface ModelPerformanceProps {
  models: ModelInfo[];
  onRetrain?: (modelId: string) => void;
  retraining?: boolean;
}

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
                <td className="py-2.5 px-3 font-semibold text-white text-sm">
                  {model.name}
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
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {(model.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {(model.precision * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {(model.recall * 100).toFixed(1)}%
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {model.f1_score.toFixed(3)}
                </td>
                <td
                  className={clsx(
                    'py-2.5 px-3 font-mono text-sm font-semibold',
                    model.sharpe_ratio >= 1 ? 'text-green-500' : 'text-yellow-500',
                  )}
                >
                  {model.sharpe_ratio.toFixed(2)}
                </td>
                <td className="py-2.5 px-3 font-mono text-sm text-gray-300">
                  {model.prediction_count.toLocaleString()}
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
