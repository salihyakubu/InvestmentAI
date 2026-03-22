import { ArrowUpCircle, ArrowDownCircle, MinusCircle } from 'lucide-react';
import clsx from 'clsx';
import { format } from 'date-fns';
import type { Prediction } from '../../types';

interface PredictionViewProps {
  predictions: Prediction[];
}

export default function PredictionView({ predictions }: PredictionViewProps) {
  const signalConfig = {
    buy: {
      icon: ArrowUpCircle,
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/20',
    },
    sell: {
      icon: ArrowDownCircle,
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/20',
    },
    hold: {
      icon: MinusCircle,
      color: 'text-gray-400',
      bgColor: 'bg-gray-500/10',
      borderColor: 'border-gray-500/20',
    },
  };

  if (predictions.length === 0) {
    return (
      <div className="card">
        <div className="card-header">Latest Predictions</div>
        <div className="text-center py-8 text-gray-500">
          No predictions available
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Latest Predictions</div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {predictions.map((pred) => {
          const cfg = signalConfig[pred.signal];
          const Icon = cfg.icon;
          return (
            <div
              key={pred.id}
              className={clsx(
                'bg-gray-900 rounded-lg p-3 border',
                cfg.borderColor,
              )}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono font-bold text-white">
                  {pred.symbol}
                </span>
                <div className="flex items-center gap-1.5">
                  <Icon className={clsx('w-5 h-5', cfg.color)} />
                  <span
                    className={clsx(
                      'text-xs font-bold uppercase',
                      cfg.color,
                    )}
                  >
                    {pred.signal}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-gray-500">Confidence</span>
                  <div className="font-mono font-semibold text-gray-200 mt-0.5">
                    {(pred.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Predicted Return</span>
                  <div
                    className={clsx(
                      'font-mono font-semibold mt-0.5',
                      pred.predicted_return >= 0
                        ? 'text-green-500'
                        : 'text-red-500',
                    )}
                  >
                    {pred.predicted_return >= 0 ? '+' : ''}
                    {(pred.predicted_return * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Model</span>
                  <div className="text-gray-300 mt-0.5 truncate">
                    {pred.model_name}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Horizon</span>
                  <div className="text-gray-300 mt-0.5">{pred.horizon}</div>
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-gray-800">
                <div className="w-full bg-gray-800 rounded-full h-1.5">
                  <div
                    className={clsx('h-full rounded-full', {
                      'bg-green-500': pred.confidence >= 0.7,
                      'bg-yellow-500':
                        pred.confidence >= 0.5 && pred.confidence < 0.7,
                      'bg-red-500': pred.confidence < 0.5,
                    })}
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
                <div className="text-[10px] text-gray-600 mt-1">
                  {format(new Date(pred.timestamp), 'MMM dd, HH:mm:ss')}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
