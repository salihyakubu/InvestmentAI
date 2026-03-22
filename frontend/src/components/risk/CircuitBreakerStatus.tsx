import { Shield, ShieldAlert, ShieldOff } from 'lucide-react';
import clsx from 'clsx';

interface CircuitBreakerStatusProps {
  status: 'CLOSED' | 'OPEN' | 'HALF_OPEN';
  reason?: string;
}

export default function CircuitBreakerStatus({
  status,
  reason,
}: CircuitBreakerStatusProps) {
  const config = {
    CLOSED: {
      icon: Shield,
      label: 'CLOSED',
      description: 'Trading is active. All systems operating normally.',
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/30',
      pulseColor: 'bg-green-500',
    },
    OPEN: {
      icon: ShieldOff,
      label: 'OPEN',
      description: 'Trading is halted. Circuit breaker has been triggered.',
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/30',
      pulseColor: 'bg-red-500',
    },
    HALF_OPEN: {
      icon: ShieldAlert,
      label: 'HALF OPEN',
      description: 'Testing recovery. Limited trading allowed.',
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-500/10',
      borderColor: 'border-yellow-500/30',
      pulseColor: 'bg-yellow-500',
    },
  };

  const cfg = config[status];
  const Icon = cfg.icon;

  return (
    <div className={clsx('card border', cfg.borderColor)}>
      <div className="card-header">Circuit Breaker</div>
      <div className="flex items-center gap-4">
        <div className={clsx('p-3 rounded-xl', cfg.bgColor)}>
          <Icon className={clsx('w-8 h-8', cfg.color)} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className={clsx('text-xl font-bold', cfg.color)}>
              {cfg.label}
            </span>
            <span className="relative flex h-3 w-3">
              <span
                className={clsx(
                  'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
                  cfg.pulseColor,
                )}
              />
              <span
                className={clsx(
                  'relative inline-flex rounded-full h-3 w-3',
                  cfg.pulseColor,
                )}
              />
            </span>
          </div>
          <p className="text-sm text-gray-400 mt-1">{cfg.description}</p>
          {reason && (
            <p className="text-xs text-gray-500 mt-2 bg-gray-900 rounded p-2 font-mono">
              Reason: {reason}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
