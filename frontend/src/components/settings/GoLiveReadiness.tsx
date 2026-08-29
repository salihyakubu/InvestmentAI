import { CheckCircle2, XCircle, CircleDashed, ShieldAlert } from 'lucide-react';
import clsx from 'clsx';
import {
  useBenchmark,
  useFundingWatch,
  usePortfolioSummary,
  useRiskMetrics,
  useTradingMode,
} from '../../api/hooks';

type CheckState = 'pass' | 'fail' | 'pending';

interface Check {
  label: string;
  state: CheckState;
  detail: string;
}

/**
 * The go-live DECISION SURFACE -- deliberately not a trigger.
 *
 * There is no activation button anywhere in this interface, by design: a
 * compromised browser session must never be one click away from live
 * capital. This panel does the two honest things a UI can do: report the
 * registered preconditions against live data, and document the operator
 * procedure (environment variable + redeploy) that IS the button.
 */
export default function GoLiveReadiness() {
  const { data: tradingMode = 'paper' } = useTradingMode();
  const { data: watch } = useFundingWatch();
  const { data: benchmark } = useBenchmark();
  const { data: portfolio } = usePortfolioSummary();
  const { data: risk } = useRiskMetrics();

  const checks: Check[] = [];

  // 1. Registered evidence of edge: the bar no factor may skip.
  if (!watch) {
    checks.push({
      label: 'A registered factor has met its four-quarter bar',
      state: 'pending',
      detail: 'waiting for the adjudication record…',
    });
  } else {
    const met = watch.factors.filter(
      (f) =>
        f.current_positive_streak >= watch.required_consecutive_positive_quarters,
    );
    checks.push({
      label: 'A registered factor has met its four-quarter bar',
      state: met.length > 0 ? 'pass' : 'fail',
      detail:
        met.length > 0
          ? `${met.length} of ${watch.factors.length} factors qualified: ${met.map((f) => f.factor).join(', ')}`
          : `0 of ${watch.factors.length} registered factors have completed ${watch.required_consecutive_positive_quarters} consecutive positive quarters on unseen data`,
    });
  }

  // 2. The account beats doing nothing.
  const lastBench = benchmark?.[benchmark.length - 1]?.benchmark_equity;
  if (portfolio?.total_equity === undefined || lastBench === undefined) {
    checks.push({
      label: 'Account equity beats the do-nothing benchmark',
      state: 'pending',
      detail: 'waiting for equity and benchmark series…',
    });
  } else {
    const ahead = portfolio.total_equity > lastBench;
    checks.push({
      label: 'Account equity beats the do-nothing benchmark',
      state: ahead ? 'pass' : 'fail',
      detail: `account ${portfolio.total_equity.toFixed(2)} vs equal-weight buy-and-hold ${lastBench.toFixed(2)}`,
    });
  }

  // 3. Risk stack green.
  if (!risk?.circuit_breaker_status) {
    checks.push({
      label: 'Circuit breaker closed',
      state: 'pending',
      detail: 'waiting for risk telemetry…',
    });
  } else {
    const closed = risk.circuit_breaker_status.toLowerCase() === 'closed';
    checks.push({
      label: 'Circuit breaker closed',
      state: closed ? 'pass' : 'fail',
      detail: `breaker state: ${risk.circuit_breaker_status}`,
    });
  }

  const passed = checks.filter((c) => c.state === 'pass').length;
  const allPass = passed === checks.length;

  const icon = (state: CheckState) =>
    state === 'pass' ? (
      <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0 mt-0.5" />
    ) : state === 'fail' ? (
      <XCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
    ) : (
      <CircleDashed className="w-4 h-4 text-gray-500 shrink-0 mt-0.5" />
    );

  return (
    <div className="card">
      <div className="card-header flex items-center gap-2">
        <ShieldAlert className="w-4 h-4 text-red-400" />
        Go-Live Readiness
        <span className="text-xs text-gray-500 normal-case font-normal">
          a decision surface — deliberately not a trigger
        </span>
      </div>

      <div
        className={clsx(
          'rounded-xl px-3.5 py-2.5 text-sm font-medium mb-4',
          tradingMode === 'live'
            ? 'bg-red-500/10 text-red-400 border border-red-500/30'
            : allPass
              ? 'bg-green-500/10 text-green-400 border border-green-500/30'
              : 'bg-white/[0.04] text-gray-300 border border-ink-hairline',
        )}
      >
        {tradingMode === 'live'
          ? 'LIVE trading is active on the deployment.'
          : `NOT READY — ${passed} of ${checks.length} registered preconditions met. The platform's own instruments currently argue against going live.`}
      </div>

      <div className="space-y-2.5 mb-4">
        {checks.map((c) => (
          <div key={c.label} className="flex items-start gap-2.5">
            {icon(c.state)}
            <div>
              <div className="text-sm text-gray-200">{c.label}</div>
              <div className="text-xs text-gray-500 font-mono">{c.detail}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-xl bg-black/25 border border-ink-hairline px-4 py-3">
        <div className="text-[11px] font-medium text-gray-500 uppercase tracking-[0.14em] mb-2">
          The activation procedure (operator only)
        </div>
        <ol className="text-xs text-gray-400 space-y-1 list-decimal list-inside">
          <li>
            Confirm the checklist above passes and the registered evidence in
            GO_LIVE.md supports it.
          </li>
          <li>
            On the deployment (Railway), set{' '}
            <code className="font-mono text-gray-300">TRADING_MODE=live</code>{' '}
            and configure live broker credentials as environment variables —
            never in this interface.
          </li>
          <li>
            Redeploy. The header badge reflects the backend&rsquo;s real mode.
          </li>
        </ol>
        <p className="text-[11px] text-gray-600 mt-2.5">
          There is no button for this, by design: a compromised browser session
          must never be one click away from live capital.
        </p>
      </div>
    </div>
  );
}
