# Go-Live Runbook

Staged path from "code-correct" to real-money trading. **Do not skip stages.**
Status legend: ✅ verified in dev · ⚙️ you run it · 🔴 hard gate.

Verdict: **paper-trade after Stage 1; do not risk real money until Stage 2 (edge)
and Stage 3 (paper soak) pass.**

---

## Stage 0 — Code correctness ✅ (done)
- 124 tests green, `ruff` clean, `mypy --strict` clean (now a blocking gate), CI in place.
- Migration verified on real Postgres **and** real TimescaleDB (0002 builds 4 hypertables +
  retention; no-op on plain Postgres); full pipeline verified over real Redis
  (`scripts/smoke_test_pipeline.py`); models train + predict (`scripts/validate_model.py`);
  tree probabilities calibrated; auth path verified on real Postgres.
- Frontend: login/auth gate in place, FE↔BE paths reconciled, production nginx build verified
  (tsc strict + vite); login render + auth-redirect confirmed in-browser.

## Stage 1 — Deploy + smoke test on real infra ⚙️
1. ✅ **Rotate credentials** (done locally 2026-06-21; old keys revoked at source):
   - New Alpaca key verified live against the paper endpoint (HTTP 200, ACTIVE).
   - New Binance key set (64/64); old key deleted in console = revoked.
   - Strong `JWT_SECRET` (64 random hex chars) in place — no longer the default.
   - `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (paper) confirmed.
   - ⚠️ Still TODO at deploy time: put these same 5 secrets in Railway's secret
     store (the local `.env` is not shipped). In live mode the app refuses to
     start with the default `JWT_SECRET` — by design.
2. **Provision** Railway services: Postgres + Redis. For TimescaleDB hypertables use a
   TimescaleDB image/template; on vanilla Railway Postgres migration `0002` safely no-ops
   (hypertables skipped, schema otherwise identical).
3. **Two app services from this repo** (same `Dockerfile`):
   - `api` (web) — start = Dockerfile CMD (`uvicorn api.main:app`); healthcheck `/healthz`
     (already in `railway.toml`); `releaseCommand = alembic upgrade head` runs automatically.
   - `worker` — override start command to `python -m services.worker` (no healthcheck).
4. **Env vars** (both services): `DATABASE_URL=${{Postgres.DATABASE_URL}}` (the app converts
   `postgresql://` → asyncpg itself), `REDIS_URL=${{Redis.REDIS_URL}}`, the 5 rotated secrets
   (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`,
   `JWT_SECRET`), `ALPACA_BASE_URL=https://paper-api.alpaca.markets`, `TRADING_MODE=paper`,
   `INITIAL_CAPITAL=100.00`. (`CORS_ALLOW_ORIGINS` if a deployed frontend origin is added.)
5. Create a login user: `railway run python scripts/create_user.py you@x.com '<pw>' admin`.
6. **Smoke test** against the deployed Redis:
   `railway run python scripts/smoke_test_pipeline.py` → must print `SMOKE TEST PASSED`.
7. Sanity: `GET /healthz` 200; `GET /api/v1/health` → db+redis ok;
   `POST /api/v1/auth/token` returns a JWT; an authed request succeeds.

> **Deploy dry-run (done 2026-06-21):** the production image was built and exercised locally
> against TimescaleDB + Redis containers — it builds, runs `alembic upgrade head` (0001+0002,
> 4 hypertables created), serves, connects DB+Redis, and answers `/healthz` 200,
> `/api/v1/health` `{database:ok,redis:ok}`, `/api/v1/auth/token` 401 on bad creds. A root
> `.dockerignore` was added so the image no longer bakes `.env`/`.venv`/`.git` (5.07GB → 2.87GB).
>
> **Full-stack compose deploy (done 2026-07-19):** the platform's own `docker compose` stack
> (TimescaleDB + Redis + api + worker) was brought up end-to-end: migrations applied (0001+0002,
> 4 hypertables), api answers `/healthz` 200 and `/api/v1/health` `{database:ok,redis:ok}`, the
> worker logs `all_services_started` / `account_sync_started` / **`live_brokers_disabled
> trading_mode=paper`** (the safety gate live in a running deployment), a user was created and
> the JWT auth path enforced (authed 200 / unauthed 401), and the pipeline smoke test **PASSED**
> through the composed Redis. Compose now injects service-name `DATABASE_URL`/`REDIS_URL` so it
> works out of the box. This stack is soak-ready (Stage 3) locally.
>
> **Cloud deploy (done 2026-07-19):** Railway project `investai` is live — Postgres + Redis +
> **api** + **worker**, all provisioned and deployed via CLI (zero dashboard config). The api's
> pre-deploy phase runs `alembic upgrade head` (0001+0002 confirmed in deploy logs); public URL
> `https://api-production-bd56.up.railway.app` answers `/healthz` 200, `/api/v1/health`
> `{database:ok,redis:ok}`, auth 401 fails closed; the worker passes its own `/healthz` liveness
> endpoint and logs `live_brokers_disabled trading_mode=paper`. Two deploy bugs were found and
> fixed by the real run: `optuna` missing from requirements.txt (worker crash) and
> `releaseCommand` being an unknown Railway key (migrations silently skipped — the key is
> `preDeployCommand`).
>
> **Stage 1 SIGN-OFF (2026-07-19):** secrets loaded on both services and verified by name;
> both deployments green; api logs clean (JWT-default warning **gone**, DB+Redis connected);
> public URL answers `/healthz` 200, `/api/v1/health` `{database:ok,redis:ok}`, auth 401
> fails closed; the pipeline smoke test **PASSED against the cloud Redis**. The worker owns
> the single Alpaca data websocket (stop any local worker running the same keys — Alpaca
> allows one concurrent stream per account, and two workers fight for it).
>
> ⚠️ **Railway UI trap (learned the hard way):** the dashboard's *Raw Editor* REPLACES the
> entire variable set with the pasted text — a partial paste deletes every other variable
> (it wiped `DATABASE_URL`/`REDIS_URL`/`START_CMD` and broke 3 deploys in seconds). Prefer
> `railway variables --set` from the CLI, which merges. If you use the Raw Editor, always
> edit the complete set.
>
> Remaining for the operator: create your login user —
> `railway ssh keys add` then
> `railway ssh -s api -- python scripts/create_user.py you@x.com '<pw>' admin`.

## Stage 2 — Prove the strategy has edge 🔴
1. **Ingest real history for a small universe** (the harness reads a `close` column;
   provider auto-selected by symbol, both keyless):
   - `PYTHONPATH=. python scripts/fetch_history.py AAPL --start 2015-01-01 --out aapl.csv`
   - repeat for a few names (e.g. `MSFT`, `SPY`) and/or `BTC/USDT` for crypto.
2. **Run the edge gate across the whole universe in one shot:**
   `PYTHONPATH=. python scripts/validate_model.py aapl.csv msft.csv spy.csv --cost-bps 5`
   (prefix `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib` on macOS; libgomp1 is already
   in the Docker image). The gate is deliberately conservative:
   - **Net of costs** — returns subtract a round-trip commission+slippage charge
     (`--cost-bps`, default 5; raise for crypto / illiquid names).
   - **Non-overlapping** holding periods — no Sharpe inflation from overlapping windows.
   - **Stability** — the edge must be net-positive in ≥75% of out-of-sample sub-periods,
     not one lucky stretch.
3. **Gate — per symbol:** hit-rate > 52% AND Sharpe > 0.5 AND stability ≥ 0.75.
   **Overall:** the verdict is `EDGE STABLE across k/n symbols` only when a majority of
   **≥3** symbols pass. A single-symbol run prints `necessary but NOT sufficient` and is
   **not** a green light. If the verdict is anything else, **stop — do not trade.**
   Self-test: run with no args (zero-drift random walk) — it must report
   `NO demonstrable edge`. The null and random data correctly report no edge.

> **Edge validation (run 2026-07-18):** the gate was run across a **diverse 10-symbol
> universe** (AAPL, MSFT, NVDA, GOOGL, JPM, JNJ, XOM, PG, KO, SPY) over **two
> non-overlapping out-of-sample windows** — 2010→2026 (test ≈2021–26, incl. the 2022
> bear) and 2004→2019 (test ≈2015–19). Net of 5 bps: **7/10 pass** in the recent window
> and **9/10** in the earlier one, clearing the ≥3-symbol majority gate in both.
>
> Read it honestly: this is a **momentum signal** (all features are trend/momentum). It
> wins on trending names in each regime and fails on whatever is *not* trending (energy
> 2015–19; defensives MSFT/PG/JNJ 2021–26), with **large drawdowns** (NVDA −55/−66%,
> GOOGL −19/−43%). That is real, cross-period *factor* exposure — not a magic signal, and
> not universal alpha. It is **necessary, not sufficient**: costs are modelled optimistically
> (5 bps, low for a micro account), and the risk engine (position sizing + 7%-daily circuit
> breaker) is what makes those drawdowns survivable — its **live behaviour is only proven in
> the Stage 3 paper soak**, which remains the real gate before any capital.

## Stage 3 — Paper-trading soak (weeks) ⚙️
Run the worker in paper mode and watch for **weeks**:
- Orders fill; the equity curve is sane.
- The **circuit breaker actually trips** on a drawdown day (risk
  `sync_account()` must be fed — see Operational glue).
- `reconcile_positions()` shows **no drift** vs the broker.
- No crashes / stuck consumers; alerts fire.

## Stage 4 — Tiny live, supervised ⚙️
- $100, hard caps, manual supervision.
- Verify the **kill switch** before funding: `ExecutionEngineService.emergency_flatten()`
  cancels open orders, flattens positions, and halts. (Unit-tested; rehearse it live.)
- Scale only if Stage 4 matches expectations.

---

## Operational glue — status
- **Risk equity/PnL feed**: ✅ the worker runs `risk.run_account_sync(equity_provider)` on a
  timer (`services/worker.py`), feeding the drawdown monitor / circuit breaker. ✅ The loop now
  also detects UTC session rollover and calls `risk.reset_daily()` automatically, so the daily
  drawdown baseline re-bases each day of a multi-day soak (unit-tested).
- **Manual order API → live execution** (#11): ✅ `POST /orders` publishes an `OrderIntentEvent`
  the execution worker consumes (`_handle_order_intent`); fills sync back to the DB order row.
- **Frontend** (#8): ✅ login/auth gate + production nginx build + FE↔BE path reconcile;
  deployed in the compose stack (host port via `FRONTEND_PORT`, default 3000) and verified
  through the nginx `/api` proxy.
- **ML** (#12): ✅ tree-probability calibration + sequence double-normalisation fix. ✅ Torch
  sequence models now calibrate too (temperature scaling fit on the validation split, persisted,
  T=1.0 backwards-compatible). Deliberate: torch stays OUT of the deploy image
  (requirements.txt) — sequence models remain inert stubs in prod until they earn their +2GB by
  passing the same edge validation as the trees.
- **Timescale** (#14): ✅ hypertables + retention as a guarded migration (`0002`, verified on
  real TimescaleDB and no-op on plain Postgres).

## Emergency procedures
- **Halt + flatten:** call `emergency_flatten()` (or `halt()` to just stop new orders).
- **Drawdown auto-stop:** the circuit breaker trips at `circuit_breaker_loss_pct`
  (default 7% daily) **once `sync_account()` is feeding it** — verify in Stage 3.

## Soak activation record (2026-07-19)
- **Bootstrap model deployed**: xgboost+lightgbm v1 trained on real 1-minute history by
  replaying the live 200-bar feature pipeline exactly (`scripts/train_and_promote.py`);
  lightgbm 0.505 val accuracy (3-class, random 0.333); promoted in the registry, artifacts
  ship via the repo, `ModelServer` now wired in the worker (it never was — predictions were
  hardcoded flat before this).
- **Worker runs in `eu-west` (Amsterdam)**: Binance geo-blocks US IPs (HTTP 451), so the
  original `us-west` placement had produced ZERO data since deploy. Do not move the worker
  back to a US region while Binance is the crypto data source.
- Verified live after activation: market.bars → features.ready → predictions.ready all
  flowing (one cycle per symbol per minute, 24/7 via crypto). Orders correctly gated at 0
  until feature buffers mature (~200 min) and both models agree ≥0.6 calibrated confidence.

## Rehearsal record (2026-07-19) — kill switch + restart drills PASSED
- **Kill-switch drill (live, cloud paper account):** opened a real position via the manual
  order path (0.0002 BTC/USDT, filled at $64,560.26 — the live price feed), then
  `POST /api/v1/admin/control {"action":"flatten"}` → worker received the command in ~1s,
  closed the position, halted; `resume` lifted the halt. Unauthenticated access 401;
  non-admin 403 (tested). **The remote kill switch is real**: in an incident, log into
  /docs → Authorize → POST /api/v1/admin/control with `{"action":"flatten"}`.
- **Restart drill:** worker bounced via `railway service restart`; the full pipeline
  resumed at its normal rate (+10 predictions/90s) with no manual intervention.
- The equity snapshots recorded the drill's round-trip cost (100.0000 → 99.9966, the
  paper broker's modeled slippage) — the soak audit trail works.
- **Known trade-off:** `emergency_flatten()` submits closing orders directly to the
  broker, bypassing DB order bookkeeping — speed over paperwork in an emergency; the
  worker's `control_flatten_result` log line is the audit record.
- Remaining rehearsal item: alert delivery — set `ALERT_WEBHOOK_URL` (Slack incoming
  webhook) on the worker and confirm a test alert arrives.

## Learning-loop rehearsal (2026-07-19) — full cycle observed live, PASSED

The continuous-learning cycle (predict → resolve real outcomes → evaluate → drift →
retrain → champion/challenger gate → promote → hot-reload → serve) was built (PR #22)
and then exercised end-to-end on the deployed system:

- **19:32:45Z** `POST /api/v1/admin/control {"action":"retrain"}` accepted (admin JWT).
- **19:32:45–57Z (12 s)** AutoRetrainer loaded the platform's OWN 1m `ohlcv` rows
  (collected by the soak itself), replayed the live 200-bar feature pipeline via the
  shared `dataset_builder`, and trained an xgboost challenger: val_accuracy **0.68**
  (floor 0.34, champion beaten) → registered + promoted **xgboost v2**, mirrored to
  `model_metadata` (ML Models page shows v2 active).
- **19:32:58.0Z** `ModelRetrainedEvent` published on the `system` stream →
  prediction service hot-reloaded the ensemble (`load_active_models` build-then-swap);
  the next prediction (BTC/USDT, 19:32:58.6Z) was served by the reloaded ensemble.
  **Zero downtime, no restart.**
- **19:37:26Z** first outcome-resolution pass: **19 predictions scored against
  realised 1m closes** (t0 vs t+5m, ±5bp deadband) — `outcomes_resolved count=19`.
- **19:38:41Z** second cycle: evaluator reported **live accuracy 0.4737 on
  sample_size=19 REAL outcomes** (previously impossible — sample_size was always 0),
  and the interval governor correctly **declined** to retrain (last trained 6 min
  ago, no drift): the gate works in both directions.

Honest caveats, recorded so the numbers are not over-read:
- The 0.68 challenger val accuracy comes from ~7 h of self-collected bars with heavy
  train overfit (train_acc 0.97) — a small, same-regime validation set. The gate
  decision was legitimate; the number is not evidence of durable edge (Stage 2 is).
- Drift detection is armed on real actuals but needs ≥100 resolved outcomes per model
  before it can fire — it engages as the soak accumulates.
- Known limitation: predictions carry the ensemble id (`ensemble:xgboost,lightgbm`)
  and `_resolve_model_type` maps it to the FIRST member, so auto-retraining currently
  refreshes xgboost only; lightgbm retrains require per-member iteration (follow-up).
- Outcome tracking is in-memory: a worker restart clears unresolved predictions
  (they re-accumulate within minutes; the `predictions` DB table is unaffected).

## Soak-hardening record (2026-07-19) — adversarial audit + fixes

A 4-dimension adversarial audit (memory growth, outcome correctness, drift wiring,
live health) of the new learning loop produced 30 findings; the load-bearing ones
were fixed the same evening:

- **Autonomous trade path was DEAD (critical):** the portfolio optimizer subscribed
  to stream `predictions` while predictions publish on `predictions.ready`, and
  nothing ever called `optimize()`/`trigger_rebalance()`. The flat equity and zero
  positions of the soak were a dead wire, not conservative gating. Now: correct
  stream, a rebalance trigger (long-only, confidence ≥ 0.6, 120 s freshness,
  300 s cooldown), reference prices from `market.prices`, and explicit weight-0
  exits for symbols that stop qualifying. **The AI can now open paper positions
  autonomously**; risk sizing/limits and the kill switch govern it.
- **Retraining froze the worker (high, observed live):** the 19:32Z retrain ran
  synchronously on the event loop for ~60 s — every Redis consumer timed out and
  Railway dropped 682 log lines. Training now runs in a worker thread.
- **Memory retention (critical):** the learning stack quadruple-stored every
  prediction in unbounded structures (~15-30 MB/day; OOM risk mid-soak) — all
  four stores now capped/pruned (20 k records per model, 30-day feedback prune,
  1 k prediction history) with drift/evaluation semantics preserved.
- **Redis streams capped (critical):** events were XADD'd with no MAXLEN and acks
  never delete — Redis itself would fill within weeks and stall all publishing.
  All publishes now cap at ~100 k entries per stream.
- **Outcome windows bounded (high):** a stock prediction near session close was
  scored against the NEXT session's first bar (overnight gap recorded as a
  5-minute outcome). Lookups are now bar-grid-bounded (t0 within 10 min back,
  t+5 m within 3 min slack); unresolvable records expire after 60 min. Tracking
  now uses event creation time and skips stale backlog (> 10 min old).
- **Drift statistics fixed (medium):** the ≥100-record gate made the 5 pp drift
  threshold a coin flip (~31 % false fire) — now ≥1000 resolved records over a
  rolling 2000-record window (~1 %). `DriftDetectedEvent` now carries `model_id`.
- **Feedback loop de-garbaged (medium):** returns are now signed by the predicted
  direction (Sharpe measures skill, not market drift) and the notional-as-PnL
  order-fill write was removed.
- **Log hygiene:** worker no longer logs the full Redis URL (password) at startup;
  log level is settings-driven (INFO default — DEBUG tripped Railway's 500
  logs/sec drop limit); stdlib-logging INFO lines (e.g. hot-reload confirmation)
  are now visible in production.

**Operator actions arising:**
- **Rotate the Redis password** (Railway → Redis service): the old value is baked
  into historical deploy logs.
- `ALERT_WEBHOOK_URL` still unset — alert-delivery rehearsal remains open.

**Known gaps accepted for now (documented, not hidden):** feature-drift detection
(`DataDriftDetector.detect_feature_drift`) remains unwired (needs per-symbol
reference distributions; pooled shortcuts would false-alarm nightly); learning
state is in-memory and resets on deploy; no consumer-group XAUTOCLAIM reclaim
(at-most-once delivery on crash mid-handler); lightgbm auto-retrain pending the
ensemble-member fix (separate session).

## Post-hardening drill + overnight endurance (2026-07-19 → 07-20) — PASSED

**Async-retrain drill (19 Jul 20:31:48Z):** `{"action":"retrain"}` triggered on the
hardened build. Training ran ~14 s in a worker thread while the event loop stayed
fully live — health beats completed and predictions kept persisting *during*
training; zero Redis consumer errors (vs. the 60 s freeze + 682 dropped log lines
in the pre-fix drill). The champion/challenger gate correctly **declined** the
challenger (val 0.40 ≥ floor 0.34, but < champion 0.505) — the gate now proven in
both directions live.

**Discovery — cloud-promoted models are ephemeral:** the 19:32Z rehearsal's
xgboost v2 was promoted on the container filesystem; the next deploy rebuilt from
the repo and silently reverted serving to v1, while `model_metadata` still says
v2 is active (the ML Models page overstates). Railway has no volumes. Follow-up
task filed: persist promoted artifacts durably (DB) + reconcile the mirror at
worker startup. Until then: **a soak-period deploy discards any cloud-promoted
model** — retrains re-promote later if still better.

**Overnight endurance (20:30Z → 06:59Z, first ~10.5 h unattended on hardened
build):** zero errors/exceptions, zero consumer failures, every health beat
healthy; outcome resolution steady at 45–50 predictions per ~10-min pass;
+3,075 predictions persisted overnight (exact 5/min crypto cadence); ohlcv
~270 rows/h; snapshots on 5-min cadence; equity 100.00 with zero positions —
**correct**, since every prediction overnight was `flat` (max confidence 0.765;
zero long ≥ 0.6 calls). The autonomous trade path is armed and verified by unit
tests; live it is rightly holding until the ensemble sees an edge.

**Predictions API implemented (PR #26):** `/api/v1/predictions/{latest,history,id}`
were 501 stubs while the worker had been persisting every prediction — the
dashboard's prediction views had no data source. Now serving live rows
(verified: fresh predictions seconds old via the deployed API).

## ML ascent record (2026-07-20 → 07-21) — deep-data 3-model conformal ensemble LIVE

The prediction stack was upgraded end-to-end (PR #29) and new champions trained
and deployed (PR #30). Every change remains subject to the champion/challenger
gate; nothing bypasses it.

**Pipeline upgrades:** triple-barrier labeling (volatility-scaled barriers,
first-touch, per-symbol) replaces percentile thresholds; hyperopt now maximizes
mean accuracy across purged walk-forward folds with embargo (no label-horizon
leakage); calibration method (isotonic vs Platt) selected per model by Brier
score; split-conformal prediction sets (α=0.10) added as an abstention layer —
the ensemble emits a non-flat signal only when every agreeing member conformally
supports it; CatBoost added as a third tree family with full contract parity;
overfit guard at the gate (train−val gap > 0.35 rejected).

**Deep-data champions (trained on 90 days of real 1m bars — 105,432 train /
26,363 val rows, 48 features):**
- xgboost v2: val_accuracy 0.5471, train−val gap −0.0180
- lightgbm v2: val_accuracy 0.5473, gap −0.0125
- catboost v1: val_accuracy 0.5469, gap −0.0144

All gaps NEGATIVE (validation beats training — no overfit). Previous champions
were ~0.505 on a few hundred validation rows. NOTE: labeling changed between
generations, so accuracies are not strictly comparable across them; within-
generation, the numbers are honest out-of-sample results on 26k rows.

**Verified in production (2026-07-21 06:21Z):** worker serves
`ensemble:xgboost,lightgbm,catboost`; predictions flowing at cadence;
model_metadata mirrored to match the serving registry (UI truthful).

**Feature-drift detection ARMED:** `feature_reference.npz` (per-symbol training
distributions) now ships with artifacts; live per-symbol feature buffers
compare against it each evaluation cycle once ≥500 rows accumulate (~8h after
deploy). Drift publishes `DriftDetectedEvent(drift_type="data")` → monitoring
alerts (Slack once ALERT_WEBHOOK_URL is set) and arms retraining.

**Deep history backfill:** ~648k rows of 90-day 1m crypto history loaded into
`ohlcv` (idempotent script, `scripts/backfill_history.py`); cloud retrains now
use a 30-day lookback (was 7). Ops note: run it with modest `--batch` values —
Railway's public Postgres proxy kills very long-lived connections.

**Ops trap recorded:** deploy the `ui` service from `frontend/` (it has its own
railway link); a repo-root `railway up -s ui` builds the backend image onto it
and fails healthcheck.

**Cloud deep-data retrain drill (2026-07-21 10:33–10:41Z) — PASSED:** with the
backfill complete (649,405 rows of 90-day 1m history across 5 pairs), an
operator `retrain` command made the WORKER train on 30 days of its own data
through the full new pipeline (~8 min: 216k rows loaded, features replayed,
triple-barrier labels, hyperopt, calibration selection — isotonic won, Brier
0.1983 vs 0.1994). The worker served 141 predictions and every health beat
stayed green DURING training (the async fix holding under real load). The
champion gate then correctly **declined** the challenger: 30-day model
val 0.5279 < 90-day champion 0.5471 — less data should lose, and the gate
proved it with real numbers. The learning loop is verified end-to-end in
production at full depth.

**Trade-gate tuning (2026-07-21, operator-approved):** 28h of the new calibrated
3-model ensemble produced 1,628 predictions — 100% flat, zero non-flat calls. A
live probe of the production models showed why: honestly calibrated
probabilities sit near the label base rates (31% long / 38% flat / 31% short),
so the old `MIN_PREDICTION_CONFIDENCE=0.6` bar (p(long) ≥ 0.6 vs a 0.31 base
rate) is structurally unreachable — it dated from the pre-calibration era of
overconfident outputs. Set to **0.40** on worker+api (env var, reversible in
seconds) so the paper soak can actually exercise the trade/execution/P&L path;
3-model unanimity and conformal gating remain. Follow-up filed: replace the
absolute-confidence trigger with a calibrated edge margin (p_long − p_short).

## Incident postmortem (2026-07-21 18:43Z → 07-22 ~23:0xZ) — pipeline stall, RESOLVED

**Impact:** no bars/predictions persisted for ~4.5h (crypto gap later backfilled:
6,602 rows re-inserted; final per-symbol counts ~131k verified — no data loss).
Paper account unaffected (no positions were open).

**Root cause:** three Postgres sessions from the MORNING'S ADA backfill client sat
`idle in transaction` for 14h49m holding uncommitted `ohlcv` writes. Live inserts
eventually queued behind their transaction ids; the 18:43Z deploy restart re-formed
the lock convoy and the whole worker write path stalled. Two amplifiers made
diagnosis hard: (1) the Alpaca websocket retrying dead credentials (the API now
returns 401 — keys need re-issuing) in a tight loop with giant rich tracebacks,
flooding Railway's 500 logs/sec cap (thousands of lines dropped, burying the real
error); (2) a code rollback that changed nothing — proving the cause was state,
not code.

**Resolution:** terminated the zombie transactions (pipeline recovered within
seconds); blanked ALPACA_API_KEY/SECRET on the worker (ingestion falls back to
Yahoo; crypto unaffected) until fresh keys are issued; restored `main` on the
worker (its Alpaca backoff + single-line warnings prevent the log-storm class);
gap-backfilled the outage window.

**Prevention (applied):** `ALTER DATABASE ... SET
idle_in_transaction_session_timeout = '10min'` — server-side, any future zombie
transaction self-terminates instead of strangling the table. Follow-up filed:
backfill-script connection hygiene (guaranteed engine disposal + per-session
timeout).

**Operator action required:** issue fresh Alpaca paper keys (old ones return 401)
and set ALPACA_API_KEY / ALPACA_SECRET_KEY on the worker to restore the live
stocks feed.

**Postmortem addendum (2026-07-22):** re-enabling fresh Alpaca keys re-triggered
the storm — the 401 was only a trigger, not the cause. True root cause: the
Alpaca DataStream was constructed on the worker's main event loop but run on a
private loop the SDK creates in a thread, with bar callbacks awaited back across
the divide ("Future attached to a different loop" storms → log-cap floods →
event-loop starvation → poisoned DB sessions). Fixed: the stream now lives
entirely on a dedicated thread's own loop (constructed AND driven there), bars
marshal to the main loop thread-safely, shutdown joins cleanly, and non-transient
stream errors log once + retry with capped backoff instead of tracebacking at
multiple/sec. Keys restored from parked variables the same day.

**Edge-margin trade gate (2026-07-21):** the follow-up above is implemented.
The rebalance trigger now qualifies a long prediction on calibrated edge —
`p(long) − p(short) >= MIN_EDGE_MARGIN` (settings.min_edge_margin, default
0.10) — instead of an absolute confidence bar; `PredictionReadyEvent` carries
the full probability map (events without one fail the gate closed, so mixed
old/new versions during a deploy cannot trade on missing data).
`min_prediction_confidence` remains only as the serving-side abstain (flat
below the floor). Ops: the `MIN_PREDICTION_CONFIDENCE=0.40` env var on
worker+api no longer gates trades — keep it as the abstain floor or remove it;
tune the trade gate via `MIN_EDGE_MARGIN` instead.

## Durable learning state (2026-07-23) — VERIFIED

Learning-loop state was in-memory, so every deploy reset the evaluator window
and the drift detector's >=1000-resolved-outcome gate to zero — a soak with
regular deploys never accumulated the history it needs. Now durable
(PR #39, migration 0004):

- predictions table gained event_id (indexed), actual_direction, actual_return,
  resolved_at. The outcome resolver best-effort writes each realised outcome
  back to its row; startup rehydrates the last 7 days of resolved rows (newest
  20k/model, oldest-first) into the evaluator and tracked-prediction history.
- Verified live: migration applied cleanly (columns + ix_predictions_event_id
  present); new predictions persist with event_id; the resolver wrote back 17
  outcomes within the first pass (sample: flat prediction -> actual long,
  +0.094%, resolved_at set); startup rehydration ran (reported empty, correct
  for the first deploy with write-back — the NEXT deploy rehydrates this data).

The soak's evidence is now cumulative across deploys.

## Trade gate (2026-07-23) — calibrated edge margin LIVE
The absolute-confidence trigger (unreachable once probabilities were honestly
calibrated) was replaced by a calibrated edge margin: a long qualifies when
p(long) - p(short) >= MIN_EDGE_MARGIN (default 0.10). Events without a
probability map fail the gate closed. Deployed (PR #38). Per-member ensemble
retraining also merged (PR #24) — all three families retrain independently.

## Remaining open gaps (for a future session)
- Wired strategy backtester (Backtesting page still a 501 stub).
- Shorting in the autonomous path + symbol-universe expansion.
- Aux regime features (funding/VIX/F&G/SPY) with train-serve replay parity.
- Alert delivery: set ALERT_WEBHOOK_URL (Slack webhook) — still unset.
(A four-agent build fleet for these hit the account spend limit mid-run and
wrote nothing; they remain to be built, ideally one at a time.)

## Alert delivery (2026-07-23) — VERIFIED, final rehearsal item CLOSED
ALERT_WEBHOOK_URL (Slack incoming webhook) set on the worker by the operator.
End-to-end test: a DriftDetectedEvent published to the SYSTEM stream was
consumed by the deployed MonitoringService -> AlertManager -> Slack POST
returned HTTP 200; "Model Drift Detected" landed in the channel; the
worker.alerts_log_only startup warning is gone. Circuit-breaker trips, drift
detections, and health criticals now page the operator instead of dying in logs.

**Security finding + fix (same session):** httpx logs every request URL at INFO,
and the Slack webhook URL embeds a secret token, so the first alert wrote the
token into the worker log store. Fixed (PR #41): httpx/httpcore capped at
WARNING so request URLs are never logged. **Operator action recommended:**
rotate the Slack webhook (the token was exposed in logs before the fix) —
Slack app -> Incoming Webhooks -> remove the old webhook, Add New Webhook,
update ALERT_WEBHOOK_URL. Low urgency (a webhook only posts to one channel),
but clean hygiene.

## Flat equity is CORRECT, not a bug (2026-07-23) — operator decision recorded
Observed: equity flat at $100, every prediction HOLD. Diagnosed by probing the
live models on current bars: p(flat) ~0.48-0.62 dominates every symbol, and the
long-vs-short edge (p_long - p_short) is ~0 (+/-0.03), so the edge-margin gate
(>= 0.10, long argmax) correctly does not fire. Root cause is honest, not
broken: triple-barrier labeling + calibrated probabilities + a calm market mean
the models genuinely see no tradeable directional edge at the 5-minute horizon.
The small positive "predicted return" (~+0.07% avg) is the regressor's drift
estimate, far below the volatility barrier and trading costs.

DECISION (operator, 2026-07-23): KEEP THE GATE HONEST. Do not lower
min_edge_margin to manufacture soak activity -- trading on a ~0.03 edge is
trading on noise, the exact self-deception the platform exists to avoid. The
soak currently validates the LEARNING loop (predict -> resolve -> evaluate ->
drift -> retrain), which is its present purpose. Real trades will appear when
volatility rises and genuine edges emerge, OR when a regressor/expected-return
trigger is proven to have edge via the (not-yet-built) backtester. A flat
equity curve during a calm regime is the system being truthful.
