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

## Market-regime features (2026-07-23) — pipeline LIVE, activation pending
Closed the "does it use news/other feeds?" gap for the achievable, honest half:
four keyless market-regime signals now feed the feature pipeline with strict
train/serve parity (PR #44, migration 0005):
- aux_funding_rate (per crypto symbol, Binance futures)
- aux_fear_greed (alternative.me), aux_vix_close, aux_spy_daily_return (Yahoo)

Design: a shared as-of AuxFeatureProvider used by BOTH live serving (latest
snapshot) and training replay (HistoricalAuxProvider, as-of each bar time), so
a feature is computed identically in both paths -- no look-ahead. Backward-
compatible: serving selects features by each model's trained names, so the
current 48-feature champions ignore the new keys. AuxMarketService polls every
5 min into aux_market_state; the retrainer replays those rows as-of during
training. Verified live: real values flowing (VIX 16.64, F&G 31, SPY -0.12%,
per-symbol funding), pipeline healthy.

NEWS/SENTIMENT deliberately OUT of scope: honest point-in-time historical news
has no reliable keyless source, and per-minute LLM sentiment over 90d of history
is cost-prohibitive. Documented as blocked on a paid historical-news source.

ACTIVATION (remaining): the features become live in DECISIONS only after a
retrain whose training window contains aux history. Two paths: (a) wait ~30d
for AuxMarketService to fill the cloud retrainer's window naturally, or (b)
backfill historical aux (Binance funding history, alternative.me F&G history,
Yahoo VIX/SPY daily) then trigger a retrain -- the champion/challenger gate then
decides whether the regime features actually improve out-of-sample accuracy.
The retrain is real cloud compute; run it deliberately.

## Regime-feature A/B verdict (2026-07-23) — NULL RESULT, honestly recorded
Controlled experiment on 90 days of production bars (103,171 train / 25,796 val
rows, identical purged-CV training, only difference = real aux values vs zeros):
  xgboost   no-aux 0.5474  with-aux 0.5448  (-0.0026)
  lightgbm  no-aux 0.5466  with-aux 0.5465  (-0.0000)
  catboost  no-aux 0.5453  with-aux 0.5460  (+0.0007)
With ~25.8k val rows the standard error is ~0.003, so every delta is within
noise: the regime features neither help nor hurt 5-minute directional accuracy
in this window. Interpretation: slow-moving daily/8-hourly signals (VIX, F&G,
funding) carry little discriminative power for the NEXT 5 MINUTES -- plausible
and now measured, not assumed.

DECISION: no promotion on this evidence (the gate would rightly have declined).
The pipeline stays live and collecting at zero risk (serving models ignore the
keys); re-test as history deepens, at longer horizons, or with interaction
features. Also learned operationally: 90-day retrains are too heavy for the
small worker (event-loop starvation during the row load; reverted its lookback
to 30d) -- heavyweight training belongs off-box.

The original gap ("does the platform use news/other feeds?") is now closed to
the extent honesty allows: the knowledge-base MACHINERY exists end-to-end with
train/serve parity, four reliable feeds flow live, and the first measurement
says today's models don't yet benefit. That measurement is the platform working.

## Strategy backtester (2026-07-23) — BUILT, DEPLOYED, VERIFIED LIVE
The last major product gap is closed. The Backtesting page was a dead 501 stub
with no engine behind it; the platform's proven methodology lived only in the
Stage-2 CLI harness. Now (PR #48, migration 0006):
- services/backtesting/edge.py is the canonical harness (net-of-cost,
  NON-overlapping holds, stability-gated, cross-symbol verdict) with a
  portfolio layer; the CLI imports from it and the null random-walk self-test
  still reports NO edge (the honesty invariant held through the refactor).
- Job-based API (run/status/results/history; orphaned jobs marked failed) and
  a job-polling UI with VISIBLE error states and an edge-verdict banner.
- Verified live through the deployed API: a 3-symbol 2018-2025 run completed in
  ~5s -- verdict "EDGE STABLE across 2/3 symbols" (AAPL and SPY pass the gate
  out-of-sample, MSFT fails), portfolio +52.2% OOS, Sharpe 1.17, maxDD 13.1%,
  359 trades. Consistent with the 2026-07-18 Stage-2 study, now reproducible
  from the dashboard by the operator.
CAVEAT (unchanged honesty): this is the DAILY-horizon harness strategy. It does
not validate the live 5-minute soak models; it is the research tool for
measuring whether any candidate signal has edge net of costs before it is
trusted with capital.

## Exploration paper-trading LIVE — first autonomous round trip (2026-07-23)
Operator direction: the learning period must also exercise the trade path.
Built (PRs #50, #51): when nothing clears the full edge gate, the single BEST
sub-gate signal (calibrated margin >= exploration_edge_margin 0.03, read from
UNDER the abstention label -- serving flattens sub-gate directions, so the
picker must use the probability margin) opens a small tagged position (3% of
equity, max 2 concurrent, auto-exit after 15 min, "explore-" order ids,
paper-mode hard gate). Closed round trips feed REAL commission-aware P&L into
the feedback loop via FIFO lot attribution.

FIRST ROUND TRIP, verified live:
- 21:24:30Z BUY 0.0395 SOL/USDT @ 76.08 (explore-tagged, 5bps slippage)
- 21:29:57Z second entry ETH/USDT (cooldown pacing; cap reached at 2)
- 21:39:42Z hold expired -> SELL @ 76.03; same cycle opened BTC/USDT
- 21:39:44Z continuous_learning.trade_closed exploration=True
  realised_return=-0.066% pnl=-$0.002 -- a real, tagged, learning outcome.

POLICY (recorded): exploration P&L is learning data, never edge evidence; the
equity curve now moves for learning reasons and must be read accordingly.
Known startup artifact: fills consumed before the first post-restart
prediction lack prediction attribution (empty prediction_id; P&L logged but
not fed to the feedback loop) -- steady-state fills are fully attributed.

## Soak review + exploration tuition tuning (2026-07-27)
Operator asked why equity looked like a straight line with "no trades". The
review proved the OPPOSITE: equity moved every day (100.00 -> 98.85 over five
days) and trade_closed events were minutes old. Two defects were HIDING the
activity, both fixed in PR #54 and verified live:
- order_sync only UPDATEd rows keyed by DB order ids, so autonomous fills
  (correlation-id client_order_ids) never reached the orders table -- the
  entire exploration history was invisible to the Trading page and audit
  trail. Now inserted as completed Order+Fill rows with the "explore-"/
  "rebal-" tag preserved in external_id. Verified live minutes after deploy:
  first autonomous row "DOT/USDT buy filled tag=explore-182cd4e6...".
- The equity chart's fixed 0-100 y-domain rendered the $1.15 soak move ~8px
  tall; the axis now hugs the data (dataMin/dataMax +/- 0.5).

TUITION TUNING (operator-approved): the -$1.15/5d bleed was pure exploration
friction (~5bps slippage per side on ~100 tiny round trips/day). Set
EXPLORATION_HOLD_MINUTES=60 (env override; code default stays 15) -- ~4x fewer
round trips, ~0.06%/day expected friction, exit outcomes span a more
meaningful horizon. Size (3%), cap (2), and margin floor (0.03) unchanged so
the learning data keeps flowing. Verified live: worker redeployed healthy and
the first entry under the new hold (BTC/USDT) fired within minutes.
POLICY unchanged: exploration P&L is learning data, never edge evidence.

## Dashboard audit + truthful metrics (2026-07-29) — PR #56
Operator: "the various dashboard indicate feels like its not functional." The
screenshot showed $99.94 equity beside +0.00% return, 0.00 Sharpe, 0.0% win
rate, 0.00% drawdown. A 5-dimension multi-agent audit (45 confirmed defects,
1 refuted) found none of those four were measurements: the frontend
normaliser coerced every field the API never sent to 0.

FIXED (PR #56, migration 0007):
1. EQUITY WAS NOT DURABLE. PaperBroker held cash/positions/cost-basis in
   process memory, so every deploy rebased the account to initial_capital.
   Proven in production data -- exactly one upward discontinuity in ten days
   of snapshots: 2026-07-27 20:40:43Z, 98.8588 -> 100.0000, the redeploy
   documented in the section above. FIVE DAYS OF SOAK P&L WERE ERASED, and
   the same thing had silently happened on earlier restarts near 100.00 where
   no jump was visible. Now: the book is checkpointed into `positions` in the
   SAME TRANSACTION as its equity snapshot; restore runs before any service
   starts; fills landing after the checkpoint are replayed. Cost basis and
   realised P&L are tracked and restored too (a test caught that restoring
   without cost basis reports a position's whole market value as unrealised
   profit). `positions` gains trading_mode so a live worker can never restore
   the paper book; account-broker selection is explicit, not dict order.
2. /portfolio/summary now carries the KPI contract, derived from stored
   history: total return MARK-TO-MARKET vs the account baseline (not
   realized_pnl, which excludes open marks), daily P&L vs the prior close,
   peak-to-trough drawdown, win rate from FIFO round trips over the fill
   ledger. Scoped by trading_mode, cached 60s against the 5s poll.
3. HONESTY GATE: a metric the data cannot support returns null and renders as
   a grey em-dash, never a coloured zero. Sharpe needs 20+ daily
   observations, so it reads "—  needs 20+ days". The circuit-breaker tile
   reads UNKNOWN instead of a green "all systems operating normally" that was
   synthesised from an unwritten risk_metrics table -- the one failure mode a
   safety indicator must not have.
4. The equity curve was served NEWEST-FIRST and drawn mirrored in time, over
   a 100-row window = 8h20m at the 5-min cadence (hence every x label reading
   "Jul 28", and the $1.15 move rendering flat). Now chronological, windowed
   in days with downsampling that keeps the newest point, on a real time axis
   with span-aware tick precision. The Risk page's drawdown had been measured
   against a running peak that included FUTURE equity.
5. Snapshots write immediately instead of sleeping an interval first (a
   crash-loop restarting faster than 300s wrote none). Prediction cards show
   their real horizon_minutes instead of a hardcoded '5m'.

VERIFIED LIVE after deploy:
- alembic head 0007_positions_mode; positions.trading_mode present.
- The one-time rebase fired exactly once, as designed: equity carried FORWARD
  at 100.16791081 onto a flat book (not reset to initial capital). Every
  restart from here is an exact restore.
- `positions` now holds real open rows (it had been 0 rows forever despite
  position_count=9), and portfolio_snapshots.unrealized_pnl is non-zero for
  the first time (0.0365) -- real cost-basis tracking is live.

HONEST CAVEAT RECORDED: the equity series still contains the 2026-07-27
20:40:43Z synthetic jump, which UNDERSTATES the soak's true loss by ~$1.14.
Total return and max drawdown are computed across that seam. The series is
continuous from 2026-07-29 onward; earlier figures must be read as two
accounts spliced together, not one track record.

STILL OPEN (documented, not shipped): the risk_metrics writer -- VaR/CVaR/
beta/volatility and the correlation heatmap remain UNAVAILABLE rather than
zero, and MaxCorrelationRule/MaxVaRRule pass vacuously because
update_returns() has no caller. Durable round-trip P&L persistence. The
secondary-page cleanup (order fill prices, Trading price chart 422s on '1D',
audit-log columns, ML precision/recall/F1, feature-importance chart, retrain
buttons, notification bell).

## LIVE SIGNAL EDGE VERDICT (2026-07-30) — NO EDGE at the traded horizon
The platform's only published edge verdict ("EDGE STABLE across 2/3 symbols",
2026-07-23) came from the DAILY harness in services/backtesting/edge.py. It
had never been pointed at the 5-minute ensemble that actually trades. This
closes that gap, using the strongest evidence available: 60,732 of the
platform's OWN recorded predictions, emitted live and resolved by the
learning loop -- genuinely point-in-time and out-of-sample, no re-simulation.

Tooling: services/backtesting/live_signal.py + scripts/evaluate_live_edge.py.
Self-tested BOTH ways before being trusted (tests/unit/test_live_signal_edge.py,
24 tests): it must return NO EDGE on noise AND detect a planted edge. That
second test caught a real bug in my own deflated-Sharpe implementation, which
had been reporting NO EDGE for a signal with IC +0.96 because the
expected-maximum benchmark was not scaled by the Sharpe estimator's standard
error. A tool that can only say "no" proves nothing.

FINDING 1 -- the 5-minute signal has NO EDGE, gross or net.
  IC +0.0066, p=0.466, 95% CI [-0.0088, +0.0237] (straddles zero)
  12,147 independent observations after removing 5x overlap
  gross expectancy -0.141 bps/period (SE ~0.16 -> indistinguishable from 0)
  decile ladder non-monotonic; 0/12 symbols pass
  This is NOT a cost problem. There is no gross signal to rescue.

FINDING 2 -- *** RETRACTED 2026-07-30, see the correction section below. ***
The claim was: "confidence is INVERTED -- directional agreement falls
monotonically as the model's own confidence rises (Q1 48.4% -> Q5 41.7%), and
the top quintile has a significantly negative IC (-0.0646, p=0.001)."
It was an artifact of unchanged price bars, not a defect in the models.

FINDING 3 -- a real but UNECONOMIC signal appears at ~60 minutes.
  mean IC by hold: 5m +0.005, 15m +0.055, 30m +0.074, 60m +0.117, 240m -0.001
  at 60m: ADA +0.196*, BTC +0.193*, SOL +0.206* (p<0.05); DOT and ETH ~0
  pooled net: +5.08 bps/hold GROSS (t=+2.68), +0.27 at 5bps, -4.54 at 10bps
  deflated Sharpe <= 0.19 everywhere (bar is 0.95); second-half holdout does
  not confirm (only BTC marginal, p=0.05)
  Read honestly: there is ~5bps of gross edge per hour, and the execution path
  costs 5-10bps per round trip. It breaks even at best-case paper slippage and
  loses at realistic venue costs. The horizon was CHOSEN AFTER seeing the
  5-minute null, so it is a hypothesis generated from the same data -- it must
  be confirmed on a window this scan never saw before it is traded.

CORRECTION TO EARLIER ADVICE (recorded deliberately): I had ranked "fix the
cost/horizon mismatch" as the highest-leverage move, on the assumption that
slippage was eating a real 5-minute edge. The data says otherwise -- at the
traded horizon there is nothing to eat. Cost reduction only matters for the
60-minute signal, and only if that signal survives out-of-sample confirmation.

IMPLICATION FOR TRADING_MODE: unchanged and reinforced. Nothing here supports
risking capital. The operator's live decision stays exactly where it was.

## CORRECTION (2026-07-30) — the "confidence inversion" was an artifact
Finding 2 of the edge verdict above is RETRACTED. It was published in PR #58
and it was wrong. Recording the retraction with the same prominence as the
original claim, because a platform whose whole differentiator is honest
measurement cannot quietly delete a bad result.

WHAT WAS CLAIMED: directional agreement falls as the model's confidence rises,
so the abstention gate keeps the worse predictions.

WHY IT WAS WRONG -- two compounding mistakes, both mine:

1. MISREAD FIELD SEMANTICS. `EnsemblePredictor._combine` sets
   `confidence = combined_probs[best_direction]`, then OVERWRITES it with
   `combined_probs["flat"]` whenever the agreement vote or conformal gate
   flattens the signal. ~100% of stored predictions are flattened, so the
   recorded "confidence" is p(flat) -- the model's belief that price will NOT
   move. It is an abstention score, not directional confidence. 98.4% of rows
   have p(flat) >= 1/3 (genuinely flat); only 1.6% are suppressed directional
   views.

2. ZERO-RETURN CONTAMINATION. Low-priced assets with coarse ticks leave the
   price literally unchanged over a 5-minute window: DOT 32.8% of bars, ADA
   18.0%, SOL 6.3% (BTC/ETH/equities all < 1%). `np.sign(0)` is 0 and matches
   no position, so every unchanged bar counts as a disagreement. That fraction
   rises with p(flat) BY CONSTRUCTION -- p(flat) is precisely the model
   predicting no move -- which manufactures a clean monotone "inversion" out
   of a model that was RIGHT about flatness.

THE EVIDENCE THAT SETTLES IT (unchanged bars excluded, de-overlapped):
  agreement by p(flat) quintile: 50.40%, 50.09%, 49.59%, 50.11%, 49.69%
    -- flat across strata; the apparent decline tracked the zero fraction
       rising 3.5% -> 14.4% exactly.
  P(up | expected_return ABOVE median) = 48.80%  (n=5,516)
  P(up | expected_return BELOW median) = 48.80%  (n=5,529)
    difference +0.01pp, z=+0.01. The signal carries ZERO directional
    information -- which STRENGTHENS Finding 1, the real result.

FINDING 1 (no edge at the traded horizon) is unaffected and reinforced.
FINDING 3 (60-minute signal) re-checked against the same artifact and SURVIVES:
  ADA +0.203 (p=0.017), BTC +0.193 (p=0.019), SOL +0.204 (p=0.013) with
  unchanged bars excluded; DOT and ETH remain ~0. Its own caveats stand --
  post-hoc horizon choice, deflated Sharpe below bar, holdout unconfirmed.

GUARD ADDED: `confidence_strata` now excludes unchanged bars from agreement
and reports the zero fraction it removed; `test_unchanged_bars_do_not_
manufacture_an_inversion` fails on the exact data shape that fooled me, and a
companion test proves the detector still catches a genuine inversion.

## CROSS-SECTIONAL RESEARCH (2026-07-30) — real signal found, and it is unharvestable
Breadth was the #1 item after the live-signal null. Built a 117-symbol hourly
crypto universe (17,532 bars x 2 years, local research cache, NOT the
production DB) and tested whether ranking symbols against each other predicts
their RELATIVE returns -- a different question from the per-symbol "will this
go up?" the platform asks today.

Tooling: services/backtesting/cross_section.py, scripts/build_universe.py,
scripts/evaluate_cross_section.py, 20 tests. Same both-ways discipline: the
harness must find a planted cross-sectional signal AND report the null on
random walks, or its verdicts mean nothing.

THE FINDING -- the first genuinely stable predictive structure this platform
has produced:
  reversal_1h   mean IC +0.0436  t=+20.9  17,530 rebalances
  quintile ladder is cleanly MONOTONE: -1.38, -0.69, -0.07, +0.53, +1.66 bps
  6/6 factors kept their IC sign on a chronological holdout never used for
  selection (IS +0.0466 -> OOS +0.0436 for reversal_1h)
This is not noise. Short-horizon cross-sectional reversal in crypto is real.

WHY IT IS STILL WORTHLESS TO US -- the breakeven cost:
  reversal_1h earns +2.73 bps GROSS per rebalance but turns over 3.02 of the
  book each hour, so it breaks even at 0.90 bps per unit turnover. Binance
  spot charges ~10 bps PER SIDE at VIP0. The signal is ~11x too small to pay
  for the spread it must cross to capture it.
  Horizon sweep (1h..48h) does not rescue it: reversal_1h's turnover stays
  ~3.0 at every holding period while its IC decays, so breakeven peaks at
  2.29 bps and then falls.
INTERPRETATION: this alpha IS the liquidity-provision premium. It can only be
harvested by POSTING liquidity (maker), never by taking it. That is a
statement about execution, not about models.

REJECTED -- momentum_168h. It showed the best breakeven in the whole horizon
sweep (11.05 bps at 48h) and it is a TAIL ARTIFACT: quintile profile
-2.235, -1.226, -0.210, -0.417, +6.694, i.e. flat middle with one jump into
the top bucket carrying 80% of the spread. Two guards now catch this
automatically -- decile monotonicity AND tail_dominance (share of the spread
in the single largest bucket step). Rank correlation ALONE was not enough:
the bad profile scores 0.9 on 5-bucket Spearman, which is why the guard is
a pair, not one number.

METHOD CAVEATS, standing:
  - SURVIVORSHIP: universe is pairs listed today; delisted losers are absent.
    Long/short is less exposed than long-only but every figure is an upper
    bound.
  - The horizon sweep is 42 cells (6 factors x 7 horizons). Anything picked
    from it post-hoc is a hypothesis, not a result. reversal_1h is exempt
    because it was pre-declared and confirmed on the holdout.

WHAT THIS CHANGES: the roadmap ordering. Cost/execution was #3; it is now #1,
because we no longer lack a signal -- we lack a way to trade one we have.
The open question is whether a MAKER-ONLY implementation (posting limit
orders, accepting partial and missed fills) can capture any part of a
0.9 bps/turnover premium. Adverse selection is the obvious threat: the fills
you get are disproportionately the ones you did not want.

## EXECUTION STUDY (2026-07-30) — turnover reduction does NOT rescue reversal_1h
Follow-on to the cross-sectional finding. The question was whether the
0.90 bps/turnover premium can be reached by executing more cheaply.

FEE REALITY, verified against the live exchange (ccxt market metadata):
  Binance SPOT    maker 10.00 bps   taker 10.00 bps
  Binance FUTURES maker  2.00 bps   taker  5.00 bps
Maker execution on SPOT is nearly pointless here: maker and taker fees are
identical, so posting instead of taking saves only the spread (~1-2 bps), not
the fee. The real 5x cost reduction is the venue, not the order type.

At 2 bps futures maker fee, reversal_1h still costs 3.02 turnover x 2 bps =
6.04 bps against 2.73 bps gross -- 2.2x short. So the binding constraint is
TURNOVER, and the test is whether damping it closes a 2.2x gap.

METHOD: swept 20 configurations -- signal smoothing (EWMA span 1/3/6/12/24)
x leg-membership buffering (none/0.3/0.4/0.5 exit quantile). Both are
standard, both reliably cut turnover. Every configuration scored on BOTH the
in-sample half and the holdout.

RESULT -- and this is the whole point of splitting first:
  12/20 configurations are PROFITABLE IN SAMPLE at 2 bps.
  0/20 survive on the HOLDOUT.
  Best in-sample: smooth 6 / buffer 0.5 -> +0.722 bps per rebalance.
  Same configuration on holdout: -0.312 bps.
Had the parameter search been run on the full sample, or had only the winner
been carried forward, it would have produced a profitable-looking strategy
that loses money. The sweep IS the overfit; the holdout is the only number
that meant anything.

WHY IT FAILS: the reversal alpha lives at the very shortest horizon and
decays fast. Smoothing cuts turnover but discards exactly the component that
carries the edge -- holdout gross falls 2.733 -> 0.106 as smoothing goes
1 -> 24, and turns NEGATIVE beyond that. There is no setting where enough
alpha survives to pay even 2 bps.

VERDICT: reversal_1h is real (IC +0.0436, t=+20.9, monotone, holdout-
confirmed) and is NOT harvestable by this platform at any tested combination
of venue, order type and turnover damping. Recorded as a closed question, not
an open one.

NOT TESTED, and honestly out of reach without new data: true maker-fill
modelling. Estimating what fraction of posted orders actually fill, and how
adversely selected those fills are, needs order-book or trade-level data the
platform does not collect. Assuming clean fills would manufacture a fake
edge, so the study was not faked -- it was scoped out and said so.

TOOLING: turnover_frontier() + format_frontier() in cross_section.py report
in-sample and holdout for every configuration by construction, and print an
explicit warning when in-sample winners have no holdout survivors. 529 tests.

## FUNDING-RATE RESEARCH (2026-07-31) — declared hypothesis REJECTED
Fourth research pass, and the first using an input that is not in the price
series. Built a 117-perpetual cache (13,149 hourly bars x 1.5 years) with
matched 8-hourly funding history, forward-filled strictly causally onto the
hourly grid.

PRE-DECLARED HYPOTHESIS (written into the module before any evaluation):
persistently positive funding marks crowded long positioning, and crowded
positioning underperforms -- so the signal is NEGATED funding. The direction
was declared up front precisely so it could not be chosen after the fact.

RESULT -- rejected, consistently:
              factor            IS t-stat   OOS t-stat   sign
  funding_carry_24h                 -1.83        -3.24   consistently NEGATIVE
  funding_carry_72h                 -1.74        -3.53   consistently NEGATIVE
  funding_level                     -1.31        -3.34   consistently NEGATIVE
  funding_zscore_7d                 +0.63        -0.21   flips -> noise
Under the declared sign convention a negative IC means the hypothesis is
backwards: high-funding contracts OUTPERFORMED in this sample. No funding
factor passed the edge gate on either half.

CONTROL -- funding lost to the price factors it was meant to beat:
  in-sample  best |t|: funding 1.8  vs price 6.5
  holdout    best |t|: funding 3.5  vs price 5.8
The new data source did not add information the price series lacked.

THE STRUCTURAL CLAIM DID HOLD, and is worth keeping. Funding factors turn
over 0.35-1.88 per rebalance against reversal's 2.94-2.98 -- up to 8x lower,
exactly as an 8-hourly input should. The architecture reasoning was right;
there was simply no alpha to carry through it.

WHAT IS DELIBERATELY NOT CLAIMED. Reversing the sign would turn the holdout
into roughly +18.7 bps per 8h rebalance at 0.35 turnover -- a breakeven near
54 bps, which would be a spectacular strategy. It is not claimed, for three
reasons, any one of which is sufficient:
  1. SURVIVORSHIP. The universe is contracts listed today. Assets that
     sustained high funding AND survived are winners by construction, and
     69.5% price coverage means many contracts listed mid-sample. "High
     funding predicts outperformance" is the exact result this bias
     manufactures.
  2. INSTABILITY. Gross magnitude differs ~5x between halves (-3.68 vs
     -18.02 bps for carry_24h). A real effect of this size should not.
  3. IT WOULD BE THE SEARCH I FORBADE. The sign was pre-declared in the
     module docstring specifically so it could not be flipped after seeing
     the data. Flipping it now doubles the search and halves the meaning.
The reversed direction is a NEW hypothesis. Testing it honestly requires a
survivorship-free universe with point-in-time listings including delisted
contracts -- a data problem, not a modelling one.

BUG CAUGHT BY THE TESTS: trailing_std returns ~6.5e-19 rather than 0 for a
constant series, which passed a `std > 0` guard and produced order-1 garbage
z-scores. Because those scores feed a cross-sectional RANKING, one
numerically degenerate contract could have landed at the top or bottom of the
book on pure float residue. Guarded with an explicit _MIN_STD floor.

RUNNING TALLY OF HONEST NULLS: live 5-min signal (no edge), cross-sectional
reversal (real, uneconomic), turnover damping (in-sample only), funding
(hypothesis rejected). Four negative results, each costing hours rather than
months. Nothing tradeable has been found. That is the state of the evidence.

## HORIZON LADDER (2026-07-31) — every level measured; none passes
Operator direction: the platform should mix ALL levels, not live in one band.
Correct critique -- every prior pass sat inside 1h-48h, exactly where cost per
rebalance dominates. Closed by building the full ladder:

  hourly cache  2y x 117 symbols: 1h, 4h, 12h, 24h, 72h
  daily cache   5y x  53 symbols: 1d, 3d, 7d, 14d, 30d   (fetched fresh)
  + a signal-level multi-factor blend per band, holdout-only

TWO COMPARABILITY FIXES REQUIRED FIRST, both of which would otherwise have
manufactured results out of arithmetic:
  - bps-per-rebalance is meaningless across horizons (2.7bps hourly and
    2.7bps weekly differ by three orders of magnitude annually). Everything
    is now annualised via an explicit bars_per_year parameter.
  - the harness hardcoded hours-per-year; on daily bars that overstates every
    Sharpe by sqrt(24). Now a required, tested parameter.
  - rungs with < 60 non-overlapping rebalances are marked UNJUDGED (a 30-day
    hold over 5y is ~24 observations; no statistic on 24 points may reach a
    capital decision).

RESULT: 60 (factor, horizon) cells scored on both halves. 5 rungs were
positive in both halves with a stable IC sign -- and EVERY ONE fails the full
guard battery, in the most instructive way possible:

  rung                    OOS ann%   OOS IC      t    tail   DSR
  momentum_168h @ 12h      +151.8   -0.0017   -0.23   0.50  0.79
  momentum_168h @ 24h      +125.0   -0.0001   -0.01   0.78  0.58
  momentum_24h  @  1d       +65.9   -0.0154   -1.71   0.80  0.36
  volume_surge  @  3d       +60.1   -0.0054   -0.48   0.62  0.39
  reversal_1h   @  7d        +5.6   +0.0021   +0.10   1.84  0.03

Read that table carefully: triple-digit annual returns sitting next to ZERO
or NEGATIVE information coefficients. Money without ranking skill means the
P&L comes from a handful of extreme names, not from the factor -- confirmed
by tail dominance 0.50-1.84. On a survivorship-biased universe (the daily
cache DOUBLY so: 5 years of history is only possessed by 5-year survivors),
this is precisely the shape of fake edge the guards were built to refuse. A
plain equity-curve backtest would have showed +65.9%/yr at Sharpe 1.2 and
called it a strategy.

BLEND: mean pairwise correlation of the six factors is ~0.0 in both bands --
genuinely diversifiable -- but there is nothing to diversify: blend Sharpe
0.26/0.49 vs best component 1.52/1.22, DSR 0.03/0.08. "Blend does NOT beat
its best component" in both bands.

NAMING CAVEAT recorded: factor lookbacks are in BARS, so on the daily cache
"momentum_24h" is 24-DAY momentum, "reversal_1h" is 1-day reversal. The
labels lie on daily bars; the arithmetic does not.

VERDICT: the gap is closed -- the platform measures every level from 1 hour
to 30 days, annualised and comparable, plus blends. FIFTH honest null.
Nothing at any level passes the edge gate on available data. The binding
constraints by band: costs at intraday, statistics (n and survivorship) at
multi-week.

## PRE-REGISTRATION (2026-07-31, before the survivorship-free data was fetched)
PR #62 rejected "fade high funding" and observed that the REVERSED direction
looked spectacular (+18.7 bps per 8h rebalance on the holdout) but refused to
claim it because the universe was survivorship-biased. Discovery: Binance's
public archive (data.binance.vision) retains DELISTED contracts -- 788 USDT
perpetuals ever vs 726 trading today -- with full funding and kline history.
The survivorship hole can therefore be closed for free.

Registered BEFORE evaluation, so nothing below can be adjusted after seeing
results:

  H1 (primary): on the survivorship-FREE universe (all contracts ever listed,
      including delisted), the factor +funding_carry_24h (HIGH funding
      predicts relative OUTPERFORMANCE -- the reverse of the PR #62 prior)
      evaluated at a 24h horizon on an 8h grid, 2 bps cost, 40% chronological
      holdout, judged by the standard gate (|IC| > 0.01, |t| > 3, net > 0,
      DSR > 0.95, monotone, tail < 0.5) on BOTH halves.
  H2 (the survivorship measurement): the same factor evaluated on the
      survivors-only subset of the SAME period must show a MORE positive
      IC than on the full universe. The difference IS the survivorship bias,
      measured directly.
  PREDICTION, stated now: if PR #62's reversal was a survivorship artifact,
      H1 fails (effect attenuates toward zero when the delisted losers are
      restored) and H2 shows a material gap. If H1 passes in full, the effect
      is real and was hiding behind the wrong prior.
  Funding rates are normalised to per-8h equivalents (the archive reveals
      some contracts fund every 4h; per-stamp rates are not comparable
      without this).
  Everything else in the run is EXPLORATORY and will be labelled as such.

## SURVIVORSHIP-FREE FUNDING TEST (2026-07-31) — registered H1 REJECTED;
## the registered mechanism was wrong; the real structure is a regime flip
Executed exactly as pre-registered (commit ed277cb, before the data existed).
Data: Binance public archive -- 2,190 8h stamps x 706 contracts, including
129 DELISTED contracts no prior universe contained. Funding normalised to
per-8h equivalents (the archive revealed some contracts fund 4-hourly; raw
per-stamp rates are not cross-sectionally comparable).

H1 (registered: +funding_carry_24h, full universe, both halves): REJECTED.
  in-sample IC -0.0092 (t -1.77), holdout IC +0.0270 (t +3.46). The sign
  FLIPS between halves; the gate fails, as it must for a sign-unstable
  factor.

H2 (registered: survivorship measurement): bias = -0.0004 IC, i.e. ZERO.
  The registered prediction -- that PR #62's reversal was a survivorship
  artifact -- is WRONG. Restoring 129 delisted contracts moved the IC by
  less than half a basis point of rank correlation. PR #62's refusal to
  claim was right in discipline but wrong about the mechanism. Survivorship,
  repeatedly flagged as the suspect, is measured and small for this
  long/short factor on this window.

WHAT THE STRUCTURE ACTUALLY IS -- a coherent regime flip, quarterly IC of
+funding_carry_24h:
  2024-Q3 -0.0124 | 2024-Q4 -0.0293 | 2025-Q1 -0.0130   (fade-crowding works)
  2025-Q2 +0.0009                                        (transition)
  2025-Q3 +0.0096 | 2025-Q4 +0.0291 | 2026-Q1 +0.0292 | 2026-Q2 +0.0284
Four consecutive positive quarters. PR #62's 1.5y window sat inside the
positive regime, which is why both its halves agreed; the 2y window straddles
the flip, which is why the registered gate failed. Both results are correct;
the factor's sign is regime-dependent on ~year scales.

WHAT THE REGISTRATION PREVENTED: the holdout-only numbers were spectacular --
+146.9% annualised for the primary, +207.2% for a carry variant, DSR 1.00,
monotone ladders. An undisciplined process would be calling that a discovery
today. It is one regime's half-sample, and the other half has the opposite
sign.

STATUS OF THE REGIME HYPOTHESIS ("since mid-2025, high-funding contracts
outperform"): generated BY this data, therefore untestable ON this data. The
only honest adjudicator is FUTURE data. It is registered here as a
walk-forward watch item -- the paper soak can track the live quarterly IC of
+funding_carry_24h at zero cost and zero capital, and four more positive
quarters on unseen data would justify a real proposal. No claim is made now.

SIXTH consecutive honest verdict. The evidence tally: no tradeable edge
found anywhere; one real-but-uneconomic factor (reversal); one measured
near-zero bias (survivorship on funding long/short); one regime-dependent
factor identified and parked for walk-forward adjudication.

## WALK-FORWARD WATCH LIVE (2026-07-31) — the adjudicator is now a platform component
The regime hypothesis parked in PR #64 needed a judge that cannot be leaned
on. Built (PR #65): FundingWatchService + factor_watch table (migration
0008) + /api/v1/research/funding-watch + a "Walk-Forward Watch" card on the
ML Models page.

Design properties, each pinned by a test:
  - POINT-IN-TIME BY CONSTRUCTION: each cycle observes whatever perpetuals
    are TRADING at that moment; a stamp's IC is computed once, when its 24h
    forward window closes, and INSERTED IMMUTABLY. Recomputing history from
    a later listing set would quietly reintroduce survivorship bias into the
    adjudication record itself -- the test plants a delisting and asserts
    the stored rows do not move.
  - CAUSAL: the factor at stamp t uses funding up to t only (a planted
    future-only relationship scores ~0); open windows are never scored.
  - REGISTRATION BOUNDARY ENFORCED IN CODE: stamps before 2026-07-01 (data
    the study saw) are refused at insert time. The registered bar -- FOUR
    consecutive positive calendar quarters on unseen data -- is restated in
    the API response, and the UI card is deliberately undramatic: purple,
    "registered hypothesis -- not a strategy", verdict text verbatim from
    the API.
  - Funding normalised to per-8h equivalents live, same as the study.
The watch has no opinion; it counts. Nothing in it places orders or touches
account state.

## RISK METRICS WRITER (2026-07-31) — breaker visible, vacuous rules re-armed
The circuit breaker always worked; nothing reported it. risk_metrics had
readers and NO writer (0 rows ever), so the dashboard showed a false green
until PR #56 made it an honest grey UNKNOWN. And the quieter, worse defect:
update_returns()/update_position() had no callers, so the risk engine's
return history was permanently empty -- VaR/CVaR computed to 0.0
STRUCTURALLY, and MaxVaRRule + MaxCorrelationRule approved every order
without checking anything.

Built (PR #66): RiskMetricsWriter in the worker. Every 120s (first write
immediate) it: syncs the engine's position map to the broker's book
(including removing closed positions, so stale exposure cannot inflate VaR
forever); feeds ~200 1m returns per held symbol from stored bars; runs
check_portfolio_risk() + new RiskManagerService.extended_metrics() (VaR/CVaR
99%, volatility, correlation max); and inserts a risk_metrics row with the
breaker state string and failed-rule names in details JSONB.

THE TEST THAT MATTERS: with a tiny VaR cap, MaxVaRRule passes VACUOUSLY
before the first writer cycle (VaR 0.0) and REJECTS after it. The pre-trade
guards are armed again, and that property is pinned in CI
(test_the_vacuous_rules_are_re_armed).

HONESTY CARRIED THROUGH: an unsupported metric is NULL end to end -- writer
writes NULL, API serves null, UI renders a grey em-dash. Beta has NO
producer, so the tile was DELETED rather than shipped as 0.00, and the
column stays NULL by explicit comment. The breaker card now prefers the
engine's own state string (closed/open/half_open), keeps UNKNOWN for
absence, and shows failed rule names as the reason line.

No migration needed: the table has existed since 0001; it was simply never
written.

## FIRST CONVICTION TRADE (2026-08-01 18:49Z) + breaker-card fix (PR #67)
MILESTONE, recorded with its honest frame: at 18:49Z the platform executed
its FIRST conviction-path trade ever -- a rebal-tagged buy of 0.003835
ETH/USDT (~$7), meaning the calibrated edge-margin gate (p_long - p_short >=
0.10) was cleared for the first time in ~90k live predictions. Order-tag mix
over the surrounding 36h: 130 explore vs 3 rebal. The optimizer sized ETH at
the 10% cap; with the existing exploration lot the position is ~$10.05 =
10.1% of equity, and effective_positions = 1.00 (the rest of the book is
dust).

NOT edge evidence: one gate-clearing signal is one observation. What it IS:
proof the conviction path works end to end in production (signal -> gate ->
optimizer -> risk -> execution -> persistence), which had never been
exercised outside drills.

THE RISK SYSTEM RESPONDED CORRECTLY AND VISIBLY: from 18:50Z every risk row
shows MaxPositionSize (max_weight 0.101 > 0.10, price drift on a cap-sized
position) and MaxConcentration (a one-position book) failing -- 66 rows and
counting. These pre-trade rules now REJECT further ETH adds; they do not
trim existing positions (by design -- caps gate entries, the liquidation
manager handles exits).

BUG FIXED (mine, introduced in PR #66): the API stuffed failing-rule names
into circuit_breaker_reason, so the dashboard card read "CLOSED -- all
systems operating normally" AND "Reason: failed rules: ..." in the same
breath. Two unrelated facts on one line. Now: circuit_breaker_reason is
reserved for WHY THE BREAKER TRIPPED (populated only when state != closed);
failing rules are their own field end to end (schema failing_rules -> UI
amber "Pre-trade rules failing" notice on the Risk page, with an explicit
note that they gate new orders and neither trip the breaker nor close
positions). Regression test pins the exact Aug 1 shape: closed breaker +
failed rules -> reason must be None.

## PRE-REGISTRATION (2026-08-02, before any gauntlet run) — auditing the
## "EDGE STABLE, Sharpe 1.17" daily-harness claim
The claim under audit, recovered verbatim from the production backtest_jobs
record: edge_harness on AAPL/MSFT/SPY, 2018-01-01..2025-12-31, 5 bps -> AAPL
pass (Sharpe 1.03), SPY pass (Sharpe 1.20, hit 64.5%), MSFT fail; portfolio
+52.2% OOS, Sharpe 1.166, verdict "EDGE STABLE across 2/3 symbols". Graded
2026-07-23, BEFORE the deflated Sharpe, tail guards, beta controls,
survivorship measurement and pre-registration existed. Its OOS window (last
30% of bars, ~2023-07..2025-12) is a strong bull era, and its universe is
three hand-picked mega-cap survivors.

Registered now, unchangeable after results are seen:

UNIVERSE (fixed): the original three plus 61 liquid US names/ETFs across
sectors: AAPL MSFT SPY QQQ AMZN GOOGL META NVDA TSLA JPM BAC WFC GS XOM CVX
COP JNJ PFE MRK UNH ABBV LLY PG KO PEP WMT COST HD MCD NKE DIS NFLX CRM ORCL
ADBE INTC AMD QCOM CSCO IBM T VZ CMCSA BA CAT GE MMM HON UPS FDX LMT RTX DE
F GM V MA PYPL AXP BRK-B IWM DIA XLF XLE GLD. Survivorship caveat: these are
today's liquid names, i.e. survivors -> every pass statistic is an UPPER
BOUND, stated in all outputs.

LEGS (claim survives only if ALL pass):
  R1 REPRODUCTION: identical 3-symbol config must reproduce the pass/fail
     pattern (AAPL+SPY pass, MSFT fail); Sharpes within +/-0.15 (Yahoo data
     revisions tolerated, pattern is not).
  H1 BREADTH: over the fixed universe (2018-01-01..2025-12-31, 5 bps,
     harness's own gate), pass fraction among judged symbols must be >=
     0.60 -- the harness's OWN MIN_SYMBOL_PASS_FRAC, now applied to a
     universe that was not chosen after the fact.
  H2 BETA CONTROL: strategy OOS Sharpe must beat buy-and-hold Sharpe on the
     SAME symbols over the SAME OOS periods for a MAJORITY of judged
     symbols. If not, the "edge" is market beta wearing a model.
  H3 UNSEEN ERA: OOS periods dated strictly AFTER 2025-12-31 (data the
     original run never saw, ~7 months). Pooled equal-weight per-period net
     return across the universe must be positive.
  DSR: equal-weight portfolio per-period Sharpe over the broad universe,
     deflated for n_trials = 24 (declared: Stage-2 feature/threshold/horizon
     decisions plus symbol picks), must exceed 0.95.

PREDICTION, stated now: R1 reproduces; H1 fails well below 0.60; H2 fails
(SPY-era hit rates are beta); the claim is DEMOTED from "edge stable" to
"selection + beta, honestly retired". If instead all five legs pass, the
strategy graduates to a registered walk-forward watch before any capital
discussion.

## GAUNTLET RESULT (2026-08-02) — the "EDGE STABLE, Sharpe 1.17" claim is RETIRED
Executed exactly as registered (commit cc85ae5, before the run). One prose
correction: the registration text said "the original three plus 61" names;
the registered LIST (authoritative, pinned by test) contains 65. The claim
FAILED four of five legs:

  R1 REPRODUCTION: FAIL. On dividend-adjusted closes MSFT flips to PASS
     (Sharpe 0.81, stability 0.75) where the original recorded FAIL (0.47,
     0.50); AAPL/SPY magnitudes reproduce. The pass/fail pattern is not
     stable under routine data adjustment -- the gate verdict sits on a
     knife edge, which is itself disqualifying for a capital decision.
  H1 BREADTH: FAIL. 37/65 judged symbols pass (56.9%) vs the harness's own
     required 60% -- and that is an UPPER BOUND (survivor universe).
  H2 BETA CONTROL: FAIL, and this is the mechanism. Only 22/65 (33.8%)
     beat costless buy-and-hold on the same OOS periods. The flagship
     exhibits are the exposé: SPY strategy 1.20 vs B&H 1.35; GLD 1.71 vs
     2.33; GE 1.95 vs 2.02; JPM 1.53 vs 1.56. The model is a long-tilted
     beta proxy that captures MOST of buy-and-hold while paying costs to
     do it. Two-thirds of "edge" symbols made less than doing nothing.
  H3 UNSEEN ERA: PASS as registered (+41.7 bps/period, t=+2.41, 28
     portfolio periods after 2025-12-31) -- but the leg as registered has
     no beta control, and given H2, its positivity is explained by a rising
     market, not skill. Recorded as passed; not repurposed as evidence.
  DSR: FAIL. Pooled portfolio Sharpe +2.11 (itself beta-inflated) deflates
     to 0.891 over 120 periods at the declared 24 trials -- below the 0.95
     bar.

VERDICT: the 2026-07-23 "EDGE STABLE across 2/3 symbols" job result is
DEMOTED from evidence to artifact: selection (3 mega-cap survivors) + beta
(a bull-era OOS window) + a knife-edge gate. The edge.py HARNESS remains a
valid research tool -- non-overlap, net-of-cost and stability logic are
sound; what is retired is this specific claim. Without the gauntlet, this
was the one number on the books an operator might have funded, and it
underperforms buy-and-hold on its own flagship symbol.

CONSEQUENCE: the platform now carries ZERO unaudited positive performance
claims. Every number either passed the modern gauntlet, is labelled
exploratory, or sits in a registered walk-forward watch. Seventh
consecutive honest verdict.

METHOD NOTE for future daily-harness work: any equity daily-bar result must
include the H2 beta control BY DEFAULT -- long-capable strategies evaluated
on 2023-2025 US equities are measuring the era unless proven otherwise.

## PRE-REGISTRATION (2026-08-02) — three additional walk-forward watches +
## the July backfill of the funding watch
Direction accepted by the operator: profit cannot come from the current
(measured no-edge) live models, so the pipeline of registered, free options
on future edge is widened. Registered BEFORE any of these factors is
computed on live data:

NEW WATCHED FACTORS -- all on the existing 8h perpetual grid, horizon 3
stamps (24h), cross-sectional Spearman IC vs demeaned forward return, same
integrity properties as the funding watch (append-only, causal, open windows
never scored). Signs and definitions are fixed NOW; provenance is the spot
hourly study (PR #60) where each kept its IC sign 6/6 across chronological
halves. That study was a different venue and grid, so these are HYPOTHESES,
not results; the watch adjudicates.
  reversal_8h_minus    signal = -(close_t/close_{t-1} - 1)      [fade the last 8h move]
  momentum_72h_minus   signal = -(close_t/close_{t-9} - 1)      [fade the 3-day move]
  low_vol_7d_minus     signal = -std(1-stamp returns, 21 stamps) [prefer quiet contracts]
UNSEEN BOUNDARY for the three new factors: 2026-08-02 (this registration).
Stamps before it never count toward their adjudication. Bar per factor,
identical to the funding watch: four consecutive positive calendar quarters
of mean IC on unseen data before any trading proposal may cite it.

JULY BACKFILL of funding_carry_24h_plus (unchanged factor, unchanged bar):
its registered unseen boundary is 2026-07-01 but the live watch only reaches
~10 days back, leaving Jul 1-21 unrecorded. Those stamps are backfilled ONCE
from the data.binance.vision monthly archive (verified published for
2026-07), which retains contracts delisted mid-month -- i.e. the
survivorship-safe source. Existing rows are never touched (append-only
holds); the backfilled rows are identifiable by their resolved_at date.

BENCHMARK BOOK (accepted alongside): a do-nothing paper benchmark -- equal
weight BTC/USDT, ETH/USDT, SPY, inception at this registration -- computed
from stored bars and drawn on the equity chart. It is the live bar any
future strategy must beat; the platform's own gauntlet showed buy-and-hold
beating its models on 2/3 of symbols, and that fact belongs on the
dashboard, not only in the runbook.

## PIPELINE WIDENED + BENCHMARK LIVE (2026-08-02) — PRs #69, #70
Executed per registration 74c5b07, in response to the operator's correct
observation that the platform makes no profit. The live models are measured
no-edge; profit can only come from the registered-hypothesis pipeline, so
the pipeline was widened from one watched factor to four, the funding
record was completed, and the do-nothing bar went on the dashboard.

WATCH REGISTRY LIVE: reversal_8h_minus, momentum_72h_minus, low_vol_7d_minus
join funding_carry_24h_plus on the 8h perp grid -- one fetch, per-factor
unseen boundaries (the three new: 2026-08-02), same append-only/causal
contract, all pinned by tests including one proving the same July dataset is
accepted for the funding factor and refused for the new three. First
post-deploy cycle logged new_observations=0, which is CORRECT: the new
factors' first stamps cannot resolve until their 24h windows close.

JULY BACKFILL EXECUTED: 63 observations for Jul 1-21 inserted from the
data.binance.vision monthly archive (800 contracts incl. delisted), existing
rows untouched, provenance visible in resolved_at. The funding factor's Q3
record is now COMPLETE: n=95 (Jul 1 - Aug 1), mean IC +0.0124, t=+1.52,
60/95 positive -- the regime leans positive on unseen data but is not yet
significant, and no one may call it before the quarter closes.

BENCHMARK LIVE: /portfolio/benchmark serves equal-weight BTC/ETH/SPY from
inception 2026-08-02, normalised to the account's inception equity, sampled
at snapshot times, drawn dashed on the equity chart. Found and fixed in the
process (PR #70): SPY was never in live ingestion -- its bars were a stale
one-off backfill -- so the registered three-asset benchmark would have
silently degraded to a crypto pair. SPY now streams; the close-lookup window
is 10 days so weekends/holidays forward-fill instead of dropping a leg.

THE SCOREBOARD THE OPERATOR NOW HAS: four independent registered shots at a
promotable edge, each adjudicated by unseen data at zero cost; a complete
Q3 funding record; and a dashboard where every strategy is judged, live,
against doing nothing. Profit remains zero because edge remains unproven --
and the machinery to change that honestly is now running at full width.

## PRE-REGISTRATION (2026-08-07) — the pipeline doubles again: four more watches
Operator direction, recorded precisely: asked what would double the
platform's return in a day, the measured answer was "nothing legitimate --
leverage at doubling scale is ruin in 2-13 median days" (see the 2026-08-06
analysis). The only thing this platform may double in a day is registered
hypotheses. Pipeline goes 4 -> 8. Registered BEFORE implementation; unseen
boundary for all four: 2026-08-07.

NEW WATCHED FACTORS -- same 8h perp grid, 24h horizon, same integrity
contract (append-only, causal, per-factor boundary, open windows unscored):
  momentum_24h_minus   signal = -(close_t/close_{t-3} - 1)
      provenance: spot hourly study (PR #60), sign held both halves.
  volume_surge_minus   signal = -(volume_t / trailing_mean(volume, 21))
      provenance: spot study volume_surge IC negative in BOTH halves ->
      fading volume spikes scored positive. Requires volume in the watch
      fetch (available from the same OHLCV call, no extra requests).
  funding_delta_minus  signal = -(funding_t - trailing_mean(funding, 21))
      provenance: economic prior only, stated now -- RISING funding marks
      BUILDING crowding; fade it. No prior study; weakest provenance of the
      eight, labelled as such.
  blend_registered_v1  signal = mean of the FOUR 2026-07/08-registered
      factors' cross-sectional z-scores (funding_carry_24h_plus,
      reversal_8h_minus, momentum_72h_minus, low_vol_7d_minus), weights
      equal and FROZEN at this registration.
      provenance: diversification -- a blend of weakly correlated signals
      can beat its best component; the meta-hypothesis is that the
      registered book is better than any single registered factor.
Bar per factor, unchanged: four consecutive positive calendar quarters of
mean IC on data after its own boundary. Nothing here trades.

## LEARNING METRICS LIVE (2026-08-08) — the loop now measures itself
The platform retrains, gates and promotes -- but nothing measured whether
promotions IMPROVE the served signal on live data. Built (PR #73):
services/continuous_learning/learning_metrics.py + /models/learning-metrics
+ a Learning Metrics card on the ML Models page.

WHAT IT MEASURES, per the repo's hard-won rules:
- Live IC by MODEL ERA. predictions.model_version is constant (ensemble
  stamps v1 always), so the only honest version-over-version comparison is
  time-segmented by promotion dates from model_metadata; any future retrain
  opens a new era automatically. This is the "is retraining learning or
  churning?" instrument.
- Daily de-overlapped IC + sign agreement with the PR #59 zero-return guard
  baked in structurally (unchanged bars excluded from agreement).
- Abstention calibration: stored confidence is p(flat) for gated
  predictions, so Brier + reliability bins measure exactly that event,
  restricted to gated rows (a test proves directional rows cannot pollute
  the bins).
- De-overlap AT FETCH TIME: minute %% 5 == 0 sampling gives independent
  observations and a 5x cheaper scan in one stroke.
- None is never 0: thin days and empty eras serve null end to end.

## FIRST LEARNING-METRICS READING (2026-08-08) — two significant findings
The instrument's first pass over 28,516 de-overlapped live observations:

FINDING 1 -- PROMOTIONS ARE NOT IMPROVING THE LIVE SIGNAL. Era table:
  era 3 (v3, Jul 24-28):        live IC +0.0034  (t=+0.26)  -- nothing
  era 4 (v4, Jul 28-Aug 08):    live IC -0.0255  (t=-5.07)  -- significantly
                                                                NEGATIVE
The v4 ensemble passed the champion/challenger VALIDATION gate and its live
era shows significantly anti-predictive expected_return over 12 days. Honest
alternatives before blaming the models: the eras straddle different market
weeks, so regime is a candidate explanation alongside genuine degradation.
Either way the conclusion stands: VALIDATION-fold accuracy, the promotion
criterion, is NOT aligned with live outcome quality. The gate is promoting
on a number that does not transfer.

FINDING 2 -- p(flat) IS BADLY OVER-STATED AT HIGH CONFIDENCE. Reliability:
  [0.0,0.4): stated 36.0% -> realized 40.8%   (+4.8pp)
  [0.4,0.5): stated 45.6% -> realized 44.0%   (-1.6pp)
  [0.5,0.6): stated 55.0% -> realized 35.4%   (-19.5pp)
  [0.6,1.0): stated 62.9% -> realized 27.0%   (-35.9pp)
When the gate is most sure the market will stay flat, it moves MOST often --
realized flat rate FALLS as stated p(flat) rises. The isotonic/Platt
calibration fitted at training time does not hold on live distributions.
This is a well-defined event (inside/outside the +/-5bp deadband), immune to
the sign(0) artifact, over thousands of observations per bin.

NO CAPITAL CONSEQUENCE TODAY: the no-edge verdict already stands, exposure
is ~9%, and the risk stack caps everything. WHAT THIS ENABLES, as future
REGISTERED work (not reactive tinkering on 12 days of data):
  (a) a live-IC criterion alongside validation accuracy in the champion
      gate, so promotion requires evidence that transfers;
  (b) recalibrating p(flat) on live resolved outcomes rather than training
      folds -- the reliability table above is the training data for it.
Both would be pre-registered with success criteria before implementation.

## PRE-REGISTRATION (2026-08-09) — closing the live learning loop
Operator direction: improve the platform's ML ability "a thousand fold".
Honest reading, grounded in the 2026-08-08 learning-metrics findings: the
multiplier on learning is not model size, it is LEARNING FROM THE RIGHT
SIGNAL. Two registered changes, success criteria fixed before a line of code:

PHASE 1 (built now) -- LIVE p(flat) RECALIBRATION. The abstention gate's
stated flat-probability is refuted by live outcomes (-35.9pp in the top
bin). A post-hoc isotonic layer, FITTED ON THE PLATFORM'S OWN RESOLVED LIVE
OUTCOMES (stated p(flat) vs realized inside-deadband frequency, last 14
days, de-overlapped, >= 2,000 pairs required), recalibrates the ensemble's
flat probability at serve time; long/short mass rescales proportionally so
probabilities stay a simplex. Refit every 6h; fitted at boot from stored
data (deterministic, no new persistence). Env kill switch:
LIVE_CALIBRATION_ENABLED (operator-controlled).
  SUCCESS CRITERIA, judged by the ALREADY-DEPLOYED learning metrics after 7
  days of post-deploy data: (i) mean daily brier_flat improves by >= 0.01
  vs the 14 days pre-deploy; (ii) the top calibration bin's |gap| falls
  below 15pp (from 35.9pp). MISS -> the layer is disabled and the miss is
  recorded. Behavioral note, stated in advance: better-calibrated (lower)
  p(flat) mechanically widens directional margins, so the conviction gate
  may fire more often; the risk stack is unchanged and caps all of it.

PHASE 2 (registered, NOT built today) -- LIVE-TRANSFER PROMOTION GATE.
Root cause of Finding 1 is that challengers are judged on validation folds
that do not transfer. The honest fix needs served FEATURE VECTORS persisted
(features_used is currently NULL on every row) so future challengers can be
scored on true live feature->outcome pairs before promotion. Registered
plan: persist compact features on de-overlapped predictions, accumulate >=
14 days, then add a champion-gate criterion "challenger live-replay IC must
beat champion's era IC". To be pre-registered in detail when phase 1's
verdict is in.
