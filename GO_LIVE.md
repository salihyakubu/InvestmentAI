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
