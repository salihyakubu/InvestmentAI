# Go-Live Runbook

Staged path from "code-correct" to real-money trading. **Do not skip stages.**
Status legend: ✅ verified in dev · ⚙️ you run it · 🔴 hard gate.

Verdict: **paper-trade after Stage 1; do not risk real money until Stage 2 (edge)
and Stage 3 (paper soak) pass.**

---

## Stage 0 — Code correctness ✅ (done)
- 100 tests green, `ruff` clean, CI in place.
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
> Remaining for the operator: create the Railway project, set the env vars above (CLI auth is
> yours), and trigger the deploy.

## Stage 2 — Prove the strategy has edge 🔴
1. **Ingest real history into a CSV** (the harness reads a `close` column; provider is
   auto-selected by symbol, both keyless):
   - Stock/ETF: `PYTHONPATH=. python scripts/fetch_history.py AAPL --start 2015-01-01 --out aapl.csv`
   - Crypto: `PYTHONPATH=. python scripts/fetch_history.py BTC/USDT --start 2021-01-01 --out btc.csv`
2. **Run the edge harness** on each file:
   `PYTHONPATH=. python scripts/validate_model.py aapl.csv`
   (prefix `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib` on macOS; on Linux/Docker
   libgomp1 is already in the image).
3. **Gate:** out-of-sample **hit-rate > 52%** AND **Sharpe > 0.5**. If
   `VERDICT: NO demonstrable edge`, **stop — do not trade.** Notes:
   - The harness now measures **non-overlapping** holding periods, so Sharpe/return
     are not inflated by overlapping forward-return windows (an early version reported
     a fictitious +5588% / Sharpe 2.3 on AAPL; the honest figure is ~+125% / Sharpe ~1.0).
   - Run it **per symbol** and require the edge to be **stable across symbols/periods**,
     not one lucky fit. A single-symbol pass still ignores costs/slippage and tunes the
     label threshold in-sample — treat it as necessary, not sufficient.
   - Self-test: run with no args (zero-drift random walk) — it must report
     **NO demonstrable edge**. No amount of code quality substitutes for a real signal.

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

## Operational glue still required (deferred, tracked)
These are wired in code but need final integration/verification:
- **Risk equity/PnL feed**: have the worker call `risk.sync_account(broker.get_account()["equity"])`
  on a timer (e.g. every 30–60s) and `risk.reset_daily()` at session open. The method is
  tested; the periodic call is deploy glue.
- **Manual order API → live execution** (#11): the `POST /orders` endpoint persists a PENDING
  order; wiring it to the worker's execution needs an async order-intent consumer.
- **Frontend** (#8): login page + prod build + FE↔BE path reconcile (UI is otherwise unusable).
- **ML** (#12): probability calibration; sequence-model leakage fix.
- **Timescale** (#14): re-add hypertables/retention as a guarded follow-up migration.

## Emergency procedures
- **Halt + flatten:** call `emergency_flatten()` (or `halt()` to just stop new orders).
- **Drawdown auto-stop:** the circuit breaker trips at `circuit_breaker_loss_pct`
  (default 7% daily) **once `sync_account()` is feeding it** — verify in Stage 3.
