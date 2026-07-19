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
> Remaining for the operator: create the Railway project, set the env vars above (CLI auth is
> yours), and trigger the deploy.

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
  timer (`services/worker.py`), feeding the drawdown monitor / circuit breaker. Small remainder:
  a `risk.reset_daily()` call at session open (verify during the Stage 3 soak).
- **Manual order API → live execution** (#11): ✅ `POST /orders` publishes an `OrderIntentEvent`
  the execution worker consumes (`_handle_order_intent`); fills sync back to the DB order row.
- **Frontend** (#8): ✅ login/auth gate + production nginx build + FE↔BE path reconcile.
- **ML** (#12): ✅ tree-probability calibration + sequence double-normalisation fix. (Torch
  sequence-model calibration still deferred — gated on torch in the deploy image.)
- **Timescale** (#14): ✅ hypertables + retention as a guarded migration (`0002`, verified on
  real TimescaleDB and no-op on plain Postgres).

## Emergency procedures
- **Halt + flatten:** call `emergency_flatten()` (or `halt()` to just stop new orders).
- **Drawdown auto-stop:** the circuit breaker trips at `circuit_breaker_loss_pct`
  (default 7% daily) **once `sync_account()` is feeding it** — verify in Stage 3.
