# Go-Live Runbook

Staged path from "code-correct" to real-money trading. **Do not skip stages.**
Status legend: ✅ verified in dev · ⚙️ you run it · 🔴 hard gate.

Verdict: **paper-trade after Stage 1; do not risk real money until Stage 2 (edge)
and Stage 3 (paper soak) pass.**

---

## Stage 0 — Code correctness ✅ (done)
- 88 tests green, `ruff` clean, CI in place.
- Migration verified on real Postgres; full pipeline verified over real Redis
  (`scripts/smoke_test_pipeline.py`); models train + predict
  (`scripts/validate_model.py`); auth path verified on real Postgres.

## Stage 1 — Deploy + smoke test on real infra ⚙️
1. 🔴 **Rotate credentials** (the leaked ones are compromised):
   - New Alpaca + Binance API keys (broker dashboards).
   - `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (paper).
   - Strong `JWT_SECRET` (≥32 random chars). In live mode the app refuses to
     start with the default — by design.
   - Put all of these in Railway's secret store, not a file.
2. Deploy (Railway uses `Dockerfile`; release runs `alembic upgrade head`).
   Locally: `docker compose up` then `alembic upgrade head`.
3. Create a login user: `PYTHONPATH=. python scripts/create_user.py you@x.com '<pw>' admin`.
4. **Smoke test** against the deployed Redis:
   `PYTHONPATH=. python scripts/smoke_test_pipeline.py` → must print `SMOKE TEST PASSED`.
5. Sanity: `GET /api/v1/healthz` 200; `POST /api/v1/auth/token` returns a JWT;
   an authed request succeeds.

## Stage 2 — Prove the strategy has edge 🔴
1. Ingest real historical OHLCV for your universe.
2. `PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python scripts/validate_model.py <prices.csv>`
   (on Linux/Docker, libgomp1 is already in the image — no DYLD needed).
3. **Gate:** out-of-sample **hit-rate > 52%** and **Sharpe > ~0.5**, stable across
   periods. If `VERDICT: NO demonstrable edge`, **stop — do not trade.** No amount
   of code quality substitutes for a profitable signal. (On random data it
   correctly reports no edge.)

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
