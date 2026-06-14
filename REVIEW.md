# InvestAI Platform — Engineering Review

_Review date: 2026-06-13 · Scope: full repository (~19k LOC Python + React frontend)_

## Verdict

An **impressively architected prototype that is not yet a working trading system and is
not safe to point at real money.** Individual modules are strong — clean event-driven
boundaries, `Decimal` discipline in the core, real ML scaffolding, a real risk-math
library, a real React UI. But the **end-to-end pipeline does not function**, the **live
ML predictions are currently meaningless**, the **risk controls are not connected to the
order path**, and the **backtest results cannot be trusted**. There are also several
security and operational-safety issues that are urgent independent of everything else.

Roughly: ~80% of a serious platform's *surface area*, ~30% of its *working wiring*. The
hard part that remains — making the pieces talk to each other correctly and safely — is
exactly the part that is missing.

---

## 🔴 URGENT (operational safety)

1. **Real broker credentials were pointed at the LIVE Alpaca endpoint.** `.env` contained
   populated Alpaca/Binance keys with `ALPACA_BASE_URL=https://api.alpaca.markets` (live
   money), despite `TRADING_MODE=paper`. In Alpaca clients the base URL — not the mode
   flag — routes to real money. **Action: rotate all four broker keys** (only the account
   owner can, via the broker dashboards) and keep secrets in Railway's secret store, not a
   file on disk. _(The base URL has been reverted to paper in this commit; see Appendix.)_
2. **JWT secret is the published default** (`change-this-to-a-random-secret-key`). With
   HS256 (symmetric) anyone can forge a token for any user/role — a full auth bypass.
   **Action: set a strong `JWT_SECRET`.** _(A startup guard that fails closed in live mode
   has been added; see Appendix.)_

Good news: `.env` is gitignored and has **never** been committed (`git log --all -- .env`
is empty), so these were not leaked through the repository.

---

## The big picture: one pipeline, severed in several places

Advertised flow: **market data → features → ML prediction → optimize → rebalance → risk
check → execution → broker.** Traced end to end, it is broken in multiple independent
places, each of which alone stops trading:

| Break | Where | Effect |
|---|---|---|
| Risk → Execution uses synthetic order IDs | `services/risk/service.py:171` emits `order_id="rebal-{event_id}-{symbol}"`; `services/execution/service.py:99` does `get_order(order_id)` → not found → returns | **No order ever reaches a broker** via the risk path; the event carries no symbol/side/qty/price. |
| Fills published to the wrong stream | execution publishes `OrderFilledEvent` to `"orders"`; risk subscribes to `"order_fills"` (`services/risk/service.py:147`) | Risk **never sees fills** → equity/positions/drawdown never update from reality. |
| Order API bypasses risk entirely | `api/routers/orders.py:97` — `# TODO: Integrate with execution service and risk checks` | A direct API order runs **zero** risk checks. |
| Risk engine inputs are orphaned | `update_returns` / `update_position` / `update_daily_pnl` / `update_equity` are never called outside tests | VaR/correlation/drawdown run on empty state → they return ~0 and **pass everything**. The risk engine is effectively blind. |

The sophisticated, unit-tested risk library is therefore **disconnected** from the live
system — a well-built engine with no fuel line attached.

> **Update (this commit):** rows 1–2 are now **fixed and verified** by an end-to-end
> integration test (`tests/integration/test_trading_pipeline.py`): `RebalanceRequestEvent`
> → risk sizes + risk-checks → `RiskApprovedEvent` (full order params) → execution creates +
> submits → paper broker fills → `OrderFilledEvent` → risk position feedback. Rows 3–4 (the
> order-API bypass and the orphaned live inputs) remain — that is roadmap item 4 below.

---

## 🔴 Critical findings by theme

### A — Predictions that drive trades are currently meaningless
- **Train/serve skew (tree models):** trained on `StandardScaler`-normalized features
  (`services/prediction/training/data_loader.py:157`) but the scaler is **never saved**
  (`services/prediction/models/xgboost_model.py:182`) and inference feeds **raw** features
  (`services/prediction/service.py:144`).
- **Feature ordering differs:** inference sorts features alphabetically
  (`services/prediction/service.py:144`); training uses the feature store's insertion order.
- **Scaler leakage:** scaler `fit` on the entire dataset before the train/val split
  (`data_loader.py:157`) → optimistic validation metrics that gate promotion.
- **Champion/challenger gate is dead:** `retrainer.py:106` reads `result.metrics`, which
  does not exist on `TrainResult` → `new_metrics` is always `{}` → every retrain rejected;
  and the retrainer is fed `np.empty((0,0))` (`retrainer.py:173`) so it never trains.
- **`expected_return` is fake:** all models map labels to `{-0.01, 0, +0.01}`
  (`xgboost_model.py:137` and siblings), discarding the real forward returns. _(Fixed for the
  live tree models — the regressor now trains on real forward returns; see update below.)_
- **LSTM/Transformer never contribute** at inference (sequence input hardcoded to `None`,
  `service.py:148`), so the live signal is only the (broken) tree models.

> **Update (this commit):** the tree (direction) path is fixed — training no longer scales
> features (train==serve) and inference orders features by the persisted `feature_names`; the
> champion/challenger metrics bug is fixed. Verified by `tests/unit/test_prediction_serving.py`.
> The **tree regressor now trains on real forward returns** (the data loader exposes them and
> the trainer threads them through), so `expected_return` carries real magnitude. Still open:
> probability calibration / confidence-gating, and the sequence-model double-normalisation +
> NN regressor (sequence models don't run at inference) — tracked separately.

### B — Backtest results cannot be trusted
- **Look-ahead bias:** strategy sees bar *i* (close/high/low), decides, then is **filled at
  that same bar's close** (`backtesting/engine.py:208`, `backtesting/simulator.py:120`).
- **Shorts have inverted cash signs:** open subtracts proceeds, cover adds them
  (`backtesting/engine.py:343`, `:366`), and double-counts against MTM.
- **No buying-power model:** cash can go arbitrarily negative (`engine.py:364`) → infinite
  implicit leverage.
- **Multi-symbol bars aligned by index, not timestamp** (`engine.py:210`); `min(len(...))`
  silently truncates to the shortest series.

> **Update (this commit):** all four are fixed and verified by
> `tests/unit/test_backtesting.py` — signals decided on bar *i* fill at bar *i+1*'s open;
> short cash flows are correct under a consistent signed-equity model; buys are buying-power
> capped (no negative cash); multi-symbol bars align on the union of timestamps
> (forward-filled) and warn on length mismatch; Sharpe/Sortino/annual-return are
> timeframe-aware (Sortino uses full-period downside deviation).

### C — Money-path safety gaps (bite the moment wiring is fixed)
- **No buying-power/cash check** before sending a real order (`execution/service.py:129`);
  computed `buying_power` is never consumed.
- **No idempotency:** Alpaca submit wrapped in `@retry(3)` with no client-order-id
  (`alpaca_broker.py:55`) → a network blip after acceptance sends a **second live order**.
- **Paper broker unrealistic:** buys always fill regardless of cash (`paper_broker.py:198`);
  fills quantized to `0.01` (wrong for crypto).
- **Short stops don't fire:** `StopLossManager.check_stops` is long-only; liquidation
  triggers emit `quantity=0.0` (`liquidation/service.py:115`) that nothing resolves to a
  real close.
- **Risk `post_trade_update` is sign-blind** — adds `fill_price*fill_quantity` regardless
  of buy/sell (`services/risk/service.py:323`), so a sell increases tracked exposure.

### D — Security / API
- **Forgeable JWT** (above) — the master key.
- **Order endpoint unguarded:** `api/schemas/orders.py:14` only checks `quantity > 0` — no
  max size, no notional cap, no `limit_price > 0`, no required price for limit/stop; the
  authenticated user is injected then discarded (orders not scoped to a user).
- **No authorization:** `role` read but never enforced; "admin only" endpoints
  (`api/routers/config_router.py`, `api/routers/risk.py`) don't check it; all
  order/position/portfolio queries are global.
- **`/debug/env` was public** (`api/main.py`) leaking DB/Redis URL prefixes. _(Removed in
  this commit.)_
- **CORS `allow_origins=["*"]` with `allow_credentials=True`** (`api/main.py`) — invalid and
  insecure. _(Replaced with a configurable allowlist in this commit.)_
- **WebSockets have no auth** (`api/websockets/streams.py:35`) and no connection cap.
- **Rate limiter implemented but wired to no route** (`api/middleware/rate_limit.py`).

### E — Deploy / data integrity
- **No migrations** (`alembic/versions/` empty) yet `railway.toml` runs `alembic upgrade
  head` → fresh prod DB comes up **empty** and every DB endpoint 500s.
- **Three divergent schemas:** `init.sql` (native enums + Timescale hypertables), the ORM
  (enums as `String`, **no `Prediction` model**), and empty Alembic. `core/models/orders.py:38`
  FKs `predictions.id`, a table the ORM doesn't define and whose real PK is composite.

> **Update (this commit):** the empty-DB blocker and the broken FK are fixed. Added the
> `Prediction` ORM model (simple `id` PK so `orders.prediction_id` resolves), a baseline Alembic
> migration (`0001_baseline`, `create_all` from the ORM — now the single source of truth), and
> `prepend_sys_path = .` so `alembic upgrade head` can import `core`/`config` on Railway.
> Verified: alembic discovers the head, all 11 tables compile to PostgreSQL DDL, FK resolves
> (`tests/unit/test_models_metadata.py`). Deferred: retiring `init.sql` and re-adding Timescale
> hypertables/retention as a follow-up migration.

- **No CI/CD** — `ruff`, `mypy --strict`, `pytest` all configured but nothing runs them.
  _(Fixed: see roadmap item 6 / the CI commit.)_
- **Frontend can't authenticate:** no login/token endpoint or login page exists, yet all
  routes need a JWT; the client redirects 401s to a nonexistent `/login`. Several hooks call
  paths the API doesn't expose; the WS URL omits the `/api/v1` prefix.

> **Update (this commit):** the backend now issues tokens — `POST /api/v1/auth/token` verifies
> credentials (bcrypt) and returns a JWT (`api/routers/auth.py`), verified end-to-end by
> `tests/integration/test_auth_login.py`. The frontend login page and the FE↔BE path /
> WS-prefix reconcile still remain (need the JS toolchain).

---

## 🟠 High / 🟡 Medium (condensed)
- Docker images run as **root**; the "frontend" prod image ships the **Vite dev server**.
  _(Fixed for the Python images: non-root `USER` + the missing `libgomp1` (xgboost/lightgbm
  OpenMP runtime) added to the root and api Dockerfiles. The frontend prod build remains.)_
- Two divergent Dockerfiles (ports 8000 vs 8080; one installs dev deps into prod).
- Redis EventBus **silently drops messages** on a failing handler — no retry/`XAUTOCLAIM`/DLQ
  (`core/events/base.py:131`); the in-process bus lets one bad subscriber crash the producer.
- Performance metrics hardcode `252` trading days — Sharpe/Sortino/annual-return wrong for
  intraday timeframes; Sortino downside-deviation denominator is non-standard
  (`backtesting/performance.py`).
- Provider retries only catch `ConnectionError`/`TimeoutError`, missing HTTP 429 / CCXT
  `RateLimitExceeded` → whole-symbol data loss.
- Predictions never calibrated or confidence-gated; NN training has no random seed.
- The "e2e" test asserts **both** branches of an if/else
  (`tests/e2e/test_full_trading_cycle.py:159`) — it can never fail. Zero DB-layer and zero
  API-layer tests. _(Fixed: tautology replaced with deterministic cases; DB + API tests added —
  see roadmap item 7.)_
- Prometheus scrapes a worker target serving no metrics; Grafana has no provisioning.

---

## 🧪 Test suite status (measured this review)

Rebuilt the environment (the committed `.venv` was broken) and ran the suite: it was
**5 of 40 red**, contradicting the impression of a tested codebase (there is no CI to
surface it). All five have since been **fixed in this commit — the suite is now green
(40/40)**. What was wrong:

- **Real bug — concentration rule is degenerate for new positions.**
  `services/risk/correlation_monitor.py` `portfolio_concentration` renormalizes weights to
  sum to 1 (`w / w.sum()`), but `pre_trade_check` passes weights as fractions of *equity*
  (which sum to <1, the remainder being cash). So a single proposed position of any size is
  scored as **100% concentration** and `MaxConcentrationRule` rejects it. Were the risk
  layer wired in, it would reject essentially every opening trade. (Surfaces in
  `test_full_cycle` and both risk-manager tests.)
- **Wiring gap — drawdown not coupled to equity.** `test_drawdown_halt` drives the
  `drawdown_monitor` to 8000, but `pre_trade_check` recomputes drawdown from `self._equity`
  (still 10000, never updated), so the drawdown rule never fires — same root cause as the
  orphaned-inputs finding.
- **Tests were never run green.** The risk tests assert rejection substrings
  `"MaxPositionSizeRule"` / `"MaxDrawdownRule"`, but the rule `name`s are `"MaxPositionSize"`
  / `"MaxDrawdown"` — the assertions cannot match. Proof the suite was not executed before
  commit.
- **MACD returns an all-NaN histogram** on numpy 2.4.6 / scipy 1.17.1
  (`test_macd_signal_cross`) — a warm-up bug or version drift; dependencies are unpinned
  (`>=`, no lockfile), so production can resolve to any version.
- **Over-tight assertion** — Bollinger containment is 82.7% vs a hardcoded ≥85% on seeded
  random-walk data (`test_bollinger_bands_contain_price`); likely test-tightness, not a code
  bug.

Net: the passing tests are real and meaningful; the suite as committed was red, and fixing
it surfaced (and corrected) a genuine risk-logic bug plus a dead-feature indicator bug.
Still missing: dependency pinning (the MACD bug was masked by running against whatever numpy
resolves — pin it). _(DB-layer and API-layer coverage have since been added; see item 7.)_

## ✅ What's done well (build on this)
- Clean, well-separated architecture; event-driven core; consistent typing/logging.
- `Decimal` discipline throughout the core ledger; floats only leak at the broker boundary.
- SQL is uniformly parameterized — no injection surface.
- Risk **math** is correct in isolation (Kelly, parametric/historical/MC VaR, circuit-breaker
  state machine, two-sided trailing stop) — but see _Test suite status_: the concentration
  check has a real bug and the committed suite is currently red.
- Indicators computed causally (NaN warm-up, no future bars) and well unit-tested.
- Right primitives: Redis consumer groups + ack, walk-forward validator, idempotent DB upserts
  (`ON CONFLICT DO NOTHING`), UTC discipline, graceful worker shutdown.
- The frontend is a real, fairly complete UI (React 18 + TS strict, TanStack Query, Zustand).

---

## Remediation roadmap (checklist)

- [ ] **0. Rotate broker keys + JWT secret** (owner action; brokers + Railway secrets).
- [x] **1. Security quick-wins** — remove `/debug/env`, configurable CORS, JWT live-mode
      guard, `.env` → paper endpoint. _(Done — see Appendix.)_
- [x] **2. Make it trade end-to-end** — `RiskApprovedEvent` now carries the full sized order
      intent; execution creates + submits it; stream names unified in `core/events/streams.py`;
      base `Event` preserves subclass fields on the Redis path; paper broker reports fills;
      and `subscribe()` no longer blocks startup. Verified by
      `tests/integration/test_trading_pipeline.py`. _(Done — see Appendix.)_
- [x] **3. Fix ML serving (direction path)** — trees no longer scale (train==serve); inference
      orders features by persisted `feature_names`; empty-metrics promotion bug fixed via
      `TrainResult.to_metrics()`; the tree regressor now trains on real forward returns.
      _(Deferred: probability calibration; sequence-path leakage.)_
- [~] **4. Connect risk to the order path** — **done:** buying-power check + client-order-id
      idempotency in execution; strict order-input validation; positions fed from fills (item 2).
      **Deferred:** synchronous manual-API → execution wiring (needs an async order-intent
      consumer) and the equity/PnL feed. See Appendix.
- [x] **5. Trustworthy backtest** — next-bar-open fills (no look-ahead); correct short cash
      signs + signed-equity model; buying-power cap; timestamp alignment; timeframe-aware
      Sharpe/Sortino/annual-return. Verified by `tests/unit/test_backtesting.py`.
- [~] **6. Productionize** — **done:** CI + lint cleanup; baseline Alembic migration +
      `Prediction` model + FK fix; non-root Python containers (+ `libgomp1`); backend
      auth/login endpoint (`POST /auth/token`, bcrypt). **Pending:** frontend prod build +
      login page + FE/BE API-path reconcile; `init.sql` retirement + Timescale hypertables.
- [x] **7. Money-path tests** — DB-backed order/fill persistence tests (SQLite), API-level
      auth-enforcement + order-validation tests (FastAPI TestClient), and the tautological e2e
      replaced with deterministic cases. Suite: 71 passing.

**Do not enable live trading until items 0–5 are done and verified.** Today a single flag
flip could send unbounded, un-risk-checked orders to a real account.

---

## Appendix — quick-wins applied in this commit
- `.env`: `ALPACA_BASE_URL` reverted to `https://paper-api.alpaca.markets`.
- `config/settings.py`: added `cors_allow_origins` (explicit allowlist, env-overridable) and
  a `model_validator` that **refuses the default JWT secret when `TRADING_MODE=live`** and
  warns otherwise.
- `api/main.py`: removed the public `/debug/env` endpoint; CORS now uses the configured
  allowlist instead of `["*"]`.

Test-suite fixes (suite now 40/40 green):
- `services/feature_engineering/technical/indicators.py`: `ema()` now seeds from the first
  *non-NaN* window, so the MACD signal line / histogram are no longer always-NaN (they were
  dead features feeding the ML pipeline).
- `services/risk/service.py`: pre-trade concentration now uses the largest position as a
  fraction of equity (no renormalisation), fixing the "single position = 100% concentration"
  bug that would reject every opening trade.
- `tests/unit/test_risk_manager.py`, `tests/unit/test_indicators.py`: corrected rule-name
  assertions (`MaxPositionSize`/`MaxDrawdown`), made the drawdown test drive equity through
  the real path, and relaxed an over-tight Bollinger containment threshold.

End-to-end wiring fix (roadmap item 2, verified by `tests/integration/test_trading_pipeline.py`):
- `core/events/streams.py` (new): canonical stream-name constants (kills the
  `risk_approved`↔`risk.approved` and `orders`↔`order_fills` drift).
- `core/events/risk_events.py`: `RiskApprovedEvent` now carries the full sized order intent
  (symbol/side/type/quantity/prices); `RebalanceRequestEvent` carries `reference_prices`.
- `core/events/order_events.py`: `OrderFilledEvent` carries `symbol`/`side`.
- `core/events/base.py`: base `Event` uses `extra="allow"` so subclass fields survive the
  Redis round-trip; `subscribe()` runs the consumer loop as a background task (was an
  infinite blocking loop that hung worker startup).
- `services/risk/service.py`: sizes weight deltas into share quantities from `reference_prices`,
  publishes full `RiskApprovedEvent`s, listens on the shared `orders` stream (type-guarded),
  and tracks positions sign-aware (sells reduce exposure).
- `services/execution/service.py`: `_handle_risk_approved` creates + submits the order from
  event params; published fills include symbol/side.
- `services/execution/brokers/paper_broker.py`: `get_order_status` reports
  `filled_qty`/`filled_avg_price` (what the monitor needs to record a fill).

Order-path safety (roadmap item 4, verified by `tests/integration/test_trading_pipeline.py`
and `tests/unit/test_order_schema.py`):
- `services/execution/service.py`: pre-trade **buying-power check** before any broker submit
  (rejects buys whose notional exceeds available buying power); **idempotency** via a
  correlation-id dedup set (a redelivered `RiskApprovedEvent` is skipped) and a stable client
  order id on the broker order.
- `services/execution/brokers/{alpaca,ccxt}_broker.py`: client order id passed to the venue
  (`client_order_id` / `newClientOrderId`) so a retry after a lost response is deduped rather
  than placing a second live order. _(Live-venue behaviour unverified here — smoke-test on deploy.)_
- `services/execution/brokers/paper_broker.py`: a buy cannot spend more cash than available.
- `api/schemas/orders.py`: strict `OrderCreate` validation (positive, bounded quantity; prices
  > 0; limit/stop price required by order type). `api/routers/orders.py`: infers asset class.

ML serving skew (roadmap item 3, verified by `tests/unit/test_prediction_serving.py`):
- `services/prediction/training/data_loader.py`: the tabular path no longer scales features
  (trees are scale-invariant and serving feeds raw features; the unpersisted scaler was the skew).
- `services/prediction/service.py` + `serving.py` + `models/base.py`: inference builds the
  feature vector in the model's persisted `feature_names` order (was alphabetical), so each
  column matches training; a missing feature defaults to 0 with a warning.
- `services/continuous_learning/retrainer.py` + `models/base.py`: promotion uses
  `TrainResult.to_metrics()` (the old code read a non-existent `.metrics`, always `{}`).
  _(Runtime-unverified here: xgboost/lightgbm need OpenMP, absent in this environment.)_
- `services/prediction/training/data_loader.py` + `trainer.py` + the tree models: the loader
  now exposes the real forward returns and the trainer threads them into the regressor, so
  `expected_return` is a real magnitude instead of a fabricated ±1% step. Verified by
  `tests/unit/test_data_loader_returns.py`.

Backtest correctness (roadmap item 5, verified by `tests/unit/test_backtesting.py`):
- `backtesting/engine.py` + `simulator.py`: signals decided on a bar execute on the NEXT bar
  at its open (was filled at the same bar's close -> look-ahead).
- `backtesting/engine.py`: short opens credit proceeds and covers debit (were inverted);
  equity uses a consistent signed mark (cash + long*mark - short*mark); buys are rejected when
  cost exceeds cash (no negative-cash leverage); multi-symbol bars align on the union of
  timestamps (forward-filled), warning on length mismatch instead of silently truncating.
- `backtesting/performance.py`: Sharpe/Sortino/annualised-return take periods_per_year from
  the timeframe (was hard-coded 252); Sortino uses full-period downside deviation.

CI + lint (roadmap item 6, partial):
- `.github/workflows/ci.yml` (new): `lint` (`ruff check .`) and `test` (`pytest`) gate every
  PR; `typecheck` (mypy `--strict`) runs non-blocking until the type debt is paid down.
- Repo-wide lint cleanup: `ruff check .` is now green (was 253 issues). Auto-fixed unused
  imports / import order / `datetime.UTC`; removed genuinely-dead locals; renamed an ambiguous
  `l`; raised `line-length` to 120 and ignored `N803`/`N806` (ML matrix-naming convention) and
  `UP042` (a `(str, Enum)`→`StrEnum` change would alter `str()` semantics). `ruff format` is
  *not* enforced in CI yet — a one-off `ruff format` pass (74 files) is a separate change.

Schema & migrations (roadmap item 6, verified by `tests/unit/test_models_metadata.py`):
- `core/models/predictions.py` (new): the missing `Prediction` ORM model, with a simple `id`
  PK so `orders.prediction_id` resolves to a valid FK (was `NoReferencedTableError`).
- `alembic/versions/0001_baseline_schema.py` (new): baseline migration that `create_all`s the
  ORM schema — `alembic upgrade head` now builds the DB (was a no-op against an empty
  `versions/`, so prod came up empty). ORM/alembic is the single source of truth.
- `alembic.ini`: `prepend_sys_path = .` so alembic can import `core`/`config` (this would also
  have broken on Railway, not just locally). `alembic/env.py` + `core/models/__init__.py`
  register the new model.

Money-path tests (roadmap item 7):
- `tests/integration/test_db_persistence.py` (new): order + fill ORM round-trip on in-memory
  SQLite — insert, query, the order→fills relationship, Decimal round-trip, status update.
- `tests/integration/test_api_auth.py` (new): the order endpoints reject unauthenticated
  requests (401) and reject invalid order bodies (422) via `OrderCreate` validation.
- `tests/e2e/test_full_trading_cycle.py`: the tautological if/else assertion (asserted both
  branches, so it could never fail) replaced with deterministic approve/reject cases.

Auth + Docker (roadmap item 6, verified by `tests/integration/test_auth_login.py`):
- `api/routers/auth.py` (new): `POST /api/v1/auth/token` verifies email/password and issues a
  JWT. `api/middleware/auth.py`: bcrypt `hash_password`/`verify_password` (using the bcrypt
  library directly — passlib is incompatible with bcrypt 4.x), with a dummy-hash path so login
  timing doesn't reveal whether an email exists. `noload` on the api_keys relationship keeps
  login to one query. Deps: `passlib[bcrypt]` → `bcrypt` in pyproject + requirements.
- `Dockerfile` + `infrastructure/docker/Dockerfile.api`: run as a non-root `appuser` and add
  `libgomp1` (the OpenMP runtime xgboost/lightgbm load at import — was missing).

Still deferred: the manual-order **API → live execution** path (needs an async order-intent
consumer in the worker) and feeding risk live equity/PnL; plus the rest of productionisation
(frontend prod build + login page + FE/BE path reconcile, `init.sql` retirement + Timescale
hypertables) — left for deliberate, separately reviewed changes.
