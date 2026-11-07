# Agent-fleet review of the platform — 2026-11-02

Sixteen role agents were defined in `.claude/agents/` (see `.claude/AGENT_FLEET.md`).
Fifteen ran a read-only review of the platform in parallel, each with a narrow file scope
and a six-finding cap; the sixteenth (`product-lead`) ran afterwards on the collected
findings. This file is the full record. The condensed, ranked version and the registered
build set are in GO_LIVE.md under the same date.

Confidence labels are the agents' own: CONFIRMED = traced end to end in code;
PLAUSIBLE = not fully traced. The lead re-verified every finding that the build set below
acts on (the stream-name mismatch, the fill-monitor poisoning, the in-memory order manager,
the unauthenticated WebSockets, the unguarded order routes, the ML-page zero coercion).

## Verdicts in one line each

| Role | Verdict |
|---|---|
| risk-manager | **BLOCK** go-live: no working exit path; safety state not durable; manual orders unguarded; breaker trip reaches nobody |
| trading-systems-engineer | Ledger not reconstructable after a restart or a venue timeout; durable orders before any Stage 4 capital |
| portfolio-manager | Paper telemetry only; would not survive real capital (no exits, exits can manufacture shorts, fill monitor can silently stop recording) |
| security-engineer | Paper acceptable; not live-ready (viewer can trade, open WebSockets, admin JWT written to logs via URL) |
| sre-devops | Fails safe on crashes, not on silence (no staleness watchdog, no external heartbeat, restart lifts a halt) |
| compliance-lead | Record says WHAT traded, not WHY, WHO or what the venue said; nothing append-only |
| crypto-market-specialist | Paper path safe but flattering; live Binance path broken three independent ways |
| data-engineer | Bar semantics and idempotency sound; two confirmed train/serve breaks (cold buffer after deploy; midnight-stamped daily aux history) |
| ml-engineer | Live IC instrument honest; promotion gate's validation leg contaminated four ways; label horizon 25 bars vs 5-minute scoring; catboost outside the retrain loop |
| quant-researcher | Honest in intent; two load-bearing numbers structurally wrong; readiness passable by one of 14 null factors with ~60% probability |
| evaluation-specialist | Money-path figures honest; ML page still coerces absent metrics to zero and shows base-rate accuracy |
| fullstack-product-designer | Truthful about WHAT, not inspectable about WHY; Audit Log page broken against its API |
| ai-agent-engineer | Coverage complete and factually current; least privilege advisory, not enforced |
| blockchain-data-analyst | No on-chain data exists; three registered-watch specs and a point-in-time pipeline spec supplied |
| securities-crypto-lawyer | Issues memo for counsel; first fact to establish is the operator's jurisdiction and the Binance entity |

## Ranked gap list (de-duplicated across roles)

Rank order: money-at-risk, then trust-at-risk, then effort. "Built today" marks the items
closed in the same change set as this review; everything else is registered work.

1. **Exit path disconnected** (risk-manager, portfolio-manager, compliance-lead). Liquidation
   subscribes to `risk.breaches`; risk publishes `risk.breached`; monitoring listens on
   `risk`; nothing consumes `liquidation.triggered`; the breaker trip publishes no event, so
   no alert and no audit row. *Built today:* stream constants unified, CircuitBreakerEvent
   published on every breaker transition with alert + audit row, execution consumes
   liquidation events through the tracked order path, breaker-trip flatten behind
   `breaker_auto_flatten` (default OFF, operator decision).
2. **Order state is process memory** (trading-systems, portfolio-manager, sre, risk,
   compliance). In-flight orders lost on restart; one failed publish poisons the fill monitor
   for ever; autonomous fill persistence swallows DB errors and acks; no fill dedupe; venue
   ids, venue time and fees never persisted; realised P&L per position never written.
   *Built today:* fill-monitor fault isolation and idempotent recording; Order row written
   before submit; startup reconciliation; fill dedupe by venue fill id; order_sync no longer
   swallows; per-position realised P&L.
3. **Safety state not durable** (risk, sre). Halt flag, breaker state, day-start and peak
   equity reset on every restart; a halted book resumes trading after a deploy. *Built
   today:* a `safety_state` row restored before services start.
4. **Authorisation and WebSockets** (security, risk, ai-agent, fullstack). Any JWT can place
   and cancel orders; 501 stubs unguarded; WebSockets unauthenticated and uncapped; the
   frontend puts the JWT in the WebSocket URL, which uvicorn and nginx log; privileged API
   actions leave no audit row. *Built today:* role gates, first-message WebSocket auth,
   connection cap, audit rows with actor and IP on `/admin/control` and order routes.
5. **Training label is 25 bars; everything downstream scores 5 minutes** (ml, quant).
   Registered, not built: changing it silently re-bases every live measurement.
6. **Promotion gate's validation leg contaminated** (ml, quant): hyperopt, early stopping,
   calibration, conformal and the gate metric all use one split; the edge harness's OOS rows
   are the calibrator's fit set; catboost is outside the retrain loop. Registered.
7. **Readiness panel passable by a null factor** (quant): no family-wise bar across 14
   watches, current quarter counted, no minimum n. Registered.
8. **Train/serve breaks in data** (data, quant, blockchain): cold 200-bar feature buffer after
   every deploy; daily VIX/SPY backfill stamped ~16h before publication. Registered.
9. **Live Binance path** (crypto): quantity rounds to zero; status/cancel need the symbol;
   equity counts USDT only; no min-notional enforcement; paper costs and ticks unrealistic.
   Registered (live-only).
10. **Operational silence** (sre): `/healthz` unconditional; no bar/prediction staleness
    watchdog; no external dead-man; no backups; no commit stamp in production. Registered.
11. **UI truth** (evaluation, fullstack, compliance): ML page coerces absent metrics to 0;
    Audit Log page reads fields the API does not send; Trading page fabricates fill state;
    dead REST feed renders live. *Built today:* ML page and Audit Log page. Rest registered.
12. **Exits sized from a stale dollar book** (portfolio-manager): weight-0 exits leave dust or
    an unintended short; the paper broker accepts naked shorts. Registered.
13. **Fleet least privilege** (ai-agent): advisory roles had Bash/Edit/Write. *Built today.*
14. **Hygiene**: macOS " 2.py" duplicate tests collected locally; the local venv is Python
    3.14 while the image is 3.11 and ML imports stall on this Mac. *Built today:* pytest
    ignore-glob. Venv rebuild is an operator action.

## Operator-only actions collected from the roles

- Enable Railway Postgres backups and run one recorded restore drill (sre).
- Provision an external dead-man monitor and supply its URL once the heartbeat lands (sre).
- Rotate `JWT_SECRET` after the WebSocket URL leak fix ships; existing tokens are in logs (security).
- Set `FORWARDED_ALLOW_IPS` on the api service; set `CORS_ALLOW_ORIGINS` to the UI origin (security).
- Confirm broker keys are trade-only, withdrawals disabled, IP-restricted (crypto, security).
- Decide whether manual orders exist in live at all; create admin vs trader vs viewer users accordingly (risk).
- Decide `breaker_auto_flatten` (risk-reducing flatten on a breaker trip) — default OFF (risk).
- Write the one-paragraph fact sheet (residence, entity, Binance entity, Railway region) and take it to counsel (lawyer).
- Rebuild the local venv on Python 3.11 to match CI and the image (evaluation, security).

## Verification of the build set (same day)

ruff clean across the repo; mypy strict clean on every changed file (three pre-existing
`type-arg` errors in the untouched `services/research/funding_watch.py` under this venv's
mypy 2.1.0 on Python 3.14; CI on 3.11 is the gate); `tsc -b` clean. Targeted pytest batches,
once the host load allowed them to run: fill monitor + order durability 8/8; safety state +
exit path + account sync 16/16; API authz + auth 25/25; adjacent existing suites (order
persistence, kill switch, trading pipeline, broker-state durability, snapshot writer, memory
pruning, order intent, trading control, state machine, circuit breaker, ledger
reconciliation) 64/64. See GO_LIVE.md "SHIPPED (2026-11-02)" for the per-item verdicts.

## Per-role reports

### risk-manager


Control table: max single order % NOT enforced (sized_quantity discarded); daily drawdown rule absent (only 7% breaker); total drawdown/peak and day_start_equity NOT durable (reset on restart); breaker not durable, no event/alert/audit on trip; halt not durable (_halted=False on boot); reconcile_positions has no caller; VaR cap compared vs 1-minute returns (~38x loose).
F1 CRITICAL CONFIRMED: liquidation/service.py:22 subscribes "risk.breaches" but risk publishes RISK_BREACHED="risk.breached" (core/events/streams.py:18); monitoring listens on "risk" (monitoring/service.py:25). Liquidation acts only on action in ("reduce","liquidate") (liquidation/service.py:152) but risk publishes action="block" (risk/service.py:201). register_position/set_stop_loss/set_trailing_stop have no callers; nothing subscribes to "liquidation.triggered". => No auto-exit exists.
F2 HIGH CONFIRMED: safety state not durable: execution/service.py:71 _halted; risk/service.py:140-144,557-562 _day_start_equity=initial_capital; drawdown_monitor.py:78-82 peak=first equity; breaker in-memory; worker.py restores broker state only.
F3 HIGH CONFIRMED: POST /orders needs only get_current_user (viewer); _handle_order_intent skips pre_trade_check; live router would route to venue.
F4 HIGH CONFIRMED: breaker trip only logs (circuit_breaker.py:140-151); CircuitBreakerEvent never published; monitoring/service.py:188-198 dead; RiskBreachedEvent carries 0.0 values and rule_name "MaxDrawdown: ..." so alert match at :206 never fires.
F5 MEDIUM: max_single_order_pct dead (risk/service.py:350 vs :215-222); max_daily_drawdown_pct read by no rule; VaR horizon mismatch (rules.py:186 vs risk_metrics_writer.py:164).
F6 MEDIUM: pre_trade_check side-blind -> blocks exits when risk highest (Aug 1 incident); reconcile_positions (execution/service.py:558) never called.
Lift condition: F1-F4 fixed with regression tests + a paper soak where breaker trips and book auto-exits at least once.

### trading-systems-engineer


F1 CRITICAL CONFIRMED: order manager + _in_flight process memory; no startup reconciliation (service.py:62-67,598; order_manager.py:128; worker.py:376-397 restores broker book then starts). SIGTERM between submit (paper fill applied) and next poll -> no fills row, no event; checkpoint shows position (ADA 252/249 pattern); API row PENDING forever.
F2 HIGH CONFIRMED: order_sync.py:170-172 _insert_autonomous_fill swallows Exception -> _dispatch acks -> never retried; transient DB error = fill permanently absent. Fix: let it raise (bus redelivers, DLQ after 5) — only safe with dedupe (F4).
F3 CRITICAL(live) CONFIRMED: submit exception after venue acceptance -> REJECTED while order is live (service.py:445-452; alpaca_broker.py:56-61 3 retries). Venue book changes silently.
F4 HIGH CONFIRMED: duplicates across restart: in-memory correlation set (service.py:70,142-144) + fresh uuid as client order id (:341-350,:423) + no dedupe in order_sync (fills.external_fill_id never written). XAUTOCLAIM redelivery after restart -> second venue order / second fills row.
F5 HIGH(live) CONFIRMED: partial-then-cancel loses filled qty; EXPIRED never handled (service.py:615-669) -> polled forever.
F6 MEDIUM CONFIRMED: API DELETE only flips DB row (orders.py:172-181); resting paper limit later fills and order_sync overwrites CANCELLED->FILLED; worker cancel_order drops from _in_flight without final poll (service.py:468-474).
Unranked: execution.stop() kills monitor while intent consumers keep submitting until event_bus.close(); fill replay keyed on persistence time can double-apply (PLAUSIBLE).
DESIGN A (durable order manager): migration 0011: orders.broker_name, orders.correlation_id (indexed), partial unique index fills.external_fill_id. Order row IS the durable order; id = engine order_id = venue client_order_id. Write row BEFORE submit (pending -> broker_name/external_id=id commit -> submit -> submitted/submitted_at/venue id). Transport exception stays `submitted`, never REJECTED; reconciler asks venue by client id. Immediate post-submit poll. Monitor emits delta fills, handles expired/cancelled-with-fills; dedupe = venue fill id or f"{venue_id}:{cum_qty}". reconcile_on_start() before subscribe: load non-terminal rows, poll, publish missed fills, paper rows unrecoverable -> expired lost_on_restart + alert, seed _in_flight. Shutdown: _stopping makes intent handlers raise before monitor cancelled. API DELETE publishes cancel intent. Tests in tests/integration/test_order_durability.py incl. restart drill and 20-cycle random-kill asserting ledger_reconciles. Criterion: zero unmatched closes across 20 drills; ledger_reconciles true 7 days post-deploy.
DESIGN B (per-position realised P&L; no migration): paper_broker _realized_by_symbol accumulated at :281/:293, reset when opening from flat; get_positions adds realized_pnl; realized_by_symbol(); restore_state(realized_by_symbol=); broker_state.checkpoint writes live["realized_pnl"] on open rows and realized_by_symbol[symbol] on rows being closed; restore feeds back. Criterion: sum(positions.realized_pnl) == portfolio_snapshots.realized_pnl to the cent.
Verified good: bus acks after handler; XAUTOCLAIM + DLQ; live gates; flatten via submit_order; Alpaca retry idempotent via client_order_id.

### portfolio-manager


F1 CRITICAL CONFIRMED: no working exit path: liquidation.triggered has zero subscribers; set_stop_loss/set_trailing_stop/register_position no callers; execution_service_ref unused; stops publish quantity=0.0; StopLossManager.check_stops side-blind (short's stop fires first tick). Only exits: 15-min exploration timer (in-memory) and weight-0 conviction exits. Fix: execution subscribes STREAM_LIQUIDATION, resolves qty 0 to broker signed qty, submits opposite side; register stop from each fill.
F2 HIGH CONFIRMED (reproduced): exits sized in dollars at current price vs stale dollar book -> dust or unintended short (risk/service.py:216-224,384-389; risk_metrics_writer.py:134); paper broker accepts naked shorts (paper_broker.py:354-364: 1000 SOL sell on $100 fills, cash -> $150k); _feed_positions feeds abs(market_value) so short reads as long. Matches prod drift signature (DOT -0.0012 short, ADA 0.0125 dust). Fix: size weight-0 exits from broker signed qty; paper rejects sells taking position negative.
F3 HIGH CONFIRMED (reproduced): one failed publish poisons the fill monitor forever (service.py:624-649,673-675): record_fill on FILLED order raises OrderError every 2s; later orders starved; fill never published. Fix: per-order try/except; skip record_fill if already FILLED, only publish.
F4 HIGH CONFIRMED: restart: (a) paper market order fills synchronously, published on next 2s poll -> kill in window loses the record; checkpoint in between carries fill ledger never gets; (b) inverse: checkpoint at T includes fill F, F persists at T+2s with filled_at > snapshot.time -> _replay_fills_after re-applies -> DOUBLED position/cash; (c) portfolio _last_target_symbols/_exploration_positions not restored -> held exploration never expires, re-entered at full weight. Fix: Order row at submit (SUBMITTED), rebuild _in_flight at boot, replay fills by order id not timestamp, persist held book.
F5 MEDIUM: realised P&L per position never written (broker_state.py:203-207; paper_broker.py:281). Fix: _realized_by_symbol in _apply_to_book, exposed via get_positions/get_account, written to row.realized_pnl on update/close.
F6 MEDIUM: sizing ignores venue minimums (_MIN_TRADE_NOTIONAL=1.0; exploration $1 orders; ccxt warns only); FixedFractionalSizer output never used (order size = target_weight x equity: 10% target = $10 order vs 5% cap).
Gaps: cost model in trade gate (Aug ~33 round trips/day x $3; paper slippage ~1.5%/month ≈ whole -1.4%; live would be ~6%/month); boot-time list_open_orders reconciliation; partial fills ignored (cancelled branch drops filled_qty); persisted stop/time-stop per position (columns exist, never written).

### security-engineer


Route table: 1 of 45 routes checks role (admin.py:48). POST/DELETE /orders any JWT; 501 stubs (config risk-rules/trading-mode, breaker reset, models retrain, portfolio rebalance) no role; /health, /market-data unauthenticated; WS /ws/* unauthenticated; trader role enforced nowhere.
F1 HIGH CONFIRMED: viewer token can place/cancel orders (orders.py:99-148,156-181); in live reaches real broker up to buying power; Order row stamped trading_mode="paper" regardless (orders.py:125); no user_id. Fix: require_role({"admin","trader"}) on POST/DELETE; stamp settings.trading_mode; add user_id.
F2 HIGH CONFIRMED: WebSockets unauthenticated and uncapped (streams.py:22-71, manager.py:34-37, fanout.py:31-35) broadcasting fills/risk alerts; rate limiter keys by request.client.host and FORWARDED_ALLOW_IPS unset -> behind Railway edge all clients share one login bucket (PLAUSIBLE lockout). Fix: verify JWT before accept (header/first message, not URL), 1008 close; per-channel cap; FORWARDED_ALLOW_IPS.
F3 HIGH CONFIRMED: frontend puts JWT in WS URL (websocket.ts:26-27 ?token=); uvicorn 0.49 logs WS path+query at INFO; nginx access log records $request; server never reads the param. 24h admin bearer in logs. Fix: drop ?token=; pass as first WS message; operator rotates JWT_SECRET.
F4 MEDIUM: privileged API actions leave no audit row (admin.py:53-60); ip_address populated nowhere; AuditLogger.log_action has no user_id; no trigger/REVOKE on audit_logs.
F5 MEDIUM: unpinned deps (requirements.txt all >=), no lockfile/hashes; Actions by tag; no permissions: block; local venv py3.14 vs image 3.11. Fix: requirements.lock --generate-hashes; pin Actions by SHA; permissions: contents: read.
F6 MEDIUM: 24h JWT in localStorage, no revocation (get_current_user never consults DB), no CSP/XFO/HSTS/nosniff in nginx template. Fix: headers; expiry <=60 min; is_active check or token_version claim.
Operator: rotate JWT_SECRET after F3; FORWARDED_ALLOW_IPS; confirm broker keys trade-only/no-withdrawal/IP-restricted; CORS_ALLOW_ORIGINS to UI origin; delete local .env.save; branch protection.
Verified good: HS256 pinned; live-mode secret guard; bcrypt; login limiter; /debug/env gone; CORS allowlist; SecretStr; .dockerignore; non-root uid 10001; live brokers gated twice.

### sre-devops


F1 CRITICAL CONFIRMED: /healthz unconditional (worker.py:71-86); health loop checks only SELECT 1/PING/broker URLs; no last-bar/last-prediction age or consumer-lag watchdog anywhere; ccxt realtime task dies once (data_ingestion/service.py:315-319) and nothing restarts it. 07-21 stall class undetectable. Fix: record last_bar_at/last_prediction_at in monitoring handlers; CRITICAL "Pipeline stale" alert >10 min; daily heartbeat.
F2 HIGH CONFIRMED: alerting lives inside the worker; railway.toml restarts ON_FAILURE max 5 then leaves service down; no external dead-man's switch; dedupe title-keyed 5min in-memory, no "recovered" alert (DB outage posts 288x/day). Redis-down: 24 consumer loops log ~300-400 lines/s (near platform cap). Fix: heartbeat POST to external monitor URL from _periodic_health_check.
F3 HIGH CONFIRMED: restart lifts halt (_halted), forgets _in_flight/_processed_correlation_ids, rebases daily breaker (drawdown_monitor.py:76-81; risk/service.py:496-508), breaker boots CLOSED; XAUTOCLAIM of a pending RiskApproved after crash can re-submit (idempotency set empty) — PLAUSIBLE narrow. Fix: worker_state row (halted, session_date, day_start_equity, breaker) read before start(); rehydrate _in_flight from orders rows.
F4 HIGH: no backup/restore procedure; retention no-op on vanilla Railway Postgres (0002:60-63); predictions excluded from retention even with Timescale; risk_metrics 720 rows/day; no pruning job. Fix: platform backups + restore drill; scripts/prune_tables.py.
F5 MEDIUM: no commit SHA in /healthz or logs; api/worker deploy independently; no prod-vs-main drift check.
F6 MEDIUM: no runbook per alert; health.py:78-80 Alpaca /v2/clock unauthenticated (401 counts healthy); health.py:84-86 Binance block -> "Degraded" every 5 min in paper mode.
Restart table: halt, in-flight, idempotency set, risk baselines/peak/breaker, portfolio _exploration_positions all lost; paper broker restored at 300s cadence; buffers/evaluator/registry OK.
Operator: enable Postgres backups + restore drill; verify Redis persistence; preDeployCommand on worker?; external dead-man monitor; Slack webhook rotation; disk sizing; RAILWAY_GIT_COMMIT_SHA availability.

### compliance-lead


Lineage: risk verdict per order MISSING everywhere (RuleResults discarded risk/service.py:339-343; RiskApprovedEvent carries none; RiskBreached publishes 0.0/0.0); submission partial (engine uuid stored nowhere; venue id only in memory; orders.external_id holds correlation id; submitted_at never set); fills partial (external_fill_id never set; commission omitted at publish service.py:639-648; filled_at = receipt time); realised P&L per position MISSING; liquidation path inert; flatten legs untagged (no client_order_id -> external_id NULL); manual orders: actor not stored, DELETE cancel DB-only never reaches engine (orders.py:156-181).
F1 HIGH: risk verdict/limits never persisted. Fix: rule_results + limits snapshot on RiskApprovedEvent; risk_decisions row keyed by correlation id.
F2 HIGH: audit rows empty/unjoinable: monitoring/service.py:157-172 details=event.payload which is {}; entity_id = engine uuid no orders column stores; /audit/decision-trace 404s; log_trade_decision has no caller. Fix: details=event.model_dump(exclude={"payload"}); engine_order_id on Order.
F3 HIGH: venue ids/time/fees never reach the record (order_events.py:18-27; brokers have them). Fix: add venue_order_id, venue_fill_id, venue_filled_at, commission to OrderFilledEvent + Fill.
F4 HIGH: privileged actions no durable attributable record: admin.py:53-60 no audit row/IP; monitoring never subscribes CONTROL; AuditLogger.log_action has no user_id/ip params (monitoring/audit.py:27-35); breaker trip logs only; promotion = JSON+log; user creation CLI no record. Fix: audit row before publish; tag flatten legs client_order_id=f"flatten-{event_id}"; publish CircuitBreakerEvent.
F5 MEDIUM(now)/HIGH(live): nothing append-only; one DB role; orders rewritten in place; audit write failures swallowed (audit.py:68-74). Fix: BEFORE UPDATE/DELETE raise triggers on audit_logs/fills; order_status_history.
F6 MEDIUM: AuditLog.tsx:12-16,130-151 sends component/risk_level, API returns service/action -> blank columns; 1000-row cap no offset; no CSV export; risk_metrics 3y < audit 7y.
Self-match: no explicit control; exposure low (exploration excludes held symbols, paper-only); residual manual-vs-autonomous. Autonomous path has no order-count/notional cap or venue-minimum check.

### crypto-market-specialist


F1 CRITICAL(live) CONFIRMED: ccxt_broker.py:225-229 _adjust_quantity treats precision as decimal count but ccxt Binance precisionMode=TICK_SIZE so precision IS stepSize (1e-05) -> int()=0 -> quantize to whole units -> $1-5 clips send amount=0 (rejected -1013); DOT rounds UP. Fix: exchange.amount_to_precision (truncate) + refuse below limits.amount.min. No test.
F2 CRITICAL(live) CONFIRMED: ccxt_broker.py:124,139 get_order_status/cancel_order pass symbol=None; engine calls with external_id only; ccxt Binance raises ArgumentsRequired -> poll_status_failed forever, no fills recorded, cancel always False. Fix: remember symbol per external_id in submit_order.
F3 CRITICAL(live) CONFIRMED: ccxt_broker.py:187-193 equity = USDT only (coins unvalued); first $5 buy reads as 5% drawdown; second trips 7% breaker. Fix: value non-USDT balances via fetch_tickers.
F4 HIGH CONFIRMED: no MIN_NOTIONAL/LOT_SIZE enforcement (ccxt_broker.py:234-240 warns only; limits.cost.min never read); exploration $1 clips and $5 single-order cap are below Binance 5 USDT minimum; PaperBroker applies no step (prod paper shows 0.0125 ADA). Soak evidence does not transfer. Fix: venue-filter shim in PaperBroker for "/" symbols; raise crypto min_trade_notional.
F5 MEDIUM CONFIRMED: paper commission 0, slippage 0-5bps vs Binance 10bps/side taker; Aug 2,045 fills x $3 x 0.1% ~ $6 fees never charged (~6% of account vs recorded -1.4%).
F6 MEDIUM CONFIRMED: paper_broker.py:351,417 quantize(0.01) on fill prices — ADA cent tick = ±1.2% noise.
Gaps: key hygiene (trade-only/no-withdrawal/IP) undocumented in GO_LIVE/.env.example; live positions carry market_value 0; both-live _account_broker picks Alpaca only; zero tests on CCXTBroker order path. Low: worker.py:28 docstring stale (ccxt does route to testnet outside live).

### data-engineer


F1 HIGH CONFIRMED: feature_engineering/service.py:62,101-124 _bar_buffers empty on restart; _load_recent_bars (168-212) never called and queries nonexistent ohlcv_bars.bar_time; first ~200 bars post-deploy yield NaN->0.0 for sma_200/ema_200/rsi/bb/adx (feature_store.py:183-186) beside real close; no warm-up gate in prediction/service.py:181-230; vectors persisted, resolved, feed evaluator/calibrator/PSI; exploration can trade on them. Fix: seed buffer from ohlcv on handle_bar_close when len<200 (fix table/column names) and/or suppress FeaturesReadyEvent until warm.
F2 HIGH CONFIRMED: backfill_aux_market.py:123,129 stamps VIX/SPY at session midnight (yfinance daily index) -> as-of serves day's final close to every 1m bar from 04:00Z-20:00Z; the 07-23 regime A/B ran on these rows. Live rows poll-time stamped. Fix: stamp daily rows at date+1d 00:00Z.
F3 MEDIUM CONFIRMED: ccxt_provider.py:186-228 sleep(60) not grid-aligned + limit=2/candles[-2] -> drops ~1 bar/symbol every 40-120 min; exception aborts remaining symbols that minute; nothing refetches. Fix: fetch_ohlcv(since=last+1, limit=10), deliver all closed, sleep to next :02.
F4 MEDIUM CONFIRMED: retrainer.py:476 loads aux rows time>=cutoff only -> first 8-24h of training window aux_*=0.0. Fix: also select latest row per metric before cutoff.
F5 MEDIUM CONFIRMED: retry decorators (ccxt_provider.py:90-95, alpaca_provider.py:143-148) catch only builtin ConnectionError/TimeoutError; ccxt errors derive from BaseError; one 429 drops a symbol in backfill_history.py:203-207; LiveAuxProvider serves last value forever with no age (aux_features.py:119-127).
F6 MEDIUM CONFIRMED: frozen prices value book/stops/breaker during feed stall (paper_broker.py:70-72,181-183; risk_metrics_writer.py:162-170); PriceUpdateEvent.market_timestamp is bar OPEN. Fix: carry bar_time+60s; flag symbols >3 min stale.
Gaps: filings/news/social/on-chain do not exist (alternative/{onchain,sentiment}.py, fundamental/metrics.py hard-coded placeholders imported nowhere); EDGAR submissions keyed by accession stamped acceptanceDateTime; GDELT 15-min seen_at; social forward-collect at fetched_at.
Verified good: closed bars only; bar OPEN stamping + single _ensure_utc; ON CONFLICT DO NOTHING on true PK; funding at fundingTime; bisect_right as-of; retention covers lookbacks.

### ml-engineer


F1 HIGH CONFIRMED: dataset_builder.py:120,135,181 at stride=HORIZON=5, forward_returns and TB span = 5 replay steps = 25 one-minute bars; resolver (continuous_learning/service.py:67,741), prediction_sync.py:105, learning metrics IC and live-transfer gate (live_replay.py:220-225) all use 5-min return. Regressor trained on 25-min, gated on 5-min. No test pins label span == served horizon. Fix: label on bar offsets (closes_bar[e+5], TB span 5 bars) independent of stride; re-tune TB_K; register in GO_LIVE.
F2 HIGH CONFIRMED: retrainer.py:243-250 calls train_model without cv_folds -> hyperopt.py:142-150 30 trials minimise val_loss on X_val; same X_val early-stops LGBM/CatBoost, fits calibrator (xgboost_model.py:250), conformal (:255), and produces gate balanced_accuracy (:265,297). One split, five uses. train_and_promote.py:291 passes purged folds; auto-retrain does not. Fix: pass purged_chrono_folds(len(X_train),3,embargo=HORIZON+1) in _retrain_member.
F3 MEDIUM CONFIRMED: models/base.py:187-207 calibrator fit on X_val and Brier scored on same rows; conformal fit on in-sample calibrated probs; served balanced_accuracy with calibrator that saw every val row. Fix: split X_val chronologically into calibration half and metric/conformal half.
F4 MEDIUM CONFIRMED: walk_forward.py:85 splits arange(n) over symbol-concatenated array (dataset_builder.py:314) -> fold k validation = one symbol's rows while others' same-minute rows in train (cross-sectional leakage); hyperopt objective maximises raw val_accuracy (hyperopt.py:162). Fix: K contiguous blocks per symbol, union block k; score by balanced accuracy/log loss.
F5 HIGH CONFIRMED: retrainer.py:49 _KNOWN_MODEL_TYPES omits "catboost"; served ensemble is xgboost,lightgbm,catboost (serving.py:53); catboost stays 2026-07-20 bootstrap forever; with min_agreement=3 a stale member vetoes every directional vote; feature_names drift unchecked (serving.py:162). Fix: add "catboost"; test every ModelServer type retrainable.
F6 MEDIUM(latent) CONFIRMED: live_calibration.py:138-154 fits on PredictionRow.confidence (post-mapping) not p_flat_raw; GO_LIVE:1699-1707 required raw; settings default ON. Fix: coalesce(p_flat_raw, confidence).
Gaps: no n_train/n_val guard on TrainResult; no significance margin on gates (balanced_accuracy > 1/3 strict; IC compare with SE≈0.022 passes no-skill ~50%); chance level hard-coded 1/3 while balanced_accuracy averages classes present (2-class tail -> constant scores 0.5); feature-hash deadlock on feature addition; artifact pickles vs unpinned sklearn; outer 80/20 split has no purge.
Verified good: aux parity (poll-time stamps + bisect), serving by persisted feature_names, embargoed replay, seeds fixed, TB sigma trailing-only.

### quant-researcher


F1 HIGH CONFIRMED: training label 25 bars vs 5-min outcome/gate/trade horizon (same as ML F1).
F2 HIGH CONFIRMED: edge.py:184-200 hands ds.X[split:] to train() as X_val -> calibrators fit on OOS rows -> predict_batch argmax on same rows; hit_rate/Sharpe/stability partly in-sample; reached by Stage-2 gate, gauntlet (gauntlet_daily_harness.py:129), dashboard backtest (runner.py:101). Fix: split 70% into fit/calibration (60/10); leave 30% untouched.
F3 HIGH CONFIRMED: no family-wise control across 14 watches; research.py:114-120,137 walks reversed(quarters) incl. current quarter; quarterly_rollup (funding_watch.py:746-770) has no min n (single stamp = positive quarter); GoLiveReadiness.tsx:47-51 green when met.length > 0. P(>=1 of 14 null factors shows 4 positive quarters) ≈ 0.60; near-mirror factors in family. Fix: exclude current quarter; n>=~90 stamps/quarter; register family bar (Holm-adjusted t on pooled 4-quarter IC) for readiness.
F4 MEDIUM CONFIRMED: new listings top-rank low_vol_7d_minus (funding_watch.py:246-254 -trailing_std with nanstd: one return -> -0.0 = max); compute_signal_ics (:489-493) gates on row index not per-contract history; post-listing drift enters append-only record. Fix: require 21 finite returns; mask symbols below min_history finite closes.
F5 LOW-MED CONFIRMED: aux backfill midnight stamping (same as data F2).
F6 MEDIUM CONFIRMED: cost model: edge.py:32 5bps round trip; backtest.py:42 default 0.0005; Binance VIP0 20bps RT taker + spread; run_backtest initial_capital=10_000 (edge.py:258); no min-notional/lot step (:342). Fix: VenueCosts preset threaded through net_period_returns; initial_capital=100 in dashboard path.
Gaps: Holm-adjusted family adjudication + pairwise IC correlation matrix (promised 2026-08-09); per-contract history enforcement; buy-and-hold control in run_backtest (2026-08-02 method note unimplemented); venue-aware cost/lot model.
Verified good: watch core (settled-close exits, append-only w/ heal cap, trust gate, exact-grid snapshots); net_period_returns stride; stability_fraction; deflated_sharpe; purged_chrono_folds; balanced_accuracy.

### evaluation-specialist


F1 HIGH CONFIRMED: ML page renders Precision/Recall/F1/Sharpe/Predictions as 0 (fields don't exist in models.py:261-282 payload; normalize.ts:192-197 num() coerces absent->0; ModelPerformance.tsx:97-118); Accuracy column = raw val_accuracy (~0.594 = base rate); majority_class_rate/balanced_accuracy in payload rendered nowhere. Listed STILL OPEN 2026-07-29; contradicts 2026-09-12 criterion (ii). Fix: numOpt + em-dash withhold; show balanced accuracy beside base rate.
F2 MEDIUM CONFIRMED: retrainer.py:647-668 falls back to new_acc >= old_acc when challenger lacks balanced_accuracy (torch models omit it); test_retrainer_gate.py:22-33 fixtures exercise only the fallback. Fix: refuse when new_skill is None; fix fixtures.
F3 MEDIUM CONFIRMED: evaluator.py:133-139 accuracy=0.0 at sample_size 0; :169-179 flat prediction counts as directional match when |ret|<=5bp so constant classifier scores flat base rate as "directional accuracy"; detect_accuracy_drift compares raw-accuracy halves -> drift = base-rate shift. Fix: None for absence; balanced accuracy for drift.
F4 MED-LOW CONFIRMED: learning_metrics.py:124-126 daily sign agreement from as few as 25 obs; era value unweighted mean of daily; no SE/n in payload (models.py:117-121); LearningMetrics.tsx:103-107 prints "49.8%". Fix: pool over moved obs, Wald SE, show n.
F5 LOW CONFIRMED: 7 macOS duplicate " 2.py" test files collected locally (identical to originals); pyproject has no ignore. Fix: addopts = "--ignore-glob=* 2.py". NOTE: pytest --collect-only hung in this environment (killed).
F6 LOW: test_risk_metrics_writer.py:240-242 sleep(0.05); assert not task.done() vacuous; test_risk_account_sync.py:56-57 fixed sleep with unused _wait_for_calls helper.
Gaps: validator.py has no staleness/outlier/duplicate checks; detect_gaps log-only; equity staleness invisible (/portfolio/summary no as-of; metrics.py:245 treats last snapshot date as today); no test pins normaliser absent->undefined for ModelInfo; registry reconcile after DB restore opens spurious era (PLAUSIBLE).
Verified good: WITHHOLD policy end-to-end; RiskDashboard em-dashes; live_replay embargo strict; skill metrics on served _proba; test_trading_pipeline negative assertion sound.

### fullstack-product-designer


Build: tsc -b hung (environment stall, not code); last verified dist 2026-09-12; eslint not installed (npm run lint exit 127) — the lint gate does not exist.
F1 HIGH CONFIRMED: AuditLog.tsx:131-150 renders component/event_type/risk_level/user; API returns service/action/entity_type/entity_id/user_id (audit.py:47-57); filters send component/risk_level which API ignores. Fix: map fields; send service param.
F2 HIGH CONFIRMED: Trading page: normalize.ts:130-132 expects filled_quantity/filled_avg_price absent from OrderResponse (api/schemas/orders.py:38-59) -> "0/qty"; status partial_fill vs UI 'partial'; useMarketData default '1D' vs enum "1d" = 422 every load (hooks.ts:201); symbol list US equities only (Trading.tsx:14). Fix: fill summary on OrderResponse; align literals; symbols from settings.
F3 HIGH CONFIRMED: dead REST feed renders live: no page reads isError/dataUpdatedAt; header chip reflects WS only (Header.tsx:42-54) which carries only prices nobody displays; after 10 failed reconnects (websocket.ts:97) reconnection stops forever silently. Fix: global "data as of / STALE" banner; capped unlimited backoff.
F4 MED-HIGH CONFIRMED: ML page fabricates zeros (normalize.ts:193-196; ModelPerformance.tsx:100-116); nullable expected_return shown as "+0.00%" green (normalize.ts:225); Retrain button posts to a 501 endpoint with no error state.
F5 MEDIUM CONFIRMED: client.ts:22-25 401 interceptor also catches wrong-password 401 from /auth/token -> full reload wipes login error; expired JWT = hard reload no return path.
F6 MEDIUM CONFIRMED: JWT in WS query string (websocket.ts:26-27) for an endpoint that never checks it.
Lower: stop orders impossible from UI (no stop_price field); timestamps browser-local no zone; fixed 240px sidebar; nginx no security headers; AuditLog fragments lack keys.
Inspectability: prediction partial (API yes, DB has p_flat_raw/explanation/features_used unexposed); margin vs gate logged only; risk rule outcomes exist nowhere durable at time T (RiskBreached 0/0 to an alerts channel UI never subscribes); order yes but prediction_id never populated; fills: no UI, no API endpoint; P&L per trade nowhere; /audit/decision-trace always 404 (log_trade_decision has no callers).
Gaps: durable per-decision record; time-indexed drill-down UI; orders/alerts WS channels + bell dead (no addAlert caller); admin kill switch has no UI; UI never decodes JWT role; no confirmation on manual order/cancel.
Verified good: TRADING_MODE display-only end to end; withheld stats render honestly ("ledger unreconciled" vs "no closed trades"); VaR/vol/breaker UNKNOWN never zero; learning metrics show n/t-stat; risk-rule names match; nginx SPA fallback + /api proxy with WS upgrade.

### ai-agent-engineer


F1 HIGH: all 16 agents grant Bash, 14 grant Edit/Write incl. blockchain analyst ("propose, do not build"), lawyer, product-lead; no .claude/settings.json deny list. Fix: drop Bash from lawyer/product-lead/blockchain; add deny list for Read(./.env), Bash(railway up*), Bash(git push*); reword FLEET "read-only" claim.
F2 HIGH: no read-only API role; POST/DELETE /orders need only any JWT (orders.py:93-104); only role check is admin.py:48.
F3 MEDIUM: ai-agent-engineer told to load claude-api skill but tools allowlist excludes Skill. Fix: add Skill.
F4 MEDIUM: ownership overlaps vs "one writer per file set": services/execution (PM vs TSE), position_sizer (PM vs risk), settings.py (4 roles), infra files (security vs sre), audit (3), funding_watch (quant vs crypto), training/* (quant vs ML), continuous_learning metrics (eval vs ML). Fix: Shared-paths table with one primary writer.
F5 MEDIUM: 501 stubs PUT /config/trading-mode (config_router.py:74-90), PUT /risk-rules (:41-59), breaker reset (risk.py:120-137) — no prompt says they must stay 501/be deleted. Fix: security-engineer rule.
F6 LOW: sre-devops "Railway drops logs above 500 lines/sec" unsourced.
Verified: all factual claims in prompts current; house rules byte-identical; frontmatter valid; all 31 referenced paths exist.
Gaps: no coordination mechanism (hand-run protocol); no agent-output evaluation instrument/finding schema; LLM analyst layer absent (no scoped credential possible until F2); lawyer/blockchain lack web access (say "repo evidence only").

### blockchain-data-analyst


Status: NO on-chain ingestion (confirmed; aux_market.py:46-47,152-208; funding_watch.py:163-177). Dead placeholder services/feature_engineering/alternative/onchain.py returns constants (0 / MVRV 1.0) with datetime.now — if ever wired, constants would look like fresh data (LOW, CONFIRMED dead). aux_market_state has no available_at column.
F2 MEDIUM PLAUSIBLE: scripts/backfill_aux_market.py:123,129 stamps VIX/SPY at Yahoo bar time (midnight exchange-local, yahoo_provider.py:94-100) while live stores intraday last bar (aux_market.py:187-208): replay look-ahead up to ~16h + train/serve skew (features currently null/ignored by champions).
F3 LOW CONFIRMED: aux_features.py:159-165 returns 0.0 for missing data, indistinguishable from real zero.
Specs: W1 stablecoin_issuance_7d_plus; W2 eth_cex_netflow_24h_minus (frozen labels); W3 unlock_pressure_7d_minus. Pipeline: append-only onchain_observation table with available_at = max(block_time+finality, observed_at); shadow-only until a watch passes; test pins no onchain_ key reaches features.

### securities-crypto-lawyer


Assumed facts confirmed in repo: paper mode $100 (settings.py:39-40); live brokers only in live mode (worker.py:39-41); Alpaca paper (settings.py:32); global ccxt.binance not Binance.US (ccxt_broker.py:43-46, providers/ccxt_provider.py:73); Railway public URL login-gated; no self-service registration (scripts/create_user.py); orders/fills carry no user_id (core/models/orders.py:19,85) = single shared account. NOT in repo: operator jurisdiction, entity, Binance entity, Railway region.

Issues: (1) Binance eligibility/geo-restriction — medium now, blocking at launch. (2) Advice/publisher exclusion: BUY/SELL labels with confidence (normalize.ts:205-208, PredictionView.tsx:72-96, Dashboard.tsx:72-74, predictions.py:55-62) — low now, high at launch. (3) Outside money/pooling/custody — gating at launch. (4) Token classification SOL/ADA/DOT — low/medium. (5) Derivatives boundary (perps research only) — low; high if traded. (6) Data/API ToS — medium now (Yahoo in LIVE path despite "backfill only" docstring yahoo_provider.py:1; aux_market.py:45,185-216). (7) No disclaimer anywhere in tracked files; backtests unlabelled as hypothetical (Backtesting.tsx:176-181); PAPER badge exists (Header.tsx:38). (8) Records/data protection: audit_logs keeps user_id+IP (audit.py:28,35); retention no-op on plain Postgres. (9) Tax/PDT: no tax-lot logic.

Disclosures to add now: Login/Layout footer private-research statement; rename BUY/SELL/HOLD to "model lean: long/short/flat" + estimate disclaimer; backtest/equity "simulated performance" label; mode badge tooltips; README/NOTICE on data sources.

Next action: operator fact sheet (residence, entity, Binance entity, Railway region) -> fintech/securities counsel in residence jurisdiction; first question = Binance.com use from jurisdiction/server region; where personal use becomes advice/pooling.


## Product-lead synthesis (ran on the fifteen reports above)

### Roadmap — items not built on 2026-11-02, in priority order

Ranking rule: money-path integrity and honesty instruments, then signals, then UI; legal
gates any outside user. Dates are earliest-possible, not promises.

**Next session (after today's build set is verified in production)**
1. Working exit path: dollar-sized exits leave dust or unintended shorts, paper accepts naked shorts, stops never registered (portfolio-manager, risk sign-off). Done: one paper soak where the breaker trips and the book auto-flattens; no negative paper position over 7 days. M.
2. Silence watchdog + external dead-man: `/healthz` unconditional, a 07-21-class stall undetectable, alerting dies with the worker (sre-devops). Done: a simulated feed stall pages Slack within 10 min; a dead worker is paged externally within 15. S.
3. Verdicts on a clock: the Q3-2026 funding quarter closed 09-30 with no verdict in the ledger (evaluation-specialist, scheduled). Done: Q3 verdict recorded; every quarter-end auto-writes a dated entry. S.
4. Readiness-panel honesty: ~60% chance a registry of 14 null factors goes green; current quarter counted; one stamp counts as a quarter; new-listing artefact (quant-researcher). Done: a simulated all-null registry goes green under 5% of the time; per-quarter n shown. S.
5. Risk verdict and venue facts persisted: no risk verdict, venue order id, venue time or fee has ever been stored; decision-trace 404 (trading-systems-engineer, compliance sign-off). Done: one query per fill returns all four. M.
6. Venue-true paper costs: zero commission, no lot step, no min-notional; August's -1.4% is roughly -6% at live fees (crypto-market-specialist, risk sign-off). Done: 10 bps per side on crypto pairs; no sub-minimum order generated. S.
7. Feature warm-up after restart: ~200 bars of NaN-coerced-to-zero features traded on and learned from after every deploy (data-engineer). Done: zero unwarmed predictions in a restart drill. S.
8. Backups, restore drill, pruning (sre-devops; operator enables). Done: a scratch restore reproduces the equity series to the cent. S.
9. Fleet hygiene: finding schema, agent-output verification ledger (ai-agent-engineer). S.

**Before Stage 4 live** (by the platform's own bar the first watched factor cannot clear four unseen quarters before 2027-06-30)
10. Live Binance path: precision, symbol per order, non-USDT balances valued (crypto). Done: testnet $5 round trip fills, polls and cancels. M.
11. Live lifecycle edges: transport exception keeps the order submitted, partial-then-cancel keeps filled quantity, EXPIRED handled, API cancel reaches the engine (trading-systems). Done: 20 random-kill drills with zero unmatched closes. M.
12. Risk control table: single-order cap enforced, daily-drawdown rule, VaR horizon, side-aware pre-trade check, scheduled position reconcile (risk-manager). M.
13. Live-grade security: hashed lockfile, SHA-pinned Actions, security headers, JWT at most 60 min with revocation, proxy IP trust (security-engineer). M.
14. Append-only record: triggers on audit and fills, order status history, audit write failures page someone (compliance-lead). S.
15. Operator hygiene, no code: rotate `JWT_SECRET` after today's WebSocket change, the Slack webhook and the Redis password; confirm broker keys trade-only, no-withdrawal, IP-locked; backups on.

**Before any outside user**
16. Fact sheet to licensed counsel: residence, entity, Binance entity, Railway region; Binance.com eligibility is the first question. Gates this whole group.
17. Disclosures: private-research footer, "model lean" instead of BUY/SELL, "simulated" on backtests and equity, data-source notice (fullstack, compliance sign-off). Cheap; may be pulled forward. S.
18. Per-user attribution and real retention: user id on orders and fills, actor on manual orders, a prune job since retention is a no-op on plain Postgres. M.
19. UI truth beyond today: staleness banner, Trading page fill summary and the 422 on load, login interceptor, reconnect cap, kill-switch UI with confirmation. M.

**Research-process (each needs a dated registration before code)**
20. Promotion-gate decontamination (ml-engineer with evaluation-specialist): register first, fix in the same session; re-score the current champion through the clean gate and record whatever it says. One carve-out ships without registration: refusing promotion when balanced accuracy is absent, because it can only refuse, never grant. M.
21. Label span equals served horizon, with a test pinning it; served balanced accuracy and live IC re-measured at 14 days, no requirement to beat chance. M.
22. CatBoost inside the retrain loop plus a feature-name drift check. S.
23. Aux stamping and cutoff fixes, then re-run the regime A/B. S.
24. Venue-aware costs and a buy-and-hold control in the research harness (the 08-02 method note). S.
25. Calibration v2 on raw p(flat) (prerequisite shipped 08-28). M.
26. On-chain watches W1-W3, shadow-only, only after the registry produces its first four-quarter verdict. L.

### Claims statement (as of 2026-11-02)

CAN be said: deployed on Railway in paper mode with $100 notional since 2026-07-19; kill switch and restart drills passed live (07-19), flatten routes through the tracked path (09-24), alerts reach Slack (07-23); equity is broker-authoritative and survives restarts (07-29); the learning loop runs end to end and its gate has declined challengers on real numbers; the research process pre-registers, records nulls and retracts in place (seven consecutive nulls 07-30 to 08-02, two retractions, one registered MISS executed as written); fourteen registered factor watches run as a point-in-time append-only adjudicator and none trades; honesty instruments exist (withheld trade statistics, balanced accuracy with a chance floor, live IC by model era, feature-drift PSI, em-dash never zero).

CANNOT be said: that it has edge (live IC -0.0045 on 82,452 observations; the served ensemble is a constant classifier); that it has made money or has any performance number (Sharpe 1.17 retired; August -1.4% against a rising do-nothing benchmark; realised -$1.53); that trade-level statistics are right (withheld; ~0.6% of closes missing, cause targeted today but unverified in production); that the risk stack would protect a live account (breaker never tripped; no auto-exit path existed before today); that it is ready for capital (readiness unmet; earliest four-quarter clearance 2027-06-30); that paper results transfer to Binance (no fees or lot steps in paper; three confirmed live-path defects); that it is safe for anyone but the operator (one shared account, no attribution, no disclaimers, Binance.com eligibility unresolved; JWTs were logged through WebSocket URLs until today's fix and the secret is not yet rotated); that the Q3-2026 watch quarter has been judged.

### Scope decisions
(a) Exits and the cost model before the label-horizon fix: exits are the risk-manager's BLOCK condition and a money-path invariant; the cost model is an honesty instrument (soak costs understated ~4x). The label mismatch mis-trains a model measured to have nothing to say at 5 minutes; fixing it cannot create edge and the live instruments already score served output at the correct horizon. It goes to the registered track with one hypothesis stated in that registration: the regressor was trained on ~25-minute labels, and the only non-null gross signal ever seen was at 60 minutes (07-30).
(b) The BLOCK stands. Lift condition: findings 1-4 fixed with regression tests plus one paper soak in which the breaker trips and the book auto-exits. "Taking risk" is moot on the facts: there is no edge to take risk on, and Stage 4 cannot open before 2027-06-30 by the registered bar. The operator keeps the final human say on `TRADING_MODE` after the lift condition, not instead of it.
(c) Register first, fix in the same session (promotion gate). The 08-28 embargo amendment shows what an unregistered gate change would have done.

### Human roles needed
Licensed securities/crypto counsel before any outside user or pooling, and before Stage 4 if the venue is Binance.com. The operator as risk sign-off and credential holder (rotations pending now). A named accountable human actor for admin actions and promotions before any outside user. An accountant for tax lots before Stage 4 funding. And a scheduled, non-human judge for registered verdicts, because two late verdicts show the ledger cannot depend on someone opening a session.
