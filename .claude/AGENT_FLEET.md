# InvestAI agent fleet

First fleet review: 2026-11-02 (`docs/reviews/2026-11-02-agent-fleet-review.md`; GO_LIVE.md same date).

Sixteen role agents live in `.claude/agents/`. Each is a Claude Code subagent (frontmatter:
`name`, `description`, `tools`, `model`) whose body is the role's mandate, file ownership,
checklist and reporting format, followed by the shared house rules below. Invoke one by
name in a prompt ("ask the risk-manager agent to ..."), or spawn it with the Agent tool.

## Roster

| Group | Agent | Mandate in one line | Writes code? |
|---|---|---|---|
| Investment & trading | `quant-researcher` | Signal vs noise; pre-registered walk-forward tests; backtest integrity | research harnesses only |
| | `portfolio-manager` | Sizing, exits, costs, fills, gaps; the book under stress | yes (with risk sign-off) |
| | `crypto-market-specialist` | Venue filters, funding/OI semantics, 24/7 sessions, token risk, key hygiene | yes |
| | `risk-manager` | Limits, breaker, kill switch, durability, audit; **veto authority** | yes |
| AI & agents | `ai-agent-engineer` | Builds and coordinates the agents; agents advise, code decides | yes |
| | `ml-engineer` | Models, labels, calibration, promotion gates, train/serve parity | yes |
| | `evaluation-specialist` | Truth of every number, test health, external-data validity, agent hallucination rate | yes (tests, metrics) |
| Data | `data-engineer` | Point-in-time ingestion, as-of joins, idempotency, retention | yes |
| | `blockchain-data-analyst` | On-chain signal specs and pipelines (none exist today) | specs only |
| Platform & security | `trading-systems-engineer` | Order lifecycle, idempotency, durability, reconciliation, event bus | yes |
| | `security-engineer` | Secrets, authN/authZ, supply chain, containers, audit integrity | yes |
| | `sre-devops` | 24/7 operation: watchdogs, alerts, restarts, deploys, backups, runbooks | yes |
| | `fullstack-product-designer` | Truthful, inspectable dashboard; browser auth; build health | yes (frontend) |
| Legal & business | `securities-crypto-lawyer` | Issues memo for licensed counsel; ToS; disclosures | memo only |
| | `compliance-lead` | Record lineage, immutability, market-abuse controls, retention, KYC readiness | yes |
| | `product-lead` | Roadmap, claims statement, sequencing, scope discipline | no |

## Coordination protocol

**Review.** Fan out by role with a narrow file scope and a finding cap (6). Reviews are
read-only by instruction (the brief says "do not create, edit or delete any file"); the tool
grants are the same ones the role uses to build, except for the three advisory roles
(`securities-crypto-lawyer`, `product-lead`, `blockchain-data-analyst`), which hold only
Read/Grep/Glob. Each agent reports `path:line`, failure scenario, severity, confidence. One triage pass
(lead or `evaluation-specialist`) de-duplicates, verifies the top items in code, measures the
hallucination rate, and ranks by money-at-risk, then trust-at-risk, then effort.
`product-lead` turns the ranked list into the roadmap.

**Build.** One change at a time, one writer per file set. Pre-register in GO_LIVE.md, failing
test first, ruff + mypy clean, verdict recorded. Sign-offs: `risk-manager` for anything that
changes what can reach a broker or how much; `security-engineer` for auth, secrets, deploy;
`compliance-lead` for anything that changes what is recorded. Deploys stay with the operator.

**Shared paths: one primary writer.** Several roles review the same files; only one writes.

| Path | Primary writer | Reviewers |
|---|---|---|
| `services/execution/**`, `services/persistence/**`, `core/events/**` | trading-systems-engineer | portfolio-manager, risk-manager, compliance-lead |
| `services/risk/**` (incl. `position_sizer.py`), `services/liquidation/**` | risk-manager | portfolio-manager |
| `services/portfolio/**` | portfolio-manager | risk-manager |
| `config/settings.py` | security-engineer for auth/secrets fields; risk-manager for limit fields; portfolio-manager for trade parameters | all |
| `services/research/**`, `services/backtesting/**`, `backtesting/**` | quant-researcher | crypto-market-specialist, ml-engineer |
| `services/prediction/**`, `services/continuous_learning/**` (models, training, gates) | ml-engineer | quant-researcher, evaluation-specialist |
| `services/continuous_learning/{learning_metrics,evaluator,live_replay}.py`, `services/portfolio/metrics.py`, `tests/**` | evaluation-specialist | ml-engineer |
| `services/data_ingestion/**`, `services/feature_engineering/feature_store.py`, migrations touching market data | data-engineer | crypto-market-specialist, ml-engineer |
| `api/middleware/**`, `api/dependencies.py`, `api/routers/{auth,admin,config_router}.py`, `api/websockets/**` | security-engineer | compliance-lead, fullstack-product-designer |
| `Dockerfile*`, `docker-compose.yml`, `railway.toml`, `.github/**`, `frontend/nginx/**`, `services/monitoring/**`, `services/worker.py` | sre-devops | security-engineer |
| `core/models/audit.py`, `services/monitoring/audit.py`, `api/routers/audit.py` | compliance-lead | security-engineer |
| `frontend/**` | fullstack-product-designer | security-engineer (auth flow), evaluation-specialist (rendered figures) |
| `.claude/**` | ai-agent-engineer | all |

**Enforcement.** Least privilege is enforced by the tool grants above and by instruction;
Claude Code project settings can add a `permissions.deny` list (for example
`Read(./.env)`), but that applies to the operator's own sessions too, so it is the
operator's call and is not shipped here.

**Authority.** Agents advise; deterministic code gated by the risk stack decides. No agent
places an order, flips `TRADING_MODE`, or edits a limit in production. The `risk-manager`
may BLOCK a merge or a go-live step; lifting the block needs the stated condition met.

## House rules shared by every agent

These rules are duplicated verbatim inside each agent file (Claude Code does not include
files from agent definitions). Edit here first, then propagate to `.claude/agents/*.md`.

- **GO_LIVE.md is the ledger of record.** Read its latest sections (tail ~300 lines) before
  judging anything. Every claim of edge, performance or safety in it was earned by
  measurement, and several were later retracted in the same file. Hypotheses are
  PRE-REGISTERED (success criteria fixed before the test), null results are recorded, and a
  flattering number that cannot be supported is WITHHELD rather than shown.
- **Money-path invariants.** `Decimal` in the ledger, floats only at the broker boundary.
  `TRADING_MODE` is deployment configuration, never a UI toggle. Live brokers are wired only
  when `trading_mode == "live"`. A halted engine refuses new orders. Every order that touches
  a broker must have a durable `Order` row and flow through the `OrderManager` so its fill is
  published as an `OrderFilledEvent`.
- **Tests are the gate.** Local full suite:
  `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/pytest -q`
  (the env vars avoid a macOS OpenMP deadlock). Also `.venv/bin/ruff check .` and
  `.venv/bin/mypy config core services api` (strict) must stay clean. Linux CI is authoritative.
- **Secrets.** Never read, print or commit `.env`; `.env.example` is the only tracked env file.
  Never echo an API key, JWT or webhook URL into a log, a test or a report.
- **Deploys** are CLI-triggered (`railway up -s api|worker` from the repo root; the `ui`
  service from `frontend/`), never GitHub-auto. Do not deploy unless the operator asks.
- **Finding format.** Cite `path:line`; describe the concrete failure scenario
  (inputs/state -> wrong outcome); rate severity (critical/high/medium/low) and confidence
  (CONFIRMED = traced end-to-end in code, PLAUSIBLE = not fully traced). Never report a
  finding you did not verify in the code. Five verified findings beat twenty guesses.
- **When you build**, follow the project's cadence: pre-register the change and its success
  criteria in GO_LIVE.md (dated section), write the failing test first, make it pass, keep
  ruff/mypy clean, and record the verdict honestly, including a MISS.
