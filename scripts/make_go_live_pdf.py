"""Generate the 'Steps to Go Live' PDF from the GO_LIVE runbook (current state)."""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = "InvestAI_Go_Live_Steps.pdf"

NAVY = colors.HexColor("#0f2a4a")
BLUE = colors.HexColor("#2563eb")
GREEN = colors.HexColor("#15803d")
AMBER = colors.HexColor("#b45309")
RED = colors.HexColor("#b91c1c")
GREY = colors.HexColor("#4b5563")
LIGHT = colors.HexColor("#f3f4f6")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], textColor=NAVY, fontSize=24, spaceAfter=2, alignment=TA_LEFT)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], textColor=GREY, fontSize=11, spaceAfter=2)
STAGE = ParagraphStyle("STAGE", parent=styles["Heading2"], textColor=NAVY, fontSize=14, spaceBefore=14, spaceAfter=4)
SECT = ParagraphStyle("SECT", parent=styles["Heading3"], textColor=NAVY, fontSize=11, spaceBefore=8, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontSize=9.5, leading=14, spaceAfter=4)
ITEM = ParagraphStyle("ITEM", parent=BODY, spaceAfter=2)
CODE = ParagraphStyle(
    "CODE", parent=styles["Code"], fontName="Courier", fontSize=8.5, leading=12,
    backColor=LIGHT, borderPadding=5, textColor=colors.HexColor("#111827"), spaceBefore=2, spaceAfter=6,
)
CALL = ParagraphStyle("CALL", parent=BODY, fontSize=9.5, leading=13)


def badge(text: str, color: colors.Color) -> str:
    return f'<font color="{color.hexval()}"><b>[ {text} ]</b></font>'


def callout(html: str, bg: str, border: colors.Color) -> Table:
    p = Paragraph(html, CALL)
    t = Table([[p]], colWidths=[6.7 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("BOX", (0, 0), (-1, -1), 0.75, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def bullets(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, ITEM), leftIndent=12, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=14,
    )


def steps(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, ITEM), leftIndent=14) for t in items],
        bulletType="1", leftIndent=18,
    )


def rule() -> HRFlowable:
    return HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#d1d5db"), spaceBefore=6, spaceAfter=6)


story: list = []

# ---- Header ----
story.append(Paragraph("InvestAI Trading Platform", H1))
story.append(Paragraph("Steps to Go Live &mdash; staged runbook", SUB))
story.append(Paragraph("Generated 2026-06-22 &nbsp;|&nbsp; branch merged to <b>main</b> (083145d)", SUB))
story.append(Spacer(1, 8))

story.append(callout(
    f"{badge('STATUS', GREEN)} &nbsp; Code remediation is <b>complete and merged to main</b>: "
    "100 tests green, CI green, tree probabilities calibrated, live-venue trading gated to live mode, "
    "and the production Docker image dry-run verified end-to-end (build &rarr; migrate &rarr; serve &rarr; "
    "DB+Redis &rarr; <font face='Courier'>/healthz</font> 200). What remains is operational: the Railway "
    "deploy, then the edge and paper-soak gates below.",
    "#ecfdf5", GREEN))
story.append(Spacer(1, 6))
story.append(callout(
    f"{badge('VERDICT', AMBER)} &nbsp; <b>Paper-trade only after Stage 1.</b> Do <b>not</b> risk real money "
    "until Stage 2 (proven edge) <b>and</b> Stage 3 (paper soak) both pass. Stages are sequential &mdash; do not skip.",
    "#fffbeb", AMBER))
story.append(Spacer(1, 8))

# ---- Legend ----
story.append(Paragraph(
    f"{badge('DONE', GREEN)} verified in dev &nbsp;&nbsp; {badge('YOU RUN', BLUE)} operator action "
    f"&nbsp;&nbsp; {badge('HARD GATE', RED)} must pass before proceeding", BODY))
story.append(rule())

# ---- Stage 0 ----
story.append(Paragraph(f"Stage 0 &mdash; Code correctness &nbsp; {badge('DONE', GREEN)}", STAGE))
story.append(bullets([
    "100 unit/integration/e2e tests green; <font face='Courier'>ruff</font> clean; CI (lint + pytest) in place.",
    "Migrations verified on real PostgreSQL <b>and</b> real TimescaleDB (migration 0002 builds 4 hypertables "
    "+ retention; safe no-op on plain Postgres).",
    "Full pipeline verified over real Redis (rebalance &rarr; risk &rarr; execution &rarr; fill &rarr; risk feedback).",
    "Tree-model probabilities calibrated; auth path verified on real Postgres.",
    "Frontend login/auth gate, FE&harr;BE paths reconciled, production nginx build verified (tsc strict + vite).",
    "Safety invariant: paper/backtest mode <b>cannot reach a live venue</b> (gated at broker wiring, router, "
    "execution-engine hard guard, and CCXT sandbox).",
]))

# ---- Stage 1 ----
story.append(Paragraph(f"Stage 1 &mdash; Deploy + smoke test on real infra &nbsp; {badge('YOU RUN', BLUE)}", STAGE))

story.append(Paragraph("Part A &mdash; what only you can do (identity / secrets)", SECT))
story.append(steps([
    "Authenticate the Railway CLI on this machine: <font face='Courier'>railway login</font> (opens a browser).",
    "Rotate keys are already done; at deploy time put the <b>5 secrets</b> into Railway&rsquo;s secret store "
    "(the local <font face='Courier'>.env</font> is never shipped): "
    "<font face='Courier'>ALPACA_API_KEY, ALPACA_SECRET_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, JWT_SECRET</font>.",
]))

story.append(Paragraph("Part B &mdash; provision &amp; deploy", SECT))
story.append(steps([
    "Create a Railway project; add <b>Postgres</b> + <b>Redis</b>. For TimescaleDB hypertables use a "
    "TimescaleDB image/template; on vanilla Postgres, migration 0002 safely no-ops.",
    "Create <b>two services from this repo</b> (same <font face='Courier'>Dockerfile</font>): "
    "<b>api</b> (Dockerfile CMD <font face='Courier'>uvicorn api.main:app</font>; healthcheck "
    "<font face='Courier'>/healthz</font>; <font face='Courier'>releaseCommand = alembic upgrade head</font> "
    "runs automatically) and <b>worker</b> (override start to <font face='Courier'>python -m services.worker</font>).",
    "Set environment variables on both services (table below).",
    "Deploy. The release command runs migrations; the api service starts and the healthcheck must go green.",
]))

env_rows = [
    ["Variable", "Value / source"],
    ["DATABASE_URL", "${{Postgres.DATABASE_URL}}  (app converts to asyncpg itself)"],
    ["REDIS_URL", "${{Redis.REDIS_URL}}"],
    ["ALPACA_API_KEY / ..._SECRET_KEY", "rotated secret (Railway secret store)"],
    ["BINANCE_API_KEY / ..._SECRET_KEY", "rotated secret (Railway secret store)"],
    ["JWT_SECRET", "rotated 64-char secret (refused if default in live mode)"],
    ["ALPACA_BASE_URL", "https://paper-api.alpaca.markets  (paper)"],
    ["TRADING_MODE", "paper"],
    ["INITIAL_CAPITAL", "100.00"],
]
env_tbl = Table(env_rows, colWidths=[2.3 * inch, 4.4 * inch])
env_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (0, -1), "Courier"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(Spacer(1, 2))
story.append(env_tbl)
story.append(Spacer(1, 6))

story.append(Paragraph("Part C &mdash; verify the deployment", SECT))
story.append(steps([
    "Create a login user: <font face='Courier'>railway run python scripts/create_user.py "
    "you@x.com '&lt;pw&gt;' admin</font>",
    "Smoke test the pipeline: <font face='Courier'>railway run python scripts/smoke_test_pipeline.py</font> "
    "&rarr; must print <font face='Courier'>SMOKE TEST PASSED</font>.",
    "Sanity: <font face='Courier'>GET /healthz</font> 200; <font face='Courier'>GET /api/v1/health</font> "
    "shows db+redis ok; <font face='Courier'>POST /api/v1/auth/token</font> returns a JWT; an authed request succeeds.",
]))
story.append(callout(
    "<b>Already proven locally (deploy dry-run):</b> the exact production image builds, runs 0001+0002 "
    "(4 hypertables), serves, connects DB+Redis, and answers /healthz 200, /api/v1/health "
    "{database:ok, redis:ok}, /api/v1/auth/token 401 on bad creds. A root .dockerignore keeps "
    "secrets/venv/git out of the image (5.07GB &rarr; 2.87GB).",
    "#eff6ff", BLUE))

# ---- Stage 2 ----
story.append(Paragraph(f"Stage 2 &mdash; Prove the strategy has edge &nbsp; {badge('HARD GATE', RED)}", STAGE))
story.append(steps([
    "<b>Ingest real history for a small universe</b> (harness reads a <font face='Courier'>close</font> "
    "column; provider auto-selected by symbol, both keyless):<br/>"
    "<font face='Courier'>python scripts/fetch_history.py AAPL --start 2015-01-01 --out aapl.csv</font><br/>"
    "... repeat for MSFT, SPY, and/or BTC/USDT.",
    "<b>Run the gate across the whole universe at once</b> (net of costs):<br/>"
    "<font face='Courier'>python scripts/validate_model.py aapl.csv msft.csv spy.csv --cost-bps 5</font><br/>"
    "(prefix <font face='Courier'>DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib</font> on macOS; "
    "libgomp1 is already in the Docker image).",
]))
story.append(callout(
    f"{badge('GATE', RED)} &nbsp; <b>Per symbol:</b> hit-rate &gt; 52% AND Sharpe &gt; 0.5 AND stability &gt;= 0.75 "
    "(net-of-cost, non-overlapping holding periods). <b>Overall:</b> a green light needs a majority of "
    "<b>3+ symbols</b> to pass &mdash; a single-symbol pass prints <i>necessary but NOT sufficient</i>. The null "
    "self-test (zero-drift random walk, run with no args) must report <b>NO demonstrable edge</b>. Anything "
    "else means <b>STOP &mdash; do not trade.</b>",
    "#fef2f2", RED))

# ---- Stage 3 ----
story.append(Paragraph(f"Stage 3 &mdash; Paper-trading soak (weeks) &nbsp; {badge('YOU RUN', BLUE)}", STAGE))
story.append(Paragraph("Run the worker in paper mode and watch for <b>weeks</b>:", BODY))
story.append(bullets([
    "Orders fill and the equity curve is sane.",
    "The <b>circuit breaker actually trips</b> on a drawdown day (the risk engine&rsquo;s "
    "<font face='Courier'>sync_account()</font> must be fed equity on a 30&ndash;60s timer).",
    "<font face='Courier'>reconcile_positions()</font> shows <b>no drift</b> vs the broker.",
    "No crashes or stuck consumers; alerts fire as expected.",
]))

# ---- Stage 4 ----
story.append(Paragraph(f"Stage 4 &mdash; Tiny live, supervised &nbsp; {badge('YOU RUN', BLUE)}", STAGE))
story.append(bullets([
    "Start at <b>$100</b> with hard caps and manual supervision.",
    "<b>Rehearse the kill switch before funding:</b> <font face='Courier'>emergency_flatten()</font> cancels "
    "open orders, flattens positions, and halts (unit-tested &mdash; rehearse it live).",
    "Set <font face='Courier'>TRADING_MODE=live</font> only at this stage; scale only if results match expectations.",
]))

# ---- Emergency ----
story.append(Paragraph("Emergency procedures", STAGE))
story.append(bullets([
    "<b>Halt + flatten:</b> call <font face='Courier'>emergency_flatten()</font> "
    "(or <font face='Courier'>halt()</font> to just stop new orders).",
    "<b>Drawdown auto-stop:</b> the circuit breaker trips at <font face='Courier'>circuit_breaker_loss_pct</font> "
    "(default 7% daily) once <font face='Courier'>sync_account()</font> is feeding it &mdash; verify in Stage 3.",
]))
story.append(Spacer(1, 6))
story.append(callout(
    "<b>Operational glue to wire at deploy:</b> have the worker call "
    "<font face='Courier'>risk.sync_account(broker.get_account()['equity'])</font> on a 30&ndash;60s timer and "
    "<font face='Courier'>risk.reset_daily()</font> at session open. The methods are tested; "
    "the periodic call is deploy glue.",
    "#f9fafb", GREY))


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(0.9 * inch, 0.55 * inch, "InvestAI - Steps to Go Live - confidential")
    canvas.drawRightString(7.6 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.9 * inch, rightMargin=0.9 * inch, topMargin=0.8 * inch, bottomMargin=0.8 * inch,
    title="InvestAI - Steps to Go Live", author="InvestAI",
)
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print(f"wrote {OUT}")
