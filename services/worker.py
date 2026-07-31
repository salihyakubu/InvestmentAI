"""Main worker process -- initialises and runs all platform services concurrently."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import structlog

from config.settings import Settings, get_settings
from core.events.base import EventBus

logger = structlog.get_logger(__name__)


def _build_brokers(settings: Settings) -> dict[str, Any]:
    """Build the broker registry for the configured trading mode.

    Live venues are wired ONLY in live mode. Credential presence is not
    sufficient: in paper/backtest mode we must never reach a real venue, even if
    production keys happen to be in the environment. This matters most for crypto
    -- CCXTBroker talks to the real Binance spot venue with no testnet/sandbox,
    so wiring it outside live mode would route "paper" crypto orders to real
    money. Paper/backtest therefore keep only the PaperBroker.
    """
    from decimal import Decimal

    from services.execution.brokers.paper_broker import PaperBroker

    paper_broker = PaperBroker(initial_cash=Decimal(str(settings.initial_capital)))
    brokers: dict[str, Any] = {"paper": paper_broker}

    if settings.trading_mode != "live":
        logger.info("worker.live_brokers_disabled", trading_mode=settings.trading_mode)
        return brokers

    alpaca_key = settings.alpaca_api_key.get_secret_value()
    if alpaca_key and alpaca_key != "your_alpaca_api_key":
        try:
            from services.execution.brokers.alpaca_broker import AlpacaBroker

            brokers["alpaca"] = AlpacaBroker(settings=settings)
        except Exception as exc:
            logger.warning("alpaca_broker.init_failed", error=str(exc))

    binance_key = settings.binance_api_key.get_secret_value()
    if binance_key and binance_key != "your_binance_api_key":
        try:
            from services.execution.brokers.ccxt_broker import CCXTBroker

            brokers["ccxt"] = CCXTBroker(settings=settings)
        except Exception as exc:
            logger.warning("ccxt_broker.init_failed", error=str(exc))

    return brokers


async def _health_app(
    scope: dict[str, Any],
    receive: Any,
    send: Any,
) -> None:
    """Minimal ASGI app exposing /healthz so platform healthchecks (Railway)
    can verify worker liveness. The worker has no other HTTP surface."""
    if scope["type"] != "http":
        return
    ok = scope.get("path") in ("/", "/healthz")
    await send(
        {
            "type": "http.response.start",
            "status": 200 if ok else 404,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": b'{"status":"alive","service":"worker"}' if ok else b"{}",
        }
    )


def _start_health_server() -> asyncio.Task[None] | None:
    """Serve the liveness endpoint on $PORT when the platform provides one.

    Railway injects PORT and (via railway.toml) healthchecks /healthz; without
    this the worker deployment could never pass that check. Locally PORT is
    usually unset and this is a no-op. Failures are non-fatal: liveness
    reporting must never take down the trading worker itself.
    """
    port = os.environ.get("PORT")
    if not port:
        return None
    try:
        import uvicorn

        config = uvicorn.Config(
            _health_app, host="0.0.0.0", port=int(port), log_level="warning"
        )
        server = uvicorn.Server(config)
        task: asyncio.Task[None] = asyncio.create_task(server.serve())
        logger.info("worker.health_server_started", port=port)
        return task
    except Exception as exc:
        logger.warning("worker.health_server_failed", error=str(exc))
        return None


async def _build_services(
    event_bus: EventBus,
    settings: Settings,
) -> list[Any]:
    """Instantiate all services and return them as a list."""

    from core.models.base import get_async_session_factory

    session_factory = get_async_session_factory()

    # -- Data Ingestion --
    from services.data_ingestion.service import DataIngestionService

    data_ingestion = DataIngestionService(event_bus=event_bus, settings=settings)

    # -- Auxiliary market-regime ingestion + shared live snapshot --
    # ONE LiveAuxProvider is shared: the AuxMarketService refreshes it from
    # keyless public sources, and the FeatureEngineeringService reads it (via
    # its FeatureStore) so every live feature vector carries current
    # market-regime values under the same keys training replay uses.
    from services.data_ingestion.aux_market import AuxMarketService
    from services.feature_engineering.aux_features import LiveAuxProvider

    live_aux_provider = LiveAuxProvider()
    aux_market = AuxMarketService(
        session_factory=session_factory,
        live_provider=live_aux_provider,
        settings=settings,
    )

    # -- Feature Engineering --
    from services.feature_engineering.service import FeatureEngineeringService

    feature_engineering = FeatureEngineeringService(
        event_bus=event_bus, settings=settings, aux_provider=live_aux_provider
    )

    # -- Prediction --
    # The ModelServer bridges the registry to serving; without it the service
    # has no models and every prediction is hardcoded flat (the registry alone
    # is not enough -- this wiring is what makes promoted models actually serve).
    from services.prediction.registry import ModelRegistry, restore_and_reconcile
    from services.prediction.service import PredictionService
    from services.prediction.serving import ModelServer

    model_registry = ModelRegistry(artifact_base=Path(settings.model_artifact_path))
    # Railway has no volumes: cloud-promoted artifacts outlive deploys only via
    # the DB blob mirror. Restore anything newer than the shipped registry and
    # reconcile model_metadata to what actually serves. Never raises.
    await restore_and_reconcile(model_registry, session_factory)
    model_server = ModelServer(registry=model_registry)
    prediction = PredictionService(
        event_bus=event_bus, settings=settings, model_server=model_server
    )

    # -- Portfolio Optimizer --
    from services.portfolio.service import PortfolioOptimizerService

    portfolio = PortfolioOptimizerService(event_bus=event_bus, settings=settings)

    # -- Risk Manager --
    from services.risk.service import RiskManagerService

    risk_manager = RiskManagerService(event_bus=event_bus, settings=settings)

    # -- Execution Engine --
    from services.execution.service import ExecutionEngineService

    brokers = _build_brokers(settings)

    execution = ExecutionEngineService(
        event_bus=event_bus, settings=settings, brokers=brokers,
    )

    # -- Liquidation Manager --
    from services.liquidation.service import LiquidationManagerService

    liquidation = LiquidationManagerService(
        event_bus=event_bus, settings=settings, execution_service_ref=execution,
    )

    # -- Walk-forward research watch (registered funding-regime hypothesis) --
    if settings.funding_watch_enabled:
        from services.research.funding_watch import FundingWatchService

        services_extra_funding_watch = FundingWatchService(
            session_factory=session_factory
        )
    else:
        services_extra_funding_watch = None

    # -- Continuous Learning --
    from services.continuous_learning.drift_detector import DataDriftDetector
    from services.continuous_learning.evaluator import ModelEvaluator
    from services.continuous_learning.feedback_loop import TradingFeedbackLoop
    from services.continuous_learning.retrainer import AutoRetrainer
    from services.continuous_learning.service import ContinuousLearningService
    from services.prediction.training.trainer import ModelTrainer

    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    drift_detector = DataDriftDetector()
    feedback_loop = TradingFeedbackLoop()
    # Reuse the serving registry: two instances would race on registry.json.
    retrainer = AutoRetrainer(
        trainer=trainer,
        registry=model_registry,
        settings=settings,
        session_factory=session_factory,
    )

    continuous_learning = ContinuousLearningService(
        event_bus=event_bus,
        settings=settings,
        evaluator=evaluator,
        retrainer=retrainer,
        drift_detector=drift_detector,
        feedback_loop=feedback_loop,
        session_factory=session_factory,
    )

    # -- Monitoring --
    from services.monitoring.alerting import AlertManager
    from services.monitoring.audit import AuditLogger
    from services.monitoring.health import HealthChecker
    from services.monitoring.metrics import MetricsCollector
    from services.monitoring.service import MonitoringService

    metrics = MetricsCollector()
    health_checker = HealthChecker(
        # async_database_url, not the raw URL: Railway supplies postgresql://,
        # whose default SQLAlchemy dialect is psycopg2 (not installed) -- the
        # async probe needs the +asyncpg scheme or it crashes and fires false
        # "database Unhealthy" criticals every cycle.
        database_url=settings.async_database_url,
        redis_url=settings.redis_url,
    )
    if not settings.alert_webhook_url:
        # An unattended soak with no alert delivery is a silent failure mode:
        # circuit-breaker trips and drift warnings would go only to logs nobody
        # is watching. Make running without a webhook a visible choice.
        logger.warning(
            "worker.alerts_log_only",
            hint="set ALERT_WEBHOOK_URL (Slack incoming webhook) for operator alerts",
        )
    alert_manager = AlertManager(webhook_url=settings.alert_webhook_url or None)
    audit_logger = AuditLogger()

    monitoring = MonitoringService(
        event_bus=event_bus,
        settings=settings,
        metrics=metrics,
        health_checker=health_checker,
        alert_manager=alert_manager,
        audit_logger=audit_logger,
    )

    # -- Persistence (sync DB rows from event streams for reporting/metrics) --
    from services.persistence.order_sync import OrderPersistenceService
    from services.persistence.prediction_sync import PredictionPersistenceService

    order_persistence = OrderPersistenceService(
        event_bus=event_bus,
        session_factory=session_factory,
        trading_mode=settings.trading_mode,
    )
    prediction_persistence = PredictionPersistenceService(
        event_bus=event_bus, session_factory=session_factory
    )

    services = [
        data_ingestion,
        aux_market,
        feature_engineering,
        prediction,
        portfolio,
        risk_manager,
        execution,
        liquidation,
        continuous_learning,
        monitoring,
        order_persistence,
        prediction_persistence,
    ]
    if settings.funding_watch_enabled and services_extra_funding_watch is not None:
        services.append(services_extra_funding_watch)
    return services


def _account_broker(brokers: dict[str, Any]) -> Any:
    """Pick the broker that owns account state, deterministically.

    Dict insertion order decides nothing here: in paper/backtest the paper
    broker is the account, and in live the venue adapter is. Falling through
    to "first registered" would silently poll the wrong account the moment a
    second broker is wired.
    """
    for name in ("alpaca", "ccxt"):
        if name in brokers and getattr(brokers[name], "supports_live", False):
            return brokers[name]
    return brokers.get("paper") or next(iter(brokers.values()))


async def _restore_broker_state(services: list[Any], settings: Settings) -> None:
    """Restore the simulated broker's cash/book from the last checkpoint.

    Best effort: a restore failure must not stop the worker from booting, but
    it is loud, because silently continuing means trading from a rebased
    account.
    """
    from core.models.base import get_async_session_factory
    from services.execution.service import ExecutionEngineService
    from services.persistence.broker_state import BrokerStateStore

    execution = next(
        (s for s in services if isinstance(s, ExecutionEngineService)), None
    )
    if execution is None or not execution.brokers:
        return
    broker = _account_broker(execution.brokers)
    if not hasattr(broker, "restore_state"):
        return  # live brokers hold account state at the venue
    store = BrokerStateStore(
        session_factory=get_async_session_factory(),
        trading_mode=settings.trading_mode,
    )
    try:
        outcome = await store.restore(broker)
        logger.info("worker.broker_state_restore", outcome=outcome)
    except Exception:
        logger.exception("worker.broker_state_restore_failed")


async def _run() -> None:
    """Main async entry point: build services, start them, wait for shutdown."""
    settings = get_settings()
    event_bus = EventBus(redis_url=settings.redis_url)

    # Log only host:port -- the full URL embeds the Redis password, and log
    # stores are far more widely readable than the secret store.
    redis_host = urlsplit(settings.redis_url).netloc.rsplit("@", 1)[-1]
    logger.info("worker.initializing", redis_host=redis_host)

    services = await _build_services(event_bus, settings)

    # Rehydrate simulated account state BEFORE anything can trade: the paper
    # broker holds cash/positions in memory, so an unrestored restart rebases
    # equity to initial capital and orphans the open book.
    await _restore_broker_state(services, settings)

    # Graceful shutdown handling
    shutdown_event = asyncio.Event()

    def _signal_handler(sig: int, _frame: Any) -> None:
        logger.info("worker.signal_received", signal=signal.Signals(sig).name)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Start all services
    logger.info("worker.starting_services", count=len(services))
    start_tasks = [svc.start() for svc in services]
    await asyncio.gather(*start_tasks)
    logger.info("worker.all_services_started")

    # Periodically sync the live broker account into the risk engine so the
    # drawdown monitor / circuit breaker act on real capital.
    from decimal import Decimal

    from services.execution.service import ExecutionEngineService
    from services.risk.service import RiskManagerService

    risk = next((s for s in services if isinstance(s, RiskManagerService)), None)
    execution = next((s for s in services if isinstance(s, ExecutionEngineService)), None)
    account_sync_task: asyncio.Task[None] | None = None
    if risk is not None and execution is not None and execution.brokers:
        broker = _account_broker(execution.brokers)

        async def _equity_provider() -> Decimal:
            account = await broker.get_account()
            return Decimal(str(account.get("equity", "0")))

        account_sync_task = asyncio.create_task(risk.run_account_sync(_equity_provider))
        logger.info("worker.account_sync_started")

    # Periodic equity snapshots -> portfolio_snapshots (the dashboard's equity
    # curve and the soak's paper trail; nothing else writes that table).
    snapshot_task: asyncio.Task[None] | None = None
    if execution is not None and execution.brokers:
        from core.models.base import get_async_session_factory
        from services.persistence.broker_state import BrokerStateStore
        from services.persistence.snapshot_writer import PortfolioSnapshotWriter

        snapshot_broker = _account_broker(execution.brokers)
        state_store = BrokerStateStore(
            session_factory=get_async_session_factory(),
            trading_mode=settings.trading_mode,
        )
        snapshot_writer = PortfolioSnapshotWriter(
            session_factory=get_async_session_factory(),
            trading_mode=settings.trading_mode,
            state_store=state_store,
        )
        snapshot_task = asyncio.create_task(
            snapshot_writer.run(
                account_provider=snapshot_broker.get_account,
                positions_provider=snapshot_broker.get_positions,
                interval_seconds=300.0,
            )
        )
        logger.info("worker.snapshot_writer_started")

    # Liveness endpoint for platform healthchecks (no-op when PORT is unset).
    health_task = _start_health_server()

    # Wait until shutdown signal
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("worker.shutting_down")
    if health_task is not None:
        health_task.cancel()
    if snapshot_task is not None:
        snapshot_task.cancel()
    if account_sync_task is not None:
        account_sync_task.cancel()
    stop_tasks = []
    for svc in reversed(services):
        if hasattr(svc, "stop"):
            stop_tasks.append(svc.stop())
    if stop_tasks:
        await asyncio.gather(*stop_tasks, return_exceptions=True)

    await event_bus.close()
    logger.info("worker.shutdown_complete")


def main() -> None:
    """Synchronous entry point."""
    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    # Configure BOTH logging stacks: structlog carries most worker services,
    # but prediction/portfolio/execution use stdlib logging, whose default
    # WARNING threshold was silently hiding their INFO lines (e.g. the
    # hot-reload confirmation) from production logs.
    logging.basicConfig(level=level, format="%(levelname)s %(name)s %(message)s")
    # httpx/httpcore log every request URL at INFO -- including the Slack alert
    # webhook, whose URL embeds a secret token. Cap them at WARNING so the
    # token is never written to the log store (errors still surface).
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger.info("worker.main", python=sys.version)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("worker.interrupted")


if __name__ == "__main__":
    main()
