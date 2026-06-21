"""Main worker process -- initialises and runs all platform services concurrently."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

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


async def _build_services(
    event_bus: EventBus,
    settings: Settings,
) -> list[Any]:
    """Instantiate all services and return them as a list."""

    # -- Data Ingestion --
    from services.data_ingestion.service import DataIngestionService

    data_ingestion = DataIngestionService(event_bus=event_bus, settings=settings)

    # -- Feature Engineering --
    from services.feature_engineering.service import FeatureEngineeringService

    feature_engineering = FeatureEngineeringService(event_bus=event_bus, settings=settings)

    # -- Prediction --
    from services.prediction.service import PredictionService

    prediction = PredictionService(event_bus=event_bus, settings=settings)

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

    # -- Continuous Learning --
    from services.continuous_learning.drift_detector import DataDriftDetector
    from services.continuous_learning.evaluator import ModelEvaluator
    from services.continuous_learning.feedback_loop import TradingFeedbackLoop
    from services.continuous_learning.retrainer import AutoRetrainer
    from services.continuous_learning.service import ContinuousLearningService
    from services.prediction.registry import ModelRegistry
    from services.prediction.training.trainer import ModelTrainer

    trainer = ModelTrainer()
    registry = ModelRegistry(artifact_base=Path(settings.model_artifact_path))
    evaluator = ModelEvaluator()
    drift_detector = DataDriftDetector()
    feedback_loop = TradingFeedbackLoop()
    retrainer = AutoRetrainer(trainer=trainer, registry=registry, settings=settings)

    continuous_learning = ContinuousLearningService(
        event_bus=event_bus,
        settings=settings,
        evaluator=evaluator,
        retrainer=retrainer,
        drift_detector=drift_detector,
        feedback_loop=feedback_loop,
    )

    # -- Monitoring --
    from services.monitoring.alerting import AlertManager
    from services.monitoring.audit import AuditLogger
    from services.monitoring.health import HealthChecker
    from services.monitoring.metrics import MetricsCollector
    from services.monitoring.service import MonitoringService

    metrics = MetricsCollector()
    health_checker = HealthChecker(
        database_url=settings.database_url,
        redis_url=settings.redis_url,
    )
    alert_manager = AlertManager(webhook_url=None)  # Set via env in production
    audit_logger = AuditLogger()

    monitoring = MonitoringService(
        event_bus=event_bus,
        settings=settings,
        metrics=metrics,
        health_checker=health_checker,
        alert_manager=alert_manager,
        audit_logger=audit_logger,
    )

    # -- Order persistence (sync DB order/fill rows from execution events) --
    from core.models.base import get_async_session_factory
    from services.persistence.order_sync import OrderPersistenceService

    order_persistence = OrderPersistenceService(
        event_bus=event_bus, session_factory=get_async_session_factory()
    )

    return [
        data_ingestion,
        feature_engineering,
        prediction,
        portfolio,
        risk_manager,
        execution,
        liquidation,
        continuous_learning,
        monitoring,
        order_persistence,
    ]


async def _run() -> None:
    """Main async entry point: build services, start them, wait for shutdown."""
    settings = get_settings()
    event_bus = EventBus(redis_url=settings.redis_url)

    logger.info("worker.initializing", redis_url=settings.redis_url)

    services = await _build_services(event_bus, settings)

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
    account_sync_task: asyncio.Task | None = None
    if risk is not None and execution is not None and execution.brokers:
        broker = next(iter(execution.brokers.values()))

        async def _equity_provider() -> Decimal:
            account = await broker.get_account()
            return Decimal(str(account.get("equity", "0")))

        account_sync_task = asyncio.create_task(risk.run_account_sync(_equity_provider))
        logger.info("worker.account_sync_started")

    # Wait until shutdown signal
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("worker.shutting_down")
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
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
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
