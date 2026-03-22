"""End-to-end test verifying the full trading pipeline.

This test exercises the entire flow:
    data ingestion -> feature engineering -> prediction ->
    portfolio optimization -> risk check -> execution -> fill

All components use in-memory / paper implementations so no external
services (Redis, PostgreSQL, brokers) are required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from config.settings import Settings
from core.enums import OrderSide, OrderStatus, OrderType
from core.events.base import Event, InProcessEventBus
from core.events.market_events import BarCloseEvent
from core.events.order_events import OrderFilledEvent
from core.events.signal_events import FeaturesReadyEvent, PredictionReadyEvent
from services.execution.brokers.base import BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.order_manager import OrderManager
from services.feature_engineering.feature_store import FeatureStore
from services.feature_engineering.service import FeatureEngineeringService
from services.portfolio.optimizers.mean_variance import MeanVarianceOptimizer
from services.risk.service import RiskManagerService


@pytest.mark.asyncio
async def test_full_cycle(
    event_bus: InProcessEventBus,
    mock_settings: Settings,
    sample_ohlcv_bars: list[dict],
) -> None:
    """Run the full trading pipeline end-to-end using in-memory
    components.

    Steps
    -----
    1. Feed OHLCV bars into the feature engineering service.
    2. Verify features are computed and a FeaturesReadyEvent is emitted.
    3. Simulate a prediction from the ML layer.
    4. Run portfolio optimisation on the prediction.
    5. Run risk checks on the proposed allocation.
    6. Submit the order to the paper broker and verify the fill.
    """
    # ------------------------------------------------------------------
    # 1. DATA INGESTION -> FEATURE ENGINEERING
    # ------------------------------------------------------------------
    feature_store = FeatureStore()
    fe_service = FeatureEngineeringService(
        event_bus=event_bus,
        settings=mock_settings,
        feature_store=feature_store,
    )
    await fe_service.start()

    # Track FeaturesReadyEvent emissions.
    features_received: list[Event] = []

    async def on_features(event: Event) -> None:
        features_received.append(event)

    await event_bus.subscribe(
        stream="features.ready",
        group="test",
        consumer="test-1",
        handler=on_features,
    )

    # Feed all bars to build up a buffer, then emit the last one as
    # a BarCloseEvent so features can be computed.
    for bar in sample_ohlcv_bars[:-1]:
        fe_service._bar_buffers.setdefault("AAPL:1d", []).append(bar)

    last_bar = sample_ohlcv_bars[-1]
    bar_event = BarCloseEvent(
        symbol="AAPL",
        timeframe="1d",
        open=last_bar["open"],
        high=last_bar["high"],
        low=last_bar["low"],
        close=last_bar["close"],
        volume=last_bar["volume"],
        vwap=last_bar["vwap"],
        bar_time=datetime.now(timezone.utc),
        source_service="data_ingestion",
    )
    await fe_service.handle_bar_close(bar_event)

    # Verify features were computed and event was published.
    assert len(features_received) == 1
    features_event = FeaturesReadyEvent.model_validate(
        features_received[0].model_dump()
    )
    assert features_event.symbol == "AAPL"
    assert len(features_event.feature_vector) > 10  # Many features expected

    # ------------------------------------------------------------------
    # 2. PREDICTION (simulated)
    # ------------------------------------------------------------------
    prediction = PredictionReadyEvent(
        symbol="AAPL",
        direction="long",
        confidence=0.78,
        expected_return=0.025,
        model_id="test_ensemble_v1",
        source_service="prediction",
    )
    await event_bus.publish("predictions.ready", prediction)

    # ------------------------------------------------------------------
    # 3. PORTFOLIO OPTIMISATION
    # ------------------------------------------------------------------
    # Use a simple 4-asset universe with AAPL having the best expected
    # return (reflecting the prediction).
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    expected_returns = np.array([0.025, 0.01, 0.012, 0.008])
    cov_matrix = np.array(
        [
            [0.04, 0.012, 0.010, 0.008],
            [0.012, 0.0324, 0.011, 0.009],
            [0.010, 0.011, 0.0625, 0.007],
            [0.008, 0.009, 0.007, 0.0484],
        ]
    )

    optimizer = MeanVarianceOptimizer(max_position_pct=0.50)
    opt_result = optimizer.optimize(expected_returns, cov_matrix, symbols)

    # Weights should sum to 1.
    total_weight = sum(opt_result.weights.values())
    assert abs(total_weight - 1.0) < 1e-6

    # AAPL should have a meaningful allocation given the higher return.
    aapl_weight = opt_result.weights["AAPL"]
    assert aapl_weight > 0.0

    # ------------------------------------------------------------------
    # 4. RISK CHECK
    # ------------------------------------------------------------------
    risk_svc = RiskManagerService(event_bus=event_bus, settings=mock_settings)
    risk_svc.update_equity(mock_settings.initial_capital)
    risk_svc.drawdown_monitor.update(float(mock_settings.initial_capital))

    decision = risk_svc.pre_trade_check(
        {"symbol": "AAPL", "target_weight": aapl_weight}
    )

    # The allocation from the optimizer should be within risk limits
    # if max_position_pct is 0.10 and the optimizer caps at 0.50.
    # The risk manager will reject if aapl_weight > 0.10.
    # We test both paths:
    if aapl_weight <= mock_settings.max_position_pct:
        assert decision.approved
    else:
        assert not decision.approved
        # Retry with a clipped weight.
        decision = risk_svc.pre_trade_check(
            {"symbol": "AAPL", "target_weight": min(aapl_weight, 0.08)}
        )
        assert decision.approved

    # ------------------------------------------------------------------
    # 5. EXECUTION via PaperBroker
    # ------------------------------------------------------------------
    broker = PaperBroker(initial_cash=mock_settings.initial_capital)
    broker.update_price("AAPL", Decimal(str(last_bar["close"])))

    order_qty = Decimal("5")
    broker_order = BrokerOrder(
        external_id="e2e-order-1",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=order_qty,
    )
    ext_id = await broker.submit_order(broker_order)

    # ------------------------------------------------------------------
    # 6. VERIFY FILL
    # ------------------------------------------------------------------
    status = await broker.get_order_status(ext_id)
    assert status["status"] == "filled"

    fills = broker.fills
    assert len(fills) >= 1
    assert fills[-1].quantity == order_qty
    assert fills[-1].price > Decimal("0")

    # Account should reflect the trade.
    account = await broker.get_account()
    assert Decimal(account["cash"]) < mock_settings.initial_capital

    # ------------------------------------------------------------------
    # 7. POST-TRADE RISK UPDATE
    # ------------------------------------------------------------------
    fill_event = OrderFilledEvent(
        order_id="e2e-order-1",
        fill_price=float(fills[-1].price),
        fill_quantity=float(fills[-1].quantity),
        commission=0.0,
        source_service="execution",
        payload={"symbol": "AAPL"},
    )
    risk_svc.post_trade_update(fill_event)

    # The risk report should now show a position.
    report = risk_svc.check_portfolio_risk()
    assert report.circuit_breaker_state == "closed"

    # Verify the event bus history contains all the events we published.
    event_types = [e.event_type for _, e in event_bus.history]
    assert "FeaturesReadyEvent" in event_types
    assert "PredictionReadyEvent" in event_types
