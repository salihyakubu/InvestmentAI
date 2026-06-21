"""Worker broker-wiring safety: live venues are gated on trading_mode.

Regression guard for the bug where live brokers were attached based purely on
credential presence. That let "paper" mode reach a live venue -- dangerous for
crypto in particular, since CCXTBroker talks to the real Binance spot venue with
no testnet/sandbox. These tests pin the invariant: only live mode wires live
brokers; paper/backtest keep nothing but the PaperBroker, even when real-looking
credentials are present in the environment.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.settings import Settings
from services.worker import _build_brokers


def _settings(trading_mode: str) -> Settings:
    """Settings carrying real-looking (non-placeholder) broker credentials."""
    return Settings(
        trading_mode=trading_mode,
        alpaca_api_key="AKREALLOOKINGKEY1234",
        alpaca_secret_key="real-looking-alpaca-secret",
        binance_api_key="real-looking-binance-key",
        binance_secret_key="real-looking-binance-secret",
        initial_capital=Decimal("100.00"),
        # Strong, non-default JWT secret so live mode passes the security check.
        jwt_secret="a" * 40,
    )


@pytest.mark.parametrize("trading_mode", ["paper", "backtest"])
def test_non_live_mode_wires_only_paper_broker(trading_mode: str) -> None:
    """With real creds present but mode != 'live', only PaperBroker is wired."""
    brokers = _build_brokers(_settings(trading_mode))

    assert set(brokers) == {"paper"}, (
        f"trading_mode={trading_mode!r} must not reach any live venue; "
        f"got brokers={sorted(brokers)}"
    )


def test_live_mode_wires_live_brokers_when_creds_present() -> None:
    """Live mode does attach the live venues when real creds are present."""
    brokers = _build_brokers(_settings("live"))

    assert set(brokers) == {"paper", "alpaca", "ccxt"}, (
        f"live mode should wire live brokers; got brokers={sorted(brokers)}"
    )
