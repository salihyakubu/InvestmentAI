"""Market-regime (aux) feature pipeline: providers, feature-store integration,
train/serve replay parity, and the ingestion service's failure isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from config.settings import Settings
from services.data_ingestion.aux_market import (
    AuxMarketService,
    _to_binance_futures_symbol,
)
from services.feature_engineering.aux_features import (
    AUX_FEATURE_NAMES,
    GLOBAL_SYMBOL,
    METRIC_FEAR_GREED,
    METRIC_FUNDING_RATE,
    METRIC_VIX_CLOSE,
    HistoricalAuxProvider,
    LiveAuxProvider,
    zero_aux,
)
from services.feature_engineering.feature_store import FeatureStore
from services.prediction.training.dataset_builder import (
    WINDOW,
    bars_matrix,
    replay_features,
)

_BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _ohlcv_df(n: int = 30, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    high = close * 1.01
    low = close * 0.99
    open_ = np.concatenate(([100.0], close[:-1]))
    volume = rng.uniform(1_000.0, 5_000.0, size=n)
    return pl.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# Fixed key set
# ---------------------------------------------------------------------------


def test_aux_feature_names_are_fixed_and_sorted() -> None:
    assert AUX_FEATURE_NAMES == sorted(AUX_FEATURE_NAMES)
    assert AUX_FEATURE_NAMES == [
        "aux_fear_greed",
        "aux_funding_rate",
        "aux_spy_daily_return",
        "aux_vix_close",
    ]


def test_zero_aux_is_full_key_set_at_zero() -> None:
    z = zero_aux()
    assert set(z) == set(AUX_FEATURE_NAMES)
    assert all(v == 0.0 for v in z.values())


# ---------------------------------------------------------------------------
# compute_all_features always emits the aux key set
# ---------------------------------------------------------------------------


def test_compute_all_features_emits_aux_zero_without_provider() -> None:
    store = FeatureStore()  # no aux provider
    df = _ohlcv_df()
    # With and without a timestamp, the aux keys are present and zero.
    for ts in (None, _BASE):
        feats = store.compute_all_features("AAPL", df, timestamp=ts)
        for name in AUX_FEATURE_NAMES:
            assert name in feats
            assert feats[name] == 0.0


def test_compute_all_features_uses_provider_values_with_timestamp() -> None:
    provider = LiveAuxProvider()
    provider.update_fear_greed(72.0)
    provider.update_vix_close(18.5)
    provider.update_spy_daily_return(0.004)
    provider.update_funding("BTC/USDT", 0.0002)

    store = FeatureStore(aux_provider=provider)
    df = _ohlcv_df()

    feats = store.compute_all_features("BTC/USDT", df, timestamp=_BASE)
    assert feats["aux_fear_greed"] == 72.0
    assert feats["aux_vix_close"] == 18.5
    assert feats["aux_spy_daily_return"] == 0.004
    assert feats["aux_funding_rate"] == 0.0002

    # A stock has no per-symbol funding -> 0.0, globals still apply.
    stock = store.compute_all_features("AAPL", df, timestamp=_BASE)
    assert stock["aux_funding_rate"] == 0.0
    assert stock["aux_vix_close"] == 18.5


def test_compute_all_features_provider_ignored_without_timestamp() -> None:
    provider = LiveAuxProvider()
    provider.update_vix_close(18.5)
    store = FeatureStore(aux_provider=provider)
    # No timestamp -> aux keys fall back to 0.0 (matches a live path with no
    # provider), so the key set is stable regardless.
    feats = store.compute_all_features("BTC/USDT", _ohlcv_df(), timestamp=None)
    assert feats["aux_vix_close"] == 0.0
    for name in AUX_FEATURE_NAMES:
        assert name in feats


# ---------------------------------------------------------------------------
# LiveAuxProvider
# ---------------------------------------------------------------------------


def test_live_provider_snapshot_update_and_asof() -> None:
    provider = LiveAuxProvider()
    # Before any update everything is zero.
    assert provider.features_asof("BTC/USDT", _BASE) == zero_aux()

    provider.update_fear_greed(40.0)
    provider.update_vix_close(22.0)
    provider.update_spy_daily_return(-0.01)
    provider.update_funding("ETH/USDT", 0.0005)

    # ts is ignored (live == latest); a much earlier ts yields the same snapshot.
    got = provider.features_asof("ETH/USDT", _BASE - timedelta(days=999))
    assert got == {
        "aux_fear_greed": 40.0,
        "aux_funding_rate": 0.0005,
        "aux_spy_daily_return": -0.01,
        "aux_vix_close": 22.0,
    }
    # Unknown-symbol funding -> 0.0.
    assert provider.features_asof("BTC/USDT", _BASE)["aux_funding_rate"] == 0.0


# ---------------------------------------------------------------------------
# HistoricalAuxProvider as-of lookup
# ---------------------------------------------------------------------------


def test_historical_provider_asof_latest_at_or_before() -> None:
    rows = [
        (_BASE + timedelta(days=10), METRIC_VIX_CLOSE, GLOBAL_SYMBOL, 15.0),
        (_BASE + timedelta(days=20), METRIC_VIX_CLOSE, GLOBAL_SYMBOL, 25.0),
        (_BASE + timedelta(days=5), METRIC_FEAR_GREED, GLOBAL_SYMBOL, 60.0),
        (_BASE + timedelta(days=12), METRIC_FUNDING_RATE, "BTC/USDT", 0.0001),
        (_BASE + timedelta(days=12), METRIC_FUNDING_RATE, "ETH/USDT", 0.0009),
    ]
    p = HistoricalAuxProvider(rows)

    # Before any VIX row -> 0.0.
    assert p.features_asof("BTC/USDT", _BASE)["aux_vix_close"] == 0.0
    # Exactly at a row's time counts (at-or-before).
    assert p.features_asof("BTC/USDT", _BASE + timedelta(days=10))["aux_vix_close"] == 15.0
    # Between rows -> the earlier one.
    assert p.features_asof("BTC/USDT", _BASE + timedelta(days=15))["aux_vix_close"] == 15.0
    # At-or-after the later row -> the later one.
    assert p.features_asof("BTC/USDT", _BASE + timedelta(days=30))["aux_vix_close"] == 25.0

    # Per-symbol funding is isolated.
    at = _BASE + timedelta(days=20)
    assert p.features_asof("BTC/USDT", at)["aux_funding_rate"] == 0.0001
    assert p.features_asof("ETH/USDT", at)["aux_funding_rate"] == 0.0009
    assert p.features_asof("SOL/USDT", at)["aux_funding_rate"] == 0.0  # missing

    # A metric with no rows at all -> 0.0.
    assert p.features_asof("BTC/USDT", at)["aux_spy_daily_return"] == 0.0
    # F&G present from day 5.
    assert p.features_asof("BTC/USDT", at)["aux_fear_greed"] == 60.0


def test_historical_provider_returns_full_key_set() -> None:
    p = HistoricalAuxProvider([])  # completely empty
    got = p.features_asof("BTC/USDT", _BASE)
    assert set(got) == set(AUX_FEATURE_NAMES)
    assert all(v == 0.0 for v in got.values())


# ---------------------------------------------------------------------------
# dataset_builder replay parity
# ---------------------------------------------------------------------------


def _timed_bars(n: int, *, with_time: bool, seed: int = 1) -> list[Any]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.005, size=n))
    bars: list[Any] = []
    for i in range(n):
        fields = {
            "open": float(close[i - 1]) if i else 100.0,
            "high": float(close[i]) * 1.01,
            "low": float(close[i]) * 0.99,
            "close": float(close[i]),
            "volume": 1000.0 + i,
        }
        if with_time:
            fields["time"] = _BASE + timedelta(days=i)
        bars.append(SimpleNamespace(**fields))
    return bars


def test_replay_injects_asof_aux_at_window_end_time() -> None:
    n = WINDOW + 5  # -> windows ending at index 199 and 204 with stride 5
    bars = _timed_bars(n, with_time=True)
    cols = bars_matrix(bars)
    assert cols["time"] is not None  # bars carry .time

    # VIX steps up between the two windows' end times (days 199 and 204).
    rows = [
        (_BASE + timedelta(days=50), METRIC_VIX_CLOSE, GLOBAL_SYMBOL, 15.0),
        (_BASE + timedelta(days=201), METRIC_VIX_CLOSE, GLOBAL_SYMBOL, 25.0),
        (_BASE + timedelta(days=10), METRIC_FUNDING_RATE, "BTC/USDT", 0.0003),
    ]
    store = FeatureStore(aux_provider=HistoricalAuxProvider(rows))
    X, _closes, names = replay_features("BTC/USDT", cols, 5, store)

    assert set(AUX_FEATURE_NAMES).issubset(names)
    assert X.shape[0] == 2
    i_vix = names.index("aux_vix_close")
    i_fund = names.index("aux_funding_rate")
    i_fng = names.index("aux_fear_greed")
    i_spy = names.index("aux_spy_daily_return")

    # Window 0 ends at day 199 -> as-of VIX is the day-50 value (15.0).
    # Window 1 ends at day 204 -> as-of VIX is the day-201 value (25.0).
    assert X[0, i_vix] == 15.0
    assert X[1, i_vix] == 25.0
    # Funding is constant across both windows for BTC/USDT.
    assert X[0, i_fund] == 0.0003
    assert X[1, i_fund] == 0.0003
    # No F&G or SPY rows -> 0.0 everywhere.
    assert np.all(X[:, i_fng] == 0.0)
    assert np.all(X[:, i_spy] == 0.0)


def test_replay_without_provider_emits_zero_aux() -> None:
    bars = _timed_bars(WINDOW + 5, with_time=True)
    cols = bars_matrix(bars)
    X, _closes, names = replay_features("BTC/USDT", cols, 5, FeatureStore())
    for name in AUX_FEATURE_NAMES:
        assert name in names
        assert np.all(X[:, names.index(name)] == 0.0)


def test_replay_no_time_falls_back_to_zero_even_with_provider() -> None:
    # Bars without a .time attribute -> cols["time"] is None -> timestamp=None,
    # so aux stays 0.0 even though a provider is wired (parity with a live path
    # that supplies no timestamp).
    bars = _timed_bars(WINDOW + 5, with_time=False)
    cols = bars_matrix(bars)
    assert cols["time"] is None
    provider = HistoricalAuxProvider(
        [(_BASE, METRIC_VIX_CLOSE, GLOBAL_SYMBOL, 99.0)]
    )
    X, _closes, names = replay_features("BTC/USDT", cols, 5, FeatureStore(aux_provider=provider))
    assert np.all(X[:, names.index("aux_vix_close")] == 0.0)


# ---------------------------------------------------------------------------
# AuxMarketService: source-failure isolation (no real network / DB)
# ---------------------------------------------------------------------------


class _NoopSession:
    async def __aenter__(self) -> _NoopSession:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def merge(self, obj: Any) -> Any:
        return obj

    async def commit(self) -> None:
        return None


def _noop_factory() -> _NoopSession:
    return _NoopSession()


class _FailingHTTPClient:
    async def __aenter__(self) -> _FailingHTTPClient:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def get(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("simulated network failure")


class _FakeYahoo:
    """Returns good daily bars for ^VIX and SPY, nothing else."""

    async def fetch_historical_bars(
        self, symbol: str, timeframe: str, start: Any, end: Any
    ) -> list[Any]:
        if symbol == "^VIX":
            return [SimpleNamespace(close=20.0, time=_BASE)]
        if symbol == "SPY":
            return [
                SimpleNamespace(close=100.0, time=_BASE),
                SimpleNamespace(close=101.0, time=_BASE + timedelta(days=1)),
            ]
        return []


def _mk_settings() -> Settings:
    return Settings(active_symbols_crypto=["BTC/USDT", "ETH/USDT"])


def test_to_binance_futures_symbol() -> None:
    assert _to_binance_futures_symbol("BTC/USDT") == "BTCUSDT"
    assert _to_binance_futures_symbol("ETH-USDT") == "ETHUSDT"


@pytest.mark.asyncio
async def test_poll_once_isolates_source_failures() -> None:
    live = LiveAuxProvider()
    svc = AuxMarketService(
        session_factory=_noop_factory,  # type: ignore[arg-type]
        live_provider=live,
        settings=_mk_settings(),
        http_client_factory=_FailingHTTPClient,  # type: ignore[arg-type]
        yahoo_provider=_FakeYahoo(),  # type: ignore[arg-type]
    )

    # Funding + F&G raise (http fails); VIX + SPY succeed. Must not raise.
    await svc.poll_once()

    snap = live.features_asof("BTC/USDT", _BASE)
    assert snap["aux_vix_close"] == 20.0
    assert snap["aux_spy_daily_return"] == pytest.approx(0.01)
    # The failed sources left their values at the zero default.
    assert snap["aux_fear_greed"] == 0.0
    assert snap["aux_funding_rate"] == 0.0


class _FakeSuccessHTTPClient:
    """Serves both the Binance funding and the F&G endpoints."""

    async def __aenter__(self) -> _FakeSuccessHTTPClient:
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        if "premiumIndex" in url:
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"lastFundingRate": "0.00012"},
            )
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"value": "55"}]},
        )


@pytest.mark.asyncio
async def test_poll_once_updates_snapshot_on_success() -> None:
    live = LiveAuxProvider()
    svc = AuxMarketService(
        session_factory=_noop_factory,  # type: ignore[arg-type]
        live_provider=live,
        settings=_mk_settings(),
        http_client_factory=_FakeSuccessHTTPClient,  # type: ignore[arg-type]
        yahoo_provider=_FakeYahoo(),  # type: ignore[arg-type]
    )
    await svc.poll_once()

    snap = live.features_asof("BTC/USDT", _BASE)
    assert snap["aux_funding_rate"] == pytest.approx(0.00012)
    assert snap["aux_fear_greed"] == 55.0
    assert snap["aux_vix_close"] == 20.0
    assert snap["aux_spy_daily_return"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Migration: imports cleanly and leaves a single head
# ---------------------------------------------------------------------------


def test_alembic_single_head_includes_aux_migration() -> None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    sd = ScriptDirectory.from_config(Config("alembic.ini"))
    heads = sd.get_heads()
    # Single linear head, with the aux migration somewhere in its ancestry
    # (later migrations may extend the chain past it).
    assert len(heads) == 1
    chain = {rev.revision for rev in sd.walk_revisions()}
    assert "0005_aux_market" in chain
