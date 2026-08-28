"""The walk-forward watch: append-only, causally resolved, honestly counted.

The watch is the adjudicator for a registered hypothesis, so its own
integrity IS the experiment. Three properties matter above all: a stamp is
only resolved after its forward window fully closes (no peeking), resolved
rows are never recomputed or duplicated (no survivorship creep), and the
quarterly count cannot be nudged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.models.base import AsyncBase
from core.models.research import FactorWatchObservation
from services.research.funding_watch import (
    FACTOR_NAME,
    HORIZON_STAMPS,
    UNSEEN_FROM,
    FundingWatchService,
    compute_resolvable_ics,
    quarterly_rollup,
)

_MS_8H = 8 * 3_600_000


def _grid(n: int, start: datetime | None = None) -> np.ndarray:
    base = int((start or datetime(2026, 7, 2, tzinfo=UTC)).timestamp() * 1000)
    base = (base // _MS_8H) * _MS_8H
    return base + np.arange(n, dtype=np.int64) * _MS_8H


async def _factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all,
            tables=[FactorWatchObservation.__table__],
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


# ---------------------------------------------------------------------------
# Resolution core
# ---------------------------------------------------------------------------


def test_a_stamp_is_only_resolved_after_its_window_closes() -> None:
    """The last HORIZON stamps have open windows and must not be scored."""
    n, syms = 12, 30
    rng = np.random.default_rng(1)
    times = _grid(n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, syms)), axis=0)
    funding = rng.normal(0, 0.0005, (n, syms))
    resolved = {t for t, _, _ in compute_resolvable_ics(times, close, funding)}
    for open_stamp in times[-HORIZON_STAMPS:]:
        assert int(open_stamp) not in resolved


def test_the_factor_uses_only_past_funding() -> None:
    """Planted future-only relationship must produce ~zero IC: the factor at
    t may use funding up to t, never beyond."""
    n, syms = 400, 40
    rng = np.random.default_rng(2)
    times = _grid(n)
    funding = rng.normal(0, 0.0005, (n, syms))
    # Returns driven by funding THREE stamps ahead -- pure future.
    rel = np.zeros((n, syms))
    rel[:-3] = funding[3:] * 5.0
    close = 100 * np.cumprod(1 + rel + rng.normal(0, 0.004, (n, syms)), axis=0)
    ics = [ic for _, ic, _ in compute_resolvable_ics(times, close, funding)]
    assert abs(float(np.mean(ics))) < 0.05


def test_registered_sign_high_funding_scored_high() -> None:
    """If high funding really does predict outperformance, the recorded IC
    must be POSITIVE -- the registered orientation, not its mirror."""
    n, syms = 400, 40
    rng = np.random.default_rng(3)
    times = _grid(n)
    funding = rng.normal(0, 0.0005, (n, syms))
    rel = np.zeros((n, syms))
    # Next-stamp relative return follows CURRENT funding, positively.
    rel[1:] = funding[:-1] * 5.0
    close = 100 * np.cumprod(1 + rel + rng.normal(0, 0.002, (n, syms)), axis=0)
    ics = [ic for _, ic, _ in compute_resolvable_ics(times, close, funding)]
    assert float(np.mean(ics)) > 0.1


def test_thin_cross_sections_are_skipped_not_scored() -> None:
    n = 20
    times = _grid(n)
    close = np.full((n, 10), 100.0)  # only 10 symbols: below the minimum
    funding = np.random.default_rng(4).normal(0, 0.0005, (n, 10))
    assert compute_resolvable_ics(times, close, funding) == []


# ---------------------------------------------------------------------------
# Append-only persistence
# ---------------------------------------------------------------------------


class _StubbedWatch(FundingWatchService):
    """Watch with the network fetch replaced by a fixed dataset."""

    def __init__(self, session_factory, dataset):
        super().__init__(session_factory)
        self._dataset = dataset

    async def observe_once(self) -> int:  # type: ignore[override]
        grid, close, funding = self._dataset
        candidates = compute_resolvable_ics(grid, close, funding)
        if not candidates:
            return 0
        async with self._session_factory() as session:
            existing = await self._existing_stamps(session)
            inserted = 0
            now = datetime.now(UTC)
            for stamp_ms, ic, n_symbols in candidates:
                when = datetime.fromtimestamp(stamp_ms / 1000, UTC)
                if when < UNSEEN_FROM or when in existing:
                    continue
                session.add(
                    FactorWatchObservation(
                        time=when, factor=FACTOR_NAME,
                        horizon_stamps=HORIZON_STAMPS, ic=ic,
                        n_symbols=n_symbols, resolved_at=now,
                    )
                )
                inserted += 1
            if inserted:
                await session.commit()
        return inserted


def _dataset(n: int = 30, start: datetime | None = None):
    rng = np.random.default_rng(7)
    times = _grid(n, start)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 40)), axis=0)
    funding = rng.normal(0, 0.0005, (n, 40))
    return times, close, funding


@pytest.mark.asyncio
async def test_observing_twice_inserts_nothing_new() -> None:
    """Idempotence is the append-only guarantee: re-running a cycle over the
    same window must not duplicate or overwrite anything."""
    engine, factory = await _factory()
    watch = _StubbedWatch(factory, _dataset())
    first = await watch.observe_once()
    assert first > 0
    second = await watch.observe_once()
    assert second == 0
    async with factory() as session:
        rows = (await session.execute(select(FactorWatchObservation))).scalars().all()
    assert len(rows) == first
    await engine.dispose()


@pytest.mark.asyncio
async def test_resolved_ics_are_never_recomputed() -> None:
    """Even if later data would give a different IC for an old stamp (e.g. a
    contract delisted since), the stored row must stand."""
    engine, factory = await _factory()
    times, close, funding = _dataset()
    watch = _StubbedWatch(factory, (times, close, funding))
    await watch.observe_once()
    async with factory() as session:
        before = {
            r.time: r.ic
            for r in (await session.execute(select(FactorWatchObservation))).scalars()
        }

    # The same window, but half the universe has vanished (delistings).
    watch._dataset = (times, close[:, :20], funding[:, :20])
    await watch.observe_once()
    async with factory() as session:
        after = {
            r.time: r.ic
            for r in (await session.execute(select(FactorWatchObservation))).scalars()
        }
    assert after == before
    await engine.dispose()


@pytest.mark.asyncio
async def test_stamps_before_the_registration_boundary_are_refused() -> None:
    """Data the study saw can never count toward its own adjudication."""
    engine, factory = await _factory()
    seen_start = UNSEEN_FROM - timedelta(days=30)
    watch = _StubbedWatch(factory, _dataset(30, start=seen_start))
    inserted = await watch.observe_once()
    async with factory() as session:
        rows = (await session.execute(select(FactorWatchObservation))).scalars().all()
    for row in rows:
        when = row.time if row.time.tzinfo else row.time.replace(tzinfo=UTC)
        assert when >= UNSEEN_FROM
    assert inserted == len(rows)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Quarterly rollup
# ---------------------------------------------------------------------------


def test_quarterly_rollup_counts_and_signs() -> None:
    observations = [
        {"time": datetime(2026, 7, 5, tzinfo=UTC), "ic": +0.03},
        {"time": datetime(2026, 8, 5, tzinfo=UTC), "ic": +0.01},
        {"time": datetime(2026, 10, 5, tzinfo=UTC), "ic": -0.02},
    ]
    rollup = quarterly_rollup(observations)
    assert [q["quarter"] for q in rollup] == ["2026-Q3", "2026-Q4"]
    assert rollup[0]["positive"] is True
    assert rollup[0]["n"] == 2
    assert rollup[1]["positive"] is False


def test_rollup_mean_not_last_value_decides_the_sign() -> None:
    """A quarter that ends on a good day but averaged negative is negative."""
    observations = [
        {"time": datetime(2026, 7, 1 + i, tzinfo=UTC), "ic": -0.05}
        for i in range(9)
    ] + [{"time": datetime(2026, 7, 20, tzinfo=UTC), "ic": +0.01}]
    rollup = quarterly_rollup(observations)
    assert rollup[0]["positive"] is False


def test_rollup_of_nothing_is_empty() -> None:
    assert quarterly_rollup([]) == []


# ---------------------------------------------------------------------------
# Multi-factor registry (registered 2026-08-02)
# ---------------------------------------------------------------------------


def test_registry_is_pinned_to_the_registration() -> None:
    """Names, boundaries and history requirements are fixed by GO_LIVE.md.
    Editing them after the fact is the move the registration forbids."""
    from services.research.funding_watch import WATCHED_FACTORS

    by_name = {f.name: f for f in WATCHED_FACTORS}
    assert set(by_name) == {
        "funding_carry_24h_plus",
        "reversal_8h_minus",
        "momentum_72h_minus",
        "low_vol_7d_minus",
        "momentum_24h_minus",
        "volume_surge_minus",
        "funding_delta_minus",
        "blend_registered_v1",
        "oi_buildup_24h_minus",
        "crowded_longs_minus",
        "taker_buying_24h_minus",
        "near_high_fade_minus",
    }
    assert by_name["funding_carry_24h_plus"].unseen_from == datetime(2026, 7, 1, tzinfo=UTC)
    for name in ("reversal_8h_minus", "momentum_72h_minus", "low_vol_7d_minus"):
        assert by_name[name].unseen_from == datetime(2026, 8, 2, tzinfo=UTC)
    for name in (
        "momentum_24h_minus",
        "volume_surge_minus",
        "funding_delta_minus",
        "blend_registered_v1",
    ):
        assert by_name[name].unseen_from == datetime(2026, 8, 7, tzinfo=UTC)
    for name in (
        "oi_buildup_24h_minus",
        "crowded_longs_minus",
        "taker_buying_24h_minus",
    ):
        assert by_name[name].unseen_from == datetime(2026, 8, 9, tzinfo=UTC)
    assert by_name["low_vol_7d_minus"].min_history == 22
    # The orthogonal-data batch declares its series; price-only factors
    # declare nothing (a series requirement is part of the registration).
    assert by_name["oi_buildup_24h_minus"].requires == ("open_interest",)
    assert by_name["oi_buildup_24h_minus"].min_history == 4
    assert by_name["crowded_longs_minus"].requires == ("global_long_short",)
    assert by_name["crowded_longs_minus"].min_history == 1
    assert by_name["taker_buying_24h_minus"].requires == ("taker_buy_ratio",)
    assert by_name["taker_buying_24h_minus"].min_history == 4
    assert by_name["funding_carry_24h_plus"].requires == ()
    assert by_name["near_high_fade_minus"].unseen_from == datetime(2026, 8, 28, tzinfo=UTC)
    assert by_name["near_high_fade_minus"].min_history == 22
    assert by_name["near_high_fade_minus"].requires == ()


def test_reversal_signal_fades_the_last_move_causally() -> None:
    """A contract that just jumped must score LOW (fade), and the signal at t
    must depend only on closes up to t."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "reversal_8h_minus").build
    close = np.full((10, 2), 100.0)
    close[5, 0] = 110.0  # symbol 0 jumps at t=5
    signal = build(close, np.zeros_like(close), np.ones_like(close))
    assert signal[5, 0] < signal[5, 1]  # the jumper is faded
    assert np.allclose(signal[1:5, 0], signal[1:5, 1])  # nothing before t=5 reacts
    assert np.all(np.isnan(signal[0]))  # no prior close -> no signal


def test_momentum_fade_uses_the_9_stamp_lookback() -> None:
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "momentum_72h_minus").build
    close = np.full((15, 2), 100.0)
    close[:, 0] = np.linspace(100, 150, 15)  # steady riser
    signal = build(close, np.zeros_like(close), np.ones_like(close))
    assert np.all(np.isnan(signal[:9]))
    assert signal[12, 0] < signal[12, 1]  # the riser is faded


def test_low_vol_prefers_the_quiet_contract() -> None:
    from services.research.funding_watch import WATCHED_FACTORS

    rng = np.random.default_rng(30)
    build = next(f for f in WATCHED_FACTORS if f.name == "low_vol_7d_minus").build
    n = 40
    close = np.empty((n, 2))
    close[:, 0] = 100 * np.cumprod(1 + rng.normal(0, 0.05, n))  # wild
    close[:, 1] = 100 * np.cumprod(1 + rng.normal(0, 0.001, n))  # quiet
    signal = build(close, np.zeros_like(close), np.ones_like(close))
    assert signal[-1, 1] > signal[-1, 0]  # quiet scores higher


@pytest.mark.asyncio
async def test_each_factor_respects_its_own_unseen_boundary() -> None:
    """Stamps in July are legitimate for the funding factor but SEEN data for
    the three factors registered on Aug 2 -- one dataset, different verdicts."""
    from services.research.funding_watch import WATCHED_FACTORS

    engine, factory = await _factory()

    class _MultiStub(FundingWatchService):
        def __init__(self, session_factory, dataset):
            super().__init__(session_factory)
            self._dataset = dataset

        async def observe_once(self) -> int:  # type: ignore[override]
            from services.research.funding_watch import compute_signal_ics

            grid, close, funding = self._dataset
            # Plausible synthetic extras so the orthogonal-data factors also
            # produce signals -- the boundary must be what refuses them.
            rng = np.random.default_rng(99)
            extras = {
                "open_interest": rng.uniform(1e6, 2e6, close.shape),
                "global_long_short": rng.uniform(0.5, 3.0, close.shape),
                "taker_buy_ratio": rng.uniform(0.7, 1.4, close.shape),
            }
            inserted = 0
            async with self._session_factory() as session:
                now = datetime.now(UTC)
                for f in WATCHED_FACTORS:
                    if f.requires:
                        signal = f.build(close, funding, np.ones_like(close), extras)
                    else:
                        signal = f.build(close, funding, np.ones_like(close))
                    for stamp_ms, ic, n_symbols in compute_signal_ics(
                        grid, close, signal, HORIZON_STAMPS, f.min_history
                    ):
                        when = datetime.fromtimestamp(stamp_ms / 1000, UTC)
                        if when < f.unseen_from:
                            continue
                        session.add(
                            FactorWatchObservation(
                                time=when, factor=f.name,
                                horizon_stamps=HORIZON_STAMPS, ic=ic,
                                n_symbols=n_symbols, resolved_at=now,
                            )
                        )
                        inserted += 1
                await session.commit()
            return inserted

    # A July window: 40 stamps starting Jul 3 -- unseen for funding, seen
    # for the Aug-2 registrations.
    watch = _MultiStub(factory, _dataset(40, start=datetime(2026, 7, 3, tzinfo=UTC)))
    await watch.observe_once()
    async with factory() as session:
        rows = (await session.execute(select(FactorWatchObservation))).scalars().all()
    factors_present = {r.factor for r in rows}
    assert factors_present == {"funding_carry_24h_plus"}
    await engine.dispose()



def test_volume_surge_fades_the_spiking_contract() -> None:
    """A contract trading 5x its trailing volume must score LOW (fade)."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "volume_surge_minus").build
    n = 40
    close = np.full((n, 2), 100.0)
    volume = np.full((n, 2), 10.0)
    volume[-1, 0] = 50.0  # symbol 0 spikes to 5x
    signal = build(close, np.zeros_like(close), volume)
    assert signal[-1, 0] < signal[-1, 1]
    # Causality: nothing before the spike reacts differently across symbols.
    assert np.allclose(signal[25:-1, 0], signal[25:-1, 1])


def test_funding_delta_fades_building_crowding() -> None:
    """Funding rising above its own 7d mean scores LOW; the level alone,
    however high, scores neutral -- delta is the thing, not level."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "funding_delta_minus").build
    n = 40
    close = np.full((n, 2), 100.0)
    funding = np.zeros((n, 2))
    funding[:, 0] = 0.001          # high but CONSTANT -> delta ~ 0
    funding[-1, 1] = 0.0008        # low level but RISING from 0
    signal = build(close, funding, np.ones_like(close))
    assert signal[-1, 1] < signal[-1, 0]  # the riser is faded, not the level


def test_oi_buildup_fades_the_ramping_contract() -> None:
    """The contract whose open interest grew fastest over 24h scores LOW;
    no signal before a full 3-stamp lookback; dead (zero) OI stays absent."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "oi_buildup_24h_minus").build
    n = 10
    close = np.full((n, 3), 100.0)
    oi = np.full((n, 3), 1_000.0)
    oi[:, 0] = 1_000.0 * 1.5 ** np.arange(n)  # symbol 0: OI ramping hard
    oi[:, 2] = 0.0                            # symbol 2: dead contract
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros, {"open_interest": oi})
    assert np.all(np.isnan(signal[:3]))          # lookback not yet available
    assert signal[-1, 0] < signal[-1, 1]         # the ramper is faded
    assert np.all(np.isnan(signal[3:, 2]))       # zero OI -> absent, not inf


def test_crowded_longs_fades_the_long_crowd() -> None:
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "crowded_longs_minus").build
    n = 5
    close = np.full((n, 2), 100.0)
    ls = np.full((n, 2), 1.0)
    ls[:, 0] = 3.0  # symbol 0: three accounts long per short -- the crowd
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros, {"global_long_short": ls})
    assert signal[-1, 0] < signal[-1, 1]


def test_taker_buying_is_lagged_one_full_stamp() -> None:
    """A taker-buying spike at stamp k must NOT move the signal at k -- the
    registered one-stamp lag makes the signal causal regardless of the
    endpoint's period-stamping convention. It shows up from k+1."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "taker_buying_24h_minus").build
    n = 12
    close = np.full((n, 2), 100.0)
    taker = np.full((n, 2), 1.0)
    k = 6
    taker[k, 0] = 5.0  # symbol 0: heavy taker buying at stamp k
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros, {"taker_buy_ratio": taker})
    assert signal[k, 0] == signal[k, 1]          # nothing leaks into stamp k
    assert signal[k + 1, 0] < signal[k + 1, 1]   # the buyer is faded after


@pytest.mark.asyncio
async def test_missing_series_skips_the_factor_not_the_cycle(monkeypatch) -> None:
    """A cycle where a data endpoint fails must still record every factor
    whose inputs exist -- and must record NOTHING for the starved factor."""
    import services.research.funding_watch as fw

    engine, factory = await _factory()
    n, syms = 30, 40
    rng = np.random.default_rng(11)
    # Future-dated grid: past every factor's unseen_from boundary.
    grid = _grid(n, start=datetime(2026, 8, 10, tzinfo=UTC))
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, syms)), axis=0)
    funding = rng.normal(0, 0.0005, (n, syms))
    volume = rng.lognormal(3, 0.5, (n, syms))
    extras = {
        "open_interest": rng.uniform(1e6, 2e6, (n, syms)),
        "global_long_short": np.full((n, syms), np.nan),  # endpoint down
        "taker_buy_ratio": rng.uniform(0.7, 1.4, (n, syms)),
    }
    monkeypatch.setattr(
        fw, "_fetch_recent", lambda: (grid, close, funding, volume, extras)
    )
    watch = FundingWatchService(factory)
    inserted = await watch.observe_once()
    assert inserted > 0
    async with factory() as session:
        rows = (await session.execute(select(FactorWatchObservation))).scalars().all()
    recorded = {r.factor for r in rows}
    assert "oi_buildup_24h_minus" in recorded
    assert "taker_buying_24h_minus" in recorded
    assert "crowded_longs_minus" not in recorded  # starved -> silent absence
    await engine.dispose()


def test_near_high_fade_fades_the_contract_at_its_own_high() -> None:
    """The 2026-08-28 external-claim kernel: the contract AT its 7-day high
    scores LOW (faded); one 20% below its high scores higher; the signal at
    t must not react to a future spike."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "near_high_fade_minus").build
    n = 40
    close = np.full((n, 2), 100.0)
    close[:, 0] = np.linspace(80, 100, n)   # symbol 0: closing AT its high
    close[30, 1] = 125.0                     # symbol 1: high 9 stamps back, IN window
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros)
    assert signal[-1, 0] < signal[-1, 1]     # at-high is faded harder
    # Causality: plant a future spike; nothing before it may change.
    spiked = close.copy()
    spiked[-1, 1] = 500.0
    signal2 = build(spiked, zeros, zeros)
    assert np.allclose(signal[:-1], signal2[:-1], equal_nan=True)
    # The ratio form is bounded: at-high is exactly -1, never below.
    assert signal[-1, 0] == -1.0


def test_near_high_window_is_21_stamps_not_all_history() -> None:
    """The registered variable is the max over the LAST 21 stamps: a high
    that has aged out of the window must stop mattering. An unbounded
    running max would fail here (it passed the basic test above -- that gap
    was itself a review finding)."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "near_high_fade_minus").build
    n = 40
    close = np.full((n, 2), 100.0)
    close[5, 0] = 125.0   # symbol 0: spike aged OUT of the window at t=39
    close[30, 1] = 125.0  # symbol 1: same spike, still IN the window
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros)
    assert signal[-1, 0] == -1.0                      # back AT its 21-stamp high
    assert signal[-1, 1] == -(100.0 / 125.0)          # still below its high
    assert signal[-1, 0] < signal[-1, 1]


def test_near_high_refuses_short_history_symbols() -> None:
    """A perp listing mid-grid must be ABSENT, not scored: nanmax over its
    NaN-padded 'window' would pin its first stamp at exactly -1.0 -- the
    fade extreme, price-independent -- and new-listing drift would enter the
    permanent record as fake positive IC (review finding, 2026-08-28)."""
    from services.research.funding_watch import WATCHED_FACTORS

    build = next(f for f in WATCHED_FACTORS if f.name == "near_high_fade_minus").build
    n = 40
    close = np.full((n, 2), 100.0)
    close[:, 1] = np.nan
    close[37:, 1] = [50.0, 40.0, 30.0]  # a dumping brand-new listing
    zeros = np.zeros_like(close)
    signal = build(close, zeros, zeros)
    assert np.all(np.isnan(signal[:, 1]))   # never scored, at any stamp
    assert signal[-1, 0] == -1.0            # the full-history symbol still is


# ---------------------------------------------------------------------------
# Fetch integrity (2026-08-09 review fixes)
# ---------------------------------------------------------------------------


def test_the_in_progress_bar_is_never_a_close() -> None:
    """Binance includes the current unfinished candle in fetch_ohlcv; its
    live mid-bar price must never become a permanently recorded IC exit."""
    from services.research.funding_watch import _closed_bars

    open_time = 1786262400000  # an exact 8h boundary
    bars = [
        [open_time - _MS_8H, 0, 0, 0, 101.0, 5.0],  # closed
        [open_time, 0, 0, 0, 999.0, 1.0],           # in progress
    ]
    mid_bar_now = open_time + _MS_8H // 2
    kept = _closed_bars(bars, mid_bar_now)
    assert open_time - _MS_8H in kept
    assert open_time not in kept
    # Once the bar closes, the next cycle picks it up.
    kept_later = _closed_bars(bars, open_time + _MS_8H)
    assert open_time in kept_later


def test_a_degraded_series_is_dropped_not_prefix_scored() -> None:
    """Rate limiting or a partial outage mid-loop leaves an alphabetical
    PREFIX of the universe; an IC on that biased sub-universe must never be
    recorded, so the trust gate refuses the series for the cycle."""
    from services.research.funding_watch import _series_is_trustworthy

    assert _series_is_trustworthy(attempts=450, errors=0)
    assert _series_is_trustworthy(attempts=450, errors=45)      # tolerated flake
    assert not _series_is_trustworthy(attempts=450, errors=120)  # mid-loop 429s
    assert not _series_is_trustworthy(attempts=0, errors=0)      # nothing fetched


def test_taker_8h_ratio_composes_both_4h_halves() -> None:
    """Period-START stamping (verified empirically): the 8h ratio at stamp t
    is the volume sum of the points stamped t and t+4h. A stamp missing
    either half stays absent -- a half-window ratio is a different variable
    than the registered one."""
    from services.research.funding_watch import _MS_4H, _parse_taker_8h

    t = 1786233600000  # exact 8h boundary
    rows = [
        {"timestamp": t, "buyVol": "30.0", "sellVol": "10.0"},
        {"timestamp": t + _MS_4H, "buyVol": "10.0", "sellVol": "30.0"},
        # Next stamp: only the first half arrived.
        {"timestamp": t + _MS_8H, "buyVol": "50.0", "sellVol": "1.0"},
    ]
    out = _parse_taker_8h(rows)
    assert out == {t: (30.0 + 10.0) / (10.0 + 30.0)}  # both halves, vol-summed
    assert t + _MS_8H not in out                       # half-window refused
    assert t + _MS_4H not in out                       # off-grid never a stamp


def test_blend_is_the_frozen_equal_weight_of_registered_components() -> None:
    """One symbol strong on every component must top the blend; a cell absent
    in all components stays absent instead of becoming a tradeable zero."""
    from services.research.funding_watch import WATCHED_FACTORS

    rng = np.random.default_rng(41)
    build = next(f for f in WATCHED_FACTORS if f.name == "blend_registered_v1").build
    n, m = 60, 30
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, (n, m)), axis=0)
    funding = rng.normal(0, 0.0005, (n, m))
    volume = rng.lognormal(3, 0.5, (n, m))
    # Symbol 0: engineered to score high on ALL components at the last stamp:
    # highest funding carry, just fell (reversal+), fell over 3d (momentum
    # fade+), and quiet.
    funding[-5:, 0] = 0.004
    close[-10:, 0] = np.linspace(100, 92, 10)  # steady decline, low vol
    signal = build(close, funding, volume)
    assert np.nanargmax(signal[-1]) == 0
    # An all-absent column stays absent.
    close2 = close.copy()
    close2[:, 1] = np.nan
    funding2 = funding.copy()
    funding2[:, 1] = np.nan
    signal2 = build(close2, funding2, volume)
    assert np.all(np.isnan(signal2[:, 1]))
