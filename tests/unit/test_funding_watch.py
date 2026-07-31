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
