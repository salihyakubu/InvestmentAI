"""One-time backfill of the funding watch's July 1-21 gap. Survivorship-safe.

    export DATABASE_URL=...   # production Postgres
    python scripts/backfill_funding_watch_july.py

The funding factor's registered unseen boundary is 2026-07-01, but the live
watch only reaches ~10 days back, so its record starts Jul 22 -- three weeks
of legitimate unseen data unrecorded. Fetching those stamps from the LIVE
exchange now would quietly exclude anything delisted since July; the
data.binance.vision monthly archive retains delisted contracts, so the
backfill is computed from the same survivorship-free source as PR #64 and is
registered in GO_LIVE.md (2026-08-02).

Append-only holds: stamps already recorded by the live watch are NEVER
touched (the insert skips them), only missing stamps strictly inside
[unseen_from, first-recorded) are added, and the rows are identifiable by
their resolved_at date. Idempotent: a second run inserts nothing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.research.funding_watch import (  # noqa: E402
    FACTOR_NAME,
    HORIZON_STAMPS,
    UNSEEN_FROM,
    compute_resolvable_ics,
)

_MS_8H = 8 * 3_600_000
# June is fetched only to warm the carry lookback for the first July stamps;
# no June stamp is ever inserted (the boundary check forbids it anyway).
_MONTHS = ["2026-06", "2026-07"]


def _load_archive_fetcher():
    spec = importlib.util.spec_from_file_location(
        "svfree", Path(__file__).resolve().parent / "build_survivorship_free_funding.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_july_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch Jun+Jul bars/funding for every archived contract, on the 8h grid."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    archive = _load_archive_fetcher()
    symbols = archive.enumerate_all_symbols()
    print(f"{len(symbols)} archived USDT perpetuals; fetching {_MONTHS}...", flush=True)

    results: dict[str, tuple[dict, dict]] = {}
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {
            pool.submit(archive.fetch_symbol, s, _MONTHS): s for s in symbols
        }
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                payload = future.result()
            except Exception:
                payload = None
            if payload is not None and payload[0]:
                results[symbol] = payload
            if done % 150 == 0:
                print(f"  {done}/{len(symbols)}  kept={len(results)}", flush=True)

    grid = np.array(
        sorted({stamp for bars, _ in results.values() for stamp in bars}),
        dtype=np.int64,
    )
    kept = sorted(results)
    close = np.full((grid.size, len(kept)), np.nan)
    funding = np.full((grid.size, len(kept)), np.nan)
    index_of = {stamp: i for i, stamp in enumerate(grid)}
    for j, symbol in enumerate(kept):
        bars, rates = results[symbol]
        last = np.nan
        for stamp in sorted(bars):
            i = index_of[stamp]
            close[i, j] = bars[stamp][0]
            if stamp in rates:
                last = rates[stamp]
            funding[i, j] = last
    print(
        f"grid: {grid.size} stamps x {len(kept)} contracts "
        f"({datetime.fromtimestamp(grid[0] / 1000, UTC):%b %d} .. "
        f"{datetime.fromtimestamp(grid[-1] / 1000, UTC):%b %d})",
        flush=True,
    )
    return grid, close, funding


async def insert_missing(candidates: list[tuple[int, float, int]]) -> int:
    import asyncpg

    url = os.environ["DATABASE_URL"].replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch(
            "select time from factor_watch where factor = $1", FACTOR_NAME
        )
        existing = {
            (r["time"].replace(tzinfo=UTC) if r["time"].tzinfo is None else r["time"])
            for r in rows
        }
        earliest_recorded = min(existing) if existing else None
        now = datetime.now(UTC)
        inserted = 0
        for stamp_ms, ic, n_symbols in candidates:
            when = datetime.fromtimestamp(stamp_ms / 1000, UTC)
            if when < UNSEEN_FROM or when in existing:
                continue
            # Only fill the gap BEFORE the live record began; the live watch
            # owns everything from its first row onward.
            if earliest_recorded is not None and when >= earliest_recorded:
                continue
            await conn.execute(
                """insert into factor_watch
                   (time, factor, horizon_stamps, ic, n_symbols, resolved_at)
                   values ($1, $2, $3, $4, $5, $6)
                   on conflict do nothing""",
                when, FACTOR_NAME, HORIZON_STAMPS, ic, n_symbols, now,
            )
            inserted += 1
        return inserted
    finally:
        await conn.close()


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        print("error: DATABASE_URL not set", file=sys.stderr)
        return 2
    grid, close, funding = build_july_grid()
    candidates = compute_resolvable_ics(grid, close, funding)
    july = [
        c for c in candidates
        if datetime.fromtimestamp(c[0] / 1000, UTC) >= UNSEEN_FROM
    ]
    print(f"resolvable stamps at/after {UNSEEN_FROM.date()}: {len(july)}")
    inserted = asyncio.run(insert_missing(candidates))
    print(f"inserted {inserted} backfilled observations (existing rows untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
