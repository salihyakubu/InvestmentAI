"""Build a survivorship-FREE perpetuals universe from Binance's public archive.

    python scripts/build_survivorship_free_funding.py --months 24

Every prior universe in this repo was assembled from the live API, which only
lists contracts trading TODAY -- delisted contracts, i.e. the losers, are
invisible. That bias is why the reversed funding result in PR #62 was refused
rather than claimed.

data.binance.vision closes the hole for free: the public archive retains
monthly funding-rate and kline files for EVERY contract that ever listed
(788 USDT perpetuals in the archive vs 726 trading live at the time of
writing). A contract that died mid-sample simply stops having files after its
delisting month -- which is exactly the truth the cross-section needs.

Grid choice: 8-hour bars, matching the canonical funding settlement times
(00/08/16 UTC). The archive publishes 8h klines directly, so no resampling.

    FUNDING INTERVAL NORMALISATION (this would have silently corrupted the
    ranking): the archive's fundingRate files carry a funding_interval_hours
    column, and it is NOT always 8 -- some contracts settle every 4 hours.
    A 4h-interval contract's per-stamp rate covers half the time of an
    8h-interval one, so raw per-stamp rates are not comparable across the
    cross-section. All rates are normalised to per-8h equivalents:
    rate * (8 / interval_hours).

The output carries an ``active_at_end`` flag per symbol (from the live
exchange), so the same file supports both the full-universe test and the
survivors-only comparison that measures the bias directly.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ARCHIVE = "https://data.binance.vision/data/futures/um/monthly"
_LISTING = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
    "?delimiter=/&prefix=data/futures/um/monthly/fundingRate/"
)
_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_MS_8H = 8 * 3_600_000


def _get(url: str, retries: int = 2) -> bytes | None:
    """Fetch a URL; None on 404 (month before listing / after delisting)."""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == retries:
                return None
            time.sleep(1.0 + attempt)
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.0 + attempt)
    return None


def enumerate_all_symbols() -> list[str]:
    """Every USDT perpetual that EVER had funding history -- the whole point."""
    raw = _get(_LISTING)
    if raw is None:
        raise RuntimeError("archive listing unreachable")
    prefixes = re.findall(
        r"<Prefix>data/futures/um/monthly/fundingRate/([A-Z0-9]+USDT)/</Prefix>",
        raw.decode(),
    )
    return sorted(set(prefixes))


def live_trading_symbols() -> set[str]:
    raw = _get(_EXCHANGE_INFO)
    if raw is None:
        return set()
    info = json.loads(raw)
    return {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING"}


def _read_csv_rows(blob: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode()
    rows = list(csv.reader(io.StringIO(text)))
    # Newer files carry a header row; older ones do not.
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]
    return [r for r in rows if r and r[0]]


def fetch_symbol(
    symbol: str, months: list[str]
) -> tuple[dict[int, tuple[float, float]], dict[int, float]] | None:
    """Return ({stamp: (close, quote_volume)}, {stamp: per-8h funding})."""
    bars: dict[int, tuple[float, float]] = {}
    funding: dict[int, float] = {}
    for month in months:
        kline_blob = _get(f"{_ARCHIVE}/klines/{symbol}/8h/{symbol}-8h-{month}.zip")
        if kline_blob is not None:
            for row in _read_csv_rows(kline_blob):
                stamp = (int(row[0]) // _MS_8H) * _MS_8H
                bars[stamp] = (float(row[4]), float(row[7]))
        funding_blob = _get(f"{_ARCHIVE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip")
        if funding_blob is not None:
            for row in _read_csv_rows(funding_blob):
                stamp = (int(row[0]) // _MS_8H) * _MS_8H
                interval = float(row[1]) if row[1] else 8.0
                rate = float(row[2])
                # Normalise to a per-8h equivalent (see module docstring).
                funding[stamp] = rate * (8.0 / max(interval, 1e-9))
    if not bars:
        return None
    return bars, funding


def month_range(n_months: int) -> list[str]:
    now = datetime.now(UTC)
    out: list[str] = []
    year, month = now.year, now.month
    for _ in range(n_months):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        out.append(f"{year:04d}-{month:02d}")
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--min-stamps", type=int, default=90, help="~30 days minimum life")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", default="data/perp_funding_8h_full.npz")
    args = parser.parse_args()

    months = month_range(args.months)
    symbols = enumerate_all_symbols()
    trading_now = live_trading_symbols()
    print(
        f"{len(symbols)} USDT perpetuals ever archived; "
        f"{len(trading_now)} trading today -> "
        f"{len([s for s in symbols if s not in trading_now])} delisted/settling "
        f"contracts RESTORED to the sample",
        flush=True,
    )
    print(f"window: {months[0]} .. {months[-1]}", flush=True)

    started = time.time()
    results: dict[str, tuple[dict[int, tuple[float, float]], dict[int, float]]] = {}
    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_symbol, s, months): s for s in symbols}
        done = 0
        for future in as_completed(futures):
            symbol = futures[future]
            done += 1
            try:
                payload = future.result()
            except Exception:
                payload = None
            if payload is not None and len(payload[0]) >= args.min_stamps:
                results[symbol] = payload
            if done % 100 == 0:
                print(
                    f"  {done}/{len(symbols)}  kept={len(results)}  "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )

    if not results:
        print("error: nothing fetched", file=sys.stderr)
        return 1

    grid = np.array(
        sorted({stamp for bars, _ in results.values() for stamp in bars}),
        dtype=np.int64,
    )
    kept = np.array(sorted(results))
    close = np.full((grid.size, kept.size), np.nan)
    volume = np.full((grid.size, kept.size), np.nan)
    funding_grid = np.full((grid.size, kept.size), np.nan)
    index_of = {stamp: i for i, stamp in enumerate(grid)}

    for j, symbol in enumerate(kept):
        bars, funding = results[str(symbol)]
        last_rate = np.nan
        for stamp in sorted(bars):
            i = index_of[stamp]
            price, qvol = bars[stamp]
            close[i, j] = price
            volume[i, j] = qvol
            # Strictly causal forward-fill: the rate known AT stamp t is the
            # last one settled at or before t.
            if stamp in funding:
                last_rate = funding[stamp]
            funding_grid[i, j] = last_rate

    active = np.array([s in trading_now for s in kept])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times=grid,
        symbols=kept,
        close=close,
        volume=volume,
        funding=funding_grid,
        active_at_end=active,
        note=np.array(
            ["SURVIVORSHIP-FREE: includes every contract ever listed in the window, "
             "delisted included; funding normalised to per-8h equivalents"]
        ),
    )
    print(
        f"\nwrote {out}  grid={grid.size} stamps x {kept.size} contracts  "
        f"({int(active.sum())} survivors, {int((~active).sum())} delisted)  "
        f"price coverage={np.isfinite(close).mean() * 100:.1f}%  "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
