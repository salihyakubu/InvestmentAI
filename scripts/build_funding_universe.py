"""Fetch perpetual-futures bars AND funding-rate history into a local cache.

    python scripts/build_funding_universe.py --top 150 --years 1.5

Why funding, and why now. Three research passes have established that the
price series alone yields nothing this platform can trade: the live 5-minute
signal has no edge, cross-sectional reversal is real but costs more to
harvest than it pays, and no turnover setting rescues it. Every one of those
signals was computed from prices -- the same series every other participant
is already ranking.

Funding rates are different in two ways that both matter here:

1. They are NOT in the price series. Funding is the periodic payment between
   perpetual longs and shorts, so it measures POSITIONING -- who is crowded
   and how badly they want to stay. That is information about other traders,
   not about price history.
2. They are stamped every 8 hours, so a funding-based signal changes slowly
   BY CONSTRUCTION. Turnover was the binding constraint on everything found
   so far; this attacks it at the source rather than by damping a fast signal
   after the fact (which was already tested, and failed).

Bars are perpetuals, not spot, because that is the instrument a funding
signal would actually trade -- and it is also the cheaper venue (2bps maker
against spot's 10bps).

    CAUSALITY. Funding is published on an 8-hour grid and forward-filled onto
    the hourly grid, so the value at hour t is the last rate PUBLISHED at or
    before t. Never the next one. A one-stamp lookahead here would invent a
    spectacular strategy out of nothing.

    SURVIVORSHIP. As with the spot universe, contracts are selected from
    those listed today. Delisted perps are absent and they are the losers.
    Every figure is an upper bound.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_MS_PER_HOUR = 3_600_000


def select_perps(exchange, top: int, min_quote_volume: float) -> list[str]:
    """Rank active linear USDT perpetuals by 24h quote volume."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates: list[tuple[float, str]] = []
    for symbol, market in markets.items():
        if not (market.get("swap") and market.get("linear") and market.get("active")):
            continue
        if market.get("quote") != "USDT":
            continue
        volume = float((tickers.get(symbol) or {}).get("quoteVolume") or 0.0)
        if volume < min_quote_volume:
            continue
        candidates.append((volume, symbol))
    candidates.sort(reverse=True)
    return [s for _, s in candidates[:top]]


def fetch_bars(exchange, symbol: str, since_ms: int, max_calls: int = 60):
    times: list[int] = []
    closes: list[float] = []
    volumes: list[float] = []
    cursor = since_ms
    for _ in range(max_calls):
        batch = exchange.fetch_ohlcv(symbol, "1h", since=cursor, limit=1000)
        if not batch:
            break
        for row in batch:
            times.append(int(row[0]))
            closes.append(float(row[4]))
            volumes.append(float(row[5]))
        nxt = int(batch[-1][0]) + 1
        if nxt <= cursor or len(batch) < 1000:
            break
        cursor = nxt
    return (
        np.array(times, dtype=np.int64),
        np.array(closes, dtype=np.float64),
        np.array(volumes, dtype=np.float64),
    )


def fetch_funding(exchange, symbol: str, since_ms: int, max_calls: int = 20):
    """Page through the 8-hourly funding-rate history."""
    stamps: list[int] = []
    rates: list[float] = []
    cursor = since_ms
    for _ in range(max_calls):
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=500)
        if not batch:
            break
        for row in batch:
            stamps.append(int(row["timestamp"]))
            rates.append(float(row["fundingRate"]))
        nxt = int(batch[-1]["timestamp"]) + 1
        if nxt <= cursor or len(batch) < 500:
            break
        cursor = nxt
    return np.array(stamps, dtype=np.int64), np.array(rates, dtype=np.float64)


def forward_fill_onto_grid(
    grid: np.ndarray, stamps: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Map an 8-hourly series onto the hourly grid, strictly causally.

    ``searchsorted(..., side="right") - 1`` picks the last stamp at or before
    each grid point. Anything before the first stamp stays NaN rather than
    borrowing the earliest known value backwards in time.
    """
    out = np.full(grid.size, np.nan)
    if stamps.size == 0:
        return out
    order = np.argsort(stamps)
    stamps, values = stamps[order], values[order]
    idx = np.searchsorted(stamps, grid, side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=150)
    parser.add_argument("--years", type=float, default=1.5)
    parser.add_argument("--min-quote-volume", type=float, default=5_000_000.0)
    parser.add_argument("--min-bars", type=int, default=2_000)
    parser.add_argument("--out", default="data/perp_funding_1h.npz")
    args = parser.parse_args()

    import ccxt

    exchange = ccxt.binance({"options": {"defaultType": "future"}, "enableRateLimit": True})
    print(f"selecting top {args.top} USDT perpetuals by 24h quote volume...", flush=True)
    universe = select_perps(exchange, args.top, args.min_quote_volume)
    print(f"  {len(universe)} contracts pass the liquidity filter", flush=True)

    since_ms = int((time.time() - args.years * 365.25 * 86400) * 1000)
    bars: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    funding: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []
    started = time.time()

    for i, symbol in enumerate(universe, 1):
        try:
            times, closes, volumes = fetch_bars(exchange, symbol, since_ms)
            if times.size < args.min_bars:
                skipped.append(f"{symbol} ({times.size} bars)")
                continue
            stamps, rates = fetch_funding(exchange, symbol, since_ms)
            if stamps.size < 50:
                skipped.append(f"{symbol} ({stamps.size} funding pts)")
                continue
        except Exception as exc:
            skipped.append(f"{symbol} ({type(exc).__name__})")
            continue
        bars[symbol] = (times, closes, volumes)
        funding[symbol] = (stamps, rates)
        if i % 20 == 0:
            print(
                f"  {i}/{len(universe)}  kept={len(bars)}  "
                f"{time.time() - started:.0f}s",
                flush=True,
            )

    if not bars:
        print("error: no contract produced usable history", file=sys.stderr)
        return 1

    grid = np.unique(np.concatenate([t for t, _, _ in bars.values()]))
    symbols = np.array(sorted(bars))
    close = np.full((grid.size, symbols.size), np.nan)
    volume = np.full((grid.size, symbols.size), np.nan)
    funding_grid = np.full((grid.size, symbols.size), np.nan)
    index_of = {t: i for i, t in enumerate(grid)}

    for j, symbol in enumerate(symbols):
        times, closes, volumes = bars[str(symbol)]
        rows = np.fromiter((index_of[t] for t in times), dtype=np.int64, count=times.size)
        close[rows, j] = closes
        volume[rows, j] = volumes
        stamps, rates = funding[str(symbol)]
        filled = forward_fill_onto_grid(grid, stamps, rates)
        # Only carry funding where the contract actually has a bar; otherwise
        # a delisted or not-yet-listed contract would appear tradeable.
        listed = np.isfinite(close[:, j])
        funding_grid[listed, j] = filled[listed]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times=grid,
        symbols=symbols,
        close=close,
        volume=volume,
        funding=funding_grid,
        survivorship_note=np.array(
            ["perpetuals ACTIVE TODAY; delisted contracts absent -> upper bound"]
        ),
    )
    print(
        f"\nwrote {out}  grid={grid.size} bars x {symbols.size} contracts  "
        f"price coverage={np.isfinite(close).mean() * 100:.1f}%  "
        f"funding coverage={np.isfinite(funding_grid).mean() * 100:.1f}%  "
        f"({time.time() - started:.0f}s)"
    )
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped[:6])}"
              + (" ..." if len(skipped) > 6 else ""))
    print("\nSURVIVORSHIP: today's survivors only. Results are an UPPER BOUND.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
