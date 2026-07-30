"""Fetch a broad crypto universe at hourly resolution into a local cache.

    python scripts/build_universe.py --top 300 --years 2

Why this exists: the live signal was measured across FIVE crypto symbols and
found to have no edge at the traded horizon, with a faint hint at ~60 minutes
resting on ~150 independent observations per symbol. That is far too little to
conclude anything. Breadth is the only way to buy statistical power without
waiting months, and a wide universe is also the precondition for testing
CROSS-SECTIONAL structure (rank symbols against each other) rather than the
per-symbol time-series prediction the platform does today.

Resolution is matched to the hypothesis: the signal under test lives at ~1
hour, so hourly bars are the right grain. Pulling 1-minute bars for 300
symbols would be ~40M rows to answer an hourly question.

The cache is LOCAL and deliberately not written to the production database.
This is research data; nothing in the live platform reads it. Only if a
strategy is promoted does the production universe expand.

    SURVIVORSHIP BIAS -- READ THIS.
    The universe is selected from pairs that are listed and active TODAY.
    Coins that were delisted, collapsed, or lost their listing during the
    sample are absent, and they are exactly the losers. Any result computed
    on this cache is therefore an UPPER BOUND on what was actually
    achievable, and a long-only result is affected far more than a
    market-neutral long/short one. Every report built on this data must
    repeat that caveat.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Leveraged tokens track a multiple of the underlying and rebalance daily;
# their returns are not comparable to spot. Stable-to-stable pairs have no
# meaningful direction. Both would pollute a cross-sectional ranking.
_EXCLUDE_SUFFIXES = ("UP/USDT", "DOWN/USDT", "BULL/USDT", "BEAR/USDT")
_STABLES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "EUR", "GBP", "AEUR",
    "PAXG", "XUSD", "USD1",
}

_MS_PER_HOUR = 3_600_000


def select_universe(exchange, top: int, min_quote_volume: float) -> list[str]:
    """Rank active USDT spot pairs by 24h quote volume."""
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    candidates: list[tuple[float, str]] = []
    for symbol, market in markets.items():
        if not (market.get("spot") and market.get("active")):
            continue
        if market.get("quote") != "USDT":
            continue
        if symbol.endswith(_EXCLUDE_SUFFIXES):
            continue
        if market.get("base") in _STABLES:
            continue
        ticker = tickers.get(symbol) or {}
        volume = float(ticker.get("quoteVolume") or 0.0)
        if volume < min_quote_volume:
            continue
        candidates.append((volume, symbol))
    candidates.sort(reverse=True)
    return [s for _, s in candidates[:top]]


def fetch_series(
    exchange, symbol: str, timeframe: str, since_ms: int, max_calls: int = 60
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Page through OHLCV for one symbol. Returns (times_ms, close, volume)."""
    times: list[int] = []
    closes: list[float] = []
    volumes: list[float] = []
    cursor = since_ms
    for _ in range(max_calls):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
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


def build_grid(
    per_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align every symbol onto one time grid; absent history stays NaN.

    NaN rather than forward-fill or zero: a symbol that had not listed yet
    must be *absent* from the cross-section at that timestamp, not present at
    a stale price. Filling would invent tradeable history.
    """
    all_times = np.unique(np.concatenate([t for t, _, _ in per_symbol.values()]))
    symbols = np.array(sorted(per_symbol))
    close = np.full((all_times.size, symbols.size), np.nan)
    volume = np.full((all_times.size, symbols.size), np.nan)
    index_of = {t: i for i, t in enumerate(all_times)}
    for j, symbol in enumerate(symbols):
        times, closes, volumes = per_symbol[str(symbol)]
        rows = np.fromiter((index_of[t] for t in times), dtype=np.int64, count=times.size)
        close[rows, j] = closes
        volume[rows, j] = volumes
    return all_times, symbols, close, volume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--min-quote-volume", type=float, default=1_000_000.0)
    parser.add_argument("--min-bars", type=int, default=2_000)
    parser.add_argument("--out", default="data/universe_1h.npz")
    args = parser.parse_args()

    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    print(f"selecting top {args.top} USDT spot pairs by 24h quote volume...")
    universe = select_universe(exchange, args.top, args.min_quote_volume)
    print(f"  {len(universe)} symbols pass the liquidity filter")

    since_ms = int((time.time() - args.years * 365.25 * 86400) * 1000)
    per_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    skipped: list[str] = []
    started = time.time()
    for i, symbol in enumerate(universe, 1):
        try:
            times, closes, volumes = fetch_series(
                exchange, symbol, args.timeframe, since_ms
            )
        except Exception as exc:  # a single bad symbol must not end the run
            skipped.append(f"{symbol} ({type(exc).__name__})")
            continue
        if times.size < args.min_bars:
            skipped.append(f"{symbol} (only {times.size} bars)")
            continue
        per_symbol[symbol] = (times, closes, volumes)
        if i % 25 == 0:
            print(
                f"  {i}/{len(universe)}  kept={len(per_symbol)}  "
                f"{time.time() - started:.0f}s elapsed"
            )

    if not per_symbol:
        print("error: no symbol produced usable history", file=sys.stderr)
        return 1

    times, symbols, close, volume = build_grid(per_symbol)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times=times,
        symbols=symbols,
        close=close,
        volume=volume,
        timeframe=np.array([args.timeframe]),
        # Stamped into the artifact so no downstream report can quietly
        # forget how the universe was chosen.
        survivorship_note=np.array(
            ["universe selected from pairs ACTIVE TODAY; delisted losers absent"]
        ),
    )
    coverage = float(np.isfinite(close).mean())
    print(
        f"\nwrote {out}  grid={times.size} bars x {symbols.size} symbols  "
        f"coverage={coverage * 100:.1f}%  ({time.time() - started:.0f}s)"
    )
    if skipped:
        print(f"skipped {len(skipped)}: {', '.join(skipped[:8])}"
              + (" ..." if len(skipped) > 8 else ""))
    print(
        "\nSURVIVORSHIP: this universe is today's survivors. Results computed "
        "on it are an UPPER BOUND."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
