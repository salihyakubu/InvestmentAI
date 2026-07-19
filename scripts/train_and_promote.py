"""Bootstrap-train the tree models on REAL history and promote them for serving.

    PYTHONPATH=. python scripts/train_and_promote.py                 # full run
    PYTHONPATH=. python scripts/train_and_promote.py --trials 5 \
        --crypto-days 7 --stride 10                                  # faster run

This is the missing initial-training path: the AutoRetrainer's data loading is a
stub and nothing else ever registers a model, so the serving registry stays
empty and the prediction service emits only flat. This script:

1. Fetches real 1-MINUTE bars (matching the live ingestion cadence): deep
   history for the active crypto pairs via Binance public data, and the last
   ~7 days for the active stocks via Yahoo (their 1m depth limit).
2. Replays the live feature pipeline EXACTLY: for each timestamp it computes
   ``FeatureStore.compute_all_features`` over the trailing 200-bar window --
   the same code, window size, and therefore the same values the worker will
   feed the model at serve time (vwap excluded for cross-provider uniformity).
3. Labels with 5-bar (= 5-minute) forward returns using DATA-DRIVEN thresholds
   (pooled 30th/70th percentiles) -- fixed daily-scale thresholds would label
   nearly everything flat at this cadence.
4. Trains XGBoost + LightGBM through the platform's own ModelTrainer (with
   real forward returns for the regressors and probability calibration),
   using a chronological per-symbol 80/20 split.
5. Gates on out-of-sample accuracy, then registers + promotes both models in
   the filesystem ModelRegistry the worker serves from, and smoke-checks the
   actual serving path (ModelServer -> ensemble -> prediction).
6. Optionally mirrors metadata into the DB ``model_metadata`` table (which
   backs GET /api/v1/models) when DATABASE_URL is set and --write-db is given.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from core.enums import TimeFrame
from services.prediction.training.dataset_builder import (
    HORIZON,
    MIN_VAL_ACCURACY,
    WINDOW,  # noqa: F401  # re-exported: window size is part of this script's documented contract
    bars_matrix,
    build_dataset,
)

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]


async def _fetch_crypto(symbol: str, days: int) -> list[Any]:
    from services.data_ingestion.providers.ccxt_provider import CCXTDataProvider

    provider = CCXTDataProvider()
    try:
        end = datetime.now(UTC)
        return await provider.fetch_historical_bars(
            symbol, TimeFrame.M1, end - timedelta(days=days), end
        )
    finally:
        await provider._close_exchange()


async def _fetch_stock(symbol: str) -> list[Any]:
    from services.data_ingestion.providers.yahoo_provider import YahooDataProvider

    provider = YahooDataProvider()
    end = datetime.now(UTC)
    # Yahoo serves 1m data for ~the last 7 days only.
    return await provider.fetch_historical_bars(
        symbol, TimeFrame.M1, end - timedelta(days=7), end
    )


async def _build_dataset(
    crypto_days: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fetch real bars, then delegate replay/label/split to the shared builder."""
    per_symbol_cols: dict[str, dict[str, np.ndarray]] = {}

    for sym in CRYPTO_SYMBOLS:
        bars = await _fetch_crypto(sym, crypto_days)
        print(f"  {sym}: {len(bars)} 1m bars")
        per_symbol_cols[sym] = bars_matrix(bars)

    for sym in STOCK_SYMBOLS:
        try:
            bars = await _fetch_stock(sym)
        except Exception as exc:  # yahoo hiccups shouldn't kill the run
            print(f"  {sym}: fetch failed ({exc}); skipping")
            continue
        print(f"  {sym}: {len(bars)} 1m bars")
        per_symbol_cols[sym] = bars_matrix(bars)

    try:
        return build_dataset(per_symbol_cols, stride=stride)
    except ValueError as exc:
        raise SystemExit(f"{exc} -- aborting") from exc


def _write_db_metadata(rows: list[dict[str, Any]]) -> None:
    """Mirror registry entries into model_metadata (backs GET /api/v1/models)."""
    import os

    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL not set -- skipping DB metadata mirror")
        return

    async def _run() -> None:
        import asyncpg

        conn = await asyncpg.connect(
            url.replace("postgresql+asyncpg://", "postgres://", 1)
            .replace("postgresql://", "postgres://", 1),
            timeout=20,
        )
        try:
            for r in rows:
                await conn.execute(
                    """
                    INSERT INTO model_metadata
                      (id, model_name, model_type, version, hyperparameters,
                       validation_metrics, artifact_path, trained_at, is_active,
                       created_at)
                    VALUES (gen_random_uuid(), $1, $2, $3, $4::jsonb, $5::jsonb,
                            $6, $7, TRUE, $7)
                    ON CONFLICT (model_name, version) DO UPDATE
                      SET validation_metrics = EXCLUDED.validation_metrics,
                          is_active = TRUE
                    """,
                    r["model_name"], r["model_type"], r["version"],
                    r["hyperparameters"], r["metrics_json"],
                    r["artifact_path"], datetime.now(UTC),
                )
            print(f"DB metadata mirrored for {len(rows)} model(s)")
        finally:
            await conn.close()

    asyncio.run(_run())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crypto-days", type=int, default=30)
    parser.add_argument("--stride", type=int, default=HORIZON,
                        help="window stride in bars (default = label horizon)")
    parser.add_argument("--trials", type=int, default=15, help="hyperopt trials per model")
    parser.add_argument("--artifacts", default="model_artifacts")
    parser.add_argument("--write-db", action="store_true",
                        help="mirror metadata into model_metadata via DATABASE_URL")
    args = parser.parse_args(argv[1:])

    # Surface the shared dataset builder's label-threshold line on stdout.
    import logging

    logging.basicConfig(format="%(message)s")
    logging.getLogger("services.prediction.training.dataset_builder").setLevel(logging.INFO)

    print("[1/5] Fetching real 1-minute history + replaying live features ...")
    tr_X, tr_y, tr_r, va_X, va_y, va_r, names = asyncio.run(
        _build_dataset(args.crypto_days, args.stride)
    )
    print(f"      train rows: {len(tr_X)}  val rows: {len(va_X)}  features: {len(names)}")
    for name, arr in (("train", tr_y), ("val", va_y)):
        u, c = np.unique(arr, return_counts=True)
        print(f"      {name} class counts: {dict(zip(u.tolist(), c.tolist()))}")

    print(f"[2/5] Training xgboost + lightgbm (hyperopt trials={args.trials}) ...")
    from services.prediction.registry import ModelRegistry
    from services.prediction.training.trainer import ModelTrainer

    trainer = ModelTrainer()
    trained: dict[str, Any] = {}
    for model_type in ("xgboost", "lightgbm"):
        result, model = trainer.train_model(
            model_type=model_type,
            X_train=tr_X, y_train=tr_y, X_val=va_X, y_val=va_y,
            hyperopt=args.trials > 0, n_trials=args.trials,
            feature_names=names,
            returns_train=tr_r, returns_val=va_r,
        )
        print(f"      {model_type}: val_accuracy={result.val_accuracy:.4f} "
              f"val_loss={result.val_loss:.4f}")
        trained[model_type] = (result, model)

    print("[3/5] Out-of-sample gate ...")
    failures = [t for t, (res, _) in trained.items() if res.val_accuracy < MIN_VAL_ACCURACY]
    if failures:
        print(f"GATE FAILED: {failures} below {MIN_VAL_ACCURACY} val accuracy -> NOT promoting")
        return 1
    print(f"      both models beat the {MIN_VAL_ACCURACY} floor (random = 0.333)")

    print("[4/5] Registering + promoting in the serving registry ...")
    registry = ModelRegistry(artifact_base=Path(args.artifacts))
    db_rows: list[dict[str, Any]] = []
    for model_type, (result, model) in trained.items():
        model_id, version = registry.register(
            model=model, model_name=model_type, metrics=result.to_metrics()
        )
        registry.promote(model_id, version)
        print(f"      {model_type} v{version} promoted ({model_id})")
        import json as _json

        db_rows.append({
            "model_name": model_type, "model_type": model_type, "version": version,
            "hyperparameters": _json.dumps({"bootstrap": True, "trials": args.trials}),
            "metrics_json": _json.dumps(result.to_metrics()),
            "artifact_path": str(Path(args.artifacts) / model_type / f"v{version}"),
        })

    print("[5/5] Serving-path smoke check (ModelServer -> ensemble) ...")
    from services.prediction.serving import ModelServer

    server = ModelServer(registry=registry)
    server.load_active_models()
    if not server.feature_names:
        print("SMOKE FAILED: served feature_names empty")
        return 1
    probe = va_X[-1]
    out = server.predict(symbol="PROBE", features_flat=probe)
    print(f"      served prediction: direction={out.direction} "
          f"confidence={out.confidence:.3f} expected_return={out.expected_return:.5f}")

    if args.write_db:
        _write_db_metadata(db_rows)

    print("\nBOOTSTRAP COMPLETE: models are promoted; restart the worker to serve them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
