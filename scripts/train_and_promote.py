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
3. Labels per symbol with triple-barrier first-touch labels (volatility-scaled
   price barriers, time barrier = the 5-bar-horizon close; the dataset
   builder's default) while the regressors keep the real 5-step forward
   returns.
4. Trains XGBoost + LightGBM + CatBoost through the platform's own
   ModelTrainer (real forward returns for the regressors, isotonic-vs-sigmoid
   probability calibration, split-conformal state), with hyperopt scored by
   mean accuracy over purged, embargoed chronological CV folds.
5. Gates on out-of-sample accuracy AND a train/val overfit gap limit, then
   registers + promotes the models in the filesystem ModelRegistry the worker
   serves from, writes the per-symbol ``feature_reference.npz`` the
   continuous-learning feature-drift check reads, and smoke-checks the actual
   serving path (ModelServer -> ensemble -> prediction).
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
    MIN_BARS,
    MIN_VAL_ACCURACY,
    WINDOW,
    bars_matrix,
    build_dataset,
)
from services.prediction.training.walk_forward import purged_chrono_folds

CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# The models to bootstrap-train and (gate permitting) promote for serving.
MODEL_TYPES = ("xgboost", "lightgbm", "catboost")

# Purged chronological CV folds for hyperopt. 3 folds is enough at bootstrap
# data scale; more folds would shrink each fold's training side below what the
# tree models need to differentiate hyperparameters.
CV_FOLDS = 3

# OVERFIT GUARD: maximum tolerated train_accuracy - val_accuracy gap.
# Promoted-champion metrics live in the registry and later retrains only have
# to beat them, so a memorizing model must not pass merely by beating the
# MIN_VAL_ACCURACY floor -- it would poison the champion baseline.
MAX_TRAIN_VAL_GAP = 0.35

# Written next to the registry; the continuous-learning service reads it for
# per-symbol feature (data) drift checks. Absence just skips those checks.
FEATURE_REFERENCE_FILENAME = "feature_reference.npz"


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
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
    dict[str, dict[str, np.ndarray]],
]:
    """Fetch real bars, then delegate replay/label/split to the shared builder.

    Also returns the raw per-symbol OHLCV columns so the caller can slice the
    pooled validation rows back into per-symbol blocks (feature reference).
    """
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
        return build_dataset(per_symbol_cols, stride=stride), per_symbol_cols
    except ValueError as exc:
        raise SystemExit(f"{exc} -- aborting") from exc


def gate_failures(results: dict[str, Any]) -> list[str]:
    """Out-of-sample gate: accuracy floor + overfit guard.

    Each entry of *results* maps model_type -> TrainResult (anything exposing
    ``train_accuracy`` / ``val_accuracy``). Returns human-readable failure
    strings; empty means every model may be promoted.
    """
    failures: list[str] = []
    for model_type, result in results.items():
        if result.val_accuracy < MIN_VAL_ACCURACY:
            failures.append(
                f"{model_type}: val_accuracy {result.val_accuracy:.4f} "
                f"< floor {MIN_VAL_ACCURACY}"
            )
        gap = result.train_accuracy - result.val_accuracy
        if gap > MAX_TRAIN_VAL_GAP:
            failures.append(
                f"{model_type}: train-val accuracy gap {gap:.4f} "
                f"> {MAX_TRAIN_VAL_GAP} (memorizing, not learning)"
            )
    return failures


def per_symbol_val_rows(
    per_symbol_cols: dict[str, dict[str, np.ndarray]],
    va_X: np.ndarray,
    stride: int,
) -> dict[str, np.ndarray]:
    """Slice the pooled validation feature rows back into per-symbol blocks.

    Mirrors ``build_dataset``'s row arithmetic exactly: symbols iterate in
    insertion order, those under ``MIN_BARS`` are skipped, each contributes
    ``max(n_replay_steps - HORIZON, 0)`` labeled rows split 80/20
    chronologically, and the validation tails are concatenated in symbol
    order. Symbols with zero validation rows are omitted.

    Raises:
        ValueError: if the reconstructed row count does not equal
            ``len(va_X)`` (dataset-builder contract drift) -- better no
            feature reference than a misaligned one.
    """
    out: dict[str, np.ndarray] = {}
    offset = 0
    for sym, cols in per_symbol_cols.items():
        n_bars = len(cols["close"])
        if n_bars < MIN_BARS:
            continue
        n_steps = (n_bars - WINDOW) // stride + 1
        n_labeled = max(n_steps - HORIZON, 0)
        n_val = n_labeled - int(n_labeled * 0.8)
        if n_val > 0:
            out[sym] = np.asarray(va_X[offset : offset + n_val], dtype=np.float32)
        offset += n_val
    if offset != len(va_X):
        raise ValueError(
            f"per-symbol row reconstruction produced {offset} rows, "
            f"but the pooled validation split has {len(va_X)}"
        )
    return out


def write_feature_reference(
    artifact_dir: Path,
    feature_names: list[str],
    rows_by_symbol: dict[str, np.ndarray],
) -> Path:
    """Write the per-symbol feature-reference archive for drift detection.

    Schema (continuous-learning contract): ``feature_names`` = 1-D str array
    in the dataset builder's sorted-name order; one 2-D float array per symbol
    keyed by the raw symbol string (slashes round-trip through np.savez),
    columns aligned with ``feature_names``.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / FEATURE_REFERENCE_FILENAME
    arrays: dict[str, Any] = {"feature_names": np.array(feature_names), **rows_by_symbol}
    np.savez(path, **arrays)
    return path


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

    print("[1/6] Fetching real 1-minute history + replaying live features ...")
    (tr_X, tr_y, tr_r, va_X, va_y, va_r, names), per_symbol_cols = asyncio.run(
        _build_dataset(args.crypto_days, args.stride)
    )
    print(f"      train rows: {len(tr_X)}  val rows: {len(va_X)}  features: {len(names)}")
    for name, arr in (("train", tr_y), ("val", va_y)):
        u, c = np.unique(arr, return_counts=True)
        print(f"      {name} class counts: {dict(zip(u.tolist(), c.tolist()))}")

    # Purged chronological CV folds over the TRAIN split for hyperopt: each
    # trial is scored by mean accuracy across folds instead of one temporal
    # slice. embargo = HORIZON + 1 covers the label span in replay-step units
    # at any stride (see walk_forward.default_embargo's caveat).
    cv_folds = None
    if args.trials > 0:
        try:
            cv_folds = purged_chrono_folds(len(tr_X), CV_FOLDS, embargo=HORIZON + 1)
            print(f"      hyperopt CV: {CV_FOLDS} purged folds (embargo={HORIZON + 1})")
        except ValueError as exc:
            print(f"      purged CV unavailable ({exc}); hyperopt uses the single split")

    print(f"[2/6] Training {' + '.join(MODEL_TYPES)} (hyperopt trials={args.trials}) ...")
    from services.prediction.registry import ModelRegistry
    from services.prediction.training.trainer import ModelTrainer

    trainer = ModelTrainer()
    trained: dict[str, Any] = {}
    for model_type in MODEL_TYPES:
        result, model = trainer.train_model(
            model_type=model_type,
            X_train=tr_X, y_train=tr_y, X_val=va_X, y_val=va_y,
            hyperopt=args.trials > 0, n_trials=args.trials,
            feature_names=names,
            returns_train=tr_r, returns_val=va_r,
            cv_folds=cv_folds,
        )
        print(f"      {model_type}: val_accuracy={result.val_accuracy:.4f} "
              f"val_loss={result.val_loss:.4f} "
              f"train-val gap={result.train_accuracy - result.val_accuracy:.4f}")
        trained[model_type] = (result, model)

    print("[3/6] Out-of-sample gate (accuracy floor + overfit guard) ...")
    failures = gate_failures({t: res for t, (res, _) in trained.items()})
    if failures:
        for f in failures:
            print(f"      GATE FAILED: {f}")
        print("      -> NOT promoting")
        return 1
    print(f"      all models beat the {MIN_VAL_ACCURACY} floor (random = 0.333) "
          f"within the {MAX_TRAIN_VAL_GAP} overfit gap")

    print("[4/6] Registering + promoting in the serving registry ...")
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

    print("[5/6] Writing the per-symbol feature reference for drift detection ...")
    try:
        rows_by_symbol = per_symbol_val_rows(per_symbol_cols, va_X, args.stride)
        ref_path = write_feature_reference(Path(args.artifacts), names, rows_by_symbol)
        counts = {sym: len(rows) for sym, rows in rows_by_symbol.items()}
        print(f"      {ref_path}: {counts}")
    except ValueError as exc:
        # A misaligned reference would fire bogus drift alerts; absence merely
        # skips the feature-drift check (the service logs that once at info).
        print(f"      SKIPPED: {exc}")

    print("[6/6] Serving-path smoke check (ModelServer -> ensemble) ...")
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
