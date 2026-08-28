"""Phase 2 live-transfer gate: promotion earns its place on embargoed rows.

The gate's failure modes ARE promotion policy: replaying a challenger on
its own training rows (the review-caught inversion), a hash mismatch
silently scoring misaligned columns, a floor that lets a thin sample
decide, a tie promoting on equality, a poisoned outcome row blamed on a
model, or a degenerate model sneaking past would each put a worse model
into serving with the record saying otherwise. Every registered rule
(GO_LIVE 2026-08-28 + the same-day embargo amendment) is pinned here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core.models.base import AsyncBase
from core.models.predictions import Prediction as PredictionRow
from services.continuous_learning.live_replay import (
    EMBARGO_DAYS,
    MIN_ROWS,
    GateDecision,
    feature_ordering_hash,
    fetch_replay_rows,
    live_transfer_gate,
    prepare_replay,
    replay_ic,
)
from services.prediction.models.base import PredictionOutput


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


_NAMES = ["f_a", "f_b", "f_c"]
_HASH = feature_ordering_hash(_NAMES)
_NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
# The embargo boundary: the challenger's training data ends here; only rows
# strictly after it may judge the challenger.
_TRAIN_END = _NOW - timedelta(days=EMBARGO_DAYS)


def test_hash_is_the_phase_2a_formula() -> None:
    """The stored rows were stamped with sha256(",".join(names))[:16]; the
    gate must compute the identical digest or refuse its own data."""
    assert _HASH == hashlib.sha256(b"f_a,f_b,f_c").hexdigest()[:16]
    assert len(_HASH) == 16


@dataclass
class _StubModel:
    """predict_batch scores rows by a fixed linear rule."""

    weights: tuple[float, float, float]

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        scores = features @ np.asarray(self.weights)
        return [
            PredictionOutput(
                direction="long", confidence=0.5, expected_return=float(s),
                probabilities={"long": 0.4, "short": 0.3, "flat": 0.3},
            )
            for s in scores
        ]


async def _factory():
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[PredictionRow.__table__]
        )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed(
    factory,
    n: int = MIN_ROWS + 100,
    step_minutes: int = 10,  # 2,100 rows x 10min ~= 14.6d: fills the embargo window
    digest: str = _HASH,
    seed: int = 0,
    start: datetime | None = None,
) -> np.ndarray:
    """Post-embargo rows whose outcome is a known function of the vector:
    y = x0 - x2 (+noise)."""
    rng = np.random.default_rng(seed)
    start = start or (_TRAIN_END + timedelta(minutes=1))
    ys = []
    async with factory() as session:
        for i in range(n):
            vec = [float(v) for v in rng.normal(0, 1, 3)]
            y = vec[0] - vec[2] + float(rng.normal(0, 0.1))
            ys.append(y)
            when = start + timedelta(minutes=step_minutes * i)
            session.add(
                PredictionRow(
                    symbol="BTC/USDT", model_id="ensemble:test", model_version=1,
                    direction="flat", confidence=0.4, expected_return=0.0,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    actual_return=y,
                    features_used={"v": vec, "h": digest},
                )
            )
        await session.commit()
    return np.asarray(ys)


# ---------------------------------------------------------------------------
# Row fetch: THE EMBARGO, hash scoping, hygiene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rows_at_or_before_the_training_boundary_never_enter() -> None:
    """THE inversion regression (review 2026-08-28): rows from the
    challenger's own training period would score it in-sample -- boosted
    trees memorize -- and every challenger would 'beat' every champion.
    The boundary is strict: at-or-before is excluded."""
    engine, factory = await _factory()
    await _seed(factory, n=40)  # legitimate post-embargo rows
    async with factory() as session:
        for offset_minutes in (0, 1, 60, 60 * 24 * 10):  # at + before boundary
            when = _TRAIN_END - timedelta(minutes=offset_minutes)
            session.add(
                PredictionRow(
                    symbol="BTC/USDT", model_id="m", model_version=1,
                    direction="flat", confidence=0.4, expected_return=0.0,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    actual_return=0.5,
                    features_used={"v": [1.0, 2.0, 3.0], "h": _HASH},
                )
            )
        await session.commit()
    X, _y, _span = await fetch_replay_rows(factory, _HASH, _TRAIN_END)
    assert X.shape == (40, 3)  # only the post-boundary rows
    await engine.dispose()


@pytest.mark.asyncio
async def test_only_matching_hash_resolved_rows_enter() -> None:
    engine, factory = await _factory()
    await _seed(factory, n=50, digest=_HASH)
    await _seed(factory, n=50, digest="0123456789abcdef", seed=1)  # other gen
    async with factory() as session:  # unresolved row: no outcome yet
        session.add(
            PredictionRow(
                symbol="BTC/USDT", model_id="m", model_version=1,
                direction="flat", confidence=0.4, expected_return=0.0,
                horizon_minutes=5, predicted_at=_NOW, created_at=_NOW,
                features_used={"v": [1.0, 2.0, 3.0], "h": _HASH},
            )
        )
        await session.commit()
    X, y, _span = await fetch_replay_rows(factory, _HASH, _TRAIN_END)
    assert X.shape == (50, 3)
    assert y.shape == (50,)
    await engine.dispose()


@pytest.mark.asyncio
async def test_wrong_width_and_non_finite_rows_are_dropped() -> None:
    """A poisoned row (NaN outcome or NaN feature) must never decide a
    promotion; a wrong-width vector cannot be the same generation."""
    engine, factory = await _factory()
    await _seed(factory, n=50)
    when = _TRAIN_END + timedelta(minutes=30)
    async with factory() as session:
        for vec, actual in (
            ([1.0, 2.0], 0.5),                        # wrong width
            ([1.0, float("nan"), 3.0], 0.5),          # NaN feature
            ([1.0, 2.0, 3.0], float("nan")),          # NaN outcome
        ):
            session.add(
                PredictionRow(
                    symbol="BTC/USDT", model_id="m", model_version=1,
                    direction="flat", confidence=0.4, expected_return=0.0,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    actual_return=actual,
                    features_used={"v": vec, "h": _HASH},
                )
            )
        await session.commit()
    X, y, _span = await fetch_replay_rows(factory, _HASH, _TRAIN_END)
    assert X.shape == (50, 3)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    await engine.dispose()


# ---------------------------------------------------------------------------
# prepare_replay: boundary and floor refusals, named outcome degeneracy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_training_boundary_refuses() -> None:
    engine, factory = await _factory()
    await _seed(factory)
    prepared = await prepare_replay(factory, _NAMES, None, _NOW)
    assert isinstance(prepared, GateDecision)
    assert not prepared.promote
    assert prepared.reason == "unknown_training_boundary"
    await engine.dispose()


@pytest.mark.asyncio
async def test_stale_training_boundary_refuses() -> None:
    engine, factory = await _factory()
    await _seed(factory)
    ancient = _NOW - timedelta(days=45)
    prepared = await prepare_replay(factory, _NAMES, ancient, _NOW)
    assert isinstance(prepared, GateDecision)
    assert prepared.reason == "stale_training_boundary"
    await engine.dispose()


@pytest.mark.asyncio
async def test_insufficient_rows_refuse_before_any_model_is_scored() -> None:
    engine, factory = await _factory()
    await _seed(factory, n=300)
    prepared = await prepare_replay(factory, _NAMES, _TRAIN_END, _NOW)
    assert isinstance(prepared, GateDecision)
    assert prepared.reason == "insufficient_live_rows"
    await engine.dispose()


@pytest.mark.asyncio
async def test_degenerate_outcomes_are_named_not_blamed_on_a_model() -> None:
    """Constant stored outcomes are a property of the outcome pipeline; the
    refusal must say so instead of pointing diagnosis at either regressor."""
    engine, factory = await _factory()
    start = _TRAIN_END + timedelta(minutes=1)
    async with factory() as session:
        for i in range(MIN_ROWS + 100):
            when = start + timedelta(minutes=10 * i)
            session.add(
                PredictionRow(
                    symbol="BTC/USDT", model_id="m", model_version=1,
                    direction="flat", confidence=0.4, expected_return=0.0,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    actual_return=0.0,  # every outcome identical
                    features_used={"v": [float(i % 7), 1.0, 2.0], "h": _HASH},
                )
            )
        await session.commit()
    prepared = await prepare_replay(factory, _NAMES, _TRAIN_END, _NOW)
    assert isinstance(prepared, GateDecision)
    assert prepared.reason == "degenerate_outcomes"
    await engine.dispose()


@pytest.mark.asyncio
async def test_hash_mismatch_refuses_loudly() -> None:
    engine, factory = await _factory()
    await _seed(factory)  # stored generation: _HASH
    other_names = ["f_a", "f_b", "f_NEW"]
    prepared = await prepare_replay(factory, other_names, _TRAIN_END, _NOW)
    assert isinstance(prepared, GateDecision)
    assert not prepared.promote
    assert feature_ordering_hash(other_names) in prepared.reason
    await engine.dispose()


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_better_challenger_is_promoted_with_both_ics_recorded() -> None:
    engine, factory = await _factory()
    await _seed(factory)
    challenger = _StubModel((1.0, 0.0, -1.0))   # the true mapping
    champion = _StubModel((0.0, 1.0, 0.0))      # scores on noise
    decision = await live_transfer_gate(
        factory, challenger, champion, _NAMES, _TRAIN_END, _NOW
    )
    assert decision.promote
    assert decision.challenger_ic > 0.9
    assert abs(decision.champion_ic) < 0.2
    assert decision.n_rows >= MIN_ROWS
    await engine.dispose()


@pytest.mark.asyncio
async def test_worse_challenger_is_refused() -> None:
    engine, factory = await _factory()
    await _seed(factory)
    decision = await live_transfer_gate(
        factory,
        challenger=_StubModel((0.0, 1.0, 0.0)),
        champion=_StubModel((1.0, 0.0, -1.0)),
        feature_names=_NAMES,
        training_end=_TRAIN_END,
        now=_NOW,
    )
    assert not decision.promote
    assert decision.reason == "challenger_does_not_beat_champion_on_live_rows"
    await engine.dispose()


@pytest.mark.asyncio
async def test_a_tie_does_not_promote() -> None:
    """The criterion is STRICT: an identical challenger must not replace the
    champion -- churn without improvement is how eras degrade."""
    engine, factory = await _factory()
    await _seed(factory)
    same = _StubModel((1.0, 0.0, -1.0))
    decision = await live_transfer_gate(factory, same, same, _NAMES, _TRAIN_END, _NOW)
    assert not decision.promote
    await engine.dispose()


@pytest.mark.asyncio
async def test_no_champion_passes_vacuously() -> None:
    engine, factory = await _factory()
    await _seed(factory)
    decision = await live_transfer_gate(
        factory, _StubModel((1, 0, -1)), None, _NAMES, _TRAIN_END, _NOW
    )
    assert decision.promote
    assert decision.reason == "no_champion_vacuous_pass"
    await engine.dispose()


@pytest.mark.asyncio
async def test_degenerate_challenger_cannot_pass() -> None:
    """Constant scores have no ordering: fabricating IC 0.0 would let a
    broken regressor tie into a promotion path. It must refuse."""
    engine, factory = await _factory()
    await _seed(factory)
    decision = await live_transfer_gate(
        factory, _StubModel((0.0, 0.0, 0.0)), _StubModel((0, 1, 0)),
        _NAMES, _TRAIN_END, _NOW,
    )
    assert not decision.promote
    assert decision.reason == "degenerate_challenger_scores"
    await engine.dispose()


def test_replay_ic_recovers_a_planted_relationship() -> None:
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (500, 3))
    y = X[:, 0] - X[:, 2]
    assert replay_ic(_StubModel((1.0, 0.0, -1.0)), X, y) > 0.99
    assert replay_ic(_StubModel((-1.0, 0.0, 1.0)), X, y) < -0.99
    assert replay_ic(_StubModel((0.0, 0.0, 0.0)), X, y) is None
