"""Live p(flat) recalibration: fixes real miscalibration, refuses noise.

The layer sits in the serving path, so its failure modes are trading
behaviour. Pinned here: an over-confident gate is corrected toward observed
frequencies; too little data yields the identity; probabilities remain a
simplex; the direction label is never touched; and the boot-time fit works
from stored rows alone.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.pool import StaticPool

from core.models.base import AsyncBase
from core.models.predictions import Prediction as PredictionRow
from services.continuous_learning.live_calibration import (
    MIN_PAIRS,
    LiveCalibrator,
    apply_mapping,
    fit_isotonic,
)
from services.prediction.models.base import PredictionOutput


@compiles(JSONB, "sqlite")
def _jsonb_as_json(type_, compiler, **kw):  # noqa: ANN001, ANN003, ANN201, ARG001
    return "JSON"


def _overconfident_pairs(n: int = 6000, seed: int = 0):
    """The live pathology: stated p(flat) high, realized frequency low."""
    rng = np.random.default_rng(seed)
    stated = rng.uniform(0.3, 0.75, n)
    # True flat probability is much lower than stated at the top end
    # (realized 27% where stated 63%, as measured live).
    true_p = 0.55 - 0.45 * (stated - 0.3)
    realized = (rng.random(n) < true_p).astype(float)
    return stated, realized


# ---------------------------------------------------------------------------
# The mapping
# ---------------------------------------------------------------------------


def test_overconfidence_is_corrected_toward_observed_frequency() -> None:
    stated, realized = _overconfident_pairs()
    x, y = fit_isotonic(stated, realized)
    # Where the gate said 0.65 flat, reality was ~0.39: the mapping must
    # pull the stated value DOWN materially.
    assert apply_mapping(0.65, x, y) < 0.50
    # And it must not destroy the calibrated low end.
    assert abs(apply_mapping(0.32, x, y) - 0.54) < 0.10


def test_mapping_is_monotone_nondecreasing() -> None:
    """Isotonic repairs levels without inventing ordering: higher stated
    p(flat) can never map to lower calibrated p(flat)."""
    stated, realized = _overconfident_pairs(seed=1)
    x, y = fit_isotonic(stated, realized)
    grid = np.linspace(0.0, 1.0, 101)
    values = [apply_mapping(p, x, y) for p in grid]
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:], strict=False))


def test_mapping_never_expresses_certainty() -> None:
    stated, realized = _overconfident_pairs(seed=2)
    x, y = fit_isotonic(stated, realized)
    assert 0.02 <= apply_mapping(0.0, x, y)
    assert apply_mapping(1.0, x, y) <= 0.98


def test_a_calibrated_gate_is_left_nearly_alone() -> None:
    """If stated already matches realized, the mapping must be ~identity --
    the layer exists to fix miscalibration, not to add drift."""
    rng = np.random.default_rng(3)
    stated = rng.uniform(0.3, 0.7, 8000)
    realized = (rng.random(8000) < stated).astype(float)
    x, y = fit_isotonic(stated, realized)
    for p in (0.35, 0.5, 0.65):
        assert abs(apply_mapping(p, x, y) - p) < 0.05


# ---------------------------------------------------------------------------
# Recalibrate: the serving-path contract
# ---------------------------------------------------------------------------


def _fitted_calibrator() -> LiveCalibrator:
    calibrator = LiveCalibrator(session_factory=None, enabled=True)  # type: ignore[arg-type]
    stated, realized = _overconfident_pairs(seed=4)
    calibrator._x, calibrator._y = fit_isotonic(stated, realized)
    calibrator._fitted_on = stated.size
    return calibrator


def test_probabilities_stay_a_simplex_and_direction_is_untouched() -> None:
    calibrator = _fitted_calibrator()
    output = PredictionOutput(
        direction="flat",
        confidence=0.65,
        expected_return=0.001,
        probabilities={"long": 0.20, "short": 0.15, "flat": 0.65},
    )
    result = calibrator.recalibrate(output)
    assert result.direction == "flat"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    # The over-stated flat mass fell; directional mass grew proportionally.
    assert result.probabilities["flat"] < 0.50
    ratio = result.probabilities["long"] / result.probabilities["short"]
    assert ratio == pytest.approx(0.20 / 0.15)
    # Confidence tracks the (unchanged) direction's calibrated probability.
    assert result.confidence == pytest.approx(result.probabilities["flat"])
    assert result.metadata["live_calibrated"] is True
    assert result.metadata["p_flat_raw"] == pytest.approx(0.65)


def test_unfitted_or_disabled_calibrator_is_identity() -> None:
    output = PredictionOutput(
        direction="flat", confidence=0.65, expected_return=0.001,
        probabilities={"long": 0.2, "short": 0.15, "flat": 0.65},
    )
    unfitted = LiveCalibrator(session_factory=None, enabled=True)  # type: ignore[arg-type]
    assert unfitted.recalibrate(output) is output
    disabled = _fitted_calibrator()
    disabled._enabled = False
    assert disabled.recalibrate(output) is output


def test_degenerate_probabilities_pass_through() -> None:
    calibrator = _fitted_calibrator()
    no_directional = PredictionOutput(
        direction="flat", confidence=1.0, expected_return=0.0,
        probabilities={"long": 0.0, "short": 0.0, "flat": 1.0},
    )
    assert calibrator.recalibrate(no_directional) is no_directional


# ---------------------------------------------------------------------------
# Refit from stored rows (the boot path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refit_fits_from_stored_gated_predictions_only() -> None:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[PredictionRow.__table__]
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)

    stated, realized = _overconfident_pairs(n=MIN_PAIRS + 500, seed=5)
    # Pin the base to a 5-minute boundary: rows inherit base.minute mod 5,
    # and the refit's de-overlap filter keeps only minute % 5 == 0 -- an
    # unaligned wall clock would silently zero out every seeded pair.
    base = (datetime.now(UTC) - timedelta(days=7)).replace(
        minute=0, second=0, microsecond=0
    )
    async with factory() as session:
        for i, (p, flat) in enumerate(zip(stated, realized, strict=True)):
            when = base + timedelta(minutes=5 * i)  # minute%5==0 by construction
            session.add(
                PredictionRow(
                    symbol="BTC/USDT", model_id="ensemble:test", model_version=1,
                    direction="flat", confidence=float(p), expected_return=0.001,
                    horizon_minutes=5, predicted_at=when, created_at=when,
                    actual_return=0.001,
                    actual_direction="flat" if flat else "long",
                    resolved_at=when + timedelta(minutes=6),
                )
            )
        await session.commit()

    calibrator = LiveCalibrator(session_factory=factory, enabled=True)
    fitted = await calibrator.refit()
    assert fitted >= MIN_PAIRS
    assert calibrator.is_active
    assert apply_mapping(0.65, calibrator._x, calibrator._y) < 0.50
    await engine.dispose()


@pytest.mark.asyncio
async def test_too_few_pairs_leaves_identity() -> None:
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(
            AsyncBase.metadata.create_all, tables=[PredictionRow.__table__]
        )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    calibrator = LiveCalibrator(session_factory=factory, enabled=True)
    assert await calibrator.refit() == 0
    assert not calibrator.is_active
    await engine.dispose()
