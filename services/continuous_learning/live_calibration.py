"""Live recalibration of the abstention gate's flat-probability.

The learning-metrics instrument (PR #73) measured the ensemble's stated
p(flat) against reality and found it badly over-stated: the top bin claims
~63% flat and delivers 27% -- a 36-point gap that WORSENS as stated
confidence rises. The training-time isotonic/Platt calibration was fitted on
historical folds and does not hold on live distributions.

This module closes that loop: an isotonic mapping fitted on the platform's
OWN resolved live outcomes -- stated p(flat) against realized
inside-the-deadband frequency -- applied to the ensemble's output at serve
time. The platform literally learns from its own recorded mistakes, which is
the registered reading of "improve the ML ability" (GO_LIVE.md 2026-08-09):
not bigger models, but learning from the signal that actually matters.

Design rules, all load-bearing:

* Fitted ONLY on gated (flattened) predictions, where the stored confidence
  IS p(flat) -- the semantics discovery from PR #59's post-mortem.
* De-overlapped pairs (minute % 5 == 0) so the fit sees independent
  observations, and a >= MIN_PAIRS floor below which the layer is the
  IDENTITY -- an uncalibrated gate is better than one calibrated on noise.
* Probabilities stay a simplex: the flat mass moves, long/short rescale
  proportionally, and the direction label is NEVER changed here -- votes and
  conformal gating already happened upstream. Widened margins downstream
  (a mechanically more active conviction gate) are a registered, expected
  behavioural consequence.
* Deterministic at boot: the mapping refits from stored predictions, so no
  new persistence and no state to corrupt.
* Kill switch: settings.live_calibration_enabled; identity when disabled.

Success criteria are registered in GO_LIVE.md (2026-08-09) and judged by the
learning-metrics endpoint after 7 days. A miss disables the layer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from sqlalchemy import extract, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.models.predictions import Prediction as PredictionRow

if TYPE_CHECKING:
    from services.prediction.models.base import PredictionOutput

logger = structlog.get_logger(__name__)

# Below this many resolved de-overlapped pairs the mapping is identity.
MIN_PAIRS = 2_000
# Fit window: recent enough to track the live regime, long enough to be stable.
FIT_WINDOW_DAYS = 14
REFIT_SECONDS = 6 * 3600.0
# Calibrated probabilities are clipped away from 0/1: a live-fitted mapping
# must never express certainty the data cannot contain.
_CLIP = (0.02, 0.98)


def fit_isotonic(stated: np.ndarray, realized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit stated p(flat) -> realized flat frequency, monotone by construction.

    Returns the (x, y) breakpoints of the fitted step function. Monotonicity
    is the point: the gate's ORDERING of confidence may be fine even when its
    LEVELS are wrong, and isotonic repairs levels without inventing ordering.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=_CLIP[0], y_max=_CLIP[1], out_of_bounds="clip")
    iso.fit(stated, realized)
    x = np.asarray(iso.X_thresholds_, dtype=float)
    y = np.asarray(iso.y_thresholds_, dtype=float)
    return x, y


def apply_mapping(p: float, x: np.ndarray, y: np.ndarray) -> float:
    """Evaluate the fitted mapping at *p* (linear interpolation, clipped)."""
    if x.size == 0:
        return p
    return float(np.clip(np.interp(p, x, y), _CLIP[0], _CLIP[1]))


class LiveCalibrator:
    """Holds the current mapping; identity until enough live pairs exist."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        enabled: bool = True,
    ) -> None:
        self._session_factory = session_factory
        self._enabled = enabled
        self._x: np.ndarray = np.array([])
        self._y: np.ndarray = np.array([])
        self._fitted_on = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def is_active(self) -> bool:
        return self._enabled and self._x.size > 0

    @property
    def fitted_on(self) -> int:
        return self._fitted_on

    async def start(self) -> None:
        """Fit once at boot, then refit on the interval."""
        if not self._enabled:
            logger.info("live_calibration_disabled")
            return
        await self.refit()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(REFIT_SECONDS)
                await self.refit()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("live_calibration_refit_failed")

    async def refit(self) -> int:
        """Refit from resolved, gated, de-overlapped predictions."""
        since = datetime.now(UTC) - timedelta(days=FIT_WINDOW_DAYS)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(PredictionRow.confidence, PredictionRow.actual_direction)
                    .where(
                        PredictionRow.actual_return.is_not(None),
                        PredictionRow.direction == "flat",
                        PredictionRow.predicted_at >= since,
                        extract("minute", PredictionRow.predicted_at) % 5 == 0,
                    )
                )
            ).all()
        if len(rows) < MIN_PAIRS:
            # Identity beats a mapping fitted on noise. Keep any previous fit.
            logger.info(
                "live_calibration_insufficient", pairs=len(rows), required=MIN_PAIRS
            )
            return 0
        stated = np.array([float(r[0] or 0.0) for r in rows])
        realized = np.array([1.0 if r[1] == "flat" else 0.0 for r in rows])
        x, y = await asyncio.to_thread(fit_isotonic, stated, realized)
        self._x, self._y = x, y
        self._fitted_on = len(rows)
        logger.info(
            "live_calibration_fitted",
            pairs=len(rows),
            window_days=FIT_WINDOW_DAYS,
            mean_stated=round(float(stated.mean()), 4),
            mean_realized=round(float(realized.mean()), 4),
        )
        return len(rows)

    def recalibrate(self, output: PredictionOutput) -> PredictionOutput:
        """Return a new output with live-calibrated probabilities.

        The flat mass is remapped; long/short rescale proportionally so the
        simplex holds; the DIRECTION IS NEVER CHANGED here -- agreement votes
        and conformal gating already decided it upstream. Confidence is the
        calibrated probability of the (unchanged) direction.
        """
        if not self.is_active:
            return output
        probs: dict[str, float] = dict(output.probabilities or {})
        p_flat = float(probs.get("flat", 0.0))
        p_long = float(probs.get("long", 0.0))
        p_short = float(probs.get("short", 0.0))
        if p_flat <= 0.0 or (p_long + p_short) <= 0.0:
            return output

        p_flat_cal = apply_mapping(p_flat, self._x, self._y)
        directional = 1.0 - p_flat_cal
        scale = directional / (p_long + p_short)
        new_probs = {
            "flat": p_flat_cal,
            "long": p_long * scale,
            "short": p_short * scale,
        }
        from services.prediction.models.base import PredictionOutput as Output

        metadata: dict[str, Any] = dict(output.metadata or {})
        metadata["live_calibrated"] = True
        metadata["p_flat_raw"] = round(p_flat, 6)
        return Output(
            direction=output.direction,
            confidence=float(new_probs.get(output.direction, p_flat_cal)),
            expected_return=output.expected_return,
            probabilities=new_probs,
            metadata=metadata,
        )
