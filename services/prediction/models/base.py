"""Abstract base predictor and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PredictionOutput:
    """Result of a single prediction."""

    direction: str  # "long", "short", "flat"
    confidence: float  # 0-1
    expected_return: float  # predicted % return
    probabilities: dict[str, float] = field(default_factory=dict)  # class probabilities


@dataclass
class TrainResult:
    """Metrics returned after a training run."""

    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    epochs_trained: int
    feature_importance: dict[str, float] | None = None


class BasePredictor(ABC):
    """Interface that every prediction model must implement."""

    model_type: str = "base"
    required_features: list[str] = []

    @abstractmethod
    def predict(self, features: np.ndarray) -> PredictionOutput:
        """Predict direction and return for a single sample."""
        ...

    @abstractmethod
    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        """Predict direction and return for a batch of samples."""
        ...

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        """Train the model and return training metrics."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artefacts to *path*."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model artefacts from *path*."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return a mapping of feature name to importance score."""
        ...
