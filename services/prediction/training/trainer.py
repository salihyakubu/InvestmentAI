"""Model trainer orchestration -- data loading, hyperopt, training, evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from services.prediction.models.base import BasePredictor, TrainResult
from services.prediction.models.lightgbm_model import LightGBMPredictor
from services.prediction.models.lstm_model import LSTMPredictor
from services.prediction.models.transformer_model import TransformerPredictor
from services.prediction.models.xgboost_model import XGBoostPredictor
from services.prediction.training.data_loader import TrainingDataLoader
from services.prediction.training.hyperopt import HyperOptimizer
from services.prediction.training.walk_forward import WalkForwardValidator

logger = logging.getLogger(__name__)

_MODEL_CLASSES: dict[str, type[BasePredictor]] = {
    "xgboost": XGBoostPredictor,
    "lightgbm": LightGBMPredictor,
    "lstm": LSTMPredictor,
    "transformer": TransformerPredictor,
}


class ModelTrainer:
    """Orchestrates end-to-end model training.

    Combines optional hyperparameter search, model training, and
    walk-forward evaluation into a single workflow.
    """

    def __init__(
        self,
        data_loader: TrainingDataLoader | None = None,
        walk_forward: WalkForwardValidator | None = None,
    ) -> None:
        self.data_loader = data_loader or TrainingDataLoader()
        self.walk_forward = walk_forward or WalkForwardValidator()

    # ------------------------------------------------------------------
    # Single model training
    # ------------------------------------------------------------------

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperopt: bool = True,
        n_trials: int = 50,
        extra_params: dict[str, Any] | None = None,
        feature_names: list[str] | None = None,
    ) -> tuple[TrainResult, BasePredictor]:
        """Train a single model, optionally preceded by hyperopt.

        Args:
            model_type: one of ``xgboost``, ``lightgbm``, ``lstm``, ``transformer``.
            X_train / y_train: training data.
            X_val / y_val: validation data.
            hyperopt: whether to run Bayesian hyperparameter optimisation first.
            n_trials: number of optuna trials if *hyperopt* is ``True``.
            extra_params: additional params merged into the model constructor.
            feature_names: optional feature name list (tree models only).

        Returns:
            ``(TrainResult, trained_model)``
        """
        if model_type not in _MODEL_CLASSES:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_MODEL_CLASSES)}")

        params: dict[str, Any] = {}

        # Optional hyperparameter search
        if hyperopt:
            logger.info("Running hyperparameter optimisation for %s (%d trials)", model_type, n_trials)
            optimizer = HyperOptimizer(model_type=model_type, n_trials=n_trials)
            best_params = optimizer.optimize(X_train, y_train, X_val, y_val)
            params.update(best_params)

        if extra_params:
            params.update(extra_params)

        # Instantiate model
        model = self._create_model(model_type, params, feature_names)

        # Train
        logger.info("Training %s model", model_type)
        result = model.train(X_train, y_train, X_val, y_val)

        return result, model

    # ------------------------------------------------------------------
    # Train all model types
    # ------------------------------------------------------------------

    def train_all_models(
        self,
        features: np.ndarray,
        close_prices: np.ndarray,
        feature_names: list[str] | None = None,
        hyperopt: bool = True,
        n_trials: int = 30,
        val_ratio: float = 0.2,
        seq_length_lstm: int = 60,
        seq_length_transformer: int = 120,
    ) -> dict[str, tuple[TrainResult, BasePredictor]]:
        """Train all four model types on the same underlying data.

        Returns:
            Mapping of model_type -> (TrainResult, trained model).
        """
        results: dict[str, tuple[TrainResult, BasePredictor]] = {}

        # --- Flat data for tree models ---
        flat_data = self.data_loader.load_training_data(features, close_prices, feature_names)
        split_idx = int(len(flat_data.X) * (1 - val_ratio))
        X_train_flat, X_val_flat = flat_data.X[:split_idx], flat_data.X[split_idx:]
        y_train_flat, y_val_flat = flat_data.y[:split_idx], flat_data.y[split_idx:]

        for model_type in ("xgboost", "lightgbm"):
            try:
                result, model = self.train_model(
                    model_type=model_type,
                    X_train=X_train_flat,
                    y_train=y_train_flat,
                    X_val=X_val_flat,
                    y_val=y_val_flat,
                    hyperopt=hyperopt,
                    n_trials=n_trials,
                    feature_names=flat_data.feature_names,
                )
                results[model_type] = (result, model)
            except Exception:
                logger.exception("Failed to train %s", model_type)

        # --- Sequence data for LSTM ---
        for model_type, seq_len in (("lstm", seq_length_lstm), ("transformer", seq_length_transformer)):
            try:
                seq_data = self.data_loader.load_sequence_data(
                    features, close_prices, seq_length=seq_len, feature_names=feature_names,
                )
                split_idx_seq = int(len(seq_data.X) * (1 - val_ratio))
                X_train_seq = seq_data.X[:split_idx_seq]
                y_train_seq = seq_data.y[:split_idx_seq]
                X_val_seq = seq_data.X[split_idx_seq:]
                y_val_seq = seq_data.y[split_idx_seq:]

                result, model = self.train_model(
                    model_type=model_type,
                    X_train=X_train_seq,
                    y_train=y_train_seq,
                    X_val=X_val_seq,
                    y_val=y_val_seq,
                    hyperopt=hyperopt,
                    n_trials=n_trials,
                    feature_names=seq_data.feature_names,
                )
                results[model_type] = (result, model)
            except Exception:
                logger.exception("Failed to train %s", model_type)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_model(
        model_type: str,
        params: dict[str, Any],
        feature_names: list[str] | None = None,
    ) -> BasePredictor:
        if model_type == "xgboost":
            return XGBoostPredictor(
                classifier_params=params,
                regressor_params=params,
                feature_names=feature_names,
            )
        elif model_type == "lightgbm":
            return LightGBMPredictor(
                classifier_params=params,
                regressor_params=params,
                feature_names=feature_names,
            )
        elif model_type == "lstm":
            return LSTMPredictor(**params)
        elif model_type == "transformer":
            return TransformerPredictor(**params)
        raise ValueError(f"Unsupported model_type: {model_type}")
