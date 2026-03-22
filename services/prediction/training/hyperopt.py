"""Hyperparameter optimisation via Optuna for each model type."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from services.prediction.models.base import BasePredictor

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------
# Search space definitions per model type
# ------------------------------------------------------------------

def _xgboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lightgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


def _lstm_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def _transformer_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 6),
        "d_ff": trial.suggest_categorical("d_ff", [128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


_SEARCH_SPACES: dict[str, Any] = {
    "xgboost": _xgboost_space,
    "lightgbm": _lightgbm_space,
    "lstm": _lstm_space,
    "transformer": _transformer_space,
}


# ------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------


class HyperOptimizer:
    """Bayesian hyperparameter optimisation using Optuna."""

    def __init__(self, model_type: str, n_trials: int = 50) -> None:
        if model_type not in _SEARCH_SPACES:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(_SEARCH_SPACES)}")
        self.model_type = model_type
        self.n_trials = n_trials

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, Any]:
        """Run hyperparameter search and return the best parameter dict.

        The objective minimises validation loss returned by
        ``model.train(...).val_loss``.
        """
        space_fn = _SEARCH_SPACES[self.model_type]

        def objective(trial: optuna.Trial) -> float:
            params = space_fn(trial)
            model = self._create_model(params)
            try:
                result = model.train(X_train, y_train, X_val, y_val)
                return result.val_loss
            except Exception:
                logger.exception("Trial %d failed", trial.number)
                return float("inf")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        logger.info(
            "Hyperopt complete for %s  best_val_loss=%.6f  best_params=%s",
            self.model_type,
            study.best_value,
            study.best_params,
        )
        return dict(study.best_params)

    def _create_model(self, params: dict[str, Any]) -> BasePredictor:
        """Instantiate a predictor with the given hyperparameters."""
        if self.model_type == "xgboost":
            from services.prediction.models.xgboost_model import XGBoostPredictor
            return XGBoostPredictor(classifier_params=params, regressor_params=params)
        elif self.model_type == "lightgbm":
            from services.prediction.models.lightgbm_model import LightGBMPredictor
            return LightGBMPredictor(classifier_params=params, regressor_params=params)
        elif self.model_type == "lstm":
            from services.prediction.models.lstm_model import LSTMPredictor
            return LSTMPredictor(**params)
        elif self.model_type == "transformer":
            from services.prediction.models.transformer_model import TransformerPredictor
            return TransformerPredictor(**params)
        raise ValueError(f"Unsupported model_type: {self.model_type}")
