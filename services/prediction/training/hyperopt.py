"""Hyperparameter optimisation via Optuna for each model type.

Two objective modes:

- Single split (default): minimise ``model.train(...).val_loss`` on the given
  train/val split -- the original behaviour.
- Purged CV: when the caller passes ``folds`` (``(train_idx, val_idx)`` index
  pairs into the training arrays, produced by
  ``walk_forward.purged_chrono_folds``), each trial is scored by the MEAN
  validation accuracy across those purged, embargoed chronological folds and
  the study maximises it. This stops the search from overfitting its
  hyperparameters to a single temporal slice while keeping the outer
  validation split untouched for the champion/challenger gate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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


def _catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    # Symmetric (oblivious) trees: depth capped at 10 and iterations at 500 to
    # keep per-trial cost comparable to the xgboost space.
    return {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
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
    "catboost": _catboost_space,
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
        folds: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, Any]:
        """Run hyperparameter search and return the best parameter dict.

        Without *folds*, the objective minimises the validation loss returned
        by ``model.train(...).val_loss`` on the single given split. With
        *folds* -- ``(train_idx, val_idx)`` index pairs into *X_train* /
        *y_train* from ``walk_forward.purged_chrono_folds`` -- each trial
        trains one fresh model per fold and the study MAXIMISES the mean
        validation accuracy across folds; *X_val* / *y_val* are left untouched
        so the outer split stays a clean gate for the final fit.
        """
        space_fn = _SEARCH_SPACES[self.model_type]

        def objective_single(trial: optuna.Trial) -> float:
            params = space_fn(trial)
            model = self._create_model(params)
            try:
                result = model.train(X_train, y_train, X_val, y_val)
                return result.val_loss
            except Exception:
                logger.exception("Trial %d failed", trial.number)
                return float("inf")

        def objective_folds(trial: optuna.Trial) -> float:
            params = space_fn(trial)
            try:
                accuracies: list[float] = []
                for train_idx, val_idx in folds or []:
                    model = self._create_model(params)  # fresh model per fold
                    result = model.train(
                        X_train[train_idx], y_train[train_idx],
                        X_train[val_idx], y_train[val_idx],
                    )
                    accuracies.append(result.val_accuracy)
                return float(np.mean(accuracies))
            except Exception:
                logger.exception("Trial %d failed", trial.number)
                return 0.0  # worst possible accuracy under maximisation

        use_folds = bool(folds)
        study = optuna.create_study(
            direction="maximize" if use_folds else "minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(
            objective_folds if use_folds else objective_single,
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        logger.info(
            "Hyperopt complete for %s  %s=%.6f  best_params=%s",
            self.model_type,
            "best_mean_cv_accuracy" if use_folds else "best_val_loss",
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
        elif self.model_type == "catboost":
            from services.prediction.models.catboost_model import CatBoostPredictor
            return CatBoostPredictor(classifier_params=params, regressor_params=params)
        elif self.model_type == "lstm":
            from services.prediction.models.lstm_model import LSTMPredictor
            return LSTMPredictor(**params)
        elif self.model_type == "transformer":
            from services.prediction.models.transformer_model import TransformerPredictor
            return TransformerPredictor(**params)
        raise ValueError(f"Unsupported model_type: {self.model_type}")
