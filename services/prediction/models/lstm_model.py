"""PyTorch LSTM predictor for direction classification and return regression."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from services.prediction.models.base import BasePredictor, PredictionOutput, TrainResult

logger = logging.getLogger(__name__)

DIRECTION_MAP = {0: "short", 1: "flat", 2: "long"}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------


class LSTMNetwork(nn.Module):
    """Two-layer LSTM with dual heads for classification and regression."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Classification head
        self.fc_cls = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Regression head (expected return)
        self.fc_reg = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, sequence_length, num_features)

        Returns:
            cls_logits: (batch, num_classes)
            reg_output: (batch, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        last_hidden = self.layer_norm(self.dropout(last_hidden))

        cls_logits = self.fc_cls(last_hidden)
        reg_output = self.fc_reg(last_hidden)

        return cls_logits, reg_output


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------


class LSTMPredictor(BasePredictor):
    """LSTM-based predictor wrapping :class:`LSTMNetwork`."""

    model_type: str = "lstm"
    required_features: list[str] = []

    def __init__(
        self,
        num_features: int = 1,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-3,
    ) -> None:
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        self.device = _get_device()
        self._network: LSTMNetwork | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._is_fitted = False

    def _build_network(self) -> LSTMNetwork:
        net = LSTMNetwork(
            num_features=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return net.to(self.device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> PredictionOutput:
        """Predict from a sequence array of shape (seq_length, num_features)."""
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # add batch dim
        return self.predict_batch(features)[0]

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
        """Predict from a batch array of shape (batch, seq_length, num_features)."""
        assert self._network is not None, "Model must be trained or loaded before prediction"
        self._network.eval()

        features = self._normalize_input(features)
        x = torch.tensor(features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            cls_logits, reg_output = self._network(x)
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            returns = reg_output.squeeze(-1).cpu().numpy()

        outputs: list[PredictionOutput] = []
        for i in range(len(probs)):
            direction_idx = int(np.argmax(probs[i]))
            outputs.append(
                PredictionOutput(
                    direction=DIRECTION_MAP[direction_idx],
                    confidence=float(probs[i][direction_idx]),
                    expected_return=float(returns[i]),
                    probabilities={DIRECTION_MAP[j]: float(probs[i][j]) for j in range(3)},
                )
            )
        return outputs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainResult:
        """Train the LSTM.

        X: (n_samples, sequence_length, num_features)
        y: integer class labels (0/1/2)
        """
        self.num_features = X_train.shape[-1]
        self._compute_normalization(X_train)

        X_train_norm = self._normalize_input(X_train)
        X_val_norm = self._normalize_input(X_val)

        self._network = self._build_network()

        # Approximate returns from labels for regression target
        return_map = {0: -0.01, 1: 0.0, 2: 0.01}
        y_train_returns = np.array([return_map.get(int(y), 0.0) for y in y_train])
        y_val_returns = np.array([return_map.get(int(y), 0.0) for y in y_val])

        train_ds = TensorDataset(
            torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
            torch.tensor(y_train_returns, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_norm, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
            torch.tensor(y_val_returns, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        cls_criterion = nn.CrossEntropyLoss()
        reg_criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=self.max_epochs,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        epochs_trained = 0

        for epoch in range(self.max_epochs):
            # --- Train ---
            self._network.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for x_batch, y_cls, y_reg in train_loader:
                x_batch = x_batch.to(self.device)
                y_cls = y_cls.to(self.device)
                y_reg = y_reg.to(self.device)

                optimizer.zero_grad()
                cls_logits, reg_out = self._network(x_batch)
                loss_cls = cls_criterion(cls_logits, y_cls)
                loss_reg = reg_criterion(reg_out.squeeze(-1), y_reg)
                loss = loss_cls + 0.5 * loss_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss_sum += loss.item() * len(x_batch)
                preds = cls_logits.argmax(dim=1)
                train_correct += (preds == y_cls).sum().item()
                train_total += len(x_batch)

            avg_train_loss = train_loss_sum / train_total

            # --- Validate ---
            self._network.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for x_batch, y_cls, y_reg in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_cls = y_cls.to(self.device)
                    y_reg = y_reg.to(self.device)

                    cls_logits, reg_out = self._network(x_batch)
                    loss_cls = cls_criterion(cls_logits, y_cls)
                    loss_reg = reg_criterion(reg_out.squeeze(-1), y_reg)
                    loss = loss_cls + 0.5 * loss_reg

                    val_loss_sum += loss.item() * len(x_batch)
                    preds = cls_logits.argmax(dim=1)
                    val_correct += (preds == y_cls).sum().item()
                    val_total += len(x_batch)

            avg_val_loss = val_loss_sum / val_total
            epochs_trained = epoch + 1

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._network.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best weights
        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self.device)

        self._is_fitted = True

        # Final metrics
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            "LSTM training complete  epochs=%d  train_acc=%.4f  val_acc=%.4f",
            epochs_trained,
            train_acc,
            val_acc,
        )

        return TrainResult(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            epochs_trained=epochs_trained,
            feature_importance=None,
        )

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _compute_normalization(self, X: np.ndarray) -> None:
        """Compute mean/std across all samples and timesteps."""
        # X shape: (n, seq, features)
        flat = X.reshape(-1, X.shape[-1])
        self._feature_mean = flat.mean(axis=0)
        self._feature_std = flat.std(axis=0)
        self._feature_std[self._feature_std < 1e-8] = 1.0

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        if self._feature_mean is None or self._feature_std is None:
            return X
        return (X - self._feature_mean) / self._feature_std

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        assert self._network is not None
        torch.save(
            {
                "state_dict": self._network.state_dict(),
                "num_features": self.num_features,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "sequence_length": self.sequence_length,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
            },
            path / "lstm_model.pt",
        )
        logger.info("LSTM model saved to %s", path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path / "lstm_model.pt", map_location=self.device, weights_only=False)
        self.num_features = checkpoint["num_features"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"]

        self._network = self._build_network()
        self._network.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = True
        logger.info("LSTM model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance  (not natively available for LSTMs)
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        return {}
