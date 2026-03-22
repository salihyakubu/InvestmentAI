"""PyTorch Transformer predictor for direction classification and return regression."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
# Positional encoding
# ------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------


class TransformerNetwork(nn.Module):
    """Transformer encoder with dual classification/regression heads."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        max_seq_len: int = 120,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head
        self.fc_cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Regression head
        self.fc_reg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, sequence_length, num_features)

        Returns:
            cls_logits: (batch, num_classes)
            reg_output: (batch, 1)
        """
        # Project to d_model and add positional encoding
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq, d_model)

        # Use mean pooling over the sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.layer_norm(x)

        cls_logits = self.fc_cls(x)
        reg_output = self.fc_reg(x)

        return cls_logits, reg_output


# ------------------------------------------------------------------
# Predictor
# ------------------------------------------------------------------


class TransformerPredictor(BasePredictor):
    """Transformer-based predictor wrapping :class:`TransformerNetwork`."""

    model_type: str = "transformer"
    required_features: list[str] = []

    def __init__(
        self,
        num_features: int = 1,
        sequence_length: int = 120,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-3,
    ) -> None:
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate

        self.device = _get_device()
        self._network: TransformerNetwork | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._is_fitted = False

    def _build_network(self) -> TransformerNetwork:
        net = TransformerNetwork(
            num_features=self.num_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.sequence_length,
        )
        return net.to(self.device)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> PredictionOutput:
        if features.ndim == 2:
            features = features[np.newaxis, ...]
        return self.predict_batch(features)[0]

    def predict_batch(self, features: np.ndarray) -> list[PredictionOutput]:
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
        """Train the Transformer.

        X: (n_samples, sequence_length, num_features)
        y: integer class labels (0/1/2)
        """
        self.num_features = X_train.shape[-1]
        self._compute_normalization(X_train)

        X_train_norm = self._normalize_input(X_train)
        X_val_norm = self._normalize_input(X_val)

        self._network = self._build_network()

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
        optimizer = torch.optim.AdamW(self._network.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
        avg_train_loss = 0.0
        avg_val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

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

            avg_train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

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

            avg_val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
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

        if best_state is not None:
            self._network.load_state_dict(best_state)
            self._network.to(self.device)

        self._is_fitted = True

        logger.info(
            "Transformer training complete  epochs=%d  train_acc=%.4f  val_acc=%.4f",
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
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
                "d_ff": self.d_ff,
                "dropout": self.dropout,
                "sequence_length": self.sequence_length,
                "feature_mean": self._feature_mean,
                "feature_std": self._feature_std,
            },
            path / "transformer_model.pt",
        )
        logger.info("Transformer model saved to %s", path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path / "transformer_model.pt", map_location=self.device, weights_only=False)
        self.num_features = checkpoint["num_features"]
        self.d_model = checkpoint["d_model"]
        self.nhead = checkpoint["nhead"]
        self.num_encoder_layers = checkpoint["num_encoder_layers"]
        self.d_ff = checkpoint["d_ff"]
        self.dropout = checkpoint["dropout"]
        self.sequence_length = checkpoint["sequence_length"]
        self._feature_mean = checkpoint["feature_mean"]
        self._feature_std = checkpoint["feature_std"]

        self._network = self._build_network()
        self._network.load_state_dict(checkpoint["state_dict"])
        self._is_fitted = True
        logger.info("Transformer model loaded from %s", path)

    # ------------------------------------------------------------------
    # Feature importance  (not natively available for Transformers)
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        return {}
