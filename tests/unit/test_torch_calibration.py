"""Temperature-scaling calibration for the torch sequence models.

The live confidence gate (``min_prediction_confidence``) only protects capital
if the reported confidence is a real probability. The tree models fit a Platt
calibrator on the validation split; the torch models must analogously fit a
softmax temperature in ``train()``, apply it in ``predict_batch()``, and
persist it through ``save()``/``load()``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from services.prediction.models.lstm_model import LSTMPredictor
from services.prediction.models.transformer_model import TransformerPredictor

torch = pytest.importorskip("torch")


def _make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny synthetic sequence dataset: (120, 10, 4) with a learnable signal."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 10, 4)).astype(np.float32)
    signal = X[:, :, 0].mean(axis=1)
    thresholds = np.quantile(signal, [1.0 / 3.0, 2.0 / 3.0])
    y = np.digitize(signal, thresholds).astype(np.int64)  # 0=short, 1=flat, 2=long
    x_tr, y_tr, x_val, y_val = X[:88], y[:88], X[88:], y[88:]
    assert len(y_val) >= 30 and len(np.unique(y_val)) >= 2  # calibration must not skip
    return x_tr, y_tr, x_val, y_val


def _tiny_lstm() -> LSTMPredictor:
    model = LSTMPredictor(
        num_features=4,
        sequence_length=10,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        batch_size=32,
        max_epochs=2,
        patience=5,
        learning_rate=1e-2,
    )
    model.device = torch.device("cpu")  # keep the test deterministic and fast
    return model


def _tiny_transformer() -> TransformerPredictor:
    model = TransformerPredictor(
        num_features=4,
        sequence_length=10,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        d_ff=16,
        dropout=0.0,
        batch_size=32,
        max_epochs=2,
        patience=5,
        learning_rate=1e-2,
    )
    model.device = torch.device("cpu")
    return model


@pytest.fixture(scope="module")
def trained_lstm() -> tuple[LSTMPredictor, np.ndarray]:
    torch.manual_seed(0)
    x_tr, y_tr, x_val, y_val = _make_dataset()
    model = _tiny_lstm()
    model.train(x_tr, y_tr, x_val, y_val)
    return model, x_val


@pytest.fixture(scope="module")
def trained_transformer() -> tuple[TransformerPredictor, np.ndarray]:
    torch.manual_seed(0)
    x_tr, y_tr, x_val, y_val = _make_dataset()
    model = _tiny_transformer()
    model.train(x_tr, y_tr, x_val, y_val)
    return model, x_val


def _assert_valid_probabilities(model: LSTMPredictor | TransformerPredictor, x: np.ndarray) -> None:
    outputs = model.predict_batch(x)
    assert len(outputs) == len(x)
    for out in outputs:
        probs = np.array([out.probabilities[k] for k in ("short", "flat", "long")])
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
        assert abs(float(probs.sum()) - 1.0) < 1e-5
        assert out.confidence == pytest.approx(float(probs.max()))
        assert out.direction in {"short", "flat", "long"}


def _assert_temperature_softens(
    model: LSTMPredictor | TransformerPredictor, x: np.ndarray
) -> None:
    """T=2.0 must yield strictly softer max probabilities than T=1.0 on the same logits."""
    original = model._temperature
    try:
        model._temperature = 1.0
        base = np.array([o.confidence for o in model.predict_batch(x)])
        model._temperature = 2.0
        soft = np.array([o.confidence for o in model.predict_batch(x)])
    finally:
        model._temperature = original

    assert base.max() > 1.0 / 3.0 + 1e-4  # logits are not degenerate-uniform
    idx = int(base.argmax())
    assert soft[idx] < base[idx]  # strictly softer where it matters most
    assert np.all(soft <= base + 1e-9)  # and never sharper anywhere


# ----------------------------------------------------------------------
# LSTM (full end-to-end: train -> calibrate -> predict -> persist)
# ----------------------------------------------------------------------


def test_lstm_train_fits_positive_temperature(trained_lstm: tuple[LSTMPredictor, np.ndarray]) -> None:
    model, _ = trained_lstm
    assert isinstance(model._temperature, float)
    assert math.isfinite(model._temperature)
    assert 0.05 <= model._temperature <= 10.0


def test_lstm_predict_batch_probabilities_are_valid(trained_lstm: tuple[LSTMPredictor, np.ndarray]) -> None:
    model, x_val = trained_lstm
    _assert_valid_probabilities(model, x_val)


def test_lstm_temperature_is_actually_applied(trained_lstm: tuple[LSTMPredictor, np.ndarray]) -> None:
    model, x_val = trained_lstm
    _assert_temperature_softens(model, x_val[:16])


def test_lstm_save_load_roundtrips_temperature(
    tmp_path: Path, trained_lstm: tuple[LSTMPredictor, np.ndarray]
) -> None:
    model, x_val = trained_lstm
    model.save(tmp_path)

    restored = LSTMPredictor()
    restored.device = torch.device("cpu")
    restored.load(tmp_path)
    assert restored._temperature == pytest.approx(model._temperature)

    before = np.array([o.confidence for o in model.predict_batch(x_val)])
    after = np.array([o.confidence for o in restored.predict_batch(x_val)])
    assert np.allclose(before, after, atol=1e-6)


def test_lstm_legacy_artifact_defaults_to_temperature_one(
    tmp_path: Path, trained_lstm: tuple[LSTMPredictor, np.ndarray]
) -> None:
    model, _ = trained_lstm
    model.save(tmp_path)

    # Simulate an artifact written before temperature scaling existed.
    checkpoint = torch.load(tmp_path / "lstm_model.pt", map_location="cpu", weights_only=False)
    checkpoint.pop("temperature")
    torch.save(checkpoint, tmp_path / "lstm_model.pt")

    restored = LSTMPredictor()
    restored.device = torch.device("cpu")
    restored.load(tmp_path)
    assert restored._temperature == 1.0


def test_lstm_skips_calibration_on_degenerate_validation(
    trained_lstm: tuple[LSTMPredictor, np.ndarray]
) -> None:
    model, x_val = trained_lstm
    # Too few validation rows -> keep T=1.0.
    assert model._fit_temperature(x_val[:10], np.zeros(10, dtype=np.int64)) == 1.0
    # Single-class validation split -> keep T=1.0.
    assert model._fit_temperature(x_val, np.zeros(len(x_val), dtype=np.int64)) == 1.0


# ----------------------------------------------------------------------
# Transformer (same contract, trained end-to-end on the same tiny data)
# ----------------------------------------------------------------------


def test_transformer_train_fits_positive_temperature(
    trained_transformer: tuple[TransformerPredictor, np.ndarray]
) -> None:
    model, _ = trained_transformer
    assert isinstance(model._temperature, float)
    assert math.isfinite(model._temperature)
    assert 0.05 <= model._temperature <= 10.0


def test_transformer_predict_batch_probabilities_are_valid(
    trained_transformer: tuple[TransformerPredictor, np.ndarray]
) -> None:
    model, x_val = trained_transformer
    _assert_valid_probabilities(model, x_val)


def test_transformer_temperature_is_actually_applied(
    trained_transformer: tuple[TransformerPredictor, np.ndarray]
) -> None:
    model, x_val = trained_transformer
    _assert_temperature_softens(model, x_val[:16])


def test_transformer_save_load_roundtrips_temperature(
    tmp_path: Path, trained_transformer: tuple[TransformerPredictor, np.ndarray]
) -> None:
    model, x_val = trained_transformer
    model.save(tmp_path)

    restored = TransformerPredictor()
    restored.device = torch.device("cpu")
    restored.load(tmp_path)
    assert restored._temperature == pytest.approx(model._temperature)

    before = np.array([o.confidence for o in model.predict_batch(x_val)])
    after = np.array([o.confidence for o in restored.predict_batch(x_val)])
    assert np.allclose(before, after, atol=1e-6)
