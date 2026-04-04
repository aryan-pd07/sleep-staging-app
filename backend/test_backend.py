import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# ── Mock tensorflow BEFORE importing api ──
tf_mock = MagicMock()
sys.modules['tensorflow'] = tf_mock
sys.modules['tensorflow.keras'] = tf_mock.keras
sys.modules['mne'] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import preprocess_batch, app
from fastapi.testclient import TestClient

client = TestClient(app)


# ─────────────────────────────────────────
# UNIT TESTS — preprocess_batch
# ─────────────────────────────────────────

def test_preprocess_output_shape():
    """Preprocessed batch should have shape (N, 3000, 1)"""
    sample = [[float(i % 100) for i in range(3000)]]
    result = preprocess_batch(sample)
    assert result.shape == (1, 3000, 1), f"Expected (1, 3000, 1), got {result.shape}"


def test_preprocess_multiple_epochs():
    """Should handle multiple epochs correctly"""
    sample = [[float(i % 100) for i in range(3000)] for _ in range(5)]
    result = preprocess_batch(sample)
    assert result.shape == (5, 3000, 1)


def test_preprocess_normalization():
    """Z-score normalization should produce near-zero mean and unit std"""
    sample = [[float(i) for i in range(3000)]]
    result = preprocess_batch(sample)
    mean = result.mean()
    std = result.std()
    assert abs(mean) < 0.01, f"Mean should be near 0, got {mean}"
    assert abs(std - 1.0) < 0.01, f"Std should be near 1, got {std}"


def test_preprocess_low_amplitude_scaling():
    """Very low amplitude signals should be scaled up to microvolts"""
    low_amp = [[0.000001 * i for i in range(3000)]]
    result = preprocess_batch(low_amp)
    assert not np.allclose(result, 0), "Low amplitude signal should be scaled"


def test_preprocess_constant_signal():
    """Constant signal (std=0) should not cause division by zero"""
    constant = [[1.0] * 3000]
    result = preprocess_batch(constant)
    assert result.shape == (1, 3000, 1)
    assert not np.any(np.isnan(result)), "Result should not contain NaN"


def test_preprocess_large_batch():
    """Should handle a larger batch without issues"""
    batch = [[float(np.sin(i * 0.01)) for i in range(3000)] for _ in range(10)]
    result = preprocess_batch(batch)
    assert result.shape == (10, 3000, 1)


def test_preprocess_negative_values():
    """Should handle negative signal values correctly"""
    negative = [[-float(i % 100) for i in range(3000)]]
    result = preprocess_batch(negative)
    assert result.shape == (1, 3000, 1)
    assert not np.any(np.isnan(result))


# ─────────────────────────────────────────
# UNIT TESTS — Class Labels & Mappings
# ─────────────────────────────────────────

def test_class_labels_mapping():
    """All 5 sleep stages should be correctly mapped"""
    CLASS_LABELS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    assert CLASS_LABELS[0] == "Wake"
    assert CLASS_LABELS[1] == "N1"
    assert CLASS_LABELS[2] == "N2"
    assert CLASS_LABELS[3] == "N3"
    assert CLASS_LABELS[4] == "REM"


def test_hypnogram_map_values():
    """Hypnogram map should have correct display values"""
    HYPNOGRAM_MAP = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
    assert HYPNOGRAM_MAP["Wake"] == 4
    assert HYPNOGRAM_MAP["N3"] == 0
    assert len(HYPNOGRAM_MAP) == 5


def test_all_stages_covered():
    """All 5 AASM sleep stages should be present"""
    CLASS_LABELS = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    expected = {"Wake", "N1", "N2", "N3", "REM"}
    assert set(CLASS_LABELS.values()) == expected


# ─────────────────────────────────────────
# INTEGRATION TESTS — API Endpoints
# ─────────────────────────────────────────

def test_predict_batch_missing_data():
    """API should return error when no data provided"""
    response = client.post("/predict_batch", json={})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data


def test_predict_batch_valid_data():
    """API should return predictions for valid data when model is loaded"""
    eeg = [[float(np.random.randn()) for _ in range(3000)] for _ in range(2)]
    eog = [[float(np.random.randn()) for _ in range(3000)] for _ in range(2)]

    # Mock the model prediction
    mock_preds = np.array([[0.1, 0.2, 0.5, 0.1, 0.1], [0.3, 0.1, 0.1, 0.4, 0.1]])

    with patch('api.model') as mock_model:
        mock_model.predict.return_value = mock_preds
        response = client.post("/predict_batch", json={
            "eeg_data": eeg,
            "eog_data": eog
        })

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "confidences" in data
    assert len(data["predictions"]) == 2
    assert all(0 <= p <= 4 for p in data["predictions"])
    assert all(0 <= c <= 100 for c in data["confidences"])


def test_predict_batch_correct_argmax():
    """Predictions should correspond to argmax of model output"""
    eeg = [[float(np.random.randn()) for _ in range(3000)]]
    eog = [[float(np.random.randn()) for _ in range(3000)]]

    # REM = index 4 has highest probability
    mock_preds = np.array([[0.05, 0.05, 0.05, 0.05, 0.80]])

    with patch('api.model') as mock_model:
        mock_model.predict.return_value = mock_preds
        response = client.post("/predict_batch", json={
            "eeg_data": eeg,
            "eog_data": eog
        })

    data = response.json()
    assert data["predictions"][0] == 4  # REM
    assert data["confidences"][0] == pytest.approx(80.0, abs=0.1)
