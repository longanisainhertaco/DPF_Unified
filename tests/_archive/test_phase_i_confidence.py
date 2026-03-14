"""Phase I: Tests for ensemble prediction and confidence estimation.

Tests cover:
- PredictionWithConfidence dataclass defaults and custom values
- EnsemblePredictor initialization (with PyTorch requirement)
- Ensemble prediction with mean/std computation
- Confidence scoring based on ensemble agreement
- Out-of-distribution detection
- Confidence threshold checking
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from dpf.ai.confidence import EnsemblePredictor, PredictionWithConfidence


def _make_state(shape=(8, 8, 8), noise=0.0):
    """Create a synthetic DPF state dict with optional noise."""
    return {
        "rho": np.ones(shape) + np.random.normal(0, noise, shape),
        "velocity": np.zeros((3, *shape)) + np.random.normal(0, noise, (3, *shape)),
        "pressure": np.ones(shape) * 100.0 + np.random.normal(0, noise, shape),
        "B": np.zeros((3, *shape)),
        "Te": np.ones(shape) * 1e4 + np.random.normal(0, noise, shape),
        "Ti": np.ones(shape) * 1e4 + np.random.normal(0, noise, shape),
        "psi": np.zeros(shape),
    }


class MockModel:
    """Mock DPFSurrogate model for testing without PyTorch."""

    def __init__(self, noise=0.0):
        self.history_length = 4
        self.is_loaded = True
        self.noise = noise

    def predict_next_step(self, history):
        """Return last state with optional noise."""
        last = history[-1]
        return {k: v.copy() + np.random.normal(0, self.noise, v.shape) for k, v in last.items()}


@pytest.fixture
def ensemble():
    """Create EnsemblePredictor with mocked internals, bypassing __init__."""
    predictor = object.__new__(EnsemblePredictor)
    predictor.checkpoint_paths = []
    predictor.device = "cpu"
    predictor.history_length = 4
    predictor.confidence_threshold = 0.8
    predictor._models = [MockModel(noise=0.01), MockModel(noise=0.01), MockModel(noise=0.01)]
    return predictor


# ── PredictionWithConfidence Tests ──────────────────────────────────


def test_prediction_with_confidence_defaults():
    """PredictionWithConfidence initializes with correct default values."""
    pred = PredictionWithConfidence()
    assert pred.mean_state == {}
    assert pred.std_state == {}
    assert pred.confidence == pytest.approx(1.0)
    assert pred.ood_score == pytest.approx(0.0)
    assert pred.n_models == 1


def test_prediction_with_confidence_custom_values():
    """PredictionWithConfidence stores custom values correctly."""
    mean_state = {"rho": np.ones((8, 8, 8))}
    std_state = {"rho": np.zeros((8, 8, 8))}
    pred = PredictionWithConfidence(
        mean_state=mean_state,
        std_state=std_state,
        confidence=0.95,
        ood_score=0.1,
        n_models=5,
    )
    assert pred.mean_state == mean_state
    assert pred.std_state == std_state
    assert pred.confidence == pytest.approx(0.95)
    assert pred.ood_score == pytest.approx(0.1)
    assert pred.n_models == 5


# ── EnsemblePredictor Initialization Tests ──────────────────────────


def test_ensemble_predictor_raises_importerror_without_torch():
    """EnsemblePredictor raises ImportError when HAS_TORCH is False."""
    with (
        patch("dpf.ai.confidence.HAS_TORCH", False),
        pytest.raises(ImportError, match="PyTorch required"),
    ):
        EnsemblePredictor(checkpoint_paths=["dummy.pt"])


def test_ensemble_predictor_n_models_property(ensemble):
    """n_models property returns correct count of ensemble members."""
    assert ensemble.n_models == 3


# ── Ensemble Prediction Tests ───────────────────────────────────────


def test_predict_returns_prediction_with_confidence(ensemble):
    """predict returns PredictionWithConfidence instance."""
    history = [_make_state() for _ in range(4)]
    result = ensemble.predict(history)
    assert isinstance(result, PredictionWithConfidence)


def test_predict_mean_state_has_correct_keys(ensemble):
    """predict mean_state contains all expected DPF state keys."""
    history = [_make_state() for _ in range(4)]
    result = ensemble.predict(history)
    expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
    assert set(result.mean_state.keys()) == expected_keys


def test_predict_std_state_has_correct_keys(ensemble):
    """predict std_state contains all expected DPF state keys."""
    history = [_make_state() for _ in range(4)]
    result = ensemble.predict(history)
    expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
    assert set(result.std_state.keys()) == expected_keys


def test_predict_mean_state_values_are_mean_of_predictions(ensemble):
    """predict mean_state values are the mean of individual model predictions."""
    np.random.seed(42)  # For reproducibility

    # Create ensemble with predictable noise
    predictor = object.__new__(EnsemblePredictor)
    predictor.checkpoint_paths = []
    predictor.device = "cpu"
    predictor.history_length = 4
    predictor.confidence_threshold = 0.8
    predictor._models = [MockModel(noise=0.0), MockModel(noise=0.0), MockModel(noise=0.0)]

    history = [_make_state(noise=0.0) for _ in range(4)]
    result = predictor.predict(history)

    # With zero noise, all predictions should be identical to last state
    last_state = history[-1]
    for key in last_state:
        np.testing.assert_allclose(result.mean_state[key], last_state[key], rtol=1e-6)


def test_predict_confidence_in_valid_range(ensemble):
    """predict confidence is in [0, 1] range."""
    history = [_make_state() for _ in range(4)]
    result = ensemble.predict(history)
    assert 0.0 <= result.confidence <= 1.0


def test_predict_n_models_matches_ensemble_size(ensemble):
    """predict result contains correct n_models count."""
    history = [_make_state() for _ in range(4)]
    result = ensemble.predict(history)
    assert result.n_models == 3


# ── Confidence Threshold Tests ──────────────────────────────────────


def test_is_confident_returns_true_when_above_threshold(ensemble):
    """is_confident returns True when confidence >= threshold."""
    pred = PredictionWithConfidence(confidence=0.85)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is True


def test_is_confident_returns_false_when_below_threshold(ensemble):
    """is_confident returns False when confidence < threshold."""
    pred = PredictionWithConfidence(confidence=0.75)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is False


def test_is_confident_returns_true_at_exact_threshold(ensemble):
    """is_confident returns True when confidence equals threshold."""
    pred = PredictionWithConfidence(confidence=0.8)
    ensemble.confidence_threshold = 0.8
    assert ensemble.is_confident(pred) is True


# ── Out-of-Distribution Detection Tests ─────────────────────────────


def test_ood_detection_returns_zero_without_training_stats(ensemble):
    """ood_detection returns 0.0 when no training_stats provided."""
    state = _make_state()
    ood_score = ensemble.ood_detection(state, training_stats=None)
    assert ood_score == pytest.approx(0.0)


def test_ood_detection_returns_float_score_with_training_stats(ensemble):
    """ood_detection returns float score when training_stats provided."""
    state = _make_state()
    training_stats = {
        "rho": {"mean": 1.0, "std": 0.1},
        "Te": {"mean": 1e4, "std": 1e3},
        "Ti": {"mean": 1e4, "std": 1e3},
        "pressure": {"mean": 100.0, "std": 10.0},
        "psi": {"mean": 0.0, "std": 1.0},
    }
    ood_score = ensemble.ood_detection(state, training_stats=training_stats)
    assert isinstance(ood_score, float)
    assert ood_score >= 0.0


def test_ood_detection_score_increases_for_out_of_distribution_states(ensemble):
    """ood_detection score increases for states far from training distribution."""
    # In-distribution state
    in_dist_state = _make_state(noise=0.0)
    training_stats = {
        "rho": {"mean": 1.0, "std": 0.1},
        "Te": {"mean": 1e4, "std": 1e3},
        "Ti": {"mean": 1e4, "std": 1e3},
        "pressure": {"mean": 100.0, "std": 10.0},
        "psi": {"mean": 0.0, "std": 1.0},
    }
    in_dist_score = ensemble.ood_detection(in_dist_state, training_stats=training_stats)

    # Out-of-distribution state (10x higher values)
    ood_state = {
        "rho": np.ones((8, 8, 8)) * 10.0,
        "velocity": np.zeros((3, 8, 8, 8)),
        "pressure": np.ones((8, 8, 8)) * 1000.0,
        "B": np.zeros((3, 8, 8, 8)),
        "Te": np.ones((8, 8, 8)) * 1e5,
        "Ti": np.ones((8, 8, 8)) * 1e5,
        "psi": np.zeros((8, 8, 8)),
    }
    ood_score = ensemble.ood_detection(ood_state, training_stats=training_stats)

    # OOD score should be significantly higher
    assert ood_score > in_dist_score
    assert ood_score > 1.0  # Should be well above in-distribution


# ── Confidence Computation Tests ────────────────────────────────────


def test_compute_confidence_returns_one_for_zero_std():
    """_compute_confidence returns 1.0 for zero std (perfect agreement)."""
    predictor = object.__new__(EnsemblePredictor)
    std_state = {
        "rho": np.zeros((8, 8, 8)),
        "Te": np.zeros((8, 8, 8)),
        "Ti": np.zeros((8, 8, 8)),
        "pressure": np.zeros((8, 8, 8)),
    }
    confidence = predictor._compute_confidence(std_state)
    assert confidence == pytest.approx(1.0)


def test_compute_confidence_decreases_with_higher_std():
    """_compute_confidence returns lower values for higher std (disagreement)."""
    predictor = object.__new__(EnsemblePredictor)

    # Low std case
    low_std_state = {
        "rho": np.ones((8, 8, 8)) * 0.1,
        "Te": np.ones((8, 8, 8)) * 0.1,
    }
    low_confidence = predictor._compute_confidence(low_std_state)

    # High std case
    high_std_state = {
        "rho": np.ones((8, 8, 8)) * 10.0,
        "Te": np.ones((8, 8, 8)) * 10.0,
    }
    high_confidence = predictor._compute_confidence(high_std_state)

    # Higher std should give lower confidence
    assert high_confidence < low_confidence
    assert 0.0 <= high_confidence <= 1.0
    assert 0.0 <= low_confidence <= 1.0


def test_compute_confidence_returns_zero_for_empty_std_state():
    """_compute_confidence returns 0.0 for empty std_state dict."""
    predictor = object.__new__(EnsemblePredictor)
    std_state = {}
    confidence = predictor._compute_confidence(std_state)
    assert confidence == pytest.approx(0.0)
