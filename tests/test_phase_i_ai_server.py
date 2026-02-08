"""Phase I: AI server endpoint tests.

Tests FastAPI REST endpoints and utility functions for WALRUS surrogate inference.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402, I001


# ── Mock Surrogate and Ensemble ─────────────────────────────────


class MockSurrogate:
    """Mock WALRUS surrogate for testing."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True
        self.device = "cpu"

    def predict_next_step(self, history):
        """Return copy of last state."""
        last = history[-1]
        return {k: v.copy() for k, v in last.items()}

    def rollout(self, initial_states, n_steps):
        """Return n_steps copies of last state."""
        last = initial_states[-1]
        return [{k: v.copy() for k, v in last.items()} for _ in range(n_steps)]


class MockEnsemble:
    """Mock ensemble predictor for testing."""

    def __init__(self):
        self.n_models = 5
        self.device = "cpu"

    def predict(self, history):
        """Return PredictionWithConfidence dataclass."""
        from dpf.ai.confidence import PredictionWithConfidence

        last = history[-1]
        mean = {k: v.copy() for k, v in last.items()}
        std = {k: np.ones_like(v) * 0.1 for k, v in last.items()}
        return PredictionWithConfidence(
            mean_state=mean,
            std_state=std,
            confidence=0.95,
            ood_score=0.05,
            n_models=5,
        )


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def client():
    """TestClient with no models loaded."""
    from dpf.server.app import app

    return TestClient(app)


@pytest.fixture
def client_with_surrogate():
    """TestClient with mock surrogate loaded."""
    from dpf.ai import realtime_server
    from dpf.server.app import app

    mock = MockSurrogate()
    original = realtime_server._surrogate
    realtime_server._surrogate = mock

    client = TestClient(app)
    yield client

    realtime_server._surrogate = original


@pytest.fixture
def client_with_ensemble():
    """TestClient with mock ensemble loaded."""
    from dpf.ai import realtime_server
    from dpf.server.app import app

    mock = MockEnsemble()
    original_surrogate = realtime_server._surrogate
    original_ensemble = realtime_server._ensemble
    realtime_server._surrogate = MockSurrogate()  # Some endpoints need both
    realtime_server._ensemble = mock

    client = TestClient(app)
    yield client

    realtime_server._surrogate = original_surrogate
    realtime_server._ensemble = original_ensemble


@pytest.fixture
def sample_state():
    """Single state dict for testing."""
    return {
        "rho": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        "Te": np.array([[[100.0, 200.0], [300.0, 400.0]]]),
        "velocity": np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    }


@pytest.fixture
def sample_history(sample_state):
    """History of 4 states."""
    return [sample_state.copy() for _ in range(4)]


# ── Utility Function Tests ──────────────────────────────────────


def test_arrays_to_lists_converts_numpy_arrays():
    """_arrays_to_lists converts numpy arrays to nested lists."""
    from dpf.ai.realtime_server import _arrays_to_lists

    state = {"rho": np.array([[1.0, 2.0], [3.0, 4.0]])}
    result = _arrays_to_lists(state)

    assert isinstance(result["rho"], list)
    assert result["rho"] == [[1.0, 2.0], [3.0, 4.0]]


def test_arrays_to_lists_handles_nested_dicts():
    """_arrays_to_lists handles nested dictionaries."""
    from dpf.ai.realtime_server import _arrays_to_lists

    state = {"outer": {"inner": np.array([1.0, 2.0])}}
    result = _arrays_to_lists(state)

    assert isinstance(result["outer"]["inner"], list)
    assert result["outer"]["inner"] == [1.0, 2.0]


def test_arrays_to_lists_passes_through_non_array_values():
    """_arrays_to_lists preserves scalars and strings."""
    from dpf.ai.realtime_server import _arrays_to_lists

    state = {"scalar": 42, "string": "test", "none": None}
    result = _arrays_to_lists(state)

    assert result["scalar"] == 42
    assert result["string"] == "test"
    assert result["none"] is None


def test_arrays_to_lists_handles_lists_of_dicts():
    """_arrays_to_lists recursively processes lists of dicts."""
    from dpf.ai.realtime_server import _arrays_to_lists

    state = {"items": [{"data": np.array([1.0])}, {"data": np.array([2.0])}]}
    result = _arrays_to_lists(state)

    assert result["items"][0]["data"] == [1.0]
    assert result["items"][1]["data"] == [2.0]


def test_lists_to_arrays_converts_numeric_lists():
    """_lists_to_arrays converts numeric lists to numpy arrays."""
    from dpf.ai.realtime_server import _lists_to_arrays

    state = {"rho": [[1.0, 2.0], [3.0, 4.0]]}
    result = _lists_to_arrays(state)

    assert isinstance(result["rho"], np.ndarray)
    np.testing.assert_array_equal(result["rho"], np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_lists_to_arrays_handles_nested_dicts():
    """_lists_to_arrays handles nested dictionaries."""
    from dpf.ai.realtime_server import _lists_to_arrays

    state = {"outer": {"inner": [1.0, 2.0]}}
    result = _lists_to_arrays(state)

    assert isinstance(result["outer"]["inner"], np.ndarray)
    np.testing.assert_array_equal(result["outer"]["inner"], np.array([1.0, 2.0]))


def test_lists_to_arrays_handles_non_numeric_lists():
    """_lists_to_arrays converts string lists to numpy arrays (np.array succeeds on strings)."""
    from dpf.ai.realtime_server import _lists_to_arrays

    state = {"strings": ["a", "b", "c"]}
    result = _lists_to_arrays(state)

    # np.array(["a", "b", "c"]) succeeds, producing a string array
    assert isinstance(result["strings"], np.ndarray)
    np.testing.assert_array_equal(result["strings"], np.array(["a", "b", "c"]))


def test_lists_to_arrays_handles_lists_of_dicts():
    """_lists_to_arrays converts list of dicts via np.array (produces object array)."""
    from dpf.ai.realtime_server import _lists_to_arrays

    # np.array([{"data": [1.0]}, {"data": [2.0]}]) succeeds (creates object array)
    # so the function will convert the list to a numpy array, not recurse into dicts
    state = {"items": [{"data": [1.0]}, {"data": [2.0]}]}
    result = _lists_to_arrays(state)

    # The result is a numpy object array containing the dicts
    assert isinstance(result["items"], np.ndarray)


# ── REST Endpoint Tests ─────────────────────────────────────────


def test_status_returns_correct_structure_when_no_model_loaded(client):
    """GET /api/ai/status returns correct structure when no model loaded."""
    response = client.get("/api/ai/status")

    assert response.status_code == 200
    data = response.json()
    assert "torch_available" in data
    assert data["model_loaded"] is False
    assert data["device"] == "none"
    assert data["ensemble_size"] == 0


def test_status_returns_model_loaded_true_when_surrogate_set(client_with_surrogate):
    """GET /api/ai/status returns model_loaded=True when surrogate set."""
    response = client_with_surrogate.get("/api/ai/status")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"


def test_predict_returns_503_when_no_model_loaded(client, sample_history):
    """POST /api/ai/predict returns 503 when no model loaded."""
    # Convert to JSON-serializable format
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client.post("/api/ai/predict", json=history_json)

    assert response.status_code == 503
    assert "No surrogate model loaded" in response.json()["detail"]


def test_predict_returns_prediction_when_model_loaded(client_with_surrogate, sample_history):
    """POST /api/ai/predict returns prediction when model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client_with_surrogate.post("/api/ai/predict", json=history_json)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_state" in data
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] >= 0.0
    # Check structure preserved
    assert "rho" in data["predicted_state"]
    assert isinstance(data["predicted_state"]["rho"], list)


def test_rollout_returns_503_when_no_model_loaded(client, sample_history):
    """POST /api/ai/rollout returns 503 when no model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client.post("/api/ai/rollout?n_steps=5", json=history_json)

    assert response.status_code == 503


def test_rollout_returns_trajectory_when_model_loaded(client_with_surrogate, sample_history):
    """POST /api/ai/rollout returns trajectory when model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client_with_surrogate.post("/api/ai/rollout?n_steps=5", json=history_json)

    assert response.status_code == 200
    data = response.json()
    assert "trajectory" in data
    assert data["n_steps"] == 5
    assert len(data["trajectory"]) == 5
    assert "total_inference_time_ms" in data
    # Check structure
    assert "rho" in data["trajectory"][0]


def test_rollout_rejects_non_positive_n_steps(client_with_surrogate, sample_history):
    """POST /api/ai/rollout rejects n_steps <= 0."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client_with_surrogate.post("/api/ai/rollout?n_steps=0", json=history_json)

    assert response.status_code == 422
    assert "must be positive" in response.json()["detail"]


def test_rollout_rejects_excessive_n_steps(client_with_surrogate, sample_history):
    """POST /api/ai/rollout rejects n_steps > 1000."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client_with_surrogate.post("/api/ai/rollout?n_steps=1001", json=history_json)

    assert response.status_code == 422
    assert "too large" in response.json()["detail"]


def test_confidence_returns_503_when_no_ensemble_loaded(client, sample_history):
    """POST /api/ai/confidence returns 503 when no ensemble loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client.post("/api/ai/confidence", json=history_json)

    assert response.status_code == 503
    assert "No ensemble model loaded" in response.json()["detail"]


def test_confidence_returns_uncertainty_when_ensemble_loaded(client_with_ensemble, sample_history):
    """POST /api/ai/confidence returns uncertainty estimates when ensemble loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in sample_history]

    response = client_with_ensemble.post("/api/ai/confidence", json=history_json)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_state" in data
    assert "confidence" in data
    assert "ood_score" in data
    assert "confidence_score" in data
    assert "n_models" in data
    assert "inference_time_ms" in data
    assert data["ood_score"] == pytest.approx(0.05, abs=1e-6)
    assert data["confidence_score"] == pytest.approx(0.95, abs=1e-6)
    assert data["n_models"] == 5


# ── Module-level Function Tests ─────────────────────────────────


def test_require_surrogate_raises_http_exception_when_none():
    """_require_surrogate raises HTTPException when None."""
    from fastapi import HTTPException

    from dpf.ai import realtime_server

    original = realtime_server._surrogate
    realtime_server._surrogate = None

    with pytest.raises(HTTPException) as exc_info:
        realtime_server._require_surrogate()

    assert exc_info.value.status_code == 503
    assert "No surrogate model loaded" in exc_info.value.detail

    realtime_server._surrogate = original


def test_require_ensemble_raises_http_exception_when_none():
    """_require_ensemble raises HTTPException when None."""
    from fastapi import HTTPException

    from dpf.ai import realtime_server

    original = realtime_server._ensemble
    realtime_server._ensemble = None

    with pytest.raises(HTTPException) as exc_info:
        realtime_server._require_ensemble()

    assert exc_info.value.status_code == 503
    assert "No ensemble model loaded" in exc_info.value.detail

    realtime_server._ensemble = original


def test_load_surrogate_raises_runtime_error_when_no_torch():
    """load_surrogate raises RuntimeError when HAS_TORCH is False."""
    from dpf.ai import realtime_server

    with patch("dpf.ai.realtime_server.HAS_TORCH", False), \
         pytest.raises(RuntimeError, match="PyTorch not available"):
        realtime_server.load_surrogate("/fake/path")


def test_load_ensemble_raises_runtime_error_when_no_torch():
    """load_ensemble raises RuntimeError when HAS_TORCH is False."""
    from dpf.ai import realtime_server

    with patch("dpf.ai.realtime_server.HAS_TORCH", False), \
         pytest.raises(RuntimeError, match="PyTorch not available"):
        realtime_server.load_ensemble(["/fake/path"])
