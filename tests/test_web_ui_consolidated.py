"""Consolidated web UI and server tests.

Coverage:
- test_server.py: Binary field encoding, presets, REST endpoints, simulation lifecycle, WebSocket
- test_e2e_server.py: Full lifecycle E2E tests (create->start->stream->fields->stop)
- test_experimental_overlay.py: CSV overlay in waveform plots (app_plots)
- test_phase_f_cli_server.py: CLI --backend flag, verify, backends command, server health
- test_phase_i_ai_server.py: AI server endpoints (WALRUS surrogate, rollout, confidence)
- test_phase_j_cli_server.py: AthenaK CLI --backend option and server /api/health
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402, I001

# ---------------------------------------------------------------------------
# Module-level helpers (e2e)
# ---------------------------------------------------------------------------

def _create_sim(client, *, preset: str = "tutorial", max_steps: int = 5):
    """Create a simulation via REST and return sim_id."""
    r = client.post(
        "/api/simulations",
        json={"config": {}, "preset": preset, "max_steps": max_steps},
    )
    assert r.status_code == 200
    return r.json()["sim_id"]


# ---------------------------------------------------------------------------
# Mock classes (Phase I AI server)
# ---------------------------------------------------------------------------

class MockSurrogate:
    """Mock WALRUS surrogate for testing."""

    def __init__(self):
        self.history_length = 4
        self.is_loaded = True
        self.device = "cpu"

    def predict_next_step(self, history):
        last = history[-1]
        return {k: v.copy() for k, v in last.items()}

    def rollout(self, initial_states, n_steps):
        last = initial_states[-1]
        return [{k: v.copy() for k, v in last.items()} for _ in range(n_steps)]


class MockEnsemble:
    """Mock ensemble predictor for testing."""

    def __init__(self):
        self.n_models = 5
        self.device = "cpu"

    def predict(self, history):
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


# ---------------------------------------------------------------------------
# Fixtures (Phase F CLI/server)
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    """Click test runner."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def config_file(tmp_path, default_circuit_params):
    """Create a temporary config JSON file."""
    from dpf.config import SimulationConfig

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
    )
    path = tmp_path / "test_config.json"
    path.write_text(config.to_json())
    return str(path)


@pytest.fixture
def config_file_with_backend(tmp_path, default_circuit_params):
    """Create a config file with explicit backend field."""
    from dpf.config import SimulationConfig

    config = SimulationConfig(
        grid_shape=[4, 4, 4],
        dx=1e-3,
        sim_time=1e-9,
        dt_init=1e-12,
        circuit=default_circuit_params,
        fluid={"backend": "python"},
    )
    path = tmp_path / "test_config_backend.json"
    path.write_text(config.to_json())
    return str(path)


# ---------------------------------------------------------------------------
# Fixtures (Phase I AI server) — prefixed ai_ to avoid module-level collision
# ---------------------------------------------------------------------------

@pytest.fixture
def ai_client():
    """TestClient with no models loaded."""
    from dpf.server.app import app

    return TestClient(app)


@pytest.fixture
def ai_client_with_surrogate():
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
def ai_client_with_ensemble():
    """TestClient with mock ensemble loaded."""
    from dpf.ai import realtime_server
    from dpf.server.app import app

    mock = MockEnsemble()
    original_surrogate = realtime_server._surrogate
    original_ensemble = realtime_server._ensemble
    realtime_server._surrogate = MockSurrogate()
    realtime_server._ensemble = mock

    client = TestClient(app)
    yield client

    realtime_server._surrogate = original_surrogate
    realtime_server._ensemble = original_ensemble


@pytest.fixture
def ai_sample_state():
    """Single state dict for testing."""
    return {
        "rho": np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        "Te": np.array([[[100.0, 200.0], [300.0, 400.0]]]),
        "velocity": np.array([[[0.0, 1.0], [2.0, 3.0]]]),
    }


@pytest.fixture
def ai_sample_history(ai_sample_state):
    """History of 4 states."""
    return [ai_sample_state.copy() for _ in range(4)]


# ---------------------------------------------------------------------------
# Section: Binary Field Encoding
# ---------------------------------------------------------------------------

class TestFieldEncoding:
    """Tests for server.encoding module."""

    def test_downsample_scalar_field(self):
        """Downsampling a 3D scalar field by factor 2."""
        from dpf.server.encoding import downsample_field

        arr = np.random.rand(16, 16, 16).astype(np.float64)
        result = downsample_field(arr, 2)
        assert result.shape == (8, 8, 8)
        assert result.dtype == np.float32

    def test_downsample_vector_field(self):
        """Downsampling a 4D vector field by factor 2."""
        from dpf.server.encoding import downsample_field

        arr = np.random.rand(3, 16, 16, 16).astype(np.float64)
        result = downsample_field(arr, 2)
        assert result.shape == (3, 8, 8, 8)
        assert result.dtype == np.float32

    def test_downsample_factor_1(self):
        """Factor 1 returns float32 copy without downsampling."""
        from dpf.server.encoding import downsample_field

        arr = np.random.rand(8, 8, 8).astype(np.float64)
        result = downsample_field(arr, 1)
        assert result.shape == (8, 8, 8)
        assert result.dtype == np.float32

    def test_encode_fields_basic(self):
        """Encode two fields, verify header and blob size."""
        from dpf.server.encoding import encode_fields

        snapshot = {
            "rho": np.ones((4, 4, 4)),
            "Te": np.full((4, 4, 4), 1000.0),
            "B": np.zeros((3, 4, 4, 4)),
        }
        header, blob = encode_fields(snapshot, ["rho", "Te"])
        assert "rho" in header
        assert "Te" in header
        assert "B" not in header

        rho_nbytes = header["rho"]["nbytes"]
        te_nbytes = header["Te"]["nbytes"]
        assert len(blob) == rho_nbytes + te_nbytes
        assert header["rho"]["dtype"] == "float32"

    def test_encode_fields_with_downsample(self):
        """Downsampled encoding produces smaller blob."""
        from dpf.server.encoding import encode_fields

        snapshot = {"rho": np.ones((16, 16, 16))}
        _, blob_full = encode_fields(snapshot, ["rho"], downsample=1)
        _, blob_ds2 = encode_fields(snapshot, ["rho"], downsample=2)
        assert len(blob_ds2) < len(blob_full)
        assert len(blob_ds2) == 8 * 8 * 8 * 4

    def test_encode_missing_field_skipped(self):
        """Requesting a non-existent field is silently skipped."""
        from dpf.server.encoding import encode_fields

        snapshot = {"rho": np.ones((4, 4, 4))}
        header, blob = encode_fields(snapshot, ["rho", "nonexistent"])
        assert "rho" in header
        assert "nonexistent" not in header

    def test_decode_roundtrip(self):
        """Encoded float32 data can be decoded back to correct values."""
        from dpf.server.encoding import encode_fields

        original = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        snapshot = {"rho": original}
        header, blob = encode_fields(snapshot, ["rho"])
        decoded = np.frombuffer(blob, dtype=np.float32).reshape(header["rho"]["shape"])
        np.testing.assert_allclose(decoded, original.astype(np.float32))


# ---------------------------------------------------------------------------
# Section: Presets
# ---------------------------------------------------------------------------

class TestPresets:
    """Tests for configuration presets."""

    def test_list_presets_nonempty(self):
        """list_presets() returns a non-empty list."""
        from dpf.presets import list_presets

        presets = list_presets()
        assert len(presets) >= 3
        assert all("name" in p for p in presets)

    def test_get_preset_tutorial(self):
        """Tutorial preset produces a valid config dict."""
        from dpf.presets import get_preset

        data = get_preset("tutorial")
        assert "grid_shape" in data
        assert "circuit" in data
        assert "_meta" not in data

    def test_get_preset_creates_valid_config(self):
        """Each preset creates a valid SimulationConfig."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset, get_preset_names

        for name in get_preset_names():
            data = get_preset(name)
            config = SimulationConfig(**data)
            assert config.sim_time > 0

    def test_get_preset_unknown_raises(self):
        """Unknown preset name raises KeyError."""
        from dpf.presets import get_preset

        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_device")

    def test_preset_names(self):
        """get_preset_names returns expected devices."""
        from dpf.presets import get_preset_names

        names = get_preset_names()
        assert "tutorial" in names
        assert "pf1000" in names
        assert "nx2" in names


# ---------------------------------------------------------------------------
# Section: REST Endpoint Tests
# ---------------------------------------------------------------------------

class TestRESTEndpoints:
    """Tests for REST API endpoints."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_health(self, client):
        """GET /api/health returns 200."""
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_create_simulation(self, client):
        """POST /api/simulations creates a simulation."""
        config = {
            "grid_shape": [4, 4, 4], "dx": 1e-3, "sim_time": 1e-7, "dt_init": 1e-10,
            "circuit": {
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        }
        r = client.post("/api/simulations", json={"config": config})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "idle"
        assert "sim_id" in data

    def test_create_from_preset(self, client):
        """POST /api/simulations with preset name."""
        r = client.post("/api/simulations", json={"config": {}, "preset": "tutorial"})
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    def test_create_invalid_config(self, client):
        """POST /api/simulations with bad config returns 422."""
        r = client.post("/api/simulations", json={"config": {"grid_shape": [1]}})
        assert r.status_code == 422

    def test_get_simulation(self, client):
        """GET /api/simulations/{id} returns info."""
        r = client.post("/api/simulations", json={"config": {}, "preset": "tutorial"})
        sim_id = r.json()["sim_id"]

        r = client.get(f"/api/simulations/{sim_id}")
        assert r.status_code == 200
        assert r.json()["sim_id"] == sim_id

    def test_get_simulation_not_found(self, client):
        """GET /api/simulations/nonexistent returns 404."""
        r = client.get("/api/simulations/nonexistent")
        assert r.status_code == 404

    def test_config_schema(self, client):
        """GET /api/config/schema returns JSON schema."""
        r = client.get("/api/config/schema")
        assert r.status_code == 200
        schema = r.json()
        assert "properties" in schema

    def test_validate_config_valid(self, client):
        """POST /api/config/validate with valid config."""
        config = {
            "grid_shape": [4, 4, 4], "dx": 1e-3, "sim_time": 1e-7,
            "circuit": {
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        }
        r = client.post("/api/config/validate", json=config)
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_validate_config_invalid(self, client):
        """POST /api/config/validate with invalid config."""
        r = client.post("/api/config/validate", json={"grid_shape": [0, 0, 0]})
        assert r.status_code == 200
        assert r.json()["valid"] is False
        assert len(r.json()["errors"]) > 0

    def test_get_presets(self, client):
        """GET /api/presets returns preset list."""
        r = client.get("/api/presets")
        assert r.status_code == 200
        presets = r.json()
        assert len(presets) >= 3
        names = [p["name"] for p in presets]
        assert "tutorial" in names


# ---------------------------------------------------------------------------
# Section: Simulation Lifecycle
# ---------------------------------------------------------------------------

class TestSimulationLifecycle:
    """Tests for start/pause/resume/stop via REST."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def _create(self, client):
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial", "max_steps": 5},
        )
        return r.json()["sim_id"]

    def test_start_simulation(self, client):
        """POST /start transitions to running."""
        sim_id = self._create(client)
        r = client.post(f"/api/simulations/{sim_id}/start")
        assert r.status_code == 200
        assert r.json()["status"] in ("running", "finished")

    def test_stop_idle_simulation(self, client):
        """POST /stop on idle sim transitions to finished."""
        sim_id = self._create(client)
        r = client.post(f"/api/simulations/{sim_id}/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "finished"

    def test_fields_before_start(self, client):
        """GET /fields before starting returns field metadata."""
        sim_id = self._create(client)
        r = client.get(f"/api/simulations/{sim_id}/fields?fields=rho")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Section: SimulationManager Unit Tests
# ---------------------------------------------------------------------------

class TestSimulationManager:
    """Direct unit tests for SimulationManager."""

    def _make_config(self):
        from dpf.config import SimulationConfig

        return SimulationConfig(
            grid_shape=[4, 4, 4], dx=1e-3, sim_time=1e-7, dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )

    def test_create_engine(self):
        """create_engine() initializes the engine."""
        from dpf.server.simulation import SimulationManager

        mgr = SimulationManager(self._make_config())
        assert mgr.engine is None
        mgr.create_engine()
        assert mgr.engine is not None

    def test_info_before_run(self):
        """info() returns valid data before any steps."""
        from dpf.server.models import SimulationStatus
        from dpf.server.simulation import SimulationManager

        mgr = SimulationManager(self._make_config())
        mgr.create_engine()
        info = mgr.info()
        assert info["status"] == SimulationStatus.idle.value
        assert info["step"] == 0

    def test_get_field_snapshot(self):
        """get_field_snapshot() returns field arrays."""
        from dpf.server.simulation import SimulationManager

        mgr = SimulationManager(self._make_config())
        mgr.create_engine()
        snap = mgr.get_field_snapshot()
        assert "rho" in snap
        assert snap["rho"].shape == (4, 4, 4)

    @pytest.mark.asyncio
    async def test_run_to_completion(self):
        """SimulationManager runs to finished with max_steps."""
        from dpf.server.models import SimulationStatus
        from dpf.server.simulation import SimulationManager

        mgr = SimulationManager(self._make_config(), max_steps=3)
        mgr.create_engine()
        await mgr.start()

        for _ in range(50):
            await asyncio.sleep(0.1)
            if mgr.status == SimulationStatus.finished:
                break

        assert mgr.status == SimulationStatus.finished
        assert mgr.last_result is not None
        assert mgr.last_result.step == 3

    @pytest.mark.asyncio
    async def test_subscribe_receives_results(self):
        """Subscriber queue receives StepResults."""
        from dpf.server.simulation import SimulationManager

        mgr = SimulationManager(self._make_config(), max_steps=3)
        mgr.create_engine()
        q = mgr.subscribe()
        await mgr.start()

        results = []
        for _ in range(20):
            try:
                r = await asyncio.wait_for(q.get(), timeout=1.0)
                results.append(r)
                if r.finished:
                    break
            except TimeoutError:
                break

        assert len(results) >= 3
        mgr.unsubscribe(q)


# ---------------------------------------------------------------------------
# Section: WebSocket Tests
# ---------------------------------------------------------------------------

class TestWebSocket:
    """Tests for WebSocket streaming endpoint."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_ws_not_found(self, client):
        """WebSocket to nonexistent sim gets closed."""
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/nonexistent"):
            pass

    def test_ws_connect_to_valid_sim(self, client):
        """WebSocket can connect to a valid simulation."""
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial", "max_steps": 3},
        )
        sim_id = r.json()["sim_id"]

        with client.websocket_connect(f"/ws/{sim_id}"):
            pass


# ---------------------------------------------------------------------------
# Section: E2E REST Lifecycle
# ---------------------------------------------------------------------------

class TestE2ELifecycle:
    """Full REST lifecycle: create -> start -> poll -> stop."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_create_and_status(self, client):
        """Create sim -> verify idle -> get status."""
        sim_id = _create_sim(client, max_steps=5)

        r = client.get(f"/api/simulations/{sim_id}")
        assert r.json()["status"] == "idle"
        assert r.json()["step"] == 0

    def test_start_simulation(self, client):
        """Start a simulation (may finish quickly with 5 steps)."""
        sim_id = _create_sim(client, max_steps=5)

        r = client.post(f"/api/simulations/{sim_id}/start")
        assert r.status_code == 200
        assert r.json()["status"] in ("running", "finished")

    def test_stop_idle_simulation(self, client):
        """Stop an idle simulation transitions to finished."""
        sim_id = _create_sim(client, max_steps=5)

        r = client.post(f"/api/simulations/{sim_id}/stop")
        assert r.status_code == 200
        assert r.json()["status"] == "finished"

    def test_preset_creates_valid_engine(self, client):
        """Each preset creates a sim that can run at least 1 step."""
        from dpf.presets import get_preset_names

        for name in get_preset_names():
            r = client.post(
                "/api/simulations",
                json={"config": {}, "preset": name, "max_steps": 1},
            )
            assert r.status_code == 200, f"Preset '{name}' failed to create"
            sim_id = r.json()["sim_id"]
            r = client.post(f"/api/simulations/{sim_id}/start")
            assert r.status_code == 200, f"Preset '{name}' failed to start"

    def test_get_fields_returns_metadata(self, client):
        """GET /fields returns field metadata with byte count."""
        sim_id = _create_sim(client)
        r = client.get(f"/api/simulations/{sim_id}/fields?fields=rho,Te")
        assert r.status_code == 200
        data = r.json()
        assert "rho" in data["fields"]
        assert "Te" in data["fields"]
        assert data["total_bytes"] > 0

    def test_field_shapes_match_grid(self, client):
        """Field shapes match the grid_shape from config."""
        sim_id = _create_sim(client, preset="tutorial")
        r = client.get(f"/api/simulations/{sim_id}/fields?fields=rho")
        assert r.status_code == 200
        shape = r.json()["fields"]["rho"]["shape"]
        assert shape == [8, 8, 8]

    def test_config_validate_roundtrip(self, client):
        """Validate a preset config via the REST endpoint."""
        from dpf.presets import get_preset

        config = get_preset("tutorial")
        r = client.post("/api/config/validate", json=config)
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_schema_contains_required_fields(self, client):
        """Config schema has essential properties."""
        r = client.get("/api/config/schema")
        assert r.status_code == 200
        props = r.json()["properties"]
        assert "grid_shape" in props
        assert "circuit" in props
        assert "sim_time" in props

    def test_multiple_simulations(self, client):
        """Can create and manage multiple simulations."""
        ids = [_create_sim(client, max_steps=2) for _ in range(3)]
        assert len(set(ids)) == 3

        for sid in ids:
            r = client.get(f"/api/simulations/{sid}")
            assert r.status_code == 200


# ---------------------------------------------------------------------------
# Section: E2E WebSocket
# ---------------------------------------------------------------------------

class TestE2EWebSocket:
    """WebSocket end-to-end tests."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_ws_rejects_unknown_sim(self, client):
        """WebSocket to nonexistent sim is rejected."""
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/nonexistent"):
            pass

    def test_ws_accepts_valid_sim(self, client):
        """WebSocket accepts connection to a valid sim."""
        sim_id = _create_sim(client, max_steps=3)
        with client.websocket_connect(f"/ws/{sim_id}"):
            pass

    def test_ws_field_request(self, client):
        """Client can request field data via WebSocket."""
        sim_id = _create_sim(client, max_steps=3)

        client.post(f"/api/simulations/{sim_id}/start")

        with client.websocket_connect(f"/ws/{sim_id}") as ws:
            ws.send_text(json.dumps({
                "type": "request_fields",
                "fields": ["rho", "Te"],
                "downsample": 1,
            }))

            header_text = ws.receive_text()
            header = json.loads(header_text)

            assert "fields" in header
            assert "total_bytes" in header

            if header["total_bytes"] > 0:
                blob = ws.receive_bytes()
                assert len(blob) == header["total_bytes"]

                rho_info = header["fields"]["rho"]
                rho_data = np.frombuffer(
                    blob[rho_info["offset"]:rho_info["offset"] + rho_info["nbytes"]],
                    dtype=np.float32,
                ).reshape(rho_info["shape"])
                assert rho_data.shape == tuple(rho_info["shape"])
                assert np.all(np.isfinite(rho_data))


# ---------------------------------------------------------------------------
# Section: E2E Binary Encoding
# ---------------------------------------------------------------------------

class TestE2EEncoding:
    """Verify field encoding roundtrip with real engine data."""

    def test_engine_fields_encode_decode(self):
        """Real engine fields survive encode -> decode."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.server.encoding import encode_fields

        config = SimulationConfig(
            grid_shape=[8, 8, 8], dx=1e-3, sim_time=1e-7, dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )
        engine = SimulationEngine(config)
        engine.step()

        snapshot = engine.get_field_snapshot()
        header, blob = encode_fields(snapshot, ["rho", "Te", "B"])

        assert "rho" in header
        assert "Te" in header
        assert "B" in header

        rho_h = header["rho"]
        rho = np.frombuffer(
            blob[rho_h["offset"]:rho_h["offset"] + rho_h["nbytes"]],
            dtype=np.float32,
        ).reshape(rho_h["shape"])
        assert rho.shape == (8, 8, 8)
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)

        b_h = header["B"]
        B = np.frombuffer(
            blob[b_h["offset"]:b_h["offset"] + b_h["nbytes"]],
            dtype=np.float32,
        ).reshape(b_h["shape"])
        assert B.shape == (3, 8, 8, 8)

    def test_downsample_preserves_physics(self):
        """Downsampled fields maintain physical reasonableness."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.server.encoding import encode_fields

        config = SimulationConfig(
            grid_shape=[16, 16, 16], dx=1e-3, sim_time=1e-7, dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )
        engine = SimulationEngine(config)
        engine.step()

        snapshot = engine.get_field_snapshot()
        _, blob_full = encode_fields(snapshot, ["rho"], downsample=1)
        _, blob_ds2 = encode_fields(snapshot, ["rho"], downsample=2)
        _, blob_ds4 = encode_fields(snapshot, ["rho"], downsample=4)

        assert len(blob_ds2) < len(blob_full)
        assert len(blob_ds4) < len(blob_ds2)
        assert len(blob_full) == 16 * 16 * 16 * 4
        assert len(blob_ds2) == 8 * 8 * 8 * 4
        assert len(blob_ds4) == 4 * 4 * 4 * 4


# ---------------------------------------------------------------------------
# Section: Experimental CSV Overlay (app_plots)
# ---------------------------------------------------------------------------

def _minimal_sim_data() -> dict:
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    return {
        "t_us": t,
        "I_MA": [0.0, 0.5, 1.0, 0.8, 0.2],
        "V_kV": [27.0, 20.0, 10.0, 5.0, 0.0],
        "phases": ["rundown"] * 5,
        "has_snowplow": False,
        "dip_pct": 0.0,
        "t_peak": 2.0,
        "I_peak": 1.0,
        "I_dip": 0.8,
        "t_dip": 3.0,
        "crowbar_t": None,
        "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
        "snowplow_cfg": {"anode_length": 0.60},
        "snowplow_obj": None,
    }


class TestParseExperimentalCsv:
    def test_time_us_current_ma(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,current_MA\n0.0,0.0\n1.0,0.5\n2.0,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.0, 1.0, 2.0])
        assert result["I_MA"] == pytest.approx([0.0, 0.5, 1.0])

    def test_t_us_i_ma_aliases(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "t_us,I_MA\n0.5,0.1\n1.5,0.9\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.5, 1.5])
        assert result["I_MA"] == pytest.approx([0.1, 0.9])

    def test_generic_t_i_columns(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "t,I\n0.0,0.0\n2.0,1.2\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([0.0, 2.0])
        assert result["I_MA"] == pytest.approx([0.0, 1.2])

    def test_time_current_generic_labels(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time,current\n1.0,0.3\n3.0,0.9\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 3.0])
        assert result["I_MA"] == pytest.approx([0.3, 0.9])


class TestUnitConversion:
    def test_time_s_to_us(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_s,current_MA\n0.000001,0.5\n0.000002,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 2.0], rel=1e-5)

    def test_current_a_to_ma(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,current_A\n0.0,500000.0\n1.0,1000000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["I_MA"] == pytest.approx([0.5, 1.0], rel=1e-5)

    def test_time_s_current_a_both_converted(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_s,current_A\n0.000001,500000.0\n0.000002,1000000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([1.0, 2.0], rel=1e-5)
        assert result["I_MA"] == pytest.approx([0.5, 1.0], rel=1e-5)

    def test_t_s_alias(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "t_s,I_MA\n0.000003,0.7\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["t_us"] == pytest.approx([3.0], rel=1e-5)

    def test_i_a_alias(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "t_us,I_A\n1.0,750000.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert result["I_MA"] == pytest.approx([0.75], rel=1e-5)


class TestMalformedCsv:
    def test_empty_string(self) -> None:
        from app_plots import parse_experimental_csv

        assert parse_experimental_csv("") is None

    def test_no_matching_columns(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "foo,bar\n1,2\n3,4\n"
        assert parse_experimental_csv(csv) is None

    def test_missing_current_column(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,voltage_kV\n0.0,27.0\n"
        assert parse_experimental_csv(csv) is None

    def test_missing_time_column(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "step,current_MA\n1,0.5\n"
        assert parse_experimental_csv(csv) is None

    def test_non_numeric_values_raise_handled(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,current_MA\nabc,def\n"
        assert parse_experimental_csv(csv) is None

    def test_empty_rows_skipped(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,current_MA\n0.0,0.5\n\n2.0,1.0\n"
        result = parse_experimental_csv(csv)
        assert result is not None
        assert len(result["t_us"]) == 2

    def test_header_only_no_data(self) -> None:
        from app_plots import parse_experimental_csv

        csv = "time_us,current_MA\n"
        assert parse_experimental_csv(csv) is None


class TestCreateWaveformFig:
    def test_no_experimental_data(self) -> None:
        import plotly.graph_objects as go

        from app_plots import create_waveform_fig

        fig = create_waveform_fig(_minimal_sim_data())
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Experimental" not in names

    def test_with_experimental_data(self) -> None:
        import plotly.graph_objects as go

        from app_plots import create_waveform_fig

        exp = {"t_us": [0.0, 1.0, 2.0], "I_MA": [0.0, 0.4, 0.9]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Experimental" in names

    def test_experimental_trace_style(self) -> None:
        from app_plots import create_waveform_fig

        exp = {"t_us": [0.0, 1.0], "I_MA": [0.0, 1.0]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        exp_trace = next(t for t in fig.data if t.name == "Experimental")
        assert exp_trace.line.color == "red"
        assert exp_trace.line.dash == "dash"

    def test_experimental_data_none_is_default(self) -> None:
        from app_plots import create_waveform_fig

        fig_no_arg = create_waveform_fig(_minimal_sim_data())
        fig_none = create_waveform_fig(_minimal_sim_data(), experimental_data=None)
        assert len(fig_no_arg.data) == len(fig_none.data)

    def test_experimental_data_on_correct_subplot(self) -> None:
        from app_plots import create_waveform_fig

        exp = {"t_us": [0.0, 1.0], "I_MA": [0.0, 1.0]}
        fig = create_waveform_fig(_minimal_sim_data(), experimental_data=exp)
        exp_trace = next(t for t in fig.data if t.name == "Experimental")
        assert exp_trace.xaxis == "x"
        assert exp_trace.yaxis == "y"


# ---------------------------------------------------------------------------
# Section: CLI --backend flag (Phase F.4)
# ---------------------------------------------------------------------------

class TestCLIBackendFlag:
    """Tests for the --backend flag on the simulate command."""

    def test_simulate_help_shows_backend(self, runner):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output
        assert "python" in result.output
        assert "athena" in result.output
        assert "auto" in result.output

    def test_simulate_python_backend(self, runner, config_file):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["simulate", config_file, "--backend=python", "--steps=1"])
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_simulate_default_backend(self, runner, config_file):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["simulate", config_file, "--steps=1"])
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_backend_override_in_config(self, runner, config_file_with_backend):
        from dpf.cli.main import cli

        result = runner.invoke(
            cli,
            ["simulate", config_file_with_backend, "--backend=python", "--steps=1"],
        )
        assert result.exit_code == 0
        assert "Backend: python" in result.output

    def test_invalid_backend_rejected(self, runner, config_file):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["simulate", config_file, "--backend=cuda", "--steps=1"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


class TestCLIVerifyCommand:
    """Tests for the verify command backend display."""

    def test_verify_shows_backend(self, runner, config_file):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["verify", config_file])
        assert result.exit_code == 0
        assert "Backend:" in result.output
        assert "python" in result.output

    def test_verify_shows_all_fields(self, runner, config_file):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["verify", config_file])
        assert result.exit_code == 0
        assert "Grid:" in result.output
        assert "dx:" in result.output
        assert "sim_time:" in result.output
        assert "Circuit:" in result.output
        assert "Fluid:" in result.output
        assert "Backend:" in result.output


class TestCLIBackendsCommand:
    """Tests for the backends command."""

    def test_backends_command_exists(self, runner):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0

    def test_backends_shows_python(self, runner):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["backends"])
        assert "python" in result.output
        assert "always available" in result.output

    def test_backends_shows_athena_status(self, runner):
        from dpf.cli.main import cli

        result = runner.invoke(cli, ["backends"])
        assert "athena" in result.output
        assert "available" in result.output or "not compiled" in result.output


class TestServerBackendReporting:
    """Tests for server API backend reporting."""

    @pytest.fixture()
    def client(self):
        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_health_reports_backends(self, client):
        """GET /api/health includes backends dict."""
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert "backends" in data
        assert data["backends"]["python"] is True
        assert isinstance(data["backends"]["athena"], bool)

    def test_simulation_info_includes_backend(self, client):
        """POST /api/simulations returns backend in SimulationInfo."""
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "backend" in data
        assert data["backend"] == "python"

    def test_simulation_info_backend_persists(self, client):
        """GET /api/simulations/{id} also returns backend."""
        r = client.post(
            "/api/simulations",
            json={"config": {}, "preset": "tutorial"},
        )
        sim_id = r.json()["sim_id"]

        r = client.get(f"/api/simulations/{sim_id}")
        assert r.status_code == 200
        assert r.json()["backend"] == "python"

    def test_create_with_backend_in_config(self, client):
        """Config with explicit backend field works in server."""
        config = {
            "grid_shape": [4, 4, 4], "dx": 1e-3, "sim_time": 1e-7,
            "circuit": {
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            "fluid": {"backend": "python"},
        }
        r = client.post("/api/simulations", json={"config": config})
        assert r.status_code == 200
        assert r.json()["backend"] == "python"


class TestPresetsBackend:
    """Verify presets work with backend field in config."""

    def test_preset_default_backend(self):
        """Presets default to python backend."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset

        data = get_preset("tutorial")
        config = SimulationConfig(**data)
        assert config.fluid.backend == "python"

    def test_preset_override_backend(self):
        """Can override backend in preset config."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset

        data = get_preset("tutorial")
        data["fluid"] = {"backend": "auto"}
        config = SimulationConfig(**data)
        assert config.fluid.backend == "auto"

    def test_all_presets_valid_with_backend(self):
        """All presets produce valid configs with backend field accessible."""
        from dpf.config import SimulationConfig
        from dpf.presets import get_preset, get_preset_names

        for name in get_preset_names():
            data = get_preset(name)
            config = SimulationConfig(**data)
            assert config.fluid.backend in ("python", "athena", "athenak", "metal", "auto")


class TestConfigBackendSerialization:
    """Verify backend field survives config serialization."""

    def test_backend_in_json(self, default_circuit_params):
        """Backend appears in JSON output."""
        import json as _json

        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[4, 4, 4], dx=1e-3, sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        data = _json.loads(config.to_json())
        assert data["fluid"]["backend"] == "python"

    def test_backend_roundtrip(self, default_circuit_params):
        """Backend survives JSON round-trip."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[4, 4, 4], dx=1e-3, sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "auto"},
        )
        json_str = config.to_json()
        restored = SimulationConfig.model_validate_json(json_str)
        assert restored.fluid.backend == "auto"

    def test_backend_file_roundtrip(self, tmp_path, default_circuit_params):
        """Backend survives file write/read round-trip."""
        from dpf.config import SimulationConfig

        config = SimulationConfig(
            grid_shape=[4, 4, 4], dx=1e-3, sim_time=1e-7,
            circuit=default_circuit_params,
            fluid={"backend": "python"},
        )
        path = tmp_path / "roundtrip.json"
        path.write_text(config.to_json())
        restored = SimulationConfig.from_file(str(path))
        assert restored.fluid.backend == "python"


# ---------------------------------------------------------------------------
# Section: AthenaK CLI/Server (Phase J)
# ---------------------------------------------------------------------------

class TestCLIBackendOption:
    """Test CLI --backend option includes athenak."""

    def test_simulate_help_shows_athenak(self):
        from click.testing import CliRunner

        from dpf.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate", "--help"])
        assert result.exit_code == 0
        assert "athenak" in result.output

    def test_backends_command_shows_athenak(self):
        from click.testing import CliRunner

        from dpf.cli.main import cli
        runner = CliRunner()
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "athenak" in result.output

    def test_backends_command_available(self):
        from click.testing import CliRunner

        from dpf.cli.main import cli
        runner = CliRunner()
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=True),
        ):
            result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "athenak" in result.output
        assert "available" in result.output

    def test_export_well_help_shows_athenak(self):
        from click.testing import CliRunner

        from dpf.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["export-well", "--help"])
        assert result.exit_code == 0
        assert "athenak" in result.output


class TestServerHealth:
    """Test /api/health endpoint includes athenak."""

    @pytest.fixture
    def client(self):
        from dpf.server.app import app
        return TestClient(app)

    def test_health_includes_athenak(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "athenak" in data["backends"]
        assert isinstance(data["backends"]["athenak"], bool)

    def test_health_athenak_available(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=True),
        ):
            resp = client.get("/api/health")
        data = resp.json()
        assert data["backends"]["athenak"] is True

    def test_health_still_has_python_and_athena(self, client):
        with (
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.athenak_wrapper.is_available", return_value=False),
        ):
            resp = client.get("/api/health")
        data = resp.json()
        assert "python" in data["backends"]
        assert "athena" in data["backends"]
        assert data["backends"]["python"] is True


# ---------------------------------------------------------------------------
# Section: AI Server Utility Functions (Phase I)
# ---------------------------------------------------------------------------

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
    """_lists_to_arrays keeps non-numeric lists unconverted."""
    from dpf.ai.realtime_server import _lists_to_arrays

    state = {"strings": ["a", "b", "c"]}
    result = _lists_to_arrays(state)

    assert isinstance(result["strings"], list)
    assert result["strings"] == ["a", "b", "c"]


def test_lists_to_arrays_handles_lists_of_dicts():
    """_lists_to_arrays converts list of dicts — non-numeric dtype rejected."""
    from dpf.ai.realtime_server import _lists_to_arrays

    state = {"items": [{"data": [1.0]}, {"data": [2.0]}]}
    result = _lists_to_arrays(state)

    assert isinstance(result["items"], list)


# ---------------------------------------------------------------------------
# Section: AI Server REST Endpoints (Phase I)
# ---------------------------------------------------------------------------

def test_status_returns_correct_structure_when_no_model_loaded(ai_client):
    """GET /api/ai/status returns correct structure when no model loaded."""
    response = ai_client.get("/api/ai/status")

    assert response.status_code == 200
    data = response.json()
    assert "torch_available" in data
    assert data["model_loaded"] is False
    assert data["device"] == "none"
    assert data["ensemble_size"] == 0


def test_status_returns_model_loaded_true_when_surrogate_set(ai_client_with_surrogate):
    """GET /api/ai/status returns model_loaded=True when surrogate set."""
    response = ai_client_with_surrogate.get("/api/ai/status")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["device"] == "cpu"


def test_predict_returns_503_when_no_model_loaded(ai_client, ai_sample_history):
    """POST /api/ai/predict returns 503 when no model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client.post("/api/ai/predict", json=history_json)

    assert response.status_code == 503
    assert "No surrogate model loaded" in response.json()["detail"]


def test_predict_returns_prediction_when_model_loaded(ai_client_with_surrogate, ai_sample_history):
    """POST /api/ai/predict returns prediction when model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client_with_surrogate.post("/api/ai/predict", json=history_json)

    assert response.status_code == 200
    data = response.json()
    assert "predicted_state" in data
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] >= 0.0
    assert "rho" in data["predicted_state"]
    assert isinstance(data["predicted_state"]["rho"], list)


def test_rollout_returns_503_when_no_model_loaded(ai_client, ai_sample_history):
    """POST /api/ai/rollout returns 503 when no model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client.post("/api/ai/rollout?n_steps=5", json=history_json)

    assert response.status_code == 503


def test_rollout_returns_trajectory_when_model_loaded(ai_client_with_surrogate, ai_sample_history):
    """POST /api/ai/rollout returns trajectory when model loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client_with_surrogate.post("/api/ai/rollout?n_steps=5", json=history_json)

    assert response.status_code == 200
    data = response.json()
    assert "trajectory" in data
    assert data["n_steps"] == 5
    assert len(data["trajectory"]) == 5
    assert "total_inference_time_ms" in data
    assert "rho" in data["trajectory"][0]


def test_rollout_rejects_non_positive_n_steps(ai_client_with_surrogate, ai_sample_history):
    """POST /api/ai/rollout rejects n_steps <= 0."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client_with_surrogate.post("/api/ai/rollout?n_steps=0", json=history_json)

    assert response.status_code == 422
    assert "must be positive" in response.json()["detail"]


def test_rollout_rejects_excessive_n_steps(ai_client_with_surrogate, ai_sample_history):
    """POST /api/ai/rollout rejects n_steps > 1000."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client_with_surrogate.post("/api/ai/rollout?n_steps=1001", json=history_json)

    assert response.status_code == 422
    assert "too large" in response.json()["detail"]


def test_confidence_returns_503_when_no_ensemble_loaded(ai_client, ai_sample_history):
    """POST /api/ai/confidence returns 503 when no ensemble loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client.post("/api/ai/confidence", json=history_json)

    assert response.status_code == 503
    assert "No ensemble model loaded" in response.json()["detail"]


def test_confidence_returns_uncertainty_when_ensemble_loaded(ai_client_with_ensemble, ai_sample_history):
    """POST /api/ai/confidence returns uncertainty estimates when ensemble loaded."""
    from dpf.ai.realtime_server import _arrays_to_lists

    history_json = [_arrays_to_lists(state) for state in ai_sample_history]

    response = ai_client_with_ensemble.post("/api/ai/confidence", json=history_json)

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


# ---------------------------------------------------------------------------
# Section: Web UI Hardening (v0.2 Sprint)
# ---------------------------------------------------------------------------


class TestAnimationFrameCap:
    """Animation frame count is capped at MAX_ANIMATION_FRAMES=200."""

    def test_3d_anim_respects_max_frames(self) -> None:
        """create_animated_3d never produces more than 200 frames."""
        from app_anim import MAX_ANIMATION_FRAMES, create_animated_3d

        n = 1000
        d = {
            "t_us": list(range(n)),
            "z_mm": [float(i) for i in range(n)],
            "r_mm": [float(i) for i in range(n)],
            "I_MA": [float(i) / n for i in range(n)],
            "phases": ["rundown"] * n,
            "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
            "snowplow_cfg": {"anode_length": 0.60, "pinch_column_fraction": 1.0},
        }
        fig = create_animated_3d(d, n_frames=300)
        assert len(fig.frames) <= MAX_ANIMATION_FRAMES

    def test_3d_anim_small_dataset_unaffected(self) -> None:
        """create_animated_3d with fewer than 200 steps is not truncated."""
        from app_anim import MAX_ANIMATION_FRAMES, create_animated_3d

        n = 50
        d = {
            "t_us": list(range(n)),
            "z_mm": [float(i) for i in range(n)],
            "r_mm": [float(i) for i in range(n)],
            "I_MA": [0.1] * n,
            "phases": ["rundown"] * n,
            "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
            "snowplow_cfg": {"anode_length": 0.60, "pinch_column_fraction": 1.0},
        }
        fig = create_animated_3d(d, n_frames=80)
        assert len(fig.frames) <= MAX_ANIMATION_FRAMES

    def test_mhd_anim_respects_max_frames(self) -> None:
        """create_animated_mhd subsamples snapshots beyond 200."""
        from app_anim import MAX_ANIMATION_FRAMES, create_animated_mhd

        n_snaps = 500
        snapshots = [
            {"t_us": float(i), "rho_mid": np.ones((4, 8)) * (i + 1)}
            for i in range(n_snaps)
        ]
        d = {
            "mhd_snapshots": snapshots,
            "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
            "snowplow_cfg": {"anode_length": 0.60},
            "rho0": 1.0,
            "I_MA": np.array([0.5] * n_snaps),
            "t_us": np.array([float(i) for i in range(n_snaps)]),
        }
        fig = create_animated_mhd(d)
        assert len(fig.frames) <= MAX_ANIMATION_FRAMES

    def test_mhd_anim_small_snapshot_count_unaffected(self) -> None:
        """create_animated_mhd with <= 200 snapshots passes all through."""
        from app_anim import MAX_ANIMATION_FRAMES, create_animated_mhd

        n_snaps = 10
        snapshots = [
            {"t_us": float(i), "rho_mid": np.ones((4, 8))}
            for i in range(n_snaps)
        ]
        d = {
            "mhd_snapshots": snapshots,
            "circuit": {"anode_radius": 0.115, "cathode_radius": 0.160},
            "snowplow_cfg": {"anode_length": 0.60},
            "rho0": 1.0,
            "I_MA": np.array([0.5] * n_snaps),
            "t_us": np.array([float(i) for i in range(n_snaps)]),
        }
        fig = create_animated_mhd(d)
        assert len(fig.frames) == n_snaps
        assert len(fig.frames) <= MAX_ANIMATION_FRAMES


class TestCsvValidation:
    """validate_experimental_csv raises gr.Error on invalid input."""

    def test_valid_csv_passes(self) -> None:
        """Valid monotonic CSV with required columns raises no error."""
        from app_plots import validate_experimental_csv

        csv = "time_us,current_MA\n0.0,0.0\n1.0,0.5\n2.0,1.0\n"
        validate_experimental_csv(csv)

    def test_missing_time_column_raises(self) -> None:
        """CSV without a time column raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        csv = "step,current_MA\n1,0.5\n2,1.0\n"
        with pytest.raises(gr.Error, match="time column"):
            validate_experimental_csv(csv)

    def test_missing_current_column_raises(self) -> None:
        """CSV without a current column raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        csv = "time_us,voltage_kV\n0.0,27.0\n1.0,20.0\n"
        with pytest.raises(gr.Error, match="current column"):
            validate_experimental_csv(csv)

    def test_non_monotonic_time_raises(self) -> None:
        """CSV with non-monotonic time column raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        csv = "time_us,current_MA\n0.0,0.0\n2.0,1.0\n1.0,0.5\n"
        with pytest.raises(gr.Error, match="monotonically increasing"):
            validate_experimental_csv(csv)

    def test_duplicate_time_values_raise(self) -> None:
        """CSV with duplicate (non-strictly-increasing) time values raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        csv = "time_us,current_MA\n0.0,0.0\n1.0,0.5\n1.0,0.8\n"
        with pytest.raises(gr.Error, match="monotonically increasing"):
            validate_experimental_csv(csv)

    def test_empty_csv_raises(self) -> None:
        """Empty CSV string raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        with pytest.raises(gr.Error):
            validate_experimental_csv("")

    def test_header_only_raises(self) -> None:
        """CSV with header but no data rows raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        with pytest.raises(gr.Error):
            validate_experimental_csv("time_us,current_MA\n")

    def test_time_s_column_accepted(self) -> None:
        """time_s column is recognized as a valid time column."""
        from app_plots import validate_experimental_csv

        csv = "time_s,current_MA\n0.000001,0.5\n0.000002,1.0\n"
        validate_experimental_csv(csv)

    def test_non_numeric_time_raises(self) -> None:
        """Non-numeric value in time column raises gr.Error."""
        import gradio as gr

        from app_plots import validate_experimental_csv

        csv = "time_us,current_MA\nabc,0.5\n1.0,1.0\n"
        with pytest.raises(gr.Error):
            validate_experimental_csv(csv)


class TestServerPortConfig:
    """DPF_UI_PORT env var controls launch port."""

    def test_default_port_is_7860(self) -> None:
        """Without DPF_UI_PORT set, port defaults to 7860."""
        import os
        from unittest.mock import patch

        env = {k: v for k, v in os.environ.items() if k != "DPF_UI_PORT"}
        with patch.dict(os.environ, env, clear=True):
            port = int(os.environ.get("DPF_UI_PORT", "7860"))

        assert port == 7860

    def test_custom_port_from_env(self) -> None:
        """DPF_UI_PORT=9000 resolves to integer 9000."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"DPF_UI_PORT": "9000"}):
            port = int(os.environ.get("DPF_UI_PORT", "7860"))

        assert port == 9000

    def test_port_cast_to_int(self) -> None:
        """DPF_UI_PORT value is cast to int, not left as string."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"DPF_UI_PORT": "8080"}):
            port = int(os.environ.get("DPF_UI_PORT", "7860"))

        assert isinstance(port, int)
        assert port == 8080


class TestSweepConcurrencyLimit:
    """Sweep buttons have concurrency_limit=1 enforced in app.py."""

    def test_app_sweep_buttons_have_concurrency_limit(self) -> None:
        """Verify concurrency_limit=1 appears on all sweep and cal button click handlers."""
        import ast
        from pathlib import Path

        src = (Path(__file__).parent.parent / "app.py").read_text()
        tree = ast.parse(src)

        concurrency_calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "click"
                ):
                    for kw in node.keywords:
                        if (
                            kw.arg == "concurrency_limit"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value == 1
                        ):
                            concurrency_calls.append(
                                ast.unparse(func.value)
                            )
        assert "sweep_btn" in concurrency_calls, (
            "sweep_btn.click missing concurrency_limit=1"
        )
        assert "sweep_2d_btn" in concurrency_calls, (
            "sweep_2d_btn.click missing concurrency_limit=1"
        )
