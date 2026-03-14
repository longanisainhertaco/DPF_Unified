"""Tests for Phase 8: Server API, WebSocket streaming, encoding, presets.

Test categories:
1. Binary field encoding (encode_fields, downsample)
2. Configuration presets (list, get, create SimulationConfig)
3. REST endpoints (health, create, status, config schema/validate, presets)
4. WebSocket streaming (scalar updates, field requests)
5. Simulation lifecycle (start, pause, resume, stop)
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

# ====================================================
# Binary Field Encoding Tests
# ====================================================


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
        # Factor 2 -> 8^3 = 512 floats * 4 bytes = 2048
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


# ====================================================
# Preset Tests
# ====================================================


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
        assert "_meta" not in data  # _meta stripped

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


# ====================================================
# REST Endpoint Tests (FastAPI TestClient)
# ====================================================


class TestRESTEndpoints:
    """Tests for REST API endpoints."""

    @pytest.fixture()
    def client(self):
        """Create a FastAPI test client."""
        from fastapi.testclient import TestClient

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
            "grid_shape": [4, 4, 4],
            "dx": 1e-3,
            "sim_time": 1e-7,
            "dt_init": 1e-10,
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
        # Create first
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
            "grid_shape": [4, 4, 4],
            "dx": 1e-3,
            "sim_time": 1e-7,
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


# ====================================================
# Simulation Lifecycle Tests
# ====================================================


class TestSimulationLifecycle:
    """Tests for start/pause/resume/stop via REST."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def _create(self, client):
        """Helper: create a simulation and return sim_id."""
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
        # It will either be running or already finished (only 5 steps)
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
        # Should return field metadata (engine is created even before start)
        assert r.status_code == 200


# ====================================================
# SimulationManager Unit Tests
# ====================================================


class TestSimulationManager:
    """Direct unit tests for SimulationManager."""

    def _make_config(self):
        from dpf.config import SimulationConfig

        return SimulationConfig(
            grid_shape=[4, 4, 4],
            dx=1e-3,
            sim_time=1e-7,
            dt_init=1e-10,
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

        # Wait for completion
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

        assert len(results) >= 3  # At least 3 steps + possible sentinel
        mgr.unsubscribe(q)


# ====================================================
# WebSocket Tests
# ====================================================


class TestWebSocket:
    """Tests for WebSocket streaming endpoint."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

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

        # Just verify we can connect (don't start to avoid async timing issues)
        with client.websocket_connect(f"/ws/{sim_id}"):
            pass  # Connection accepted successfully
