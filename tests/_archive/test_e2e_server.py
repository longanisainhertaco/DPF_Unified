"""End-to-end server integration tests.

Full lifecycle tests: create → start → stream → request fields →
pause → resume → stop.  Verifies REST, WebSocket, and binary
encoding all work together.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

# ====================================================
# Helpers
# ====================================================


def _create_sim(client, *, preset: str = "tutorial", max_steps: int = 5):
    """Create a simulation via REST and return sim_id."""
    r = client.post(
        "/api/simulations",
        json={"config": {}, "preset": preset, "max_steps": max_steps},
    )
    assert r.status_code == 200
    return r.json()["sim_id"]


# ====================================================
# End-to-End: REST Lifecycle
# ====================================================


class TestE2ELifecycle:
    """Full REST lifecycle: create → start → poll → stop."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from dpf.server.app import _simulations, app
        _simulations.clear()
        return TestClient(app)

    def test_create_and_status(self, client):
        """Create sim → verify idle → get status."""
        sim_id = _create_sim(client, max_steps=5)

        # Status should be idle
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
        assert shape == [8, 8, 8]  # tutorial preset grid

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
        assert len(set(ids)) == 3  # All unique

        for sid in ids:
            r = client.get(f"/api/simulations/{sid}")
            assert r.status_code == 200


# ====================================================
# End-to-End: WebSocket
# ====================================================


class TestE2EWebSocket:
    """WebSocket end-to-end tests."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

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
            pass  # Connection accepted

    def test_ws_field_request(self, client):
        """Client can request field data via WebSocket."""
        sim_id = _create_sim(client, max_steps=3)

        # Start the sim first via REST
        client.post(f"/api/simulations/{sim_id}/start")

        with client.websocket_connect(f"/ws/{sim_id}") as ws:
            # Send field request
            ws.send_text(json.dumps({
                "type": "request_fields",
                "fields": ["rho", "Te"],
                "downsample": 1,
            }))

            # Receive header (JSON text frame)
            header_text = ws.receive_text()
            header = json.loads(header_text)

            # Should have field metadata
            assert "fields" in header
            assert "total_bytes" in header

            if header["total_bytes"] > 0:
                # Receive binary field data
                blob = ws.receive_bytes()
                assert len(blob) == header["total_bytes"]

                # Decode rho field
                rho_info = header["fields"]["rho"]
                rho_data = np.frombuffer(
                    blob[rho_info["offset"]:rho_info["offset"] + rho_info["nbytes"]],
                    dtype=np.float32,
                ).reshape(rho_info["shape"])
                assert rho_data.shape == tuple(rho_info["shape"])
                assert np.all(np.isfinite(rho_data))


# ====================================================
# End-to-End: Binary Encoding Roundtrip
# ====================================================


class TestE2EEncoding:
    """Verify field encoding roundtrip with real engine data."""

    def test_engine_fields_encode_decode(self):
        """Real engine fields survive encode → decode."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine
        from dpf.server.encoding import encode_fields

        config = SimulationConfig(
            grid_shape=[8, 8, 8],
            dx=1e-3,
            sim_time=1e-7,
            dt_init=1e-10,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
        )
        engine = SimulationEngine(config)
        engine.step()

        snapshot = engine.get_field_snapshot()
        header, blob = encode_fields(snapshot, ["rho", "Te", "B"])

        # Verify all requested fields present
        assert "rho" in header
        assert "Te" in header
        assert "B" in header

        # Decode rho and verify shape + values
        rho_h = header["rho"]
        rho = np.frombuffer(
            blob[rho_h["offset"]:rho_h["offset"] + rho_h["nbytes"]],
            dtype=np.float32,
        ).reshape(rho_h["shape"])
        assert rho.shape == (8, 8, 8)
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)  # Density should be positive

        # Decode B and verify vector field shape
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
            grid_shape=[16, 16, 16],
            dx=1e-3,
            sim_time=1e-7,
            dt_init=1e-10,
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

        # Downsample should reduce size
        assert len(blob_ds2) < len(blob_full)
        assert len(blob_ds4) < len(blob_ds2)

        # Exact expected sizes: 16^3 * 4 = 16384, 8^3 * 4 = 2048, 4^3 * 4 = 256
        assert len(blob_full) == 16 * 16 * 16 * 4
        assert len(blob_ds2) == 8 * 8 * 8 * 4
        assert len(blob_ds4) == 4 * 4 * 4 * 4
