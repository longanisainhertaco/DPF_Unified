"""FastAPI application — REST + WebSocket endpoints for DPF simulations.

REST endpoints:
    POST   /api/simulations              Create a simulation
    GET    /api/simulations/{id}         Get status
    POST   /api/simulations/{id}/start   Begin running
    POST   /api/simulations/{id}/pause   Pause
    POST   /api/simulations/{id}/resume  Resume
    POST   /api/simulations/{id}/stop    Stop
    GET    /api/simulations/{id}/fields  Binary field data
    GET    /api/config/schema            JSON Schema
    POST   /api/config/validate          Validate config
    GET    /api/presets                  List named presets
    GET    /api/health                   Health check

WebSocket:
    WS     /ws/{sim_id}                 Scalar streaming + field requests
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from dpf.config import SimulationConfig
from dpf.presets import get_preset, list_presets
from dpf.server.encoding import encode_fields
from dpf.server.models import (
    ConfigValidationResponse,
    CreateSimulationRequest,
    FieldHeader,
    FieldRequest,
    PresetInfo,
    ScalarUpdate,
    SimulationInfo,
)
from dpf.server.simulation import SimulationManager

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DPF Simulation Server",
    description="Dense Plasma Focus simulator — REST + WebSocket API for Unity GUI",
    version="0.1.0",
)

# Allow Unity to connect from any origin (localhost development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory simulation registry (single-process)
_simulations: dict[str, SimulationManager] = {}


# ── Helpers ──────────────────────────────────────────────────────


def _get_sim(sim_id: str) -> SimulationManager:
    if sim_id not in _simulations:
        raise HTTPException(status_code=404, detail=f"Simulation '{sim_id}' not found")
    return _simulations[sim_id]


# ── Health ───────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Health check with backend availability info."""
    from dpf.athena_wrapper import is_available as athena_available

    return {
        "status": "ok",
        "backends": {
            "python": True,
            "athena": athena_available(),
        },
    }


# ── Simulation CRUD ──────────────────────────────────────────────


@app.post("/api/simulations", response_model=SimulationInfo)
async def create_simulation(req: CreateSimulationRequest) -> SimulationInfo:
    """Create a new simulation from config or preset."""
    try:
        if req.preset:
            preset_data = get_preset(req.preset)
            config = SimulationConfig(**preset_data)
        else:
            config = SimulationConfig(**req.config)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mgr = SimulationManager(config, max_steps=req.max_steps)
    mgr.create_engine()
    _simulations[mgr.sim_id] = mgr
    logger.info("Created simulation %s", mgr.sim_id)
    return SimulationInfo(**mgr.info())


@app.get("/api/simulations/{sim_id}", response_model=SimulationInfo)
async def get_simulation(sim_id: str) -> SimulationInfo:
    mgr = _get_sim(sim_id)
    return SimulationInfo(**mgr.info())


@app.post("/api/simulations/{sim_id}/start", response_model=SimulationInfo)
async def start_simulation(sim_id: str) -> SimulationInfo:
    mgr = _get_sim(sim_id)
    await mgr.start()
    return SimulationInfo(**mgr.info())


@app.post("/api/simulations/{sim_id}/pause", response_model=SimulationInfo)
async def pause_simulation(sim_id: str) -> SimulationInfo:
    mgr = _get_sim(sim_id)
    await mgr.pause()
    return SimulationInfo(**mgr.info())


@app.post("/api/simulations/{sim_id}/resume", response_model=SimulationInfo)
async def resume_simulation(sim_id: str) -> SimulationInfo:
    mgr = _get_sim(sim_id)
    await mgr.resume()
    return SimulationInfo(**mgr.info())


@app.post("/api/simulations/{sim_id}/stop", response_model=SimulationInfo)
async def stop_simulation(sim_id: str) -> SimulationInfo:
    mgr = _get_sim(sim_id)
    await mgr.stop()
    return SimulationInfo(**mgr.info())


@app.get("/api/simulations/{sim_id}/fields")
async def get_fields(
    sim_id: str,
    fields: str = "rho,Te",
    downsample: int = 1,
) -> dict[str, Any]:
    """Return field snapshot metadata (binary transfer is via WebSocket)."""
    mgr = _get_sim(sim_id)
    snapshot = mgr.get_field_snapshot()
    if not snapshot:
        raise HTTPException(status_code=409, detail="No field data available")

    field_names = [f.strip() for f in fields.split(",")]
    header, blob = encode_fields(snapshot, field_names, downsample)
    # For REST, return metadata only (binary via WS)
    return {
        "sim_id": sim_id,
        "fields": header,
        "total_bytes": len(blob),
        "note": "Use WebSocket /ws/{sim_id} to retrieve binary field data",
    }


# ── Config / Presets ─────────────────────────────────────────────


@app.get("/api/config/schema")
async def config_schema() -> dict[str, Any]:
    """Return the JSON Schema for SimulationConfig."""
    return SimulationConfig.model_json_schema()


@app.post("/api/config/validate", response_model=ConfigValidationResponse)
async def validate_config(config: dict[str, Any]) -> ConfigValidationResponse:
    """Validate a config dict without running a simulation."""
    try:
        SimulationConfig(**config)
        return ConfigValidationResponse(valid=True)
    except Exception as exc:
        return ConfigValidationResponse(valid=False, errors=[str(exc)])


@app.get("/api/presets", response_model=list[PresetInfo])
async def get_presets() -> list[PresetInfo]:
    return [PresetInfo(**p) for p in list_presets()]


# ── WebSocket ────────────────────────────────────────────────────


@app.websocket("/ws/{sim_id}")
async def websocket_endpoint(websocket: WebSocket, sim_id: str) -> None:
    """WebSocket endpoint for real-time simulation streaming.

    - Server sends JSON ScalarUpdate each step.
    - Client can send FieldRequest JSON to get binary field data back.
    """
    if sim_id not in _simulations:
        await websocket.close(code=4004, reason="Simulation not found")
        return

    mgr = _simulations[sim_id]
    await websocket.accept()
    logger.info("WS client connected to sim %s", sim_id)

    # Subscribe to step results
    queue = mgr.subscribe()

    async def _send_scalars() -> None:
        """Forward step results from queue to WebSocket."""
        try:
            while True:
                result = await queue.get()
                update = ScalarUpdate(
                    step=result.step,
                    time=result.time,
                    dt=result.dt,
                    current=result.current,
                    voltage=result.voltage,
                    energy_conservation=result.energy_conservation,
                    max_Te=result.max_Te,
                    max_rho=result.max_rho,
                    Z_bar=result.Z_bar,
                    R_plasma=result.R_plasma,
                    eta_anomalous=result.eta_anomalous,
                    total_radiated_energy=result.total_radiated_energy,
                    neutron_rate=result.neutron_rate,
                    total_neutron_yield=result.total_neutron_yield,
                    finished=result.finished,
                )
                await websocket.send_text(update.model_dump_json())
                if result.finished:
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WS send error for sim %s", sim_id)

    async def _receive_commands() -> None:
        """Listen for client field requests."""
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "request_fields":
                    req = FieldRequest(**msg)
                    snapshot = mgr.get_field_snapshot()
                    if snapshot:
                        header, blob = encode_fields(
                            snapshot, req.fields, req.downsample,
                        )
                        # Send header as JSON text frame
                        fh = FieldHeader(
                            fields=header,
                            total_bytes=len(blob),
                        )
                        await websocket.send_text(fh.model_dump_json())
                        # Send binary data
                        await websocket.send_bytes(blob)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("WS receive error for sim %s", sim_id)

    # Run send and receive concurrently
    try:
        await asyncio.gather(_send_scalars(), _receive_commands())
    finally:
        mgr.unsubscribe(queue)
        logger.info("WS client disconnected from sim %s", sim_id)
