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
from pydantic import BaseModel

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

# AI router — optional, loaded only if dpf.ai is available
try:
    from dpf.ai.realtime_server import ai_router

    _HAS_AI_ROUTER = True
except ImportError:
    _HAS_AI_ROUTER = False

app = FastAPI(
    title="DPF Simulation Server",
    description="Dense Plasma Focus simulator — REST + WebSocket API for Unity GUI",
    version="0.1.0",
)

if _HAS_AI_ROUTER:
    app.include_router(ai_router)

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
    from dpf.athenak_wrapper import is_available as athenak_available

    def _metal_available() -> bool:
        try:
            from dpf.metal.metal_solver import MetalMHDSolver

            return MetalMHDSolver.is_available()
        except Exception:
            return False

    return {
        "status": "ok",
        "backends": {
            "python": True,
            "athena": athena_available(),
            "athenak": athenak_available(),
            "metal": _metal_available(),
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


# ── Thomson Scattering Diagnostic ────────────────────────────────


class ThomsonRequest(BaseModel):
    rho: list[float]
    Te_eV: list[float]
    Ti_eV: list[float] | None = None
    v_bulk: list[float] | None = None
    laser_wavelength: float = 1064e-9
    scattering_angle: float = 1.5707963267948966  # pi/2
    chord_positions: list[float] | None = None
    n_wavelength_points: int = 256


class ThomsonResponse(BaseModel):
    wavelength_nm: list[float]
    spectra: list[list[float]]  # shape (N_chords, N_wavelength)
    chord_positions_m: list[float]
    laser_wavelength_nm: float
    scattering_angle_rad: float


@app.post("/api/thomson", response_model=ThomsonResponse)
async def thomson_diagnostic(req: ThomsonRequest) -> ThomsonResponse:
    """Compute synthetic Thomson scattering spectra from simulation state.

    Accepts plasma parameters along a 1-D chord and returns the scattered
    power spectrum at each chord position using the full Salpeter form factor.
    """
    try:
        import numpy as np  # noqa: I001, PLC0415
        from dpf.diagnostics.thomson_scattering import thomson_spectrum  # noqa: I001, PLC0415
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Thomson module unavailable: {exc}") from exc

    ne_arr = np.array(req.rho, dtype=np.float64)
    Te_arr = np.array(req.Te_eV, dtype=np.float64)
    Ti_arr = np.array(req.Ti_eV, dtype=np.float64) if req.Ti_eV else Te_arr.copy()
    v_arr = np.array(req.v_bulk, dtype=np.float64) if req.v_bulk else np.zeros_like(ne_arr)

    if not (len(ne_arr) == len(Te_arr) == len(Ti_arr) == len(v_arr)):
        raise HTTPException(
            status_code=422,
            detail="rho, Te_eV, Ti_eV, and v_bulk must all have the same length",
        )

    lambda0 = req.laser_wavelength
    delta_max = 50e-9  # +/- 50 nm window around laser wavelength
    wl_grid = np.linspace(lambda0 - delta_max, lambda0 + delta_max, req.n_wavelength_points)

    spectra = thomson_spectrum(
        ne=ne_arr,
        Te_eV=Te_arr,
        v_bulk=v_arr,
        wavelength_grid=wl_grid,
        Ti_eV=Ti_arr,
        scattering_angle=req.scattering_angle,
        laser_wavelength=lambda0,
    )

    chord_positions = req.chord_positions if req.chord_positions else list(range(len(ne_arr)))

    return ThomsonResponse(
        wavelength_nm=(wl_grid * 1e9).tolist(),
        spectra=spectra.tolist(),
        chord_positions_m=chord_positions,
        laser_wavelength_nm=lambda0 * 1e9,
        scattering_angle_rad=req.scattering_angle,
    )


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
