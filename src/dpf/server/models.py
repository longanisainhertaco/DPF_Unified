"""Pydantic models for the DPF server API.

Defines request/response schemas for REST endpoints and WebSocket messages.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


class SimulationStatus(str, enum.Enum):
    """Lifecycle state of a simulation."""

    idle = "idle"
    running = "running"
    paused = "paused"
    finished = "finished"
    error = "error"


# ── REST request / response models ──────────────────────────────────


class CreateSimulationRequest(BaseModel):
    """Request to create a new simulation."""

    config: dict[str, Any] = Field(..., description="Full SimulationConfig as JSON dict")
    max_steps: int | None = Field(None, ge=1, description="Optional step limit")
    preset: str | None = Field(None, description="Named preset (overrides config if set)")


class SimulationInfo(BaseModel):
    """Status snapshot returned by GET /api/simulations/{id}."""

    sim_id: str
    status: SimulationStatus
    backend: str = "python"
    step: int = 0
    time: float = 0.0
    current: float = 0.0
    voltage: float = 0.0
    energy_conservation: float = 1.0
    max_Te: float = 0.0
    max_rho: float = 0.0
    total_radiated_energy: float = 0.0
    error_message: str | None = None


class ConfigValidationResponse(BaseModel):
    """Result of POST /api/config/validate."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class PresetInfo(BaseModel):
    """Summary of a named configuration preset."""

    name: str
    description: str
    device: str
    geometry: str
    grid_shape: list[int]


# ── WebSocket message models ────────────────────────────────────────


class ScalarUpdate(BaseModel):
    """Per-step scalar data streamed over WebSocket (JSON text frame)."""

    type: str = "scalar"
    step: int
    time: float
    dt: float
    current: float
    voltage: float
    energy_conservation: float
    max_Te: float
    max_rho: float
    Z_bar: float
    R_plasma: float
    eta_anomalous: float
    total_radiated_energy: float
    neutron_rate: float
    total_neutron_yield: float
    finished: bool


class FieldRequest(BaseModel):
    """Client request for field data (sent as JSON text frame on WS)."""

    type: str = "request_fields"
    fields: list[str] = Field(
        default_factory=lambda: ["rho", "Te"],
        description="Field names to retrieve",
    )
    downsample: int = Field(1, ge=1, le=8, description="Spatial downsample factor")


class FieldHeader(BaseModel):
    """Header preceding a binary field frame on WebSocket."""

    type: str = "field_header"
    fields: dict[str, dict[str, Any]] = Field(
        ..., description="Per-field metadata: shape, dtype, offset, nbytes"
    )
    total_bytes: int


# ── AI request / response models ───────────────────────────────────


class PredictRequest(BaseModel):
    """Request for single next-step AI prediction."""

    type: str = "predict"
    history: list[dict[str, Any]] = Field(
        ..., description="List of state dicts (arrays as nested lists)"
    )


class PredictResponse(BaseModel):
    """Response with predicted state."""

    type: str = "prediction"
    predicted_state: dict[str, Any] = Field(
        ..., description="Predicted state dict (arrays as nested lists)"
    )
    inference_time_ms: float = 0.0


class RolloutRequest(BaseModel):
    """Request for multi-step rollout."""

    type: str = "rollout"
    initial_states: list[dict[str, Any]] = Field(
        ..., description="Initial history states"
    )
    n_steps: int = Field(10, ge=1, le=1000, description="Number of rollout steps")


class RolloutResponse(BaseModel):
    """Response with rollout trajectory."""

    type: str = "rollout_result"
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    n_steps: int = 0
    total_inference_time_ms: float = 0.0


class SweepRequest(BaseModel):
    """Request for parameter sweep."""

    type: str = "sweep"
    configs: list[dict[str, Any]] = Field(
        ..., description="List of config parameter dicts"
    )
    n_steps: int = Field(100, ge=1, description="Rollout steps per config")


class SweepResponse(BaseModel):
    """Response with sweep results."""

    type: str = "sweep_result"
    results: list[dict[str, Any]] = Field(default_factory=list)
    n_configs: int = 0


class InverseRequest(BaseModel):
    """Request for inverse design."""

    type: str = "inverse"
    targets: dict[str, float] = Field(..., description="Target outputs")
    constraints: dict[str, float] = Field(default_factory=dict)
    method: str = Field("bayesian", description="Optimization method")
    n_trials: int = Field(100, ge=1, le=10000)


class InverseResponse(BaseModel):
    """Response with inverse design results."""

    type: str = "inverse_result"
    best_params: dict[str, float] = Field(default_factory=dict)
    best_score: float = 0.0
    n_trials: int = 0


class ConfidenceResponse(BaseModel):
    """Response with prediction + confidence."""

    type: str = "confidence"
    predicted_state: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    ood_score: float = 0.0


class AIStatusResponse(BaseModel):
    """AI module status."""

    type: str = "ai_status"
    torch_available: bool = False
    model_loaded: bool = False
    device: str = "cpu"
    ensemble_size: int = 0
