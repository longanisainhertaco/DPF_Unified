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
