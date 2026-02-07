"""Pydantic v2 configuration system for DPF simulations.

Provides validated, typed configuration with submodels for each physics
component. Supports JSON/YAML I/O and cross-field validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class CircuitConfig(BaseModel):
    """RLC circuit parameters."""

    C: float = Field(..., gt=0, description="Capacitance [F]")
    V0: float = Field(..., gt=0, description="Initial voltage [V]")
    L0: float = Field(..., gt=0, description="External inductance [H]")
    R0: float = Field(0.0, ge=0, description="External resistance [Ohm]")
    anode_radius: float = Field(..., gt=0, description="Anode radius [m]")
    cathode_radius: float = Field(..., gt=0, description="Cathode radius [m]")
    ESR: float = Field(0.0, ge=0, description="Equivalent series resistance [Ohm]")
    ESL: float = Field(0.0, ge=0, description="Equivalent series inductance [H]")

    @model_validator(mode="after")
    def check_radii(self) -> CircuitConfig:
        if self.anode_radius >= self.cathode_radius:
            raise ValueError("anode_radius must be less than cathode_radius")
        return self


class CollisionConfig(BaseModel):
    """Collision model parameters."""

    coulomb_log: float = Field(10.0, gt=0, description="Coulomb logarithm (fixed or initial)")
    dynamic_coulomb_log: bool = Field(True, description="Compute Coulomb log dynamically")
    sigma_en: float = Field(1e-19, gt=0, description="Electron-neutral cross section [m^2]")


class RadiationConfig(BaseModel):
    """Radiation transport parameters."""

    bremsstrahlung_enabled: bool = Field(True, description="Enable bremsstrahlung")
    gaunt_factor: float = Field(1.2, gt=0, description="Gaunt factor")
    fld_enabled: bool = Field(False, description="Enable flux-limited diffusion")
    flux_limiter: float = Field(1.0 / 3.0, gt=0, le=1.0, description="Flux limiter lambda")


class SheathConfig(BaseModel):
    """Plasma sheath boundary condition parameters."""

    enabled: bool = Field(False, description="Enable sheath BCs at electrodes")
    boundary: str = Field("z_high", description="Boundary to apply sheath ('z_high', 'z_low')")
    V_sheath: float = Field(0.0, ge=0, description="Sheath voltage drop [V] (0 = auto from Te)")


class GeometryConfig(BaseModel):
    """Coordinate system configuration."""

    type: str = Field(
        "cartesian",
        description="Coordinate system: 'cartesian' (3D) or 'cylindrical' (2D axisymmetric r,z)",
    )
    dz: float | None = Field(
        None, gt=0,
        description="Axial grid spacing [m] (cylindrical only; defaults to dx if not set)",
    )

    @model_validator(mode="after")
    def validate_type(self) -> GeometryConfig:
        if self.type not in ("cartesian", "cylindrical"):
            raise ValueError(f"geometry type must be 'cartesian' or 'cylindrical', got '{self.type}'")
        return self


class FluidConfig(BaseModel):
    """Fluid / MHD solver parameters."""

    reconstruction: str = Field("weno5", description="Reconstruction scheme")
    riemann_solver: str = Field("hll", description="Riemann solver type")
    cfl: float = Field(0.4, gt=0, lt=1, description="CFL number")
    dedner_ch: float = Field(0.0, ge=0, description="Dedner cleaning speed (0 = auto)")
    gamma: float = Field(5.0 / 3.0, gt=1, description="Adiabatic index")


class DiagnosticsConfig(BaseModel):
    """Diagnostics output parameters."""

    hdf5_filename: str = Field("diagnostics.h5", description="Output HDF5 file")
    output_interval: int = Field(10, gt=0, description="Steps between outputs")
    field_output_interval: int = Field(
        0, ge=0,
        description="Steps between field snapshots in HDF5 (0 = off)",
    )


class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""

    grid_shape: list[int] = Field(..., min_length=3, max_length=3, description="Grid (nx, ny, nz)")
    dx: float = Field(..., gt=0, description="Grid spacing [m]")
    sim_time: float = Field(..., gt=0, description="Total simulation time [s]")
    dt_init: float | None = Field(None, gt=0, description="Initial timestep [s]")

    # Initial conditions (exposed for GUI / parameter sweeps)
    rho0: float = Field(1e-4, gt=0, description="Initial fill gas density [kg/m^3]")
    T0: float = Field(300.0, gt=0, description="Initial temperature [K]")
    anomalous_alpha: float = Field(
        0.05, ge=0, le=1.0,
        description="Buneman anomalous resistivity alpha parameter",
    )

    circuit: CircuitConfig
    collision: CollisionConfig = Field(default_factory=CollisionConfig)
    radiation: RadiationConfig = Field(default_factory=RadiationConfig)
    sheath: SheathConfig = Field(default_factory=SheathConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    fluid: FluidConfig = Field(default_factory=FluidConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    @model_validator(mode="after")
    def validate_grid(self) -> SimulationConfig:
        if any(n <= 0 for n in self.grid_shape):
            raise ValueError("grid_shape values must be positive integers")
        if self.geometry.type == "cylindrical" and self.grid_shape[1] != 1:
            # For cylindrical, grid_shape is [nr, 1, nz] — ny must be 1
            raise ValueError(
                f"cylindrical geometry requires grid_shape[1]=1 (axisymmetric), "
                f"got {self.grid_shape[1]}"
            )
        return self

    # --- I/O helpers ---

    @classmethod
    def from_file(cls, path: str | Path) -> SimulationConfig:
        """Load configuration from a JSON file."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize to JSON string, optionally writing to file."""
        out = self.model_dump_json(indent=2)
        if path is not None:
            Path(path).write_text(out)
        return out
