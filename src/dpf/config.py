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
    line_radiation_enabled: bool = Field(
        False,
        description="Enable impurity line + recombination radiation cooling",
    )
    impurity_Z: float = Field(
        29.0, ge=1, le=74,
        description="Atomic number of dominant impurity (default 29 = copper)",
    )
    impurity_fraction: float = Field(
        0.0, ge=0, le=1.0,
        description="Impurity number density as fraction of ne (e.g. 0.01 = 1%)",
    )


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


class BoundaryConfig(BaseModel):
    """Electrode and domain boundary conditions."""

    electrode_bc: bool = Field(
        False,
        description="Apply electrode B-field BC: B_theta = mu0*I/(2*pi*r) at electrode radii",
    )
    axis_bc: bool = Field(
        True,
        description="Enforce symmetry BCs at r=0 axis (cylindrical only): B_r=0, dB_z/dr=0",
    )


class FluidConfig(BaseModel):
    """Fluid / MHD solver parameters."""

    backend: str = Field(
        "python",
        description=(
            "MHD solver backend: 'python' (NumPy/Numba), 'athena' (Athena++ C++), "
            "'metal' (PyTorch MPS on Apple GPU), "
            "or 'auto' (Athena++ if available, else Python)"
        ),
    )
    reconstruction: str = Field("weno5", description="Reconstruction scheme")
    riemann_solver: str = Field("hlld", description="Riemann solver type (Phase P default: HLLD)")
    cfl: float = Field(0.4, gt=0, lt=1, description="CFL number")
    dedner_ch: float = Field(0.0, ge=0, description="Dedner cleaning speed (0 = auto)")
    gamma: float = Field(5.0 / 3.0, gt=1, description="Adiabatic index")
    enable_resistive: bool = Field(True, description="Enable resistive MHD (eta*J in induction)")
    enable_energy_equation: bool = Field(True, description="Use conservative total energy equation")
    enable_nernst: bool = Field(False, description="Enable Nernst B-field advection by grad(Te)")
    enable_viscosity: bool = Field(False, description="Enable Braginskii ion viscosity (eta_0)")
    diffusion_method: str = Field(
        "explicit",
        description=(
            "Diffusion treatment: 'explicit' (standard CFL-limited), "
            "'sts' (super time-stepping RKL2), 'implicit' (Crank-Nicolson ADI)"
        ),
    )
    sts_stages: int = Field(8, ge=2, le=32, description="RKL2 super time-stepping stages")
    implicit_tol: float = Field(1e-8, gt=0, description="Implicit diffusion solver tolerance")
    enable_powell: bool = Field(False, description="Enable Powell 8-wave div(B) source terms")
    dedner_cr: float = Field(0.0, ge=0, description="Dedner damping rate (0 = auto)")
    enable_anisotropic_conduction: bool = Field(
        False, description="Enable field-aligned Braginskii thermal conduction"
    )
    full_braginskii_viscosity: bool = Field(
        False, description="Enable full Braginskii viscosity (eta_0 + eta_1 + eta_3)"
    )
    time_integrator: str = Field(
        "ssp_rk3",
        description="Time integrator: 'ssp_rk2' (2nd-order SSP), 'ssp_rk3' (3rd-order SSP, Phase P default)",
    )
    precision: str = Field(
        "float32",
        description="Floating-point precision: 'float32' (fast, GPU) or 'float64' (accurate, CPU only)",
    )
    use_ct: bool = Field(
        False,
        description="Use Constrained Transport for div(B)=0 (Metal GPU only, requires MPS device)",
    )

    @model_validator(mode="after")
    def validate_backend(self) -> FluidConfig:
        if self.backend not in ("python", "athena", "athenak", "metal", "auto"):
            raise ValueError(
                f"backend must be 'python', 'athena', 'athenak', 'metal', or 'auto', got '{self.backend}'"
            )
        return self


class DiagnosticsConfig(BaseModel):
    """Diagnostics output parameters."""

    hdf5_filename: str = Field("diagnostics.h5", description="Output HDF5 file")
    output_interval: int = Field(10, gt=0, description="Steps between outputs")
    field_output_interval: int = Field(
        0, ge=0,
        description="Steps between field snapshots in HDF5 (0 = off)",
    )


class SweepConfig(BaseModel):
    """Parameter sweep configuration for batch trajectory generation."""

    method: str = Field(
        "lhs",
        description="Sampling method: 'lhs' (Latin Hypercube), 'grid', 'random'",
    )
    n_samples: int = Field(100, ge=1, le=10000, description="Number of parameter samples")
    parameter_ranges: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Parameter ranges: {'circuit.V0': {'low': 10e3, 'high': 40e3}}",
    )
    resolution: list[int] = Field(
        default_factory=lambda: [64, 1, 128],
        min_length=3,
        max_length=3,
        description="Grid resolution for sweep trajectories",
    )

    @model_validator(mode="after")
    def validate_method(self) -> SweepConfig:
        if self.method not in ("lhs", "grid", "random"):
            raise ValueError(f"method must be 'lhs', 'grid', or 'random', got '{self.method}'")
        return self


class InverseConfig(BaseModel):
    """Inverse design optimization configuration."""

    method: str = Field(
        "bayesian",
        description="Optimization method: 'bayesian' or 'evolutionary'",
    )
    targets: dict[str, float] = Field(
        default_factory=dict,
        description="Target outputs: {'neutron_yield': 1e10, 'I_peak': 2e6}",
    )
    constraints: dict[str, float] = Field(
        default_factory=dict,
        description="Constraints: {'V0_max': 30e3, 'C_max': 50e-6}",
    )
    n_trials: int = Field(100, ge=1, le=10000, description="Number of optimization trials")

    @model_validator(mode="after")
    def validate_method(self) -> InverseConfig:
        if self.method not in ("bayesian", "evolutionary"):
            raise ValueError(
                f"method must be 'bayesian' or 'evolutionary', got '{self.method}'"
            )
        return self


class AIConfig(BaseModel):
    """AI/ML surrogate model configuration."""

    surrogate_checkpoint: str | None = Field(
        None, description="Path to fine-tuned WALRUS checkpoint"
    )
    device: str = Field("cpu", description="Inference device: 'cpu', 'mps', 'cuda'")
    history_length: int = Field(
        4, ge=1, le=32, description="Number of history timesteps for WALRUS input"
    )
    ensemble_size: int = Field(
        1, ge=1, le=10, description="Number of ensemble models for uncertainty estimation"
    )
    confidence_threshold: float = Field(
        0.8, gt=0, le=1.0, description="Minimum confidence for predictions"
    )
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    inverse: InverseConfig = Field(default_factory=InverseConfig)

    @model_validator(mode="after")
    def validate_device(self) -> AIConfig:
        if self.device not in ("cpu", "mps", "cuda"):
            raise ValueError(f"device must be 'cpu', 'mps', or 'cuda', got '{self.device}'")
        return self


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
    ion_mass: float = Field(
        3.34358377e-27, gt=0,
        description="Ion mass [kg] (default: deuterium m_d = 3.34e-27 kg)",
    )

    circuit: CircuitConfig
    collision: CollisionConfig = Field(default_factory=CollisionConfig)
    radiation: RadiationConfig = Field(default_factory=RadiationConfig)
    sheath: SheathConfig = Field(default_factory=SheathConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    fluid: FluidConfig = Field(default_factory=FluidConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    ai: AIConfig | None = Field(None, description="AI/ML surrogate configuration (optional)")

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
