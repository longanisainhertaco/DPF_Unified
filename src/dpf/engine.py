"""Simulation engine — orchestrates the DPF simulation loop.

Wires together: config -> circuit -> fluid -> collision -> radiation -> diagnostics
into a working timestep loop where every physics module is called each step.

This is the central coordination layer that ensures:
1. Circuit and plasma are properly coupled via CouplingState
2. Energy is tracked and conservation checked
3. Radiation losses are applied to electron energy
4. Diagnostics are recorded at the configured interval
5. The simulation terminates cleanly

Supports dual-engine architecture via ``config.fluid.backend``:
- ``"python"`` — NumPy/Numba MHD solver (default, full feature set)
- ``"athena"`` — Athena++ C++ MHD solver (10-100x faster, requires build)
- ``"auto"``   — Athena++ if available, else fallback to Python
"""

from __future__ import annotations

import contextlib
import logging
import time as wall_time
from typing import Any

import numpy as np

from dpf.atomic.ionization import saha_ionization_fraction_array
from dpf.circuit.rlc_solver import RLCSolver
from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures, spitzer_resistivity
from dpf.config import SimulationConfig
from dpf.constants import k_B, pi
from dpf.constants import mu_0 as _mu_0
from dpf.core.bases import CouplingState, StepResult

# FieldManager (Phase 5)
from dpf.core.field_manager import FieldManager
from dpf.diagnostics.checkpoint import load_checkpoint, save_checkpoint
from dpf.diagnostics.hdf5_writer import HDF5Writer
from dpf.diagnostics.interferometry import abel_transform, fringe_shift
from dpf.diagnostics.neutron_yield import neutron_yield_rate
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
from dpf.fluid.eos import IdealEOS
from dpf.fluid.implicit_diffusion import implicit_resistive_diffusion, implicit_thermal_diffusion
from dpf.fluid.mhd_solver import MHDSolver
from dpf.fluid.nernst import apply_nernst_advection
from dpf.fluid.snowplow import SnowplowModel
from dpf.fluid.super_time_step import rkl2_diffusion_3d, rkl2_thermal_step
from dpf.fluid.viscosity import (
    braginskii_eta0,
    braginskii_eta1,
    ion_collision_time,
    viscous_heating_rate,
    viscous_stress_rate,
)
from dpf.kinetic.manager import KineticManager
from dpf.radiation.bremsstrahlung import apply_bremsstrahlung_losses
from dpf.radiation.line_radiation import apply_line_radiation_losses
from dpf.radiation.transport import apply_radiation_transport
from dpf.sheath.bohm import apply_sheath_bc, floating_potential
from dpf.turbulence.anomalous import (
    anomalous_resistivity_field,
    anomalous_resistivity_scalar,
    total_resistivity_scalar,
)

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Dense Plasma Focus simulation engine.

    Orchestrates the coupled circuit-plasma simulation loop:
    1. Initialize fields from config
    2. Time loop:
       a. Compute CFL-limited dt
       b. Advance circuit (get current, voltage)
       c. Advance plasma (MHD step with circuit forcing)
       d. Apply collisions (temperature relaxation)
       e. Update coupling state
       f. Record diagnostics
    3. Finalize and write output

    Args:
        config: Validated SimulationConfig.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.time = 0.0
        self.step_count = 0

        # Initialize sub-solvers
        nx, ny, nz = config.grid_shape
        dx = config.dx

        # Circuit
        cc = config.circuit
        self.circuit = RLCSolver(
            C=cc.C,
            V0=cc.V0,
            L0=cc.L0,
            R0=cc.R0,
            ESR=cc.ESR,
            ESL=cc.ESL,
            anode_radius=cc.anode_radius,
            cathode_radius=cc.cathode_radius,
            crowbar_enabled=cc.crowbar_enabled,
            crowbar_mode=cc.crowbar_mode,
            crowbar_time=cc.crowbar_time,
            crowbar_resistance=cc.crowbar_resistance,
        )

        # Fluid / MHD — select solver based on geometry and backend
        fc = config.fluid
        self.geometry_type = config.geometry.type

        # Ion mass from config
        self.ion_mass = config.ion_mass

        # Boundary config
        self.boundary_cfg = config.boundary

        # Backend selection: "python", "athena", "athenak", "metal", "hybrid", or "auto"
        self.backend = self._resolve_backend(fc.backend)

        if self.backend == "hybrid":
            # Hybrid: Athena++ for physics phase, WALRUS surrogate for acceleration
            from dpf.athena_wrapper import AthenaPPSolver
            self.fluid = AthenaPPSolver(config)
            self._cell_volume = None
            self._hybrid_engine = None  # Lazily initialized in run()
            logger.info(
                "Using hybrid backend (Athena++ + WALRUS surrogate, "
                "handoff=%.0f%%)", fc.handoff_fraction * 100
            )
        elif self.backend == "athenak":
            from dpf.athenak_wrapper import AthenaKSolver
            self.fluid = AthenaKSolver(config)
            self._cell_volume = None
            logger.info("Using AthenaK backend (Kokkos)")
        elif self.backend == "athena":
            from dpf.athena_wrapper import AthenaPPSolver
            self.fluid = AthenaPPSolver(config)
            self._cell_volume = None
            logger.info("Using Athena++ backend (mode: %s)", self.fluid.mode)
        elif self.backend == "metal":
            from dpf.metal.metal_solver import MetalMHDSolver
            dz = config.geometry.dz if config.geometry.dz is not None else dx
            self.fluid = MetalMHDSolver(
                grid_shape=(nx, ny, nz),
                dx=dx,
                dz=dz,
                gamma=fc.gamma,
                cfl=fc.cfl,
                device="mps",
                limiter="minmod",
                use_ct=fc.use_ct,
                riemann_solver=fc.riemann_solver,
                reconstruction=fc.reconstruction,
                time_integrator=fc.time_integrator,
                precision=fc.precision,
                enable_hall=True,
                enable_braginskii_conduction=fc.enable_anisotropic_conduction,
                enable_braginskii_viscosity=fc.enable_viscosity,
                enable_nernst=fc.enable_nernst,
                enable_bremsstrahlung=getattr(
                    config.radiation, "bremsstrahlung_enabled", False
                ),
                ion_mass=self.ion_mass,
                coordinates=self.geometry_type,
            )
            # Attach cylindrical geometry provider for diagnostics
            if self.geometry_type == "cylindrical":
                from dpf.geometry.cylindrical import CylindricalGeometry
                self.fluid.geom = CylindricalGeometry(
                    nr=nx, nz=nz, dr=dx, dz=dz,
                )
            self._cell_volume = dx * dx * dz
            logger.info(
                "Using Metal GPU backend (PyTorch MPS, %s, %s+%s+%s)",
                self.geometry_type, fc.reconstruction,
                fc.riemann_solver, fc.time_integrator,
            )
        elif self.geometry_type == "cylindrical":
            dz = config.geometry.dz if config.geometry.dz is not None else dx
            self.fluid = CylindricalMHDSolver(
                nr=nx,
                nz=nz,
                dr=dx,
                dz=dz,
                gamma=fc.gamma,
                cfl=fc.cfl,
                dedner_ch=fc.dedner_ch,
                enable_hall=True,
                enable_resistive=fc.enable_resistive,
                enable_energy_equation=fc.enable_energy_equation,
                ion_mass=self.ion_mass,
                riemann_solver=fc.riemann_solver,
                conservative_energy=fc.conservative_energy,
            )
            # Cylindrical cell volumes from geometry: pi*(r_out^2-r_in^2)*dz
            # Expand (nr, nz) → (nr, 1, nz) for broadcast with 3D state arrays
            self._cell_volume = self.fluid.geom.cell_volumes()[:, np.newaxis, :]
            # Validate grid covers electrodes for cylindrical geometry
            r_max = nx * dx
            if r_max < cc.cathode_radius:
                logger.warning(
                    "Cylindrical grid r_max=%.3f m < cathode_radius=%.3f m. "
                    "Electrode BCs will degenerate (both map to outermost cell). "
                    "Increase nr or dx so r_max >= cathode_radius.",
                    r_max, cc.cathode_radius,
                )
            elif r_max < cc.cathode_radius * 1.05:
                logger.warning(
                    "Cylindrical grid r_max=%.3f m barely covers cathode_radius=%.3f m "
                    "(< 5%% margin). Consider increasing nr for proper boundary resolution.",
                    r_max, cc.cathode_radius,
                )
        else:
            self.fluid = MHDSolver(
                grid_shape=(nx, ny, nz),
                dx=dx,
                gamma=fc.gamma,
                cfl=fc.cfl,
                dedner_ch=fc.dedner_ch,
                enable_resistive=fc.enable_resistive,
                enable_energy_equation=fc.enable_energy_equation,
                ion_mass=self.ion_mass,
                riemann_solver=fc.riemann_solver,
                time_integrator=fc.time_integrator,
                use_ct=fc.use_ct,
            )
            self._cell_volume = dx**3  # Uniform Cartesian cell volume

        # EOS
        self.eos = IdealEOS(gamma=fc.gamma)

        # Diagnostics
        dc = config.diagnostics
        self.diagnostics = HDF5Writer(
            filename=dc.hdf5_filename,
            field_output_interval=dc.field_output_interval,
        )
        self.diag_interval = dc.output_interval

        # Well Exporter (Phase J.2)
        from pathlib import Path

        from dpf.io.well_exporter import WellExporter

        # Use same directory as HDF5 output
        out_dir = Path(dc.hdf5_filename).parent
        self.well_exporter = WellExporter(
            output_dir=out_dir,
            filename_prefix=dc.well_filename_prefix,
            enable=(dc.well_output_interval > 0),
        )
        self.well_interval = dc.well_output_interval

        # Initialize plasma state
        self.state = self._initial_state(nx, ny, nz)

        # Coupling
        self._coupling = CouplingState()
        self._prev_L_plasma: float = 0.0

        # Diagnostics tracking
        self._last_R_plasma: float = 0.0
        self._last_Z_bar: float = 0.0
        self._last_eta_anom: float = 0.0
        self._last_div_B: float = 0.0

        # Radiation config
        self.rad_cfg = config.radiation
        self.total_radiated_energy = 0.0

        # Sheath config
        self.sheath_cfg = config.sheath

        # Plasma column geometry for resistance estimate
        # Column length ~ anode length ~ cathode_radius (order of magnitude)
        self.column_length = cc.cathode_radius
        # Initial column cross-section ~ pi * anode_radius^2
        self.anode_radius = cc.anode_radius

        # Energy tracking
        self.initial_energy: float | None = None

        # Neutron yield tracking
        self.total_neutron_yield: float = 0.0
        self._last_neutron_rate: float = 0.0

        # Interferometry (cylindrical only)
        self._last_fringe_shifts: np.ndarray | None = None

        # Regime validity diagnostics (Phase AE)
        self._last_regime_result: dict | None = None

        # Source terms for coupling (e.g. J_kin from PIC)
        self._current_source_terms: dict[str, np.ndarray] | None = None

        # Ohmic heating gap tracking (circuit→plasma energy consistency)
        self._ohmic_gap_history: list[float] = []
        self._last_ohmic_gap: float = 0.0

        # Snowplow → MHD B-field coupling: one-shot initialization at radial entry
        self._radial_bfield_initialized: bool = False

        # Checkpoint settings
        self.checkpoint_interval: int = 0  # 0 = disabled
        self.checkpoint_filename: str = "checkpoint.h5"

        # Kinetic (PIC) Manager
        self.kinetic: KineticManager | None = None
        if config.kinetic.enabled:
            self.kinetic = KineticManager(config)

        # Snowplow dynamics (Phase S)
        self.snowplow: SnowplowModel | None = None
        if config.snowplow.enabled:
            cc = config.circuit
            self.snowplow = SnowplowModel(
                anode_radius=cc.anode_radius,
                cathode_radius=cc.cathode_radius,
                fill_density=config.rho0,
                anode_length=config.snowplow.anode_length,
                mass_fraction=config.snowplow.mass_fraction,
                current_fraction=config.snowplow.current_fraction,
                radial_mass_fraction=config.snowplow.radial_mass_fraction,
                fill_pressure_Pa=config.snowplow.fill_pressure_Pa,
                pinch_column_fraction=config.snowplow.pinch_column_fraction,
                radial_current_fraction=config.snowplow.radial_current_fraction,
                radial_current_fraction_2=config.snowplow.radial_current_fraction_2,
                radial_transition_time=config.snowplow.radial_transition_time,
            )

        # Field Manager for vector calculus and inductance (Phase 5)
        self.field_manager = FieldManager(
            grid_shape=(nx, ny, nz),
            dx=dx,
            # geometry handled internally in FieldManager via logic
            geometry=self.geometry_type,
        )
        if self.geometry_type == "cylindrical":
            self.field_manager.dz = config.geometry.dz if config.geometry.dz else dx

        logger.info(
            "SimulationEngine initialized: grid=(%d,%d,%d), geometry=%s, backend=%s, "
            "sim_time=%.2e s, bremsstrahlung=%s, fld=%s, sheath=%s",
            nx, ny, nz, self.geometry_type, self.backend, config.sim_time,
            self.rad_cfg.bremsstrahlung_enabled,
            self.rad_cfg.fld_enabled,
            self.sheath_cfg.enabled,
        )

        # Warn about physics modules skipped by non-Python backends.
        # Note: Metal and Python backends share the engine's operator-split
        # loop, so radiation/sheath/diffusion physics ARE applied for Metal.
        # Athena++/AthenaK have their own fast path and skip these modules.
        if self.backend in ("athenak", "athena", "hybrid"):
            skipped = []
            if fc.enable_viscosity:
                skipped.append("Braginskii viscosity")
            if fc.enable_nernst:
                skipped.append("Nernst effect")
            if fc.enable_anisotropic_conduction:
                skipped.append("anisotropic thermal conduction")
            if self.rad_cfg.bremsstrahlung_enabled or self.rad_cfg.line_radiation_enabled:
                skipped.append("radiation transport (bremsstrahlung/line)")
            if self.sheath_cfg.enabled:
                skipped.append("sheath boundary conditions")
            if fc.diffusion_method == "sts":
                skipped.append("RKL2 super time-stepping")
            if fc.diffusion_method == "implicit":
                skipped.append("implicit diffusion (ADI)")

            if skipped:
                logger.warning(
                    "Backend '%s' skips physics modules: %s. "
                    "These modules are handled by the Python engine's operator-split "
                    "loop but are NOT applied for Athena++/AthenaK backends. "
                    "Use backend='metal' (supports all operator-split physics) "
                    "or backend='python' (all physics, but non-conservative).",
                    self.backend, ", ".join(skipped),
                )
        elif self.backend == "metal":
            # Metal shares the engine operator-split loop (radiation, sheath,
            # collision) AND has its own transport physics (Hall, Braginskii,
            # Nernst, resistive MHD, bremsstrahlung).
            metal_only = []
            if fc.diffusion_method == "sts":
                metal_only.append("RKL2 super time-stepping (using explicit instead)")
            if fc.diffusion_method == "implicit":
                metal_only.append("implicit diffusion (using explicit instead)")
            if metal_only:
                logger.info(
                    "Metal backend note: %s", ", ".join(metal_only),
                )

    @property
    def engine_tier(self) -> str:
        """Return the engine tier based on backend.

        Returns:
            ``"production"`` for conservative backends (Athena++, Metal),
            ``"teaching"`` for the Python backend (non-conservative dp/dt).
        """
        if self.backend in ("athena", "metal", "hybrid"):
            return "production"
        return "teaching"

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """Resolve the requested backend to an actual backend name.

        Args:
            requested: ``"python"``, ``"athena"``, ``"athenak"``,
                ``"metal"``, or ``"auto"``.

        Returns:
            ``"python"``, ``"athena"``, ``"athenak"``, or ``"metal"``.

        Raises:
            RuntimeError: If an explicit backend was requested but is
                not available.
        """
        if requested == "python":
            return "python"

        if requested == "hybrid":
            # Hybrid uses Athena++ for physics + WALRUS surrogate for acceleration
            return "hybrid"

        if requested == "metal":
            from dpf.metal.metal_solver import MetalMHDSolver
            if not MetalMHDSolver.is_available():
                raise RuntimeError(
                    "Metal GPU backend requested but PyTorch MPS is not available. "
                    "Requires Apple Silicon with PyTorch >= 2.0.\n"
                    "Or use backend='python' or backend='auto'."
                )
            return "metal"

        if requested == "athenak":
            from dpf.athenak_wrapper import is_available as athenak_available
            if not athenak_available():
                raise RuntimeError(
                    "AthenaK backend requested but binary not found. Build with:\n"
                    "  bash scripts/setup_athenak.sh\n"
                    "  bash scripts/build_athenak.sh\n"
                    "Or use backend='python' or backend='auto'."
                )
            return "athenak"

        if requested == "athena":
            from dpf.athena_wrapper import is_available
            if not is_available():
                raise RuntimeError(
                    "Athena++ backend requested but _athena_core extension "
                    "is not compiled.  Build with:\n"
                    "  cd src/dpf/athena_wrapper/cpp && mkdir -p build && cd build\n"
                    "  cmake .. -DATHENA_ROOT=../../external/athena && make -j8\n"
                    "Or use backend='python' or backend='auto'."
                )
            return "athena"

        if requested == "auto":
            # Priority: Athena++ > Metal > AthenaK > Python
            # Athena++ and Metal are "production" tier (conservative dE/dt).
            # Python is "teaching" tier (non-conservative dp/dt).
            try:
                from dpf.athena_wrapper import is_available
                if is_available():
                    logger.info("Auto-selected Athena++ backend (primary)")
                    return "athena"
            except ImportError:
                pass
            try:
                from dpf.metal.metal_solver import MetalMHDSolver
                if MetalMHDSolver.is_available():
                    logger.info("Auto-selected Metal backend (GPU)")
                    return "metal"
            except ImportError:
                pass
            try:
                from dpf.athenak_wrapper import is_available as athenak_available
                if athenak_available():
                    logger.info("Auto-selected AthenaK backend")
                    return "athenak"
            except ImportError:
                pass
            logger.info("Auto-selected Python backend (no C++ backends available)")
            return "python"

        raise ValueError(f"Unknown backend: {requested!r}")

    def _initial_state(self, nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
        """Create initial plasma state (uniform fill gas)."""
        rho0 = self.config.rho0
        T0 = self.config.T0
        # Total pressure = p_e + p_i = n_i*k_B*Te + n_i*k_B*Ti
        # With Te = Ti = T0: p_total = 2 * n_i * k_B * T0
        n_i = rho0 / self.ion_mass
        p0 = 2.0 * n_i * k_B * T0

        return {
            "rho": np.full((nx, ny, nz), rho0),
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.full((nx, ny, nz), p0),
            "B": np.zeros((3, nx, ny, nz)),
            "Te": np.full((nx, ny, nz), T0),
            "Ti": np.full((nx, ny, nz), T0),
            "psi": np.zeros((nx, ny, nz)),  # Dedner cleaning scalar
        }

    def _compute_dt(self) -> float:
        """Compute global timestep from CFL and circuit constraints."""
        dt_fluid = self.fluid._compute_dt(self.state)
        # Circuit timescale: L/R or sqrt(LC)
        L = self.circuit.L_ext + self._coupling.Lp
        R = self.circuit.R_total
        C = self.circuit.C
        dt_circuit = 0.1 * min(
            L / max(R, 1e-30),
            np.sqrt(L * C),
        )
        dt = min(dt_fluid, dt_circuit)

        # Honor user-specified initial dt
        if self.config.dt_init is not None and self.step_count == 0:
            dt = min(dt, self.config.dt_init)

        # Cap at reasonable fraction of sim_time
        dt = min(dt, self.config.sim_time / 10.0)

        return dt

    def save_checkpoint(self, filename: str | None = None) -> None:
        """Save current simulation state to an HDF5 checkpoint file.

        Args:
            filename: Output file path (default: self.checkpoint_filename).
        """
        fname = filename or self.checkpoint_filename
        circuit_state = {
            "current": self.circuit.current,
            "voltage": self.circuit.voltage,
            "energy_cap": self.circuit.state.energy_cap,
            "energy_ind": self.circuit.state.energy_ind,
            "energy_res": self.circuit.state.energy_res,
        }
        snowplow_state = None
        if self.snowplow is not None:
            snowplow_state = {
                "z": self.snowplow.z,
                "v": self.snowplow.v,
                "r_shock": self.snowplow.r_shock,
                "v_r": self.snowplow.vr,
                "phase": self.snowplow.phase,
                "swept_mass": self.snowplow.swept_mass,
                "rundown_complete": self.snowplow.rundown_complete,
            }
        config_json = self.config.model_dump_json()
        save_checkpoint(
            fname, self.state, circuit_state,
            self.time, self.step_count, config_json,
            snowplow_state=snowplow_state,
        )

    def load_from_checkpoint(self, filename: str) -> None:
        """Restore simulation state from an HDF5 checkpoint file.

        Args:
            filename: Input checkpoint file path.
        """
        data = load_checkpoint(filename)
        self.state = data["state"]
        self.time = data["time"]
        self.step_count = data["step_count"]

        # Restore circuit state
        circ = data["circuit"]
        self.circuit.state.current = circ.get("current", 0.0)
        self.circuit.state.voltage = circ.get("voltage", 0.0)
        self.circuit.state.energy_cap = circ.get("energy_cap", 0.0)
        self.circuit.state.energy_ind = circ.get("energy_ind", 0.0)
        self.circuit.state.energy_res = circ.get("energy_res", 0.0)

        # Restore snowplow state if present
        if self.snowplow is not None and "snowplow" in data:
            sp = data["snowplow"]
            for attr, val in sp.items():
                # Some attributes are read-only properties (e.g. rundown_complete)
                # with private backing fields (_rundown_complete).
                try:
                    setattr(self.snowplow, attr, val)
                except AttributeError:
                    private = f"_{attr}"
                    if hasattr(self.snowplow, private):
                        setattr(self.snowplow, private, val)

        # Set initial energy for conservation tracking
        self.initial_energy = self.circuit.total_energy()

        logger.info(
            "Restored from checkpoint: t=%.4e s, step=%d, I=%.1f A",
            self.time, self.step_count, self.circuit.current,
        )

    # ------------------------------------------------------------------
    # NaN / Inf guard
    # ------------------------------------------------------------------

    def _sanitize_state(self, label: str) -> int:
        """Check for and repair NaN/Inf values in state arrays.

        Args:
            label: Human-readable label for logging (e.g. "after fluid step").

        Returns:
            Total number of non-finite values repaired.

        Raises:
            RuntimeError: If cumulative repairs exceed 10000, indicating solver
                instability rather than benign boundary artifacts.
        """
        total_repaired = 0
        floors = {
            "rho": 1e-20,
            "pressure": 1e-20,
            "Te": 1.0,
            "Ti": 1.0,
        }
        for key, arr in self.state.items():
            if not isinstance(arr, np.ndarray):
                continue
            bad = ~np.isfinite(arr)
            count = int(np.sum(bad))
            if count > 0:
                floor = floors.get(key, 0.0)
                arr[bad] = floor
                total_repaired += count
                logger.warning(
                    "%s: %d non-finite values in '%s', replaced with %.1e",
                    label, count, key, floor,
                )
        if total_repaired > 0:
            self._cumulative_repairs = getattr(self, "_cumulative_repairs", 0) + total_repaired
            if self._cumulative_repairs > 10000:
                raise RuntimeError(
                    f"Solver instability: {self._cumulative_repairs} cumulative NaN/Inf "
                    f"repairs. Latest: {total_repaired} in '{label}'."
                )
        return total_repaired

    # ------------------------------------------------------------------
    # Single-step interface
    # ------------------------------------------------------------------

    def step(self, *, _max_steps: int | None = None) -> StepResult:
        """Advance the simulation by a single timestep.

        Returns:
            StepResult with scalar diagnostics and ``finished`` flag.
        """
        sim_time = self.config.sim_time

        # Check termination conditions *before* stepping
        if self.time >= sim_time:
            return self._make_step_result(dt=0.0, finished=True)
        if _max_steps is not None and self.step_count >= _max_steps:
            return self._make_step_result(dt=0.0, finished=True)

        # Set initial energy on first call
        if self.initial_energy is None:
            self.initial_energy = self.circuit.total_energy()

        dt = self._compute_dt()
        # Don't overshoot
        if self.time + dt > sim_time:
            dt = sim_time - self.time

        # === Athena++ fast path ===
        # When using the Athena++ backend, delegate the MHD step to C++
        # and use a simplified coupling loop.  The full Python physics
        # operators (Spitzer, Nernst, viscosity, radiation) will be moved
        # to Athena++ source terms in Phase G.  For now, only circuit
        # coupling is active.
        if self.backend in ("athena", "hybrid"):
            return self._step_athena(dt, sim_time, _max_steps)

        # Deprecation warning for Python Cartesian backend on production workloads.
        # The Cartesian MHDSolver uses dp/dt (non-conservative) which violates
        # Rankine-Hugoniot at shocks.  The CylindricalMHDSolver now defaults to
        # conservative_energy=True, so this warning only applies to Cartesian.
        if self.step_count == 0 and self.backend == "python":
            is_conservative = getattr(self.fluid, "conservative_energy", False)
            if not is_conservative:
                nx, ny, nz = self.config.grid_shape
                import warnings
                if nx * ny * nz > 16**3 or self.config.sim_time > 1e-7:
                    warnings.warn(
                        "Python MHD backend uses a non-conservative pressure equation "
                        "(dp/dt instead of dE/dt) which violates Rankine-Hugoniot at "
                        f"shocks (grid {nx}x{ny}x{nz}, sim_time={self.config.sim_time:.1e}). "
                        "For production accuracy, use backend='metal' (conservative, GPU) "
                        "or backend='athena' (Athena++ C++). "
                        "The Python engine is recommended only for teaching and prototyping.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

        # === Phase 5: Pulse Power Circuit Step ===
        # 1. Update fields for inductance calculation (copy to avoid aliasing —
        # the fluid step may later modify state["B"] in-place, and we need the
        # pre-step B for inductance computation).
        self.field_manager.B = self.state["B"].copy()

        # 2. Track field-based inductance for _prev_L_plasma history.
        # The actual dL/dt used by the circuit is computed from snowplow or
        # volume-integral L_plasma in the circuit step below (Step 2).
        L_p = self.field_manager.compute_plasma_inductance(self.circuit.current)
        self._prev_L_plasma = L_p

        # 3. Apply magnetic boundary conditions using current from previous step.
        # The circuit is advanced ONCE per step, after computing R_plasma and L_plasma
        # from the full MHD state (see Step 2 below).  Using previous-step current here
        # is standard explicit coupling — the electrode B_theta BC is set before the
        # MHD advance, which is then used to compute the updated plasma state.
        if self.boundary_cfg.electrode_bc:
            self._apply_electrode_bc(self._coupling.current)

        # === Step 1: Compute ionization state and plasma resistance ===
        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = rho / self.ion_mass  # Number density (assume fully ionized for ne)

        # Volume-averaged quantities for scalar coupling
        Te_avg = float(np.mean(Te))
        ne_avg = float(np.mean(ne))

        # Spatially-varying ionization state from Saha equation
        Te_flat = Te.ravel()
        ne_flat = ne.ravel()
        Z_bar_field = saha_ionization_fraction_array(Te_flat, ne_flat).reshape(Te.shape)
        Z_bar_field = np.maximum(Z_bar_field, 0.01)  # Floor to avoid division by zero
        # Scalar Z_bar for circuit coupling and scalar operators
        Z_bar = max(float(np.mean(Z_bar_field)), 0.01)

        # Compute spatially-resolved resistivity field for MHD solver
        eta_field = None
        eta_anom = 0.0
        R_plasma = 0.0
        L_plasma = 0.0

        if Te_avg > 1000.0 and ne_avg > 1e10:
            # Cell-by-cell Spitzer resistivity with temperature floor
            # and spatially-varying Z_bar for accurate transport
            Te_floored = np.maximum(Te, 1000.0)
            ne_floored = np.maximum(ne, 1e10)
            lnL_field = coulomb_log(ne_floored, Te_floored)
            eta_spitzer_field = spitzer_resistivity(
                ne_floored, Te_floored, lnL_field, Z=Z_bar_field,
            )

            # Volume-averaged for fallback and scalar coupling
            lnL_avg = coulomb_log(
                np.array([ne_avg]), np.array([Te_avg])
            )[0]
            eta_spitzer_avg = float(spitzer_resistivity(
                np.array([ne_avg]), np.array([Te_avg]), lnL_avg, Z=Z_bar
            )[0])

            # Spatially-resolved anomalous resistivity from Buneman instability
            # Compute J field from curl(B)/mu_0 for threshold check
            I_current = self._coupling.current
            A_column = pi * self.anode_radius**2
            B_field = self.state["B"]
            Ti_field = self.state["Ti"]

            if self.geometry_type == "cylindrical":
                # Cylindrical: curl(B)/mu_0 for J magnitude
                B_2d = np.squeeze(B_field, axis=2) if B_field.ndim == 4 else B_field
                curl_B = self.fluid.geom.curl(B_2d)
                J_field = curl_B / _mu_0
                J_mag = np.sqrt(np.sum(J_field**2, axis=0))  # (nr, nz)
                ne_2d = np.squeeze(ne, axis=1) if ne.ndim == 3 else ne
                Ti_2d = np.squeeze(Ti_field, axis=1) if Ti_field.ndim == 3 else Ti_field
                Te_2d = np.squeeze(self.state["Te"], axis=1) if self.state["Te"].ndim == 3 else self.state["Te"]
                eta_anom_field = anomalous_resistivity_field(
                    J_mag, np.maximum(ne_2d, 1e10), np.maximum(Ti_2d, 1.0),
                    alpha=self.config.anomalous_alpha,
                    mi=self.ion_mass,
                    threshold_model=self.config.anomalous_threshold_model,
                    Te=np.maximum(Te_2d, 1.0),
                )
                # Unsqueeze to (nr, 1, nz)
                eta_anom_field_3d = eta_anom_field[:, np.newaxis, :]
                eta_field = eta_spitzer_field + eta_anom_field_3d
            else:
                # Cartesian: compute J from curl(B)
                dx = self.config.dx
                J_field = np.array([
                    np.gradient(B_field[2], dx, axis=1) - np.gradient(B_field[1], dx, axis=2),
                    np.gradient(B_field[0], dx, axis=2) - np.gradient(B_field[2], dx, axis=0),
                    np.gradient(B_field[1], dx, axis=0) - np.gradient(B_field[0], dx, axis=1),
                ]) / _mu_0
                J_mag = np.sqrt(np.sum(J_field**2, axis=0))
                eta_anom_field = anomalous_resistivity_field(
                    J_mag, np.maximum(ne, 1e10), np.maximum(Ti_field, 1.0),
                    alpha=self.config.anomalous_alpha,
                    mi=self.ion_mass,
                    threshold_model=self.config.anomalous_threshold_model,
                    Te=np.maximum(self.state["Te"], 1.0),
                )
                eta_field = eta_spitzer_field + eta_anom_field

            # Scalar anomalous for diagnostics
            J_avg = abs(I_current) / max(A_column, 1e-30)
            Ti_avg = float(np.mean(self.state["Ti"]))
            Te_avg_scalar = float(np.mean(self.state["Te"]))
            eta_anom = anomalous_resistivity_scalar(
                J_avg, ne_avg, Ti_avg, alpha=self.config.anomalous_alpha,
                mi=self.ion_mass,
                threshold_model=self.config.anomalous_threshold_model,
                Te_val=Te_avg_scalar,
            )
            eta_total_avg = total_resistivity_scalar(eta_spitzer_avg, eta_anom)

            # Sanitize: cap extreme values and NaN
            eta_field = np.where(np.isfinite(eta_field), eta_field, eta_total_avg)
            eta_field = np.minimum(eta_field, 1.0)

            # --- Volume-integral R_plasma: R = integral(eta*|J|^2 dV) / I^2 ---
            I_sq = max(I_current**2, 1e-30)
            if self.geometry_type == "cylindrical":
                cell_vol = self.fluid.geom.cell_volumes()  # (nr, nz)
                eta_2d = np.squeeze(eta_field, axis=1) if eta_field.ndim == 3 else eta_field
                J_sq = J_mag**2
                R_plasma = float(np.sum(eta_2d * J_sq * cell_vol)) / I_sq
            else:
                dV = self.config.dx**3
                J_sq = np.sum(J_field**2, axis=0)
                R_plasma = float(np.sum(eta_field * J_sq * dV)) / I_sq

            # Cap R_plasma to physical range (DPF: ~10 mOhm normal, ~1-10 Ohm at pinch disruption)
            R_plasma = min(R_plasma, 10.0)

            # --- Volume-integral L_plasma: L = 2 * integral(B^2/(2*mu_0) dV) / I^2 ---
            B_sq = np.sum(B_field**2, axis=0)
            if self.geometry_type == "cylindrical":
                B_sq_2d = np.squeeze(B_sq, axis=1) if B_sq.ndim == 3 else B_sq
                L_plasma = float(np.sum(B_sq_2d / _mu_0 * cell_vol)) / I_sq
            else:
                L_plasma = float(np.sum(B_sq / _mu_0 * dV)) / I_sq

        # === Step 1b: Collision+Radiation (first half-step of Strang) ===
        self._apply_collision_radiation(dt / 2.0, Z_bar, Z_bar_field=Z_bar_field)

        # === Step 2: Circuit advance (with plasma resistance + inductance) ===
        # Sub-cycle the circuit + snowplow to resolve their dynamics properly.
        # The MHD CFL timestep can be ~1 µs for cold gas, but the circuit LC
        # period is ~40 µs and the snowplow needs ~10 ns steps for accurate
        # sheath trajectory.  Without sub-cycling, the snowplow blows through
        # the entire anode in a single oversized step.
        coupling = self.fluid.coupling_interface()
        coupling.R_plasma = R_plasma
        coupling.Z_bar = Z_bar

        # Compute sub-step size: resolve the circuit LC timescale
        L_total = self.circuit.L_ext + self._coupling.Lp
        dt_lc = np.sqrt(max(L_total, 1e-12) * self.circuit.C)
        # Target ~500 sub-steps per quarter period for accurate snowplow trajectory
        dt_sub_target = max(dt_lc / 500.0, 1e-12)
        n_sub = max(1, int(np.ceil(dt / dt_sub_target)))
        dt_sub = dt / n_sub

        sheath_pressure = self._dynamic_sheath_pressure()

        # back-EMF treatment: when the snowplow model is active, the motional
        # EMF (integral of v x B . dl) is already captured by I * dL/dt inside
        # the circuit solver's R_star = R_eff + dLp/dt.  Computing a separate
        # MHD-field-based back_emf AND including dL/dt in R_star would double-
        # count the inductive coupling.  Only use MHD back-EMF when there is
        # no snowplow model providing dL/dt.
        # (Debate #20 finding: Python engine was double-counting ~100-300 kV
        # at pinch compression while Athena++ correctly set back_emf=0.)
        if self.snowplow is not None:
            back_emf = 0.0
        else:
            back_emf = self._compute_back_emf(dt)

        for _isub in range(n_sub):
            # Snowplow dynamics: sheath-derived L_plasma overrides field-based
            # L_plasma when the snowplow model is active (Lee model Phases 2-5).
            if self.snowplow is not None and self.snowplow.is_active:
                sp_result = self.snowplow.step(
                    dt_sub, self._coupling.current,
                    pressure=sheath_pressure,
                )
                coupling.Lp = sp_result["L_plasma"]
                coupling.dL_dt = sp_result["dL_dt"]
                self._prev_L_plasma = sp_result["L_plasma"]
                self._last_sp_dL_dt = sp_result["dL_dt"]

            elif self.snowplow is not None and not self.snowplow.is_active:
                # Snowplow reached final pinch — freeze L_plasma at its pinch
                # value and set dL/dt=0.  The snowplow.plasma_inductance property
                # returns L_axial_frozen + L_radial(r_pinch_min), which is the
                # correct inductance for a stagnated Z-pinch column.
                #
                # Without this branch, the code falls through to the MHD field-
                # based L_plasma, which is ~0 on coarse grids because the B-field
                # hasn't been properly evolved by the Metal solver at ~16 MHD
                # steps.  The resulting L_total → L_ext = 33.5 nH produces a
                # catastrophic current spike (10^7 MA).
                coupling.Lp = self.snowplow.plasma_inductance
                coupling.dL_dt = 0.0

            elif L_plasma > 0:
                # Fallback: use volume-integral L_plasma from MHD fields
                coupling.Lp = L_plasma
                if self._prev_L_plasma > 0 and dt_sub > 0:
                    coupling.dL_dt = (L_plasma - self._prev_L_plasma) / dt_sub
                self._prev_L_plasma = L_plasma

            # Advance circuit one sub-step
            new_coupling = self.circuit.step(coupling, back_emf, dt_sub)
            self._coupling = new_coupling
            # Update coupling R_plasma for next sub-step (R_plasma is constant
            # during sub-cycling since MHD state is frozen)
            coupling.R_plasma = R_plasma

        # One-shot B-field initialization at axial→radial phase transition
        if (
            self.snowplow is not None
            and self.snowplow.phase in ("radial", "reflected", "pinch")
            and not self._radial_bfield_initialized
        ):
            self._initialize_radial_bfield()

        # === Step 2.5: Kinetic / PIC Step ===
        # Run kinetic step *before* fluid to provide J_kin source terms for this step
        if self.kinetic and self.kinetic.kc.enabled:
            # Convert E, B to (nx, ny, nz, 3) for HybridPIC
            # Engine state["B"] is (3, nx, ny, nz)
            B_fld = np.moveaxis(self.state["B"], 0, -1)

            # E field reconstruction: E = -v x B + eta*J (simplified to -v x B for pushing)
            # Ideally should use E from previous step or predictor-corrector
            v = np.moveaxis(self.state["velocity"], 0, -1)
            E_fld = -np.cross(v, B_fld)

            self.kinetic.step(dt, self.time, E_fld, B_fld)

            # Get current density deposition for feedback to MHD
            Jx, Jy, Jz = self.kinetic.get_current_density()
            # Pack into source_terms (3, nx, ny, nz)
            # KineticManager returns (nx, ny, nz) arrays, Engine expects (3, nx, ny, nz)
            J_kin = np.stack([Jx, Jy, Jz], axis=0)
            self._current_source_terms = {"J_kin": J_kin}
        else:
            self._current_source_terms = None

        # === Step 3: Fluid/MHD advance (with resistivity + electrode BCs + Kinetics) ===
        # Measure ohmic gap using CURRENT state and inject correction in THIS step
        # (eliminates one-step lag from previous approach)
        if (
            self.config.fluid.enable_ohmic_correction
            and eta_field is not None
            and self._cell_volume is not None
        ):
            self._measure_ohmic_gap(eta_field, new_coupling, dt)
            if self._last_ohmic_gap != 0.0:
                src = self._current_source_terms or {}
                src["Q_ohmic_correction"] = self._compute_ohmic_correction(
                    eta_field, new_coupling.current, dt,
                )
                self._current_source_terms = src

        # Inject snowplow source terms into MHD grid
        if self.config.snowplow.enable_mhd_coupling and self.snowplow is not None:
            sp_src = self._compute_snowplow_source_terms(dt)
            if sp_src:
                src = self._current_source_terms or {}
                src.update(sp_src)
                self._current_source_terms = src

        cc = self.config.circuit
        self.state = self.fluid.step(
            self.state,
            dt,
            current=new_coupling.current,
            voltage=new_coupling.voltage,
            eta_field=eta_field,
            source_terms=self._current_source_terms,
            anode_radius=cc.anode_radius,
            cathode_radius=cc.cathode_radius,
            apply_electrode_bc=self.boundary_cfg.electrode_bc,
        )
        self._sanitize_state("after fluid step")

        # (ohmic gap measurement moved to before fluid step — no longer lagged)

        # === Step 3.1: Ablation operator-split step ===
        if self.config.ablation.enabled:
            from dpf.atomic.ablation import ablation_source_array
            I_abl = abs(new_coupling.current)
            A_col = pi * self.anode_radius**2
            J_bdy = I_abl / max(A_col, 1e-30)
            J_arr = np.full_like(self.state["rho"], J_bdy)
            # Boundary mask: anode face (inner radial boundary)
            mask = np.zeros(self.state["rho"].shape, dtype=np.int64)
            mask[0, :, :] = 1
            eta_abl = eta_field if eta_field is not None else np.full_like(
                self.state["rho"], 1e-7,
            )
            S_rho = ablation_source_array(
                J_arr, eta_abl, self.config.ablation.efficiency, mask,
            )
            self.state["rho"] = self.state["rho"] + S_rho * dt
            self._sanitize_state("after ablation step")

        # === Step 3.5: Powell 8-wave div(B) source terms ===
        if self.config.fluid.enable_powell:
            self._apply_powell_sources(dt)
            self._sanitize_state("after Powell step")

        # === Step 3a: Nernst B-field advection ===
        fc = self.config.fluid
        if fc.enable_nernst and self.backend != "metal":
            self._apply_nernst(dt, Z_bar)
            self._sanitize_state("after Nernst step")

        # === Step 3b: Sheath boundary conditions ===
        if self.sheath_cfg.enabled:
            Te_bc = self.state["Te"]
            ne_bc = self.state["rho"] / self.ion_mass
            Te_boundary = float(np.mean(Te_bc))
            ne_boundary = float(np.mean(ne_bc))
            V_sh = self.sheath_cfg.V_sheath
            if V_sh <= 0.0 and Te_boundary > 100.0:
                V_sh = abs(float(floating_potential(Te_boundary)))
            if V_sh > 0.0 and Te_boundary > 100.0 and ne_boundary > 1e10:
                self.state = apply_sheath_bc(
                    self.state,
                    ne_boundary=ne_boundary,
                    Te_boundary=Te_boundary,
                    V_sheath=V_sh,
                    boundary=self.sheath_cfg.boundary,
                )

        # === Step 4+5: Collision+Radiation (second half-step of Strang) ===
        # Recompute Z_bar from post-MHD Te to avoid stale ionization state.
        # The MHD step may have significantly changed Te (e.g. shock heating),
        # so using pre-step Z_bar introduces O(dt) splitting error.
        Te_post = self.state["Te"]
        ne_post = self.state["rho"] / self.ion_mass
        Z_bar_field_post = saha_ionization_fraction_array(
            Te_post.ravel(), ne_post.ravel(),
        ).reshape(Te_post.shape)
        Z_bar_field_post = np.maximum(Z_bar_field_post, 0.01)
        Z_bar_post = max(float(np.mean(Z_bar_field_post)), 0.01)
        self._apply_collision_radiation(dt / 2.0, Z_bar_post, Z_bar_field=Z_bar_field_post)

        # === Step 5b: Neutron yield (DD thermonuclear) ===
        Ti_yield = self.state["Ti"]
        rho_yield = self.state["rho"]
        n_D = rho_yield / self.ion_mass  # Number density for neutron yield
        if self.geometry_type == "cylindrical":
            cell_vol = self.fluid.geom.cell_volumes()
            # Expand from (nr, nz) to (nr, 1, nz) for broadcasting
            Ti_2d = np.squeeze(Ti_yield, axis=1) if Ti_yield.ndim == 3 else Ti_yield
            nD_2d = np.squeeze(n_D, axis=1) if n_D.ndim == 3 else n_D
            _, neutron_rate = neutron_yield_rate(nD_2d, Ti_2d, cell_vol)
        else:
            cell_vol_cart = self.config.dx**3
            _, neutron_rate = neutron_yield_rate(n_D, Ti_yield, cell_vol_cart)
        self._last_neutron_rate = neutron_rate
        self.total_neutron_yield += neutron_rate * dt

        # === Step 5b2: Beam-target neutron yield (during pinch/reflected) ===
        self._last_beam_target_rate = 0.0
        if self.snowplow and self.snowplow.phase in ("reflected", "pinch"):
            from dpf.diagnostics.beam_target import beam_target_yield_rate
            _sp_dL_dt = getattr(self, "_last_sp_dL_dt", 0.0)
            V_pinch = abs(self._coupling.current * _sp_dL_dt)
            n_target = float(np.mean(self.state["rho"])) / self.ion_mass
            bt_rate = beam_target_yield_rate(
                abs(self._coupling.current), V_pinch, n_target,
                self.snowplow.L_anode, f_beam=0.2,
            )
            self.total_neutron_yield += bt_rate * dt
            self._last_beam_target_rate = bt_rate

        # === Step 5b3: m=0 sausage instability growth rate ===
        self._last_m0_result = None
        if self.snowplow and self.snowplow.phase in ("radial", "reflected", "pinch"):
            from dpf.diagnostics.instability import m0_growth_rate_from_state
            self._last_m0_result = m0_growth_rate_from_state(
                self.state, self.snowplow, self.config,
            )

        # === Step 5b4: Pease-Braginskii radiative collapse check ===
        from dpf.diagnostics.pease_braginskii import check_pease_braginskii
        self._last_pb_result = check_pease_braginskii(
            I_current=abs(self._coupling.current),
            Z=Z_bar,
            gaunt_factor=1.2,
            ln_Lambda=self.config.collision.coulomb_log,
        )

        # === Step 5c: Well Exporter ===
        if self.well_interval > 0 and self.step_count % self.well_interval == 0:
            self.well_exporter.append_state(self.state, time=self.time)




        # === Step 5c: Synthetic interferometry (cylindrical only) ===
        if self.geometry_type == "cylindrical":
            ne_interf = rho_yield / self.ion_mass
            # Take midplane slice (z = nz//2)
            nz_mid = ne_interf.shape[2] // 2
            ne_midplane = ne_interf[:, 0, nz_mid]  # shape (nr,)
            r_grid = self.fluid.geom.r  # Radial coordinate array
            N_L = abel_transform(ne_midplane, r_grid)
            self._last_fringe_shifts = fringe_shift(N_L)

        # === Step 5d: Plasma regime validity check (Phase AE) ===
        # Periodic check (every 100 steps) of MHD validity criteria:
        # ND (collisionality), Rm (frozen-in flux), Debye length, ion skin depth.
        if self.step_count % 100 == 0:
            from dpf.diagnostics.plasma_regime import regime_validity
            ne_rv = self.state["rho"] / self.ion_mass
            Te_rv = self.state["Te"]
            Ti_rv = self.state["Ti"]
            v_mag = np.sqrt(np.sum(self.state["velocity"] ** 2, axis=0))
            rv = regime_validity(ne_rv, Te_rv, Ti_rv, v_mag, dx=self.config.dx)
            self._last_regime_result = rv
            frac = rv["fraction_valid"]
            if frac < 0.5:
                logger.warning(
                    "MHD regime validity: %.0f%% of cells outside MHD-valid regime "
                    "(ND>1 or dx<10*lambda_De). Consider kinetic model.",
                    (1.0 - frac) * 100,
                )

        # === Step 6: Advance time and record diagnostics ===
        self.time += dt
        self.step_count += 1

        # Store latest step-level scalars for StepResult
        self._last_R_plasma = R_plasma
        self._last_Z_bar = Z_bar
        self._last_eta_anom = eta_anom

        if self.step_count % self.diag_interval == 0:
            circ = self.circuit.state
            diag_state = {
                **self.state,
                "circuit": {
                    "current": circ.current,
                    "voltage": circ.voltage,
                    "energy_cap": circ.energy_cap,
                    "energy_ind": circ.energy_ind,
                    "energy_res": circ.energy_res,
                    "energy_total": self.circuit.total_energy(),
                },
                "radiation": {
                    "total_radiated_energy": self.total_radiated_energy,
                    "bremsstrahlung_enabled": self.rad_cfg.bremsstrahlung_enabled,
                    "fld_enabled": self.rad_cfg.fld_enabled,
                },
                "plasma": {
                    "R_plasma": R_plasma,
                    "Z_bar": Z_bar,
                    "eta_anomalous": eta_anom,
                    "sheath_enabled": self.sheath_cfg.enabled,
                    "geometry": self.geometry_type,
                },
                "neutrons": {
                    "neutron_rate": self._last_neutron_rate,
                    "beam_target_rate": getattr(self, "_last_beam_target_rate", 0.0),
                    "total_neutron_yield": self.total_neutron_yield,
                },
                "instability": {
                    "m0_growth_rate": (
                        self._last_m0_result["growth_rate"]
                        if self._last_m0_result else 0.0
                    ),
                    "m0_growth_time": (
                        self._last_m0_result["growth_time"]
                        if self._last_m0_result else float("inf")
                    ),
                    "m0_beta_p": (
                        self._last_m0_result["beta_p"]
                        if self._last_m0_result else 0.0
                    ),
                    "m0_is_unstable": (
                        self._last_m0_result["is_unstable"]
                        if self._last_m0_result else False
                    ),
                },
                "snowplow": {
                    "z_sheath": self.snowplow.z if self.snowplow else 0.0,
                    "v_sheath": self.snowplow.v if self.snowplow else 0.0,
                    "swept_mass": self.snowplow.swept_mass if self.snowplow else 0.0,
                    "rundown_complete": (
                        self.snowplow.rundown_complete if self.snowplow else False
                    ),
                    "r_shock": self.snowplow.r_shock if self.snowplow else 0.0,
                    "phase": self.snowplow.phase if self.snowplow else "none",
                },
                "pease_braginskii": {
                    "I_PB_MA": self._last_pb_result["I_PB_MA"],
                    "ratio": self._last_pb_result["ratio"],
                    "exceeds_PB": self._last_pb_result["exceeds_PB"],
                    "regime": self._last_pb_result["regime"],
                },
                "regime_validity": {
                    "fraction_valid": (
                        self._last_regime_result["fraction_valid"]
                        if self._last_regime_result else 1.0
                    ),
                    "ND_max": (
                        float(np.max(self._last_regime_result["ND"]))
                        if self._last_regime_result else 0.0
                    ),
                    "Rm_min": (
                        float(np.min(self._last_regime_result["Rm"]))
                        if self._last_regime_result else 0.0
                    ),
                },
            }
            self.diagnostics.record(diag_state, self.time)

        if self.step_count % 100 == 0:
            E_total = self.circuit.total_energy()
            logger.info(
                "Step %d: t=%.4e s, dt=%.2e s, I=%.1f A, V=%.1f V, E_cons=%.4f",
                self.step_count,
                self.time,
                dt,
                new_coupling.current,
                new_coupling.voltage,
                E_total / max(self.initial_energy, 1e-30),
            )

        # Auto-checkpoint
        if (
            self.checkpoint_interval > 0
            and self.step_count % self.checkpoint_interval == 0
        ):
            self.save_checkpoint()

        # Check if finished after this step
        finished = self.time >= sim_time
        if _max_steps is not None and self.step_count >= _max_steps:
            finished = True

        return self._make_step_result(dt=dt, finished=finished)

    def _compute_J_from_B(self, B: np.ndarray) -> np.ndarray:
        """Compute current density J = curl(B) / mu_0.

        Handles both Cartesian (3D) and cylindrical (ny=1) grids.
        Returns J with same shape as B.
        """
        mu_0 = _mu_0
        dx = self.config.dx

        if self.geometry_type == "cylindrical" and hasattr(self.fluid, "geom"):
            # Use cylindrical curl operator (handles ny=1)
            B_2d = np.squeeze(B, axis=2) if B.ndim == 4 else B
            curl_B = self.fluid.geom.curl(B_2d)
            J_2d = curl_B / mu_0
            if B.ndim == 4:
                return J_2d[:, :, np.newaxis, :]
            return J_2d

        # Cartesian: standard gradient curl
        J = np.array([
            np.gradient(B[2], dx, axis=1) - np.gradient(B[1], dx, axis=2),
            np.gradient(B[0], dx, axis=2) - np.gradient(B[2], dx, axis=0),
            np.gradient(B[1], dx, axis=0) - np.gradient(B[0], dx, axis=1),
        ]) / mu_0
        return J

    def _measure_ohmic_gap(
        self,
        eta_field: np.ndarray,
        coupling: CouplingState,
        dt: float,
    ) -> None:
        """Measure gap between circuit and MHD ohmic heating rates.

        Circuit says: P_ohmic = R_plasma * I^2
        MHD says:     P_ohmic = integral(eta * J^2 * dV)

        The difference is stored for J^2-weighted correction on the next step.
        """
        B = self.state.get("B")
        if B is None or B.shape[0] < 3:
            return

        J = self._compute_J_from_B(B)
        J_sq = np.sum(J**2, axis=0)
        dV = self._cell_volume
        Q_mhd = float(np.sum(eta_field * J_sq * dV))
        Q_circuit = coupling.R_plasma * coupling.current**2

        gap = Q_circuit - Q_mhd
        self._last_ohmic_gap = gap
        self._ohmic_gap_history.append(gap)
        if len(self._ohmic_gap_history) > 100:
            self._ohmic_gap_history = self._ohmic_gap_history[-50:]

    def _compute_ohmic_correction(
        self,
        eta_field: np.ndarray,
        current: float,
        dt: float,
    ) -> np.ndarray:
        """Distribute the ohmic gap as a J^2-weighted pressure source [W/m^3].

        Returns Q_correction(x,y,z) such that:
            integral(Q_correction * dV) = last_ohmic_gap
        and the spatial distribution follows local J^2.
        """
        B = self.state.get("B")
        if B is None or B.shape[0] < 3:
            return np.zeros_like(self.state["rho"])

        J = self._compute_J_from_B(B)
        J_sq = np.sum(J**2, axis=0)
        dV = self._cell_volume
        total_J_sq_dV = float(np.sum(J_sq * dV))

        if total_J_sq_dV < 1e-30:
            return np.zeros_like(self.state["rho"])

        # Distribute gap proportional to local J^2 [W/m^3]
        return (self._last_ohmic_gap / total_J_sq_dV) * J_sq

    def _compute_snowplow_source_terms(self, dt: float) -> dict:
        """Compute density, momentum, and energy source terms from snowplow sheath.

        Injects swept mass, momentum, and post-shock energy into the MHD grid
        using Gaussian-smeared windows centered at the sheath position. This
        couples the 0D snowplow model to the 2D/3D MHD fluid.

        Returns dict with keys: S_rho_snowplow, S_mom_snowplow, S_energy_snowplow.
        Empty dict if coupling is disabled or snowplow inactive.
        """
        if not self.config.snowplow.enable_mhd_coupling:
            return {}
        if self.snowplow is None or not self.snowplow.is_active:
            return {}

        nx, ny, nz = self.config.grid_shape
        dx = self.config.dx
        dz_cfg = self.config.geometry.dz
        dz = dz_cfg if dz_cfg else dx
        gamma = self.config.fluid.gamma
        rho0 = self.config.rho0
        m_ion = self.ion_mass
        k_B = 1.381e-23

        source_terms = {}
        grid_shape = self.state["rho"].shape  # (nx, ny, nz)

        if self.snowplow.phase == "rundown":
            z_sheath = self.snowplow.sheath_position
            v_sheath = self.snowplow.sheath_velocity

            if abs(v_sheath) < 1e-6:
                return {}

            # Axial coordinate array
            z_arr = np.array([(k + 0.5) * dz for k in range(nz)])

            # Gaussian window: sigma = 2*dz, normalized
            sigma_z = 2.0 * dz
            W_z = np.exp(-0.5 * ((z_arr - z_sheath) / sigma_z) ** 2)
            W_z /= np.sum(W_z) * dz + 1e-30  # Normalize

            # Mass injection rate [kg/s]
            f_m = self.snowplow.f_m
            A_ann = self.snowplow.A_annular
            dm_dt = rho0 * A_ann * abs(v_sheath) * f_m

            # Broadcast to grid shape
            S_rho = np.zeros(grid_shape)
            if self.geometry_type == "cylindrical":
                # (nr, 1, nz) — inject along z at all radii
                S_rho[:, :, :] = dm_dt * W_z[np.newaxis, np.newaxis, :]
            else:
                S_rho[:, :, :] = dm_dt * W_z[np.newaxis, np.newaxis, :]

            # Momentum: injected gas enters at sheath velocity (z-direction)
            S_mom = np.zeros((3, *grid_shape))
            S_mom[2] = S_rho * v_sheath  # z-momentum

            # Energy: Rankine-Hugoniot post-shock
            rho_post = ((gamma + 1) / (gamma - 1)) * rho0
            T_post = (3.0 / 16.0) * m_ion * v_sheath**2 / k_B
            p_post = rho_post * k_B * T_post / m_ion
            e_kin = 0.5 * rho_post * v_sheath**2
            E_post = p_post / (gamma - 1.0) + e_kin
            S_energy = np.zeros(grid_shape)
            S_energy[:, :, :] = E_post * abs(v_sheath) * W_z[np.newaxis, np.newaxis, :]

            source_terms["S_rho_snowplow"] = S_rho
            source_terms["S_mom_snowplow"] = S_mom
            source_terms["S_energy_snowplow"] = S_energy

        elif self.snowplow.phase == "radial":
            r_shock = self.snowplow.shock_radius
            vr_shock = self.snowplow.vr

            if abs(vr_shock) < 1e-6:
                return {}

            # Radial coordinate array
            r_arr = np.array([(i + 0.5) * dx for i in range(nx)])

            # Gaussian window in r
            sigma_r = 2.0 * dx
            W_r = np.exp(-0.5 * ((r_arr - r_shock) / sigma_r) ** 2)
            W_r /= np.sum(W_r) * dx + 1e-30

            # Radial mass injection rate [kg/s]
            f_mr = self.snowplow.f_mr
            z_f = self.snowplow.z_f
            dm_dt = rho0 * 2.0 * np.pi * r_shock * abs(vr_shock) * z_f * f_mr

            S_rho = np.zeros(grid_shape)
            if self.geometry_type == "cylindrical":
                S_rho[:, :, :] = dm_dt * W_r[:, np.newaxis, np.newaxis]
            else:
                S_rho[:, :, :] = dm_dt * W_r[:, np.newaxis, np.newaxis]

            # Momentum: radial inward
            S_mom = np.zeros((3, *grid_shape))
            S_mom[0] = S_rho * vr_shock  # r-momentum (negative = inward)

            # Energy: Rankine-Hugoniot post-shock (radial)
            rho_post = ((gamma + 1) / (gamma - 1)) * rho0
            T_post = (3.0 / 16.0) * m_ion * vr_shock**2 / k_B
            p_post = rho_post * k_B * T_post / m_ion
            e_kin = 0.5 * rho_post * vr_shock**2
            E_post = p_post / (gamma - 1.0) + e_kin
            S_energy = np.zeros(grid_shape)
            S_energy[:, :, :] = E_post * abs(vr_shock) * W_r[:, np.newaxis, np.newaxis]

            source_terms["S_rho_snowplow"] = S_rho
            source_terms["S_mom_snowplow"] = S_mom
            source_terms["S_energy_snowplow"] = S_energy

        return source_terms

    def _compute_back_emf(self, dt: float) -> float:
        """Compute motional back-EMF from MHD field advection.

        The back-EMF arises from the -(v x B) electric field in the plasma.
        For a cylindrical Z-pinch, the z-component is -(v_r * B_theta).
        For Cartesian geometry, it is -(v_x * B_y - v_y * B_x).

        Returns the volume-averaged motional EMF times the axial length [V].
        """
        velocity = self.state.get("velocity")
        B = self.state.get("B")
        if velocity is None or B is None:
            return 0.0
        if velocity.shape[0] < 2 or B.shape[0] < 2:
            return 0.0

        # Compute z-component of -(v x B) as electric field density [V/m]
        if self.geometry_type == "cylindrical":
            # (v x B)_z = v_r * B_theta (components [0] and [1])
            emf_density = -(velocity[0] * B[1])
        else:
            # (v x B)_z = v_x * B_y - v_y * B_x
            emf_density = -(velocity[0] * B[1] - velocity[1] * B[0])

        # Convert from E-field [V/m] to circuit voltage [V]
        # by multiplying by the axial length (electrode gap)
        dx = self.config.dx
        dz = self.config.geometry.dz if self.config.geometry.dz else dx
        nz = self.config.grid_shape[2]
        z_length = nz * dz

        return float(np.mean(emf_density)) * z_length

    def _apply_electrode_bc(self, current: float) -> None:
        """Apply circuit-driven magnetic boundary conditions."""
        # Backend-specific electrode B-field BC (sets B_theta from current)
        if self.backend == "python" and self.geometry_type == "cylindrical":
            if hasattr(self.fluid, "apply_electrode_bfield_bc"):
                cc = self.config.circuit
                self.state["B"] = self.fluid.apply_electrode_bfield_bc(
                    self.state["B"], current, cc.anode_radius, cc.cathode_radius
                )
        elif self.geometry_type == "cylindrical":
            # Generic electrode BC for Metal, Athena++, AthenaK, and future
            # backends.  Operates directly on self.state["B"] (always NumPy).
            # Sets B_theta = mu_0 * I / (2*pi*r) between electrodes, zero inside anode.
            cc = self.config.circuit
            dr = self.config.dx
            nr = self.config.grid_shape[0]
            B = self.state["B"]
            for ir in range(nr):
                r = (ir + 0.5) * dr
                if cc.anode_radius <= r <= cc.cathode_radius and r > 0:
                    val = _mu_0 * current / (2.0 * pi * r)
                    if B.ndim == 4:
                        B[1, ir, :, :] = val
                    else:
                        B[1, ir, :] = val
                elif r < cc.anode_radius:
                    if B.ndim == 4:
                        B[1, ir, :, :] = 0.0
                    else:
                        B[1, ir, :] = 0.0
            self.state["B"] = B

        # Snowplow zipper BC: applies to ALL backends with cylindrical geometry
        if self.geometry_type == "cylindrical" and self.snowplow and self.snowplow.is_active:
            z_sheath = self.snowplow.z
            dz = self.config.geometry.dz if self.config.geometry.dz else self.config.dx
            iz_sheath = round(z_sheath / dz)

            nx, ny, nz = self.config.grid_shape
            B = self.state["B"]
            if 0 <= iz_sheath < nz:
                # Zero B_theta for z > z_sheath (ahead of axial sheath)
                if B.ndim == 4:
                    B[1, :, :, iz_sheath + 1:] = 0.0
                else:  # 3D (3, nr, nz) — squeezed cylindrical
                    B[1, :, iz_sheath + 1:] = 0.0

            # Radial zipper: zero B_theta outside radial shock front
            if self.snowplow.phase in ("radial", "reflected"):
                r_shock = self.snowplow.r_shock
                dr = self.config.dx
                ir_shock = round(r_shock / dr)
                if 0 <= ir_shock < nx:
                    if B.ndim == 4:
                        B[1, ir_shock + 1:, :, :] = 0.0
                    else:  # 3D
                        B[1, ir_shock + 1:, :] = 0.0

    def _initialize_radial_bfield(self) -> None:
        """One-shot B_theta initialization when snowplow enters radial phase.

        Sets B_theta(r) = mu_0 * I / (2*pi*r) for r < r_shock and zero outside,
        closing the snowplow→MHD coupling loop.  Called once at the axial→radial
        phase transition.  The MHD solver then evolves B freely inside r_shock
        while the zipper BC (in ``_apply_electrode_bc``) maintains B_theta = 0
        outside.

        Works for all backends (Python, Metal, Athena++, AthenaK) because it
        writes directly to ``self.state["B"]``, which is always a NumPy array
        regardless of the active fluid solver backend.  Cell-centre radial
        positions are obtained from ``self.fluid.geom.r`` (Python backend) or
        derived from ``self.config.grid_shape`` and ``self.config.dx`` (all
        other backends).

        Physics:
            At the instant the sheath reaches the anode end and begins radial
            implosion, the azimuthal field inside the sheath is that of a
            current-carrying wire: B_theta = mu_0 * I / (2*pi*r).  Outside the
            sheath (thin-sheath approximation), B_theta = 0.
        """
        if self.snowplow is None:
            return
        if self.geometry_type != "cylindrical":
            return

        I_current = abs(self._coupling.current)
        r_shock = self.snowplow.r_shock
        dr = self.config.dx

        # Build cell-centre radial positions.
        # Python CylindricalMHDSolver exposes self.fluid.geom.r; all other
        # backends (Metal, Athena++, AthenaK) derive r from the grid config.
        if hasattr(self.fluid, "geom") and hasattr(self.fluid.geom, "r"):
            r_grid = self.fluid.geom.r
        else:
            nr = self.config.grid_shape[0]
            r_grid = np.array([(ir + 0.5) * dr for ir in range(nr)])

        ir_shock = round(r_shock / dr) if dr > 0 else len(r_grid)
        ir_shock = min(ir_shock, len(r_grid))

        B = self.state["B"]  # shape (3, nr, 1, nz)

        # Inside shock: B_theta = mu_0 * I / (2*pi*r)
        for ir in range(ir_shock):
            r_val = r_grid[ir]
            if r_val > 0:
                B[1, ir, :, :] = _mu_0 * I_current / (2.0 * pi * r_val)
            else:
                # On-axis: B_theta → 0 by symmetry
                B[1, ir, :, :] = 0.0

        # Outside shock: B_theta = 0 (thin-sheath assumption)
        if ir_shock < B.shape[1]:
            B[1, ir_shock:, :, :] = 0.0

        self.state["B"] = B
        self._radial_bfield_initialized = True

        logger.info(
            "Radial B-field initialized: I=%.2e A, r_shock=%.3e m, "
            "ir_shock=%d/%d, B_theta_max=%.2f T",
            I_current, r_shock, ir_shock, len(r_grid),
            float(np.max(np.abs(B[1]))),
        )

    def _dynamic_sheath_pressure(self) -> float:
        """Compute volume-averaged MHD pressure near the sheath/shock front.

        During axial phase: averages pressure for z > z_sheath cells.
        During radial/reflected phase: averages pressure for r < r_shock cells.
        Falls back to config fill_pressure_Pa if snowplow inactive or no valid cells.

        Returns:
            Pressure [Pa] from MHD state, or fill_pressure_Pa as fallback.
        """
        fallback = self.config.snowplow.fill_pressure_Pa
        if self.snowplow is None or not self.snowplow.is_active:
            return fallback

        p = self.state.get("pressure")
        if p is None:
            return fallback

        dr = self.config.dx
        dz = self.config.geometry.dz if self.config.geometry.dz else dr

        if self.snowplow.phase == "rundown":
            # Axial: average pressure ahead of sheath (z > z_sheath)
            iz = round(self.snowplow.z / dz) if dz > 0 else 0
            nz = p.shape[-1]
            if 0 <= iz < nz - 1:
                p_ahead = p[..., iz + 1:]
                if p_ahead.size > 0:
                    return max(float(np.mean(p_ahead)), fallback)
        elif self.snowplow.phase in ("radial", "reflected"):
            # Radial: average pressure inside shock front (r < r_shock)
            ir = round(self.snowplow.r_shock / dr) if dr > 0 else 0
            nx = p.shape[0]
            if 0 < ir <= nx:
                p_inside = p[:ir]
                if p_inside.size > 0:
                    return max(float(np.mean(p_inside)), fallback)

        return fallback

    # ------------------------------------------------------------------
    # Athena++ backend step
    # ------------------------------------------------------------------

    def _step_athena(
        self, dt: float, sim_time: float, _max_steps: int | None
    ) -> StepResult:
        """Simplified timestep using the Athena++ MHD backend.

        Circuit coupling is active (Python RLC solver), but the detailed
        Python physics operators (Spitzer resistivity, Nernst, viscosity,
        radiation) are bypassed.  These will be added as Athena++ source
        terms in Phase G.

        Args:
            dt: Timestep size [s].
            sim_time: Target simulation time [s].
            _max_steps: Optional step limit.

        Returns:
            StepResult with scalar diagnostics.
        """
        # --- Circuit advance ---
        # Use coupling data from Athena++ C++ (R_plasma, L_plasma computed by
        # dpf_zpinch.cpp UserWorkInLoop via volume integrals)
        coupling = self.fluid.coupling_interface()
        coupling.Z_bar = 1.0

        # Track dL/dt for inductive coupling
        L_plasma = coupling.Lp
        if L_plasma > 0:
            if self._prev_L_plasma > 0 and dt > 0:
                coupling.dL_dt = (L_plasma - self._prev_L_plasma) / dt
            self._prev_L_plasma = L_plasma

        back_emf = 0.0  # dL/dt already in R_star inside rlc_solver.py
        new_coupling = self.circuit.step(coupling, back_emf, dt)
        self._coupling = new_coupling

        # --- MHD advance via Athena++ ---
        self.state = self.fluid.step(
            self.state,
            dt,
            current=new_coupling.current,
            voltage=new_coupling.voltage,
        )

        # --- Apply electrode BC post-hoc (Athena++ manages its own BCs but
        # does not know about the DPF circuit-coupled B_theta prescription) ---
        _bc = getattr(self, "boundary_cfg", None)
        if _bc is not None and _bc.electrode_bc:
            self._apply_electrode_bc(new_coupling.current)

        # --- Advance time ---
        self.time += dt
        self.step_count += 1

        # --- Diagnostics ---
        R_plasma = coupling.R_plasma
        self._last_R_plasma = R_plasma
        self._last_Z_bar = 1.0
        self._last_eta_anom = 0.0

        from dpf.diagnostics.pease_braginskii import check_pease_braginskii
        self._last_pb_result = check_pease_braginskii(
            I_current=abs(self._coupling.current),
            Z=1.0,
            gaunt_factor=1.2,
            ln_Lambda=self.config.collision.coulomb_log,
        )

        if self.step_count % self.diag_interval == 0:
            circ = self.circuit.state
            diag_state = {
                **self.state,
                "circuit": {
                    "current": circ.current,
                    "voltage": circ.voltage,
                    "energy_cap": circ.energy_cap,
                    "energy_ind": circ.energy_ind,
                    "energy_res": circ.energy_res,
                    "energy_total": self.circuit.total_energy(),
                },
                "plasma": {
                    "R_plasma": R_plasma,
                    "Z_bar": 1.0,
                    "eta_anomalous": 0.0,
                    "sheath_enabled": False,
                    "geometry": self.geometry_type,
                    "backend": "athena",
                },
            }
            # Athena++ may produce arrays with shapes that the Python
            # diagnostics recorder doesn't expect (e.g. 2D cylindrical
            # with nx3=1).  This is non-fatal.
            with contextlib.suppress(ValueError, IndexError):
                self.diagnostics.record(diag_state, self.time)

        if self.step_count % 100 == 0:
            E_total = self.circuit.total_energy()
            logger.info(
                "Step %d [athena]: t=%.4e s, dt=%.2e s, I=%.1f A, V=%.1f V, E_cons=%.4f",
                self.step_count,
                self.time,
                dt,
                new_coupling.current,
                new_coupling.voltage,
                E_total / max(self.initial_energy, 1e-30),
            )

        # === Step 5c: Well Exporter ===
        if self.well_interval > 0 and self.step_count % self.well_interval == 0:
            self.well_exporter.append_state(self.state, time=self.time)

        # Check if finished
        finished = self.time >= sim_time
        if _max_steps is not None and self.step_count >= _max_steps:
            finished = True

        return self._make_step_result(dt=dt, finished=finished)

    def _make_step_result(self, *, dt: float, finished: bool) -> StepResult:
        """Build a StepResult from current engine state."""
        E_total = self.circuit.total_energy()
        conservation = E_total / max(self.initial_energy or E_total, 1e-30)
        return StepResult(
            time=self.time,
            step=self.step_count,
            dt=dt,
            current=self.circuit.current,
            voltage=self.circuit.voltage,
            energy_conservation=conservation,
            max_Te=float(np.max(self.state["Te"])),
            max_rho=float(np.max(self.state["rho"])),
            Z_bar=getattr(self, "_last_Z_bar", 1.0),
            R_plasma=getattr(self, "_last_R_plasma", 0.0),
            eta_anomalous=getattr(self, "_last_eta_anom", 0.0),
            total_radiated_energy=self.total_radiated_energy,
            neutron_rate=getattr(self, "_last_neutron_rate", 0.0),
            total_neutron_yield=self.total_neutron_yield,
            finished=finished,
        )

    # ------------------------------------------------------------------
    # Strang-split collision + radiation sub-step
    # ------------------------------------------------------------------

    def _apply_collision_radiation(
        self,
        dt_sub: float,
        Z_bar: float,
        *,
        Z_bar_field: np.ndarray | None = None,
    ) -> None:
        """Apply collision (temperature relaxation) and radiation losses.

        This is the combined collision + radiation operator used in Strang
        splitting.  Called twice per timestep with dt/2 each (once before
        and once after the MHD advance) for 2nd-order temporal accuracy.

        Args:
            dt_sub: Sub-step duration [s] (typically dt/2).
            Z_bar: Scalar average ionization state (fallback).
            Z_bar_field: Spatially-varying ionization state array, same shape
                as Te/rho. If provided, used for collision/radiation operators
                instead of scalar Z_bar for improved physics fidelity.
        """
        # --- Collision physics (electron-ion temperature relaxation) ---
        Te = self.state["Te"]
        Ti = self.state["Ti"]
        rho = self.state["rho"]
        ne = rho / self.ion_mass

        col_cfg = self.config.collision
        if col_cfg.dynamic_coulomb_log:
            lnL = coulomb_log(ne, Te)
        else:
            lnL = col_cfg.coulomb_log

        # Use spatially-varying Z for collision rate if available
        Z_for_collisions = Z_bar_field if Z_bar_field is not None else Z_bar
        freq_ei = nu_ei(ne, Te, lnL, Z=Z_for_collisions)
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt_sub)
        self.state["Te"] = Te_new
        self.state["Ti"] = Ti_new

        # Update pressure from new temperatures
        self.state["pressure"] = self.eos.total_pressure(rho, Ti_new, Te_new)
        self._sanitize_state("after collision step")

        # --- Braginskii ion viscosity ---
        if self.config.fluid.enable_viscosity and self.backend != "metal":
            self._apply_viscosity(dt_sub)
            self._sanitize_state("after viscosity step")

        # --- Radiation losses ---
        # Use spatially-varying Z for bremsstrahlung (P ~ Z^2) if available
        Z_for_rad = Z_bar_field if Z_bar_field is not None else Z_bar
        if self.rad_cfg.bremsstrahlung_enabled:
            ne_rad = rho / self.ion_mass
            if self.rad_cfg.fld_enabled:
                self.state = apply_radiation_transport(
                    self.state,
                    dx=self.config.dx,
                    dt=dt_sub,
                    Z=Z_for_rad,
                    gaunt_factor=self.rad_cfg.gaunt_factor,
                )
            else:
                Te_rad, P_rad = apply_bremsstrahlung_losses(
                    self.state["Te"],
                    ne_rad,
                    dt_sub,
                    Z=Z_for_rad,
                    gaunt_factor=self.rad_cfg.gaunt_factor,
                )
                self.state["Te"] = Te_rad
                if self.geometry_type == "cylindrical":
                    cell_vol = self.fluid.geom.cell_volumes()
                    P_rad_2d = np.squeeze(P_rad, axis=1) if P_rad.ndim == 3 else P_rad
                    self.total_radiated_energy += float(
                        np.sum(P_rad_2d * cell_vol) * dt_sub
                    )
                else:
                    self.total_radiated_energy += float(
                        np.sum(P_rad) * np.prod([self.config.dx] * 3) * dt_sub
                    )

            # Update pressure after radiation
            self.state["pressure"] = self.eos.total_pressure(
                self.state["rho"], self.state["Ti"], self.state["Te"]
            )
            self._sanitize_state("after radiation step")

        # --- Line radiation (impurity cooling) ---
        if self.rad_cfg.line_radiation_enabled and self.rad_cfg.impurity_fraction > 0:
            ne_line = self.state["rho"] / self.ion_mass
            Te_line, P_line = apply_line_radiation_losses(
                self.state["Te"],
                ne_line,
                dt_sub,
                Z_eff=0.0,  # bremsstrahlung already applied above; only line + recomb here
                n_imp_frac=self.rad_cfg.impurity_fraction,
                Z_imp=self.rad_cfg.impurity_Z,
                Te_floor=1.0,
            )
            self.state["Te"] = Te_line
            # Track radiated energy from line radiation
            if self.geometry_type == "cylindrical":
                cell_vol = self.fluid.geom.cell_volumes()
                P_line_2d = np.squeeze(P_line, axis=1) if P_line.ndim == 3 else P_line
                self.total_radiated_energy += float(np.sum(P_line_2d * cell_vol) * dt_sub)
            else:
                self.total_radiated_energy += float(
                    np.sum(P_line) * np.prod([self.config.dx] * 3) * dt_sub
                )
            self.state["pressure"] = self.eos.total_pressure(
                self.state["rho"], self.state["Ti"], self.state["Te"]
            )
            self._sanitize_state("after line radiation step")

        # --- Implicit / STS magnetic and thermal diffusion ---
        fc = self.config.fluid
        if fc.diffusion_method != "explicit" and fc.enable_resistive:
            self._apply_diffusion(dt_sub, Z_bar, Z_bar_field=Z_bar_field)
            self._sanitize_state("after diffusion step")

    # ------------------------------------------------------------------
    # Nernst B-field advection sub-step
    # ------------------------------------------------------------------

    def _apply_nernst(self, dt: float, Z_bar: float) -> None:
        """Advect B-field by Nernst velocity (grad Te driven).

        The Nernst effect sweeps magnetic field along electron temperature
        gradients.  It is applied as an operator-split step after the MHD
        advance.

        Args:
            dt: Timestep [s].
            Z_bar: Average ionization state.
        """
        B = self.state["B"]
        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = rho / self.ion_mass

        dx = self.config.dx
        if self.geometry_type == "cylindrical":
            dz = self.config.geometry.dz if self.config.geometry.dz is not None else dx
            # Nernst module uses np.gradient on all 3 axes — needs ny >= 2.
            # Pad ny=1 -> ny=3 by repeating the single slice, then extract back.
            pad_n = 3
            B_pad = np.repeat(B, pad_n, axis=2)          # (3, nr, 3, nz)
            ne_pad = np.repeat(ne, pad_n, axis=1)         # (nr, 3, nz)
            Te_pad = np.repeat(Te, pad_n, axis=1)         # (nr, 3, nz)
            Bx_new, By_new, Bz_new = apply_nernst_advection(
                B_pad[0], B_pad[1], B_pad[2],
                ne_pad, Te_pad, dx, dx, dz, dt,
                Z_eff=max(Z_bar, 0.01),
            )
            # Extract middle y-slice back to (nr, 1, nz)
            Bx_new = Bx_new[:, 1:2, :]
            By_new = By_new[:, 1:2, :]
            Bz_new = Bz_new[:, 1:2, :]
        else:
            Bx_new, By_new, Bz_new = apply_nernst_advection(
                B[0], B[1], B[2],
                ne, Te, dx, dx, dx, dt,
                Z_eff=max(Z_bar, 0.01),
            )

        self.state["B"] = np.array([Bx_new, By_new, Bz_new])

    # ------------------------------------------------------------------
    # Powell 8-wave div(B) source terms
    # ------------------------------------------------------------------

    def _apply_powell_sources(self, dt: float) -> None:
        """Apply Powell 8-wave div(B) source terms.

        These non-conservative source terms help control magnetic field
        divergence by correcting momentum, induction, and energy proportional
        to div(B). They complement Dedner GLM cleaning.

        Reference: Powell et al., J. Comp. Phys. 154, 284 (1999).
        """
        from dpf.fluid.mhd_solver import powell_source_terms, powell_source_terms_cylindrical

        rho = self.state["rho"]
        gamma = self.config.fluid.gamma

        if self.geometry_type == "cylindrical":
            # Squeeze to 2D for cylindrical Powell
            state_2d = {}
            for key, arr in self.state.items():
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 4:  # (3, nr, 1, nz) -> (3, nr, nz)
                        state_2d[key] = np.squeeze(arr, axis=2)
                    elif arr.ndim == 3:  # (nr, 1, nz) -> (nr, nz)
                        state_2d[key] = np.squeeze(arr, axis=1)
                    else:
                        state_2d[key] = arr
                else:
                    state_2d[key] = arr

            powell = powell_source_terms_cylindrical(state_2d, self.fluid.geom)

            # Apply sources (2D) then unsqueeze back
            rho_2d = np.squeeze(rho, axis=1)
            rho_safe = np.maximum(rho_2d, 1e-20)

            vel_2d = np.squeeze(self.state["velocity"], axis=2)
            vel_2d += dt * powell["dmom_powell"] / rho_safe[np.newaxis, :, :]
            self.state["velocity"][:, :, 0, :] = vel_2d

            B_2d = np.squeeze(self.state["B"], axis=2)
            B_2d += dt * powell["dB_powell"]
            self.state["B"][:, :, 0, :] = B_2d

            p_2d = np.squeeze(self.state["pressure"], axis=1)
            p_2d += dt * powell["denergy_powell"] * (gamma - 1.0)
            self.state["pressure"][:, 0, :] = p_2d
        else:
            # Cartesian 3D
            dx = self.config.dx
            powell = powell_source_terms(self.state, dx, dx, dx)

            rho_safe = np.maximum(rho, 1e-20)
            self.state["velocity"] += dt * powell["dmom_powell"] / rho_safe[np.newaxis, :, :, :]
            self.state["B"] += dt * powell["dB_powell"]
            self.state["pressure"] += dt * powell["denergy_powell"] * (gamma - 1.0)

        # Enforce positivity
        self.state["pressure"] = np.maximum(self.state["pressure"], 1e-20)

    # ------------------------------------------------------------------
    # Braginskii ion viscosity sub-step
    # ------------------------------------------------------------------

    def _apply_viscosity(self, dt_sub: float) -> None:
        """Apply Braginskii ion viscosity.

        Updates velocity via viscous stress and adds viscous heating to
        ion temperature.  If ``full_braginskii_viscosity`` is enabled in
        the config, the full anisotropic Braginskii stress tensor
        (eta_0 parallel + eta_1 perpendicular) is used instead of the
        simple isotropic traceless approximation.

        Args:
            dt_sub: Sub-step duration [s] (typically dt/2 from Strang).
        """
        rho = self.state["rho"]
        vel = self.state["velocity"]
        Ti = self.state["Ti"]
        B = self.state["B"]

        ni = rho / self.ion_mass
        tau_i = ion_collision_time(ni, Ti)
        eta0 = braginskii_eta0(ni, Ti, tau_i)

        fc = self.config.fluid
        use_full = fc.full_braginskii_viscosity

        # Compute eta_1 if using full Braginskii
        eta1_field = None
        if use_full:
            B_mag = np.sqrt(np.sum(B**2, axis=0))
            eta1_field = braginskii_eta1(ni, Ti, tau_i, B_mag, self.ion_mass)

        dx = self.config.dx
        if self.geometry_type == "cylindrical":
            dz = self.config.geometry.dz if self.config.geometry.dz is not None else dx
            dy = dx
            # Viscosity module uses finite differences on all 3 axes,
            # which requires ny >= 2.  Pad ny=1 -> ny=3 then extract.
            pad_n = 3
            vel_pad = np.repeat(vel, pad_n, axis=2)       # (3, nr, 3, nz)
            rho_pad = np.repeat(rho, pad_n, axis=1)        # (nr, 3, nz)
            eta0_pad = np.repeat(eta0, pad_n, axis=1)      # (nr, 3, nz)

            if use_full and eta1_field is not None:
                eta1_pad = np.repeat(eta1_field, pad_n, axis=1)
                B_pad = np.repeat(B, pad_n, axis=2)
                accel_pad = viscous_stress_rate(
                    vel_pad, rho_pad, eta0_pad, dx, dy, dz,
                    full_braginskii=True, B=B_pad, eta1=eta1_pad,
                )
            else:
                accel_pad = viscous_stress_rate(vel_pad, rho_pad, eta0_pad, dx, dy, dz)
            Q_visc_pad = viscous_heating_rate(vel_pad, eta0_pad, dx, dy, dz)

            accel = accel_pad[:, :, 1:2, :]     # middle slice
            Q_visc = Q_visc_pad[:, 1:2, :]
            ni_safe = np.maximum(ni, 1e-30)
        else:
            dy = dx
            dz = dx
            if use_full and eta1_field is not None:
                accel = viscous_stress_rate(
                    vel, rho, eta0, dx, dy, dz,
                    full_braginskii=True, B=B, eta1=eta1_field,
                )
            else:
                accel = viscous_stress_rate(vel, rho, eta0, dx, dy, dz)
            Q_visc = viscous_heating_rate(vel, eta0, dx, dy, dz)
            ni_safe = np.maximum(ni, 1e-30)

        self.state["velocity"] = vel + dt_sub * accel

        # Viscous heating: Q_visc -> Ti
        dTi = (2.0 / 3.0) * Q_visc * dt_sub / (ni_safe * k_B)
        self.state["Ti"] = self.state["Ti"] + dTi

        # Update pressure after viscous heating
        self.state["pressure"] = self.eos.total_pressure(
            rho, self.state["Ti"], self.state["Te"]
        )

    # ------------------------------------------------------------------
    # Implicit / STS diffusion sub-step
    # ------------------------------------------------------------------

    def _apply_diffusion(
        self,
        dt_sub: float,
        Z_bar: float,
        *,
        Z_bar_field: np.ndarray | None = None,
    ) -> None:
        """Apply implicit or super-time-stepping magnetic and thermal diffusion.

        Called from _apply_collision_radiation when diffusion_method != 'explicit'.
        Solves the resistive diffusion dB/dt = (eta/mu_0)*Laplacian(B) and
        thermal conduction dTe/dt = kappa/(1.5*ne*kB)*Laplacian(Te) using either
        Crank-Nicolson ADI or RKL2 super time-stepping.

        Args:
            dt_sub: Sub-step duration [s] (typically dt/2 from Strang).
            Z_bar: Scalar average ionization state (fallback).
            Z_bar_field: Spatially-varying ionization state array, if available.
        """
        fc = self.config.fluid
        dx = self.config.dx
        B = self.state["B"]
        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = np.maximum(rho / self.ion_mass, 1e10)

        # Compute resistivity field for diffusion coefficient
        # Use spatially-varying Z for accurate Spitzer resistivity
        Z_for_diff = Z_bar_field if Z_bar_field is not None else Z_bar
        Te_safe = np.maximum(Te, 1000.0)
        ne_safe = np.maximum(ne, 1e10)
        from dpf.collision.spitzer import spitzer_resistivity as _spitz_eta
        lnL = coulomb_log(ne_safe, Te_safe)
        eta = _spitz_eta(ne_safe, Te_safe, lnL, Z=Z_for_diff)

        # Compute Spitzer thermal conductivity: kappa_e ~ 3.2 * ne * kB^2 * Te * tau_e / m_e
        # Simplified estimate: kappa ~ 20 * (kB * Te)^{5/2} / (m_e^{1/2} * e^4 * lnL)
        # For now, use a simplified isotropic Spitzer kappa
        from dpf.constants import m_e
        tau_e = 3.44e5 * Te_safe**1.5 / (ne_safe * lnL)  # Approximate electron collision time
        kappa = 3.2 * ne_safe * k_B**2 * Te_safe * tau_e / m_e

        if self.geometry_type == "cylindrical":
            dz = self.config.geometry.dz if self.config.geometry.dz is not None else dx
            dy = dx
        else:
            dy = dx
            dz = dx

        if fc.diffusion_method == "implicit":
            # Crank-Nicolson ADI for magnetic diffusion
            Bx_new, By_new, Bz_new = implicit_resistive_diffusion(
                B[0], B[1], B[2], eta, dt_sub, dx, dy, dz,
            )
            self.state["B"] = np.array([Bx_new, By_new, Bz_new])

            # Crank-Nicolson ADI for thermal diffusion
            Te_new = implicit_thermal_diffusion(Te, kappa, ne, dt_sub, dx, dy, dz)
            self.state["Te"] = np.maximum(Te_new, 1.0)

        elif fc.diffusion_method == "sts":
            # RKL2 super time-stepping for magnetic diffusion
            s = fc.sts_stages
            Bx_new, By_new, Bz_new = rkl2_diffusion_3d(
                B[0], B[1], B[2], eta, dt_sub, dx, dy, dz, s_stages=s,
            )
            self.state["B"] = np.array([Bx_new, By_new, Bz_new])

            # RKL2 for thermal diffusion
            Te_new = rkl2_thermal_step(Te, kappa, ne, dt_sub, dx, s_stages=s)
            self.state["Te"] = np.maximum(Te_new, 1.0)

        # --- Anisotropic thermal conduction (field-aligned Braginskii) ---
        if fc.enable_anisotropic_conduction and self.backend != "metal":
            from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction
            B_ac = self.state["B"]
            Te_ac = self.state["Te"]
            ne_ac = np.maximum(self.state["rho"] / self.ion_mass, 1e10)
            # Anisotropic conduction accepts scalar Z_eff
            Z_eff_aniso = max(Z_bar, 0.01)
            Te_aniso = anisotropic_thermal_conduction(
                Te_ac, B_ac, ne_ac, dt_sub, dx, dy, dz,
                Z_eff=Z_eff_aniso,
            )
            self.state["Te"] = np.maximum(Te_aniso, 1.0)

        # Update pressure from new Te
        self.state["pressure"] = self.eos.total_pressure(
            rho, self.state["Ti"], self.state["Te"]
        )

    # ------------------------------------------------------------------
    # Field snapshot access (for server/GUI)
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalize the simulation and flush exporters."""
        if hasattr(self, "well_exporter"):
            self.well_exporter.close()

    def get_field_snapshot(self) -> dict[str, np.ndarray]:
        """Return a copy of the current field state arrays.

        Returns:
            Dictionary with copies of rho, velocity, pressure, B, Te, Ti, psi.
        """
        return {key: arr.copy() for key, arr in self.state.items()}

    # ------------------------------------------------------------------
    # Batch run (uses step() internally)
    # ------------------------------------------------------------------

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        """Execute the simulation loop.

        Args:
            max_steps: Maximum number of timesteps (None = run to sim_time).

        Returns:
            Dictionary with summary statistics.
        """
        # Hybrid engine delegation (live integration)
        if self.backend == "hybrid":
            if self._hybrid_engine is None:
                from dpf.ai.hybrid_engine import HybridEngine
                from dpf.ai.surrogate import DPFSurrogate

                # Check config
                ckpt = self.config.ai.surrogate_checkpoint if self.config.ai else None
                handoff = self.config.fluid.handoff_fraction
                val_interval = self.config.fluid.validation_interval

                logger.info(
                    "Switching to HybridEngine (handoff=%.0f%%, validation=%d)",
                    handoff * 100, val_interval
                )

                surrogate = DPFSurrogate(checkpoint_path=ckpt, device=self.config.ai.device if self.config.ai else "cpu")
                self._hybrid_engine = HybridEngine(
                    config=self.config,
                    surrogate=surrogate,
                    handoff_fraction=handoff,
                    validation_interval=val_interval,
                )

            return self._hybrid_engine.run(max_steps=max_steps)

        t_wall_start = wall_time.monotonic()

        # Store initial energy
        self.initial_energy = self.circuit.total_energy()

        # Peak current tracking
        self._peak_current_A = 0.0
        self._peak_current_time_s = 0.0

        logger.info("Starting simulation: t_end=%.2e s", self.config.sim_time)

        while True:
            result = self.step(_max_steps=max_steps)

            # Track peak current
            I_abs = abs(self.circuit.current)
            if I_abs > self._peak_current_A:
                self._peak_current_A = I_abs
                self._peak_current_time_s = self.time

            if result.finished:
                break

        # Finalize
        self.diagnostics.finalize()

        t_wall = wall_time.monotonic() - t_wall_start
        E_final = self.circuit.total_energy()
        conservation = E_final / max(self.initial_energy, 1e-30)

        summary = {
            "steps": self.step_count,
            "sim_time": self.time,
            "wall_time_s": t_wall,
            "energy_conservation": conservation,
            "final_current_A": self.circuit.current,
            "final_voltage_V": self.circuit.voltage,
            "total_radiated_energy_J": self.total_radiated_energy,
            "total_neutron_yield": self.total_neutron_yield,
            "peak_current_A": self._peak_current_A,
            "peak_current_time_s": self._peak_current_time_s,
        }

        logger.info(
            "Simulation complete: %d steps in %.2f s (%.1f steps/s), E_cons=%.6f",
            self.step_count,
            t_wall,
            self.step_count / max(t_wall, 1e-10),
            conservation,
        )

        return summary
