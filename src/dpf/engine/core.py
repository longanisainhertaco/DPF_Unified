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

import logging
import time as wall_time
from typing import Any

import numpy as np

from dpf.atomic.ionization import saha_ionization_fraction_array
from dpf.circuit.coupler import CircuitCoupler, FeedbackResult
from dpf.circuit.rlc_solver import RLCSolver
from dpf.collision.spitzer import coulomb_log, spitzer_resistivity
from dpf.config import SimulationConfig
from dpf.constants import k_B, pi
from dpf.constants import mu_0 as _mu_0
from dpf.core.bases import CouplingState, StepResult

# FieldManager (Phase 5)
from dpf.core.field_manager import FieldManager
from dpf.diagnostics.energy_balance import EnergyTracker
from dpf.diagnostics.hdf5_writer import HDF5Writer
from dpf.diagnostics.yield_tracker import YieldTracker
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
from dpf.fluid.eos import IdealEOS
from dpf.fluid.mhd_solver import MHDSolver
from dpf.fluid.snowplow import SnowplowModel
from dpf.kinetic.manager import KineticManager
from dpf.sheath.bohm import apply_sheath_bc, floating_potential
from dpf.turbulence.anomalous import (
    anomalous_resistivity_field,
    anomalous_resistivity_scalar,
    total_resistivity,
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
            crowbar_inductance=cc.crowbar_inductance,
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
            # TODO: dpf_zpinch.cpp needs C++ source function to inject circuit
            # B-field. Python-side electrode_bc segfaults because pybind11 arrays
            # are read-only views. Fix requires: AllocateRealUserMeshDataField(6),
            # EnrollUserExplicitSourceFunction, DPFCircuitSourceFunc applying
            # B_theta = mu0*I/(2*pi*r). See observations 2026-03-22.
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
                enable_hall=fc.enable_hall,
                enable_braginskii_conduction=fc.enable_anisotropic_conduction,
                enable_braginskii_viscosity=fc.enable_viscosity,
                enable_nernst=fc.enable_nernst,
                enable_bremsstrahlung=getattr(
                    config.radiation, "bremsstrahlung_enabled", False
                ),
                ion_mass=self.ion_mass,
                coordinates=self.geometry_type,
                r_inner=config.circuit.anode_radius if self.geometry_type == "cylindrical" else None,
                convert_b_si_to_hl=self.geometry_type == "cylindrical",
            )
            # Attach cylindrical geometry provider for diagnostics
            if self.geometry_type == "cylindrical":
                from dpf.geometry.cylindrical import CylindricalGeometry
                self.fluid.geom = CylindricalGeometry(
                    nr=nx, nz=nz, dr=dx, dz=dz,
                )
            if self.geometry_type == "cylindrical" and hasattr(self.fluid, 'geom'):
                self._cell_volume = self.fluid.geom.cell_volumes()[:, np.newaxis, :]
            else:
                self._cell_volume = dx * dx * dz
            logger.info(
                "Using Metal GPU backend (PyTorch MPS, %s, %s+%s+%s)",
                self.geometry_type, fc.reconstruction,
                fc.riemann_solver, fc.time_integrator,
            )
        elif self.backend == "mlx":
            from dpf.metal.mlx_solver import MLXMHDSolver
            dz = config.geometry.dz if config.geometry.dz is not None else dx
            self.fluid = MLXMHDSolver(
                grid_shape=(nx, ny, nz),
                dx=dx,
                dz=dz,
                gamma=fc.gamma,
                cfl=fc.cfl,
                riemann_solver=fc.riemann_solver,
                reconstruction=fc.reconstruction,
                time_integrator=fc.time_integrator,
                coordinates=self.geometry_type,
                r_inner=config.circuit.anode_radius if self.geometry_type == "cylindrical" else 0.0,
                convert_b_si_to_hl=self.geometry_type == "cylindrical",
                ion_mass=self.ion_mass,
                enable_bremsstrahlung=getattr(config.radiation, "bremsstrahlung_enabled", False),
                resistivity_model=fc.resistivity_model,
                anomalous_resistivity=fc.anomalous_resistivity,
            )
            if self.geometry_type == "cylindrical":
                from dpf.geometry.cylindrical import CylindricalGeometry
                self.fluid.geom = CylindricalGeometry(
                    nr=nx, nz=nz, dr=dx, dz=dz,
                )
                self._cell_volume = self.fluid.geom.cell_volumes()[:, np.newaxis, :]
            else:
                self._cell_volume = dx * dx * dz
            logger.info("Using MLX Metal backend (Apple Silicon GPU)")
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
                enable_hall=fc.enable_hall,
                enable_resistive=fc.enable_resistive,
                enable_energy_equation=fc.enable_energy_equation,
                ion_mass=self.ion_mass,
                riemann_solver=fc.riemann_solver,
                conservative_energy=fc.conservative_energy,
                use_godunov_flux=fc.use_godunov_flux,
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

        # Snowplow-MHD mass tracking
        self._prev_swept_mass: float = 0.0
        self._prev_radial_swept_mass: float = 0.0
        # Lp handoff blending state
        self._lp_blend_alpha: float = 0.0  # 0 = snowplow, 1 = MHD
        self._lp_blend_active: bool = False
        self._skip_next_fluid_step: bool = False  # skip MHD on handoff step (CFL violation)
        self._initial_grid_mass: float = self._compute_grid_mass()

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
        self._energy_tracker = EnergyTracker(gamma=config.fluid.gamma)
        self._last_conservation_error: float = 0.0

        # Neutron yield tracking
        self.total_neutron_yield: float = 0.0
        self._last_neutron_rate: float = 0.0
        self._yield_tracker = YieldTracker(
            ion_mass=self.ion_mass,
            rho0=float(config.fluid.rho0) if hasattr(config.fluid, "rho0") else 1e-4,
        )
        self._last_bt_fraction: float = 0.0

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

        # Performance: NaN check stride (run every N steps; step 0 always runs)
        self._nan_check_stride: int = getattr(config, "nan_check_stride", 10)

        # Performance: coupling integral subcycling (R_plasma, L_plasma, Z_bar)
        # Alfven timescale ~100-500 ns >> dt_mhd ~1-5 ns → safe to cache for N steps
        self._coupling_cache_stride: int = 10
        self._cached_R_plasma: float = 0.0
        self._cached_L_plasma: float = 0.0
        self._cached_Z_bar: float = 1.0
        self._cached_Z_bar_field: np.ndarray | None = None
        self._cached_eta_field: np.ndarray | None = None
        self._cached_eta_anom: float = 0.0

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

        # Circuit-MHD coupler: density-weighted Lp extraction from MHD fields
        self.coupling_mode = config.circuit.coupling_mode
        dz_coupler = config.geometry.dz if config.geometry.dz else dx
        if self.geometry_type == "cylindrical":
            self.coupler = CircuitCoupler(
                anode_radius=cc.anode_radius,
                cathode_radius=cc.cathode_radius,
                dr=dx,
                dz=dz_coupler,
                r_inner=cc.anode_radius,
            )
        else:
            self.coupler = CircuitCoupler(
                anode_radius=cc.anode_radius,
                cathode_radius=cc.cathode_radius,
                dr=dx,
                dz=dz_coupler,
            )
        self._last_feedback: FeedbackResult | None = None

        # Perf: cache _should_use_coupler result, recompute every 10 steps
        self._coupler_decision_cache: bool | None = None

        # Suppress repeated MHD regime validity warnings
        self._mhd_regime_warned: bool = False

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
        elif self.backend in ("metal", "mlx"):
            # Metal/MLX share the engine operator-split loop (radiation, sheath,
            # collision) AND have their own transport physics (HLLD, WENO5-Z,
            # SSP-RK3, dual-energy, bremsstrahlung).
            metal_only = []
            if fc.diffusion_method == "sts":
                metal_only.append("RKL2 super time-stepping (using explicit instead)")
            if fc.diffusion_method == "implicit":
                metal_only.append("implicit diffusion (using explicit instead)")
            if metal_only:
                logger.info(
                    "%s backend note: %s", self.backend, ", ".join(metal_only),
                )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Backend dispatch — extracted to backend_dispatch.py
    # ------------------------------------------------------------------

    @property
    def engine_tier(self) -> str:
        from dpf.engine.backend_dispatch import engine_tier
        return engine_tier(self.backend)

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        from dpf.engine.backend_dispatch import resolve_backend
        return resolve_backend(requested)

    def _initial_state(self, nx: int, ny: int, nz: int) -> dict[str, np.ndarray]:
        """Create initial plasma state (uniform fill gas)."""
        rho0 = self.config.rho0
        T0 = self.config.T0
        # Total pressure = p_e + p_i = n_i*k_B*Te + n_i*k_B*Ti
        # With Te = Ti = T0: p_total = 2 * n_i * k_B * T0
        n_i = rho0 / self.ion_mass
        p0 = 2.0 * n_i * k_B * T0

        state = {
            "rho": np.full((nx, ny, nz), rho0),
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.full((nx, ny, nz), p0),
            "B": np.zeros((3, nx, ny, nz)),
            "Te": np.full((nx, ny, nz), T0),
            "Ti": np.full((nx, ny, nz), T0),
            "psi": np.zeros((nx, ny, nz)),  # Dedner cleaning scalar
        }
        if self.config.fluid.two_temperature:
            from dpf.fluid.two_temperature import initialize_electron_energy
            state["e_electron"] = initialize_electron_energy(
                state["Te"], state["Ti"], state["pressure"],
                state["rho"], self.ion_mass,
            )
        return state

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

        # Floor: prevent CFL collapse from freezing the simulation.
        # Only applies to Metal/MLX backends where sharp BCs can drive
        # dt -> 0 via extreme wavespeeds in 1-2 vacuum cells.
        # Python backend has its own stability requirements; don't override.
        backend = getattr(getattr(self.config, "fluid", None), "backend", "python")
        if backend in ("metal", "mlx") and dt < 1e-12:
            dt = 1e-12

        return dt
    # ------------------------------------------------------------------
    # State management — extracted to state_management.py
    # ------------------------------------------------------------------
    from dpf.engine.state_management import (  # noqa: E402
        _make_step_result,
        _sanitize_state,
        _step_diagnostics_and_yield,
        _step_record_and_checkpoint,
        load_from_checkpoint,
    )
    from dpf.engine.state_management import (
        save_checkpoint as _save_checkpoint_impl,
    )
    save_checkpoint = _save_checkpoint_impl



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
        if self.backend in ("athena", "athenak", "hybrid"):
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

        self._step_init_fields()

        # === Step 1: Ionization state, resistivity, R_plasma, L_plasma ===
        Z_bar, Z_bar_field, eta_field, J_field, R_plasma, L_plasma, eta_anom = (
            self._step_ionization_and_resistivity()
        )

        # === Step 1b: Collision+Radiation (first half-step of Strang) ===
        self._apply_collision_radiation(dt / 2.0, Z_bar, Z_bar_field=Z_bar_field)

        # === Step 2: Circuit sub-cycle (snowplow + inductance coupling) ===
        new_coupling = self._step_circuit_subcycle(dt, R_plasma, L_plasma, Z_bar)

        # === Step 2.5: Kinetic / PIC step ===
        self._step_pic(dt, eta_field, J_field)

        # === Step 3 + 3.1 + 3.5: Fluid/MHD advance, ablation, Powell ===
        self._step_fluid_advance(dt, eta_field, new_coupling)

        # === Steps 3a–4+5: Nernst, poloidal B, sheath BC, second Strang half ===
        self._step_post_fluid_corrections(dt, Z_bar, new_coupling)

        # === Steps 5a2–5d: Energy balance, yield, instability, diagnostics ===
        neutron_rate = self._step_diagnostics_and_yield(dt, Z_bar)

        # === Step 6: Advance time, record diagnostics, checkpoint ===
        return self._step_record_and_checkpoint(
            dt, sim_time, _max_steps,
            Z_bar, R_plasma, eta_anom, new_coupling, neutron_rate,
        )

    # ------------------------------------------------------------------
    # step() sub-methods — pure refactoring, no physics changes
    # ------------------------------------------------------------------

    def _step_init_fields(self) -> None:
        """Phase 5 pre-step: sync field manager and apply electrode BC."""
        self.field_manager.B = self.state["B"]
        L_p = self.field_manager.compute_plasma_inductance(self.circuit.current)
        self._prev_L_plasma = L_p
        if self.boundary_cfg.electrode_bc and self.backend not in ("mlx",):
            self._apply_electrode_bc(self._coupling.current)

    def _step_ionization_and_resistivity(
        self,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        float,
        float,
        float,
    ]:
        """Step 1: Ionization state, Spitzer+anomalous resistivity, R/L_plasma.

        Coupling integrals (R_plasma, L_plasma, Z_bar, eta_field) are subcycled:
        recomputed every _coupling_cache_stride steps (default 10) since the
        Alfven timescale (~100-500 ns) >> dt_mhd (~1-5 ns). J_field is only
        returned on recompute steps; cache-hit steps return J_field=None.

        Returns:
            Z_bar: volume-averaged ionization fraction
            Z_bar_field: spatially-resolved ionization field
            eta_field: spatially-resolved resistivity (None if Te/ne too low)
            J_field: current density (3, nx, ny, nz); None on cache-hit steps
            R_plasma: volume-integral plasma resistance [Ohm]
            L_plasma: volume-integral plasma inductance [H]
            eta_anom: scalar anomalous resistivity for diagnostics
        """
        stride = self._coupling_cache_stride
        cache_miss = (self.step_count == 0) or (self.step_count % stride == 0)

        if not cache_miss:
            # Return cached values; J_field not needed (PIC handles None)
            return (
                self._cached_Z_bar,
                self._cached_Z_bar_field if self._cached_Z_bar_field is not None
                    else np.full_like(self.state["Te"], self._cached_Z_bar),
                self._cached_eta_field,
                None,
                self._cached_R_plasma,
                self._cached_L_plasma,
                self._cached_eta_anom,
            )

        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = rho / self.ion_mass

        Te_avg = float(np.mean(Te))
        ne_avg = float(np.mean(ne))

        Z_bar_field = saha_ionization_fraction_array(Te.ravel(), ne.ravel()).reshape(Te.shape)
        Z_bar_field = np.maximum(Z_bar_field, 0.01)
        Z_bar = max(float(np.mean(Z_bar_field)), 0.01)

        eta_field = None
        J_field = None
        eta_anom = 0.0
        R_plasma = 0.0
        L_plasma = 0.0

        if Te_avg > 1000.0 and ne_avg > 1e10:
            Te_floored = np.maximum(Te, 1000.0)
            ne_floored = np.maximum(ne, 1e10)
            lnL_field = coulomb_log(ne_floored, Te_floored)
            eta_spitzer_field = spitzer_resistivity(
                ne_floored, Te_floored, lnL_field, Z=Z_bar_field,
            )

            lnL_avg = coulomb_log(np.array([ne_avg]), np.array([Te_avg]))[0]
            eta_spitzer_avg = float(spitzer_resistivity(
                np.array([ne_avg]), np.array([Te_avg]), lnL_avg, Z=Z_bar
            )[0])

            I_current = self._coupling.current
            A_column = pi * self.anode_radius**2
            B_field = self.state["B"]
            Ti_field = self.state["Ti"]

            if self.geometry_type == "cylindrical":
                B_2d = np.squeeze(B_field, axis=2) if B_field.ndim == 4 else B_field
                curl_B = self.fluid.geom.curl(B_2d)
                J_field_2d = curl_B / _mu_0
                J_field = J_field_2d[:, :, np.newaxis, :]
                J_mag = np.sqrt(np.sum(J_field_2d**2, axis=0))
                ne_2d = np.squeeze(ne, axis=1) if ne.ndim == 3 else ne
                Ti_2d = (
                    np.squeeze(Ti_field, axis=1) if Ti_field.ndim == 3 else Ti_field
                )
                Te_2d = (
                    np.squeeze(self.state["Te"], axis=1)
                    if self.state["Te"].ndim == 3
                    else self.state["Te"]
                )
                eta_anom_field = anomalous_resistivity_field(
                    J_mag, np.maximum(ne_2d, 1e10), np.maximum(Ti_2d, 1.0),
                    alpha=self.config.anomalous_alpha,
                    mi=self.ion_mass,
                    threshold_model=self.config.anomalous_threshold_model,
                    Te=np.maximum(Te_2d, 1.0),
                )
                eta_field = eta_spitzer_field + eta_anom_field[:, np.newaxis, :]

                # CIV anomalous resistivity (velocity-driven, active during axial rundown)
                if getattr(self.config, "anomalous_civ_enabled", False):
                    from dpf.turbulence.anomalous import CIV_VCRIT, civ_anomalous_resistivity
                    vel = self.state.get("velocity")
                    if vel is not None:
                        v2d = np.squeeze(vel, axis=2) if vel.ndim == 4 else vel
                        v_bulk = np.sqrt(np.sum(v2d**2, axis=0))
                        B2d = np.squeeze(B_field, axis=2) if B_field.ndim == 4 else B_field
                        B_mag = np.sqrt(np.sum(B2d**2, axis=0))
                        gas = getattr(self.config, "fill_gas", "deuterium")
                        v_crit = CIV_VCRIT.get(gas, 38500.0)
                        eta_civ = civ_anomalous_resistivity(
                            v_bulk, np.maximum(ne_2d, 1e10), B_mag,
                            mi=self.ion_mass,
                            alpha_civ=getattr(self.config, "anomalous_civ_alpha", 0.05),
                            v_crit=v_crit,
                        )
                        eta_field = eta_field + eta_civ[:, np.newaxis, :]
            else:
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

            J_avg = abs(I_current) / max(A_column, 1e-30)
            Ti_avg = float(np.mean(self.state["Ti"]))
            Te_avg_scalar = float(np.mean(self.state["Te"]))
            eta_anom = anomalous_resistivity_scalar(
                J_avg, ne_avg, Ti_avg, alpha=self.config.anomalous_alpha,
                mi=self.ion_mass,
                threshold_model=self.config.anomalous_threshold_model,
                Te_val=Te_avg_scalar,
            )
            eta_total_avg = total_resistivity(eta_spitzer_avg, eta_anom)

            eta_field = np.where(np.isfinite(eta_field), eta_field, eta_total_avg)
            eta_field = np.minimum(eta_field, 1.0)

            # Volume-integral R_plasma: R = integral(eta*|J|^2 dV) / I^2
            I_sq = max(I_current**2, 1e-30)
            if self.geometry_type == "cylindrical":
                cell_vol = self.fluid.geom.cell_volumes()
                eta_2d = np.squeeze(eta_field, axis=1) if eta_field.ndim == 3 else eta_field
                R_plasma = float(np.sum(eta_2d * J_mag**2 * cell_vol)) / I_sq
            else:
                dV = self.config.dx**3
                R_plasma = float(np.sum(eta_field * np.sum(J_field**2, axis=0) * dV)) / I_sq
            R_plasma = min(R_plasma, 10.0)

            # Volume-integral L_plasma: L = 2 * integral(B^2/(2*mu_0) dV) / I^2
            B_sq = np.sum(B_field**2, axis=0)
            if self.geometry_type == "cylindrical":
                B_sq_2d = np.squeeze(B_sq, axis=1) if B_sq.ndim == 3 else B_sq
                L_plasma = float(np.sum(B_sq_2d / _mu_0 * cell_vol)) / I_sq
            else:
                L_plasma = float(np.sum(B_sq / _mu_0 * dV)) / I_sq

        # Update cache
        self._cached_Z_bar = Z_bar
        self._cached_Z_bar_field = Z_bar_field
        self._cached_eta_field = eta_field
        self._cached_eta_anom = eta_anom
        self._cached_R_plasma = R_plasma
        self._cached_L_plasma = L_plasma

        return Z_bar, Z_bar_field, eta_field, J_field, R_plasma, L_plasma, eta_anom
    # ------------------------------------------------------------------
    # Circuit coupling — extracted to circuit_coupling.py
    # ------------------------------------------------------------------
    from dpf.engine.circuit_coupling import (  # noqa: E402
        _apply_electrode_bc,
        _compute_back_emf,
        _compute_grid_mass,
        _compute_J_from_B,
        _compute_ohmic_correction,
        _compute_snowplow_source_terms,
        _dynamic_sheath_pressure,
        _initialize_radial_bfield,
        _initialize_radial_state,
        _measure_ohmic_gap,
        _should_use_coupler,
        _step_circuit_subcycle,
    )



    def _step_pic(
        self,
        dt: float,
        eta_field: np.ndarray | None,
        J_field: np.ndarray | None,
    ) -> None:
        """Step 2.5: Kinetic/PIC step — provides J_kin source terms for MHD."""
        if self.kinetic and self.kinetic.kc.enabled:
            self.kinetic.update_mhd_state(self.state)

            B_fld = np.moveaxis(self.state["B"], 0, -1)
            v = np.moveaxis(self.state["velocity"], 0, -1)
            E_fld = -np.cross(v, B_fld)
            if J_field is not None and eta_field is not None:
                J_fld = np.moveaxis(J_field, 0, -1)
                E_fld = E_fld + eta_field[..., np.newaxis] * J_fld

            self.kinetic.step(dt, self.time, E_fld, B_fld)

            Jx, Jy, Jz = self.kinetic.get_current_density()
            self._current_source_terms = {"J_kin": np.stack([Jx, Jy, Jz], axis=0)}
        else:
            self._current_source_terms = None

    def _step_fluid_advance(
        self,
        dt: float,
        eta_field: np.ndarray | None,
        new_coupling: CouplingState,
    ) -> None:
        """Steps 3, 3.1, 3.5: MHD fluid advance, ablation, Powell div(B) sources."""
        # Skip MHD on the handoff step — the radial init dramatically changes
        # wave speeds and the pre-handoff dt violates CFL by ~500×.
        # The next step will recompute dt from the new state.
        if self._skip_next_fluid_step:
            self._skip_next_fluid_step = False
            logger.info("Skipping fluid advance on handoff step (CFL recomputation needed)")
            return

        # Ohmic correction: measure gap before fluid step, inject in same step
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

        if self.config.snowplow.enable_mhd_coupling and self.snowplow is not None:
            # In radial_mhd mode, only inject source terms during radial+ phases
            # (snowplow handles rundown perfectly — MHD source terms during rundown
            # cause numerical instability on the non-conservative Python solver)
            inject_ok = True
            # Source terms inject during ALL phases to create the current sheath
            # structure that confines the ghost-cell B_theta. Without the sheath,
            # the ghost BC drives uniform compression of the entire domain.
            if inject_ok:
                sp_src = self._compute_snowplow_source_terms(dt)
                if sp_src:
                    src = self._current_source_terms or {}
                    src.update(sp_src)
                    self._current_source_terms = src

        cc = self.config.circuit
        # MHD sub-stepping: if the fluid CFL dt is much smaller than the engine
        # dt (after radial handoff, CFL can be ~ps vs engine dt ~μs), sub-step.
        # Cap at 100 sub-steps to keep wall time reasonable.
        dt_fluid_cfl = self.fluid._compute_dt(self.state)
        if dt_fluid_cfl > 0 and dt_fluid_cfl < dt:
            n_mhd_sub = max(1, min(int(np.ceil(dt / dt_fluid_cfl)), 100))
        else:
            n_mhd_sub = 1
        dt_mhd = dt / n_mhd_sub
        for _mhd_sub in range(n_mhd_sub):
            # Ghost-cell BC only on first sub-step — subsequent sub-steps
            # evolve the domain freely. Repeated ghost-cell application
            # across sub-steps creates cumulative energy injection.
            apply_bc_this_sub = self.boundary_cfg.electrode_bc and _mhd_sub == 0
            self.state = self.fluid.step(
                self.state,
                dt_mhd,
                current=new_coupling.current,
                voltage=new_coupling.voltage,
                eta_field=eta_field,
                source_terms=self._current_source_terms if _mhd_sub == 0 else None,
                anode_radius=cc.anode_radius,
                cathode_radius=cc.cathode_radius,
                apply_electrode_bc=apply_bc_this_sub,
            )
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
            self._sanitize_state("after fluid step")
        self._last_div_B = getattr(self.fluid, "_last_div_B", 0.0)

        # Step 3.1: Ablation operator-split
        if self.config.ablation.enabled:
            from dpf.atomic.ablation import ablation_source_array
            I_abl = abs(new_coupling.current)
            A_col = pi * self.anode_radius**2
            J_bdy = I_abl / max(A_col, 1e-30)
            J_arr = np.full_like(self.state["rho"], J_bdy)
            mask = np.zeros(self.state["rho"].shape, dtype=np.int64)
            mask[0, :, :] = 1
            eta_abl = eta_field if eta_field is not None else np.full_like(
                self.state["rho"], 1e-7,
            )
            S_rho = ablation_source_array(
                J_arr, eta_abl, self.config.ablation.efficiency, mask,
            )
            self.state["rho"] = self.state["rho"] + S_rho * dt
            if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
                self._sanitize_state("after ablation step")

        # Step 3.5: Powell 8-wave div(B) source terms
        if self.config.fluid.enable_powell:
            self._apply_powell_sources(dt)
            if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
                self._sanitize_state("after Powell step")

    def _step_post_fluid_corrections(
        self,
        dt: float,
        Z_bar: float,
        new_coupling: CouplingState,
    ) -> None:
        """Steps 3a-4+5: Nernst, poloidal B, sheath BC, second Strang half-step."""
        fc = self.config.fluid

        # Step 3a: Nernst B-field advection
        if fc.enable_nernst and self.backend != "metal":
            self._apply_nernst(dt, Z_bar)
            if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
                self._sanitize_state("after Nernst step")

        # Step 3a2: Auluck poloidal B-field (EXPERIMENTAL)
        if fc.enable_poloidal:
            from dpf.experimental.poloidal_bfield import add_poloidal_field
            cc = self.config.circuit
            dx = self.config.dx
            dz_cfg = self.config.geometry.dz
            dz = dz_cfg if dz_cfg else dx
            self.state = add_poloidal_field(
                self.state, self.circuit.current,
                cc.anode_radius, cc.cathode_radius,
                self.config.rho0, dx, dz,
            )

        # Step 3b: Sheath boundary conditions
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

        # Steps 4+5: second Strang half-step — recompute Z_bar from post-MHD Te
        Te_post = self.state["Te"]
        ne_post = self.state["rho"] / self.ion_mass
        Z_bar_field_post = saha_ionization_fraction_array(
            Te_post.ravel(), ne_post.ravel(),
        ).reshape(Te_post.shape)
        Z_bar_field_post = np.maximum(Z_bar_field_post, 0.01)
        Z_bar_post = max(float(np.mean(Z_bar_field_post)), 0.01)
        self._apply_collision_radiation(dt / 2.0, Z_bar_post, Z_bar_field=Z_bar_field_post)
    # ------------------------------------------------------------------
    # Athena fast path — extracted to athena_step.py
    # ------------------------------------------------------------------
    from dpf.engine.athena_step import _step_athena  # noqa: E402

    # ------------------------------------------------------------------
    # Physics operators — extracted to physics_operators.py
    # ------------------------------------------------------------------
    from dpf.engine.physics_operators import (  # noqa: E402
        _apply_collision_radiation,
        _apply_diffusion,
        _apply_nernst,
        _apply_powell_sources,
        _apply_viscosity,
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
