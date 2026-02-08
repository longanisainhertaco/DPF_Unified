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

from dpf.atomic.ionization import saha_ionization_fraction
from dpf.circuit.rlc_solver import RLCSolver
from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures, spitzer_resistivity
from dpf.config import SimulationConfig
from dpf.constants import k_B, pi
from dpf.constants import mu_0 as _mu_0
from dpf.core.bases import CouplingState, StepResult
from dpf.diagnostics.checkpoint import load_checkpoint, save_checkpoint
from dpf.diagnostics.hdf5_writer import HDF5Writer
from dpf.diagnostics.interferometry import abel_transform, fringe_shift
from dpf.diagnostics.neutron_yield import neutron_yield_rate
from dpf.fluid.cylindrical_mhd import CylindricalMHDSolver
from dpf.fluid.eos import IdealEOS
from dpf.fluid.implicit_diffusion import implicit_resistive_diffusion, implicit_thermal_diffusion
from dpf.fluid.mhd_solver import MHDSolver
from dpf.fluid.nernst import apply_nernst_advection
from dpf.fluid.super_time_step import rkl2_diffusion_3d, rkl2_thermal_step
from dpf.fluid.viscosity import (
    braginskii_eta0,
    braginskii_eta1,
    ion_collision_time,
    viscous_heating_rate,
    viscous_stress_rate,
)
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
        )

        # Fluid / MHD — select solver based on geometry and backend
        fc = config.fluid
        self.geometry_type = config.geometry.type

        # Ion mass from config
        self.ion_mass = config.ion_mass

        # Boundary config
        self.boundary_cfg = config.boundary

        # Backend selection: "python", "athena", or "auto"
        self.backend = self._resolve_backend(fc.backend)

        if self.backend == "athena":
            from dpf.athena_wrapper import AthenaPPSolver
            self.fluid = AthenaPPSolver(config)
            self._cell_volume = None
            logger.info("Using Athena++ backend (mode: %s)", self.fluid.mode)
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
            )
            self._cell_volume = None  # Computed on demand from cylindrical geometry
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

        # Initialize plasma state
        self.state = self._initial_state(nx, ny, nz)

        # Coupling
        self._coupling = CouplingState()

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

        # Checkpoint settings
        self.checkpoint_interval: int = 0  # 0 = disabled
        self.checkpoint_filename: str = "checkpoint.h5"

        logger.info(
            "SimulationEngine initialized: grid=(%d,%d,%d), geometry=%s, backend=%s, "
            "sim_time=%.2e s, bremsstrahlung=%s, fld=%s, sheath=%s",
            nx, ny, nz, self.geometry_type, self.backend, config.sim_time,
            self.rad_cfg.bremsstrahlung_enabled,
            self.rad_cfg.fld_enabled,
            self.sheath_cfg.enabled,
        )

    @staticmethod
    def _resolve_backend(requested: str) -> str:
        """Resolve the requested backend to an actual backend name.

        Args:
            requested: ``"python"``, ``"athena"``, or ``"auto"``.

        Returns:
            ``"python"`` or ``"athena"``.

        Raises:
            RuntimeError: If ``"athena"`` was explicitly requested but is
                not available.
        """
        if requested == "python":
            return "python"

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
            try:
                from dpf.athena_wrapper import is_available
                if is_available():
                    logger.info("Auto-selected Athena++ backend")
                    return "athena"
            except ImportError:
                pass
            logger.info("Auto-selected Python backend (Athena++ not available)")
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
        config_json = self.config.model_dump_json()
        save_checkpoint(
            fname, self.state, circuit_state,
            self.time, self.step_count, config_json,
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
        if self.backend == "athena":
            return self._step_athena(dt, sim_time, _max_steps)

        # === Step 1: Compute ionization state and plasma resistance ===
        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = rho / self.ion_mass  # Number density (assume fully ionized for ne)

        # Volume-averaged quantities for scalar coupling
        Te_avg = float(np.mean(Te))
        ne_avg = float(np.mean(ne))

        # Compute average ionization state from Saha equation
        Z_bar = saha_ionization_fraction(Te_avg, ne_avg)
        Z_bar = max(Z_bar, 0.01)  # Floor at 0.01 to avoid division by zero

        # Compute spatially-resolved resistivity field for MHD solver
        eta_field = None
        eta_anom = 0.0
        R_plasma = 0.0
        L_plasma = 0.0

        if Te_avg > 1000.0 and ne_avg > 1e10:
            # Cell-by-cell Spitzer resistivity with temperature floor
            Te_floored = np.maximum(Te, 1000.0)
            ne_floored = np.maximum(ne, 1e10)
            lnL_field = coulomb_log(ne_floored, Te_floored)
            eta_spitzer_field = spitzer_resistivity(
                ne_floored, Te_floored, lnL_field, Z=Z_bar,
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
                eta_anom_field = anomalous_resistivity_field(
                    J_mag, np.maximum(ne_2d, 1e10), np.maximum(Ti_2d, 1.0),
                    alpha=self.config.anomalous_alpha,
                    mi=self.ion_mass,
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
                )
                eta_field = eta_spitzer_field + eta_anom_field

            # Scalar anomalous for diagnostics
            J_avg = abs(I_current) / max(A_column, 1e-30)
            Ti_avg = float(np.mean(self.state["Ti"]))
            eta_anom = anomalous_resistivity_scalar(
                J_avg, ne_avg, Ti_avg, alpha=self.config.anomalous_alpha,
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

            # Cap R_plasma to prevent runaway (physically: few Ohm max)
            R_plasma = min(R_plasma, 10.0)

            # --- Volume-integral L_plasma: L = 2 * integral(B^2/(2*mu_0) dV) / I^2 ---
            B_sq = np.sum(B_field**2, axis=0)
            if self.geometry_type == "cylindrical":
                B_sq_2d = np.squeeze(B_sq, axis=1) if B_sq.ndim == 3 else B_sq
                L_plasma = float(np.sum(B_sq_2d / _mu_0 * cell_vol)) / I_sq
            else:
                L_plasma = float(np.sum(B_sq / _mu_0 * dV)) / I_sq

        # === Step 1b: Collision+Radiation (first half-step of Strang) ===
        self._apply_collision_radiation(dt / 2.0, Z_bar)

        # === Step 2: Circuit advance (with plasma resistance + inductance) ===
        coupling = self.fluid.coupling_interface()
        coupling.R_plasma = R_plasma
        coupling.Z_bar = Z_bar
        # Use volume-integral L_plasma if available
        if L_plasma > 0:
            coupling.Lp = L_plasma
        back_emf = coupling.emf
        new_coupling = self.circuit.step(coupling, back_emf, dt)
        self._coupling = new_coupling

        # === Step 3: Fluid/MHD advance (with resistivity + electrode BCs) ===
        cc = self.config.circuit
        self.state = self.fluid.step(
            self.state,
            dt,
            current=new_coupling.current,
            voltage=new_coupling.voltage,
            eta_field=eta_field,
            anode_radius=cc.anode_radius,
            cathode_radius=cc.cathode_radius,
            apply_electrode_bc=self.boundary_cfg.electrode_bc,
        )
        self._sanitize_state("after fluid step")

        # === Step 3a: Nernst B-field advection ===
        fc = self.config.fluid
        if fc.enable_nernst:
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
        self._apply_collision_radiation(dt / 2.0, Z_bar)

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

        # === Step 5c: Synthetic interferometry (cylindrical only) ===
        if self.geometry_type == "cylindrical":
            ne_interf = rho_yield / self.ion_mass
            # Take midplane slice (z = nz//2)
            nz_mid = ne_interf.shape[2] // 2
            ne_midplane = ne_interf[:, 0, nz_mid]  # shape (nr,)
            r_grid = self.fluid.geom.r  # Radial coordinate array
            N_L = abel_transform(ne_midplane, r_grid)
            self._last_fringe_shifts = fringe_shift(N_L)

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
                    "total_neutron_yield": self.total_neutron_yield,
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
        coupling = self.fluid.coupling_interface()
        coupling.R_plasma = 0.0
        coupling.Z_bar = 1.0
        back_emf = coupling.emf
        new_coupling = self.circuit.step(coupling, back_emf, dt)
        self._coupling = new_coupling

        # --- MHD advance via Athena++ ---
        self.state = self.fluid.step(
            self.state,
            dt,
            current=new_coupling.current,
            voltage=new_coupling.voltage,
        )

        # --- Advance time ---
        self.time += dt
        self.step_count += 1

        # --- Diagnostics ---
        self._last_R_plasma = 0.0
        self._last_Z_bar = 1.0
        self._last_eta_anom = 0.0

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
                    "R_plasma": 0.0,
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

    def _apply_collision_radiation(self, dt_sub: float, Z_bar: float) -> None:
        """Apply collision (temperature relaxation) and radiation losses.

        This is the combined collision + radiation operator used in Strang
        splitting.  Called twice per timestep with dt/2 each (once before
        and once after the MHD advance) for 2nd-order temporal accuracy.

        Args:
            dt_sub: Sub-step duration [s] (typically dt/2).
            Z_bar: Average ionization state.
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

        freq_ei = nu_ei(ne, Te, lnL, Z=Z_bar)
        Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt_sub)
        self.state["Te"] = Te_new
        self.state["Ti"] = Ti_new

        # Update pressure from new temperatures
        self.state["pressure"] = self.eos.total_pressure(rho, Ti_new, Te_new)
        self._sanitize_state("after collision step")

        # --- Braginskii ion viscosity ---
        if self.config.fluid.enable_viscosity:
            self._apply_viscosity(dt_sub)
            self._sanitize_state("after viscosity step")

        # --- Radiation losses ---
        if self.rad_cfg.bremsstrahlung_enabled:
            ne_rad = rho / self.ion_mass
            if self.rad_cfg.fld_enabled:
                self.state = apply_radiation_transport(
                    self.state,
                    dx=self.config.dx,
                    dt=dt_sub,
                    Z=Z_bar,
                    gaunt_factor=self.rad_cfg.gaunt_factor,
                )
            else:
                Te_rad, P_rad = apply_bremsstrahlung_losses(
                    self.state["Te"],
                    ne_rad,
                    dt_sub,
                    Z=Z_bar,
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
            self._apply_diffusion(dt_sub, Z_bar)
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

    def _apply_diffusion(self, dt_sub: float, Z_bar: float) -> None:
        """Apply implicit or super-time-stepping magnetic and thermal diffusion.

        Called from _apply_collision_radiation when diffusion_method != 'explicit'.
        Solves the resistive diffusion dB/dt = (eta/mu_0)*Laplacian(B) and
        thermal conduction dTe/dt = kappa/(1.5*ne*kB)*Laplacian(Te) using either
        Crank-Nicolson ADI or RKL2 super time-stepping.

        Args:
            dt_sub: Sub-step duration [s] (typically dt/2 from Strang).
            Z_bar: Average ionization state.
        """
        fc = self.config.fluid
        dx = self.config.dx
        B = self.state["B"]
        Te = self.state["Te"]
        rho = self.state["rho"]
        ne = np.maximum(rho / self.ion_mass, 1e10)

        # Compute resistivity field for diffusion coefficient
        Te_safe = np.maximum(Te, 1000.0)
        ne_safe = np.maximum(ne, 1e10)
        from dpf.collision.spitzer import spitzer_resistivity as _spitz_eta
        lnL = coulomb_log(ne_safe, Te_safe)
        eta = _spitz_eta(ne_safe, Te_safe, lnL, Z=Z_bar)

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
        if fc.enable_anisotropic_conduction:
            from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction
            B_ac = self.state["B"]
            Te_ac = self.state["Te"]
            ne_ac = np.maximum(self.state["rho"] / self.ion_mass, 1e10)
            Te_aniso = anisotropic_thermal_conduction(
                Te_ac, B_ac, ne_ac, dt_sub, dx, dy, dz,
                Z_eff=max(Z_bar, 0.01),
            )
            self.state["Te"] = np.maximum(Te_aniso, 1.0)

        # Update pressure from new Te
        self.state["pressure"] = self.eos.total_pressure(
            rho, self.state["Ti"], self.state["Te"]
        )

    # ------------------------------------------------------------------
    # Field snapshot access (for server/GUI)
    # ------------------------------------------------------------------

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
        t_wall_start = wall_time.monotonic()

        # Store initial energy
        self.initial_energy = self.circuit.total_energy()

        logger.info("Starting simulation: t_end=%.2e s", self.config.sim_time)

        while True:
            result = self.step(_max_steps=max_steps)
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
        }

        logger.info(
            "Simulation complete: %d steps in %.2f s (%.1f steps/s), E_cons=%.6f",
            self.step_count,
            t_wall,
            self.step_count / max(t_wall, 1e-10),
            conservation,
        )

        return summary
