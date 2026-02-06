"""Simulation engine — orchestrates the DPF simulation loop.

Wires together: config -> circuit -> fluid -> collision -> diagnostics
into a working timestep loop where every physics module is called each step.

This is the central coordination layer that ensures:
1. Circuit and plasma are properly coupled via CouplingState
2. Energy is tracked and conservation checked
3. Diagnostics are recorded at the configured interval
4. The simulation terminates cleanly
"""

from __future__ import annotations

import logging
import time as wall_time
from typing import Any, Dict, Optional

import numpy as np

from dpf.circuit.rlc_solver import RLCSolver
from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures
from dpf.config import SimulationConfig
from dpf.constants import k_B, m_p
from dpf.core.bases import CouplingState
from dpf.diagnostics.hdf5_writer import HDF5Writer
from dpf.fluid.eos import IdealEOS
from dpf.fluid.mhd_solver import MHDSolver

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

        # Fluid / MHD
        fc = config.fluid
        self.fluid = MHDSolver(
            grid_shape=(nx, ny, nz),
            dx=dx,
            gamma=fc.gamma,
            cfl=fc.cfl,
            dedner_ch=fc.dedner_ch,
        )

        # EOS
        self.eos = IdealEOS(gamma=fc.gamma)

        # Diagnostics
        dc = config.diagnostics
        self.diagnostics = HDF5Writer(
            filename=dc.hdf5_filename,
            field_output_interval=0,
        )
        self.diag_interval = dc.output_interval

        # Initialize plasma state
        self.state = self._initial_state(nx, ny, nz)

        # Coupling
        self._coupling = CouplingState()

        # Energy tracking
        self.initial_energy: Optional[float] = None

        logger.info(
            "SimulationEngine initialized: grid=(%d,%d,%d), sim_time=%.2e s",
            nx, ny, nz, config.sim_time,
        )

    def _initial_state(self, nx: int, ny: int, nz: int) -> Dict[str, np.ndarray]:
        """Create initial plasma state (uniform fill gas)."""
        rho0 = 1e-4  # 0.1 mg/m^3 fill gas density
        T0 = 300.0   # Room temperature
        p0 = (rho0 / m_p) * k_B * T0  # Ideal gas

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

    def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Execute the simulation loop.

        Args:
            max_steps: Maximum number of timesteps (None = run to sim_time).

        Returns:
            Dictionary with summary statistics.
        """
        t_wall_start = wall_time.monotonic()
        sim_time = self.config.sim_time

        # Store initial energy
        self.initial_energy = self.circuit.total_energy()

        logger.info("Starting simulation: t_end=%.2e s", sim_time)

        while self.time < sim_time:
            if max_steps is not None and self.step_count >= max_steps:
                break

            dt = self._compute_dt()
            # Don't overshoot
            if self.time + dt > sim_time:
                dt = sim_time - self.time

            # === Step 1: Circuit advance ===
            coupling = self.fluid.coupling_interface()
            back_emf = coupling.emf
            new_coupling = self.circuit.step(coupling, back_emf, dt)
            self._coupling = new_coupling

            # === Step 2: Fluid/MHD advance ===
            self.state = self.fluid.step(
                self.state,
                dt,
                current=new_coupling.current,
                voltage=new_coupling.voltage,
            )

            # === Step 3: Collision physics (temperature relaxation) ===
            Te = self.state["Te"]
            Ti = self.state["Ti"]
            rho = self.state["rho"]
            ne = rho / m_p  # Assume Z=1

            col_cfg = self.config.collision
            if col_cfg.dynamic_coulomb_log:
                lnL = coulomb_log(ne, Te)
            else:
                lnL = col_cfg.coulomb_log

            freq_ei = nu_ei(ne, Te, lnL)
            Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt)
            self.state["Te"] = Te_new
            self.state["Ti"] = Ti_new

            # Update pressure from new temperatures
            self.state["pressure"] = self.eos.total_pressure(rho, Ti_new, Te_new)

            # === Step 4: Record diagnostics ===
            self.time += dt
            self.step_count += 1

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
        }

        logger.info(
            "Simulation complete: %d steps in %.2f s (%.1f steps/s), E_cons=%.6f",
            self.step_count,
            t_wall,
            self.step_count / max(t_wall, 1e-10),
            conservation,
        )

        return summary
