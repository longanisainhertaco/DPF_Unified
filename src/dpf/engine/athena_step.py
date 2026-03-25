"""Athena++/AthenaK fast-path timestep extracted from SimulationEngine.

Self-contained alternative step() for C++ backends.

These are methods of SimulationEngine assigned back to the class in core.py.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from dpf.core.bases import StepResult
from dpf.diagnostics.pease_braginskii import check_pease_braginskii

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Athena++ backend step
# ------------------------------------------------------------------

def _step_athena(
    self, dt: float, sim_time: float, _max_steps: int | None
) -> StepResult:
    """Timestep using the Athena++/AthenaK MHD backend.

    Uses the full circuit sub-cycling loop (snowplow dynamics, Lp
    blending, back-EMF) via ``_step_circuit_subcycle``, then delegates
    the MHD advance to the C++ backend.  Athena++ handles its own
    resistivity, radiation, and viscosity via source terms enrolled
    in dpf_zpinch.cpp.

    Args:
        dt: Timestep size [s].
        sim_time: Target simulation time [s].
        _max_steps: Optional step limit.

    Returns:
        StepResult with scalar diagnostics.
    """
    # --- Circuit advance with full sub-cycling + snowplow ---
    # Get R_plasma and L_plasma from Athena++ coupling data (dpf_zpinch.cpp
    # UserWorkInLoop computes these via volume integrals of eta*J^2 and B^2).
    coupling = self.fluid.coupling_interface()
    R_plasma = coupling.R_plasma
    L_plasma = coupling.Lp

    new_coupling = self._step_circuit_subcycle(dt, R_plasma, L_plasma, Z_bar=1.0)

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
