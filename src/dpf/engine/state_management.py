"""State management methods extracted from SimulationEngine.

Contains: checkpoint save/load, state sanitization, diagnostics
recording, yield tracking, and step result construction.

These are methods of SimulationEngine assigned back to the class in core.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from dpf.core.bases import CouplingState, StepResult
from dpf.diagnostics.checkpoint import load_checkpoint as _load_ckpt
from dpf.diagnostics.checkpoint import save_checkpoint as _save_ckpt
from dpf.diagnostics.interferometry import abel_transform, fringe_shift
from dpf.diagnostics.pease_braginskii import check_pease_braginskii
from dpf.diagnostics.plasma_regime import regime_validity

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    _save_ckpt(
        fname, self.state, circuit_state,
        self.time, self.step_count, config_json,
        snowplow_state=snowplow_state,
    )


def load_from_checkpoint(self, filename: str) -> None:
    """Restore simulation state from an HDF5 checkpoint file.

    Args:
        filename: Input checkpoint file path.
    """
    data = _load_ckpt(filename)
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


def _step_diagnostics_and_yield(self, dt: float, Z_bar: float) -> float:
    """Steps 5a2-5d: energy balance, neutron yield, instability, interferometry.

    Returns neutron_rate [s^-1] for inclusion in StepResult.
    """
    # Step 5a2: Energy balance tracking
    _circ_state = self.circuit.state
    _L_total_energy = self.circuit.L_ext + self._coupling.Lp
    if self.step_count % max(self.diag_interval, 10) == 0:
        self._energy_tracker.record(
            state=self.state,
            t=self.time,
            dt=dt,
            cell_volume=(
                float(np.mean(self._cell_volume))
                if hasattr(self._cell_volume, "__len__")
                else self._cell_volume
            ),
            radiated_power=(
                self.total_radiated_energy / max(self.time, 1e-30)
                if self.time > 0
                else 0.0
            ),
            C=self.circuit.C,
            V_cap=_circ_state.voltage,
            L_total=_L_total_energy,
            I_current=_circ_state.current,
        )
    _energy_report = self._energy_tracker.get_report()
    self._last_conservation_error = (
        _energy_report.conservation_error[-1]
        if _energy_report.conservation_error
        else 0.0
    )

    # Step 5b: Neutron yield via YieldTracker
    rho_yield = self.state["rho"]
    if self.geometry_type == "cylindrical":
        _cell_vol_yield = float(np.mean(self.fluid.geom.cell_volumes()))
    else:
        _cell_vol_yield = self.config.dx**3
    _fb = self._last_feedback
    _dL_mhd = abs(_fb.dLp_dt) if _fb is not None else 0.0
    _dL_sp = abs(getattr(self, "_last_sp_dL_dt", 0.0))
    # V_pinch = inductive (I*dL/dt) + snowplow anomalous resistance (R_anom)
    # R_anom is nonzero only during radial/disruption phases (snowplow handles this)
    _sp_R = getattr(self, "_last_sp_R_plasma", 0.0)
    _V_pinch = abs(self._coupling.current) * (max(_dL_mhd, _dL_sp) + _sp_R)
    _L_pinch = (
        self.config.snowplow.anode_length * self.config.snowplow.pinch_column_fraction
    )
    self._yield_tracker.accumulate(
        state=self.state,
        dt=dt,
        I_current=self._coupling.current,
        V_pinch=_V_pinch,
        cell_volume=_cell_vol_yield,
        L_pinch=_L_pinch,
    )
    _yield_result = self._yield_tracker.get_result()
    _dY_thermo = _yield_result.dY_thermo[-1] if _yield_result.dY_thermo else 0.0
    _dY_bt = _yield_result.dY_bt[-1] if _yield_result.dY_bt else 0.0
    neutron_rate = (_dY_thermo + _dY_bt) / max(dt, 1e-30)
    self._last_neutron_rate = neutron_rate
    self.total_neutron_yield = _yield_result.Y_total
    self._last_beam_target_rate = _dY_bt / max(dt, 1e-30)
    self._last_bt_fraction = _yield_result.bt_fraction

    # Step 5b3: m=0 sausage instability growth rate
    self._last_m0_result = None
    if self.snowplow and self.snowplow.phase in ("radial", "reflected", "pinch"):
        from dpf.diagnostics.instability import m0_growth_rate_from_state
        self._last_m0_result = m0_growth_rate_from_state(
            self.state, self.snowplow, self.config,
        )

    # Step 5b4: Pease-Braginskii radiative collapse check
    self._last_pb_result = check_pease_braginskii(
        I_current=abs(self._coupling.current),
        Z=Z_bar,
        gaunt_factor=1.2,
        ln_Lambda=self.config.collision.coulomb_log,
    )

    # Step 5c: Well Exporter
    if self.well_interval > 0 and self.step_count % self.well_interval == 0:
        self.well_exporter.append_state(self.state, time=self.time)

    # Step 5c: Synthetic interferometry (cylindrical only)
    if self.geometry_type == "cylindrical":
        ne_interf = rho_yield / self.ion_mass
        nz_mid = ne_interf.shape[2] // 2
        ne_midplane = ne_interf[:, 0, nz_mid]
        r_grid = self.fluid.geom.r
        N_L = abel_transform(ne_midplane, r_grid)
        self._last_fringe_shifts = fringe_shift(N_L)

    # Step 5d: Plasma regime validity check (every 100 steps)
    if self.step_count % 100 == 0:
        ne_rv = self.state["rho"] / self.ion_mass
        Te_rv = self.state["Te"]
        Ti_rv = self.state["Ti"]
        v_mag = np.sqrt(np.sum(self.state["velocity"] ** 2, axis=0))
        rv = regime_validity(ne_rv, Te_rv, Ti_rv, v_mag, dx=self.config.dx)
        self._last_regime_result = rv
        frac = rv["fraction_valid"]
        if frac < 0.5 and not self._mhd_regime_warned:
            logger.warning(
                "MHD regime validity: %.0f%% of cells outside MHD-valid regime "
                "(ND>1 or dx<10*lambda_De). Consider kinetic model.",
                (1.0 - frac) * 100,
            )
            self._mhd_regime_warned = True

    return neutron_rate


def _step_record_and_checkpoint(
    self,
    dt: float,
    sim_time: float,
    _max_steps: int | None,
    Z_bar: float,
    R_plasma: float,
    eta_anom: float,
    new_coupling: CouplingState,
    neutron_rate: float,
) -> StepResult:
    """Step 6: Advance time, record diagnostics, auto-checkpoint."""
    self.time += dt
    self.step_count += 1

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
                "bt_fraction": self._last_bt_fraction,
            },
            "energy_balance": {
                "conservation_error": self._last_conservation_error,
                "is_conserved": self._last_conservation_error < 0.05,
                "div_B_max": self._last_div_B,
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
            "coupler": {
                "coupling_mode": self.coupling_mode,
                "Lp": self._last_feedback.Lp if self._last_feedback else 0.0,
                "dLp_dt": self._last_feedback.dLp_dt if self._last_feedback else 0.0,
                "back_emf": self._last_feedback.back_emf if self._last_feedback else 0.0,
                "r_eff": self._last_feedback.r_eff if self._last_feedback else 0.0,
                "z_sheath": self._last_feedback.z_sheath if self._last_feedback else 0.0,
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

    if (
        self.checkpoint_interval > 0
        and self.step_count % self.checkpoint_interval == 0
    ):
        self.save_checkpoint()

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
        mass_conservation=self._compute_grid_mass() / max(self._initial_grid_mass, 1e-30),
        finished=finished,
    )
