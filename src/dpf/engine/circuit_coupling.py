"""Circuit-plasma coupling methods extracted from SimulationEngine.

Contains: circuit subcycling, snowplow source terms, Lp computation,
back-EMF, electrode boundary conditions, ohmic gap correction, and
radial field initialization.

These are methods of SimulationEngine assigned back to the class in core.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from dpf.circuit.coupler import FeedbackResult
from dpf.constants import eV, k_B, pi
from dpf.constants import mu_0 as _mu_0
from dpf.core.bases import CouplingState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _step_circuit_subcycle(
    self,
    dt: float,
    R_plasma: float,
    L_plasma: float,
    Z_bar: float,
) -> CouplingState:
    """Step 2: Sub-cycle circuit + snowplow to resolve LC dynamics.

    Returns the final CouplingState after all sub-steps.
    """
    coupling = self.fluid.coupling_interface()
    coupling.R_plasma = R_plasma
    coupling.Z_bar = Z_bar

    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError(f"Non-finite dt={dt} in circuit subcycle — MHD state likely diverged")

    Lp_safe = self._coupling.Lp if np.isfinite(self._coupling.Lp) else 0.0
    L_total = self.circuit.L_ext + Lp_safe
    dt_lc = np.sqrt(max(L_total, 1e-12) * self.circuit.C)
    # Target ~500 sub-steps per quarter period for accurate snowplow trajectory
    dt_sub_target = max(dt_lc / 500.0, 1e-12)
    if not np.isfinite(dt_sub_target):
        dt_sub_target = 1e-12
    n_sub = max(1, int(np.ceil(dt / dt_sub_target)))
    dt_sub = dt / n_sub

    sheath_pressure = self._dynamic_sheath_pressure()
    self._last_sheath_pressure = sheath_pressure

    # back-EMF: snowplow captures motional EMF via I*dL/dt.
    # For pure-snowplow runs, set zero (dL/dt handled by circuit sub-step).
    # For hybrid snowplow+MHD runs, back_emf will be set during Lp blending
    # below once MHD fields pass engineering eligibility checks.
    if self.snowplow is not None:
        back_emf = 0.0
    else:
        back_emf = self._compute_back_emf(dt)

    use_coupler = self._should_use_coupler()

    feedback: FeedbackResult | None = None
    if use_coupler and self.state.get("rho") is not None:
        feedback = self.coupler.compute_feedback(
            self.state, self._coupling.current, dt,
        )
        self._last_feedback = feedback

    for _isub in range(n_sub):
        if self.snowplow is not None and self.snowplow.is_active:
            sp_result = self.snowplow.step(
                dt_sub, self._coupling.current,
                pressure=sheath_pressure,
            )
            Lp_sp = sp_result["L_plasma"]
            dLdt_sp = sp_result["dL_dt"]
            self._last_sp_dL_dt = dLdt_sp
            self._last_sp_R_plasma = sp_result.get("R_plasma", 0.0)

            # Lp handoff: blend snowplow -> density-weighted Lp from MHD fields.
            # radial_mhd: activate at radial phase onset as an engineering mode.
            # full_mhd: activate from axial rundown onward (uses compute_feedback
            #   density-weighted Lp during axial phase as an engineering signal).
            handoff = self.config.snowplow.handoff_mode
            handoff_phases = (
                ("rundown", "radial", "reflected")
                if handoff == "full_mhd"
                else ("radial", "reflected")
            )
            # Gate: only use MHD Lp when it's comparable to snowplow Lp.
            # During early axial rundown, MHD fields are uninitialized —
            # compute_feedback returns near-zero Lp, which would let current
            # rise unrestricted (+59% I_peak observed without this gate).
            _mhd_lp_eligible = (
                feedback is not None
                and feedback.Lp > 0
                and (Lp_sp <= 0 or feedback.Lp >= 0.5 * Lp_sp)
            )
            if (
                handoff in ("radial_mhd", "full_mhd")
                and self.snowplow.phase in handoff_phases
                and _mhd_lp_eligible
            ):
                # Activate blending when MHD Lp is resolved
                if not self._lp_blend_active:
                    self._lp_blend_active = True
                    self._lp_blend_alpha = 0.0
                # Exponential ramp: α → 1 with τ = 5 sub-steps
                tau_blend = 5.0 * dt_sub
                self._lp_blend_alpha = min(
                    1.0 - (1.0 - self._lp_blend_alpha) * np.exp(-dt_sub / max(tau_blend, 1e-30)),
                    1.0,
                )
                alpha = self._lp_blend_alpha
                Lp_mhd = feedback.Lp
                dLdt_mhd = feedback.dLp_dt
                # Clamp: no >20% Lp jump per sub-step
                Lp_blend = alpha * Lp_mhd + (1.0 - alpha) * Lp_sp
                if self._prev_L_plasma > 0:
                    ratio = Lp_blend / self._prev_L_plasma
                    if ratio > 1.2:
                        Lp_blend = 1.2 * self._prev_L_plasma
                    elif ratio < 0.8:
                        Lp_blend = 0.8 * self._prev_L_plasma
                coupling.Lp = Lp_blend
                coupling.dL_dt = alpha * dLdt_mhd + (1.0 - alpha) * dLdt_sp
                # Ramp back-EMF with alpha (was stuck at 0 for 0 < alpha < 0.9)
                back_emf = alpha * feedback.back_emf
            else:
                coupling.Lp = Lp_sp
                coupling.dL_dt = dLdt_sp
            self._prev_L_plasma = coupling.Lp

        elif self.snowplow is not None and not self.snowplow.is_active:
            # Post-pinch: use snowplow's column expansion dL/dt from _frozen_result().
            # The expansion model (Goyon 2025) tracks r(t) = r_pinch + v_expand * dt
            # and computes dL/dt = -(mu_0/2pi) * z_f * v_expand / r(t).
            # Previously dL_dt=0 (frozen L), which produced no current dip.
            sp_post = self.snowplow.step(dt_sub, self._coupling.current)
            sp_Lp = sp_post.get("L_plasma", self.snowplow.plasma_inductance)
            sp_dLdt = sp_post.get("dL_dt", 0.0)
            sp_R = sp_post.get("R_plasma", 0.0)
            coupling.R_plasma = max(coupling.R_plasma, sp_R)

            if (
                self.coupling_mode == "density_weighted"
                and feedback is not None
                and feedback.Lp > 0
                and feedback.dLp_dt is not None
                and abs(feedback.dLp_dt) > abs(sp_dLdt) * 0.1
            ):
                # MHD feedback has meaningful dLp/dt post-pinch — use it.
                coupling.Lp = feedback.Lp
                coupling.dL_dt = feedback.dLp_dt
                back_emf = feedback.back_emf
            else:
                # Snowplow post-pinch expansion model provides dL/dt.
                # This is the fallback when MHD feedback's Lp is clamped
                # or when snowplow's expansion model is more physical.
                coupling.Lp = sp_Lp
                coupling.dL_dt = sp_dLdt

        elif use_coupler and feedback is not None and feedback.Lp > 0:
            # Density-weighted Lp from MHD fields (CircuitCoupler).
            coupling.Lp = feedback.Lp
            coupling.dL_dt = feedback.dLp_dt
            back_emf = feedback.back_emf
            self._prev_L_plasma = feedback.Lp

        elif L_plasma > 0:
            coupling.Lp = L_plasma
            if self._prev_L_plasma > 0 and dt_sub > 0:
                coupling.dL_dt = (L_plasma - self._prev_L_plasma) / dt_sub
            self._prev_L_plasma = L_plasma

        new_coupling = self.circuit.step(coupling, back_emf, dt_sub)
        self._coupling = new_coupling
        coupling.R_plasma = R_plasma

    # One-shot state initialization at axial->radial phase transition
    if (
        self.snowplow is not None
        and self.snowplow.phase in ("radial", "reflected", "pinch")
        and not self._radial_bfield_initialized
    ):
        handoff = self.config.snowplow.handoff_mode
        if handoff in ("radial_mhd", "full_mhd"):
            self._initialize_radial_state()
            # Force dt recomputation: the radial init dramatically changes
            # wave speeds (B=286T → v_A=4.7e6 m/s). The pre-handoff dt
            # violates CFL by ~500×. Set flag to skip THIS step's MHD advance.
            self._skip_next_fluid_step = True
        else:
            self._initialize_radial_bfield()

    return new_coupling



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
    """Mass-conserving snowplow source terms for MHD grid.

    Physics: The snowplow sweeps fill gas as it propagates. Each timestep:
    1. Compute dm = swept_mass(t) - swept_mass(t-dt) [kg] of newly swept gas
    2. Remove dm uniformly from upstream cells (ahead of sheath)
    3. Deposit dm at sheath location (Gaussian-smeared, width = 2*dx)
    4. Momentum = dm/dt × v_sheath in propagation direction
    5. Energy = Rankine-Hugoniot post-shock: T_post = (3/16) × m_i × v² / k_B

    Conservation: total grid mass is preserved (depletion balances deposition).
    """
    if not self.config.snowplow.enable_mhd_coupling:
        return {}
    if self.snowplow is None or not self.snowplow.is_active:
        return {}

    dx = self.config.dx
    dz_cfg = self.config.geometry.dz
    dz = dz_cfg if dz_cfg else dx
    gamma = self.config.fluid.gamma
    m_ion = self.ion_mass
    grid_shape = self.state["rho"].shape

    if self.snowplow.phase == "rundown":
        z_sheath = self.snowplow.sheath_position
        v_sheath = self.snowplow.sheath_velocity

        if abs(v_sheath) < 1e-6:
            return {}

        # dm swept this step [kg]
        m_swept_now = self.snowplow.swept_mass
        dm = max(m_swept_now - self._prev_swept_mass, 0.0)
        self._prev_swept_mass = m_swept_now
        if dm < 1e-30:
            return {}

        # Source rate [kg/s] for continuous source term formulation
        dm_dt = dm / max(dt, 1e-30)

        # Axial coordinate array
        nz = grid_shape[-1]
        z_arr = np.array([(k + 0.5) * dz for k in range(nz)])

        # --- Depletion: remove mass from upstream (z > z_sheath) ---
        # Uniform depletion from cells ahead of sheath
        upstream_mask = z_arr > z_sheath
        n_upstream = int(np.sum(upstream_mask))
        S_rho_deplete = np.zeros(grid_shape)
        if n_upstream > 0:
            # Deplete uniformly from upstream cells [kg/m³/s]
            if self.geometry_type == "cylindrical":
                cell_vols = self.fluid.geom.cell_volumes()  # (nr, nz)
                upstream_vol = float(np.sum(cell_vols[:, upstream_mask]))
            else:
                cell_vol = dx * dx * dz
                upstream_vol = cell_vol * n_upstream * grid_shape[0] * grid_shape[1]
            depletion_rate = dm_dt / max(upstream_vol, 1e-30)  # [kg/m³/s]
            # Cap depletion to 50% of local density per step (prevents over-depletion on small grids)
            rho_upstream = self.state["rho"]
            max_depletion = 0.5 * np.mean(rho_upstream[:, :, upstream_mask]) / max(dt, 1e-30)
            depletion_rate = min(depletion_rate, max(max_depletion, 0.0))
            if self.geometry_type == "cylindrical":
                S_rho_deplete[:, :, upstream_mask] = -depletion_rate
            else:
                S_rho_deplete[:, :, upstream_mask] = -depletion_rate

        # --- Deposition: inject at sheath location (Gaussian) ---
        sigma_z = 2.0 * dz
        W_z = np.exp(-0.5 * ((z_arr - z_sheath) / sigma_z) ** 2)
        W_sum = np.sum(W_z) + 1e-30
        W_z_norm = W_z / W_sum  # Normalized to sum=1

        S_rho_deposit = np.zeros(grid_shape)
        if self.geometry_type == "cylindrical":
            cell_vols = self.fluid.geom.cell_volumes()
            # Distribute dm_dt into cells weighted by W_z, per unit volume
            for k in range(nz):
                vol_slice = float(np.sum(cell_vols[:, k]))
                if vol_slice > 0:
                    S_rho_deposit[:, :, k] = dm_dt * W_z_norm[k] / max(vol_slice, 1e-30)
        else:
            cell_vol = dx * dx * dz
            nr, ny_g = grid_shape[0], grid_shape[1]
            total_cells_per_slice = nr * ny_g
            for k in range(nz):
                S_rho_deposit[:, :, k] = dm_dt * W_z_norm[k] / max(cell_vol * total_cells_per_slice, 1e-30)

        S_rho = S_rho_deplete + S_rho_deposit

        # Momentum: deposited gas enters at sheath velocity (z-direction)
        S_mom = np.zeros((3, *grid_shape))
        S_mom[2] = np.maximum(S_rho_deposit, 0.0) * v_sheath  # only deposited mass carries momentum

        # Energy: Rankine-Hugoniot post-shock (strong shock, γ=5/3)
        # T_ion = (3/16) × m_ion × v²_s / k_B  (NRL Formulary, ion-only)
        # v_post = 2v_s/(γ+1) = (3/4)v_s  (lab-frame post-shock velocity)
        T_post = (3.0 / 16.0) * m_ion * v_sheath**2 / k_B
        v_post = 2.0 * v_sheath / (gamma + 1.0)  # lab-frame post-shock velocity
        p_post_per_mass = k_B * max(T_post, 1.0) / m_ion
        e_thermal = p_post_per_mass / (gamma - 1.0)
        e_kinetic = 0.5 * v_post**2
        S_energy = np.maximum(S_rho_deposit, 0.0) * (e_thermal + e_kinetic)

        return {
            "S_rho_snowplow": S_rho,
            "S_mom_snowplow": S_mom,
            "S_energy_snowplow": S_energy,
        }

    elif self.snowplow.phase == "radial":
        r_shock = self.snowplow.shock_radius
        vr_shock = self.snowplow.vr

        if abs(vr_shock) < 1e-6:
            return {}

        # dm swept this step [kg]
        m_radial_now = self.snowplow.radial_swept_mass
        dm = max(m_radial_now - self._prev_radial_swept_mass, 0.0)
        self._prev_radial_swept_mass = m_radial_now
        if dm < 1e-30:
            return {}

        dm_dt = dm / max(dt, 1e-30)

        nx = grid_shape[0]
        r_arr = np.array([(i + 0.5) * dx for i in range(nx)])

        # --- Depletion: remove mass from outside shock (r > r_shock) ---
        upstream_mask_r = r_arr > r_shock
        n_upstream_r = int(np.sum(upstream_mask_r))
        S_rho_deplete = np.zeros(grid_shape)
        if n_upstream_r > 0:
            if self.geometry_type == "cylindrical":
                cell_vols = self.fluid.geom.cell_volumes()
                upstream_vol = float(np.sum(cell_vols[upstream_mask_r, :]))
            else:
                cell_vol = dx**3
                upstream_vol = cell_vol * n_upstream_r * grid_shape[1] * grid_shape[2]
            depletion_rate = dm_dt / max(upstream_vol, 1e-30)
            rho_upstream_r = self.state["rho"]
            max_depletion_r = 0.5 * np.mean(rho_upstream_r[upstream_mask_r, :, :]) / max(dt, 1e-30)
            depletion_rate = min(depletion_rate, max(max_depletion_r, 0.0))
            if self.geometry_type == "cylindrical":
                S_rho_deplete[upstream_mask_r, :, :] = -depletion_rate
            else:
                S_rho_deplete[upstream_mask_r, :, :] = -depletion_rate

        # --- Deposition: inject at shock front (Gaussian in r) ---
        sigma_r = 2.0 * dx
        W_r = np.exp(-0.5 * ((r_arr - r_shock) / sigma_r) ** 2)
        W_r_norm = W_r / (np.sum(W_r) + 1e-30)

        S_rho_deposit = np.zeros(grid_shape)
        if self.geometry_type == "cylindrical":
            cell_vols = self.fluid.geom.cell_volumes()
            for i in range(nx):
                vol_slice = float(np.sum(cell_vols[i, :]))
                if vol_slice > 0:
                    S_rho_deposit[i, :, :] = dm_dt * W_r_norm[i] / max(vol_slice, 1e-30)
        else:
            cell_vol = dx**3
            ny_g, nz_g = grid_shape[1], grid_shape[2]
            for i in range(nx):
                S_rho_deposit[i, :, :] = dm_dt * W_r_norm[i] / max(cell_vol * ny_g * nz_g, 1e-30)

        S_rho = S_rho_deplete + S_rho_deposit

        # Momentum: radial inward (vr < 0)
        S_mom = np.zeros((3, *grid_shape))
        S_mom[0] = np.maximum(S_rho_deposit, 0.0) * vr_shock

        # Energy: Rankine-Hugoniot post-shock (radial, strong shock, γ=5/3)
        T_post = (3.0 / 16.0) * m_ion * vr_shock**2 / k_B
        vr_post = 2.0 * vr_shock / (gamma + 1.0)  # lab-frame post-shock radial velocity
        p_post_per_mass = k_B * max(T_post, 1.0) / m_ion
        e_thermal = p_post_per_mass / (gamma - 1.0)
        e_kinetic = 0.5 * vr_post**2
        S_energy = np.maximum(S_rho_deposit, 0.0) * (e_thermal + e_kinetic)

        return {
            "S_rho_snowplow": S_rho,
            "S_mom_snowplow": S_mom,
            "S_energy_snowplow": S_energy,
        }

    return {}


def _mhd_coupler_trust_status(self) -> dict[str, object]:
    """Classify whether current MHD fields are usable for auto circuit loading.

    This is an engineering trust gate, not scientific validation. The auto mode
    requires more than a positive density array because uniform initial fill gas
    does not prove that MHD fields have resolved sheath/circuit information.
    """
    rho = self.state.get("rho")
    if not isinstance(rho, np.ndarray):
        return {"trusted": False, "reason": "missing_density_field"}

    finite_rho = rho[np.isfinite(rho)]
    if finite_rho.size == 0:
        return {"trusted": False, "reason": "density_not_finite"}

    rho_max = float(np.max(finite_rho))
    rho_min = float(np.min(finite_rho))
    if rho_max <= 0.0:
        return {"trusted": False, "reason": "density_nonpositive"}

    rho_dynamic = (rho_max - rho_min) > max(1e-12 * rho_max, 1e-30)
    indicators: dict[str, bool] = {"rho_dynamic": rho_dynamic}

    for key in ("B", "velocity"):
        arr = self.state.get(key)
        if isinstance(arr, np.ndarray) and arr.size > 0:
            finite = arr[np.isfinite(arr)]
            indicators[f"{key}_nonzero"] = (
                finite.size > 0 and float(np.max(np.abs(finite))) > 0.0
            )
        else:
            indicators[f"{key}_nonzero"] = False

    trusted = any(indicators.values())
    return {
        "trusted": trusted,
        "reason": "resolved_mhd_signal" if trusted else "uniform_initial_state",
        "indicators": indicators,
        "validation_status": "not_validation_evidence",
        "can_support_scientific_claims": False,
    }


def _should_use_coupler(self) -> bool:
    """Determine whether to use the CircuitCoupler for Lp extraction.

    Explicit ``density_weighted`` mode remains caller-controlled. ``auto`` mode
    requires a resolved MHD signal, not only nonzero initial density.
    """
    if self.coupling_mode == "lee_only":
        self._coupler_trust_status = {
            "trusted": False,
            "reason": "lee_only_mode",
            "validation_status": "not_validation_evidence",
            "can_support_scientific_claims": False,
        }
        return False
    if self.coupling_mode == "density_weighted":
        self._coupler_trust_status = {
            "trusted": True,
            "reason": "explicit_density_weighted_mode",
            "validation_status": "not_validation_evidence",
            "can_support_scientific_claims": False,
        }
        return True
    # Cache result; recompute every 10 steps or on first call.
    if self._coupler_decision_cache is None or self.step_count % 10 == 0:
        status = self._mhd_coupler_trust_status()
        self._coupler_trust_status = status
        self._coupler_decision_cache = bool(status["trusted"])
    return self._coupler_decision_cache


def _compute_grid_mass(self) -> float:
    """Total mass on the MHD grid [kg]."""
    rho = self.state.get("rho")
    if rho is None:
        return 0.0
    if self.geometry_type == "cylindrical" and hasattr(self.fluid, "geom"):
        cell_vols = self.fluid.geom.cell_volumes()
        rho_2d = np.squeeze(rho, axis=1) if rho.ndim == 3 else rho
        return float(np.sum(rho_2d * cell_vols))
    return float(np.sum(rho)) * self.config.dx**3


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
    elif self.geometry_type == "cylindrical" and self.backend not in ("metal",):
        # Generic electrode BC for Athena++, AthenaK (NOT Metal — Metal handles
        # its own electrode BC inside step_gpu with energy correction).
        # Sets B_theta = mu_0 * I / (2*pi*r) between electrodes.
        # Energy correction: inject delta(B²/2μ₀) into pressure to preserve
        # conservation — without this, the BC injects ~10⁸ J/m³ of magnetic
        # energy per step without bookkeeping, driving pressure negative.
        cc = self.config.circuit
        dr = self.config.dx
        nr = self.config.grid_shape[0]
        B = self.state["B"]
        if B.ndim != 4:
            nr_cfg, ny_cfg, nz_cfg = self.config.grid_shape
            B = B.reshape(3, nr_cfg, ny_cfg, nz_cfg)
            self.state["B"] = B
        B_sq_before = B[1]**2  # B_theta² before BC
        for ir in range(nr):
            r = (ir + 0.5) * dr
            if cc.anode_radius <= r <= cc.cathode_radius and r > 0:
                val = _mu_0 * current / (2.0 * pi * r)
                B[1, ir, :, :] = val
            elif r < cc.anode_radius:
                B[1, ir, :, :] = 0.0
        self.state["B"] = B
        # Energy correction: inject magnetic energy change into pressure
        B_sq_after = B[1]**2
        delta_ME = (B_sq_after - B_sq_before) / (2.0 * _mu_0)
        gamma = self.config.fluid.gamma
        self.state["pressure"] = np.maximum(
            self.state["pressure"] + delta_ME * (gamma - 1.0), 1e-20,
        )

    # Snowplow zipper BC: applies to ALL backends with cylindrical geometry
    if self.geometry_type == "cylindrical" and self.snowplow and self.snowplow.is_active:
        z_sheath = self.snowplow.z
        dz = self.config.geometry.dz if self.config.geometry.dz else self.config.dx
        if not np.isfinite(z_sheath) or dz <= 0:
            return  # snowplow diverged — skip zipper BC this step
        iz_sheath = int(round(z_sheath / dz))

        # Skip until the sheath has crossed the first z-cell. When iz_sheath=0
        # the BC zeroes B_theta beyond z=0 every step, while the electrode BC
        # re-imposes B_theta = mu0*I/(2*pi*r) at z=0 — producing a 1-cell-wide
        # B-spike that WENO reconstructs into spurious ~30 T flux. Python
        # backend (which calls _apply_electrode_bc) diverged with I_peak ~6 kA
        # vs MLX 333 kA on PF-1000 27 kV (3x backend parity gap, NRMSE=0.99).
        # MLX skipped this code path entirely (engine/core.py:667 exclusion),
        # which masked the bug. Gating fixes the backend-parity regression.
        if iz_sheath >= 1:
            nx, ny, nz = self.config.grid_shape
            B = self.state["B"]
            # Ensure 4D (3, nr, ny, nz) — normalised upstream for cylindrical
            # backends; guard Cartesian/Python paths that may still be 3D.
            if B.ndim != 4:
                B = B.reshape(3, nx, ny, nz)
                self.state["B"] = B
            if iz_sheath < nz:
                B[1, :, :, iz_sheath + 1:] = 0.0

        # Radial zipper: suppress B_theta inside radial shock front
        # (field-free interior ahead of converging sheath).
        # Use a smooth tanh ramp over 3 cells instead of a sharp cutoff
        # to prevent CFL catastrophic collapse on coarse grids. The sharp
        # B_theta=0 step function creates a 1-2 cell discontinuity that
        # amplifies |B| by 10^6 in ~10 steps (RCA: current_dip_rca.md).
        if self.snowplow.phase in ("radial", "reflected"):
            r_shock = self.snowplow.r_shock
            dr = self.config.dx
            if np.isfinite(r_shock) and dr > 0:
                r_cells = np.arange(nx) * dr + 0.5 * dr
                # Smooth transition: 0 deep inside shock, 1 outside
                # Width = 3*dr for smooth transition over 3 cells
                width = max(3.0 * dr, 1e-6)
                ramp = 0.5 * (1.0 + np.tanh((r_cells - r_shock) / (0.5 * width)))
                # Apply ramp to B_theta: inside shock → suppressed, outside → preserved
                ramp_4d = ramp[:, np.newaxis, np.newaxis]  # (nr, 1, 1)
                B[1] = B[1] * ramp_4d


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

    ir_shock = int(round(r_shock / dr)) if (dr > 0 and np.isfinite(r_shock)) else len(r_grid)
    ir_shock = min(ir_shock, len(r_grid))

    B = self.state["B"]  # shape (3, nr, 1, nz)
    cc = self.config.circuit
    r_cathode = cc.cathode_radius
    ir_cathode = int(round(r_cathode / dr)) if (dr > 0) else len(r_grid)
    ir_cathode = min(ir_cathode, len(r_grid))

    # Thin-sheath topology: current flows at r = r_shock.
    # By Ampere's law:
    #   r < r_shock  → no enclosed current → B_theta = 0  (field-free interior)
    #   r_shock < r < r_cathode → full I enclosed → B_theta = mu_0*I/(2*pi*r)
    #   r > r_cathode → net current = 0 (return current cancels) → B_theta = 0
    #
    # Previous code had this inverted (B_theta inside shock, zero outside),
    # which injected ~238% of capacitor energy via 1/r divergence near axis.

    # Field-free interior (ahead of converging sheath)
    B[1, :ir_shock, :, :] = 0.0

    # Magnetic field between sheath and cathode
    for ir in range(ir_shock, ir_cathode):
        r_val = r_grid[ir]
        if r_val > 0:
            B[1, ir, :, :] = _mu_0 * I_current / (2.0 * pi * r_val)

    # Zero outside cathode
    if ir_cathode < B.shape[1]:
        B[1, ir_cathode:, :, :] = 0.0

    self.state["B"] = B
    self._radial_bfield_initialized = True

    logger.info(
        "Radial B-field initialized: I=%.2e A, r_shock=%.3e m, "
        "ir_shock=%d/%d, B_theta_max=%.2f T",
        I_current, r_shock, ir_shock, len(r_grid),
        float(np.max(np.abs(B[1]))),
    )


def _initialize_radial_state(self) -> None:
    """Full MHD state initialization at axial→radial transition.

    Extends _initialize_radial_bfield with density, velocity, and pressure
    profiles from the snowplow's Rankine-Hugoniot solution. Called when
    handoff_mode is "radial_mhd" or "full_mhd".

    Physics: At transition, the MHD grid receives:
        - rho(r): compressed slug (4×ρ0) inside shock, fill gas outside
        - v_r(r): inward v_post = (3/4)v_s inside shock, zero outside
        - p(r): Rankine-Hugoniot post-shock pressure inside, fill outside
        - B_theta(r): μ0·I/(2πr) inside shock, zero outside
    """
    if self.snowplow is None or self.geometry_type != "cylindrical":
        self._initialize_radial_bfield()
        return

    I_current = abs(self._coupling.current)
    dr = self.config.dx
    gamma = self.config.fluid.gamma

    # Build radial grid
    if hasattr(self.fluid, "geom") and hasattr(self.fluid.geom, "r"):
        r_grid = self.fluid.geom.r
    else:
        nr = self.config.grid_shape[0]
        r_grid = np.array([(ir + 0.5) * dr for ir in range(nr)])

    # Export Rankine-Hugoniot profiles from snowplow
    profiles = self.snowplow.export_radial_profiles(r_grid, I_current, gamma)

    # Write profiles to MHD state (cylindrical: shape is (nr, 1, nz))
    # B_theta profile from snowplow already uses correct thin-sheath topology
    # (nonzero between r_shock and r_cathode, zero inside r_shock).
    rho = self.state["rho"]
    vel = self.state["velocity"]
    pres = self.state["pressure"]
    B = self.state["B"]

    nr = len(r_grid)
    for ir in range(min(nr, rho.shape[0])):
        rho[ir, :, :] = profiles["rho"][ir]
        vel[0, ir, :, :] = profiles["vr"][ir]
        pres[ir, :, :] = profiles["pressure"][ir]
        B[1, ir, :, :] = profiles["B_theta"][ir]

    # Zero other velocity/B components at initialization
    vel[1, :, :, :] = 0.0  # v_theta = 0
    vel[2, :, :, :] = 0.0  # v_z = 0 (axial motion done)
    B[0, :, :, :] = 0.0    # B_r = 0
    B[2, :, :, :] = 0.0    # B_z = 0

    self.state["rho"] = rho
    self.state["velocity"] = vel
    self.state["pressure"] = pres
    self.state["B"] = B

    # Update Te/Ti from Rankine-Hugoniot pressure.
    # T = p*m_i/(2*rho*kB) for fully ionized Z=1 (n_e + n_i = 2*n_i).
    T = np.maximum(pres * self.ion_mass / (2.0 * np.maximum(rho, 1e-30) * k_B), 1.0)
    self.state["Te"] = T
    self.state["Ti"] = T

    # Recalculate initial grid mass after reinitialization
    self._initial_grid_mass = self._compute_grid_mass()

    self._radial_bfield_initialized = True

    rho_max = float(np.max(rho))
    rho_fill = self.config.rho0
    logger.info(
        "Radial MHD state initialized (handoff_mode=%s): I=%.2e A, "
        "r_shock=%.3e m, rho_max/rho0=%.1f, B_theta_max=%.2f T, "
        "T_ion_max=%.0f eV",
        self.config.snowplow.handoff_mode, I_current,
        self.snowplow.r_shock, rho_max / max(rho_fill, 1e-30),
        float(np.max(np.abs(B[1]))),
        float(np.max(T)) * k_B / eV,
    )


def _dynamic_sheath_pressure(self) -> float:
    """Compute volume-averaged MHD pressure near the sheath/shock front.

    During axial phase: uses the configured cold fill pressure.
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

    if self.snowplow.phase == "rundown":
        # The Lee/RADPF axial equation uses cold molecular fill pressure p0.
        # The MHD state stores total ion+electron plasma pressure, which is a
        # different thermodynamic quantity and over-pressurizes neutral fill.
        return fallback
    elif self.snowplow.phase in ("radial", "reflected"):
        # Radial: average pressure inside shock front (r < r_shock)
        ir = int(round(self.snowplow.r_shock / dr)) if (dr > 0 and np.isfinite(self.snowplow.r_shock)) else 0
        nx = p.shape[0]
        if 0 < ir <= nx:
            p_inside = p[:ir]
            if p_inside.size > 0:
                return max(float(np.mean(p_inside)), fallback)

    return fallback
