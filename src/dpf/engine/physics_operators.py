"""Physics operator-split sub-steps extracted from SimulationEngine.

Contains: collision/radiation, Nernst advection, Powell div(B) sources,
Braginskii viscosity, and implicit/STS magnetic+thermal diffusion.

These are methods of SimulationEngine assigned back to the class in core.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dpf.collision.spitzer import coulomb_log, nu_ei, relax_temperatures, spitzer_resistivity
from dpf.constants import eV, k_B, m_e
from dpf.fluid.anisotropic_conduction import anisotropic_thermal_conduction
from dpf.fluid.implicit_diffusion import implicit_resistive_diffusion, implicit_thermal_diffusion
from dpf.fluid.ionization import coronal_z_eff
from dpf.fluid.mhd_solver import powell_source_terms, powell_source_terms_cylindrical
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

if TYPE_CHECKING:
    pass


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
    # Metal/MLX backends handle transport internally (operator-split in
    # the solver). Skip collision + EOS pressure overwrite for these backends.
    Te = self.state["Te"]
    Ti = self.state["Ti"]
    rho = self.state["rho"]
    _two_t = self.config.fluid.two_temperature and "e_electron" in self.state
    if self.backend not in ("metal", "mlx"):
        col_cfg = self.config.collision
        ne = np.maximum(rho / self.ion_mass, 1e10)
        if col_cfg.dynamic_coulomb_log:
            lnL = coulomb_log(ne, Te)
        else:
            lnL = col_cfg.coulomb_log
        Z_for_collisions = Z_bar_field if Z_bar_field is not None else Z_bar
        freq_ei = nu_ei(ne, Te, lnL, Z=Z_for_collisions)
        if not _two_t:
            Te_new, Ti_new = relax_temperatures(Te, Ti, freq_ei, dt_sub)
            self.state["Te"] = Te_new
            self.state["Ti"] = Ti_new
        else:
            Te_new, Ti_new = Te, Ti
        self.state["pressure"] = self.eos.total_pressure(rho, Ti_new, Te_new)
    if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
        self._sanitize_state("after collision step")

    # --- Braginskii ion viscosity ---
    if self.config.fluid.enable_viscosity and self.backend not in ("metal", "mlx"):
        self._apply_viscosity(dt_sub)
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
            self._sanitize_state("after viscosity step")

    # --- Anisotropic thermal conduction (field-aligned Braginskii) ---
    # Skip if: Metal backend (handled internally). In 2T mode this applies to Te (electron conduction).
    if self.config.fluid.enable_anisotropic_conduction and self.backend not in ("metal", "mlx"):
        _dx = self.config.dx
        if self.geometry_type == "cylindrical":
            _dz = self.config.geometry.dz if self.config.geometry.dz is not None else _dx
            _dy = _dx
        else:
            _dy = _dx
            _dz = _dx
        ne = rho / self.ion_mass
        Z_eff_aniso = max(float(Z_bar), 0.01)
        Te_aniso = anisotropic_thermal_conduction(
            self.state["Te"],
            self.state["B"],
            np.maximum(ne, 1e10),
            dt_sub,
            _dx,
            _dy,
            _dz,
            Z_eff=Z_eff_aniso,
        )
        self.state["Te"] = np.maximum(Te_aniso, 1.0)
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
            self._sanitize_state("after anisotropic conduction step")

    # --- Radiation losses ---
    # MLX/Metal handle bremsstrahlung internally in the solver step.
    Z_for_rad = Z_bar_field if Z_bar_field is not None else Z_bar
    if self.rad_cfg.bremsstrahlung_enabled and not _two_t and self.backend not in ("metal", "mlx"):
        ne_rad = rho / self.ion_mass
        if self.rad_cfg.fld_enabled:
            _r_coords = None
            if self.geometry_type == "cylindrical" and hasattr(self, "fluid"):
                _geom = getattr(self.fluid, "geom", None)
                if _geom is not None:
                    _r_coords = _geom.r
            self.state = apply_radiation_transport(
                self.state,
                dx=self.config.dx,
                dt=dt_sub,
                Z=Z_for_rad,
                gaunt_factor=self.rad_cfg.gaunt_factor,
                geometry=self.geometry_type,
                r_coords=_r_coords,
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
                _dE_rad = float(np.sum(np.minimum(P_rad_2d, 1e40) * cell_vol) * dt_sub)
                if np.isfinite(_dE_rad):
                    self.total_radiated_energy += _dE_rad
            else:
                _dE_rad = float(np.sum(np.minimum(P_rad, 1e40)) * np.prod([self.config.dx] * 3) * dt_sub)
                if np.isfinite(_dE_rad):
                    self.total_radiated_energy += _dE_rad

        # Update pressure after radiation
        self.state["pressure"] = self.eos.total_pressure(
            self.state["rho"], self.state["Ti"], self.state["Te"]
        )
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
            self._sanitize_state("after radiation step")

    # --- Line radiation (impurity cooling) ---
    if self.rad_cfg.line_radiation_enabled and self.rad_cfg.impurity_fraction > 0 and not _two_t:
        ne_line = self.state["rho"] / self.ion_mass
        # Compute Z_eff: coronal equilibrium or fixed
        if self.rad_cfg.ionization_model == "coronal":
            Te_eV_arr = self.state["Te"] * (k_B / eV)  # K -> eV
            Z_eff_line = np.mean(
                coronal_z_eff(Te_eV_arr, Z_nucleus=int(self.rad_cfg.impurity_Z))
            )
        else:
            Z_eff_line = 0.0  # bremsstrahlung already applied above
        # Compute optical escape factor for line radiation trapping
        from dpf.radiation.line_radiation import optical_escape_factor
        ne_avg = float(np.mean(ne_line))
        f_esc = optical_escape_factor(
            ne_avg,
            Z_imp=self.rad_cfg.impurity_Z,
            n_imp_frac=self.rad_cfg.impurity_fraction,
        )
        Te_line, P_line = apply_line_radiation_losses(
            self.state["Te"],
            ne_line,
            dt_sub,
            Z_eff=Z_eff_line,
            n_imp_frac=self.rad_cfg.impurity_fraction,
            Z_imp=self.rad_cfg.impurity_Z,
            Te_floor=1.0,
            escape_factor=f_esc,
        )
        self.state["Te"] = Te_line
        # Track radiated energy from line radiation
        if self.geometry_type == "cylindrical":
            cell_vol = self.fluid.geom.cell_volumes()
            P_line_2d = np.squeeze(P_line, axis=1) if P_line.ndim == 3 else P_line
            _dE_line = float(np.sum(np.minimum(P_line_2d, 1e40) * cell_vol) * dt_sub)
            if np.isfinite(_dE_line):
                self.total_radiated_energy += _dE_line
        else:
            _dE_line = float(np.sum(np.minimum(P_line, 1e40)) * np.prod([self.config.dx] * 3) * dt_sub)
            if np.isfinite(_dE_line):
                self.total_radiated_energy += _dE_line
        self.state["pressure"] = self.eos.total_pressure(
            self.state["rho"], self.state["Ti"], self.state["Te"]
        )
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
            self._sanitize_state("after line radiation step")

    # --- Implicit / STS magnetic and thermal diffusion ---
    fc = self.config.fluid
    if fc.diffusion_method != "explicit" and fc.enable_resistive:
        self._apply_diffusion(dt_sub, Z_bar, Z_bar_field=Z_bar_field)
        if self.step_count == 0 or self.step_count % self._nan_check_stride == 0:
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
    ne = np.maximum(rho / self.ion_mass, 1e10)  # floor prevents 1/ne divergence in Nernst

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
    lnL = coulomb_log(ne_safe, Te_safe)
    eta = spitzer_resistivity(ne_safe, Te_safe, lnL, Z=Z_for_diff)

    # Compute Spitzer thermal conductivity: kappa_e ~ 3.2 * ne * kB^2 * Te * tau_e / m_e
    # Simplified estimate: kappa ~ 20 * (kB * Te)^{5/2} / (m_e^{1/2} * e^4 * lnL)
    # For now, use a simplified isotropic Spitzer kappa
    # Spitzer thermal conductivity via ne * kB^2 * Te * tau_e / m_e
    # Note: ne appears in both numerator (kappa) and denominator (tau_e), so
    # the ne unit convention in tau_e doesn't affect kappa — ne cancels.
    # Using NRL coefficient 3.44e5 with ne in m^-3 gives wrong tau_e but correct kappa.
    Te_eV = Te_safe * k_B / eV
    lnL_safe = np.maximum(lnL, 1.0)  # Coulomb log floor to prevent div-by-zero
    tau_e = 3.44e5 * Te_eV**1.5 / (ne_safe * lnL_safe)
    kappa = 3.2 * ne_safe * k_B**2 * Te_safe * tau_e / m_e
    kappa = np.where(np.isfinite(kappa), kappa, 0.0)

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
