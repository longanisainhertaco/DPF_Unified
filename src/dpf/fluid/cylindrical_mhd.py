"""2D axisymmetric cylindrical MHD solver for Dense Plasma Focus.

Implements the same interface as the Cartesian MHDSolver but in (r, z)
cylindrical coordinates with azimuthal symmetry. The plasma state is
stored on a 3D array with ny=1 for compatibility: shape (nr, 1, nz).

Key differences from Cartesian:
1. Geometric source terms (hoop stress, centrifugal)
2. Cylindrical divergence: div(F) = (1/r)*d(rF_r)/dr + dF_z/dz
3. Cylindrical curl for induction equation
4. Cell volumes proportional to r: dV = 2*pi*r*dr*dz
5. WENO5+HLL flux sweeps use face areas (2*pi*r*dz for radial faces)

Vector ordering: (v_r, v_theta, v_z) stored as state["velocity"][0,1,2]
Magnetic field: (B_r, B_theta, B_z) stored as state["B"][0,1,2]

For the DPF, the dominant components are:
- B_theta (azimuthal, from axial current)
- v_r (radial pinch), v_z (axial rundown)
- J_z (axial current density)

Reference:
    Stone & Norman, ApJS 80:753 (1992) — ZEUS-2D
    Mignone et al., ApJS 170:228 (2007) — PLUTO code
"""

from __future__ import annotations

import logging

import numpy as np

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_d, mu_0
from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.fluid.constrained_transport import (
    cell_centered_to_face,
    compute_div_B,
    ct_update,
    emf_from_fluxes,
    face_to_cell_centered,
)
from dpf.fluid.eos import IdealEOS
from dpf.fluid.mhd_solver import (
    _hll_flux_1d_core,
    _hlld_flux_1d_core,
    _weno5_reconstruct_1d,
)
from dpf.geometry.cylindrical import CylindricalGeometry

logger = logging.getLogger(__name__)

# Default ion mass: deuterium
_DEFAULT_ION_MASS = m_d


class CylindricalMHDSolver(PlasmaSolverBase):
    """2D axisymmetric Hall MHD solver in (r, z) cylindrical coordinates.

    Uses the same state dictionary interface as the Cartesian MHDSolver:
        rho:      shape (nr, 1, nz)
        velocity: shape (3, nr, 1, nz)  — components (v_r, v_theta, v_z)
        pressure: shape (nr, 1, nz)
        B:        shape (3, nr, 1, nz)  — components (B_r, B_theta, B_z)
        Te:       shape (nr, 1, nz)
        Ti:       shape (nr, 1, nz)
        psi:      shape (nr, 1, nz)

    The ny=1 dimension is squeezed internally for 2D operations.

    Args:
        nr: Number of radial cells.
        nz: Number of axial cells.
        dr: Radial grid spacing [m].
        dz: Axial grid spacing [m].
        gamma: Adiabatic index.
        cfl: CFL number for timestep.
        dedner_ch: Dedner cleaning speed (0 = auto).
        enable_hall: Enable Hall term.
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        dr: float,
        dz: float,
        gamma: float = 5.0 / 3.0,
        cfl: float = 0.4,
        dedner_ch: float = 0.0,
        enable_hall: bool = True,
        enable_resistive: bool = True,
        enable_energy_equation: bool = True,
        ion_mass: float | None = None,
        riemann_solver: str = "hll",
        enable_ct: bool = False,
        time_integrator: str = "ssp_rk3",
        conservative_energy: bool = True,
    ) -> None:
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        self.gamma = gamma
        self.cfl = cfl
        self.dedner_ch_init = dedner_ch
        self.enable_hall = enable_hall
        self.enable_resistive = enable_resistive
        self.enable_energy_equation = enable_energy_equation
        self.ion_mass = ion_mass if ion_mass is not None else _DEFAULT_ION_MASS
        self.riemann_solver = riemann_solver if riemann_solver in ("hll", "hlld") else "hll"
        self.time_integrator = time_integrator if time_integrator in ("ssp_rk2", "ssp_rk3") else "ssp_rk3"
        self.conservative_energy = conservative_energy
        self._last_eta_max = 0.0  # For resistive diffusion CFL
        # CT is disabled in cylindrical mode — the CT implementation uses Cartesian
        # metric (see H5 in Troubleshooting.md). Use Dedner cleaning instead.
        if enable_ct:
            logger.warning(
                "CT is not supported in cylindrical coordinates (uses Cartesian metric). "
                "Falling back to Dedner divergence cleaning."
            )
        self.enable_ct = False
        self.eos = IdealEOS(gamma=gamma, ion_mass=self.ion_mass)

        # Whether we can use WENO5 (need >= 5 cells in each direction)
        self.use_weno5 = nr >= 5 and nz >= 5

        # Geometry operator
        self.geom = CylindricalGeometry(nr, nz, dr, dz)

        # Coupling state
        self._coupling = CouplingState()
        self._prev_Lp: float | None = None

        # Grid shape for compatibility with Cartesian interface
        self.grid_shape = (nr, 1, nz)

        logger.info(
            "CylindricalMHDSolver initialized: (nr=%d, nz=%d), dr=%.2e, dz=%.2e, "
            "gamma=%.3f, Hall=%s, Resistive=%s, EnergyEq=%s, WENO5=%s, "
            "Riemann=%s, TimeInt=%s, ion_mass=%.3e kg",
            nr, nz, dr, dz, gamma, enable_hall,
            self.enable_resistive, self.enable_energy_equation,
            self.use_weno5, self.riemann_solver, self.time_integrator, self.ion_mass,
        )

    def _squeeze(self, arr: np.ndarray) -> np.ndarray:
        """Squeeze the ny=1 dimension for 2D operations.

        3D (nr, 1, nz) -> 2D (nr, nz)
        4D (3, nr, 1, nz) -> 3D (3, nr, nz)
        """
        return np.squeeze(arr, axis=-2 if arr.ndim == 3 else 2)

    def _unsqueeze(self, arr: np.ndarray) -> np.ndarray:
        """Restore the ny=1 dimension.

        2D (nr, nz) -> 3D (nr, 1, nz)
        3D (3, nr, nz) -> 4D (3, nr, 1, nz)
        """
        if arr.ndim == 2:
            return arr[:, np.newaxis, :]
        elif arr.ndim == 3 and arr.shape[0] == 3:
            return arr[:, :, np.newaxis, :]
        return arr

    def _weno5_flux_sweep_2d(
        self,
        rho: np.ndarray,
        vel_n: np.ndarray,
        pressure: np.ndarray,
        Bn: np.ndarray,
        axis: int,
    ) -> dict[str, np.ndarray]:
        """WENO5+Riemann flux sweep along one axis of a 2D (nr, nz) grid.

        Args:
            rho: Density (nr, nz).
            vel_n: Normal velocity component (nr, nz).
            pressure: Pressure (nr, nz).
            Bn: Normal B-field component (nr, nz).
            axis: 0 for radial, 1 for axial.

        Returns:
            Dict with mass_flux, momentum_flux, energy_flux, n_interfaces.
        """
        n_ax = rho.shape[axis]
        n_other = rho.shape[1 - axis]

        if n_ax < 5:
            return {
                "mass_flux": np.zeros_like(rho),
                "momentum_flux": np.zeros_like(rho),
                "energy_flux": np.zeros_like(rho),
                "n_interfaces": 0,
            }

        n_iface = n_ax - 4
        if axis == 0:
            out_shape = (n_iface, n_other)
        else:
            out_shape = (n_other, n_iface)

        F_rho = np.zeros(out_shape)
        F_mom = np.zeros(out_shape)
        F_ene = np.zeros(out_shape)

        riemann_fn = _hlld_flux_1d_core if self.riemann_solver == "hlld" else _hll_flux_1d_core

        for idx in range(n_other):
            if axis == 0:
                rho_1d = rho[:, idx]
                u_1d = vel_n[:, idx]
                p_1d = pressure[:, idx]
                Bn_1d = Bn[:, idx]
            else:
                rho_1d = rho[idx, :]
                u_1d = vel_n[idx, :]
                p_1d = pressure[idx, :]
                Bn_1d = Bn[idx, :]

            rL, rR = _weno5_reconstruct_1d(rho_1d)
            uL, uR = _weno5_reconstruct_1d(u_1d)
            pL, pR = _weno5_reconstruct_1d(p_1d)
            BnL, BnR = _weno5_reconstruct_1d(Bn_1d)

            rL = np.maximum(rL, 1e-20)
            rR = np.maximum(rR, 1e-20)
            pL = np.maximum(pL, 1e-20)
            pR = np.maximum(pR, 1e-20)

            f_rho, f_mom, f_ene = riemann_fn(
                rL, rR, uL, uR, pL, pR, BnL, BnR, self.gamma,
            )

            if axis == 0:
                F_rho[:, idx] = f_rho
                F_mom[:, idx] = f_mom
                F_ene[:, idx] = f_ene
            else:
                F_rho[idx, :] = f_rho
                F_mom[idx, :] = f_mom
                F_ene[idx, :] = f_ene

        return {
            "mass_flux": F_rho,
            "momentum_flux": F_mom,
            "energy_flux": F_ene,
            "n_interfaces": n_iface,
        }

    def _compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """Compute CFL-limited timestep for cylindrical geometry."""
        rho = self._squeeze(state["rho"])
        v = self._squeeze(state["velocity"])
        B = self._squeeze(state["B"])
        p = self._squeeze(state["pressure"])

        # Fast magnetosonic speed
        B_sq = np.sum(B**2, axis=0)
        a2 = self.gamma * p / np.maximum(rho, 1e-30)
        va2 = B_sq / (mu_0 * np.maximum(rho, 1e-30))
        cf = np.sqrt(a2 + va2)

        v_max_r = np.max(np.abs(v[0])) + np.max(cf)
        v_max_z = np.max(np.abs(v[2])) + np.max(cf)

        # Hall speed limit
        if self.enable_hall:
            ne = rho / self.ion_mass
            ne_max = np.max(ne)
            if ne_max > 0:
                B_max = np.sqrt(np.max(B_sq))
                dx_min = min(self.dr, self.dz)
                v_hall = B_max / (mu_0 * np.maximum(ne_max, 1e-20) * e_charge * dx_min)
                v_max_r = max(v_max_r, v_hall)
                v_max_z = max(v_max_z, v_hall)

        dt_r = self.cfl * self.dr / max(v_max_r, 1e-30)
        dt_z = self.cfl * self.dz / max(v_max_z, 1e-30)
        dt = min(dt_r, dt_z)

        # Resistive diffusion CFL: dt < 0.5 * dx^2 * mu_0 / eta_max
        if self.enable_resistive and hasattr(self, "_last_eta_max") and self._last_eta_max > 0:
            dx_min = min(self.dr, self.dz)
            dt_diff = 0.5 * dx_min**2 * mu_0 / self._last_eta_max
            dt = min(dt, dt_diff)

        if dt < 1e-30:
            dt = 1e-10
        return dt

    def _compute_rhs(
        self,
        rho: np.ndarray,
        vel: np.ndarray,
        p: np.ndarray,
        B: np.ndarray,
        psi: np.ndarray,
        eta_field: np.ndarray | None = None,
        source_terms: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute RHS of the MHD equations in cylindrical coordinates.

        All arrays are 2D: scalars (nr, nz), vectors (3, nr, nz).

        Args:
            rho, vel, p, B, psi: State variables.
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nr, nz).
            source_terms: Optional dict with external source terms.

        Returns time derivatives for all state variables.
        """
        geom = self.geom

        # --- Current density: J = curl(B) / mu_0 ---
        curl_B = geom.curl(B)
        J = curl_B / mu_0

        # --- Density: d(rho)/dt = -div(rho*v) ---
        if self.use_weno5:
            # WENO5+Riemann flux-based density update along r and z
            drho_dt = np.zeros_like(rho)
            # Radial sweep (axis=0): use v_r and B_r
            fl_r = self._weno5_flux_sweep_2d(rho, vel[0], p, B[0], axis=0)
            n_r = fl_r["n_interfaces"]
            if n_r >= 2:
                n_upd = n_r - 1
                dF = fl_r["mass_flux"][1:n_upd + 1, :] - fl_r["mass_flux"][:n_upd, :]
                drho_dt[2:2 + n_upd, :] -= dF / self.dr
            # Axial sweep (axis=1): use v_z and B_z
            fl_z = self._weno5_flux_sweep_2d(rho, vel[2], p, B[2], axis=1)
            n_z = fl_z["n_interfaces"]
            if n_z >= 2:
                n_upd = n_z - 1
                dF = fl_z["mass_flux"][:, 1:n_upd + 1] - fl_z["mass_flux"][:, :n_upd]
                drho_dt[:, 2:2 + n_upd] -= dF / self.dz
        else:
            rho_v = np.zeros((3, self.nr, self.nz))
            for d in range(3):
                rho_v[d] = rho * vel[d]
            drho_dt = -geom.divergence(rho_v)

        # --- Momentum: d(rho*v)/dt = -div(rho*v*v) - grad(p) + J×B + S_geom ---
        # Pressure gradient
        grad_p = geom.gradient(p)

        # J × B force
        JxB = np.zeros((3, self.nr, self.nz))
        JxB[0] = J[1] * B[2] - J[2] * B[1]
        JxB[1] = J[2] * B[0] - J[0] * B[2]
        JxB[2] = J[0] * B[1] - J[1] * B[0]

        # Momentum advection: -div(rho * v_d * v) for each component d
        dmom_dt = np.zeros((3, self.nr, self.nz))
        for d in range(3):
            mom_flux = np.zeros((3, self.nr, self.nz))
            for axis in range(3):
                mom_flux[axis] = rho * vel[d] * vel[axis]
            dmom_dt[d] = -geom.divergence(mom_flux)

        # Add forces
        for d in range(3):
            dmom_dt[d] += JxB[d] - grad_p[d]

        # Geometric source terms (hoop stress, centrifugal)
        S_geom = geom.geometric_source_momentum(rho, vel, p, B)
        dmom_dt += S_geom

        # --- Induction: dB/dt = -curl(E) ---
        # Ideal MHD: E = -v × B
        vxB = np.zeros((3, self.nr, self.nz))
        vxB[0] = vel[1] * B[2] - vel[2] * B[1]
        vxB[1] = vel[2] * B[0] - vel[0] * B[2]
        vxB[2] = vel[0] * B[1] - vel[1] * B[0]
        E_field = -vxB

        # --- Resistive term: E_resistive = eta * J ---
        ohmic_heating = np.zeros((self.nr, self.nz))
        if self.enable_resistive and eta_field is not None:
            E_resistive = np.zeros((3, self.nr, self.nz))
            for d in range(3):
                E_resistive[d] = eta_field * J[d]
            E_field = E_field + E_resistive
            # Ohmic heating: Q_ohm = eta * |J|^2 [W/m^3]
            J_sq = np.sum(J**2, axis=0)
            ohmic_heating = eta_field * J_sq

        # Hall term: E_Hall = (J × B) / (ne * e)
        if self.enable_hall:
            ne = rho / self.ion_mass
            ne_safe = np.maximum(ne, 1e-20)
            E_Hall = np.zeros((3, self.nr, self.nz))
            E_Hall[0] = (J[1] * B[2] - J[2] * B[1]) / (ne_safe * e_charge)
            E_Hall[1] = (J[2] * B[0] - J[0] * B[2]) / (ne_safe * e_charge)
            E_Hall[2] = (J[0] * B[1] - J[1] * B[0]) / (ne_safe * e_charge)
            E_field = E_field + E_Hall

        dB_dt = -geom.curl(E_field)

        # --- Energy equation ---
        # External source terms (snowplow, ohmic correction, etc.)
        src = source_terms or {}
        ext_drho = src.get("S_rho_snowplow")
        ext_dmom = src.get("S_mom_snowplow")
        ext_dE = src.get("S_energy_snowplow")
        Q_ohmic_corr = src.get("Q_ohmic_correction")

        if ext_drho is not None:
            ext_drho_2d = self._squeeze(ext_drho) if ext_drho.ndim == 3 else ext_drho
            drho_dt = drho_dt + ext_drho_2d
        if ext_dmom is not None:
            ext_dmom_2d = self._squeeze(ext_dmom) if ext_dmom.ndim == 4 else ext_dmom
            dmom_dt = dmom_dt + ext_dmom_2d

        total_heating = ohmic_heating
        if Q_ohmic_corr is not None:
            Q_corr_2d = self._squeeze(Q_ohmic_corr) if Q_ohmic_corr.ndim == 3 else Q_ohmic_corr
            total_heating = total_heating + Q_corr_2d

        if self.conservative_energy and self.enable_energy_equation:
            # Conservative total energy: E = p/(γ-1) + 0.5·ρ·v² + B²/(2μ₀)
            gm1 = self.gamma - 1.0
            v_sq = np.sum(vel**2, axis=0)
            B_sq = np.sum(B**2, axis=0)
            E_total = p / gm1 + 0.5 * rho * v_sq + B_sq / (2.0 * mu_0)
            p_total = p + B_sq / (2.0 * mu_0)
            v_dot_B = np.sum(vel * B, axis=0)

            # Energy flux vector: F_E = (E + p_tot)·v - B·(v·B)
            F_E = np.zeros((3, self.nr, self.nz))
            for d in range(3):
                F_E[d] = (E_total + p_total) * vel[d] - B[d] * v_dot_B

            # dE/dt = -div(F_E) + Q_ohm + Q_ext
            dE_dt = -geom.divergence(F_E) + total_heating
            if ext_dE is not None:
                ext_dE_2d = self._squeeze(ext_dE) if ext_dE.ndim == 3 else ext_dE
                dE_dt = dE_dt + ext_dE_2d

            # dp_dt not used when conservative — set to None sentinel
            dp_dt = None
        else:
            div_v = geom.divergence(vel)
            if self.enable_energy_equation:
                dp_dt = -self.gamma * p * div_v + (self.gamma - 1.0) * total_heating
            else:
                dp_dt = -self.gamma * p * div_v
            dE_dt = None

        # --- Dedner cleaning (skipped when CT is active) ---
        dpsi_dt = np.zeros_like(psi)
        if not self.enable_ct:
            if self.dedner_ch_init > 0:
                ch = self.dedner_ch_init
            else:
                # Use max(|v| + c_f) where c_f is the fast magnetosonic speed
                B_sq_ded = np.sum(B**2, axis=0)
                cs2_ded = self.gamma * p / np.maximum(rho, 1e-30)
                va2_ded = B_sq_ded / (mu_0 * np.maximum(rho, 1e-30))
                cf_ded = np.sqrt(cs2_ded + va2_ded)
                v_abs = np.sqrt(np.sum(vel**2, axis=0))
                ch = max(float(np.max(v_abs + cf_ded)), 1.0)
            cp = ch
            div_B = geom.div_B_cylindrical(B)
            dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi
            grad_psi = geom.gradient(psi)
            dB_dt = dB_dt - grad_psi

        result = {
            "drho_dt": drho_dt,
            "dmom_dt": dmom_dt,
            "dB_dt": dB_dt,
            "dpsi_dt": dpsi_dt,
            "ohmic_heating": ohmic_heating,
            "E_field": E_field,
        }
        if dE_dt is not None:
            result["dE_dt"] = dE_dt
        else:
            result["dp_dt"] = dp_dt
        return result

    def apply_electrode_bfield_bc(
        self,
        B: np.ndarray,
        current: float,
        anode_radius: float,
        cathode_radius: float,
    ) -> np.ndarray:
        """Apply electrode B-field BC in cylindrical coordinates.

        Imposes B_theta = mu_0 * I / (2 * pi * r) at cells near the
        electrode radii. This is the magnetic piston that drives the DPF.

        Also enforces axis symmetry: B_r = 0 at r=0.

        Args:
            B: Magnetic field (3, nr, nz) in 2D cylindrical.
            current: Circuit current [A].
            anode_radius: Anode radius [m].
            cathode_radius: Cathode radius [m].

        Returns:
            Modified B-field array.
        """
        r = self.geom.r  # shape (nr,)

        # Handle 4D input (3, nr, 1, nz) by squeezing to (3, nr, nz)
        needs_unsqueeze = B.ndim == 4
        if needs_unsqueeze:
            B = self._squeeze(B)

        # Enforce axis symmetry: B_r = 0 at r=0
        B[0, 0, :] = 0.0

        if abs(current) < 1e-10:
            if needs_unsqueeze:
                B = self._unsqueeze(B)
            return B

        # Find cells closest to cathode_radius (outer electrode)
        idx_cath = np.argmin(np.abs(r - cathode_radius))
        # Find cells closest to anode_radius (inner electrode)
        idx_anode = np.argmin(np.abs(r - anode_radius))

        # Apply B_theta = mu_0 * I / (2 * pi * r) at cathode boundary
        r_cath = max(r[idx_cath], 1e-10)
        B_theta_cath = mu_0 * current / (2.0 * np.pi * r_cath)
        B[1, idx_cath, :] = B_theta_cath

        # If there's more than one cell to the cathode, also set the last cell
        if idx_cath < self.nr - 1:
            r_outer = max(r[-1], 1e-10)
            B[1, -1, :] = mu_0 * current / (2.0 * np.pi * r_outer)

        # Apply B_theta at anode boundary
        if idx_anode > 0:
            r_an = max(r[idx_anode], 1e-10)
            B_theta_anode = mu_0 * current / (2.0 * np.pi * r_an)
            B[1, idx_anode, :] = B_theta_anode

        # For cells between anode and cathode at the closed end (z=0, insulator face),
        # impose B_theta = mu_0*I/(2*pi*r).  The open end (z=nz-1) uses zero-gradient
        # extrapolation — forcing B_theta there is non-physical for Mather-type geometry
        # where the sheath exits freely.  Reference: Lee (1984), Scholz (2006).
        for iz in [0]:  # Only closed end (insulator face)
            for ir in range(idx_anode, min(idx_cath + 1, self.nr)):
                r_local = max(r[ir], 1e-10)
                B[1, ir, iz] = mu_0 * current / (2.0 * np.pi * r_local)
        # Open end (z=nz-1): zero-gradient extrapolation
        B[1, :, -1] = B[1, :, -2]

        if needs_unsqueeze:
            B = self._unsqueeze(B)
        return B

    def _euler_stage(
        self,
        rho: np.ndarray,
        mom: np.ndarray,
        p: np.ndarray,
        B: np.ndarray,
        psi: np.ndarray,
        dt: float,
        eta_2d: np.ndarray | None,
        source_terms: dict | None = None,
    ) -> tuple:
        """Compute one forward-Euler stage: U^(1) = U^n + dt * L(U^n).

        Returns:
            (rho, mom, p, B, psi, rhs, E_total_or_None)
        """
        vel = mom / np.maximum(rho[np.newaxis, :, :], 1e-30)
        rhs = self._compute_rhs(rho, vel, p, B, psi, eta_2d, source_terms)
        rho_new = np.maximum(rho + dt * rhs["drho_dt"], 1e-10)
        mom_new = mom + dt * rhs["dmom_dt"]
        B_new = B + dt * rhs["dB_dt"]
        psi_new = psi + dt * rhs["dpsi_dt"]

        E_total_new = None
        if "dE_dt" in rhs:
            # Conservative energy path: evolve E_total, recover p
            gm1 = self.gamma - 1.0
            v_sq = np.sum(vel**2, axis=0)
            B_sq = np.sum(B**2, axis=0)
            E_n = p / gm1 + 0.5 * rho * v_sq + B_sq / (2.0 * mu_0)
            E_total_new = np.maximum(E_n + dt * rhs["dE_dt"], 1e-20)
            # Recover pressure from updated conserved variables
            vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
            v_sq_new = np.sum(vel_new**2, axis=0)
            B_sq_new = np.sum(B_new**2, axis=0)
            p_new = np.maximum(
                gm1 * (E_total_new - 0.5 * rho_new * v_sq_new - B_sq_new / (2.0 * mu_0)),
                1e-20,
            )
        else:
            p_new = np.maximum(p + dt * rhs["dp_dt"], 1e-20)

        # Axis boundary conditions: v_r=0, B_r=0 at r=0
        mom_new[0, 0, :] = 0.0
        B_new[0, 0, :] = 0.0

        return rho_new, mom_new, p_new, B_new, psi_new, rhs, E_total_new

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
        anode_radius: float = 0.0,
        cathode_radius: float = 0.0,
        apply_electrode_bc: bool = False,
        source_terms: dict | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Advance MHD state by one timestep using SSP-RK3 (default) or SSP-RK2.

        When conservative_energy=True (default), total energy E is the conserved
        variable for the SSP combination instead of pressure. Pressure is recovered
        after each stage via p = (γ-1)·(E - 0.5·ρ·v² - B²/(2μ₀)).

        Args:
            state: Dictionary with 3D arrays (nr, 1, nz).
            dt: Timestep [s].
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nr, 1, nz).
            anode_radius: Anode radius [m] for electrode BC.
            cathode_radius: Cathode radius [m] for electrode BC.
            apply_electrode_bc: Whether to apply electrode B-field BC.
            source_terms: External source terms (snowplow, ohmic correction).

        Returns:
            Updated state dictionary with 3D arrays.
        """
        # Squeeze to 2D
        rho = self._squeeze(state["rho"])
        vel = self._squeeze(state["velocity"])
        p = self._squeeze(state["pressure"])
        B = self._squeeze(state["B"])
        Te = self._squeeze(state.get("Te", np.full((self.nr, 1, self.nz), 1e4)))
        Ti = self._squeeze(state.get("Ti", np.full((self.nr, 1, self.nz), 1e4)))
        psi = self._squeeze(state.get("psi", np.zeros((self.nr, 1, self.nz))))

        # Squeeze eta_field if provided
        eta_2d = None
        if eta_field is not None:
            eta_2d = self._squeeze(eta_field) if eta_field.ndim == 3 else eta_field
            self._last_eta_max = float(np.max(eta_2d))

        # Save U^n
        rho_n = rho.copy()
        p_n = p.copy()
        B_n = B.copy()
        psi_n = psi.copy()
        mom_n = rho_n[np.newaxis, :, :] * vel.copy()

        # Compute E_total^n for conservative SSP combining
        use_E = self.conservative_energy and self.enable_energy_equation
        gm1 = self.gamma - 1.0
        if use_E:
            v_sq_n = np.sum(vel**2, axis=0)
            B_sq_n = np.sum(B_n**2, axis=0)
            E_n = p_n / gm1 + 0.5 * rho_n * v_sq_n + B_sq_n / (2.0 * mu_0)

        # === Stage 1: U^(1) = U^n + dt * L(U^n) ===
        rho_1, mom_1, p_1, B_1, psi_1, rhs1, E_1 = self._euler_stage(
            rho_n, mom_n, p_n, B_n, psi_n, dt, eta_2d, source_terms,
        )
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(B_1, current, anode_radius, cathode_radius)

        if self.time_integrator == "ssp_rk3":
            # === Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt * L(U^(1))) ===
            rho_2e, mom_2e, p_2e, B_2e, psi_2e, rhs2, E_2e = self._euler_stage(
                rho_1, mom_1, p_1, B_1, psi_1, dt, eta_2d, source_terms,
            )
            rho_2 = np.maximum(0.75 * rho_n + 0.25 * rho_2e, 1e-10)
            mom_2 = 0.75 * mom_n + 0.25 * mom_2e
            B_2 = 0.75 * B_n + 0.25 * B_2e
            psi_2 = 0.75 * psi_n + 0.25 * psi_2e

            if use_E and E_2e is not None:
                # SSP combine on conserved E_total, then recover p
                E_2 = np.maximum(0.75 * E_n + 0.25 * E_2e, 1e-20)
                vel_2 = mom_2 / np.maximum(rho_2[np.newaxis, :, :], 1e-30)
                v_sq_2 = np.sum(vel_2**2, axis=0)
                B_sq_2 = np.sum(B_2**2, axis=0)
                p_2 = np.maximum(gm1 * (E_2 - 0.5 * rho_2 * v_sq_2 - B_sq_2 / (2.0 * mu_0)), 1e-20)
            else:
                p_2 = np.maximum(0.75 * p_n + 0.25 * p_2e, 1e-20)
                E_2 = None

            if apply_electrode_bc and cathode_radius > 0:
                B_2 = self.apply_electrode_bfield_bc(B_2, current, anode_radius, cathode_radius)

            # === Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt * L(U^(2))) ===
            rho_3e, mom_3e, p_3e, B_3e, psi_3e, rhs3, E_3e = self._euler_stage(
                rho_2, mom_2, p_2, B_2, psi_2, dt, eta_2d, source_terms,
            )
            rho_new = np.maximum((1.0 / 3.0) * rho_n + (2.0 / 3.0) * rho_3e, 1e-10)
            mom_new = (1.0 / 3.0) * mom_n + (2.0 / 3.0) * mom_3e
            B_new = (1.0 / 3.0) * B_n + (2.0 / 3.0) * B_3e
            psi_new = (1.0 / 3.0) * psi_n + (2.0 / 3.0) * psi_3e

            if use_E and E_3e is not None:
                E_new = np.maximum((1.0 / 3.0) * E_n + (2.0 / 3.0) * E_3e, 1e-20)
                vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
                v_sq_new = np.sum(vel_new**2, axis=0)
                B_sq_new = np.sum(B_new**2, axis=0)
                p_new = np.maximum(gm1 * (E_new - 0.5 * rho_new * v_sq_new - B_sq_new / (2.0 * mu_0)), 1e-20)
            else:
                p_new = np.maximum((1.0 / 3.0) * p_n + (2.0 / 3.0) * p_3e, 1e-20)

            vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
            ohmic_avg = (1.0 / 3.0) * (rhs1["ohmic_heating"] + rhs2["ohmic_heating"] + rhs3["ohmic_heating"])
        else:
            # === SSP-RK2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1))) ===
            rho_2e, mom_2e, p_2e, B_2e, psi_2e, rhs2, E_2e = self._euler_stage(
                rho_1, mom_1, p_1, B_1, psi_1, dt, eta_2d, source_terms,
            )
            rho_new = np.maximum(0.5 * rho_n + 0.5 * rho_2e, 1e-10)
            mom_new = 0.5 * mom_n + 0.5 * mom_2e
            B_new = 0.5 * B_n + 0.5 * B_2e
            psi_new = 0.5 * psi_n + 0.5 * psi_2e

            if use_E and E_2e is not None:
                E_new = np.maximum(0.5 * E_n + 0.5 * E_2e, 1e-20)
                vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
                v_sq_new = np.sum(vel_new**2, axis=0)
                B_sq_new = np.sum(B_new**2, axis=0)
                p_new = np.maximum(gm1 * (E_new - 0.5 * rho_new * v_sq_new - B_sq_new / (2.0 * mu_0)), 1e-20)
            else:
                p_new = np.maximum(0.5 * p_n + 0.5 * p_2e, 1e-20)

            vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
            ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs2["ohmic_heating"])

        # Cap velocity at 10x the fast magnetosonic speed to prevent runaway
        B_sq = np.sum(B_new**2, axis=0)
        cs2 = self.gamma * p_new / np.maximum(rho_new, 1e-20)
        va2 = B_sq / (mu_0 * np.maximum(rho_new, 1e-20))
        v_max = 10.0 * np.sqrt(np.maximum(cs2 + va2, 1e-10))
        v_mag = np.sqrt(np.sum(vel_new**2, axis=0))
        v_excess = v_mag / np.maximum(v_max, 1e-30)
        limiter = np.where(v_excess > 1.0, 1.0 / np.maximum(v_excess, 1e-30), 1.0)
        vel_new *= limiter[np.newaxis, :, :]

        # Final axis BC enforcement: v_r=0, B_r=0 at r=0
        vel_new[0, 0, :] = 0.0
        B_new[0, 0, :] = 0.0

        # Apply electrode BC after stage 2
        if apply_electrode_bc and cathode_radius > 0:
            B_new = self.apply_electrode_bfield_bc(
                B_new, current, anode_radius, cathode_radius,
            )

        # --- Constrained transport correction (optional) ---
        if self.enable_ct:
            # Average E-field from both RK stages
            E_avg = 0.5 * (rhs1["E_field"] + rhs2["E_field"])
            # Expand 2D (3, nr, nz) -> 3D (3, nr, 1, nz) for CT module
            E_3d = E_avg[:, :, np.newaxis, :]
            B_3d = B_new[:, :, np.newaxis, :]

            # Convert cell-centred B to face-centred
            staggered = cell_centered_to_face(
                B_3d[0], B_3d[1], B_3d[2],
                dx=self.dr, dy=self.dr, dz=self.dz,
            )
            # Compute edge EMFs from face-centred E-field contributions
            E_face_x = np.zeros((self.nr + 1, 1, self.nz))
            E_face_y = np.zeros((self.nr, 2, self.nz))
            E_face_z = np.zeros((self.nr, 1, self.nz + 1))
            # Use E_avg components as face flux contributions
            for d in range(3):
                E_face_x[:-1, :, :] += 0.5 * E_3d[d, :, :, :] / 3.0
                E_face_x[1:, :, :] += 0.5 * E_3d[d, :, :, :] / 3.0
            Ex_edge, Ey_edge, Ez_edge = emf_from_fluxes(
                E_face_x, E_face_y, E_face_z,
                dx=self.dr, dy=self.dr, dz=self.dz,
            )
            # Apply CT update
            staggered_new = ct_update(staggered, Ex_edge, Ey_edge, Ez_edge, dt)
            # Convert back to cell-centred
            Bx_cc, By_cc, Bz_cc = face_to_cell_centered(staggered_new)
            B_new[0] = Bx_cc[:, 0, :]
            B_new[1] = By_cc[:, 0, :]
            B_new[2] = Bz_cc[:, 0, :]
            # Store div(B) for diagnostics
            self._last_div_B = float(np.max(np.abs(compute_div_B(staggered_new))))

        # --- Two-temperature update (preserve Te ≠ Ti) ---
        n_i = rho_new / self.ion_mass
        n_i_safe = np.maximum(n_i, 1e-30)

        # Preserve Te/(Te+Ti) fraction through pressure-based temperature split
        Te_old = Te
        Ti_old = Ti
        T_sum_old = np.maximum(Te_old + Ti_old, 1.0)
        f_e = Te_old / T_sum_old  # Electron fraction of total temperature

        # Total temperature from new pressure: T_total = p_new / (n_i * k_B)
        T_total_new = p_new / np.maximum(n_i_safe * k_B, 1e-30)
        Te_new = f_e * T_total_new
        Ti_new = (1.0 - f_e) * T_total_new
        # Ohmic heating preferentially heats electrons
        dTe_ohmic = (2.0 / 3.0) * ohmic_avg * dt / np.maximum(n_i_safe * k_B, 1e-30)
        Te_new = Te_new + dTe_ohmic

        Te_new = np.maximum(Te_new, 1.0)
        Ti_new = np.maximum(Ti_new, 1.0)

        # Cap temperatures at physically reasonable maximum (100 keV ~ 1.16e9 K)
        T_max = 1.16e9  # 100 keV in Kelvin
        Te_new = np.minimum(Te_new, T_max)
        Ti_new = np.minimum(Ti_new, T_max)

        # --- Update coupling ---
        # Lp from magnetic energy: Lp = 2*W_mag/I² = ∫B²/µ₀ dV / I²
        # Standard energy-based inductance formula for coaxial geometry.
        # For a z-pinch: Lp = (µ₀/2π)*z*ln(b/a) emerges naturally.
        if current > 0:
            B_sq = B_new[0] ** 2 + B_new[1] ** 2 + B_new[2] ** 2
            cell_vol = self.geom.cell_volumes()  # (nr, nz), includes 2πr factor
            Lp_est = float(np.sum(B_sq / mu_0 * cell_vol)) / (current**2 + 1e-30)
        else:
            Lp_est = 0.0

        if self._prev_Lp is not None and dt > 0:
            dL_dt: float | None = (Lp_est - self._prev_Lp) / dt
        else:
            dL_dt = None
        self._prev_Lp = Lp_est

        self._coupling = CouplingState(
            Lp=Lp_est,
            current=current,
            voltage=voltage,
            dL_dt=dL_dt,
        )

        # Unsqueeze back to 3D (nr, 1, nz)
        return {
            "rho": self._unsqueeze(rho_new),
            "velocity": self._unsqueeze(vel_new),
            "pressure": self._unsqueeze(p_new),
            "B": self._unsqueeze(B_new),
            "Te": self._unsqueeze(Te_new),
            "Ti": self._unsqueeze(Ti_new),
            "psi": self._unsqueeze(psi_new),
        }

    def coupling_interface(self) -> CouplingState:
        return self._coupling
