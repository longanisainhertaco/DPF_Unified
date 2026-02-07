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
        self.enable_ct = enable_ct
        self.eos = IdealEOS(gamma=gamma)

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
            "Riemann=%s, CT=%s, ion_mass=%.3e kg",
            nr, nz, dr, dz, gamma, enable_hall,
            self.enable_resistive, self.enable_energy_equation,
            self.use_weno5, self.riemann_solver, self.enable_ct, self.ion_mass,
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
    ) -> dict[str, np.ndarray]:
        """Compute RHS of the MHD equations in cylindrical coordinates.

        All arrays are 2D: scalars (nr, nz), vectors (3, nr, nz).

        Args:
            rho, vel, p, B, psi: State variables.
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nr, nz).

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

        # --- Pressure / Energy equation ---
        div_v = geom.divergence(vel)
        if self.enable_energy_equation:
            # dp/dt = -gamma * p * div(v) + (gamma-1) * eta * J^2
            dp_dt = -self.gamma * p * div_v + (self.gamma - 1.0) * ohmic_heating
        else:
            dp_dt = -self.gamma * p * div_v

        # --- Dedner cleaning (skipped when CT is active) ---
        dpsi_dt = np.zeros_like(psi)
        if not self.enable_ct:
            ch = self.dedner_ch_init if self.dedner_ch_init > 0 else max(np.max(np.abs(vel)), 1.0)
            cp = ch
            div_B = geom.div_B_cylindrical(B)
            dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi
            grad_psi = geom.gradient(psi)
            dB_dt = dB_dt - grad_psi

        return {
            "drho_dt": drho_dt,
            "dmom_dt": dmom_dt,
            "dp_dt": dp_dt,
            "dB_dt": dB_dt,
            "dpsi_dt": dpsi_dt,
            "ohmic_heating": ohmic_heating,
            "E_field": E_field,
        }

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

        # Enforce axis symmetry: B_r = 0 at r=0
        B[0, 0, :] = 0.0

        if abs(current) < 1e-10:
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

        # For cells between anode and cathode at the z-boundaries (electrodes),
        # impose B_theta profile at z=0 and z=nz-1 (electrode faces)
        for iz in [0, self.nz - 1]:
            for ir in range(idx_anode, min(idx_cath + 1, self.nr)):
                r_local = max(r[ir], 1e-10)
                B[1, ir, iz] = mu_0 * current / (2.0 * np.pi * r_local)

        return B

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
    ) -> dict[str, np.ndarray]:
        """Advance MHD state by one timestep using SSP-RK2.

        Same interface as Cartesian MHDSolver.step().
        Internally squeezes to 2D, computes, then unsqueezes to 3D.

        Args:
            state: Dictionary with 3D arrays (nr, 1, nz).
            dt: Timestep [s].
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nr, 1, nz).
            anode_radius: Anode radius [m] for electrode BC.
            cathode_radius: Cathode radius [m] for electrode BC.
            apply_electrode_bc: Whether to apply electrode B-field BC.

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

        # Save U^n
        rho_n = rho.copy()
        vel_n = vel.copy()
        p_n = p.copy()
        B_n = B.copy()
        psi_n = psi.copy()
        mom_n = rho_n[np.newaxis, :, :] * vel_n

        # === Stage 1: U^(1) = U^n + dt * L(U^n) ===
        rhs1 = self._compute_rhs(rho_n, vel_n, p_n, B_n, psi_n, eta_2d)

        rho_1 = np.maximum(rho_n + dt * rhs1["drho_dt"], 1e-20)
        mom_1 = mom_n + dt * rhs1["dmom_dt"]
        vel_1 = mom_1 / np.maximum(rho_1[np.newaxis, :, :], 1e-30)
        p_1 = np.maximum(p_n + dt * rhs1["dp_dt"], 1e-20)
        B_1 = B_n + dt * rhs1["dB_dt"]
        psi_1 = psi_n + dt * rhs1["dpsi_dt"]

        # Apply electrode BC after stage 1
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(
                B_1, current, anode_radius, cathode_radius,
            )

        # === Stage 2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1))) ===
        rhs2 = self._compute_rhs(rho_1, vel_1, p_1, B_1, psi_1, eta_2d)

        rho_new = np.maximum(0.5 * rho_n + 0.5 * (rho_1 + dt * rhs2["drho_dt"]), 1e-20)
        mom_new = 0.5 * mom_n + 0.5 * (mom_1 + dt * rhs2["dmom_dt"])
        vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
        p_new = np.maximum(0.5 * p_n + 0.5 * (p_1 + dt * rhs2["dp_dt"]), 1e-20)
        B_new = 0.5 * B_n + 0.5 * (B_1 + dt * rhs2["dB_dt"])
        psi_new = 0.5 * psi_n + 0.5 * (psi_1 + dt * rhs2["dpsi_dt"])

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
        ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs2["ohmic_heating"])
        dTe_ohmic = (2.0 / 3.0) * ohmic_avg * dt / np.maximum(n_i_safe * k_B, 1e-30)
        Te_new = Te_new + dTe_ohmic

        Te_new = np.maximum(Te_new, 1.0)
        Ti_new = np.maximum(Ti_new, 1.0)

        # --- Update coupling ---
        # In cylindrical coords, B_theta is the azimuthal field from axial current
        B_theta_avg = np.mean(np.abs(B_new[1]))
        if current > 0:
            Lp_est = mu_0 * B_theta_avg / (current + 1e-30) * self.dz * self.nz
        else:
            Lp_est = 0.0

        if self._prev_Lp is not None and dt > 0:
            dL_dt = (Lp_est - self._prev_Lp) / dt
        else:
            dL_dt = 0.0
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
