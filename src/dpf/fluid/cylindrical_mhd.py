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
from numba import njit, prange

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
    _weno5_reconstruct_1d,
)
from dpf.geometry.cylindrical import CylindricalGeometry

logger = logging.getLogger(__name__)

# Default ion mass: deuterium
_DEFAULT_ION_MASS = m_d


# ============================================================
# Parallel WENO5 flux sweep kernels (Numba)
# Each line along the non-sweep axis is independent — prange
# distributes them across all available cores.
# ============================================================

@njit(cache=True, parallel=True)
def _weno5_sweep_hll_parallel(
    rho: np.ndarray,
    vel_n: np.ndarray,
    pressure: np.ndarray,
    Bn: np.ndarray,
    gamma: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel WENO5+HLL flux sweep along *axis* of a 2D (nr, nz) grid.

    Each 1D line perpendicular to *axis* is reconstructed and solved
    independently, so all n_other lines can run concurrently via prange.

    Args:
        rho: (nr, nz)
        vel_n: normal velocity (nr, nz)
        pressure: (nr, nz)
        Bn: normal B-field (nr, nz)
        gamma: adiabatic index
        axis: 0 = sweep along r (n_other = nz), 1 = sweep along z (n_other = nr)

    Returns:
        (F_rho, F_mom, F_ene) each shape (n_iface, n_other) for axis=0,
        or (n_other, n_iface) for axis=1.
    """
    nr = rho.shape[0]
    nz = rho.shape[1]

    if axis == 0:
        n_ax = nr
        n_other = nz
    else:
        n_ax = nz
        n_other = nr

    n_iface = n_ax - 4

    if axis == 0:
        F_rho = np.zeros((n_iface, n_other))
        F_mom = np.zeros((n_iface, n_other))
        F_ene = np.zeros((n_iface, n_other))
        for idx in prange(n_other):
            rho_1d = rho[:, idx]
            u_1d = vel_n[:, idx]
            p_1d = pressure[:, idx]
            Bn_1d = Bn[:, idx]
            rL, rR = _weno5_reconstruct_1d(rho_1d)
            uL, uR = _weno5_reconstruct_1d(u_1d)
            pL, pR = _weno5_reconstruct_1d(p_1d)
            BnL, BnR = _weno5_reconstruct_1d(Bn_1d)
            for k in range(len(rL)):
                if rL[k] < 1e-20:
                    rL[k] = 1e-20
                if rR[k] < 1e-20:
                    rR[k] = 1e-20
                if pL[k] < 1e-20:
                    pL[k] = 1e-20
                if pR[k] < 1e-20:
                    pR[k] = 1e-20
            f_rho, f_mom, f_ene = _hll_flux_1d_core(rL, rR, uL, uR, pL, pR, BnL, BnR, gamma)
            F_rho[:, idx] = f_rho
            F_mom[:, idx] = f_mom
            F_ene[:, idx] = f_ene
    else:
        F_rho = np.zeros((n_other, n_iface))
        F_mom = np.zeros((n_other, n_iface))
        F_ene = np.zeros((n_other, n_iface))
        for idx in prange(n_other):
            rho_1d = rho[idx, :]
            u_1d = vel_n[idx, :]
            p_1d = pressure[idx, :]
            Bn_1d = Bn[idx, :]
            rL, rR = _weno5_reconstruct_1d(rho_1d)
            uL, uR = _weno5_reconstruct_1d(u_1d)
            pL, pR = _weno5_reconstruct_1d(p_1d)
            BnL, BnR = _weno5_reconstruct_1d(Bn_1d)
            for k in range(len(rL)):
                if rL[k] < 1e-20:
                    rL[k] = 1e-20
                if rR[k] < 1e-20:
                    rR[k] = 1e-20
                if pL[k] < 1e-20:
                    pL[k] = 1e-20
                if pR[k] < 1e-20:
                    pR[k] = 1e-20
            f_rho, f_mom, f_ene = _hll_flux_1d_core(rL, rR, uL, uR, pL, pR, BnL, BnR, gamma)
            F_rho[idx, :] = f_rho
            F_mom[idx, :] = f_mom
            F_ene[idx, :] = f_ene

    return F_rho, F_mom, F_ene


# NOTE: HLLD parallel kernel removed — _hlld_flux_1d_core is not @njit
# decorated, so it can't be called from parallel=True context. The WENO5
# density sweep always uses HLL (which is JIT-compiled). The full 8-component
# HLLD Riemann solver is used separately in _hll_riemann_flux (class method).


# ============================================================
# Parallel PLM geometry kernels (Numba)
# These replace the pure-NumPy CylindricalGeometry methods on the PLM
# (non-WENO5) code path.  Each operates on a full 2D field and parallelises
# over rows or columns with prange.
#
# np.gradient uses:
#   interior: (f[i+1] - f[i-1]) / (2*h)   — 2nd-order centred
#   boundary: one-sided 1st-order forward/backward
# We replicate that stencil exactly so the PLM path stays bit-compatible
# with the previous np.gradient implementation while gaining parallelism.
# ============================================================


@njit(cache=True, parallel=True)
def _plm_grad1d_r(field: np.ndarray, dr: float) -> np.ndarray:
    """d(field)/dr via 2nd-order centred differences, parallelised over z."""
    nr = field.shape[0]
    nz = field.shape[1]
    out = np.empty((nr, nz))
    for j in prange(nz):
        out[0, j] = (field[1, j] - field[0, j]) / dr
        for i in range(1, nr - 1):
            out[i, j] = (field[i + 1, j] - field[i - 1, j]) / (2.0 * dr)
        out[nr - 1, j] = (field[nr - 1, j] - field[nr - 2, j]) / dr
    return out


@njit(cache=True, parallel=True)
def _plm_grad1d_z(field: np.ndarray, dz: float) -> np.ndarray:
    """d(field)/dz via 2nd-order centred differences, parallelised over r."""
    nr = field.shape[0]
    nz = field.shape[1]
    out = np.empty((nr, nz))
    for i in prange(nr):
        out[i, 0] = (field[i, 1] - field[i, 0]) / dz
        for j in range(1, nz - 1):
            out[i, j] = (field[i, j + 1] - field[i, j - 1]) / (2.0 * dz)
        out[i, nz - 1] = (field[i, nz - 1] - field[i, nz - 2]) / dz
    return out


@njit(cache=True, parallel=True)
def _plm_divergence_parallel(
    F_r: np.ndarray,
    F_z: np.ndarray,
    r: np.ndarray,
    inv_r: np.ndarray,
    dr: float,
    dz: float,
) -> np.ndarray:
    """Cylindrical divergence: (1/r)*d(r*Fr)/dr + dFz/dz.

    Parallelised over z-columns for the radial term and over r-rows for
    the axial term; combined result is correct per-cell.
    """
    nr = F_r.shape[0]
    nz = F_r.shape[1]

    # d(r * F_r)/dr  — parallelise over z
    rFr = np.empty((nr, nz))
    for i in prange(nr):
        for j in range(nz):
            rFr[i, j] = r[i] * F_r[i, j]

    div_r = _plm_grad1d_r(rFr, dr)
    # (1/r) weighting
    for i in prange(nr):
        for j in range(nz):
            div_r[i, j] = div_r[i, j] * inv_r[i]

    dFz_dz = _plm_grad1d_z(F_z, dz)

    out = np.empty((nr, nz))
    for i in prange(nr):
        for j in range(nz):
            out[i, j] = div_r[i, j] + dFz_dz[i, j]
    return out


@njit(cache=True, parallel=True)
def _plm_gradient_parallel(
    p: np.ndarray,
    dr: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cylindrical gradient: (dp/dr, 0, dp/dz).

    Returns three (nr, nz) arrays for r, theta, z components.
    """
    grad_r = _plm_grad1d_r(p, dr)
    grad_z = _plm_grad1d_z(p, dz)
    nr = p.shape[0]
    nz = p.shape[1]
    grad_theta = np.zeros((nr, nz))
    return grad_r, grad_theta, grad_z


@njit(cache=True, parallel=True)
def _plm_curl_parallel(
    B_r: np.ndarray,
    B_theta: np.ndarray,
    B_z: np.ndarray,
    r: np.ndarray,
    inv_r: np.ndarray,
    dr: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axisymmetric cylindrical curl of B.

    (curl B)_r     = -dB_theta/dz
    (curl B)_theta = dB_r/dz - dB_z/dr
    (curl B)_z     = (1/r) * d(r * B_theta)/dr
    """
    nr = B_r.shape[0]
    nz = B_r.shape[1]

    curl_r = np.empty((nr, nz))
    dBt_dz = _plm_grad1d_z(B_theta, dz)
    for i in prange(nr):
        for j in range(nz):
            curl_r[i, j] = -dBt_dz[i, j]

    dBr_dz = _plm_grad1d_z(B_r, dz)
    dBz_dr = _plm_grad1d_r(B_z, dr)
    curl_theta = np.empty((nr, nz))
    for i in prange(nr):
        for j in range(nz):
            curl_theta[i, j] = dBr_dz[i, j] - dBz_dr[i, j]

    rBtheta = np.empty((nr, nz))
    for i in prange(nr):
        for j in range(nz):
            rBtheta[i, j] = r[i] * B_theta[i, j]
    d_rBt_dr = _plm_grad1d_r(rBtheta, dr)
    curl_z = np.empty((nr, nz))
    for i in prange(nr):
        for j in range(nz):
            curl_z[i, j] = inv_r[i] * d_rBt_dr[i, j]

    return curl_r, curl_theta, curl_z


@njit(cache=True, parallel=True)
def _plm_div_B_parallel(
    B_r: np.ndarray,
    B_z: np.ndarray,
    r: np.ndarray,
    inv_r: np.ndarray,
    dr: float,
    dz: float,
) -> np.ndarray:
    """div(B) in cylindrical coords: (1/r)*d(r*Br)/dr + dBz/dz."""
    return _plm_divergence_parallel(B_r, B_z, r, inv_r, dr, dz)


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
        use_godunov_flux: bool = False,
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
        self.use_godunov_flux = use_godunov_flux
        self._last_eta_max = 0.0  # For resistive diffusion CFL
        self._last_div_B: float = 0.0
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
            "Riemann=%s, TimeInt=%s, Godunov=%s, ion_mass=%.3e kg",
            nr, nz, dr, dz, gamma, enable_hall,
            self.enable_resistive, self.enable_energy_equation,
            self.use_weno5, self.riemann_solver, self.time_integrator,
            self.use_godunov_flux, self.ion_mass,
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

        if n_ax < 5:
            return {
                "mass_flux": np.zeros_like(rho),
                "momentum_flux": np.zeros_like(rho),
                "energy_flux": np.zeros_like(rho),
                "n_interfaces": 0,
            }

        n_iface = n_ax - 4

        # Always use HLL parallel kernel for WENO5 density sweep.
        # HLLD is used in the full 8-component _hll_riemann_flux method.
        F_rho, F_mom, F_ene = _weno5_sweep_hll_parallel(
            rho, vel_n, pressure, Bn, self.gamma, axis,
        )

        return {
            "mass_flux": F_rho,
            "momentum_flux": F_mom,
            "energy_flux": F_ene,
            "n_interfaces": n_iface,
        }

    # ================================================================
    # Godunov (HLL) flux-based spatial update — shock-safe alternative
    # to np.gradient central differences.  Selected by use_godunov_flux=True.
    # ================================================================

    @staticmethod
    def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Minmod slope limiter: returns zero where signs differ."""
        return np.where(
            a * b > 0,
            np.where(np.abs(a) < np.abs(b), a, b),
            0.0,
        )

    def _plm_reconstruct(
        self, q: np.ndarray, axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """PLM reconstruction with minmod limiter along *axis*.

        Given cell averages q, compute left and right states at each
        cell interface i+1/2.  Interface i+1/2 sits between cells i and i+1.

        For N cells there are N-1 interior interfaces (indices 0 .. N-2).
        Return arrays have the interface dimension = N-1.

        Args:
            q: Field array, shape (nr, nz) for scalars or (nr, nz) slice.
            axis: 0 = radial, 1 = axial.

        Returns:
            (q_L, q_R) each of shape with interface dim = N-1 along *axis*.
            q_L[i] = value on the left side of interface i+1/2.
            q_R[i] = value on the right side of interface i+1/2.
        """
        # Slopes: delta[i] = minmod(q[i+1]-q[i], q[i]-q[i-1])
        # For interior cells 1..N-2 only; boundary cells get zero slope.
        if axis == 0:
            dq_fwd = q[1:, :] - q[:-1, :]  # shape (N-1, nz)
            slope = np.zeros_like(q)
            slope[1:-1, :] = self._minmod(dq_fwd[1:, :], dq_fwd[:-1, :])
            # L state at interface i+1/2 = q[i] + 0.5*slope[i]
            q_L = q[:-1, :] + 0.5 * slope[:-1, :]
            # R state at interface i+1/2 = q[i+1] - 0.5*slope[i+1]
            q_R = q[1:, :] - 0.5 * slope[1:, :]
        else:
            dq_fwd = q[:, 1:] - q[:, :-1]
            slope = np.zeros_like(q)
            slope[:, 1:-1] = self._minmod(dq_fwd[:, 1:], dq_fwd[:, :-1])
            q_L = q[:, :-1] + 0.5 * slope[:, :-1]
            q_R = q[:, 1:] - 0.5 * slope[:, 1:]
        return q_L, q_R

    def _hll_flux_8(
        self,
        rho_L: np.ndarray,
        rho_R: np.ndarray,
        vn_L: np.ndarray,
        vn_R: np.ndarray,
        vt1_L: np.ndarray,
        vt1_R: np.ndarray,
        vt2_L: np.ndarray,
        vt2_R: np.ndarray,
        Bn_L: np.ndarray,
        Bn_R: np.ndarray,
        Bt1_L: np.ndarray,
        Bt1_R: np.ndarray,
        Bt2_L: np.ndarray,
        Bt2_R: np.ndarray,
        p_L: np.ndarray,
        p_R: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """8-component HLL Riemann flux for ideal MHD.

        Conservative state U = [rho, rho*vn, rho*vt1, rho*vt2,
                                 E_total, Bn, Bt1, Bt2]

        Subscripts: n = normal to interface, t1/t2 = tangential.

        Returns dict with flux arrays for each conserved variable.
        """
        gamma = self.gamma
        gm1 = gamma - 1.0

        rho_L = np.maximum(rho_L, 1e-30)
        rho_R = np.maximum(rho_R, 1e-30)
        p_L = np.maximum(p_L, 1e-20)
        p_R = np.maximum(p_R, 1e-20)

        # Magnetic field squared
        B_sq_L = Bn_L**2 + Bt1_L**2 + Bt2_L**2
        B_sq_R = Bn_R**2 + Bt1_R**2 + Bt2_R**2

        # Fast magnetosonic speed
        a2_L = gamma * p_L / rho_L
        a2_R = gamma * p_R / rho_R
        va2_L = B_sq_L / (mu_0 * rho_L)
        va2_R = B_sq_R / (mu_0 * rho_R)
        # van^2 for full fast speed formula
        van2_L = Bn_L**2 / (mu_0 * rho_L)
        van2_R = Bn_R**2 / (mu_0 * rho_R)
        disc_L = np.maximum((a2_L + va2_L)**2 - 4.0 * a2_L * van2_L, 0.0)
        disc_R = np.maximum((a2_R + va2_R)**2 - 4.0 * a2_R * van2_R, 0.0)
        cf_L = np.sqrt(0.5 * (a2_L + va2_L + np.sqrt(disc_L)))
        cf_R = np.sqrt(0.5 * (a2_R + va2_R + np.sqrt(disc_R)))

        # Davis wave speed estimates
        S_L = np.minimum(vn_L - cf_L, vn_R - cf_R)
        S_R = np.maximum(vn_L + cf_L, vn_R + cf_R)

        # Conserved states
        v_sq_L = vn_L**2 + vt1_L**2 + vt2_L**2
        v_sq_R = vn_R**2 + vt1_R**2 + vt2_R**2
        E_L = p_L / gm1 + 0.5 * rho_L * v_sq_L + 0.5 * B_sq_L / mu_0
        E_R = p_R / gm1 + 0.5 * rho_R * v_sq_R + 0.5 * B_sq_R / mu_0

        p_tot_L = p_L + 0.5 * B_sq_L / mu_0
        p_tot_R = p_R + 0.5 * B_sq_R / mu_0

        vdotB_L = vn_L * Bn_L + vt1_L * Bt1_L + vt2_L * Bt2_L
        vdotB_R = vn_R * Bn_R + vt1_R * Bt1_R + vt2_R * Bt2_R

        # Left fluxes
        FL_rho = rho_L * vn_L
        FL_mn = rho_L * vn_L**2 + p_tot_L - Bn_L**2 / mu_0
        FL_mt1 = rho_L * vn_L * vt1_L - Bn_L * Bt1_L / mu_0
        FL_mt2 = rho_L * vn_L * vt2_L - Bn_L * Bt2_L / mu_0
        FL_E = (E_L + p_tot_L) * vn_L - Bn_L * vdotB_L / mu_0
        FL_Bn = np.zeros_like(rho_L)
        FL_Bt1 = vn_L * Bt1_L - vt1_L * Bn_L
        FL_Bt2 = vn_L * Bt2_L - vt2_L * Bn_L

        # Right fluxes
        FR_rho = rho_R * vn_R
        FR_mn = rho_R * vn_R**2 + p_tot_R - Bn_R**2 / mu_0
        FR_mt1 = rho_R * vn_R * vt1_R - Bn_R * Bt1_R / mu_0
        FR_mt2 = rho_R * vn_R * vt2_R - Bn_R * Bt2_R / mu_0
        FR_E = (E_R + p_tot_R) * vn_R - Bn_R * vdotB_R / mu_0
        FR_Bn = np.zeros_like(rho_R)
        FR_Bt1 = vn_R * Bt1_R - vt1_R * Bn_R
        FR_Bt2 = vn_R * Bt2_R - vt2_R * Bn_R

        # Conserved U arrays
        U_L_rho = rho_L
        U_R_rho = rho_R
        U_L_mn = rho_L * vn_L
        U_R_mn = rho_R * vn_R
        U_L_mt1 = rho_L * vt1_L
        U_R_mt1 = rho_R * vt1_R
        U_L_mt2 = rho_L * vt2_L
        U_R_mt2 = rho_R * vt2_R
        U_L_Bn = Bn_L
        U_R_Bn = Bn_R
        U_L_Bt1 = Bt1_L
        U_R_Bt1 = Bt1_R
        U_L_Bt2 = Bt2_L
        U_R_Bt2 = Bt2_R

        # HLL formula: F = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R-U_L)) / (S_R-S_L)
        denom = np.maximum(S_R - S_L, 1e-30)

        def _hll(FL: np.ndarray, FR: np.ndarray, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
            return (S_R * FL - S_L * FR + S_L * S_R * (UR - UL)) / denom

        return {
            "F_rho": _hll(FL_rho, FR_rho, U_L_rho, U_R_rho),
            "F_mn": _hll(FL_mn, FR_mn, U_L_mn, U_R_mn),
            "F_mt1": _hll(FL_mt1, FR_mt1, U_L_mt1, U_R_mt1),
            "F_mt2": _hll(FL_mt2, FR_mt2, U_L_mt2, U_R_mt2),
            "F_E": _hll(FL_E, FR_E, E_L, E_R),
            "F_Bn": _hll(FL_Bn, FR_Bn, U_L_Bn, U_R_Bn),
            "F_Bt1": _hll(FL_Bt1, FR_Bt1, U_L_Bt1, U_R_Bt1),
            "F_Bt2": _hll(FL_Bt2, FR_Bt2, U_L_Bt2, U_R_Bt2),
        }

    def _compute_godunov_rhs(
        self,
        rho: np.ndarray,
        vel: np.ndarray,
        p: np.ndarray,
        B: np.ndarray,
        psi: np.ndarray,
        eta_field: np.ndarray | None = None,
        source_terms: dict | None = None,
        e_electron: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Godunov (PLM+HLL) RHS for cylindrical MHD.

        Replaces np.gradient central differences for hyperbolic terms with
        PLM reconstruction + HLL Riemann fluxes.  Resistant to Gibbs
        oscillations at sheath discontinuities.

        The flux divergence uses a proper finite-volume form:
            dU/dt = -(1/V) * [A_{i+1/2}*F_{i+1/2} - A_{i-1/2}*F_{i-1/2}]
        where V = cell volume and A = face area.

        Source terms (geometric, resistive, Hall, Dedner) are unchanged.
        """
        geom = self.geom
        nr, nz = self.nr, self.nz
        _r = geom.r          # (nr,) — cell-center radii
        _inv_r = geom.inv_r  # (nr,) — 1/r

        # Precompute cell volumes and face areas (could be cached)
        cell_vol = geom.cell_volumes()       # (nr, nz)
        A_r = geom.face_areas_radial()       # (nr+1, nz)
        A_z = geom.face_areas_axial()        # (nr, nz+1)

        # Protect against zero volume at axis
        cell_vol = np.maximum(cell_vol, 1e-30)

        # ---- Radial flux sweep (axis=0) ----
        # Normal = r, tangential1 = theta, tangential2 = z
        rho_L_r, rho_R_r = self._plm_reconstruct(rho, axis=0)
        vr_L, vr_R = self._plm_reconstruct(vel[0], axis=0)
        vt_L, vt_R = self._plm_reconstruct(vel[1], axis=0)
        vz_L, vz_R = self._plm_reconstruct(vel[2], axis=0)
        Br_L, Br_R = self._plm_reconstruct(B[0], axis=0)
        Bt_L, Bt_R = self._plm_reconstruct(B[1], axis=0)
        Bz_L, Bz_R = self._plm_reconstruct(B[2], axis=0)
        p_L_r, p_R_r = self._plm_reconstruct(p, axis=0)

        # Positivity floors on reconstructed states
        rho_L_r = np.maximum(rho_L_r, 1e-20)
        rho_R_r = np.maximum(rho_R_r, 1e-20)
        p_L_r = np.maximum(p_L_r, 1e-20)
        p_R_r = np.maximum(p_R_r, 1e-20)

        flux_r = self._hll_flux_8(
            rho_L_r, rho_R_r,
            vr_L, vr_R,       # normal
            vt_L, vt_R,       # tangential 1 (theta)
            vz_L, vz_R,       # tangential 2 (z)
            Br_L, Br_R,       # Bn
            Bt_L, Bt_R,       # Bt1
            Bz_L, Bz_R,       # Bt2
            p_L_r, p_R_r,
        )
        # flux_r arrays have shape (nr-1, nz) — one per interior interface

        # ---- Axial flux sweep (axis=1) ----
        # Normal = z, tangential1 = r, tangential2 = theta
        rho_L_z, rho_R_z = self._plm_reconstruct(rho, axis=1)
        vz_L_z, vz_R_z = self._plm_reconstruct(vel[2], axis=1)
        vr_L_z, vr_R_z = self._plm_reconstruct(vel[0], axis=1)
        vt_L_z, vt_R_z = self._plm_reconstruct(vel[1], axis=1)
        Bz_L_z, Bz_R_z = self._plm_reconstruct(B[2], axis=1)
        Br_L_z, Br_R_z = self._plm_reconstruct(B[0], axis=1)
        Bt_L_z, Bt_R_z = self._plm_reconstruct(B[1], axis=1)
        p_L_z, p_R_z = self._plm_reconstruct(p, axis=1)

        rho_L_z = np.maximum(rho_L_z, 1e-20)
        rho_R_z = np.maximum(rho_R_z, 1e-20)
        p_L_z = np.maximum(p_L_z, 1e-20)
        p_R_z = np.maximum(p_R_z, 1e-20)

        flux_z = self._hll_flux_8(
            rho_L_z, rho_R_z,
            vz_L_z, vz_R_z,   # normal (z)
            vr_L_z, vr_R_z,   # tangential 1 (r)
            vt_L_z, vt_R_z,   # tangential 2 (theta)
            Bz_L_z, Bz_R_z,   # Bn
            Br_L_z, Br_R_z,   # Bt1
            Bt_L_z, Bt_R_z,   # Bt2
            p_L_z, p_R_z,
        )
        # flux_z arrays have shape (nr, nz-1)

        # ---- Passive scalar flux for e_electron (upwind) ----
        if e_electron is not None:
            ee_L_r, ee_R_r = self._plm_reconstruct(e_electron, axis=0)
            ee_L_r = np.maximum(ee_L_r, 0.0)
            ee_R_r = np.maximum(ee_R_r, 0.0)
            # Upwind based on mass flux sign
            mass_flux_r = flux_r["F_rho"]
            flux_r["F_ee"] = np.where(
                mass_flux_r >= 0,
                ee_L_r * vr_L / np.maximum(rho_L_r, 1e-30) * rho_L_r,
                ee_R_r * vr_R / np.maximum(rho_R_r, 1e-30) * rho_R_r,
            )
            # Simplify: F_ee = ee * v_n at the upwind side
            flux_r["F_ee"] = np.where(mass_flux_r >= 0, ee_L_r * vr_L, ee_R_r * vr_R)

            ee_L_z, ee_R_z = self._plm_reconstruct(e_electron, axis=1)
            ee_L_z = np.maximum(ee_L_z, 0.0)
            ee_R_z = np.maximum(ee_R_z, 0.0)
            mass_flux_z = flux_z["F_rho"]
            flux_z["F_ee"] = np.where(mass_flux_z >= 0, ee_L_z * vz_L_z, ee_R_z * vz_R_z)

        # ---- Flux divergence in finite-volume form ----
        # Radial: dU/dt -= (1/V) * [A_{i+1/2}*F_{i+1/2} - A_{i-1/2}*F_{i-1/2}]
        # Interior interfaces 0..nr-2 correspond to faces 1..nr-1
        # (face 0 = axis, face nr = outer boundary)
        drho_dt = np.zeros((nr, nz))
        dmom_r_dt = np.zeros((nr, nz))
        dmom_t_dt = np.zeros((nr, nz))
        dmom_z_dt = np.zeros((nr, nz))
        dE_dt = np.zeros((nr, nz))
        dBr_dt = np.zeros((nr, nz))
        dBt_dt = np.zeros((nr, nz))
        dBz_dt = np.zeros((nr, nz))

        # Radial contribution: interfaces 0..nr-2 map to faces 1..nr-1
        # Cell i receives flux from face i (left) and face i+1 (right).
        # Interface index j corresponds to face j+1.
        def _apply_radial_flux(F_key: str) -> np.ndarray:
            """Compute -div(F_r) contribution for one conserved variable.

            Uses area-weighted fluxes on all nr+1 faces.  Face 0 (axis) has
            A=0 so contributes nothing.  Face nr (outer boundary) uses
            zero-gradient extrapolation from the last interior interface.
            """
            # Raw fluxes at interior interfaces (nr-1 values, faces 1..nr-1)
            F_int = flux_r[F_key]  # (nr-1, nz)
            # Build full face flux array (nr+1 faces: 0..nr)
            F_full = np.zeros((nr + 1, nz))
            F_full[1:nr, :] = F_int  # interior faces
            # Face 0 (axis): zero flux (A_r[0]=0 anyway)
            # Face nr (outer boundary): zero-gradient extrapolation
            F_full[nr, :] = F_int[-1, :]
            # Area-weighted flux at each face
            AF = F_full * A_r  # (nr+1, nz)
            # Flux divergence: -(AF_{i+1} - AF_i) / V_i
            result = -(AF[1:, :] - AF[:-1, :]) / cell_vol
            return result

        drho_dt += _apply_radial_flux("F_rho")
        dmom_r_dt += _apply_radial_flux("F_mn")
        dmom_t_dt += _apply_radial_flux("F_mt1")
        dmom_z_dt += _apply_radial_flux("F_mt2")
        dE_dt += _apply_radial_flux("F_E")
        dBr_dt += _apply_radial_flux("F_Bn")
        dBt_dt += _apply_radial_flux("F_Bt1")
        dBz_dt += _apply_radial_flux("F_Bt2")

        def _apply_axial_flux(F_key: str) -> np.ndarray:
            """Compute -div(F_z) contribution with zero-gradient BCs."""
            F_int = flux_z[F_key]  # (nr, nz-1)
            F_full = np.zeros((nr, nz + 1))
            F_full[:, 1:nz] = F_int
            # Face 0 (z=0): zero-gradient
            F_full[:, 0] = F_int[:, 0]
            # Face nz (z=L): zero-gradient
            F_full[:, nz] = F_int[:, -1]
            AF = F_full * A_z
            result = -(AF[:, 1:] - AF[:, :-1]) / cell_vol
            return result

        drho_dt += _apply_axial_flux("F_rho")
        # Axial sweep: n=z, t1=r, t2=theta
        dmom_z_dt += _apply_axial_flux("F_mn")    # normal momentum → z
        dmom_r_dt += _apply_axial_flux("F_mt1")   # tangential1 → r
        dmom_t_dt += _apply_axial_flux("F_mt2")   # tangential2 → theta
        dE_dt += _apply_axial_flux("F_E")
        dBz_dt += _apply_axial_flux("F_Bn")
        dBr_dt += _apply_axial_flux("F_Bt1")
        dBt_dt += _apply_axial_flux("F_Bt2")

        # Electron energy advection
        dee_dt = np.zeros((nr, nz))
        if e_electron is not None:
            dee_dt += _apply_radial_flux("F_ee")
            dee_dt += _apply_axial_flux("F_ee")

        # Assemble momentum and B-field derivatives
        dmom_dt = np.zeros((3, nr, nz))
        dmom_dt[0] = dmom_r_dt
        dmom_dt[1] = dmom_t_dt
        dmom_dt[2] = dmom_z_dt

        dB_dt = np.zeros((3, nr, nz))
        dB_dt[0] = dBr_dt
        dB_dt[1] = dBt_dt
        dB_dt[2] = dBz_dt

        # ---- Source terms (same as central-difference path) ----

        # Geometric source terms (hoop stress, centrifugal)
        S_geom = geom.geometric_source_momentum(rho, vel, p, B)
        dmom_dt += S_geom

        # Current density for resistive/Hall terms: J = curl(B) / mu_0
        cBr, cBt, cBz = _plm_curl_parallel(B[0], B[1], B[2], _r, _inv_r, self.dr, self.dz)
        curl_B = np.empty((3, nr, nz))
        curl_B[0] = cBr
        curl_B[1] = cBt
        curl_B[2] = cBz
        J = curl_B / mu_0

        # Kinetic current coupling
        J_total = J
        src = source_terms or {}
        if "J_kin" in src:
            J_kin = src["J_kin"]
            if J_kin.ndim == 4:
                J_kin = J_kin[:, :, 0, :]
            J_total = J - J_kin

        # Resistive + Hall electric field for induction correction
        E_field = np.zeros((3, nr, nz))
        ohmic_heating = np.zeros((nr, nz))
        if self.enable_resistive and eta_field is not None:
            for d in range(3):
                E_field[d] = eta_field * J_total[d]
            J_sq = np.sum(J_total**2, axis=0)
            ohmic_heating = eta_field * J_sq

        if self.enable_hall:
            ne = rho / self.ion_mass
            ne_safe = np.maximum(ne, 1e-20)
            E_Hall = np.zeros((3, nr, nz))
            E_Hall[0] = (J_total[1] * B[2] - J_total[2] * B[1]) / (ne_safe * e_charge)
            E_Hall[1] = (J_total[2] * B[0] - J_total[0] * B[2]) / (ne_safe * e_charge)
            E_Hall[2] = (J_total[0] * B[1] - J_total[1] * B[0]) / (ne_safe * e_charge)
            E_field = E_field + E_Hall

        # Resistive/Hall correction to induction (added on top of HLL ideal fluxes)
        if self.enable_resistive or self.enable_hall:
            eEr, eEt, eEz = _plm_curl_parallel(
                E_field[0], E_field[1], E_field[2], _r, _inv_r, self.dr, self.dz,
            )
            dB_dt[0] -= eEr
            dB_dt[1] -= eEt
            dB_dt[2] -= eEz

        # External source terms
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

        dE_dt += total_heating
        if ext_dE is not None:
            ext_dE_2d = self._squeeze(ext_dE) if ext_dE.ndim == 3 else ext_dE
            dE_dt = dE_dt + ext_dE_2d

        # Dedner divergence cleaning
        dpsi_dt = np.zeros_like(psi)
        if not self.enable_ct:
            if self.dedner_ch_init > 0:
                ch = self.dedner_ch_init
            else:
                B_sq_ded = np.sum(B**2, axis=0)
                cs2_ded = self.gamma * p / np.maximum(rho, 1e-30)
                va2_ded = B_sq_ded / (mu_0 * np.maximum(rho, 1e-30))
                cf_ded = np.sqrt(cs2_ded + va2_ded)
                v_abs = np.sqrt(np.sum(vel**2, axis=0))
                ch = max(float(np.max(v_abs + cf_ded)), 1.0)
            cp = ch
            div_B = _plm_div_B_parallel(B[0], B[2], _r, _inv_r, self.dr, self.dz)
            self._last_div_B = float(np.max(np.abs(div_B)))
            dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi
            gpsi_r, gpsi_t, gpsi_z = _plm_gradient_parallel(psi, self.dr, self.dz)
            dB_dt[0] -= gpsi_r
            dB_dt[1] -= gpsi_t
            dB_dt[2] -= gpsi_z

        # Godunov path always returns conservative energy
        result = {
            "drho_dt": drho_dt,
            "dmom_dt": dmom_dt,
            "dB_dt": dB_dt,
            "dpsi_dt": dpsi_dt,
            "dE_dt": dE_dt,
            "ohmic_heating": ohmic_heating,
            "E_field": E_field,
        }
        if e_electron is not None:
            result["dee_dt"] = dee_dt
        return result

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
        e_electron: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute RHS of the MHD equations in cylindrical coordinates.

        All arrays are 2D: scalars (nr, nz), vectors (3, nr, nz).

        Args:
            rho, vel, p, B, psi: State variables.
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nr, nz).
            source_terms: Optional dict with external source terms.
            e_electron: Electron energy density [J/m³], shape (nr, nz). Optional.

        Returns time derivatives for all state variables.
        """
        # Delegate to Godunov (PLM+HLL) path if enabled
        if self.use_godunov_flux:
            return self._compute_godunov_rhs(
                rho, vel, p, B, psi, eta_field, source_terms, e_electron,
            )

        geom = self.geom
        _r = geom.r          # shape (nr,)
        _inv_r = geom.inv_r  # shape (nr,)

        # --- Current density: J = curl(B) / mu_0 ---
        cBr, cBt, cBz = _plm_curl_parallel(
            B[0], B[1], B[2], _r, _inv_r, self.dr, self.dz,
        )
        curl_B = np.empty((3, self.nr, self.nz))
        curl_B[0] = cBr
        curl_B[1] = cBt
        curl_B[2] = cBz
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
            drho_dt = -_plm_divergence_parallel(
                rho * vel[0], rho * vel[2], _r, _inv_r, self.dr, self.dz,
            )

        # --- Momentum: d(rho*v)/dt = -div(rho*v*v) - grad(p) + J×B + S_geom ---
        # Pressure gradient
        gpr, gpt, gpz = _plm_gradient_parallel(p, self.dr, self.dz)
        grad_p = np.empty((3, self.nr, self.nz))
        grad_p[0] = gpr
        grad_p[1] = gpt
        grad_p[2] = gpz

        # J × B force
        JxB = np.zeros((3, self.nr, self.nz))
        JxB[0] = J[1] * B[2] - J[2] * B[1]
        JxB[1] = J[2] * B[0] - J[0] * B[2]
        JxB[2] = J[0] * B[1] - J[1] * B[0]

        # Momentum advection: -div(rho * v_d * v) for each component d
        dmom_dt = np.zeros((3, self.nr, self.nz))
        for d in range(3):
            dmom_dt[d] = -_plm_divergence_parallel(
                rho * vel[d] * vel[0], rho * vel[d] * vel[2],
                _r, _inv_r, self.dr, self.dz,
            )

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

        # --- Kinetic current coupling (Frontier E: PIC → MHD) ---
        J_total = J
        if source_terms and "J_kin" in source_terms:
            J_kin = source_terms["J_kin"]
            if J_kin.ndim == 4:  # (3, nr, 1, nz) → squeeze to (3, nr, nz)
                J_kin = J_kin[:, :, 0, :]
            J_total = J - J_kin  # J_kin already carried by kinetic particles; subtract to avoid double-counting

        # --- Resistive term: E_resistive = eta * J_total ---
        ohmic_heating = np.zeros((self.nr, self.nz))
        if self.enable_resistive and eta_field is not None:
            E_resistive = np.zeros((3, self.nr, self.nz))
            for d in range(3):
                E_resistive[d] = eta_field * J_total[d]
            E_field = E_field + E_resistive
            # Ohmic heating: Q_ohm = eta * |J_total|^2 [W/m^3]
            J_sq = np.sum(J_total**2, axis=0)
            ohmic_heating = eta_field * J_sq

        # Hall term: E_Hall = (J_total × B) / (ne * e)
        if self.enable_hall:
            ne = rho / self.ion_mass
            ne_safe = np.maximum(ne, 1e-20)
            E_Hall = np.zeros((3, self.nr, self.nz))
            E_Hall[0] = (J_total[1] * B[2] - J_total[2] * B[1]) / (ne_safe * e_charge)
            E_Hall[1] = (J_total[2] * B[0] - J_total[0] * B[2]) / (ne_safe * e_charge)
            E_Hall[2] = (J_total[0] * B[1] - J_total[1] * B[0]) / (ne_safe * e_charge)
            E_field = E_field + E_Hall

        eBr, eBt, eBz = _plm_curl_parallel(
            E_field[0], E_field[1], E_field[2], _r, _inv_r, self.dr, self.dz,
        )
        dB_dt = np.empty((3, self.nr, self.nz))
        dB_dt[0] = -eBr
        dB_dt[1] = -eBt
        dB_dt[2] = -eBz

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
            dE_dt = -_plm_divergence_parallel(
                F_E[0], F_E[2], _r, _inv_r, self.dr, self.dz,
            ) + total_heating
            if ext_dE is not None:
                ext_dE_2d = self._squeeze(ext_dE) if ext_dE.ndim == 3 else ext_dE
                dE_dt = dE_dt + ext_dE_2d

            # Convert dE/dt to dp/dt using Metal solver's pressure recovery formula
            # (matches metal_riemann.py lines 1677-1682)
            v_dot_dmom = np.sum(vel * dmom_dt, axis=0)
            B_dot_dB = np.sum(B * dB_dt, axis=0)
            dp_dt = (self.gamma - 1.0) * (dE_dt - v_dot_dmom + 0.5 * v_sq * drho_dt - B_dot_dB)
            dE_dt = None
        else:
            div_v = _plm_divergence_parallel(
                vel[0], vel[2], _r, _inv_r, self.dr, self.dz,
            )
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
            div_B = _plm_div_B_parallel(B[0], B[2], _r, _inv_r, self.dr, self.dz)
            self._last_div_B = float(np.max(np.abs(div_B)))
            dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi
            gpsi_r, gpsi_t, gpsi_z = _plm_gradient_parallel(psi, self.dr, self.dz)
            dB_dt[0] -= gpsi_r
            dB_dt[1] -= gpsi_t
            dB_dt[2] -= gpsi_z

        # Electron energy advection: dee/dt = -div(ee * v)
        if e_electron is not None:
            dee_dt = -_plm_divergence_parallel(
                e_electron * vel[0], e_electron * vel[2], _r, _inv_r, self.dr, self.dz,
            )
        else:
            dee_dt = None

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
        if dee_dt is not None:
            result["dee_dt"] = dee_dt
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
        e_electron: np.ndarray | None = None,
    ) -> tuple:
        """Compute one forward-Euler stage: U^(1) = U^n + dt * L(U^n).

        Returns:
            (rho, mom, p, B, psi, rhs, E_total_or_None, e_electron_or_None)
        """
        vel = mom / np.maximum(rho[np.newaxis, :, :], 1e-30)
        rhs = self._compute_rhs(rho, vel, p, B, psi, eta_2d, source_terms, e_electron)
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
            # Inter-stage velocity clamping: prevent kinetic energy from exceeding total energy
            B_sq_new = np.sum(B_new**2, axis=0)
            E_internal_min = 0.01 * E_total_new  # Reserve at least 1% for internal energy
            KE_max = np.maximum(E_total_new - B_sq_new / (2.0 * mu_0) - E_internal_min, 0.0)
            v_sq_new = np.sum(vel_new**2, axis=0)
            KE_actual = 0.5 * rho_new * v_sq_new
            v_scale = np.where(KE_actual > KE_max, np.sqrt(KE_max / np.maximum(KE_actual, 1e-30)), 1.0)
            vel_new = vel_new * v_scale[np.newaxis, :, :]
            mom_new = rho_new[np.newaxis, :, :] * vel_new
            v_sq_new = np.sum(vel_new**2, axis=0)
            p_new = np.maximum(
                gm1 * (E_total_new - 0.5 * rho_new * v_sq_new - B_sq_new / (2.0 * mu_0)),
                1e-20,
            )
        else:
            p_new = np.maximum(p + dt * rhs["dp_dt"], 1e-20)

        # Axis boundary conditions: v_r=0, B_r=0 at r=0
        mom_new[0, 0, :] = 0.0
        B_new[0, 0, :] = 0.0

        # Electron energy advection update
        ee_new = None
        if e_electron is not None and "dee_dt" in rhs:
            ee_new = np.maximum(e_electron + dt * rhs["dee_dt"], 0.0)

        return rho_new, mom_new, p_new, B_new, psi_new, rhs, E_total_new, ee_new

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

        # Squeeze e_electron if present
        e_electron_in = state.get("e_electron")
        ee_2d = None
        ee_n = None
        if e_electron_in is not None:
            ee_2d = self._squeeze(e_electron_in) if e_electron_in.ndim == 3 else e_electron_in
            ee_n = ee_2d.copy()

        # === Stage 1: U^(1) = U^n + dt * L(U^n) ===
        rho_1, mom_1, p_1, B_1, psi_1, rhs1, E_1, ee_1 = self._euler_stage(
            rho_n, mom_n, p_n, B_n, psi_n, dt, eta_2d, source_terms, ee_n,
        )
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(B_1, current, anode_radius, cathode_radius)

        if self.time_integrator == "ssp_rk3":
            # === Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt * L(U^(1))) ===
            rho_2e, mom_2e, p_2e, B_2e, psi_2e, rhs2, E_2e, ee_2e = self._euler_stage(
                rho_1, mom_1, p_1, B_1, psi_1, dt, eta_2d, source_terms, ee_1,
            )
            rho_2 = np.maximum(0.75 * rho_n + 0.25 * rho_2e, 1e-10)
            mom_2 = 0.75 * mom_n + 0.25 * mom_2e
            B_2 = 0.75 * B_n + 0.25 * B_2e
            psi_2 = 0.75 * psi_n + 0.25 * psi_2e
            ee_2 = np.maximum(0.75 * ee_n + 0.25 * ee_2e, 0.0) if ee_n is not None else None

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
            rho_3e, mom_3e, p_3e, B_3e, psi_3e, rhs3, E_3e, ee_3e = self._euler_stage(
                rho_2, mom_2, p_2, B_2, psi_2, dt, eta_2d, source_terms, ee_2,
            )
            rho_new = np.maximum((1.0 / 3.0) * rho_n + (2.0 / 3.0) * rho_3e, 1e-10)
            mom_new = (1.0 / 3.0) * mom_n + (2.0 / 3.0) * mom_3e
            B_new = (1.0 / 3.0) * B_n + (2.0 / 3.0) * B_3e
            psi_new = (1.0 / 3.0) * psi_n + (2.0 / 3.0) * psi_3e
            ee_new_adv = np.maximum((1.0 / 3.0) * ee_n + (2.0 / 3.0) * ee_3e, 0.0) if ee_n is not None else None

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
            rho_2e, mom_2e, p_2e, B_2e, psi_2e, rhs2, E_2e, ee_2e = self._euler_stage(
                rho_1, mom_1, p_1, B_1, psi_1, dt, eta_2d, source_terms, ee_1,
            )
            rho_new = np.maximum(0.5 * rho_n + 0.5 * rho_2e, 1e-10)
            mom_new = 0.5 * mom_n + 0.5 * mom_2e
            B_new = 0.5 * B_n + 0.5 * B_2e
            psi_new = 0.5 * psi_n + 0.5 * psi_2e
            ee_new_adv = np.maximum(0.5 * ee_n + 0.5 * ee_2e, 0.0) if ee_n is not None else None

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

        # --- Two-temperature update ---
        n_i = rho_new / self.ion_mass
        n_i_safe = np.maximum(n_i, 1e-30)

        if ee_new_adv is not None:
            # True 2T: apply source terms to ADVECTED electron energy
            from dpf.fluid.two_temperature import step_electron_energy
            e_e_2d = ee_new_adv  # Use advected value, not original state
            eta_eff = eta_2d if eta_2d is not None else np.zeros_like(rho_new)
            J_sq = ohmic_avg / np.maximum(eta_eff, 1e-30) if np.any(eta_eff > 0) else np.zeros_like(rho_new)
            e_e_new, Te_new, Ti_new = step_electron_energy(
                rho_e_e=e_e_2d, rho=rho_new,
                velocity=vel_new, eta=eta_eff,
                J_sq=J_sq, Te=Te, Ti=Ti,
                n_e=n_i_safe, n_i=n_i_safe,
                dx=self.geom.dr, dt=dt,
                Z=1.0, gaunt_factor=1.2,
                gamma=self.gamma,
            )
        else:
            # Fraction-preserving hack (legacy fallback)
            e_e_new = None
            Te_old = Te
            Ti_old = Ti
            T_sum_old = np.maximum(Te_old + Ti_old, 1.0)
            f_e = Te_old / T_sum_old
            T_total_new = p_new / np.maximum(n_i_safe * k_B, 1e-30)
            Te_new = f_e * T_total_new
            Ti_new = (1.0 - f_e) * T_total_new
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
        result = {
            "rho": self._unsqueeze(rho_new),
            "velocity": self._unsqueeze(vel_new),
            "pressure": self._unsqueeze(p_new),
            "B": self._unsqueeze(B_new),
            "Te": self._unsqueeze(Te_new),
            "Ti": self._unsqueeze(Ti_new),
            "psi": self._unsqueeze(psi_new),
        }
        if e_e_new is not None:
            result["e_electron"] = self._unsqueeze(e_e_new)
        return result

    def coupling_interface(self) -> CouplingState:
        return self._coupling
