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
from dpf.constants import k_B, m_p, mu_0
from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.fluid.eos import IdealEOS
from dpf.geometry.cylindrical import CylindricalGeometry

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        self.gamma = gamma
        self.cfl = cfl
        self.dedner_ch_init = dedner_ch
        self.enable_hall = enable_hall
        self.eos = IdealEOS(gamma=gamma)

        # Geometry operator
        self.geom = CylindricalGeometry(nr, nz, dr, dz)

        # Coupling state
        self._coupling = CouplingState()
        self._prev_Lp: float | None = None

        # Grid shape for compatibility with Cartesian interface
        self.grid_shape = (nr, 1, nz)

        logger.info(
            "CylindricalMHDSolver initialized: (nr=%d, nz=%d), dr=%.2e, dz=%.2e, "
            "gamma=%.3f, Hall=%s",
            nr, nz, dr, dz, gamma, enable_hall,
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
            ne = rho / m_p
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
    ) -> dict[str, np.ndarray]:
        """Compute RHS of the MHD equations in cylindrical coordinates.

        All arrays are 2D: scalars (nr, nz), vectors (3, nr, nz).

        Returns time derivatives for all state variables.
        """
        geom = self.geom

        # --- Current density: J = curl(B) / mu_0 ---
        curl_B = geom.curl(B)
        J = curl_B / mu_0

        # --- Density: d(rho)/dt = -div(rho*v) ---
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

        # Hall term: E_Hall = (J × B) / (ne * e)
        if self.enable_hall:
            ne = rho / m_p
            ne_safe = np.maximum(ne, 1e-20)
            E_Hall = np.zeros((3, self.nr, self.nz))
            E_Hall[0] = (J[1] * B[2] - J[2] * B[1]) / (ne_safe * e_charge)
            E_Hall[1] = (J[2] * B[0] - J[0] * B[2]) / (ne_safe * e_charge)
            E_Hall[2] = (J[0] * B[1] - J[1] * B[0]) / (ne_safe * e_charge)
            E_field = E_field + E_Hall

        dB_dt = -geom.curl(E_field)

        # --- Pressure: dp/dt = -gamma * p * div(v) ---
        div_v = geom.divergence(vel)
        dp_dt = -self.gamma * p * div_v

        # --- Dedner cleaning ---
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
        }

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
    ) -> dict[str, np.ndarray]:
        """Advance MHD state by one timestep using SSP-RK2.

        Same interface as Cartesian MHDSolver.step().
        Internally squeezes to 2D, computes, then unsqueezes to 3D.

        Args:
            state: Dictionary with 3D arrays (nr, 1, nz).
            dt: Timestep [s].
            current: Circuit current [A].
            voltage: Circuit voltage [V].

        Returns:
            Updated state dictionary with 3D arrays.
        """
        # Squeeze to 2D
        rho = self._squeeze(state["rho"])
        vel = self._squeeze(state["velocity"])
        p = self._squeeze(state["pressure"])
        B = self._squeeze(state["B"])
        psi = self._squeeze(state.get("psi", np.zeros((self.nr, 1, self.nz))))

        # Save U^n
        rho_n = rho.copy()
        vel_n = vel.copy()
        p_n = p.copy()
        B_n = B.copy()
        psi_n = psi.copy()
        mom_n = rho_n[np.newaxis, :, :] * vel_n

        # === Stage 1: U^(1) = U^n + dt * L(U^n) ===
        rhs1 = self._compute_rhs(rho_n, vel_n, p_n, B_n, psi_n)

        rho_1 = np.maximum(rho_n + dt * rhs1["drho_dt"], 1e-20)
        mom_1 = mom_n + dt * rhs1["dmom_dt"]
        vel_1 = mom_1 / np.maximum(rho_1[np.newaxis, :, :], 1e-30)
        p_1 = np.maximum(p_n + dt * rhs1["dp_dt"], 1e-20)
        B_1 = B_n + dt * rhs1["dB_dt"]
        psi_1 = psi_n + dt * rhs1["dpsi_dt"]

        # === Stage 2: U^(n+1) = 0.5*U^n + 0.5*(U^(1) + dt*L(U^(1))) ===
        rhs2 = self._compute_rhs(rho_1, vel_1, p_1, B_1, psi_1)

        rho_new = np.maximum(0.5 * rho_n + 0.5 * (rho_1 + dt * rhs2["drho_dt"]), 1e-20)
        mom_new = 0.5 * mom_n + 0.5 * (mom_1 + dt * rhs2["dmom_dt"])
        vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :], 1e-30)
        p_new = np.maximum(0.5 * p_n + 0.5 * (p_1 + dt * rhs2["dp_dt"]), 1e-20)
        B_new = 0.5 * B_n + 0.5 * (B_1 + dt * rhs2["dB_dt"])
        psi_new = 0.5 * psi_n + 0.5 * (psi_1 + dt * rhs2["dpsi_dt"])

        # --- Update temperatures from pressure ---
        n_i = rho_new / m_p
        Ti_new = p_new / (2.0 * np.maximum(n_i, 1e-30) * k_B)
        Te_new = Ti_new.copy()

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
