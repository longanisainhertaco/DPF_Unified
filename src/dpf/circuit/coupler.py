"""Circuit-MHD coupling via density-weighted plasma inductance.

Abstracts the Lp computation from MHD state fields so that ANY backend
producing spatially-resolved density/B-fields can feed back into the
lumped circuit model.  This replaces the B-energy volume integral
(which suffers from electrode BC artifacts on coarse grids) with the
density-weighted radius method from Lee model theory.

Algorithm:
    1. Find sheath position z_sheath from density peak (argmax along z)
    2. Compute density-weighted radius:
       r_eff = integral(r * rho * dV) / integral(rho * dV)
    3. Compute Lp = (mu_0 / 2*pi) * z_sheath * ln(cathode_radius / r_eff)
    4. Enforce Lp monotonicity (Lp can only increase during compression)
    5. Compute dLp/dt via BDF2 finite difference
    6. Compute back-EMF = I * dLp/dt, clamped to +/-50 kV

References:
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from dpf.constants import mu_0, pi

logger = logging.getLogger(__name__)

BACK_EMF_CLAMP_V = 50_000.0  # +/- 50 kV clamp


@dataclass
class FeedbackResult:
    """Result of a single coupling computation."""

    Lp: float = 0.0           # Plasma inductance [H]
    dLp_dt: float = 0.0       # Rate of change [H/s]
    back_emf: float = 0.0     # Motional + inductive EMF [V]
    r_eff: float = 0.0        # Density-weighted effective radius [m]
    z_sheath: float = 0.0     # Sheath axial position [m]


class CircuitCoupler:
    """Computes plasma inductance from MHD state for circuit feedback.

    Extracts Lp from the MHD density field using the density-weighted
    radius method.  This couples the distributed MHD solution to the
    lumped circuit model, replacing the volume-integral B-energy method
    which is sensitive to electrode BC artifacts.

    Parameters
    ----------
    anode_radius : float
        Inner electrode radius [m].
    cathode_radius : float
        Outer electrode radius [m].
    dr : float
        Radial grid spacing [m].
    dz : float
        Axial grid spacing [m].
    r_inner : float or None
        Inner radial offset of the grid [m].  For cylindrical grids that
        start at r = anode_radius, pass anode_radius.  None means r starts
        at 0.
    """

    def __init__(
        self,
        anode_radius: float,
        cathode_radius: float,
        dr: float,
        dz: float,
        r_inner: float | None = None,
    ) -> None:
        self.anode_radius = anode_radius
        self.cathode_radius = cathode_radius
        self.dr = dr
        self.dz = dz
        self.r_inner = r_inner if r_inner is not None else 0.0

        # Monotonicity enforcement: Lp can only increase during compression
        self._Lp_max: float = 0.0

        # BDF2 history for dLp/dt: stores (time, Lp) tuples
        self._history: deque[tuple[float, float]] = deque(maxlen=3)

        # Track time for dLp/dt computation
        self._time: float = 0.0

        logger.debug(
            "CircuitCoupler: a=%.4f m, b=%.4f m, dr=%.2e, dz=%.2e, r_inner=%.4f m",
            anode_radius, cathode_radius, dr, dz, self.r_inner,
        )

    def reset(self) -> None:
        """Reset internal state (for checkpoint/restart)."""
        self._Lp_max = 0.0
        self._history.clear()
        self._time = 0.0

    def compute_feedback(
        self,
        state: dict,
        current: float,
        dt: float,
    ) -> FeedbackResult:
        """Compute Lp, dLp/dt, and back-EMF from MHD state.

        Parameters
        ----------
        state : dict
            MHD state dict with at least ``rho`` and ``B`` arrays.
            For cylindrical: rho is (nr, 1, nz) or (nr, nz).
        current : float
            Circuit current [A].
        dt : float
            Timestep [s] (for dLp/dt finite difference).

        Returns
        -------
        FeedbackResult
            Coupling quantities for the circuit solver.
        """
        rho = state.get("rho")
        if rho is None:
            return FeedbackResult()

        # Squeeze out the ny=1 dimension for cylindrical
        if rho.ndim == 3 and rho.shape[1] == 1:
            rho_2d = rho[:, 0, :]  # (nr, nz)
        elif rho.ndim == 2:
            rho_2d = rho
        else:
            # 3D Cartesian: fall back to B-energy method (coupler not applicable)
            return self._fallback_b_energy(state, current, dt)

        nr, nz = rho_2d.shape

        # --- Step 1: Find sheath position from density peak ---
        # Integrate density along r at each z, find z with max column density
        # This is robust against noisy individual cells
        r_arr = self._r_array(nr)
        # Column density: integral(rho * 2*pi*r * dr) at each z
        col_density = np.sum(rho_2d * r_arr[:, np.newaxis], axis=0) * self.dr
        iz_sheath = int(np.argmax(col_density))
        z_sheath = (iz_sheath + 0.5) * self.dz

        # --- Step 2: Density-weighted effective radius ---
        # Use cells up to and including sheath position (the compressed region)
        # r_eff = integral(r * rho * 2*pi*r * dr * dz) / integral(rho * 2*pi*r * dr * dz)
        # over z in [0, z_sheath]
        rho_region = rho_2d[:, :iz_sheath + 1]
        r_col = r_arr[:, np.newaxis]  # (nr, 1) for broadcasting

        dV = 2.0 * pi * r_col * self.dr * self.dz  # cylindrical volume element
        mass = rho_region * dV
        total_mass = np.sum(mass)

        if total_mass > 0:
            r_eff = float(np.sum(r_col * mass) / total_mass)
        else:
            r_eff = 0.5 * (self.anode_radius + self.cathode_radius)

        # Clamp r_eff to physical range: must be between axis and cathode
        r_eff = max(r_eff, 1e-6)
        r_eff = min(r_eff, self.cathode_radius * 0.999)

        # --- Step 3: Compute Lp from Lee formula ---
        # Lp = (mu_0 / 2*pi) * z_sheath * ln(b / r_eff)
        if r_eff > 0 and z_sheath > 0:
            Lp = (mu_0 / (2.0 * pi)) * z_sheath * np.log(self.cathode_radius / r_eff)
        else:
            Lp = 0.0

        # --- Step 4: Enforce monotonicity ---
        # During compression, Lp should only increase (z advances, r_eff decreases)
        # Noisy z_sheath can cause oscillating Lp -> oscillating back-EMF
        if Lp > self._Lp_max:
            self._Lp_max = Lp
        else:
            Lp = self._Lp_max

        # --- Step 5: Compute dLp/dt via BDF2 ---
        self._time += dt
        dLp_dt = self._compute_dLp_dt(Lp)
        self._history.append((self._time, Lp))

        # --- Step 6: Back-EMF = I * dLp/dt, clamped ---
        back_emf = current * dLp_dt
        back_emf = float(np.clip(back_emf, -BACK_EMF_CLAMP_V, BACK_EMF_CLAMP_V))

        return FeedbackResult(
            Lp=Lp,
            dLp_dt=dLp_dt,
            back_emf=back_emf,
            r_eff=r_eff,
            z_sheath=z_sheath,
        )

    def _r_array(self, nr: int) -> np.ndarray:
        """Build radial coordinate array accounting for r_inner offset."""
        return self.r_inner + (np.arange(nr) + 0.5) * self.dr

    def _compute_dLp_dt(self, Lp: float) -> float:
        """BDF2 finite difference for dLp/dt.

        Uses 2nd-order backward difference when >= 2 history points
        are available, otherwise 1st-order backward difference.
        """
        n = len(self._history)

        if n == 0:
            return 0.0

        if n >= 2:
            t_nm2, Lp_nm2 = self._history[-2]
            t_nm1, Lp_nm1 = self._history[-1]
            t_now = self._time
            dt1 = t_nm1 - t_nm2
            dt2 = t_now - t_nm1
            if dt1 > 0 and dt2 > 0:
                if abs(dt1 - dt2) < 1e-8 * max(dt1, dt2, 1e-30):
                    return (3.0 * Lp - 4.0 * Lp_nm1 + Lp_nm2) / (2.0 * dt2)
                else:
                    r = dt2 / dt1
                    return (
                        (1.0 + 2.0 * r) / (1.0 + r) * Lp
                        - (1.0 + r) * Lp_nm1
                        + r**2 / (1.0 + r) * Lp_nm2
                    ) / dt2

        t_nm1, Lp_nm1 = self._history[-1]
        dt_back = self._time - t_nm1
        if dt_back > 0:
            return (Lp - Lp_nm1) / dt_back

        return 0.0

    def _fallback_b_energy(
        self, state: dict, current: float, dt: float,
    ) -> FeedbackResult:
        """Fallback: compute Lp from B-field magnetic energy (3D Cartesian).

        L = 2 * integral(B^2 / (2*mu_0) * dV) / I^2
        """
        B = state.get("B")
        if B is None or abs(current) < 1e-3:
            return FeedbackResult()

        B_sq = np.sum(B**2, axis=0)
        dV = self.dr**3  # Cartesian: dx = dy = dz = dr (approximation)
        W_B = float(np.sum(B_sq / (2.0 * mu_0) * dV))
        I_sq = max(current**2, 1e-30)
        Lp = 2.0 * W_B / I_sq

        if Lp > self._Lp_max:
            self._Lp_max = Lp
        else:
            Lp = self._Lp_max

        self._time += dt
        dLp_dt = self._compute_dLp_dt(Lp)
        self._history.append((self._time, Lp))

        back_emf = current * dLp_dt
        back_emf = float(np.clip(back_emf, -BACK_EMF_CLAMP_V, BACK_EMF_CLAMP_V))

        return FeedbackResult(Lp=Lp, dLp_dt=dLp_dt, back_emf=back_emf)
