"""
Pure MLX Lee-model snowplow for Dense Plasma Focus.

Phases: rundown (axial, Phase 2) → radial (Phase 3) → pinch (Phase 4).
All math is scalar Python / math module only. No numpy, no mlx.core.
"""

from __future__ import annotations

import math

_MU0 = 4.0 * math.pi * 1e-7
_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi


class MLXSnowplow:
    """Lee-model snowplow for DPF axial and radial phases.

    Parameters
    ----------
    anode_radius : float
        Inner (anode) electrode radius [m].
    cathode_radius : float
        Outer (cathode) electrode radius [m].
    fill_density : float
        Pre-fill gas mass density [kg/m³].
    anode_length : float
        Axial length of the anode [m].
    mass_fraction : float
        Axial snowplow mass fraction fm (default 0.15).
    current_fraction : float
        Current fraction fc carried by sheath (default 0.7).
    fill_pressure_Pa : float
        Pre-fill gas pressure [Pa] for back-pressure correction.
    radial_mass_fraction : float | None
        Radial compression mass fraction fmr. Defaults to mass_fraction.
    pinch_column_fraction : float
        Fraction of anode length participating in pinch column (default 1.0).
    """

    def __init__(
        self,
        anode_radius: float,
        cathode_radius: float,
        fill_density: float,
        anode_length: float,
        mass_fraction: float = 0.15,
        current_fraction: float = 0.7,
        fill_pressure_Pa: float = 400.0,
        radial_mass_fraction: float | None = None,
        pinch_column_fraction: float = 1.0,
    ) -> None:
        self._a = float(anode_radius)
        self._b = float(cathode_radius)
        self._rho0 = float(fill_density)
        self._L_anode = float(anode_length)
        self._fm = float(mass_fraction)
        self._fc = float(current_fraction)
        self._p0 = float(fill_pressure_Pa)
        self._fmr = float(radial_mass_fraction if radial_mass_fraction is not None else mass_fraction)
        self._pcf = float(pinch_column_fraction)

        self._ln_ba: float = math.log(self._b / self._a)
        self._A_ann: float = math.pi * (self._b**2 - self._a**2)

        # Start 1% into anode — avoids near-zero swept mass at z=0
        self._z: float = max(self._L_anode * 0.01, 1e-4)
        self._vz: float = 0.0

        # Radial state (populated at rundown→radial transition)
        self._z_f: float = self._L_anode
        self._r_s: float = self._b
        self._vr: float = 0.0

        # Inductance
        self._L_axial: float = 0.0
        self._L_plasma: float = 0.0
        self._dL_dt: float = 0.0

        self._phase: str = "rundown"
        self._active: bool = True

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._active

    def _axial_inductance(self, z: float) -> float:
        return (_MU0 / _TWO_PI) * self._ln_ba * z

    def _radial_inductance(self, r_s: float) -> float:
        if r_s <= 0.0 or r_s >= self._b:
            return 0.0
        return (_MU0 / _TWO_PI) * self._z_f * math.log(self._b / r_s)

    def _axial_force(self, I: float) -> float:
        """Net axial force on sheath [N]."""
        F_mag = (_MU0 / _FOUR_PI) * self._ln_ba * (self._fc * I) ** 2
        return F_mag - self._p0 * self._A_ann

    def _axial_mass(self, z: float) -> float:
        return max(self._fm * self._rho0 * self._A_ann * z, 1e-20)

    def _radial_force(self, I: float, r_s: float) -> float:
        """Inward radial magnetic force on slug [N]."""
        if r_s <= 0.0:
            return 0.0
        return (_MU0 / _FOUR_PI) * (self._fc * I) ** 2 * self._z_f / r_s

    def _radial_mass(self, r_s: float) -> float:
        """Slug mass for radial compression [kg].

        m_total - m_compressed so that mass decreases as r_s → 0.
        Floor at 1% of total to avoid divide-by-zero when r_s ≈ b.
        """
        r_s = max(r_s, 0.0)
        m_total = self._fmr * self._rho0 * math.pi * self._b**2 * self._z_f
        m_compressed = self._fmr * self._rho0 * math.pi * r_s**2 * self._z_f
        return max(m_total - m_compressed, max(m_total * 0.01, 1e-20))

    def _step_rundown(self, dt: float, I: float) -> None:
        z0, v0 = self._z, self._vz
        F0 = self._axial_force(I)
        a0 = F0 / self._axial_mass(z0)

        v_half = v0 + 0.5 * dt * a0
        z1 = max(z0 + dt * v_half, 0.0)

        a1 = self._axial_force(I) / self._axial_mass(z1)
        v1 = v_half + 0.5 * dt * a1

        L1 = self._axial_inductance(z1)
        self._dL_dt = (L1 - self._L_plasma) / dt if dt > 0.0 else 0.0
        self._z = z1
        self._vz = v1
        self._L_plasma = L1
        self._L_axial = L1

        if z1 >= self._L_anode:
            self._z = self._L_anode
            self._z_f = self._L_anode * self._pcf
            self._r_s = self._b
            self._vr = 0.0
            self._phase = "radial"

    def _step_radial(self, dt: float, I: float) -> None:
        r0, vr0 = self._r_s, self._vr
        a0 = -self._radial_force(I, r0) / self._radial_mass(r0)

        vr_half = vr0 + 0.5 * dt * a0
        r1 = max(r0 + dt * vr_half, 1e-6)

        a1 = -self._radial_force(I, r1) / self._radial_mass(r1)
        vr1 = vr_half + 0.5 * dt * a1

        if abs(vr1) > 1e8:
            self._terminate()
            return

        L_total = self._L_axial + self._radial_inductance(r1)
        if abs(L_total) > 1.0:
            self._terminate()
            return

        self._dL_dt = (L_total - self._L_plasma) / dt if dt > 0.0 else 0.0
        self._r_s = r1
        self._vr = vr1
        self._L_plasma = L_total

        if r1 <= 0.1 * self._a:
            self._terminate()

    def _terminate(self) -> None:
        self._phase = "pinch"
        self._active = False
        self._dL_dt = 0.0

    def step(self, dt: float, current: float) -> dict[str, float]:
        """Advance snowplow by dt [s] with instantaneous current [A].

        Returns dict with keys:
          L_plasma, dL_dt, R_plasma, z_sheath, r_shock, phase
        """
        if self._active:
            if self._phase == "rundown":
                self._step_rundown(dt, current)
            elif self._phase == "radial":
                self._step_radial(dt, current)

        return {
            "L_plasma": self._L_plasma,
            "dL_dt": self._dL_dt,
            "R_plasma": 0.0,
            "z_sheath": self._z,
            "r_shock": self._r_s,
            "phase": self._phase,
        }

    def get_Lp(self) -> float:
        return self._L_plasma

    def get_dLp_dt(self) -> float:
        return self._dL_dt

    def get_R_plasma(self) -> float:
        return 0.0

    def get_phase(self) -> str:
        return self._phase
