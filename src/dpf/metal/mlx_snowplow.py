"""
Lee-model snowplow for Dense Plasma Focus.

Implements the 5-phase Lee model equations from:
  Lee, S. (2014), J. Fusion Energy 33:319-335.
  RADPF theory document: plasmafocus.net/IPFS/modelpackage/File2Theory.pdf

Phases: rundown (axial) -> radial inward shock (slug model) -> pinch.

Axial phase: RADPF Eqs. (I)-(II) — snowplow with momentum correction.
Radial phase: RADPF Eqs. (III)-(VI) — 4-ODE slug model with separate
  shock front (r_s), piston (r_p), column elongation (z_f), and circuit (I).

All math is scalar Python / math module only. No numpy, no mlx.core.
"""

from __future__ import annotations

import math

_MU0 = 4.0 * math.pi * 1e-7
_TWO_PI = 2.0 * math.pi
_FOUR_PI = 4.0 * math.pi
_GAMMA = 5.0 / 3.0


class MLXSnowplow:
    """Lee-model snowplow for DPF axial and radial phases.

    Parameters
    ----------
    anode_radius : float
        Inner (anode) electrode radius [m].
    cathode_radius : float
        Outer (cathode) electrode radius [m].
    fill_density : float
        Pre-fill gas mass density [kg/m^3].
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
    gamma : float
        Specific heat ratio (default 5/3 for atomic gas).
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
        gamma: float = _GAMMA,
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
        self._gamma = float(gamma)

        self._c: float = self._b / self._a  # cathode/anode ratio
        self._ln_c: float = math.log(self._c)
        self._A_ann: float = math.pi * (self._b**2 - self._a**2)

        # Start 1% into anode to avoid zero swept mass at z=0
        self._z: float = max(self._L_anode * 0.01, 1e-4)
        self._vz: float = 0.0

        # Radial state (populated at rundown->radial transition)
        self._z_f: float = self._L_anode
        self._r_s: float = self._a  # shock front (starts at anode radius)
        self._r_p: float = self._a  # piston (starts at anode radius)
        self._vr_p: float = 0.0     # piston radial velocity (for Eq. V lag)

        # Inductance — note: fc multiplies Lp per Lee 2014 Eq. (II)
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

    @property
    def plasma_inductance(self) -> float:
        return self._L_plasma

    # ------------------------------------------------------------------
    # Inductance formulas — Lee 2014, Eqs. (10), (18)
    # fc multiplies Lp per Eq. (II): circuit sees fc * L_tube
    # ------------------------------------------------------------------

    def _axial_inductance(self, z: float) -> float:
        """Lee 2014, Eq. (10): L = (mu_0/2pi)*ln(b/a)*z, times fc."""
        return self._fc * (_MU0 / _TWO_PI) * self._ln_c * z

    def _radial_inductance(self, r_p: float, z_f: float) -> float:
        """Lee 2014, Eq. (18): L_rad = (mu_0/2pi)*ln(b/r_p)*z_f, times fc."""
        if r_p <= 0.0 or r_p >= self._b:
            return 0.0
        return self._fc * (_MU0 / _TWO_PI) * math.log(self._b / r_p) * z_f

    # ------------------------------------------------------------------
    # Axial phase — RADPF Eqs. (I) and (II)
    # ------------------------------------------------------------------

    def _step_rundown(self, dt: float, I: float) -> None:
        """Advance axial phase by dt.

        Lee 2014, Eq. (I):
          d2z/dt2 = [ (fc^2/fm) * (mu*ln(c)) / (4*pi^2*rho0*(c^2-1)) * (I/a)^2
                      - (dz/dt)^2 ] / z

        Integrated with velocity Verlet (kick-drift-kick).
        """
        z0, v0 = self._z, self._vz

        # Eq. (I): acceleration with momentum correction -(dz/dt)^2/z
        coeff = (self._fc**2 / self._fm) * _MU0 * self._ln_c / (
            4.0 * math.pi**2 * self._rho0 * (self._c**2 - 1.0)
        )
        z_safe = max(z0, 1e-6)
        a0 = (coeff * (I / self._a) ** 2 - v0**2) / z_safe

        # Back-pressure correction (not in base Lee model, standard extension)
        a0 -= self._p0 * self._A_ann / self._axial_mass(z0)

        # Verlet integration
        v_half = v0 + 0.5 * dt * a0
        z1 = max(z0 + dt * v_half, 1e-6)
        a1 = (coeff * (I / self._a) ** 2 - v_half**2) / max(z1, 1e-6)
        a1 -= self._p0 * self._A_ann / self._axial_mass(z1)
        v1 = v_half + 0.5 * dt * a1

        # Inductance: fc * (mu_0/2pi) * ln(c) * z
        L1 = self._axial_inductance(z1)
        self._dL_dt = (L1 - self._L_plasma) / dt if dt > 0.0 else 0.0
        self._z = z1
        self._vz = v1
        self._L_plasma = L1
        self._L_axial = L1

        # Phase transition: z >= anode length
        if z1 >= self._L_anode:
            self._z = self._L_anode
            # Lee 2014 p.11: zeta_f = 0.00001 (normalized to a), so z_f = a * 1e-5
            self._z_f = self._a * 1e-5
            self._r_s = self._a  # shock starts at anode radius
            self._r_p = self._a  # piston starts at anode radius
            self._vr_p = 0.0
            self._L_axial = self._axial_inductance(self._L_anode)
            self._phase = "radial"

    def _axial_mass(self, z: float) -> float:
        return max(self._fm * self._rho0 * self._A_ann * z, 1e-20)

    # ------------------------------------------------------------------
    # Radial phase — RADPF Eqs. (III)-(VI), 4-ODE slug model
    # ------------------------------------------------------------------

    def _step_radial(
        self, dt: float, I: float,
        V_cap: float = 0.0, R0: float = 0.0, L0: float = 0.0,
    ) -> None:
        """Advance radial slug model by dt.

        4 coupled ODEs from Lee 2014:
          Eq. (III): dr_s/dt  — shock front speed
          Eq. (IV):  dz_f/dt  — column elongation
          Eq. (V):   dr_p/dt  — piston speed (complex, coupled to dI/dt)
          Eq. (VI):  dI/dt    — circuit equation (uses previous dr_p/dt)

        The coupling between Eqs. (V) and (VI) is handled by using the
        previous timestep's dr_p/dt in Eq. (VI), per RADPF convention.
        """
        gamma = self._gamma
        r_s = self._r_s
        r_p = self._r_p
        z_f = self._z_f
        dr_p_dt_prev = self._vr_p  # from previous step

        # Safety floors
        r_p_safe = max(r_p, self._a * 1e-4)
        r_s_safe = max(r_s, self._a * 1e-4)
        z_f_safe = max(z_f, self._a * 1e-5)
        I_safe = max(abs(I), 1.0)

        # Eq. (III): shock front speed
        # dr_s/dt = -sqrt(mu_0*(gamma+1)/rho_0) * (fc/sqrt(fmr)) * I / (4*pi*r_p)
        dr_s_dt = -(
            math.sqrt(_MU0 * (gamma + 1.0) / self._rho0)
            * (self._fc / math.sqrt(self._fmr))
            * I_safe / (_FOUR_PI * r_p_safe)
        )

        # Eq. (IV): column elongation
        # dz_f/dt = -(2/(gamma+1)) * dr_s/dt
        dz_f_dt = -(2.0 / (gamma + 1.0)) * dr_s_dt

        # Eq. (VI): circuit equation during radial phase
        # Uses previous dr_p/dt for the back-EMF term (RADPF convention)
        L_ax = self._L_axial
        L_rad = self._radial_inductance(r_p_safe, z_f_safe)
        L_total = max(L_ax + L_rad, 1e-12)

        # Eq. (VI): Full circuit equation during radial phase (Lee 2014)
        #   dI/dt = [V_cap - r_0*I
        #            - fc*(mu/2pi)*ln(b/r_p)*I*dz_f/dt
        #            + fc*(mu/2pi)*(z_f/r_p)*I*dr_p/dt
        #           ] / [L_0 + fc*(mu/2pi)*ln(c)*z_0 + fc*(mu/2pi)*ln(b/r_p)*z_f]
        back_emf_zf = self._fc * (_MU0 / _TWO_PI) * math.log(self._b / r_p_safe) * I_safe * dz_f_dt
        back_emf_rp = self._fc * (_MU0 / _TWO_PI) * (z_f_safe / r_p_safe) * I_safe * dr_p_dt_prev
        L_total_circuit = max(L0 + L_total, 1e-12)
        dI_dt_full = (V_cap - R0 * I_safe - back_emf_zf + back_emf_rp) / L_total_circuit

        # Eq. (V): piston speed (full Lee 2014 form)
        ratio_sq = (r_s_safe / r_p_safe) ** 2
        one_minus_ratio = max(1.0 - ratio_sq, 1e-10)

        term1 = (2.0 / (gamma + 1.0)) * (r_s_safe / r_p_safe) * dr_s_dt
        term2 = -(r_p_safe / (gamma * I_safe)) * one_minus_ratio * dI_dt_full
        term3 = -(1.0 / (gamma + 1.0)) * (r_p_safe / z_f_safe) * one_minus_ratio * dz_f_dt

        denom_v = (gamma - 1.0) / gamma + (1.0 / gamma) * ratio_sq
        denom_v = max(denom_v, 1e-10)

        dr_p_dt = (term1 + term2 + term3) / denom_v

        # Clamp velocities for stability
        v_max = 1e7
        dr_s_dt = max(min(dr_s_dt, v_max), -v_max)
        dr_p_dt = max(min(dr_p_dt, v_max), -v_max)
        dz_f_dt = max(min(dz_f_dt, v_max), -v_max)

        # Forward Euler update (simple, matches RADPF sequential approach)
        r_s_new = max(r_s + dt * dr_s_dt, 0.0)
        r_p_new = max(r_p + dt * dr_p_dt, self._a * 1e-4)
        z_f_new = max(z_f + dt * dz_f_dt, self._a * 1e-5)

        # Piston cannot be outside shock front
        r_p_new = max(r_p_new, r_s_new)

        # Update inductance — uses r_p (piston), not r_s (shock)
        L_total_new = self._L_axial + self._radial_inductance(r_p_new, z_f_new)
        if abs(L_total_new) > 1.0:
            self._terminate()
            return

        self._dL_dt = (L_total_new - self._L_plasma) / dt if dt > 0.0 else 0.0
        self._r_s = r_s_new
        self._r_p = r_p_new
        self._z_f = z_f_new
        self._vr_p = dr_p_dt
        self._L_plasma = L_total_new

        # Terminate when shock reaches axis
        if r_s_new <= self._a * 0.01:
            self._terminate()

    def _terminate(self) -> None:
        self._phase = "pinch"
        self._active = False
        self._dL_dt = 0.0

    def step(
        self,
        dt: float,
        current: float,
        voltage: float = 0.0,
        R0: float = 0.0,
        L0: float = 0.0,
    ) -> dict[str, float]:
        """Advance snowplow by dt [s] with instantaneous current [A].

        Parameters
        ----------
        dt : float
            Timestep [s].
        current : float
            Circuit current [A].
        voltage : float
            Capacitor voltage [V] (needed for radial phase Eq. VI).
        R0 : float
            Circuit stray resistance [Ohm] (needed for radial phase Eq. VI).
        L0 : float
            Circuit stray inductance [H] (needed for radial phase Eq. VI).

        Returns dict with keys:
          L_plasma, dL_dt, R_plasma, z_sheath, r_shock, r_piston, z_focus, phase
        """
        if self._active:
            if self._phase == "rundown":
                self._step_rundown(dt, current)
            elif self._phase == "radial":
                self._step_radial(dt, current, voltage, R0, L0)

        return {
            "L_plasma": self._L_plasma,
            "dL_dt": self._dL_dt,
            "R_plasma": 0.0,
            "z_sheath": self._z,
            "r_shock": self._r_s,
            "r_piston": self._r_p,
            "z_focus": self._z_f,
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
