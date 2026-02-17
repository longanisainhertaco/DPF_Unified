"""Snowplow dynamics for Dense Plasma Focus axial and radial phases.

Implements the Lee model Phases 2-4 where the current sheath sweeps fill gas
along the coaxial electrode gap (axial rundown), then implodes radially
(radial compression), producing the characteristic DPF current dip and pinch.

**Phase 2 — Axial Rundown**:
    d/dt[m(z) * v] = F_mag - F_pressure

where:
    F_mag = (mu_0 / 4pi) * ln(b/a) * (f_c * I)^2  [N]  magnetic driving force
    F_pressure = p * pi * (b^2 - a^2)              [N]  fill gas back-pressure
    m(z) = rho_0 * pi * (b^2 - a^2) * z * f_m      [kg] swept mass
    L_plasma = (mu_0 / 2pi) * ln(b/a) * z          [H]  plasma inductance

**Phase 3 — Radial Inward Shock** (slug model):
    d/dt[M_slug * vr] = -F_rad

where:
    F_rad = (mu_0 / 4pi) * (f_c * I)^2 * z_f / r_s   [N]  radial J×B force
    M_slug = f_mr * rho_0 * pi * (b^2 - r_s^2) * z_f  [kg] radial swept mass
    L_plasma = L_axial + (mu_0 / 2pi) * z_f * ln(b / r_s)  [H]

The snowplow provides the time-varying plasma inductance L_plasma(t) that
couples to the circuit solver via CouplingState, producing the characteristic
DPF current dip signature.

References:
    Lee, S. & Saw, S.H., Phys. Plasmas 21, 072501 (2014).
    Lee, S., J. Fusion Energy 33, 319-335 (2014).
    Haines, M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
"""

from __future__ import annotations

import logging

import numpy as np

from dpf.constants import mu_0, pi

logger = logging.getLogger(__name__)


class SnowplowModel:
    """0D snowplow model for DPF axial rundown and radial compression.

    Implements Lee model Phases 2-4:
    - Phase 2: Axial rundown (snowplow) — sheath sweeps fill gas along anode
    - Phase 3: Radial inward shock — cylindrical implosion toward axis
    - Phase 4: Pinch (frozen state after shock reaches minimum radius)

    Provides plasma inductance for circuit coupling throughout all phases.

    Args:
        anode_radius: Inner electrode (anode) radius [m].
        cathode_radius: Outer electrode (cathode) radius [m].
        fill_density: Fill gas mass density rho_0 [kg/m^3].
        anode_length: Anode length (rundown distance) [m].
        mass_fraction: Fraction of gas swept by sheath axially (0 < f_m <= 1).
            Typical values: 0.1-0.5 for DPF (Lee & Saw 2014).
        fill_pressure_Pa: Fill gas pressure [Pa] for back-pressure force.
        current_fraction: Fraction of total circuit current flowing in sheath
            (f_c, Lee & Saw 2014). Typical values: 0.6-0.8 for DPF.
        radial_mass_fraction: Fraction of gas swept radially (f_mr).
            Defaults to mass_fraction if not specified.
            Typical values: 0.1-0.3 for DPF (usually < f_m).
    """

    def __init__(
        self,
        anode_radius: float,
        cathode_radius: float,
        fill_density: float,
        anode_length: float,
        mass_fraction: float = 0.3,
        fill_pressure_Pa: float = 400.0,
        current_fraction: float = 0.7,
        radial_mass_fraction: float | None = None,
    ) -> None:
        self.a = anode_radius           # [m]
        self.b = cathode_radius         # [m]
        self.rho0 = fill_density        # [kg/m^3]
        self.L_anode = anode_length     # [m]
        self.f_m = mass_fraction        # dimensionless (axial)
        self.p_fill = fill_pressure_Pa  # [Pa]
        self.f_c = current_fraction     # dimensionless (Lee & Saw 2014: ~0.7)
        self.f_mr = radial_mass_fraction if radial_mass_fraction is not None else mass_fraction

        # Derived geometric constants
        self.ln_ba = np.log(self.b / self.a)             # ln(b/a)
        self.A_annular = pi * (self.b**2 - self.a**2)    # annular area [m^2]

        # Magnetic force coefficient: (mu_0 / 4pi) * ln(b/a) [N/A^2]
        # Phase R.5 confirmed: mu_0/(4pi), NOT mu_0/2
        self.F_coeff = (mu_0 / (4.0 * pi)) * self.ln_ba

        # Inductance coefficient: (mu_0 / 2pi) * ln(b/a) [H/m]
        # From coaxial magnetic energy: L/z = mu_0/(2*pi) * ln(b/a)
        # While F/I^2 = mu_0/(4*pi) * ln(b/a), so L_coeff = 2 * F_coeff
        self.L_coeff = 2.0 * self.F_coeff

        # Initialize axial state
        self.z = 1e-4   # Initial sheath position [m] (small offset to avoid m=0)
        self.v = 0.0    # Initial sheath velocity [m/s]
        self.phase = "rundown"

        # Radial phase state (initialized when rundown completes)
        self.r_shock = self.b     # Shock front radius [m] (starts at cathode)
        self.vr = 0.0             # Radial shock velocity [m/s] (negative = inward)
        self._L_axial_frozen = 0.0  # Axial L_plasma at end of rundown [H]

        # Minimum pinch radius: 10% of anode radius (Lee standard)
        self.r_pinch_min = 0.1 * self.a

        # Track phase completion
        self._rundown_complete = False
        self._pinch_complete = False
        self._pinch_time = 0.0
        self._elapsed_time = 0.0

        logger.info(
            "SnowplowModel: a=%.1f mm, b=%.1f mm, L_anode=%.0f mm, "
            "f_m=%.2f, f_c=%.2f, f_mr=%.2f, rho0=%.2e kg/m^3, p_fill=%.0f Pa",
            self.a * 1e3, self.b * 1e3, self.L_anode * 1e3,
            self.f_m, self.f_c, self.f_mr, self.rho0, self.p_fill,
        )

    @property
    def sheath_position(self) -> float:
        """Current sheath axial position [m]."""
        return self.z

    @property
    def sheath_velocity(self) -> float:
        """Current sheath axial velocity [m/s]."""
        return self.v

    @property
    def shock_radius(self) -> float:
        """Current radial shock front radius [m]."""
        return self.r_shock

    @property
    def swept_mass(self) -> float:
        """Mass swept by the sheath [kg].

        Axial: m = rho_0 * pi * (b^2 - a^2) * z * f_m
        """
        return self.rho0 * self.A_annular * self.z * self.f_m

    @property
    def radial_swept_mass(self) -> float:
        """Mass swept by radial shock [kg].

        M_slug = f_mr * rho_0 * pi * (b^2 - r_s^2) * z_f
        """
        return self.f_mr * self.rho0 * pi * (self.b**2 - self.r_shock**2) * self.L_anode

    @property
    def plasma_inductance(self) -> float:
        """Total plasma inductance [H].

        Axial: L_plasma = (mu_0 / 2pi) * ln(b/a) * z
        Radial: L_plasma = L_axial + (mu_0 / 2pi) * z_f * ln(b / r_s)
        """
        if self.phase == "rundown":
            return self.L_coeff * self.z
        # Radial or pinch: axial contribution frozen + radial contribution
        r_eff = max(self.r_shock, self.r_pinch_min)
        L_radial = (mu_0 / (2.0 * pi)) * self.L_anode * np.log(self.b / r_eff)
        return self._L_axial_frozen + L_radial

    @property
    def rundown_complete(self) -> bool:
        """Whether the sheath has reached the end of the anode."""
        return self._rundown_complete

    @property
    def pinch_complete(self) -> bool:
        """Whether the radial shock has reached minimum radius (pinch)."""
        return self._pinch_complete

    @property
    def is_active(self) -> bool:
        """Whether the snowplow is still dynamically evolving.

        True during axial rundown, radial compression, and reflected shock.
        False after final pinch (frozen state).
        """
        return self.phase in ("rundown", "radial", "reflected")

    def step(
        self,
        dt: float,
        current: float,
        pressure: float | None = None,
    ) -> dict[str, float]:
        """Advance the snowplow by one timestep.

        Dispatches to axial or radial phase stepping as appropriate.

        Args:
            dt: Timestep [s].
            current: Circuit current [A].
            pressure: Fill gas pressure [Pa]. If None, uses self.p_fill.

        Returns:
            Dictionary with snowplow diagnostics:
                z_sheath: Sheath axial position [m].
                v_sheath: Sheath axial velocity [m/s].
                r_shock: Radial shock front radius [m].
                vr_shock: Radial shock velocity [m/s].
                L_plasma: Total plasma inductance [H].
                dL_dt: Rate of change of inductance [H/s].
                swept_mass: Mass swept by sheath [kg].
                F_magnetic: Magnetic driving force [N].
                F_pressure: Gas back-pressure force [N].
                phase: Current phase name.
        """
        self._elapsed_time += dt

        if self._pinch_complete:
            return self._frozen_result()

        if self.phase == "rundown":
            return self._step_axial(dt, current, pressure)
        elif self.phase == "reflected":
            return self._step_reflected(dt, current)
        else:
            return self._step_radial(dt, current, pressure)

    def _make_result(
        self, dL_dt: float, F_magnetic: float, F_pressure: float,
    ) -> dict[str, float]:
        """Build standard result dictionary."""
        return {
            "z_sheath": self.z,
            "v_sheath": self.v,
            "r_shock": self.r_shock,
            "vr_shock": self.vr,
            "L_plasma": self.plasma_inductance,
            "dL_dt": dL_dt,
            "swept_mass": self.swept_mass,
            "F_magnetic": F_magnetic,
            "F_pressure": F_pressure,
            "phase": self.phase,
        }

    def _frozen_result(self) -> dict[str, float]:
        """Return frozen state after pinch."""
        return self._make_result(dL_dt=0.0, F_magnetic=0.0, F_pressure=0.0)

    def _step_axial(
        self, dt: float, current: float, pressure: float | None,
    ) -> dict[str, float]:
        """Advance axial rundown phase by one timestep.

        Uses velocity-Verlet (leapfrog) integration for second-order accuracy.
        """
        p = pressure if pressure is not None else self.p_fill
        I_current = current

        # Forces
        F_mag = self.F_coeff * (self.f_c * I_current)**2  # [N]
        F_press = p * self.A_annular                        # [N]

        # Current mass and mass pickup rate
        m = max(self.swept_mass, 1e-20)
        dm_dt = self.rho0 * self.A_annular * self.f_m * abs(self.v)

        # Equation of motion: d(mv)/dt = F_mag - F_press
        # => m * dv/dt = F_mag - F_press - v * dm/dt
        a_n = (F_mag - F_press - self.v * dm_dt) / m

        # Velocity-Verlet: half-step velocity
        v_half = self.v + 0.5 * dt * a_n
        v_half = max(v_half, 0.0)

        # Position update
        z_new = self.z + dt * v_half

        # Check if rundown complete
        if z_new >= self.L_anode:
            z_new = self.L_anode
            self._rundown_complete = True
            self.z = z_new
            self.v = v_half
            self._L_axial_frozen = self.L_coeff * self.L_anode
            self.phase = "radial"
            # Start shock slightly inward from cathode to avoid zero-mass singularity
            # (analogous to axial z=1e-4 offset). The offset of 1% of (b-a) gives
            # a finite initial slug mass for the velocity-Verlet integrator.
            dr_init = 0.01 * (self.b - self.a)
            self.r_shock = self.b - dr_init
            self.vr = 0.0
            logger.info(
                "Snowplow rundown complete at t=%.2e s: z=%.3f m, v=%.0f m/s, "
                "L_p=%.2e H, m_swept=%.2e kg → entering radial phase",
                self._elapsed_time, self.z, self.v,
                self.plasma_inductance, self.swept_mass,
            )
            dL_dt = self.L_coeff * self.v
            return self._make_result(dL_dt=dL_dt, F_magnetic=F_mag, F_pressure=F_press)

        # Recompute acceleration at new position
        m_new = self.rho0 * self.A_annular * z_new * self.f_m
        m_new = max(m_new, 1e-20)
        dm_dt_new = self.rho0 * self.A_annular * self.f_m * abs(v_half)
        a_new = (F_mag - F_press - v_half * dm_dt_new) / m_new

        # Full-step velocity
        v_new = v_half + 0.5 * dt * a_new
        v_new = max(v_new, 0.0)

        # Update state
        self.z = z_new
        self.v = v_new

        dL_dt = self.L_coeff * self.v
        return self._make_result(dL_dt=dL_dt, F_magnetic=F_mag, F_pressure=F_press)

    def _adiabatic_back_pressure(self, r_s: float) -> float:
        """Adiabatic back-pressure from compressed fill gas ahead of the shock.

        As the cylindrical shock compresses inward from radius *b* to *r_s*,
        the fill gas is compressed adiabatically.  For a 2-D cylindrical
        compression the area ratio gives:

            p(r_s) = p_fill * (b / r_s)^(2 * gamma)

        where gamma = 5/3 for monatomic gas (deuterium).

        Returns the pressure [Pa] at radius *r_s*.
        """
        gamma = 5.0 / 3.0
        ratio = self.b / max(r_s, self.r_pinch_min)
        return self.p_fill * ratio ** (2.0 * gamma)

    def _step_radial(
        self, dt: float, current: float, pressure: float | None = None,
    ) -> dict[str, float]:
        """Advance radial inward shock phase by one timestep.

        Lee model Phase 3: slug model for cylindrical implosion.
        Uses velocity-Verlet integration.

        The radial J×B force drives the current sheath inward:
            F_rad = (mu_0 / 4pi) * (f_c * I)^2 * z_f / r_s

        Adiabatic back-pressure opposes the implosion:
            F_back = p_back * 2*pi * r_s * z_f
            p_back = p_fill * (b / r_s)^(2*gamma)

        Plasma inductance increases as the sheath compresses:
            L_radial = (mu_0 / 2pi) * z_f * ln(b / r_s)
            dL/dt = -(mu_0 / 2pi) * z_f * vr / r_s   (vr < 0 → dL/dt > 0)
        """
        I_current = current
        z_f = self.L_anode  # Pinch column length

        # Radial force: J×B on cylindrical current sheet
        r_s = max(self.r_shock, self.r_pinch_min)
        F_rad = (mu_0 / (4.0 * pi)) * (self.f_c * I_current)**2 * z_f / r_s

        # Adiabatic back-pressure force (opposes inward motion)
        # Use max of adiabatic estimate and external MHD pressure if provided
        p_back = self._adiabatic_back_pressure(r_s)
        if pressure is not None:
            p_back = max(p_back, pressure)
        F_pressure = p_back * 2.0 * pi * r_s * z_f

        # Current radial slug mass and mass pickup rate
        M_slug = max(self.radial_swept_mass, 1e-20)
        # dM/dt = f_mr * rho0 * 2*pi * r_s * |vr| * z_f
        dm_dt = self.f_mr * self.rho0 * 2.0 * pi * r_s * abs(self.vr) * z_f

        # Equation of motion: d(M*vr)/dt = -F_rad + F_pressure (inward)
        # => M * dvr/dt = -F_rad + F_pressure - vr * dM/dt
        a_n = (-F_rad + F_pressure - self.vr * dm_dt) / M_slug

        # Velocity-Verlet: half-step
        vr_half = self.vr + 0.5 * dt * a_n
        vr_half = min(vr_half, 0.0)  # Radial velocity must be inward (≤0)

        # Position update
        r_new = self.r_shock + dt * vr_half

        # Check if pinch reached — transition to reflected shock phase
        if r_new <= self.r_pinch_min:
            r_new = self.r_pinch_min
            self.r_shock = r_new
            self.vr = 0.0  # Stagnate at pinch, then reflected shock drives outward
            self._pinch_time = self._elapsed_time
            self.phase = "reflected"
            # Store pinch pressure for reflected shock driving force
            self._p_pinch = self._adiabatic_back_pressure(self.r_pinch_min)
            # Store slug mass at pinch for reflected phase EOM
            self._M_slug_pinch = max(self.radial_swept_mass, 1e-20)
            logger.info(
                "Radial pinch at t=%.2e s: r_s=%.2e m (%.1f%% of a), "
                "vr=%.0f m/s, L_p=%.2e H, I=%.0f A → entering reflected phase",
                self._elapsed_time, self.r_shock,
                100.0 * self.r_shock / self.a, self.vr,
                self.plasma_inductance, I_current,
            )
            # dL/dt at pinch
            dL_dt = -(mu_0 / (2.0 * pi)) * z_f * self.vr / max(self.r_shock, 1e-10)
            return self._make_result(
                dL_dt=dL_dt, F_magnetic=F_rad, F_pressure=F_pressure,
            )

        # Recompute acceleration at new position
        r_new_eff = max(r_new, self.r_pinch_min)
        M_new = self.f_mr * self.rho0 * pi * (self.b**2 - r_new**2) * z_f
        M_new = max(M_new, 1e-20)
        dm_dt_new = self.f_mr * self.rho0 * 2.0 * pi * r_new_eff * abs(vr_half) * z_f
        F_rad_new = (mu_0 / (4.0 * pi)) * (self.f_c * I_current)**2 * z_f / r_new_eff
        p_back_new = self._adiabatic_back_pressure(r_new_eff)
        F_pressure_new = p_back_new * 2.0 * pi * r_new_eff * z_f
        a_new = (-F_rad_new + F_pressure_new - vr_half * dm_dt_new) / M_new

        # Full-step velocity
        vr_new = vr_half + 0.5 * dt * a_new
        vr_new = min(vr_new, 0.0)

        # Update state
        self.r_shock = r_new
        self.vr = vr_new

        # dL/dt from radial compression
        dL_dt = -(mu_0 / (2.0 * pi)) * z_f * self.vr / max(self.r_shock, 1e-10)

        return self._make_result(dL_dt=dL_dt, F_magnetic=F_rad, F_pressure=F_pressure)

    def _step_reflected(self, dt: float, current: float) -> dict[str, float]:
        """Advance reflected shock phase (Lee Phase 5) by one timestep.

        After pinch, adiabatic back-pressure at r_min drives outward expansion
        of the shock front.  The inward J×B force opposes the expansion.

        The reflected shock terminates when:
        - r_shock reaches cathode radius b (full expansion), or
        - radial velocity reverses to negative (re-stagnation).

        Physics:
            F_pressure = p_back(r_s) * 2*pi * r_s * z_f   (drives outward)
            F_rad = (mu_0/4pi) * (f_c*I)^2 * z_f / r_s   (opposes, inward)
            M_slug * dvr/dt = F_pressure - F_rad

        During reflected phase dL/dt < 0 (inductance decreasing as r grows).
        """
        I_current = current
        z_f = self.L_anode

        r_s = max(self.r_shock, self.r_pinch_min)

        # Back-pressure at current radius (decreasing as shock expands)
        p_back = self._adiabatic_back_pressure(r_s)
        F_pressure = p_back * 2.0 * pi * r_s * z_f

        # J×B force (inward, opposing expansion)
        F_rad = (mu_0 / (4.0 * pi)) * (self.f_c * I_current)**2 * z_f / r_s

        # Slug mass from pinch (approximately constant during reflected phase)
        M_slug = self._M_slug_pinch

        # Equation of motion: M * dvr/dt = F_pressure - F_rad
        # Sign convention: vr > 0 = outward expansion
        a_n = (F_pressure - F_rad) / M_slug

        # Velocity-Verlet: half-step
        vr_half = self.vr + 0.5 * dt * a_n

        # Position update
        r_new = self.r_shock + dt * vr_half

        # Termination: reached cathode or re-stagnation (vr reverses inward)
        terminate = False
        if r_new >= self.b:
            r_new = self.b
            terminate = True
        if vr_half < 0.0 and self.vr >= 0.0:
            # Velocity reversed — re-stagnation
            terminate = True
            vr_half = 0.0

        if terminate:
            self.r_shock = r_new
            self.vr = vr_half
            self._pinch_complete = True
            self.phase = "pinch"
            logger.info(
                "Reflected shock terminated at t=%.2e s: r_s=%.2e m, "
                "vr=%.0f m/s → frozen pinch state",
                self._elapsed_time, self.r_shock, self.vr,
            )
            dL_dt = -(mu_0 / (2.0 * pi)) * z_f * self.vr / max(self.r_shock, 1e-10)
            return self._make_result(
                dL_dt=dL_dt, F_magnetic=F_rad, F_pressure=F_pressure,
            )

        # Recompute acceleration at new position
        r_new_eff = max(r_new, self.r_pinch_min)
        p_back_new = self._adiabatic_back_pressure(r_new_eff)
        F_pressure_new = p_back_new * 2.0 * pi * r_new_eff * z_f
        F_rad_new = (mu_0 / (4.0 * pi)) * (self.f_c * I_current)**2 * z_f / r_new_eff
        a_new = (F_pressure_new - F_rad_new) / M_slug

        # Full-step velocity
        vr_new = vr_half + 0.5 * dt * a_new

        # Update state
        self.r_shock = r_new
        self.vr = vr_new

        # dL/dt during expansion: r increasing → L decreasing → dL/dt < 0
        dL_dt = -(mu_0 / (2.0 * pi)) * z_f * self.vr / max(self.r_shock, 1e-10)

        return self._make_result(dL_dt=dL_dt, F_magnetic=F_rad, F_pressure=F_pressure)
