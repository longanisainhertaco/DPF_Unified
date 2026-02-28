"""Lee Model for cross-checking DPF MHD results.

The Lee Model (Lee & Saw, 2008) is a lumped-circuit 5-phase model of
Dense Plasma Focus dynamics.  It couples the electrical circuit to the
plasma dynamics through a snowplow equation for the axial and radial
phases, providing I(t), pinch timing, and estimated neutron yield.

This implementation covers phases 1, 2, and 4 (reflected shock):

1. **Axial rundown phase**: The current sheet is launched at the insulator
   and accelerates axially along the anode.  The equation of motion is a
   snowplow model:

       d^2z/dt^2 = (mu_0 / (4*pi)) * ln(b/a) * (f_m*I)^2 / M_swept

   where M_swept = m0 + rho0 * pi * (b^2 - a^2) * z is the accumulated
   mass.  The circuit equation is simultaneously integrated.

2. **Radial inward shock phase**: Once the current sheet reaches the end
   of the anode, the radial implosion begins.  The slug model gives:

       d^2r_s/dt^2 = -(mu_0 / (4*pi)) * I^2 / (rho * r_s * L_pinch)

   where r_s is the shock radius and L_pinch is the pinch column length.

The ODE system is integrated with ``scipy.integrate.solve_ivp``.

Usage::

    from dpf.validation.lee_model_comparison import LeeModel

    model = LeeModel()
    result = model.run("PF-1000")
    print(f"Peak current: {result.peak_current:.2e} A")
    print(f"Pinch time:   {result.pinch_time:.2e} s")

References
----------
- S. Lee, J. Fusion Energy **33**, 319-335 (2014).
- S. Lee & S. H. Saw, J. Fusion Energy **27**, 292-295 (2008).
- S. Lee, Radiative Dense Plasma Focus Model, IAEA/IC (2005).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dpf.constants import k_B, m_d, mu_0, pi

logger = logging.getLogger(__name__)


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class LeeModelResult:
    """Container for Lee Model simulation results.

    Attributes:
        t: Time array [s].
        I: Current waveform [A].
        V: Capacitor voltage [V].
        z_sheet: Axial position of current sheet [m] (phase 1).
        r_shock: Radial shock position [m] (phase 2).
        peak_current: Peak discharge current [A].
        peak_current_time: Time of peak current [s].
        pinch_time: Estimated pinch time [s].
        device_name: Name of the device modeled.
        phases_completed: List of completed phase numbers.
        metadata: Additional metadata.
    """

    t: np.ndarray
    I: np.ndarray  # noqa: E741
    V: np.ndarray
    z_sheet: np.ndarray
    r_shock: np.ndarray
    peak_current: float = 0.0
    peak_current_time: float = 0.0
    pinch_time: float = 0.0
    device_name: str = ""
    phases_completed: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LeeModelComparison:
    """Comparison between Lee Model and MHD simulation results.

    Attributes:
        lee_result: Lee Model output.
        peak_current_rmse: RMSE between current waveforms (if available).
        peak_current_error: Relative error on peak current.
        timing_error: Relative error on peak current timing.
        waveform_nrmse: Normalized RMSE of full I(t) vs experimental waveform.
        device_name: Device name.
    """

    lee_result: LeeModelResult
    peak_current_rmse: float = 0.0
    peak_current_error: float = 0.0
    timing_error: float = 0.0
    waveform_nrmse: float = float("nan")
    device_name: str = ""


# ============================================================
# Device parameters helper
# ============================================================

def _get_device_params(device_name: str) -> dict[str, Any]:
    """Look up device parameters from the validation registry.

    Tries both ``dpf.validation.suite.DEVICE_REGISTRY`` and
    ``dpf.validation.experimental.DEVICES``.

    Args:
        device_name: Device name (e.g. "PF-1000", "NX2").

    Returns:
        Dictionary with keys: C, V0, L0, R0, anode_radius, cathode_radius,
        anode_length, fill_pressure_torr, peak_current.

    Raises:
        KeyError: If device not found in any registry.
    """
    from dpf.validation.experimental import DEVICES

    if device_name in DEVICES:
        dev = DEVICES[device_name]
        return {
            "C": dev.capacitance,
            "V0": dev.voltage,
            "L0": dev.inductance,
            "R0": dev.resistance,
            "anode_radius": dev.anode_radius,
            "cathode_radius": dev.cathode_radius,
            "anode_length": dev.anode_length,
            "fill_pressure_torr": dev.fill_pressure_torr,
            "peak_current_exp": dev.peak_current,
            "current_rise_time_exp": dev.current_rise_time,
            "crowbar_resistance": dev.crowbar_resistance,
        }

    from dpf.validation.suite import DEVICE_REGISTRY

    if device_name in DEVICE_REGISTRY:
        dev = DEVICE_REGISTRY[device_name]
        return {
            "C": dev.C,
            "V0": dev.V0,
            "L0": dev.L0,
            "R0": dev.R0,
            "anode_radius": dev.anode_radius,
            "cathode_radius": dev.cathode_radius,
            "anode_length": 0.16,  # Default for PF-1000 scale
            "fill_pressure_torr": 3.5,
            "peak_current_exp": dev.peak_current_A,
            "current_rise_time_exp": dev.peak_current_time_s,
        }

    raise KeyError(
        f"Device '{device_name}' not found. "
        f"Available: {list(DEVICES.keys())}"
    )


# ============================================================
# Lee Model
# ============================================================

class LeeModel:
    """Lee Model for DPF dynamics (phases 1, 2, 4 + crowbar).

    Integrates the coupled circuit + snowplow ODEs for the axial
    rundown phase, then transitions to the radial slug model.
    Optionally models crowbar activation (V_cap=0 → L-R decay).

    Args:
        fill_gas_mass: Mass of fill gas ion [kg].
            Default: deuterium (3.34e-27 kg).
        current_fraction: Fraction of total current in the current sheet
            (Lee's fc factor, typically 0.7-0.9).
        mass_fraction: Fraction of swept mass retained by the sheet
            (Lee's fm factor, typically 0.5-0.7).
        radial_mass_fraction: Fraction of gas swept radially (Lee's f_mr).
            Defaults to mass_fraction if None.  Lee & Saw (2014):
            f_mr ~ 0.07-0.12 for PF-1000, typically < f_m.
        liftoff_delay: Insulator flashover delay [s].  Accounts for the
            finite time before the current sheet lifts off the insulator
            and begins sweeping fill gas.  The model output time is shifted
            by this amount.  Typical values: 0.5-1.5 us for MJ-class DPF
            (Lee 2005, IAEA/IC).  Default: 0 (no delay).
        crowbar_enabled: If True, crowbar fires when V_cap first crosses
            zero (voltage-zero trigger).  Post-crowbar, the capacitor is
            short-circuited and current decays as L-R with constant
            inductance (frozen plasma column).  Default: False.
    """

    def __init__(
        self,
        fill_gas_mass: float = 6.687e-27,  # D2 molecular mass (2 * m_d)
        current_fraction: float = 0.7,
        mass_fraction: float = 0.7,
        radial_mass_fraction: float | None = None,
        pinch_column_fraction: float = 1.0,
        liftoff_delay: float = 0.0,
        crowbar_enabled: bool = False,
        crowbar_resistance: float = 0.0,
    ) -> None:
        self.fill_gas_mass = fill_gas_mass
        self.fm = mass_fraction      # Mass fraction factor (Lee's f_m)
        self.fc = current_fraction   # Current fraction factor (Lee's f_c)
        self.f_mr = radial_mass_fraction if radial_mass_fraction is not None else mass_fraction
        self.pinch_column_fraction = max(min(pinch_column_fraction, 1.0), 0.01)
        self.liftoff_delay = liftoff_delay  # Insulator flashover delay [s]
        self.crowbar_enabled = crowbar_enabled
        self.crowbar_resistance = crowbar_resistance  # [Ohm] spark gap arc resistance

    def run(
        self,
        device_name: str | None = None,
        device_params: dict[str, Any] | None = None,
    ) -> LeeModelResult:
        """Run the Lee Model for a given device.

        Either *device_name* or *device_params* must be provided.

        Args:
            device_name: Name of a registered device (e.g. "PF-1000", "NX2").
            device_params: Manual parameter dictionary with keys:
                C, V0, L0, R0, anode_radius, cathode_radius,
                anode_length, fill_pressure_torr.

        Returns:
            :class:`LeeModelResult` with time traces and diagnostics.

        Raises:
            ValueError: If neither device_name nor device_params is given.
        """
        from scipy.integrate import solve_ivp

        if device_params is None:
            if device_name is None:
                raise ValueError("Must provide either device_name or device_params")
            device_params = _get_device_params(device_name)

        name = device_name or "custom"

        # Extract parameters
        C = device_params["C"]
        V0 = device_params["V0"]
        L0 = device_params["L0"]
        R0 = device_params["R0"]
        a = device_params["anode_radius"]
        b = device_params["cathode_radius"]
        z_max = device_params["anode_length"]
        p_torr = device_params["fill_pressure_torr"]

        # Crowbar resistance: prefer device_params, fall back to constructor
        R_crowbar = device_params.get("crowbar_resistance", self.crowbar_resistance)

        # Fill gas density from pressure
        p_Pa = p_torr * 133.322  # Torr -> Pa
        T_room = 300.0  # K
        n_fill = p_Pa / (k_B * T_room)
        rho0 = n_fill * self.fill_gas_mass

        # Inductance per unit length: L_per_length = (mu_0 / 2pi) * ln(b/a)
        L_per_length = (mu_0 / (2.0 * pi)) * np.log(b / a)

        # Initial mass of fill gas between electrodes
        annulus_area = pi * (b**2 - a**2)

        logger.info(
            "Lee Model run: device=%s, C=%.2e F, V0=%.0f V, "
            "L0=%.2e H, R0=%.2e Ohm, a=%.4f m, b=%.4f m, "
            "z_max=%.4f m, p=%.1f Torr, rho0=%.4e kg/m^3",
            name, C, V0, L0, R0, a, b, z_max, p_torr, rho0,
        )

        # ── Phase 1: Axial rundown ──
        # State vector: y = [I, V_cap, z, dz/dt]
        # dI/dt = (V_cap - R0*I - I*dL_p/dt) / L_total
        # dV/dt = -I / C
        # dz/dt = vz
        # dvz/dt = (mu_0/(4*pi)) * ln(b/a) * (fm*I)^2 / M_swept(z)

        def axial_rhs(t: float, y: np.ndarray) -> np.ndarray:
            I, Vcap, z_pos, vz = y  # noqa: E741

            # Clamp
            z_pos = max(z_pos, 0.0)
            vz = max(vz, 0.0)

            # Swept mass
            M_swept = self.fm * rho0 * annulus_area * max(z_pos, 1e-6)

            # Plasma inductance: L_p = L_per_length * z
            L_p = L_per_length * max(z_pos, 1e-6)
            L_total = L0 + L_p

            # dL_p/dt = L_per_length * vz
            dLp_dt = L_per_length * vz

            # Circuit equation
            dI_dt = (Vcap - R0 * I - I * dLp_dt) / max(L_total, 1e-15)

            # Capacitor voltage
            dV_dt = -I / C

            # Snowplow acceleration (Lee & Saw 2014)
            # F = (mu_0/(4*pi)) * ln(b/a) * (fc*I)^2 - p_fill * A_annulus
            # Full equation: M * dvz/dt + vz * dM/dt = F_mag - F_back
            # where F_back = fill gas back-pressure force (opposes sheet motion)
            # dM/dt = fm * rho0 * annulus_area * vz
            dM_dt = self.fm * rho0 * annulus_area * vz

            F_magnetic = (mu_0 / (4.0 * pi)) * np.log(b / a) * (self.fc * I)**2
            F_back = p_Pa * annulus_area

            if M_swept > 1e-15:
                dvz_dt = (F_magnetic - F_back - vz * dM_dt) / M_swept
            else:
                dvz_dt = 0.0

            # Limit acceleration to prevent blow-up
            dvz_dt = np.clip(dvz_dt, -1e15, 1e15)

            return np.array([dI_dt, dV_dt, vz, dvz_dt])

        # Estimate simulation time: ~4 quarter-periods
        T_quarter = pi * np.sqrt(L0 * C)
        t_sim = 6.0 * T_quarter

        # Initial conditions: small initial position to avoid singularity
        y0_axial = np.array([0.0, V0, 1e-6, 0.0])

        # Integrate phase 1
        sol1 = solve_ivp(
            axial_rhs,
            [0, t_sim],
            y0_axial,
            method="RK45",
            max_step=t_sim / 5000,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )

        t1 = sol1.t
        I1 = sol1.y[0]
        V1 = sol1.y[1]
        z1 = sol1.y[2]
        _vz1 = sol1.y[3]  # noqa: F841

        # Find when current sheet reaches end of anode
        phase1_end_idx = len(t1) - 1
        for i in range(len(z1)):
            if z1[i] >= z_max:
                phase1_end_idx = i
                break

        phases_completed = [1]

        # ── Phase 2: Radial implosion (slug model) ──
        # State: [I, V_cap, r_shock, dr/dt]
        # Slug model: M_slug * d^2r/dt^2 = -(mu_0/(4*pi))*I^2/r + ...
        # Simplified: treat as inward-moving cylindrical shock
        # Transition when r_shock reaches some minimum (e.g., 0.1*a)

        t2 = np.array([])
        I2 = np.array([])
        V2 = np.array([])
        r2 = np.array([])
        pinch_time = 0.0

        if phase1_end_idx < len(t1) - 1:
            # Radial phase initial conditions
            t_start_r = t1[phase1_end_idx]
            I_start = I1[phase1_end_idx]
            V_start = V1[phase1_end_idx]
            r_start = b  # Start at cathode radius
            vr_start = 0.0  # Initially at rest radially

            # Pinch column length (effective length for radial phase)
            L_pinch = z_max * self.pinch_column_fraction

            # Adiabatic back-pressure parameters
            gamma = 5.0 / 3.0  # Monatomic gas
            p_fill = p_Pa  # Fill gas pressure [Pa]
            r_min = 0.1 * a  # Minimum pinch radius

            # Plasma inductance during radial phase
            # L_p = L_per_length * z_max + (mu_0/(2*pi)) * L_pinch * ln(b/r)
            L_p_axial = L_per_length * z_max

            def radial_rhs(t: float, y: np.ndarray) -> np.ndarray:
                I_r, Vcap_r, r_s, vr = y

                r_s = max(r_s, 0.001 * a)  # Minimum radius

                # Plasma inductance
                L_p_rad = (mu_0 / (2.0 * pi)) * L_pinch * np.log(max(b / r_s, 1.01))
                L_p_total = L_p_axial + L_p_rad
                L_total = L0 + L_p_total

                # dL_p/dt for radial phase
                dLp_dt_rad = -(mu_0 / (2.0 * pi)) * L_pinch * vr / max(r_s, 1e-10)

                # Circuit
                dI_dt = (Vcap_r - R0 * I_r - I_r * dLp_dt_rad) / max(L_total, 1e-15)
                dV_dt = -I_r / C

                # Dynamic radial slug mass: f_mr * rho0 * pi * (b^2 - r_s^2) * z_f
                # Increases as shock sweeps inward (r_s decreases)
                M_slug = self.f_mr * rho0 * pi * (b**2 - r_s**2) * L_pinch
                M_slug = max(M_slug, 1e-20)

                # Mass pickup rate: dM/dt = f_mr * rho0 * 2*pi * r_s * |vr| * z_f
                dm_dt = self.f_mr * rho0 * 2.0 * pi * r_s * abs(vr) * L_pinch

                # Radial J×B force: (mu_0/(4*pi)) * (fc*I)^2 * z_f / r_s
                F_rad = (
                    (mu_0 / (4.0 * pi)) * (self.fc * I_r)**2 * L_pinch
                    / max(r_s, 1e-10)
                )

                # Adiabatic back-pressure: p_fill * (b/r_s)^(2*gamma)
                r_eff = max(r_s, r_min)
                p_back = p_fill * (b / r_eff) ** (2.0 * gamma)
                F_pressure = p_back * 2.0 * pi * r_eff * L_pinch

                # Equation of motion: M * dvr/dt = -F_rad + F_pressure - vr * dM/dt
                dvr_dt = (-F_rad + F_pressure - vr * dm_dt) / M_slug

                dvr_dt = np.clip(dvr_dt, -1e15, 1e15)

                return np.array([dI_dt, dV_dt, vr, dvr_dt])

            # Simulate for ~2 quarter-periods more
            t_end_r = t_start_r + 2.0 * T_quarter
            y0_radial = np.array([I_start, V_start, r_start, vr_start])

            # Terminal event: shock reaches 10% of anode radius
            def pinch_event(t: float, y: np.ndarray) -> float:
                return y[2] - 0.1 * a

            pinch_event.terminal = True  # type: ignore[attr-defined]
            pinch_event.direction = -1  # type: ignore[attr-defined]

            sol2 = solve_ivp(
                radial_rhs,
                [t_start_r, t_end_r],
                y0_radial,
                method="RK45",
                max_step=(t_end_r - t_start_r) / 2000,
                rtol=1e-8,
                atol=1e-10,
                events=pinch_event,
            )

            t2 = sol2.t
            I2 = sol2.y[0]
            V2 = sol2.y[1]
            r2 = sol2.y[2]
            phases_completed.append(2)

            if len(sol2.t_events[0]) > 0:
                pinch_time = float(sol2.t_events[0][0])
            else:
                pinch_time = float(t2[-1]) if len(t2) > 0 else 0.0

            # ── Phase 4: Reflected shock (outward expansion) ──
            # After pinch, the compressed gas creates maximum back-pressure
            # that drives the shock front outward against the J×B force.
            # Physics: Rankine-Hugoniot post-shock density = 4*rho0,
            # mass accumulation from swept compressed gas.
            t4 = np.array([])
            I4 = np.array([])
            V4 = np.array([])
            r4 = np.array([])

            if len(t2) > 0:
                t_start_4 = float(t2[-1])
                I_start_4 = float(I2[-1])
                V_start_4 = float(V2[-1])
                r_pinch_start = float(r2[-1])

                # Pinch slug mass at stagnation
                M_slug_pinch = (
                    self.f_mr * rho0 * pi * (b**2 - r_pinch_start**2) * L_pinch
                )
                M_slug_pinch = max(M_slug_pinch, 1e-20)

                # Post-shock density: reflected shock encounters gas already
                # compressed to 4*rho0 by the inward shock (R-H strong limit,
                # gamma=5/3).  Reflected shock re-compresses by ~2x (Mach ~2
                # in pre-heated gas), giving ~8*rho0 total.  Strong limit
                # would give 16*rho0.  (PhD Debate #21 double-shock estimate.)
                rho_post = 8.0 * rho0

                def reflected_rhs(t: float, y: np.ndarray) -> np.ndarray:
                    I_r4, Vcap_r4, r_s, vr4 = y

                    r_s = max(r_s, 0.001 * a)

                    # Plasma inductance (same formula as Phase 2)
                    L_p_rad = (
                        (mu_0 / (2.0 * pi)) * L_pinch
                        * np.log(max(b / r_s, 1.01))
                    )
                    L_p_total = L_p_axial + L_p_rad
                    L_total = L0 + L_p_total

                    # dL_p/dt: vr4 > 0 (outward) → dL/dt < 0
                    dLp_dt_r4 = (
                        -(mu_0 / (2.0 * pi)) * L_pinch * vr4
                        / max(r_s, 1e-10)
                    )

                    # Circuit equation
                    dI_dt = (
                        (Vcap_r4 - R0 * I_r4 - I_r4 * dLp_dt_r4)
                        / max(L_total, 1e-15)
                    )
                    dV_dt = -I_r4 / C

                    # Reflected shock slug mass: initial pinch mass + swept gas
                    M_slug = M_slug_pinch + (
                        self.f_mr * rho_post * pi
                        * max(r_s**2 - r_pinch_start**2, 0.0) * L_pinch
                    )
                    M_slug = max(M_slug, 1e-20)

                    # Mass pickup rate from compressed gas
                    dm_dt = (
                        self.f_mr * rho_post * 2.0 * pi * r_s
                        * abs(vr4) * L_pinch
                    )

                    # Back-pressure drives outward expansion
                    r_eff = max(r_s, r_min)
                    p_back = p_fill * (b / r_eff) ** (2.0 * gamma)
                    F_pressure = p_back * 2.0 * pi * r_eff * L_pinch

                    # J×B force opposes outward motion
                    F_rad = (
                        (mu_0 / (4.0 * pi)) * (self.fc * I_r4)**2 * L_pinch
                        / max(r_s, 1e-10)
                    )

                    # EOM: F_pressure - F_rad - drag
                    dvr_dt = (F_pressure - F_rad - vr4 * dm_dt) / M_slug
                    dvr_dt = np.clip(dvr_dt, -1e15, 1e15)

                    return np.array([dI_dt, dV_dt, vr4, dvr_dt])

                # Terminal events for Phase 4
                def cathode_event(t: float, y: np.ndarray) -> float:
                    return b - y[2]  # trigger when r_shock reaches b

                cathode_event.terminal = True  # type: ignore[attr-defined]
                cathode_event.direction = -1  # type: ignore[attr-defined]

                def reversal_event(t: float, y: np.ndarray) -> float:
                    return y[3]  # trigger when vr crosses zero (re-stagnation)

                reversal_event.terminal = True  # type: ignore[attr-defined]
                reversal_event.direction = -1  # type: ignore[attr-defined]

                t_end_4 = t_start_4 + 2.0 * T_quarter
                y0_phase4 = np.array([
                    I_start_4, V_start_4, r_pinch_start, 0.0,
                ])

                sol4 = solve_ivp(
                    reflected_rhs,
                    [t_start_4, t_end_4],
                    y0_phase4,
                    method="RK45",
                    max_step=(t_end_4 - t_start_4) / 2000,
                    rtol=1e-8,
                    atol=1e-10,
                    events=[cathode_event, reversal_event],
                )

                if len(sol4.t) > 1:
                    t4 = sol4.t[1:]  # skip duplicate first point
                    I4 = sol4.y[0][1:]
                    V4 = sol4.y[1][1:]
                    r4 = sol4.y[2][1:]
                    phases_completed.append(4)

                    # Append reflected shock data to Phase 2 arrays
                    t2 = np.concatenate([t2, t4])
                    I2 = np.concatenate([I2, I4])
                    V2 = np.concatenate([V2, V4])
                    r2 = np.concatenate([r2, r4])

                    logger.info(
                        "Phase 4 reflected shock: r_min=%.4f m → "
                        "r_final=%.4f m, duration=%.2e s",
                        r_pinch_start, float(r4[-1]),
                        float(t4[-1] - t4[0]) if len(t4) > 1 else 0.0,
                    )

            # ── Post-pinch continuation (for crowbar detection) ──
            # After reflected shock (or pinch), continue circuit integration
            # with frozen plasma inductance until V_cap crosses zero.
            if self.crowbar_enabled and len(t2) > 0 and V2[-1] > 0:
                t_post_start = float(t2[-1])
                I_post_start = float(I2[-1])
                V_post_start = float(V2[-1])
                r_pinch = float(r2[-1])

                # Frozen plasma inductance at pinch
                L_p_frozen = L_per_length * z_max
                L_p_frozen += (
                    (mu_0 / (2.0 * pi)) * z_max
                    * np.log(max(b / max(r_pinch, 0.001 * a), 1.01))
                )
                L_total_frozen = L0 + L_p_frozen

                def post_pinch_rhs(t: float, y: np.ndarray) -> np.ndarray:
                    I_pp, V_pp = y
                    dI_dt = (V_pp - R0 * I_pp) / L_total_frozen
                    dV_dt = -I_pp / C
                    return np.array([dI_dt, dV_dt])

                def voltage_zero_event(t: float, y: np.ndarray) -> float:
                    return y[1]  # V_cap

                voltage_zero_event.terminal = True  # type: ignore[attr-defined]
                voltage_zero_event.direction = -1  # type: ignore[attr-defined]

                t_post_end = t_post_start + 3.0 * T_quarter
                sol_post = solve_ivp(
                    post_pinch_rhs,
                    [t_post_start, t_post_end],
                    np.array([I_post_start, V_post_start]),
                    method="RK45",
                    max_step=(t_post_end - t_post_start) / 2000,
                    rtol=1e-8,
                    atol=1e-10,
                    events=voltage_zero_event,
                )

                # Append post-pinch data
                t2 = np.concatenate([t2, sol_post.t[1:]])
                I2 = np.concatenate([I2, sol_post.y[0][1:]])
                V2 = np.concatenate([V2, sol_post.y[1][1:]])
                r2 = np.concatenate([r2, np.full(len(sol_post.t) - 1, r_pinch)])

        # Combine phases
        if len(t2) > 0:
            t_combined = np.concatenate([t1[:phase1_end_idx + 1], t2])
            I_combined = np.concatenate([I1[:phase1_end_idx + 1], I2])
            V_combined = np.concatenate([V1[:phase1_end_idx + 1], V2])
            z_combined = np.concatenate([
                z1[:phase1_end_idx + 1],
                np.full(len(t2), z_max),
            ])
            r_combined = np.concatenate([
                np.full(phase1_end_idx + 1, b),
                r2,
            ])
        else:
            t_combined = t1
            I_combined = I1
            V_combined = V1
            z_combined = z1
            r_combined = np.full(len(t1), b)

        # ── Crowbar phase: L-R decay after V_cap crosses zero ──
        if self.crowbar_enabled:
            # Find first zero-crossing of V_combined (after peak current)
            # Look for sign change in V_combined
            cb_idx = None
            for i in range(1, len(V_combined)):
                if V_combined[i - 1] > 0 and V_combined[i] <= 0:
                    cb_idx = i
                    break

            if cb_idx is not None:
                t_cb = float(t_combined[cb_idx])
                I_cb = float(I_combined[cb_idx])

                # Total inductance at crowbar time (frozen)
                # Use the last known plasma inductance
                L_p_at_cb = L_per_length * z_max
                if len(r2) > 0:
                    # Add radial phase inductance
                    r_at_cb = max(float(r_combined[cb_idx]), 0.001 * a)
                    L_p_at_cb += (mu_0 / (2.0 * pi)) * z_max * np.log(max(b / r_at_cb, 1.01))
                L_total_cb = L0 + L_p_at_cb

                # L-R decay: I(t) = I_cb * exp(-R_post * (t - t_cb) / L_total_cb)
                # Post-crowbar resistance = R0 + crowbar spark gap arc resistance
                R_post_cb = R0 + R_crowbar
                # Extend to 3 e-folding times or until 10x the crowbar time
                tau_LR = L_total_cb / max(R_post_cb, 1e-10)
                t_end_cb = t_cb + min(5.0 * tau_LR, 3.0 * T_quarter)
                n_cb_pts = 500
                t_cb_arr = np.linspace(t_cb, t_end_cb, n_cb_pts)
                I_cb_arr = I_cb * np.exp(-R_post_cb * (t_cb_arr - t_cb) / L_total_cb)
                V_cb_arr = np.zeros(n_cb_pts)  # Capacitor shorted
                z_cb_arr = np.full(n_cb_pts, z_max)
                r_cb_arr = np.full(n_cb_pts, float(r_combined[cb_idx]))

                # Truncate pre-crowbar waveform and append crowbar decay
                t_combined = np.concatenate([t_combined[:cb_idx], t_cb_arr])
                I_combined = np.concatenate([I_combined[:cb_idx], I_cb_arr])
                V_combined = np.concatenate([V_combined[:cb_idx], V_cb_arr])
                z_combined = np.concatenate([z_combined[:cb_idx], z_cb_arr])
                r_combined = np.concatenate([r_combined[:cb_idx], r_cb_arr])
                phases_completed.append(3)  # Crowbar phase

                logger.info(
                    "Crowbar fired at t=%.2e s, I_cb=%.2e A, tau_LR=%.2e s",
                    t_cb, I_cb, tau_LR,
                )

        # Apply liftoff delay: shift time origin to account for insulator
        # flashover period before current sheet formation.
        if self.liftoff_delay > 0:
            t_combined = t_combined + self.liftoff_delay
            if pinch_time > 0:
                pinch_time += self.liftoff_delay

        # Diagnostics — use first-peak finder to avoid post-pinch oscillation
        from dpf.validation.experimental import _find_first_peak

        abs_I = np.abs(I_combined)
        peak_idx = _find_first_peak(abs_I)
        peak_current = float(abs_I[peak_idx])
        peak_current_time = float(t_combined[peak_idx])

        if pinch_time <= 0:
            pinch_time = peak_current_time * 1.2  # Rough estimate

        logger.info(
            "Lee Model completed: phases=%s, peak_I=%.2e A at t=%.2e s, "
            "pinch_time=%.2e s, liftoff_delay=%.2e s",
            phases_completed, peak_current, peak_current_time, pinch_time,
            self.liftoff_delay,
        )

        return LeeModelResult(
            t=t_combined,
            I=I_combined,
            V=V_combined,
            z_sheet=z_combined,
            r_shock=r_combined,
            peak_current=peak_current,
            peak_current_time=peak_current_time,
            pinch_time=pinch_time,
            device_name=name,
            phases_completed=phases_completed,
            metadata={
                "C": C,
                "V0": V0,
                "L0": L0,
                "R0": R0,
                "R_crowbar": R_crowbar,
                "anode_radius": a,
                "cathode_radius": b,
                "anode_length": z_max,
                "fill_pressure_torr": p_torr,
                "rho0": rho0,
                "fm": self.fm,
                "f_mr": self.f_mr,
                "fc": self.fc,
                "liftoff_delay": self.liftoff_delay,
            },
        )

    def compare_with_experiment(
        self,
        device_name: str,
        truncate_at_dip: bool = False,
    ) -> LeeModelComparison:
        """Run the Lee Model and compare against experimental data.

        Args:
            device_name: Name of the device.
            truncate_at_dip: If True, compute NRMSE only up to the current
                dip, excluding the post-pinch frozen-L region.

        Returns:
            :class:`LeeModelComparison` with error metrics.
        """
        params = _get_device_params(device_name)
        result = self.run(device_name=device_name)

        # Compare peak current
        Ipeak_exp = params.get("peak_current_exp", 0.0)
        if Ipeak_exp > 0:
            peak_err = abs(result.peak_current - Ipeak_exp) / Ipeak_exp
        else:
            peak_err = 0.0

        # Compare timing
        t_rise_exp = params.get("current_rise_time_exp", 0.0)
        if t_rise_exp > 0:
            timing_err = abs(result.peak_current_time - t_rise_exp) / t_rise_exp
        else:
            timing_err = 0.0

        # Waveform NRMSE against experimental digitized I(t)
        waveform_nrmse = float("nan")
        from dpf.validation.experimental import DEVICES, nrmse_peak
        if device_name in DEVICES:
            dev = DEVICES[device_name]
            if dev.waveform_t is not None and dev.waveform_I is not None:
                waveform_nrmse = nrmse_peak(
                    result.t, result.I, dev.waveform_t, dev.waveform_I,
                    truncate_at_dip=truncate_at_dip,
                )

        return LeeModelComparison(
            lee_result=result,
            peak_current_error=peak_err,
            timing_error=timing_err,
            waveform_nrmse=waveform_nrmse,
            device_name=device_name,
        )


# ============================================================
# Neutron yield estimation
# ============================================================


def estimate_neutron_yield_from_lee_result(result: LeeModelResult) -> float:
    """Estimate total DD neutron yield per shot from a Lee Model pinch.

    Uses the beam-target fusion mechanism, which dominates thermonuclear
    yield in DPF devices operating below ~1 MJ stored energy.

    **Algorithm** (Lee & Saw 2008, Section III):

    1. Extract pinch geometry from ``result.metadata``:
       ``a``, ``b``, ``z_f``, ``rho0``, ``fm``.
    2. Estimate pinch density via adiabatic cylindrical compression from
       the fill gas to radius ``r_pinch_min = 0.1 * a``::

           M_slug = fm * rho0 * pi * (b^2 - r_pinch_min^2) * z_f
           V_pinch = pi * r_pinch_min^2 * z_f
           n_target = M_slug / (m_D * V_pinch)

    3. Estimate pinch Alfven speed::

           B_theta = mu_0 * I_pinch / (2 * pi * r_pinch_min)
           rho_slug = M_slug / V_pinch
           v_Alfven = B_theta / sqrt(mu_0 * rho_slug)

    4. Estimate pinch voltage (back-EMF from inductance change)::

           V_pinch_V = (mu_0/(2*pi)) * z_f * v_Alfven * I_pinch / r_pinch_min

       Note: DPF pinch voltages of 100 kV – 2 MV are physically expected
       for megaampere-class devices (Lee & Saw 2008, Table II).

    5. Compute Alfven transit time (dwell time) and integrate yield::

           tau_dwell = r_pinch_min / v_Alfven
           dY_dt = beam_target_yield_rate(I_pinch, V_pinch_V, n_target, z_f)
           Y_total = dY_dt * tau_dwell

    Accuracy: This 0D model gives order-of-magnitude estimates (factor
    1–100 of experimental). Shot-to-shot variability alone is typically
    a factor 2–5x.

    Args:
        result: Completed :class:`LeeModelResult` with metadata populated.

    Returns:
        Estimated total neutron yield per shot (dimensionless count).
        Returns 0.0 if metadata is incomplete or pinch was not reached.
    """
    from dpf.diagnostics.beam_target import beam_target_yield_rate

    meta = result.metadata
    if not meta:
        return 0.0

    # Extract device geometry
    a: float = meta.get("anode_radius", 0.0)
    b: float = meta.get("cathode_radius", 0.0)
    z_f: float = meta.get("anode_length", 0.0)
    rho0: float = meta.get("rho0", 0.0)
    # Use f_mr for radial phase slug mass; fall back to fm for backward compat
    fm: float = meta.get("f_mr", meta.get("fm", 0.3))

    if a <= 0.0 or b <= 0.0 or z_f <= 0.0 or rho0 <= 0.0:
        return 0.0

    # Pinch radius (Lee standard: 10% of anode radius)
    r_pinch_min = 0.1 * a

    # Pinch-phase current: use current at pinch time, not peak current
    # Peak current occurs before pinch; pinch current is typically 60-80% of peak
    if result.pinch_time > 0 and len(result.t) > 0:
        pinch_idx = int(np.searchsorted(result.t, result.pinch_time))
        pinch_idx = min(pinch_idx, len(result.I) - 1)
        I_pinch = max(float(np.abs(result.I[pinch_idx])), 0.0)
    else:
        I_pinch = max(result.peak_current, 0.0)  # fallback
    if I_pinch <= 0.0:
        return 0.0

    # Swept mass in radial phase (cylindrical slug from b to r_pinch_min)
    M_slug = fm * rho0 * pi * (b**2 - r_pinch_min**2) * z_f
    M_slug = max(M_slug, 1.0e-30)

    # Pinch volume
    V_pinch_vol = pi * r_pinch_min**2 * z_f

    # Target number density (deuterium) in pinch column
    n_target = M_slug / (m_d * V_pinch_vol)
    n_target = max(n_target, 0.0)

    # Azimuthal B-field at pinch surface
    B_theta = mu_0 * I_pinch / (2.0 * pi * r_pinch_min)

    # Mass density in pinch slug
    rho_slug = M_slug / V_pinch_vol

    # Alfven speed in compressed slug
    v_Alfven = B_theta / max(np.sqrt(mu_0 * rho_slug), 1.0e-30)
    v_Alfven = max(v_Alfven, 1.0)  # floor at 1 m/s to avoid division by zero

    # Pinch voltage from inductive back-EMF during radial compression
    # V = (mu_0 / 2pi) * z_f * v_Alfven * I / r_pinch_min
    # [H/m * m * (m/s) * A / m] = H * A / (m * s) = V (dimensional check OK)
    V_pinch_V = (mu_0 / (2.0 * pi)) * z_f * v_Alfven * I_pinch / r_pinch_min

    # Alfven transit time = dwell time of pinch
    tau_dwell = r_pinch_min / v_Alfven

    # Beam-target yield rate [neutrons/s]
    dY_dt = beam_target_yield_rate(
        I_pinch=I_pinch,
        V_pinch=V_pinch_V,
        n_target=n_target,
        L_target=z_f,
    )

    return float(dY_dt * tau_dwell)
