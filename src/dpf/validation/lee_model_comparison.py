"""Simplified Lee Model for cross-checking DPF MHD results.

The Lee Model (Lee & Saw, 2008) is a lumped-circuit 5-phase model of
Dense Plasma Focus dynamics.  It couples the electrical circuit to the
plasma dynamics through a snowplow equation for the axial and radial
phases, providing I(t), pinch timing, and estimated neutron yield.

This implementation covers phases 1 and 2 only (MVP):

1. **Axial rundown phase**: The current sheet is launched at the insulator
   and accelerates axially along the anode.  The equation of motion is a
   snowplow model:

       d^2z/dt^2 = (mu_0 / 2) * ln(b/a) * I^2 / M_swept

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

from dpf.constants import k_B, mu_0, pi

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
    I: np.ndarray
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
        device_name: Device name.
    """

    lee_result: LeeModelResult
    peak_current_rmse: float = 0.0
    peak_current_error: float = 0.0
    timing_error: float = 0.0
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
    """Simplified Lee Model for DPF dynamics (phases 1-2).

    Integrates the coupled circuit + snowplow ODEs for the axial
    rundown phase, then transitions to the radial slug model.

    Args:
        fill_gas_mass: Mass of fill gas ion [kg].
            Default: deuterium (3.34e-27 kg).
        current_fraction: Fraction of total current in the current sheet
            (Lee's fm factor, typically 0.7-0.9).
        mass_fraction: Fraction of swept mass retained by the sheet
            (Lee's fc factor, typically 0.5-0.7).
    """

    def __init__(
        self,
        fill_gas_mass: float = 3.34e-27,
        current_fraction: float = 0.7,
        mass_fraction: float = 0.7,
    ) -> None:
        self.fill_gas_mass = fill_gas_mass
        self.fm = current_fraction  # Current fraction factor
        self.fc = mass_fraction     # Mass fraction factor

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
        # dvz/dt = (mu_0/2) * ln(b/a) * fm^2 * I^2 / M_swept(z)

        def axial_rhs(t: float, y: np.ndarray) -> np.ndarray:
            I, Vcap, z_pos, vz = y

            # Clamp
            z_pos = max(z_pos, 0.0)
            vz = max(vz, 0.0)

            # Swept mass
            M_swept = self.fc * rho0 * annulus_area * max(z_pos, 1e-6)

            # Plasma inductance: L_p = L_per_length * z
            L_p = L_per_length * max(z_pos, 1e-6)
            L_total = L0 + L_p

            # dL_p/dt = L_per_length * vz
            dLp_dt = L_per_length * vz

            # Circuit equation
            dI_dt = (Vcap - R0 * I - I * dLp_dt) / max(L_total, 1e-15)

            # Capacitor voltage
            dV_dt = -I / C

            # Snowplow acceleration
            # F = (mu_0/2) * ln(b/a) * (fm*I)^2 / (2*pi)
            # ... actually the standard Lee formulation:
            # M * d^2z/dt^2 = (mu_0*ln(b/a))/(2) * fm^2 * I^2
            # minus mass pickup: M * dvz/dt + vz * dM/dt = F
            # dM/dt = fc * rho0 * annulus_area * vz
            dM_dt = self.fc * rho0 * annulus_area * vz

            F_magnetic = 0.5 * mu_0 * np.log(b / a) * (self.fm * I)**2

            if M_swept > 1e-15:
                dvz_dt = (F_magnetic - vz * dM_dt) / M_swept
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
        vz1 = sol1.y[3]

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

            # Pinch column length
            L_pinch = z_max

            # Mass of gas in the radial slug (per unit length)
            # At the start of radial phase, gas between a and b
            M_radial = rho0 * pi * (b**2 - a**2) * L_pinch

            # Plasma inductance during radial phase
            # L_p = L_per_length * z_max + (mu_0/(2*pi)) * L_pinch * ln(b/r)
            L_p_axial = L_per_length * z_max

            def radial_rhs(t: float, y: np.ndarray) -> np.ndarray:
                I_r, Vcap_r, r_s, vr = y

                r_s = max(r_s, 0.001 * a)  # Minimum radius
                vr = min(vr, 0.0)  # vr should be negative (inward)

                # Plasma inductance
                L_p_rad = (mu_0 / (2.0 * pi)) * L_pinch * np.log(max(b / r_s, 1.01))
                L_p_total = L_p_axial + L_p_rad
                L_total = L0 + L_p_total

                # dL_p/dt for radial phase
                dLp_dt_rad = -(mu_0 / (2.0 * pi)) * L_pinch * vr / max(r_s, 1e-10)

                # Circuit
                dI_dt = (Vcap_r - R0 * I_r - I_r * dLp_dt_rad) / max(L_total, 1e-15)
                dV_dt = -I_r / C

                # Radial slug equation (simplified)
                # F = (mu_0/(4*pi)) * fm^2 * I^2 / r
                # M * d^2r/dt^2 = -F (inward)
                F_rad = (mu_0 / (4.0 * pi)) * (self.fm * I_r)**2 / max(r_s, 1e-10)

                if M_radial > 1e-15:
                    dvr_dt = -F_rad / M_radial
                else:
                    dvr_dt = 0.0

                dvr_dt = np.clip(dvr_dt, -1e15, 0.0)

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

        # Diagnostics
        abs_I = np.abs(I_combined)
        peak_idx = int(np.argmax(abs_I))
        peak_current = float(abs_I[peak_idx])
        peak_current_time = float(t_combined[peak_idx])

        if pinch_time <= 0:
            pinch_time = peak_current_time * 1.2  # Rough estimate

        logger.info(
            "Lee Model completed: phases=%s, peak_I=%.2e A at t=%.2e s, "
            "pinch_time=%.2e s",
            phases_completed, peak_current, peak_current_time, pinch_time,
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
                "anode_radius": a,
                "cathode_radius": b,
                "anode_length": z_max,
                "fill_pressure_torr": p_torr,
                "rho0": rho0,
                "fm": self.fm,
                "fc": self.fc,
            },
        )

    def compare_with_experiment(
        self,
        device_name: str,
    ) -> LeeModelComparison:
        """Run the Lee Model and compare against experimental data.

        Args:
            device_name: Name of the device.

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

        return LeeModelComparison(
            lee_result=result,
            peak_current_error=peak_err,
            timing_error=timing_err,
            device_name=device_name,
        )
