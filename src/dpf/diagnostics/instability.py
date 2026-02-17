"""m=0 sausage instability growth rate for DPF pinch analysis.

Implements Kruskal-Schwarzschild linear stability analysis for the m=0
sausage mode in a Z-pinch / Dense Plasma Focus configuration.

The m=0 mode is the dominant instability in DPF pinch columns and
determines the pinch disruption timescale.  When the growth time
tau_m0 ~ 10-100 ns is shorter than the confinement time, the pinch
disrupts and beam-target neutron production commences.

Physics:
    gamma_m0 = k * v_A * sqrt(1 - beta_p / (2 + gamma_gas * beta_p))

    where:
        k = mode_number / a_pinch        [1/m]
        v_A = B_theta / sqrt(mu_0 * rho)  [m/s]  (Alfven speed)
        beta_p = 2*mu_0*p / B_theta^2     [dimensionless] (plasma beta)

    Stability criterion (Kadomtsev):
        beta_p > 2/(gamma_gas - 1)  =>  stable (for gamma=5/3: beta_p > 3)

References:
    Kruskal, M. & Schwarzschild, M., Proc. R. Soc. A 223, 348 (1954).
    Kadomtsev, B.B., Rev. Plasma Phys. 2, 153 (1966).
    Haines, M.G., Plasma Phys. Control. Fusion 53, 093001 (2011).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dpf.constants import mu_0


def m0_growth_rate(
    B_theta: float,
    rho: float,
    pressure: float,
    a_pinch: float,
    mode_number: int = 1,
    gamma: float = 5.0 / 3.0,
) -> dict[str, Any]:
    """Compute m=0 sausage instability growth rate for a Z-pinch.

    Args:
        B_theta: Azimuthal magnetic field at pinch surface [T].
        rho: Mass density in pinch column [kg/m^3].
        pressure: Plasma pressure in pinch column [Pa].
        a_pinch: Pinch column radius [m].
        mode_number: Axial mode number (k = mode_number / a_pinch).
            Default 1 (fundamental mode).
        gamma: Ratio of specific heats (default 5/3 for monatomic).

    Returns:
        Dictionary with:
            growth_rate: Linear growth rate gamma_m0 [1/s].
            growth_time: e-folding time 1/gamma_m0 [s] (inf if stable).
            alfven_speed: Alfven speed v_A [m/s].
            beta_p: Plasma beta (2*mu_0*p / B_theta^2).
            is_unstable: True if growth_rate > 0.
            stability_margin: beta_p_critical - beta_p (positive = unstable).
    """
    B_theta = abs(B_theta)
    rho = max(rho, 1e-20)
    pressure = max(pressure, 0.0)
    a_pinch = max(a_pinch, 1e-10)

    # Alfven speed
    v_A = B_theta / np.sqrt(mu_0 * rho)

    # Plasma beta
    B_sq = B_theta**2
    beta_p = 2.0 * mu_0 * pressure / max(B_sq, 1e-30)

    # Critical beta for stability: beta_p_crit = 2/(gamma - 1)
    beta_p_crit = 2.0 / (gamma - 1.0)
    stability_margin = beta_p_crit - beta_p

    # Wave number
    k = mode_number / a_pinch

    # Growth rate: gamma_m0 = k * v_A * sqrt(1 - beta_p / (2 + gamma*beta_p))
    # The argument under the sqrt must be positive for instability
    denom = 2.0 + gamma * beta_p
    arg = 1.0 - beta_p / max(denom, 1e-30)

    if arg > 0:
        growth_rate = k * v_A * np.sqrt(arg)
        is_unstable = True
    else:
        growth_rate = 0.0
        is_unstable = False

    growth_time = 1.0 / growth_rate if growth_rate > 0 else float("inf")

    return {
        "growth_rate": growth_rate,
        "growth_time": growth_time,
        "alfven_speed": v_A,
        "beta_p": beta_p,
        "is_unstable": is_unstable,
        "stability_margin": stability_margin,
    }


def m0_growth_rate_from_state(
    state: dict[str, np.ndarray],
    snowplow: Any,
    config: Any,
) -> dict[str, Any]:
    """Extract pinch-region parameters from MHD state and compute m=0 growth rate.

    Uses the snowplow shock radius to identify the pinch region, then
    volume-averages B_theta, rho, and pressure within that region.

    Args:
        state: MHD state dictionary with 'B', 'rho', 'pressure' arrays.
        snowplow: SnowplowModel instance (provides r_shock, a, phase).
        config: SimulationConfig (provides dx, geometry).

    Returns:
        Dictionary from :func:`m0_growth_rate`, or a default stable result
        if the pinch region cannot be identified.
    """
    default_result = {
        "growth_rate": 0.0,
        "growth_time": float("inf"),
        "alfven_speed": 0.0,
        "beta_p": 0.0,
        "is_unstable": False,
        "stability_margin": 0.0,
    }

    if snowplow is None:
        return default_result

    r_shock = snowplow.r_shock
    a_pinch = max(r_shock, snowplow.r_pinch_min)

    # Extract MHD arrays
    B = state.get("B")
    rho = state.get("rho")
    pressure = state.get("pressure")

    if B is None or rho is None or pressure is None:
        return default_result

    # B_theta is the azimuthal component (index 1 in cylindrical)
    # For Cartesian, approximate B_theta from By at the pinch radius
    dr = config.dx
    ir_shock = round(r_shock / dr) if dr > 0 else 0
    nx = rho.shape[0]

    if ir_shock <= 0 or ir_shock >= nx:
        # Pinch region not resolvable on grid — use volume averages
        B_theta_avg = float(np.mean(np.abs(B[1])))
        rho_avg = float(np.mean(rho))
        p_avg = float(np.mean(pressure))
    else:
        # Average within the pinch region (r < r_shock)
        B_theta_avg = float(np.mean(np.abs(B[1, :ir_shock])))
        rho_avg = float(np.mean(rho[:ir_shock]))
        p_avg = float(np.mean(pressure[:ir_shock]))

    return m0_growth_rate(
        B_theta=B_theta_avg,
        rho=rho_avg,
        pressure=p_avg,
        a_pinch=a_pinch,
    )
