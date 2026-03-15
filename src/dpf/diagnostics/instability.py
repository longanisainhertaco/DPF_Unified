"""m=0 sausage instability growth rate for DPF pinch analysis.

Implements Kruskal-Schwarzschild linear stability analysis for the m=0
sausage mode in a Z-pinch / Dense Plasma Focus configuration.

The m=0 mode is the dominant instability in DPF pinch columns and
determines the pinch disruption timescale.  When the growth time
tau_m0 ~ 10-100 ns is shorter than the confinement time, the pinch
disrupts and beam-target neutron production commences.

Physics:
    gamma_m0 = k * v_A * sqrt(1 - gamma_gas * beta_p / 2)

    where:
        k = mode_number / a_pinch        [1/m]
        v_A = B_theta / sqrt(mu_0 * rho)  [m/s]  (Alfven speed)
        beta_p = 2*mu_0*p / B_theta^2     [dimensionless] (plasma beta)

    Stability criterion (Kadomtsev):
        beta_p > 2/gamma_gas  =>  stable (for gamma=5/3: beta_p > 1.2)

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
    B_theta = abs(float(B_theta))
    rho = max(float(rho), 1e-20)
    pressure = max(float(pressure), 0.0)
    a_pinch = max(float(a_pinch), 1e-10)

    # Cap B_theta to prevent overflow in B_sq (float64 max ~ 1.8e308)
    B_theta = min(B_theta, 1e150)

    # Alfven speed
    v_A = B_theta / np.sqrt(mu_0 * rho)

    # Plasma beta
    B_sq = B_theta**2
    beta_p = 2.0 * mu_0 * pressure / max(B_sq, 1e-30)

    # Critical beta for stability (Kadomtsev 1966): beta_p_crit = 2/gamma
    beta_p_crit = 2.0 / gamma
    stability_margin = beta_p_crit - beta_p

    # Wave number
    k = mode_number / a_pinch

    # Growth rate (Kadomtsev 1966): gamma_m0 = k * v_A * sqrt(1 - gamma*beta_p/2)
    # Stable when gamma*beta_p/2 >= 1, i.e. beta_p >= 2/gamma
    arg = 1.0 - gamma * beta_p / 2.0

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


def tearing_mode_growth_rate(
    B: np.ndarray,
    rho: np.ndarray,
    eta: float,
    dx: float,
    mu_0: float = 4 * np.pi * 1e-7,
) -> dict[str, Any]:
    """Compute tearing mode linear growth rate (Furth, Killeen, Rosenbluth 1963).

    The resistive tearing instability grows at current sheets where field lines
    reconnect.  The FKR scaling law gives:

        gamma_tearing ~ (k*delta)^(2/5) * S^(-3/5) * tau_A^(-1)

    where:
        S     = Lundquist number = mu_0 * v_A * L / eta
        tau_A = L / v_A  (Alfven transit time over system scale L)
        delta = current sheet half-thickness ~ dx  (resolved at grid scale)
        k     = pi / L  (fundamental reconnection mode wavenumber)

    The pre-factor follows the standard FKR result:
        gamma = (k * delta)^(2/5) * S^(-3/5) / tau_A

    Args:
        B: Magnetic field array.  Shape (3, ...) for vector field or (...) for
            scalar magnitude.  Units: T.
        rho: Mass density array matching B spatial shape.  Units: kg/m^3.
        eta: Resistivity (scalar).  Units: Ohm·m.
        dx: Grid cell size used as current-sheet thickness proxy.  Units: m.
        mu_0: Magnetic permeability of free space [H/m].  Default 4*pi*1e-7.

    Returns:
        Dictionary with:
            growth_rate: Linear growth rate gamma [1/s].
            growth_time: e-folding time 1/gamma [s] (inf if non-growing).
            lundquist_number: S = mu_0 * v_A * L / eta [dimensionless].
            alfven_speed: Volume-averaged Alfven speed v_A [m/s].
            alfven_time: tau_A = L / v_A [s].
            system_scale: Characteristic length L [m].
            is_tearing: True when S > 1 (resistive tearing is possible).

    References:
        Furth, H.P., Killeen, J. & Rosenbluth, M.N., Phys. Fluids 6, 459 (1963).
        Biskamp, D., Magnetic Reconnection in Plasmas, Cambridge (2000) §3.2.
    """
    B = np.asarray(B, dtype=float)
    rho = np.asarray(rho, dtype=float)
    eta = float(eta)
    dx = float(dx)

    # Magnetic field magnitude — handle both scalar and vector layouts
    if B.ndim > 1 and B.shape[0] == 3:
        B_mag = np.sqrt(np.sum(B**2, axis=0))
    else:
        B_mag = np.abs(B)

    # Volume-averaged quantities
    rho_mean = float(np.mean(np.maximum(rho, 1e-20)))
    B_mean = float(np.mean(B_mag))

    # Alfven speed
    v_A = B_mean / np.sqrt(mu_0 * rho_mean)

    # System scale: largest spatial extent of the B array
    n_cells = max(B_mag.size, 1)
    L = n_cells * dx  # full domain length

    # Current sheet thickness at grid scale
    delta = dx

    # Fundamental reconnection wavenumber
    k = np.pi / L

    # Alfven time
    tau_A = L / max(v_A, 1e-30)

    # Lundquist number S = mu_0 * v_A * L / eta
    S = mu_0 * v_A * L / max(eta, 1e-30)

    # FKR growth rate: gamma = (k*delta)^(2/5) * S^(-3/5) / tau_A
    kd = k * delta
    if S > 1.0 and kd > 0.0:
        growth_rate = (kd ** 0.4) * (S ** (-0.6)) / tau_A
        is_tearing = True
    else:
        growth_rate = 0.0
        is_tearing = False

    growth_time = 1.0 / growth_rate if growth_rate > 0.0 else float("inf")

    return {
        "growth_rate": growth_rate,
        "growth_time": growth_time,
        "lundquist_number": S,
        "alfven_speed": v_A,
        "alfven_time": tau_A,
        "system_scale": L,
        "is_tearing": is_tearing,
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
