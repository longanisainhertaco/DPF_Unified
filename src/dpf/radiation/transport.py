"""Flux-limited diffusion (FLD) radiation transport.

Implements the Levermore-Pomraning flux limiter for radiation energy
transport in the diffusion approximation:

    dE_r/dt = div(D * grad(E_r)) - kappa_a * c * E_r + eta
    D = c * lambda(R) / kappa

where:
    R = |grad(E_r)| / (kappa * E_r)    — Knudsen number
    lambda(R) = (1/R)(coth(R) - 1/R)   — Levermore-Pomraning flux limiter

In the optically thick limit (R -> 0): lambda -> 1/3 (diffusion)
In the optically thin limit (R -> inf): lambda -> 1/R (free streaming)

The FLD approximation is operator-split from the hydrodynamics and
solved with explicit sub-cycling for stability.

Reference:
    Levermore & Pomraning, ApJ 248, 321 (1981)
    Mihalas & Mihalas, "Foundations of Radiation Hydrodynamics" (1984)
"""

from __future__ import annotations

import numpy as np

from dpf.constants import c as c_light
from dpf.constants import k_B


def levermore_pomraning_limiter(R: np.ndarray) -> np.ndarray:
    """Compute the Levermore-Pomraning flux limiter lambda(R).

    lambda(R) = (1/R)(coth(R) - 1/R)

    Handles R -> 0 limit analytically: lambda(0) = 1/3.

    Args:
        R: Knudsen number array (non-negative).

    Returns:
        Flux limiter values in [0, 1/3].
    """
    lam = np.full_like(R, 1.0 / 3.0)

    # For R > small threshold, compute full expression
    mask = R > 1e-4
    Rm = R[mask]

    # coth(R) = (exp(2R) + 1) / (exp(2R) - 1)
    # For numerical stability with large R, use:
    # coth(R) = 1 + 2/(exp(2R) - 1)
    exp2R = np.exp(np.minimum(2.0 * Rm, 500.0))  # Cap to avoid overflow
    coth_R = (exp2R + 1.0) / (exp2R - 1.0)

    lam[mask] = (coth_R - 1.0 / Rm) / Rm

    # Clamp to valid range [0, 1/3]
    lam = np.clip(lam, 0.0, 1.0 / 3.0)

    return lam


def compute_radiation_energy(
    Te: np.ndarray,
    ne: np.ndarray,
) -> np.ndarray:
    """Compute radiation energy density from local thermal equilibrium.

    E_r = a_R * T^4  where a_R = 4*sigma_SB/c is the radiation constant.

    For LTE initialization. In general E_r is evolved separately.

    Args:
        Te: Electron temperature [K].
        ne: Electron density [m^-3] (unused, for interface consistency).

    Returns:
        Radiation energy density [J/m^3].
    """
    # Radiation constant a_R = 4 * sigma_SB / c
    # sigma_SB = 5.670374419e-8 W/(m^2 K^4)
    sigma_SB = 5.670374419e-8
    a_R = 4.0 * sigma_SB / c_light

    Te_safe = np.maximum(Te, 0.0)
    return a_R * Te_safe**4


def compute_rosseland_opacity(
    ne: np.ndarray,
    Te: np.ndarray,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
) -> np.ndarray:
    """Compute Rosseland mean free-free (inverse bremsstrahlung) opacity.

    The Kramers opacity law for free-free absorption:
        kappa_ff = C_ff * Z^2 * g_ff * ne * ni / Te^{3.5}   [1/m]

    where C_ff = 3.7e8 cm^5 K^{3.5} (CGS), converting to SI:
        C_ff(SI) = 3.7e8 * (1e-2)^5 = 3.7e-2 m^5 K^{3.5}

    With ni = ne / Z for quasi-neutrality:
        kappa = C * ne^2 * Z * g_ff / Te^{3.5}

    Reference: NRL Plasma Formulary, Kramers opacity.

    Args:
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].
        Z: Ion charge state.
        gaunt_factor: Gaunt factor (default 1.2).

    Returns:
        Absorption opacity [1/m].
    """
    Te_safe = np.maximum(Te, 1000.0)  # Floor at 1000 K — cold gas has negligible opacity
    ne_safe = np.maximum(ne, 0.0)

    # Kramers free-free opacity coefficient in SI
    # kappa_ff = C_ff * ne * ni * Z^2 * g_ff / Te^3.5
    # where ni = ne / Z, so kappa_ff = C_ff * ne^2 * Z * g_ff / Te^3.5
    # C_ff ~ 3.7e8 [CGS: cm^5 K^3.5] = 3.7e8 * (1e-2)^5 [SI: m^5 K^3.5] = 3.7e-2
    C_ff = 3.7e-2  # m^5 K^{3.5}
    kappa = C_ff * gaunt_factor * Z * ne_safe * ne_safe / Te_safe**3.5

    # Floor at very small opacity (optically thin limit)
    kappa = np.maximum(kappa, 1e-20)

    # Cap opacity to prevent numerical overflow (inf/nan) in diffusion solver.
    # With the Te floor at 1000 K above, opacity stays physical.
    # This cap only guards against extreme edge cases (e.g., ne > 1e28).
    kappa = np.minimum(kappa, 1e30)

    return kappa


def fld_step(
    E_rad: np.ndarray,
    Te: np.ndarray,
    ne: np.ndarray,
    dx: float,
    dt: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    brem_power: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance radiation energy density by one FLD timestep.

    Solves:
        dE_r/dt = div(D * grad(E_r)) + S_emission - kappa_a * c * E_r

    with explicit forward Euler and sub-cycling for stability.
    The diffusion coefficient is: D = c * lambda(R) / kappa

    Args:
        E_rad: Radiation energy density [J/m^3], shape (nx, ny, nz).
        Te: Electron temperature [K].
        ne: Electron number density [m^-3].
        dx: Grid spacing [m].
        dt: Outer timestep [s].
        Z: Ion charge state.
        gaunt_factor: Bremsstrahlung gaunt factor.
        brem_power: Pre-computed bremsstrahlung power [W/m^3] (optional).
                    If None, computed from ne, Te.

    Returns:
        Tuple of (E_rad_new, Q_absorbed) where:
            E_rad_new: Updated radiation energy density [J/m^3].
            Q_absorbed: Net radiation energy absorbed by matter [W/m^3]
                        (negative = net emission / cooling).
    """
    # Compute opacity
    kappa = compute_rosseland_opacity(ne, Te, Z)

    # Emission source: bremsstrahlung power
    if brem_power is None:
        from dpf.radiation.bremsstrahlung import bremsstrahlung_power
        brem_power = bremsstrahlung_power(ne, Te, Z, gaunt_factor)

    # Compute gradient of E_rad for flux limiter
    ndim = E_rad.ndim
    grad_E_mag = np.zeros_like(E_rad)
    for axis in range(ndim):
        if E_rad.shape[axis] < 3:
            continue  # Skip degenerate axes (e.g., theta=1 in cylindrical)
        grad_comp = np.gradient(E_rad, dx, axis=axis, edge_order=2)
        grad_E_mag += grad_comp**2
    grad_E_mag = np.sqrt(grad_E_mag)

    # Knudsen number R = |grad(E)| / (kappa * E)
    E_safe = np.maximum(E_rad, 1e-30)
    R = grad_E_mag / (kappa * E_safe)

    # Flux limiter
    lam = levermore_pomraning_limiter(R)

    # Diffusion coefficient D = c * lambda / kappa
    D = c_light * lam / np.maximum(kappa, 1e-30)

    # Sub-cycle for stability: dt_diff < dx^2 / (2 * ndim * D_max)
    D_max = float(np.max(D))
    if D_max > 0:
        dt_diff = dx * dx / (2.0 * ndim * D_max)
    else:
        dt_diff = dt

    n_sub = max(1, int(np.ceil(dt / dt_diff)))
    n_sub = min(n_sub, 1000)  # Safety cap
    dt_sub = dt / n_sub

    # Track total absorption for coupling back to matter
    Q_total = np.zeros_like(E_rad)
    E_new = E_rad.copy()

    for _ in range(n_sub):
        # Diffusion: div(D * grad(E))
        div_flux = np.zeros_like(E_new)
        for axis in range(ndim):
            # Skip degenerate axes (e.g., theta=1 in cylindrical)
            n = E_new.shape[axis]
            if n < 3:
                continue
            # Face-centered D (average of neighbors)
            # For axis=0: D_{i+1/2} = 0.5*(D_i + D_{i+1})

            # Build slices for i and i+1
            sl_c = [slice(None)] * ndim  # center slice
            sl_p = [slice(None)] * ndim  # plus-1 slice
            sl_m = [slice(None)] * ndim  # minus-1 slice

            sl_c[axis] = slice(1, n - 1)
            sl_p[axis] = slice(2, n)
            sl_m[axis] = slice(0, n - 2)

            # D at faces
            D_plus = 0.5 * (D[tuple(sl_c)] + D[tuple(sl_p)])
            D_minus = 0.5 * (D[tuple(sl_c)] + D[tuple(sl_m)])

            # Flux: F_{i+1/2} = D_{i+1/2} * (E_{i+1} - E_i) / dx
            flux_plus = D_plus * (E_new[tuple(sl_p)] - E_new[tuple(sl_c)]) / dx
            flux_minus = D_minus * (E_new[tuple(sl_c)] - E_new[tuple(sl_m)]) / dx

            # div(flux) = (F_{i+1/2} - F_{i-1/2}) / dx
            div_flux[tuple(sl_c)] += (flux_plus - flux_minus) / dx

        # Absorption: -kappa * c * E
        absorption = kappa * c_light * E_new

        # Source: bremsstrahlung emission into radiation field
        emission = brem_power

        # Update
        dE = dt_sub * (div_flux + emission - absorption)
        E_new = E_new + dE
        E_new = np.maximum(E_new, 0.0)  # Floor at zero

        # Track net absorbed energy (absorption - emission)
        Q_total += dt_sub * (absorption - emission)

    # Average power over full dt
    Q_absorbed = Q_total / max(dt, 1e-30)

    return E_new, Q_absorbed


def apply_radiation_transport(
    state: dict[str, np.ndarray],
    dx: float,
    dt: float,
    Z: float = 1.0,
    gaunt_factor: float = 1.2,
    Te_floor: float = 1.0,
) -> dict[str, np.ndarray]:
    """Apply full radiation step: bremsstrahlung + FLD transport.

    This is the top-level interface called by the engine. It:
    1. Computes bremsstrahlung emission
    2. Advances radiation energy via FLD
    3. Updates electron temperature from radiation-matter coupling

    If 'E_rad' is not in the state dict, it is initialized from LTE.

    Args:
        state: Simulation state dict (must contain 'Te', 'rho', and optionally 'E_rad').
        dx: Grid spacing [m].
        dt: Timestep [s].
        Z: Ion charge state.
        gaunt_factor: Bremsstrahlung gaunt factor.
        Te_floor: Minimum electron temperature [K].

    Returns:
        Updated state dict with modified 'Te' and 'E_rad'.
    """
    from dpf.constants import m_d
    from dpf.radiation.bremsstrahlung import bremsstrahlung_power

    Te = state["Te"]
    rho = state["rho"]
    ne = Z * rho / m_d  # Electron density assuming Z ionization

    # Skip FLD when plasma is too cold — radiation transport is negligible
    # below ~10,000 K (< 1 eV) and the opacity formula produces overflow.
    Te_max = float(np.max(Te))
    if Te_max < 1e4:
        # Cold plasma: no significant radiation. Return state unchanged.
        new_state = dict(state)
        if "E_rad" not in new_state:
            new_state["E_rad"] = np.zeros_like(Te)
        return new_state

    # Initialize radiation energy if not present
    if "E_rad" not in state:
        state["E_rad"] = compute_radiation_energy(Te, ne)

    E_rad = state["E_rad"]

    # Compute bremsstrahlung power
    P_brem = bremsstrahlung_power(ne, Te, Z, gaunt_factor)

    # FLD step
    E_rad_new, Q_absorbed = fld_step(
        E_rad=E_rad,
        Te=Te,
        ne=ne,
        dx=dx,
        dt=dt,
        Z=Z,
        gaunt_factor=gaunt_factor,
        brem_power=P_brem,
    )

    # Update electron temperature from radiation-matter coupling.
    #
    # Q_absorbed = (absorption - emission) [W/m^3]
    #   positive: radiation absorbed by matter > emission (net heating)
    #   negative: emission > absorption (net cooling = bremsstrahlung losses)
    #
    # Matter energy change: dE_matter = Q_absorbed * dt
    # Temperature: dTe = Q_absorbed / (1.5 * ne * k_B)   [K/s] * dt
    dTe = Q_absorbed * dt / (1.5 * np.maximum(ne, 1e-10) * k_B)
    Te_new = Te + dTe
    Te_new = np.maximum(Te_new, Te_floor)

    # Sanitize
    Te_new = np.where(np.isfinite(Te_new), Te_new, Te)

    state["Te"] = Te_new
    state["E_rad"] = E_rad_new

    return state
