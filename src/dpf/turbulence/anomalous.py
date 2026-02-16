"""Anomalous resistivity from current-driven micro-instabilities.

Implements three threshold models for anomalous resistivity in dense
plasma focus devices:

1. **Ion-acoustic** (default): v_d > v_ti — the electron drift velocity
   exceeds the ion thermal speed. This is the correct threshold for
   ion-acoustic turbulence-driven anomalous resistivity.

2. **LHDI** (lower-hybrid drift instability): v_d > (m_e/m_i)^{1/4} * v_ti.
   Lower threshold than ion-acoustic; triggers first in DPF sheaths.
   Important for the m=0 instability in the pinch column.

3. **Buneman (classic)**: v_d > v_te — the electron drift velocity exceeds
   the electron thermal speed. This is the true Buneman instability; it
   has a much higher threshold than ion-acoustic.

When the threshold is exceeded, collective plasma turbulence enhances
the effective resistivity far above the Spitzer value:

    eta_anom = alpha * m_e * omega_pe / (n_e * e^2)

where alpha ~ 0.01-0.1 is a turbulence parameter.

References:
    Buneman, Phys. Rev. 115:503 (1959) — Buneman instability threshold
    Sagdeev, Rev. Plasma Phys. 4:23 (1966) — anomalous resistivity
    Haines, Plasma Phys. Control. Fusion 53:093001 (2011) — DPF review
    Davidson & Gladd, Phys. Fluids 18:1327 (1975) — LHDI threshold
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import njit

from dpf.constants import e, epsilon_0, k_B, m_e, m_p


@njit(cache=True)
def electron_drift_velocity(
    J: np.ndarray,
    ne: np.ndarray,
) -> np.ndarray:
    """Compute electron drift velocity magnitude from current density.

    v_d = |J| / (n_e * e)

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].

    Returns:
        Electron drift velocity [m/s].
    """
    return np.abs(J) / np.maximum(ne * e, 1e-300)


@njit(cache=True)
def ion_thermal_speed(
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Compute ion thermal speed.

    v_ti = sqrt(k_B * Ti / m_i)

    Args:
        Ti: Ion temperature [K].
        mi: Ion mass [kg] (default: proton mass).

    Returns:
        Ion thermal speed [m/s].
    """
    return np.sqrt(k_B * np.maximum(Ti, 0.0) / mi)


@njit(cache=True)
def electron_thermal_speed(Te: np.ndarray) -> np.ndarray:
    """Compute electron thermal speed.

    v_te = sqrt(k_B * Te / m_e)

    Args:
        Te: Electron temperature [K].

    Returns:
        Electron thermal speed [m/s].
    """
    return np.sqrt(k_B * np.maximum(Te, 0.0) / m_e)


@njit(cache=True)
def plasma_frequency(ne: np.ndarray) -> np.ndarray:
    """Compute electron plasma frequency.

    omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))

    Args:
        ne: Electron number density [m^-3].

    Returns:
        Electron plasma frequency [rad/s].
    """
    return np.sqrt(np.maximum(ne, 0.0) * e**2 / (epsilon_0 * m_e))


# ---------------------------------------------------------------------------
# Threshold functions
# ---------------------------------------------------------------------------


@njit(cache=True)
def ion_acoustic_threshold(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Check ion-acoustic instability threshold: v_d > v_ti.

    This is the correct threshold for ion-acoustic turbulence-driven
    anomalous resistivity. Previously mislabeled as "Buneman" in this
    codebase (the Buneman threshold is v_d > v_te, much higher).

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].
        mi: Ion mass [kg].

    Returns:
        Boolean array: True where instability is active.

    References:
        Sagdeev, Rev. Plasma Phys. 4:23 (1966).
    """
    v_d = electron_drift_velocity(J, ne)
    v_ti = ion_thermal_speed(Ti, mi)
    return v_d > v_ti


def buneman_threshold(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Deprecated alias for ion_acoustic_threshold.

    The code checks v_d > v_ti (ion-acoustic), NOT the true Buneman
    threshold v_d > v_te. Use ``ion_acoustic_threshold`` instead.
    For the true Buneman threshold, use ``buneman_classic_threshold``.

    .. deprecated::
        Use ``ion_acoustic_threshold`` instead.
    """
    warnings.warn(
        "buneman_threshold is deprecated. The code checks v_d > v_ti "
        "(ion-acoustic instability), not the true Buneman threshold "
        "(v_d > v_te). Use ion_acoustic_threshold() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ion_acoustic_threshold(J, ne, Ti, mi)


@njit(cache=True)
def lhdi_threshold(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    mi: float = m_p,
) -> np.ndarray:
    """Check lower-hybrid drift instability (LHDI) threshold.

    v_d > (m_e / m_i)^{1/4} * v_ti

    The LHDI threshold is LOWER than the ion-acoustic threshold by a
    factor of (m_e/m_i)^{1/4}. For deuterium:
        (m_e / (2*m_p))^{1/4} ~ 0.129

    This means LHDI triggers before ion-acoustic instability and is
    important for the m=0 instability in DPF pinch columns.

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].
        mi: Ion mass [kg].

    Returns:
        Boolean array: True where LHDI is active.

    References:
        Davidson & Gladd, Phys. Fluids 18:1327 (1975).
        Huba et al., Phys. Fluids B 5:3779 (1993).
    """
    v_d = electron_drift_velocity(J, ne)
    v_ti = ion_thermal_speed(Ti, mi)
    # LHDI factor: (m_e / m_i)^{1/4}
    factor = (m_e / mi) ** 0.25
    return v_d > factor * v_ti


@njit(cache=True)
def buneman_classic_threshold(
    J: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
) -> np.ndarray:
    """Check true Buneman instability threshold: v_d > v_te.

    The classic Buneman threshold requires the electron drift velocity
    to exceed the electron thermal speed. This is much higher than the
    ion-acoustic threshold (by a factor of sqrt(m_i/m_e) ~ 43 for D).

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Te: Electron temperature [K].

    Returns:
        Boolean array: True where Buneman instability is active.

    References:
        Buneman, Phys. Rev. 115:503 (1959).
    """
    v_d = electron_drift_velocity(J, ne)
    v_te = electron_thermal_speed(Te)
    return v_d > v_te


# ---------------------------------------------------------------------------
# Core anomalous resistivity computation (njit, internal)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _compute_eta_anom(
    ne: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute anomalous resistivity magnitude (without threshold mask).

    eta_anom = alpha * m_e * omega_pe / (n_e * e^2)

    Args:
        ne: Electron number density [m^-3].
        alpha: Turbulence parameter.

    Returns:
        Anomalous resistivity [Ohm*m].
    """
    omega_pe = plasma_frequency(ne)
    return alpha * m_e * omega_pe / np.maximum(ne * e**2, 1e-300)


@njit(cache=True)
def anomalous_resistivity(
    J: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    alpha: float = 0.05,
    mi: float = m_p,
) -> np.ndarray:
    """Compute anomalous resistivity using ion-acoustic threshold.

    When the electron drift velocity exceeds the ion thermal speed,
    the anomalous resistivity is:

        eta_anom = alpha * m_e * omega_pe / (n_e * e^2)

    Below threshold, eta_anom = 0.

    Args:
        J: Current density magnitude [A/m^2].
        ne: Electron number density [m^-3].
        Ti: Ion temperature [K].
        alpha: Turbulence parameter (0.01-0.1, default 0.05).
        mi: Ion mass [kg].

    Returns:
        Anomalous resistivity [Ohm*m].
    """
    result = np.zeros_like(J)

    v_d = electron_drift_velocity(J, ne)
    v_ti = ion_thermal_speed(Ti, mi)
    omega_pe = plasma_frequency(ne)

    for idx in np.ndindex(J.shape):
        if v_d[idx] > v_ti[idx]:
            ne_local = ne[idx]
            if ne_local > 0.0:
                result[idx] = alpha * m_e * omega_pe[idx] / np.maximum(ne_local * e**2, 1e-300)

    return result


@njit(cache=True)
def anomalous_resistivity_scalar(
    J_mag: float,
    ne_val: float,
    Ti_val: float,
    alpha: float = 0.05,
    mi: float = m_p,
) -> float:
    """Scalar version of anomalous resistivity for volume-averaged values.

    Args:
        J_mag: Current density magnitude [A/m^2].
        ne_val: Electron number density [m^-3].
        Ti_val: Ion temperature [K].
        alpha: Turbulence parameter (0.01-0.1, default 0.05).
        mi: Ion mass [kg].

    Returns:
        Anomalous resistivity [Ohm*m].
    """
    if ne_val <= 0.0:
        return 0.0

    v_d = abs(J_mag) / max(ne_val * e, 1e-300)
    v_ti = (k_B * max(Ti_val, 0.0) / mi) ** 0.5

    if v_d <= v_ti:
        return 0.0

    omega_pe = (ne_val * e**2 / (epsilon_0 * m_e)) ** 0.5
    return alpha * m_e * omega_pe / max(ne_val * e**2, 1e-300)


@njit(cache=True)
def total_resistivity(
    eta_spitzer: np.ndarray,
    eta_anomalous: np.ndarray,
) -> np.ndarray:
    """Combine Spitzer and anomalous resistivity.

    eta_total = eta_spitzer + eta_anomalous

    The anomalous contribution is additive (parallel scattering mechanisms).

    Args:
        eta_spitzer: Spitzer (collisional) resistivity [Ohm*m].
        eta_anomalous: Anomalous (turbulent) resistivity [Ohm*m].

    Returns:
        Total resistivity [Ohm*m].
    """
    return eta_spitzer + eta_anomalous


# ---------------------------------------------------------------------------
# Vectorized field computation (dispatched by threshold model)
# ---------------------------------------------------------------------------


def anomalous_resistivity_field(
    J_mag: np.ndarray,
    ne: np.ndarray,
    Ti: np.ndarray,
    alpha: float = 0.05,
    mi: float = m_p,
    threshold_model: str = "ion_acoustic",
    Te: np.ndarray | None = None,
) -> np.ndarray:
    """Compute anomalous resistivity field from spatially-resolved J, ne, Ti.

    Vectorized version that dispatches to the appropriate threshold model.

    Args:
        J_mag: Current density magnitude [A/m^2], any shape.
        ne: Electron number density [m^-3], same shape.
        Ti: Ion temperature [K], same shape.
        alpha: Turbulence parameter (0.01-0.1, default 0.05).
        mi: Ion mass [kg].
        threshold_model: Threshold model to use:
            ``"ion_acoustic"`` — v_d > v_ti (default, most common for DPF).
            ``"lhdi"`` — v_d > (m_e/m_i)^{1/4} * v_ti (lower threshold).
            ``"buneman_classic"`` — v_d > v_te (highest threshold).
        Te: Electron temperature [K], required for ``"buneman_classic"``.

    Returns:
        Anomalous resistivity field [Ohm*m], same shape as J_mag.

    Raises:
        ValueError: If threshold_model is ``"buneman_classic"`` and Te is None,
            or if threshold_model is unknown.
    """
    if threshold_model == "ion_acoustic":
        mask = ion_acoustic_threshold(J_mag, ne, Ti, mi)
    elif threshold_model == "lhdi":
        mask = lhdi_threshold(J_mag, ne, Ti, mi)
    elif threshold_model == "buneman_classic":
        if Te is None:
            raise ValueError(
                "Te (electron temperature) is required for buneman_classic "
                "threshold model."
            )
        mask = buneman_classic_threshold(J_mag, ne, Te)
    else:
        raise ValueError(
            f"Unknown threshold_model '{threshold_model}'. "
            "Options: 'ion_acoustic', 'lhdi', 'buneman_classic'."
        )

    eta_anom = _compute_eta_anom(ne, alpha)
    return np.where(mask, eta_anom, 0.0)


@njit(cache=True)
def total_resistivity_scalar(
    eta_spitzer: float,
    eta_anomalous: float,
) -> float:
    """Scalar version of total resistivity.

    Args:
        eta_spitzer: Spitzer resistivity [Ohm*m].
        eta_anomalous: Anomalous resistivity [Ohm*m].

    Returns:
        Total resistivity [Ohm*m].
    """
    return eta_spitzer + eta_anomalous


def lhdi_factor(mi: float = m_p) -> float:
    """Return the LHDI threshold factor (m_e/m_i)^{1/4}.

    For deuterium (m_i = 2*m_p): factor ~ 0.129.

    Args:
        mi: Ion mass [kg].

    Returns:
        Dimensionless LHDI threshold factor.
    """
    return (m_e / mi) ** 0.25
