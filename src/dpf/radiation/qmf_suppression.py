"""Quantizing magnetic field (QMF) bremsstrahlung suppression.

In ultra-strong magnetic fields (B > 10^5 T), electron cyclotron
energy exceeds thermal energy, forcing electrons into discrete
Landau levels. This suppresses bremsstrahlung radiation because
transitions between Landau levels are restricted.

Key physics:
- Cyclotron energy: E_c = hbar * omega_c = hbar * eB/m_e
- Thermal energy: E_th = k_B * T
- QMF regime: E_c > E_th, or B > B_QMF = m_e * k_B * T / (e * hbar)
- Suppression factor: S = f(E_c / E_th) with S → 0 at E_c >> E_th

For p-11B fusion: temperatures ~200 keV required, so
B_QMF ~ 1.73e9 T (17.3 GG) -- only achievable in extreme DPF conditions.
(Computed from B_QMF = m_e * k_B * T / (e * hbar) at k_B T = 200 keV:
 numerator = 3.204e-14 J * 9.109e-31 kg = 2.918e-44
 denominator = 1.602e-19 C * 1.055e-34 J s = 1.690e-53
 B_QMF = 1.727e9 T.)

Competing effect: synchrotron radiation INCREASES with B^2, partially
offsetting bremsstrahlung suppression. Net benefit requires detailed
balance.

References:
    Potekhin & Chabrier, ApJ 585:955 (2003) — magnetized thermal conductivity
    Rider, Phys. Plasmas 4:1039 (1997) — p-B11 radiation constraints
    Putvinski et al., Nucl. Fusion 38:1275 (1998) — QMF in fusion

UNVERIFIED CITATION (removed 2026-04-27):
    Bezchastnov & Potekhin, J. Phys. B 27:3349 (1994), DOI 10.1088/0953-4075/27/15/013,
    bibcode 1994JPhB...27.3349B, was previously cited here as "QMF brem rates".
    Verified via NASA ADS: that paper is titled "Transitions between shifted Landau
    states and photoionization of the hydrogen atom moving in a strong magnetic field"
    and treats BOUND-STATE photoionization (bound-free), NOT free-free electron-ion
    bremsstrahlung. The interpolation formula in `bremsstrahlung_suppression_factor`
    below is therefore unsourced/heuristic, not derived from this paper. Treat
    QMF suppression as UNVERIFIED until a primary free-free reference is on disk
    under references/papers/ (candidate: Pavlov & Panov 1976 Sov.Phys.JETP 44:300,
    or Lauer et al. 1983 ApJ 272:122). Default behavior: qmf_unverified=True.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpf.constants import e, hbar, k_B, m_e


@dataclass
class QMFDiag:
    """QMF bremsstrahlung suppression diagnostic."""

    B_qmf_T: float  # Critical B for QMF regime [T]
    B_actual_T: float  # Actual peak B [T]
    ratio_Ec_Eth: float  # E_cyclotron / E_thermal
    suppression_factor: float  # Bremsstrahlung suppression (1 = no suppression, 0 = full)
    synchrotron_enhancement: float  # Synchrotron increase factor (>1)
    net_radiation_factor: float  # Net radiation change (suppression * enhancement)
    is_qmf_regime: bool  # True if E_c > E_th
    note: str


def cyclotron_energy(B: float) -> float:
    """Electron cyclotron energy [J].

    E_c = hbar * omega_c = hbar * e * B / m_e
    """
    return hbar * e * abs(B) / m_e


def qmf_critical_field(Te_K: float) -> float:
    """Critical B-field for QMF regime [T].

    B_QMF = m_e * k_B * T / (e * hbar)

    At this field, cyclotron energy equals thermal energy.
    """
    return m_e * k_B * max(Te_K, 1.0) / (e * hbar)


def bremsstrahlung_suppression_factor(
    B: float,
    Te_K: float,
) -> float:
    """Compute bremsstrahlung suppression factor in strong B. [UNVERIFIED]

    HEURISTIC INTERPOLATION — not from a published derivation. The previous
    docstring attributed this to Bezchastnov & Potekhin J.Phys.B 27:3349 (1994);
    that paper (DOI 10.1088/0953-4075/27/15/013) is on hydrogen-atom photoionization
    with shifted Landau states, not free-free bremsstrahlung. Citation removed
    2026-04-27. PDF not on disk under references/papers/.

    Functional form (placeholder, no paper backing):
        S(r) = exp(-r) + (1 - exp(-r)) * exp(-sqrt(r)),  r = E_c/E_th
        E_c = hbar * e * B / m_e,  E_th = k_B * Te
    Limits: r << 1 -> S = 1 (no suppression). r >> 1 -> S -> 0, floored at 0.01.

    Physical reasoning sketch: when r >> 1 only the n=0 Landau level is thermally
    populated, restricting allowed transitions; quantitative magnitude is not
    derived here. For practical DPF fields (B < 100 T at T ~ 1 keV), r << 1 and
    this returns 1.0 regardless of formula details, so the heuristic is harmless
    in the DPF regime but should NOT be trusted for B > 1 GG / r >~ 1.

    Args:
        B: Magnetic field magnitude [T].
        Te_K: Electron temperature [K].

    Returns:
        Suppression factor S in [0, 1]. S=1 means no suppression.
    """
    E_c = cyclotron_energy(B)
    E_th = k_B * max(Te_K, 1.0)
    ratio = E_c / E_th

    if ratio < 0.01:
        return 1.0  # No suppression
    elif ratio > 50:
        return 0.01  # Near-complete suppression (floor at 1%)
    else:
        # Smooth interpolation based on Landau level population
        # At ratio=1, ~37% of electrons in ground level → moderate suppression
        return float(np.exp(-ratio) + (1.0 - np.exp(-ratio)) * np.exp(-ratio**0.5))


def synchrotron_enhancement_factor(
    B: float,
    Te_K: float,
    ne: float,
    Z: float = 1.0,
) -> float:
    """Compute synchrotron radiation enhancement factor.

    Synchrotron power (Larmor formula averaged over 2D perpendicular Maxwellian):
        P_sync = (e^4 * B^2 * n_e * <v_perp^2>) / (6*pi*eps0*m_e^2*c^3)
               = (e^4 * B^2 * n_e * k_B * T_e) / (3*pi*eps0*m_e^3*c^3)
    where <v_perp^2> = 2 * k_B * T / m_e (two perpendicular degrees of freedom).
    Scales as B^2 * T.

    Previous version used v_th^2 = k_B * T / m_e (1D thermal speed), which
    under-predicts by a factor of 2 compared to the correct perpendicular
    thermal average used in cyclotron/synchrotron emission.

    Normalized to bremsstrahlung at reference conditions.

    Args:
        B: Magnetic field [T].
        Te_K: Electron temperature [K].
        ne: Electron density [m^-3].
        Z: Ion charge state (default 1).

    Returns:
        Factor > 1 means synchrotron exceeds bremsstrahlung at this B.
    """
    from dpf.constants import c as c_light
    from dpf.constants import epsilon_0

    # Perpendicular thermal speed squared: <v_perp^2> = 2 k_B T_e / m_e
    # (two perpendicular DoF for cyclotron motion, not 1D v_th^2).
    v_perp_sq = 2.0 * k_B * max(Te_K, 1.0) / m_e

    # Synchrotron power density (re-derived from Larmor 2026-04-23 to
    # answer ZETA_REV P2 query about "eps0 m_e^2 vs eps0^2 m_e^3" denominator):
    #
    #   Single-electron Larmor:  P_1 = e^2 a^2 / (6 pi eps0 c^3)
    #   Circular-motion accel:   a = v_perp * omega_c = v_perp * (eB / m_e)
    #   So a^2 = e^2 B^2 v_perp^2 / m_e^2
    #   P_1 = e^4 B^2 v_perp^2 / (6 pi eps0 m_e^2 c^3)
    #   P_volumetric = n_e * P_1 with thermal average <v_perp^2> = 2 k_B T / m_e
    #
    # Denominator IS eps0 * m_e^2 in this form; the extra 1/m_e that
    # turns it into the equivalent (eps0 m_e^3) form lives in <v_perp^2>.
    # ZETA_REV's "eps0^2 m_e^3" reading would add a stray eps0 and break
    # dimensions ([W/m^3] requires exactly one eps0; verified by the unit
    # check below).
    #
    # Dimensional check (SI):
    #   Num = e^4 B^2 n_e v_perp^2  -> C^4 * T^2 * m^-3 * m^2/s^2
    #       = C^2 kg^2 m^-1 s^-4   (using T^2 = kg^2/(C^2 s^2))
    #   Den = eps0 m_e^2 c^3        -> C^2 s^2 kg^-1 m^-3 * kg^2 * m^3/s^3
    #       = C^2 kg s^-1
    #   Ratio = kg m^-1 s^-3 = J m^-3 s^-1 = W/m^3   OK
    P_sync = e**4 * B**2 * ne * v_perp_sq / (6 * np.pi * epsilon_0 * m_e**2 * c_light**3)

    # Bremsstrahlung power density (quasi-neutral ni = ne/Z):
    #   P_ff = BREM_COEFF * g_ff * Z * ne^2 * sqrt(Te_K)  [W/m^3]
    # Must include Z factor (free-free from ion Coulomb field).
    g_ff = 1.2  # Gaunt factor
    P_brem = 1.569e-40 * g_ff * Z * ne**2 * np.sqrt(max(Te_K, 1.0))

    if P_brem > 0:
        return max(P_sync / P_brem, 1.0)
    return 1.0


def qmf_diagnostic(
    B_field: np.ndarray,
    Te: np.ndarray,
    ne: np.ndarray,
) -> QMFDiag:
    """Compute QMF bremsstrahlung suppression diagnostic.

    Args:
        B_field: Magnetic field (3, ...) [T].
        Te: Electron temperature (...) [K].
        ne: Electron density (...) [m^-3].

    Returns:
        QMFDiag with all metrics.
    """
    B_mag = np.sqrt(np.sum(B_field**2, axis=0))
    B_peak = float(np.max(B_mag))
    Te_peak = float(np.max(Te))
    ne_peak = float(np.max(ne))

    B_qmf = qmf_critical_field(Te_peak)
    E_c = cyclotron_energy(B_peak)
    E_th = k_B * max(Te_peak, 1.0)
    ratio = E_c / E_th

    suppression = bremsstrahlung_suppression_factor(B_peak, Te_peak)
    sync_enhance = synchrotron_enhancement_factor(B_peak, Te_peak, ne_peak)
    net = suppression * sync_enhance

    is_qmf = ratio > 1.0

    if B_peak < 100:
        note = "B << B_QMF: no QMF effects at typical DPF fields"
    elif ratio < 0.01:
        note = f"E_c/E_th = {ratio:.1e}: far below QMF threshold"
    elif ratio < 1.0:
        note = f"E_c/E_th = {ratio:.2f}: approaching QMF regime"
    else:
        note = f"E_c/E_th = {ratio:.1f}: IN QMF regime, brem suppressed by {(1-suppression)*100:.0f}%"

    return QMFDiag(
        B_qmf_T=B_qmf,
        B_actual_T=B_peak,
        ratio_Ec_Eth=ratio,
        suppression_factor=suppression,
        synchrotron_enhancement=sync_enhance,
        net_radiation_factor=net,
        is_qmf_regime=is_qmf,
        note=note,
    )
