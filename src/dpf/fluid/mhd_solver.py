"""Hall MHD solver with WENO5-Z reconstruction and Dedner divergence cleaning.

.. warning::

    **Teaching / fallback engine only.**  This Python MHD solver evolves
    **pressure** (dp/dt) rather than **total energy** (dE/dt).  This is a
    non-conservative formulation that violates the Rankine-Hugoniot jump
    conditions at shocks and can produce incorrect post-shock states.

    For production simulations and any work requiring quantitative accuracy,
    use one of the conservative backends:

    - ``backend="metal"``  — Metal GPU solver (HLLD + WENO5 + SSP-RK3, conservative)
    - ``backend="athena"`` — Athena++ C++ solver (PPM + HLLD + CT, conservative)

    The Python engine remains useful for:
    - Quick prototyping and debugging (no compilation needed)
    - Teaching / demonstrating MHD algorithms
    - Unit testing individual physics modules
    - Small-grid parameter scans where accuracy is not critical

Merges dpf2's hall_mhd_solver structure with DPF_AI's validated physics kernels:
- WENO5-Z reconstruction (Borges et al. 2008 weights, Jiang-Shu 1996 polynomials)
- HLLD Riemann solver for ideal MHD (Miyoshi & Kusano 2005, default)
- SSP-RK2/RK3 time integration (selectable, RK3 default)
- Dimension-split flux-based conservative update
- Dedner hyperbolic divergence cleaning for div(B)
- Generalized Ohm's law with Hall term
- Braginskii anisotropic heat flux (operator-split)
- dL_dt estimation from pinch dynamics

**Known limitation (Phase R assessment):**
The pressure equation dp/dt = -gamma*p*div(v) + (gamma-1)*eta*J^2 (line ~1597)
is NOT equivalent to the conservative total energy equation.  At shocks, this
produces incorrect jump conditions.  See the Metal engine
(:mod:`dpf.metal.metal_solver`) for the correct conservative formulation.

The solver operates on a state dictionary with keys:
    rho: density [kg/m^3], shape (nx, ny, nz)
    velocity: velocity [m/s], shape (3, nx, ny, nz)
    pressure: total pressure [Pa], shape (nx, ny, nz)
    B: magnetic field [T], shape (3, nx, ny, nz)
    Te: electron temperature [K], shape (nx, ny, nz)
    Ti: ion temperature [K], shape (nx, ny, nz)
    psi: Dedner cleaning scalar, shape (nx, ny, nz)
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

from dpf.constants import e as e_charge
from dpf.constants import k_B, m_d, mu_0
from dpf.core.bases import CouplingState, PlasmaSolverBase
from dpf.fluid.eos import IdealEOS

logger = logging.getLogger(__name__)

# Default ion mass: deuterium
_DEFAULT_ION_MASS = m_d


# ============================================================
# WENO5-Z reconstruction kernels (Borges et al. 2008)
# FV candidate polynomials + WENO-Z nonlinear weights
# ============================================================

@njit(cache=True)
def _weno5_reconstruct_1d(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WENO5-Z reconstruction for a 1D array at cell interfaces.

    Uses Jiang-Shu (1996) candidate polynomials with WENO-Z nonlinear
    weights (Borges et al. 2008).  The WENO-Z indicator tau5 = |beta0 - beta2|
    provides better accuracy at critical points (where f and f' vanish)
    compared to the classical WENO-JS weights.

    Note: The Python MHD solver uses a hybrid scheme (WENO flux divergence
    for density/momentum + np.gradient for induction/pressure).  FV-style
    candidate polynomials are more stable in this hybrid context than
    pure FD point-value formulas.  The Metal GPU solver uses FD formulas
    because it has a fully conservative formulation.

    Args:
        v: 1D array of cell-centered values, shape (n,).

    Returns:
        (v_left, v_right) at interfaces, shape (n-4,) each.
        Interface index j corresponds to the face between cell j+2 and j+3.

    References:
        Borges R. et al., JCP 227, 3191 (2008) — WENO-Z weights.
        Jiang G.-S. & Shu C.-W., JCP 126, 202 (1996) — WENO5 candidate polynomials.
    """
    n = len(v)
    n_iface = n - 4  # number of interfaces with full stencil
    v_L = np.empty(n_iface)
    v_R = np.empty(n_iface)

    eps = 1e-6  # Smoothness parameter

    # FV ideal weights (Jiang-Shu 1996)
    d0, d1, d2 = 0.1, 0.6, 0.3

    for i in range(2, n - 2):
        idx = i - 2  # output index

        # --- Left-biased reconstruction (v_L at i+1/2) ---
        v0 = v[i - 2]
        v1 = v[i - 1]
        v2 = v[i]
        v3 = v[i + 1]
        v4 = v[i + 2] if i + 2 < n else v[i + 1]

        # FV candidate polynomials (Jiang-Shu 1996)
        p0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
        p1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
        p2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0

        # Smoothness indicators (Jiang-Shu)
        beta0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2 + 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
        beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - v3) ** 2
        beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2

        # WENO-Z weights: tau5 = |beta0 - beta2| (Borges et al. 2008)
        tau5 = abs(beta0 - beta2)
        alpha0 = d0 * (1.0 + (tau5 / (eps + beta0)) ** 2)
        alpha1 = d1 * (1.0 + (tau5 / (eps + beta1)) ** 2)
        alpha2 = d2 * (1.0 + (tau5 / (eps + beta2)) ** 2)
        alpha_sum = alpha0 + alpha1 + alpha2
        if alpha_sum == 0.0:
            alpha_sum = 1e-30

        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum

        v_L[idx] = w0 * p0 + w1 * p1 + w2 * p2

        # --- Right-biased reconstruction (v_R at i+1/2) ---
        # Mirror stencil
        r0 = v[i + 2] if i + 2 < n else v[i + 1]
        r1 = v[i + 1]
        r2 = v[i]
        r3 = v[i - 1]
        r4 = v[i - 2]

        q0 = (2.0 * r0 - 7.0 * r1 + 11.0 * r2) / 6.0
        q1 = (-r1 + 5.0 * r2 + 2.0 * r3) / 6.0
        q2 = (2.0 * r2 + 5.0 * r3 - r4) / 6.0

        gb0 = (13.0 / 12.0) * (r0 - 2.0 * r1 + r2) ** 2 + 0.25 * (r0 - 4.0 * r1 + 3.0 * r2) ** 2
        gb1 = (13.0 / 12.0) * (r1 - 2.0 * r2 + r3) ** 2 + 0.25 * (r1 - r3) ** 2
        gb2 = (13.0 / 12.0) * (r2 - 2.0 * r3 + r4) ** 2 + 0.25 * (3.0 * r2 - 4.0 * r3 + r4) ** 2

        tau5_r = abs(gb0 - gb2)
        a0 = d0 * (1.0 + (tau5_r / (eps + gb0)) ** 2)
        a1 = d1 * (1.0 + (tau5_r / (eps + gb1)) ** 2)
        a2 = d2 * (1.0 + (tau5_r / (eps + gb2)) ** 2)
        a_sum = a0 + a1 + a2
        if a_sum == 0.0:
            a_sum = 1e-30

        v_R[idx] = (a0 / a_sum) * q0 + (a1 / a_sum) * q1 + (a2 / a_sum) * q2

    return v_L, v_R


# ============================================================
# HLL Riemann flux for ideal MHD (1D, per-interface)
# ============================================================

@njit(cache=True)
def _hll_flux_1d_core(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    u_L: np.ndarray,
    u_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn_L: np.ndarray,
    Bn_R: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HLL approximate Riemann solver core (Numba-accelerated).

    Returns tuple of (mass_flux, momentum_flux, energy_flux).
    """
    # Fast magnetosonic speed estimate
    B_sq_L = Bn_L ** 2
    B_sq_R = Bn_R ** 2
    a2_L = gamma * np.maximum(p_L, 0.0) / np.maximum(rho_L, 1e-30)
    a2_R = gamma * np.maximum(p_R, 0.0) / np.maximum(rho_R, 1e-30)
    va2_L = B_sq_L / (mu_0 * np.maximum(rho_L, 1e-30))
    va2_R = B_sq_R / (mu_0 * np.maximum(rho_R, 1e-30))
    cf_L = np.sqrt(a2_L + va2_L)
    cf_R = np.sqrt(a2_R + va2_R)

    # Davis wave speed estimates
    S_L = np.minimum(u_L - cf_L, u_R - cf_R)
    S_R = np.maximum(u_L + cf_L, u_R + cf_R)

    # Left and right fluxes
    F_rho_L = rho_L * u_L
    F_rho_R = rho_R * u_R

    ptot_L = p_L + 0.5 * B_sq_L / mu_0
    ptot_R = p_R + 0.5 * B_sq_R / mu_0

    F_mom_L = rho_L * u_L ** 2 + ptot_L
    F_mom_R = rho_R * u_R ** 2 + ptot_R

    e_int_L = p_L / (gamma - 1.0)
    e_int_R = p_R / (gamma - 1.0)
    E_L = e_int_L + 0.5 * rho_L * u_L ** 2 + 0.5 * B_sq_L / mu_0
    E_R = e_int_R + 0.5 * rho_R * u_R ** 2 + 0.5 * B_sq_R / mu_0

    F_ene_L = (E_L + ptot_L) * u_L
    F_ene_R = (E_R + ptot_R) * u_R

    # Conserved quantities
    U_rho_L = rho_L
    U_rho_R = rho_R
    U_mom_L = rho_L * u_L
    U_mom_R = rho_R * u_R

    # HLL flux
    denom = np.maximum(S_R - S_L, 1e-30)

    F_rho = (S_R * F_rho_L - S_L * F_rho_R + S_L * S_R * (U_rho_R - U_rho_L)) / denom
    F_mom = (S_R * F_mom_L - S_L * F_mom_R + S_L * S_R * (U_mom_R - U_mom_L)) / denom
    F_ene = (S_R * F_ene_L - S_L * F_ene_R + S_L * S_R * (E_R - E_L)) / denom

    return F_rho, F_mom, F_ene


def _hll_flux_1d(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    u_L: np.ndarray,
    u_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn_L: np.ndarray,
    Bn_R: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    """HLL approximate Riemann solver — dict wrapper for compatibility."""
    F_rho, F_mom, F_ene = _hll_flux_1d_core(
        rho_L, rho_R, u_L, u_R, p_L, p_R, Bn_L, Bn_R, gamma,
    )
    return {
        "mass_flux": F_rho,
        "momentum_flux": F_mom,
        "energy_flux": F_ene,
    }


# ============================================================
# HLLD Riemann solver (Miyoshi & Kusano 2005)
# ============================================================

@njit(cache=True)
def _hlld_flux_1d_8comp(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    vn_L: np.ndarray,
    vn_R: np.ndarray,
    vt1_L: np.ndarray,
    vt1_R: np.ndarray,
    vt2_L: np.ndarray,
    vt2_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn: np.ndarray,
    Bt1_L: np.ndarray,
    Bt1_R: np.ndarray,
    Bt2_L: np.ndarray,
    Bt2_R: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Full 8-component HLLD approximate Riemann solver (Numba-accelerated).

    Resolves 4 intermediate states: outer fast shocks + Alfven rotational
    discontinuities + contact.  Returns fluxes for all 7 MHD conserved
    quantities (mass, normal momentum, 2 transverse momenta, energy,
    2 transverse B components).

    Reference: Miyoshi & Kusano, JCP 208, 315 (2005).

    Args:
        rho_L, rho_R: Reconstructed density at interfaces.
        vn_L, vn_R: Normal velocity.
        vt1_L, vt1_R: First transverse velocity.
        vt2_L, vt2_R: Second transverse velocity.
        p_L, p_R: Thermal pressure.
        Bn: Normal magnetic field (continuous across interface).
        Bt1_L, Bt1_R: First transverse B-field.
        Bt2_L, Bt2_R: Second transverse B-field.
        gamma: Adiabatic index.

    Returns:
        (F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2):
        Fluxes at each interface.
    """
    n = len(rho_L)
    F_rho = np.empty(n)
    F_momn = np.empty(n)
    F_momt1 = np.empty(n)
    F_momt2 = np.empty(n)
    F_ene = np.empty(n)
    F_Bt1 = np.empty(n)
    F_Bt2 = np.empty(n)

    epsilon = 1e-20 * mu_0  # Degeneracy guard scaled to code units

    for i in range(n):
        # --- Primitives with floors ---
        rL = max(rho_L[i], 1e-30)
        rR = max(rho_R[i], 1e-30)
        vnL = vn_L[i]
        vnR = vn_R[i]
        vt1L = vt1_L[i]
        vt1R = vt1_R[i]
        vt2L = vt2_L[i]
        vt2R = vt2_R[i]
        pL = max(p_L[i], 0.0)
        pR = max(p_R[i], 0.0)
        bn = Bn[i]
        bt1L = Bt1_L[i]
        bt1R = Bt1_R[i]
        bt2L = Bt2_L[i]
        bt2R = Bt2_R[i]

        bn2 = bn * bn

        # --- Total magnetic pressure on each side ---
        B_sq_L = bn2 + bt1L * bt1L + bt2L * bt2L
        B_sq_R = bn2 + bt1R * bt1R + bt2R * bt2R
        ptL = pL + 0.5 * B_sq_L / mu_0
        ptR = pR + 0.5 * B_sq_R / mu_0

        # --- Fast magnetosonic speeds ---
        a2L = gamma * pL / rL
        a2R = gamma * pR / rR
        va2L = B_sq_L / (mu_0 * rL)
        va2R = B_sq_R / (mu_0 * rR)
        bn2_over_mu0_rL = bn2 / (mu_0 * rL)
        bn2_over_mu0_rR = bn2 / (mu_0 * rR)

        disc_L = (a2L + va2L) * (a2L + va2L) - 4.0 * a2L * bn2_over_mu0_rL
        disc_R = (a2R + va2R) * (a2R + va2R) - 4.0 * a2R * bn2_over_mu0_rR
        cf2L = 0.5 * (a2L + va2L + max(disc_L, 0.0) ** 0.5)
        cf2R = 0.5 * (a2R + va2R + max(disc_R, 0.0) ** 0.5)
        cfL = max(cf2L, 0.0) ** 0.5
        cfR = max(cf2R, 0.0) ** 0.5

        # --- Davis wave speed estimates ---
        SL = min(vnL - cfL, vnR - cfR)
        SR = max(vnL + cfL, vnR + cfR)

        # --- Conserved total energies ---
        v2L = vnL * vnL + vt1L * vt1L + vt2L * vt2L
        v2R = vnR * vnR + vt1R * vt1R + vt2R * vt2R
        eL = pL / (gamma - 1.0) + 0.5 * rL * v2L + 0.5 * B_sq_L / mu_0
        eR = pR / (gamma - 1.0) + 0.5 * rR * v2R + 0.5 * B_sq_R / mu_0

        # --- Contact speed SM (Eq. 38) ---
        denom_SM = rR * (SR - vnR) - rL * (SL - vnL)
        if abs(denom_SM) < 1e-30:
            SM = 0.5 * (vnL + vnR)
        else:
            SM = (rR * vnR * (SR - vnR) - rL * vnL * (SL - vnL) + ptL - ptR) / denom_SM

        # --- Star total pressure (Eq. 41) ---
        pt_star = ptL + rL * (SL - vnL) * (SM - vnL)

        # --- Star densities (Eq. 43) ---
        denom_L = SL - SM
        denom_R = SR - SM
        if abs(denom_L) < 1e-30:
            rho_sL = rL
        else:
            rho_sL = rL * (SL - vnL) / denom_L
        if abs(denom_R) < 1e-30:
            rho_sR = rR
        else:
            rho_sR = rR * (SR - vnR) / denom_R

        rho_sL = max(rho_sL, 1e-30)
        rho_sR = max(rho_sR, 1e-30)

        # --- Star transverse velocities and B (Eqs. 44-45) ---
        denom_star_L = rL * (SL - vnL) * (SL - SM) - bn2 / mu_0
        denom_star_R = rR * (SR - vnR) * (SR - SM) - bn2 / mu_0

        if abs(denom_star_L) < epsilon:
            vt1_sL = vt1L
            vt2_sL = vt2L
            Bt1_sL = bt1L
            Bt2_sL = bt2L
        else:
            vt1_sL = vt1L - bn * bt1L * (SM - vnL) / (mu_0 * denom_star_L)
            vt2_sL = vt2L - bn * bt2L * (SM - vnL) / (mu_0 * denom_star_L)
            num_L = rL * (SL - vnL) * (SL - vnL) - bn2 / mu_0
            Bt1_sL = bt1L * num_L / denom_star_L
            Bt2_sL = bt2L * num_L / denom_star_L

        if abs(denom_star_R) < epsilon:
            vt1_sR = vt1R
            vt2_sR = vt2R
            Bt1_sR = bt1R
            Bt2_sR = bt2R
        else:
            vt1_sR = vt1R - bn * bt1R * (SM - vnR) / (mu_0 * denom_star_R)
            vt2_sR = vt2R - bn * bt2R * (SM - vnR) / (mu_0 * denom_star_R)
            num_R = rR * (SR - vnR) * (SR - vnR) - bn2 / mu_0
            Bt1_sR = bt1R * num_R / denom_star_R
            Bt2_sR = bt2R * num_R / denom_star_R

        # --- Star energies (Eq. 48) ---
        vB_L = vnL * bn + vt1L * bt1L + vt2L * bt2L
        vB_sL = SM * bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL
        if abs(denom_L) < 1e-30:
            e_sL = eL
        else:
            e_sL = ((SL - vnL) * eL - ptL * vnL + pt_star * SM
                    + bn / mu_0 * (vB_L - vB_sL)) / denom_L

        vB_R = vnR * bn + vt1R * bt1R + vt2R * bt2R
        vB_sR = SM * bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR
        if abs(denom_R) < 1e-30:
            e_sR = eR
        else:
            e_sR = ((SR - vnR) * eR - ptR * vnR + pt_star * SM
                    + bn / mu_0 * (vB_R - vB_sR)) / denom_R

        # --- Alfven wave speeds (Eq. 51) ---
        sqrt_rho_sL = max(rho_sL, 0.0) ** 0.5
        sqrt_rho_sR = max(rho_sR, 0.0) ** 0.5
        abs_bn_over_sqrt_mu0 = abs(bn) / mu_0 ** 0.5
        SL_star = SM - abs_bn_over_sqrt_mu0 / max(sqrt_rho_sL, 1e-30)
        SR_star = SM + abs_bn_over_sqrt_mu0 / max(sqrt_rho_sR, 1e-30)

        # --- Double-star states (Eqs. 59-62) ---
        if abs(bn) < epsilon:
            # Degenerate: no Alfven rotation, double-star = star
            vt1_ssL = vt1_sL
            vt2_ssL = vt2_sL
            Bt1_ssL = Bt1_sL
            Bt2_ssL = Bt2_sL
            e_ssL = e_sL
            vt1_ssR = vt1_sR
            vt2_ssR = vt2_sR
            Bt1_ssR = Bt1_sR
            Bt2_ssR = Bt2_sR
            e_ssR = e_sR
        else:
            sign_bn = 1.0 if bn >= 0.0 else -1.0
            denom_ss = sqrt_rho_sL + sqrt_rho_sR
            if denom_ss < 1e-30:
                denom_ss = 1e-30
            sqrt_mu0 = mu_0 ** 0.5
            inv_denom_ss = 1.0 / denom_ss

            vt1_ss = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR
                      + sign_bn * (Bt1_sR - Bt1_sL) / sqrt_mu0) * inv_denom_ss
            vt2_ss = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR
                      + sign_bn * (Bt2_sR - Bt2_sL) / sqrt_mu0) * inv_denom_ss
            Bt1_ss = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL
                      + sign_bn * sqrt_mu0 * sqrt_rho_sL * sqrt_rho_sR
                      * (vt1_sR - vt1_sL)) * inv_denom_ss
            Bt2_ss = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL
                      + sign_bn * sqrt_mu0 * sqrt_rho_sL * sqrt_rho_sR
                      * (vt2_sR - vt2_sL)) * inv_denom_ss

            vB_ssL = SM * bn + vt1_ss * Bt1_ss + vt2_ss * Bt2_ss
            vB_ssR = vB_ssL  # Same by construction

            e_ssL = e_sL - sign_bn * sqrt_rho_sL / sqrt_mu0 * (vB_sL - vB_ssL)
            e_ssR = e_sR + sign_bn * sqrt_rho_sR / sqrt_mu0 * (vB_sR - vB_ssR)

            vt1_ssL = vt1_ss
            vt2_ssL = vt2_ss
            Bt1_ssL = Bt1_ss
            Bt2_ssL = Bt2_ss
            vt1_ssR = vt1_ss
            vt2_ssR = vt2_ss
            Bt1_ssR = Bt1_ss
            Bt2_ssR = Bt2_ss

        # --- Physical fluxes (left and right) ---
        Frho_L = rL * vnL
        Frho_R = rR * vnR
        Fmomn_L = rL * vnL * vnL + ptL - bn2 / mu_0
        Fmomn_R = rR * vnR * vnR + ptR - bn2 / mu_0
        Fmomt1_L = rL * vnL * vt1L - bn * bt1L / mu_0
        Fmomt1_R = rR * vnR * vt1R - bn * bt1R / mu_0
        Fmomt2_L = rL * vnL * vt2L - bn * bt2L / mu_0
        Fmomt2_R = rR * vnR * vt2R - bn * bt2R / mu_0
        Fene_L = (eL + ptL) * vnL - bn * vB_L / mu_0
        Fene_R = (eR + ptR) * vnR - bn * vB_R / mu_0
        FBt1_L = vnL * bt1L - vt1L * bn
        FBt1_R = vnR * bt1R - vt1R * bn
        FBt2_L = vnL * bt2L - vt2L * bn
        FBt2_R = vnR * bt2R - vt2R * bn

        # Note: normal momentum flux includes total pressure ptL
        # but must subtract bn^2/mu_0 because ptL already has it
        # Actually, the standard MHD flux for normal momentum is:
        #   rho*vn*vn + p_total - Bn^2/mu_0
        # But p_total = p + 0.5*(Bn^2+Bt1^2+Bt2^2)/mu_0
        # So the flux = rho*vn*vn + p + (Bt1^2+Bt2^2-Bn^2)/(2*mu_0)
        # We use the pt_star formula in the Riemann solver regions,
        # so recalculate the normal momentum flux correctly:
        Fmomn_L = rL * vnL * vnL + ptL - bn2 / mu_0
        Fmomn_R = rR * vnR * vnR + ptR - bn2 / mu_0
        # Correction: actually the standard form is rho*vn^2 + pt
        # where pt = p + |B|^2/(2*mu_0) already includes Bn^2.
        # The Rankine-Hugoniot jump conditions use pt directly.
        # For HLLD the normal momentum flux is:
        #   F_momn = rho*vn*vn + pt - Bn^2/mu_0
        # = rho*vn^2 + p + (Bt1^2+Bt2^2)/(2*mu_0) - Bn^2/(2*mu_0)
        # This is correct for the MHD equations in conservative form.

        # --- Conserved state vectors for HLL jump formula ---
        # U_L = [rho, rho*vn, rho*vt1, rho*vt2, e_total, Bt1, Bt2]
        U_rho_L = rL
        U_momn_L = rL * vnL
        U_momt1_L = rL * vt1L
        U_momt2_L = rL * vt2L
        U_ene_L = eL
        U_Bt1_L = bt1L
        U_Bt2_L = bt2L

        U_rho_R = rR
        U_momn_R = rR * vnR
        U_momt1_R = rR * vt1R
        U_momt2_R = rR * vt2R
        U_ene_R = eR
        U_Bt1_R = bt1R
        U_Bt2_R = bt2R

        # Star conserved states
        U_rho_sL = rho_sL
        U_momn_sL = rho_sL * SM
        U_momt1_sL = rho_sL * vt1_sL
        U_momt2_sL = rho_sL * vt2_sL
        U_ene_sL = e_sL
        U_Bt1_sL = Bt1_sL
        U_Bt2_sL = Bt2_sL

        U_rho_sR = rho_sR
        U_momn_sR = rho_sR * SM
        U_momt1_sR = rho_sR * vt1_sR
        U_momt2_sR = rho_sR * vt2_sR
        U_ene_sR = e_sR
        U_Bt1_sR = Bt1_sR
        U_Bt2_sR = Bt2_sR

        # Double-star conserved states
        U_rho_ssL = rho_sL
        U_momn_ssL = rho_sL * SM
        U_momt1_ssL = rho_sL * vt1_ssL
        U_momt2_ssL = rho_sL * vt2_ssL
        U_ene_ssL = e_ssL
        U_Bt1_ssL = Bt1_ssL
        U_Bt2_ssL = Bt2_ssL

        U_rho_ssR = rho_sR
        U_momn_ssR = rho_sR * SM
        U_momt1_ssR = rho_sR * vt1_ssR
        U_momt2_ssR = rho_sR * vt2_ssR
        U_ene_ssR = e_ssR
        U_Bt1_ssR = Bt1_ssR
        U_Bt2_ssR = Bt2_ssR

        # --- 5-region flux selection ---
        if SL >= 0.0:
            F_rho[i] = Frho_L
            F_momn[i] = Fmomn_L
            F_momt1[i] = Fmomt1_L
            F_momt2[i] = Fmomt2_L
            F_ene[i] = Fene_L
            F_Bt1[i] = FBt1_L
            F_Bt2[i] = FBt2_L
        elif SR <= 0.0:
            F_rho[i] = Frho_R
            F_momn[i] = Fmomn_R
            F_momt1[i] = Fmomt1_R
            F_momt2[i] = Fmomt2_R
            F_ene[i] = Fene_R
            F_Bt1[i] = FBt1_R
            F_Bt2[i] = FBt2_R
        elif SM >= 0.0:
            # Left of contact
            if SL_star >= 0.0:
                # Between SL and SL*: use star-L
                F_rho[i] = Frho_L + SL * (U_rho_sL - U_rho_L)
                F_momn[i] = Fmomn_L + SL * (U_momn_sL - U_momn_L)
                F_momt1[i] = Fmomt1_L + SL * (U_momt1_sL - U_momt1_L)
                F_momt2[i] = Fmomt2_L + SL * (U_momt2_sL - U_momt2_L)
                F_ene[i] = Fene_L + SL * (U_ene_sL - U_ene_L)
                F_Bt1[i] = FBt1_L + SL * (U_Bt1_sL - U_Bt1_L)
                F_Bt2[i] = FBt2_L + SL * (U_Bt2_sL - U_Bt2_L)
            else:
                # Between SL* and SM: use double-star-L
                F_rho[i] = (Frho_L + SL * (U_rho_sL - U_rho_L)
                            + SL_star * (U_rho_ssL - U_rho_sL))
                F_momn[i] = (Fmomn_L + SL * (U_momn_sL - U_momn_L)
                             + SL_star * (U_momn_ssL - U_momn_sL))
                F_momt1[i] = (Fmomt1_L + SL * (U_momt1_sL - U_momt1_L)
                              + SL_star * (U_momt1_ssL - U_momt1_sL))
                F_momt2[i] = (Fmomt2_L + SL * (U_momt2_sL - U_momt2_L)
                              + SL_star * (U_momt2_ssL - U_momt2_sL))
                F_ene[i] = (Fene_L + SL * (U_ene_sL - U_ene_L)
                            + SL_star * (U_ene_ssL - U_ene_sL))
                F_Bt1[i] = (FBt1_L + SL * (U_Bt1_sL - U_Bt1_L)
                            + SL_star * (U_Bt1_ssL - U_Bt1_sL))
                F_Bt2[i] = (FBt2_L + SL * (U_Bt2_sL - U_Bt2_L)
                            + SL_star * (U_Bt2_ssL - U_Bt2_sL))
        else:
            # Right of contact
            if SR_star <= 0.0:
                # Between SR* and SR: use double-star-R
                F_rho[i] = (Frho_R + SR * (U_rho_sR - U_rho_R)
                            + SR_star * (U_rho_ssR - U_rho_sR))
                F_momn[i] = (Fmomn_R + SR * (U_momn_sR - U_momn_R)
                             + SR_star * (U_momn_ssR - U_momn_sR))
                F_momt1[i] = (Fmomt1_R + SR * (U_momt1_sR - U_momt1_R)
                              + SR_star * (U_momt1_ssR - U_momt1_sR))
                F_momt2[i] = (Fmomt2_R + SR * (U_momt2_sR - U_momt2_R)
                              + SR_star * (U_momt2_ssR - U_momt2_sR))
                F_ene[i] = (Fene_R + SR * (U_ene_sR - U_ene_R)
                            + SR_star * (U_ene_ssR - U_ene_sR))
                F_Bt1[i] = (FBt1_R + SR * (U_Bt1_sR - U_Bt1_R)
                            + SR_star * (U_Bt1_ssR - U_Bt1_sR))
                F_Bt2[i] = (FBt2_R + SR * (U_Bt2_sR - U_Bt2_R)
                            + SR_star * (U_Bt2_ssR - U_Bt2_sR))
            else:
                # Between SM and SR*: use star-R
                F_rho[i] = Frho_R + SR * (U_rho_sR - U_rho_R)
                F_momn[i] = Fmomn_R + SR * (U_momn_sR - U_momn_R)
                F_momt1[i] = Fmomt1_R + SR * (U_momt1_sR - U_momt1_R)
                F_momt2[i] = Fmomt2_R + SR * (U_momt2_sR - U_momt2_R)
                F_ene[i] = Fene_R + SR * (U_ene_sR - U_ene_R)
                F_Bt1[i] = FBt1_R + SR * (U_Bt1_sR - U_Bt1_R)
                F_Bt2[i] = FBt2_R + SR * (U_Bt2_sR - U_Bt2_R)

    return F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2


def _hlld_flux_1d_core(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    u_L: np.ndarray,
    u_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn_L: np.ndarray,
    Bn_R: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible HLLD wrapper returning only (mass, momentum, energy) fluxes.

    Calls the full 8-component solver with zero transverse velocities and B-fields,
    then returns only the first 3 flux components.  Used by the cylindrical solver.
    """
    n = len(rho_L)
    zeros = np.zeros(n)
    Bn_avg = 0.5 * (Bn_L + Bn_R)
    F_rho, F_momn, _, _, F_ene, _, _ = _hlld_flux_1d_8comp(
        rho_L, rho_R, u_L, u_R, zeros, zeros, zeros, zeros,
        p_L, p_R, Bn_avg, zeros, zeros, zeros, zeros, gamma,
    )
    return F_rho, F_momn, F_ene


def _hlld_flux_1d(
    rho_L: np.ndarray,
    rho_R: np.ndarray,
    vn_L: np.ndarray,
    vn_R: np.ndarray,
    vt1_L: np.ndarray,
    vt1_R: np.ndarray,
    vt2_L: np.ndarray,
    vt2_R: np.ndarray,
    p_L: np.ndarray,
    p_R: np.ndarray,
    Bn: np.ndarray,
    Bt1_L: np.ndarray,
    Bt1_R: np.ndarray,
    Bt2_L: np.ndarray,
    Bt2_R: np.ndarray,
    gamma: float,
) -> dict[str, np.ndarray]:
    """HLLD approximate Riemann solver — dict wrapper."""
    F_rho, F_momn, F_momt1, F_momt2, F_ene, F_Bt1, F_Bt2 = _hlld_flux_1d_8comp(
        rho_L, rho_R, vn_L, vn_R, vt1_L, vt1_R, vt2_L, vt2_R,
        p_L, p_R, Bn, Bt1_L, Bt1_R, Bt2_L, Bt2_R, gamma,
    )
    return {
        "mass_flux": F_rho,
        "momentum_flux": F_momn,
        "momentum_t1_flux": F_momt1,
        "momentum_t2_flux": F_momt2,
        "energy_flux": F_ene,
        "Bt1_flux": F_Bt1,
        "Bt2_flux": F_Bt2,
    }


# ============================================================
# Dimension-split WENO5+HLL/HLLD flux computation
# ============================================================

def _compute_flux_1d_sweep(
    rho: np.ndarray,
    vel_n: np.ndarray,
    vel_t1: np.ndarray,
    vel_t2: np.ndarray,
    pressure: np.ndarray,
    Bn: np.ndarray,
    Bt1: np.ndarray,
    Bt2: np.ndarray,
    gamma: float,
    axis: int,
    riemann_solver: str = "hll",
) -> dict[str, np.ndarray]:
    """Compute WENO5-reconstructed Riemann fluxes along one axis.

    Uses dimension-by-dimension sweep: for each pencil along `axis`,
    perform WENO5 reconstruction then HLL or HLLD Riemann solve.

    Args:
        rho: Density, shape (nx, ny, nz).
        vel_n: Normal velocity component, shape (nx, ny, nz).
        vel_t1: First transverse velocity component, shape (nx, ny, nz).
        vel_t2: Second transverse velocity component, shape (nx, ny, nz).
        pressure: Thermal pressure, shape (nx, ny, nz).
        Bn: Normal B-field component, shape (nx, ny, nz).
        Bt1: First transverse B-field component, shape (nx, ny, nz).
        Bt2: Second transverse B-field component, shape (nx, ny, nz).
        gamma: Adiabatic index.
        axis: Sweep axis (0, 1, or 2).
        riemann_solver: "hll" or "hlld".

    Returns:
        Dictionary of flux arrays at interfaces (reduced by 4 along `axis`).
        For HLLD, includes transverse momentum and B-field fluxes.
    """
    shape = rho.shape
    n_ax = shape[axis]

    # If grid too small for WENO5 (need >=5 cells), fall back to np.gradient
    if n_ax < 5:
        # Return zero fluxes
        return {
            "mass_flux": np.zeros_like(rho),
            "momentum_flux": np.zeros_like(rho),
            "energy_flux": np.zeros_like(rho),
            "n_interfaces": 0,
        }

    n_iface = n_ax - 4

    # Output arrays: n_iface along `axis`, same on other axes
    out_shape = list(shape)
    out_shape[axis] = n_iface
    out_shape = tuple(out_shape)

    F_rho = np.zeros(out_shape)
    F_mom = np.zeros(out_shape)
    F_ene = np.zeros(out_shape)

    # Additional arrays for HLLD transverse fluxes
    use_hlld = riemann_solver == "hlld"
    if use_hlld:
        F_momt1 = np.zeros(out_shape)
        F_momt2 = np.zeros(out_shape)
        F_Bt1 = np.zeros(out_shape)
        F_Bt2 = np.zeros(out_shape)

    # Iterate over pencils perpendicular to axis
    other_axes = [i for i in range(3) if i != axis]
    for idx0 in range(shape[other_axes[0]]):
        for idx1 in range(shape[other_axes[1]]):
            # Build the slicer for this pencil
            slicer = [None, None, None]
            slicer[other_axes[0]] = idx0
            slicer[other_axes[1]] = idx1
            slicer[axis] = slice(None)
            s = tuple(slicer)

            # Extract 1D pencils
            rho_1d = rho[s]
            u_1d = vel_n[s]
            p_1d = pressure[s]
            Bn_1d = Bn[s]

            # WENO5 reconstruct each quantity
            rho_L, rho_R = _weno5_reconstruct_1d(rho_1d)
            u_L, u_R = _weno5_reconstruct_1d(u_1d)
            p_L, p_R = _weno5_reconstruct_1d(p_1d)
            Bn_L, Bn_R = _weno5_reconstruct_1d(Bn_1d)

            # Ensure positivity
            rho_L = np.maximum(rho_L, 1e-20)
            rho_R = np.maximum(rho_R, 1e-20)
            p_L = np.maximum(p_L, 1e-20)
            p_R = np.maximum(p_R, 1e-20)

            # Riemann solve (HLL or HLLD)
            if use_hlld:
                # Reconstruct transverse quantities for full 8-component HLLD
                vt1_1d = vel_t1[s]
                vt2_1d = vel_t2[s]
                Bt1_1d = Bt1[s]
                Bt2_1d = Bt2[s]

                vt1_L, vt1_R = _weno5_reconstruct_1d(vt1_1d)
                vt2_L, vt2_R = _weno5_reconstruct_1d(vt2_1d)
                Bt1_L_r, Bt1_R_r = _weno5_reconstruct_1d(Bt1_1d)
                Bt2_L_r, Bt2_R_r = _weno5_reconstruct_1d(Bt2_1d)

                # Average Bn at interface (continuous in ideal MHD)
                Bn_avg = 0.5 * (Bn_L + Bn_R)

                fluxes = _hlld_flux_1d(
                    rho_L, rho_R, u_L, u_R, vt1_L, vt1_R, vt2_L, vt2_R,
                    p_L, p_R, Bn_avg, Bt1_L_r, Bt1_R_r, Bt2_L_r, Bt2_R_r,
                    gamma,
                )
            else:
                fluxes = _hll_flux_1d(
                    rho_L, rho_R, u_L, u_R, p_L, p_R, Bn_L, Bn_R, gamma,
                )

            # Store in output array
            out_slicer = [None, None, None]
            out_slicer[other_axes[0]] = idx0
            out_slicer[other_axes[1]] = idx1
            out_slicer[axis] = slice(None)
            out_s = tuple(out_slicer)

            F_rho[out_s] = fluxes["mass_flux"]
            F_mom[out_s] = fluxes["momentum_flux"]
            F_ene[out_s] = fluxes["energy_flux"]

            if use_hlld:
                F_momt1[out_s] = fluxes["momentum_t1_flux"]
                F_momt2[out_s] = fluxes["momentum_t2_flux"]
                F_Bt1[out_s] = fluxes["Bt1_flux"]
                F_Bt2[out_s] = fluxes["Bt2_flux"]

    result = {
        "mass_flux": F_rho,
        "momentum_flux": F_mom,
        "energy_flux": F_ene,
        "n_interfaces": n_iface,
    }
    if use_hlld:
        result["momentum_t1_flux"] = F_momt1
        result["momentum_t2_flux"] = F_momt2
        result["Bt1_flux"] = F_Bt1
        result["Bt2_flux"] = F_Bt2

    return result


def _apply_flux_divergence(
    U: np.ndarray,
    flux: np.ndarray,
    n_interfaces: int,
    axis: int,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Apply conservative flux-difference update from WENO5+HLL fluxes.

    The flux array has (n_interfaces) entries along `axis`, corresponding
    to faces between cells [2..n_ax-3]. We update the interior cells that
    have both left and right flux faces: cells 2..(n_ax-3) → n_interfaces-1 cells.

    Args:
        U: Conservative variable, shape (nx, ny, nz).
        flux: Interface fluxes, shape with n_interfaces along axis.
        n_interfaces: Number of flux interfaces.
        axis: Sweep axis.
        dx: Grid spacing.
        dt: Timestep.

    Returns:
        Updated conservative variable.
    """
    if n_interfaces < 2:
        return U.copy()

    result = U.copy()

    # Number of cells we can update: n_interfaces - 1
    n_update = n_interfaces - 1

    # Build slice objects for the flux difference
    # F[j+1] - F[j] for j = 0..n_interfaces-2
    flux_left_slicer = [slice(None)] * 3
    flux_right_slicer = [slice(None)] * 3
    update_slicer = [slice(None)] * 3

    flux_left_slicer[axis] = slice(0, n_update)
    flux_right_slicer[axis] = slice(1, n_update + 1)
    update_slicer[axis] = slice(2, 2 + n_update)

    dF = flux[tuple(flux_right_slicer)] - flux[tuple(flux_left_slicer)]
    result[tuple(update_slicer)] -= dt / dx * dF

    return result


# ============================================================
# Dedner divergence cleaning
# ============================================================

def _dedner_source(
    psi: np.ndarray,
    B: np.ndarray,
    ch: float,
    cp: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dedner hyperbolic/parabolic divergence cleaning.

    dpsi/dt = -ch^2 * div(B) - (ch^2 / cp^2) * psi
    dB/dt += -grad(psi)

    Args:
        psi: Cleaning scalar field, shape (nx, ny, nz).
        B: Magnetic field, shape (3, nx, ny, nz).
        ch: Hyperbolic cleaning speed [m/s].
        cp: Parabolic damping speed [m/s].
        dx: Grid spacing [m].

    Returns:
        (dpsi_dt, dB_dt): Source terms.
    """
    div_B = (
        np.gradient(B[0], dx, axis=0)
        + np.gradient(B[1], dx, axis=1)
        + np.gradient(B[2], dx, axis=2)
    )

    dpsi_dt = -ch**2 * div_B - (ch**2 / (cp**2 + 1e-30)) * psi

    grad_psi = np.array([
        np.gradient(psi, dx, axis=0),
        np.gradient(psi, dx, axis=1),
        np.gradient(psi, dx, axis=2),
    ])
    dB_dt = -grad_psi

    return dpsi_dt, dB_dt


def _dedner_source_mt2010(
    psi: np.ndarray,
    B: np.ndarray,
    ch: float,
    cr: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dedner divergence cleaning with Mignone & Tzeferacos (2010) tuning.

    Improved Dedner cleaning with separate damping coefficient cr
    (Mignone & Tzeferacos, JCP 229, 5896, 2010):

        dpsi/dt = -ch^2 * div(B) - cr * psi
        dB/dt  += -grad(psi)

    The damping rate cr controls how quickly psi decays. The original
    Dedner uses cr = ch^2/cp^2, while M&T2010 prescribes cr ~ ch/dx
    for optimal damping.

    Args:
        psi: Cleaning scalar field, shape (nx, ny, nz).
        B: Magnetic field, shape (3, nx, ny, nz).
        ch: Hyperbolic cleaning speed [m/s].
        cr: Damping rate [1/s].
        dx: Grid spacing [m].

    Returns:
        (dpsi_dt, dB_dt): Source terms.
    """
    div_B = (
        np.gradient(B[0], dx, axis=0)
        + np.gradient(B[1], dx, axis=1)
        + np.gradient(B[2], dx, axis=2)
    )

    dpsi_dt = -ch**2 * div_B - cr * psi

    grad_psi = np.array([
        np.gradient(psi, dx, axis=0),
        np.gradient(psi, dx, axis=1),
        np.gradient(psi, dx, axis=2),
    ])
    dB_dt = -grad_psi

    return dpsi_dt, dB_dt


# ============================================================
# Powell 8-wave source terms
# ============================================================

def powell_source_terms(
    state: dict[str, np.ndarray],
    dx: float,
    dy: float,
    dz: float,
) -> dict[str, np.ndarray]:
    """Compute Powell 8-wave div(B) source terms.

    Powell et al. (1999) source terms that complement Dedner divergence
    cleaning by proportionally correcting momentum, energy, and induction:

        S_Powell = -div(B) * [0, Bx, By, Bz, v.B, vx, vy, vz, 0]^T

    In conservative variable ordering (rho, rho*v, E, B, psi):
        - Mass: 0
        - Momentum: -div(B) * B
        - Energy: -div(B) * (v . B)
        - Induction: -div(B) * v
        - Psi: 0

    Args:
        state: State dictionary with keys rho, velocity, pressure, B, psi.
        dx, dy, dz: Grid spacings [m].

    Returns:
        Dictionary with source term arrays:
            dmom_powell: shape (3, nx, ny, nz)
            denergy_powell: shape (nx, ny, nz)
            dB_powell: shape (3, nx, ny, nz)
    """
    B = state["B"]
    vel = state["velocity"]

    # Compute div(B) using central differences
    div_B = (
        np.gradient(B[0], dx, axis=0)
        + np.gradient(B[1], dy, axis=1)
        + np.gradient(B[2], dz, axis=2)
    )

    # Momentum source: -div(B) * B
    dmom_powell = np.zeros_like(B)
    for d in range(3):
        dmom_powell[d] = -div_B * B[d]

    # Energy source: -div(B) * (v . B)
    v_dot_B = np.sum(vel * B, axis=0)
    denergy_powell = -div_B * v_dot_B

    # Induction source: -div(B) * v
    dB_powell = np.zeros_like(B)
    for d in range(3):
        dB_powell[d] = -div_B * vel[d]

    return {
        "dmom_powell": dmom_powell,
        "denergy_powell": denergy_powell,
        "dB_powell": dB_powell,
        "div_B": div_B,
    }


def powell_source_terms_cylindrical(
    state_2d: dict[str, np.ndarray],
    geom,
) -> dict[str, np.ndarray]:
    """Powell source terms for cylindrical coordinates.

    Uses cylindrical div(B): (1/r) * d(r * B_r)/dr + dB_z/dz.

    Args:
        state_2d: State dictionary with 2D arrays (nr, nz).
        geom: CylindricalGeometry instance with div_B_cylindrical method.

    Returns:
        Dictionary with source term arrays (same as powell_source_terms).
    """
    B = state_2d["B"]
    vel = state_2d["velocity"]

    # Cylindrical divergence of B
    div_B = geom.div_B_cylindrical(B)

    # Momentum source: -div(B) * B
    dmom_powell = np.zeros_like(B)
    for d in range(3):
        dmom_powell[d] = -div_B * B[d]

    # Energy source: -div(B) * (v . B)
    v_dot_B = np.sum(vel * B, axis=0)
    denergy_powell = -div_B * v_dot_B

    # Induction source: -div(B) * v
    dB_powell = np.zeros_like(B)
    for d in range(3):
        dB_powell[d] = -div_B * vel[d]

    return {
        "dmom_powell": dmom_powell,
        "denergy_powell": denergy_powell,
        "dB_powell": dB_powell,
        "div_B": div_B,
    }


# ============================================================
# Braginskii anisotropic heat flux operator
# ============================================================

def _braginskii_heat_flux(
    Te: np.ndarray,
    ne: np.ndarray,
    B: np.ndarray,
    dx: float,
    dt: float,
) -> np.ndarray:
    """Apply Braginskii anisotropic heat flux to electron temperature.

    Heat flux: q = -kappa_par * (b_hat . grad(Te)) * b_hat
                 - kappa_perp * grad_perp(Te)

    Uses operator-split explicit diffusion with sub-cycling if needed
    for stability: dt_diff < dx^2 / (2 * max(kappa)).

    Args:
        Te: Electron temperature [K], shape (nx, ny, nz).
        ne: Electron number density [m^-3], shape (nx, ny, nz).
        B: Magnetic field [T], shape (3, nx, ny, nz).
        dx: Grid spacing [m].
        dt: Timestep [s].

    Returns:
        Updated Te array.
    """
    from dpf.collision.spitzer import braginskii_kappa

    B_mag = np.sqrt(np.sum(B**2, axis=0))

    # Compute Braginskii conductivities
    kappa_par, kappa_perp = braginskii_kappa(ne, Te, B_mag)

    # Sanitize kappa arrays — NaN/Inf from extreme conditions
    kappa_par = np.where(np.isfinite(kappa_par), kappa_par, 0.0)
    kappa_perp = np.where(np.isfinite(kappa_perp), kappa_perp, 0.0)

    # Maximum diffusion coefficient for stability check
    max_kappa = np.max(kappa_par)
    if max_kappa < 1e-30:
        return Te.copy()

    # Cap kappa to avoid extreme diffusivity (physical limit)
    kappa_cap = 1e30  # Reasonable cap for plasma heat conductivity
    kappa_par = np.minimum(kappa_par, kappa_cap)
    kappa_perp = np.minimum(kappa_perp, kappa_cap)
    max_kappa = min(max_kappa, kappa_cap)

    # Temperature gradient
    grad_Te = np.array([
        np.gradient(Te, dx, axis=0),
        np.gradient(Te, dx, axis=1),
        np.gradient(Te, dx, axis=2),
    ])

    # Magnetic unit vector
    B_hat = np.zeros_like(B)
    B_safe = np.maximum(B_mag, 1e-30)
    for i in range(3):
        B_hat[i] = B[i] / B_safe

    # Parallel gradient: (b . grad(Te)) * b
    b_dot_gradT = np.sum(B_hat * grad_Te, axis=0)
    q_par_dir = np.zeros_like(B)
    for i in range(3):
        q_par_dir[i] = b_dot_gradT * B_hat[i]

    # Perpendicular gradient: grad(Te) - (b . grad(Te)) * b
    grad_perp_Te = grad_Te - q_par_dir

    # Heat flux divergence: div(kappa_par * q_par + kappa_perp * q_perp)
    heat_flux = np.zeros_like(B)
    for i in range(3):
        heat_flux[i] = kappa_par * q_par_dir[i] + kappa_perp * grad_perp_Te[i]

    # Sanitize heat flux — catch NaN/Inf from kappa * gradient products
    heat_flux = np.where(np.isfinite(heat_flux), heat_flux, 0.0)

    div_q = (
        np.gradient(heat_flux[0], dx, axis=0)
        + np.gradient(heat_flux[1], dx, axis=1)
        + np.gradient(heat_flux[2], dx, axis=2)
    )

    # Sanitize div_q
    div_q = np.where(np.isfinite(div_q), div_q, 0.0)

    # Stability-limited timestep for explicit diffusion
    # dt_diff < dx^2 / (2 * dim * max_kappa / (n_e * k_B))
    ne_safe = np.maximum(ne, 1e-20)
    min_ne = np.min(ne_safe)
    diffusivity = max_kappa / (min_ne * k_B + 1e-30)

    if not np.isfinite(diffusivity) or diffusivity <= 0:
        return Te.copy()

    dt_diff = 0.25 * dx**2 / diffusivity

    if not np.isfinite(dt_diff) or dt_diff <= 0:
        return Te.copy()

    # Sub-cycle if needed
    n_sub = max(1, int(np.ceil(dt / dt_diff)))
    n_sub = min(n_sub, 100)  # Cap subcycles to avoid runaway
    dt_sub = dt / n_sub

    Te_new = Te.copy()
    for _ in range(n_sub):
        # dTe/dt = div(q) / (ne * kB)
        Te_new += dt_sub * div_q / (ne_safe * k_B)

    # Floor temperature
    Te_new = np.maximum(Te_new, 1.0)
    # Sanitize
    Te_new = np.where(np.isfinite(Te_new), Te_new, Te)

    return Te_new


# ============================================================
# Main MHD Solver
# ============================================================

class MHDSolver(PlasmaSolverBase):
    """Hall MHD solver with WENO5-Z reconstruction, HLLD Riemann, and Dedner cleaning.

    Features:
    - WENO5-Z reconstruction + HLLD Riemann solver for advection (5th-order spatial)
    - SSP-RK3 time integration (3rd-order temporal, default) or SSP-RK2
    - Dedner hyperbolic divergence cleaning for div(B)
    - Hall term in induction equation (J × B)/(ne)
    - Braginskii anisotropic heat flux (operator-split)
    - dL_dt estimation from pinch dynamics for circuit coupling

    Falls back to forward-Euler np.gradient if grid < 5 cells in any direction.

    Args:
        grid_shape: (nx, ny, nz).
        dx: Grid spacing [m].
        gamma: Adiabatic index.
        cfl: CFL number for timestep.
        dedner_ch: Dedner hyperbolic cleaning speed (0 = auto from max wave speed).
        enable_hall: Enable Hall term in induction equation.
        enable_braginskii: Enable Braginskii anisotropic heat flux.
        time_integrator: "ssp_rk3" (default, 3rd-order) or "ssp_rk2" (2nd-order).
        riemann_solver: "hlld" (default) or "hll".
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        dx: float,
        gamma: float = 5.0 / 3.0,
        cfl: float = 0.4,
        dedner_ch: float = 0.0,
        enable_hall: bool = True,
        enable_braginskii: bool = True,
        enable_resistive: bool = True,
        enable_energy_equation: bool = True,
        ion_mass: float | None = None,
        riemann_solver: str = "hlld",
        time_integrator: str = "ssp_rk3",
        use_ct: bool = False,
    ) -> None:
        self.grid_shape = grid_shape
        self.dx = dx
        self.gamma = gamma
        self.cfl = cfl
        self.dedner_ch_init = dedner_ch
        self.enable_hall = enable_hall
        self.enable_braginskii = enable_braginskii
        self.enable_resistive = enable_resistive
        self.enable_energy_equation = enable_energy_equation
        self.ion_mass = ion_mass if ion_mass is not None else _DEFAULT_ION_MASS
        self.riemann_solver = riemann_solver if riemann_solver in ("hll", "hlld") else "hlld"
        self.time_integrator = time_integrator if time_integrator in ("ssp_rk2", "ssp_rk3") else "ssp_rk3"
        self.use_ct = use_ct
        self.eos = IdealEOS(gamma=gamma)

        # Whether we can use WENO5 (need >= 5 cells in each direction)
        self.use_weno5 = all(n >= 5 for n in grid_shape)

        # Coupling state for circuit interaction
        self._coupling = CouplingState()
        self._prev_Lp: float | None = None  # For dL_dt computation

        logger.info(
            "MHDSolver initialized: grid=%s, dx=%.2e, gamma=%.3f, "
            "WENO5-Z=%s, Hall=%s, Braginskii=%s, Resistive=%s, EnergyEq=%s, "
            "Riemann=%s, TimeInt=%s, CT=%s, ion_mass=%.3e kg",
            grid_shape, dx, gamma,
            self.use_weno5, enable_hall, enable_braginskii,
            self.enable_resistive, self.enable_energy_equation,
            self.riemann_solver, self.time_integrator, self.use_ct,
            self.ion_mass,
        )

    def _compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """Compute CFL-limited timestep."""
        rho = state["rho"]
        v = state["velocity"]
        B = state["B"]
        p = state["pressure"]

        # Maximum wave speed: fast magnetosonic
        B_sq = np.sum(B**2, axis=0)
        a2 = self.gamma * p / np.maximum(rho, 1e-30)
        va2 = B_sq / (mu_0 * np.maximum(rho, 1e-30))
        cf = np.sqrt(a2 + va2)

        v_max = np.max(np.abs(v)) + np.max(cf)

        # Hall speed limit: omega_ci * di where di = c / omega_pi
        if self.enable_hall:
            ne = rho / self.ion_mass  # Assume Z=1
            ne_max = np.max(ne)
            if ne_max > 0:
                B_max = np.sqrt(np.max(B_sq))
                # Hall speed ~ B / (mu_0 * ne * e * dx)
                v_hall = B_max / (mu_0 * np.maximum(ne_max, 1e-20) * e_charge * self.dx)
                v_max = max(v_max, v_hall)

        if v_max < 1e-30:
            return 1e-10  # Fallback for zero-velocity initial condition
        return self.cfl * self.dx / v_max

    def _compute_rhs_euler(
        self,
        state: dict[str, np.ndarray],
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
        source_terms: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute the right-hand side (time derivative) of the MHD system.

        Uses WENO5+HLL flux-based update for advection if grid is large enough,
        otherwise falls back to np.gradient centered differences.

        Args:
            state: Current state dict.
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nx,ny,nz).
                       If None, resistive term is skipped.
            source_terms: Optional dictionary of external source terms (e.g. 'J_kin').

        Returns dict of dU/dt for each state variable.
        """
        rho = state["rho"]
        vel = state["velocity"]
        p = state["pressure"]
        B = state["B"]
        psi = state.get("psi", np.zeros_like(rho))

        drho_dt = np.zeros_like(rho)
        dmom_dt = np.zeros((3,) + rho.shape)
        dp_dt = np.zeros_like(rho)
        dB_dt = np.zeros_like(B)

        # --- Density advection + HLLD full flux update ---
        # Transverse index mapping for each sweep axis:
        #   axis=0: vn=vel[0], vt1=vel[1], vt2=vel[2], Bn=B[0], Bt1=B[1], Bt2=B[2]
        #   axis=1: vn=vel[1], vt1=vel[2], vt2=vel[0], Bn=B[1], Bt1=B[2], Bt2=B[0]
        #   axis=2: vn=vel[2], vt1=vel[0], vt2=vel[1], Bn=B[2], Bt1=B[0], Bt2=B[1]
        _transverse_map = {
            0: (1, 2),
            1: (2, 0),
            2: (0, 1),
        }

        # Store per-axis flux results for reuse by momentum and induction
        _axis_fluxes: list[dict[str, np.ndarray] | None] = [None, None, None]

        if self.use_weno5:
            # WENO5+Riemann flux-based update: accumulate flux divergence
            for axis in range(3):
                t1_idx, t2_idx = _transverse_map[axis]
                fluxes = _compute_flux_1d_sweep(
                    rho, vel[axis], vel[t1_idx], vel[t2_idx],
                    p, B[axis], B[t1_idx], B[t2_idx],
                    self.gamma, axis,
                    riemann_solver=self.riemann_solver,
                )
                _axis_fluxes[axis] = fluxes
                n_iface = fluxes["n_interfaces"]
                if n_iface >= 2:
                    n_update = n_iface - 1
                    flux_left_sl = [slice(None)] * 3
                    flux_right_sl = [slice(None)] * 3
                    update_sl = [slice(None)] * 3
                    flux_left_sl[axis] = slice(0, n_update)
                    flux_right_sl[axis] = slice(1, n_update + 1)
                    update_sl[axis] = slice(2, 2 + n_update)
                    dF = (
                        fluxes["mass_flux"][tuple(flux_right_sl)]
                        - fluxes["mass_flux"][tuple(flux_left_sl)]
                    )
                    drho_dt[tuple(update_sl)] -= dF / self.dx
        else:
            # Fallback: centered-difference with np.gradient (original MVP behavior)
            flux_rho = np.array([rho * vel[i] for i in range(3)])
            div_flux = (
                np.gradient(flux_rho[0], self.dx, axis=0)
                + np.gradient(flux_rho[1], self.dx, axis=1)
                + np.gradient(flux_rho[2], self.dx, axis=2)
            )
            drho_dt = -div_flux

        # --- Momentum: J x B force + pressure gradient ---
        # Current density: J = curl(B) / mu_0
        J = np.array([
            np.gradient(B[2], self.dx, axis=1) - np.gradient(B[1], self.dx, axis=2),
            np.gradient(B[0], self.dx, axis=2) - np.gradient(B[2], self.dx, axis=0),
            np.gradient(B[1], self.dx, axis=0) - np.gradient(B[0], self.dx, axis=1),
        ]) / mu_0

        # J x B force
        JxB = np.array([
            J[1] * B[2] - J[2] * B[1],
            J[2] * B[0] - J[0] * B[2],
            J[0] * B[1] - J[1] * B[0],
        ])

        grad_p = np.array([
            np.gradient(p, self.dx, axis=0),
            np.gradient(p, self.dx, axis=1),
            np.gradient(p, self.dx, axis=2),
        ])

        # Momentum advection (WENO5 for momentum flux if available)
        if self.use_weno5:
            use_hlld = self.riemann_solver == "hlld"
            if use_hlld:
                # Full 8-component HLLD: use momentum_flux (normal), momentum_t1_flux,
                # momentum_t2_flux from the already-computed per-axis sweeps.
                for axis in range(3):
                    fluxes = _axis_fluxes[axis]
                    if fluxes is None:
                        continue
                    n_iface = fluxes["n_interfaces"]
                    if n_iface < 2:
                        continue
                    n_update = n_iface - 1
                    flux_left_sl = [slice(None)] * 3
                    flux_right_sl = [slice(None)] * 3
                    update_sl = [slice(None)] * 3
                    flux_left_sl[axis] = slice(0, n_update)
                    flux_right_sl[axis] = slice(1, n_update + 1)
                    update_sl[axis] = slice(2, 2 + n_update)

                    t1_idx, t2_idx = _transverse_map[axis]

                    # Normal momentum flux -> updates momentum[axis]
                    dF_n = (
                        fluxes["momentum_flux"][tuple(flux_right_sl)]
                        - fluxes["momentum_flux"][tuple(flux_left_sl)]
                    )
                    dmom_dt[axis][tuple(update_sl)] -= dF_n / self.dx

                    # Transverse momentum flux t1 -> updates momentum[t1_idx]
                    dF_t1 = (
                        fluxes["momentum_t1_flux"][tuple(flux_right_sl)]
                        - fluxes["momentum_t1_flux"][tuple(flux_left_sl)]
                    )
                    dmom_dt[t1_idx][tuple(update_sl)] -= dF_t1 / self.dx

                    # Transverse momentum flux t2 -> updates momentum[t2_idx]
                    dF_t2 = (
                        fluxes["momentum_t2_flux"][tuple(flux_right_sl)]
                        - fluxes["momentum_t2_flux"][tuple(flux_left_sl)]
                    )
                    dmom_dt[t2_idx][tuple(update_sl)] -= dF_t2 / self.dx
            else:
                # HLL: only has mass_flux, use old approach
                for d in range(3):
                    for axis in range(3):
                        mom_d = rho * vel[d]
                        t1_idx, t2_idx = _transverse_map[axis]
                        fluxes = _compute_flux_1d_sweep(
                            mom_d, vel[axis], vel[t1_idx], vel[t2_idx],
                            p, B[axis], B[t1_idx], B[t2_idx],
                            self.gamma, axis,
                            riemann_solver="hll",
                        )
                        n_iface = fluxes["n_interfaces"]
                        if n_iface >= 2:
                            n_update = n_iface - 1
                            flux_left_sl = [slice(None)] * 3
                            flux_right_sl = [slice(None)] * 3
                            update_sl = [slice(None)] * 3
                            flux_left_sl[axis] = slice(0, n_update)
                            flux_right_sl[axis] = slice(1, n_update + 1)
                            update_sl[axis] = slice(2, 2 + n_update)
                            dF = (
                                fluxes["mass_flux"][tuple(flux_right_sl)]
                                - fluxes["mass_flux"][tuple(flux_left_sl)]
                            )
                            dmom_dt[d][tuple(update_sl)] -= dF / self.dx
        else:
            # Fallback: centered difference momentum advection
            for d in range(3):
                for axis in range(3):
                    flux_mom = rho * vel[d] * vel[axis]
                    dmom_dt[d] -= np.gradient(flux_mom, self.dx, axis=axis)

        # Add Lorentz force and pressure gradient
        # Kinetic Feedback: J_plasma = J_tot - J_kin
        # J_tot = curl(B)/mu_0 (calculated above as 'J')
        J_plasma = J.copy()
        if source_terms is not None and "J_kin" in source_terms:
            J_kin = source_terms["J_kin"]
            # Ensure shape matches (3, nx, ny, nz)
            if J_kin.shape == J.shape:
                J_plasma -= J_kin
            else:
                pass # TODO: Add shape check/warning

        # J x B force uses J_plasma (force on the fluid)
        JxB = np.array([
            J_plasma[1] * B[2] - J_plasma[2] * B[1],
            J_plasma[2] * B[0] - J_plasma[0] * B[2],
            J_plasma[0] * B[1] - J_plasma[1] * B[0],
        ])

        for d in range(3):
            dmom_dt[d] += JxB[d] - grad_p[d]

        # --- Induction equation: dB/dt = -curl(E) ---
        # Ideal MHD: E = -v × B
        vxB = np.array([
            vel[1] * B[2] - vel[2] * B[1],
            vel[2] * B[0] - vel[0] * B[2],
            vel[0] * B[1] - vel[1] * B[0],
        ])
        E_field = -vxB  # E = -v × B (ideal Ohm's law)

        # --- Resistive term: E_resistive = eta * J_plasma ---
        ohmic_heating = np.zeros_like(rho)
        if self.enable_resistive and eta_field is not None:
            # Resistivity acts on the conduction current (J_plasma), not total
            E_resistive = eta_field[np.newaxis, :, :, :] * J_plasma
            E_field += E_resistive
            # Ohmic heating: Q_ohm = eta * |J_plasma|^2 [W/m^3]
            J_sq = np.sum(J_plasma**2, axis=0)
            ohmic_heating = eta_field * J_sq

        # --- Hall term: E_Hall = (J_plasma × B) / (n_e * e) ---
        if self.enable_hall:
            ne = rho / self.ion_mass  # Z=1
            ne_safe = np.maximum(ne, 1e-20)
            # Hall electric field depends on J_plasma (current carried by fluid species)
            E_Hall = np.array([
                (J_plasma[1] * B[2] - J_plasma[2] * B[1]),
                (J_plasma[2] * B[0] - J_plasma[0] * B[2]),
                (J_plasma[0] * B[1] - J_plasma[1] * B[0]),
            ]) / (ne_safe * e_charge)
            E_field += E_Hall

        # dB/dt = -curl(E)
        curl_E = np.array([
            np.gradient(E_field[2], self.dx, axis=1) - np.gradient(E_field[1], self.dx, axis=2),
            np.gradient(E_field[0], self.dx, axis=2) - np.gradient(E_field[2], self.dx, axis=0),
            np.gradient(E_field[1], self.dx, axis=0) - np.gradient(E_field[0], self.dx, axis=1),
        ])
        dB_dt = -curl_E

        # --- HLLD B-field flux correction for transverse B in WENO5 interior ---
        # When HLLD provides Bt1/Bt2 fluxes, apply their divergence to interior
        # cells.  This adds the higher-order WENO5-reconstructed induction flux
        # on top of the np.gradient-based curl(E).  The HLLD flux captures the
        # same ideal MHD induction physics but with 5th-order reconstruction,
        # providing better shock-capturing for the magnetic field.
        if self.use_weno5 and self.riemann_solver == "hlld":
            for axis in range(3):
                fluxes = _axis_fluxes[axis]
                if fluxes is None:
                    continue
                n_iface = fluxes["n_interfaces"]
                if n_iface < 2 or "Bt1_flux" not in fluxes:
                    continue
                n_update = n_iface - 1
                flux_left_sl = [slice(None)] * 3
                flux_right_sl = [slice(None)] * 3
                update_sl = [slice(None)] * 3
                flux_left_sl[axis] = slice(0, n_update)
                flux_right_sl[axis] = slice(1, n_update + 1)
                update_sl[axis] = slice(2, 2 + n_update)

                t1_idx, t2_idx = _transverse_map[axis]

                # Bt1 flux -> updates B[t1_idx]
                dF_Bt1 = (
                    fluxes["Bt1_flux"][tuple(flux_right_sl)]
                    - fluxes["Bt1_flux"][tuple(flux_left_sl)]
                )
                dB_dt[t1_idx][tuple(update_sl)] -= dF_Bt1 / self.dx

                # Bt2 flux -> updates B[t2_idx]
                dF_Bt2 = (
                    fluxes["Bt2_flux"][tuple(flux_right_sl)]
                    - fluxes["Bt2_flux"][tuple(flux_left_sl)]
                )
                dB_dt[t2_idx][tuple(update_sl)] -= dF_Bt2 / self.dx

        # --- Pressure / Energy equation ---
        div_v = (
            np.gradient(vel[0], self.dx, axis=0)
            + np.gradient(vel[1], self.dx, axis=1)
            + np.gradient(vel[2], self.dx, axis=2)
        )

        if self.enable_energy_equation:
            # Conservative total energy equation:
            # dE_total/dt = -div(F_energy) + eta*J^2 - P_rad
            # E_total = p/(gamma-1) + 0.5*rho*|v|^2 + |B|^2/(2*mu_0)
            # For the pressure update, we use:
            # dp/dt = -gamma*p*div(v) + (gamma-1)*eta*J^2
            # The Ohmic heating term adds (gamma-1)*Q_ohm to pressure rate
            dp_dt = -self.gamma * p * div_v + (self.gamma - 1.0) * ohmic_heating
        else:
            # Adiabatic: dp/dt = -gamma * p * div(v)
            dp_dt = -self.gamma * p * div_v

        # --- Dedner cleaning (Mignone & Tzeferacos 2010 tuning) ---
        # Skip Dedner when CT is enabled (mutually exclusive)
        if not self.use_ct:
            # ch = max fast magnetosonic speed across the grid
            # cr = ch / dx for optimal parabolic damping (M&T2010 prescription)
            if self.dedner_ch_init > 0:
                ch = self.dedner_ch_init
            else:
                # Auto: ch = max(|v| + c_f) where c_f is fast magnetosonic speed
                B_sq = np.sum(B**2, axis=0)
                cs2 = self.gamma * p / np.maximum(rho, 1e-20)
                va2 = B_sq / (mu_0 * np.maximum(rho, 1e-20))
                cf = np.sqrt(cs2 + va2)  # Fast magnetosonic speed
                v_mag = np.sqrt(np.sum(vel**2, axis=0))
                ch = float(np.max(v_mag + cf)) + 1.0
            cr = ch / self.dx  # M&T2010 optimal damping rate
            dpsi_dt, dB_clean = _dedner_source_mt2010(psi, B, ch, cr, self.dx)
            dB_dt += dB_clean
        else:
            # CT mode: no Dedner cleaning, psi is unused
            dpsi_dt = np.zeros_like(rho)

        return {
            "drho_dt": drho_dt,
            "dmom_dt": dmom_dt,
            "dp_dt": dp_dt,
            "dB_dt": dB_dt,
            "dpsi_dt": dpsi_dt,
            "ohmic_heating": ohmic_heating,
        }

    def apply_electrode_bfield_bc(
        self,
        B: np.ndarray,
        current: float,
        anode_radius: float,
        cathode_radius: float,
    ) -> np.ndarray:
        """Apply electrode B-field boundary conditions for Cartesian solver.

        In Cartesian 3D, we approximate the azimuthal B-field from the
        circuit current at cells near the electrode radii. This is a
        simplified version — the cylindrical solver has a more physical
        implementation.

        B_theta = mu_0 * I / (2 * pi * r) at the boundary cells.

        For Cartesian, we map B_theta onto (B_x, B_y) components using
        the local azimuthal angle.

        Args:
            B: Magnetic field (3, nx, ny, nz).
            current: Circuit current [A].
            anode_radius: Anode radius [m].
            cathode_radius: Cathode radius [m].

        Returns:
            Modified B-field array.
        """
        if abs(current) < 1e-10:
            return B

        nx, ny, nz = self.grid_shape
        dx = self.dx

        # Build radial coordinate from grid center
        x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        y = (np.arange(ny) - ny / 2.0 + 0.5) * dx

        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X**2 + Y**2)
        R_safe = np.maximum(R, 1e-30)

        # Azimuthal unit vectors: theta_hat = (-sin(theta), cos(theta), 0)
        #   = (-y/r, x/r, 0)
        sin_theta = Y / R_safe  # -sin is for Bx component
        cos_theta = X / R_safe  # cos is for By component

        # Apply at outer boundary cells (near cathode radius)
        # Find cells within one cell of cathode_radius
        mask_cathode = np.abs(R - cathode_radius) < 1.5 * dx

        if np.any(mask_cathode):
            for k in range(nz):
                B_th_local = mu_0 * current / (2.0 * np.pi * np.maximum(R, cathode_radius * 0.5))
                # B_x = -B_theta * sin(theta), B_y = B_theta * cos(theta)
                B[0, :, :, k] = np.where(mask_cathode, -B_th_local * sin_theta, B[0, :, :, k])
                B[1, :, :, k] = np.where(mask_cathode, B_th_local * cos_theta, B[1, :, :, k])

        return B

    def _euler_stage(
        self,
        rho: np.ndarray,
        mom: np.ndarray,
        p: np.ndarray,
        B: np.ndarray,
        psi: np.ndarray,
        Te: np.ndarray,
        Ti: np.ndarray,
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Single Euler stage: U^(1) = U + dt * L(U).

        Returns (rho, mom, p, B, psi, rhs).

        Note: When WENO5 is active, the density/momentum flux divergence only
        updates interior cells [2, N-3].  Boundary cells get zero drho_dt from
        the WENO path but non-zero dmom_dt from np.gradient.  After the update,
        we clamp momentum in boundary cells to prevent extreme velocities from
        contaminating multi-stage RK reconstructions.
        """
        vel = mom / np.maximum(rho[np.newaxis, :, :, :], 1e-30)
        state_in = {
            "rho": rho, "velocity": vel, "pressure": p,
            "B": B, "Te": Te, "Ti": Ti, "psi": psi,
        }
        rhs = self._compute_rhs_euler(state_in, current, voltage, eta_field)

        rho_out = rho + dt * rhs["drho_dt"]
        rho_out = np.maximum(rho_out, 1e-20)
        mom_out = mom + dt * rhs["dmom_dt"]
        p_out = p + dt * rhs["dp_dt"]
        p_out = np.maximum(p_out, 1e-20)
        B_out = B + dt * rhs["dB_dt"]
        psi_out = psi + dt * rhs["dpsi_dt"]

        # --- Velocity clamping for hybrid WENO5 stability ---
        # When WENO5 is active, boundary cells (indices 0,1 and N-2,N-1 along
        # each axis) get mismatched updates (no density flux but non-zero
        # momentum flux from np.gradient).  Clamp velocity at all cells to
        # prevent extreme values from destabilizing multi-stage RK methods.
        if self.use_weno5:
            B_sq = np.sum(B_out**2, axis=0)
            a2 = self.gamma * p_out / np.maximum(rho_out, 1e-30)
            va2 = B_sq / (mu_0 * np.maximum(rho_out, 1e-30))
            v_max_local = np.sqrt(a2 + va2)  # fast magnetosonic speed
            # Allow 10× the local fast magnetosonic speed
            v_clamp = np.maximum(v_max_local, 1.0) * 10.0
            vel_out = mom_out / np.maximum(rho_out[np.newaxis, :, :, :], 1e-30)
            v_mag = np.sqrt(np.sum(vel_out**2, axis=0))
            # Only clamp where velocity exceeds the bound
            scale = np.where(v_mag > v_clamp, v_clamp / np.maximum(v_mag, 1e-30), 1.0)
            mom_out = mom_out * scale[np.newaxis, :, :, :]

        return rho_out, mom_out, p_out, B_out, psi_out, rhs

    def _apply_ct_correction_numpy(
        self,
        B_new: np.ndarray,
        B_old: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply constrained transport divergence correction to B (NumPy).

        Converts cell-centred B → face-centred, builds EMFs from the
        induction-equation dB/dt, applies the CT update, then converts back
        to cell centres.  This is the same simplified CT approach used by
        the Metal solver.

        Parameters
        ----------
        B_new : np.ndarray
            Updated cell-centred B after the RK stage, shape (3, nx, ny, nz).
        B_old : np.ndarray
            Previous cell-centred B, shape (3, nx, ny, nz).
        dt : float
            Timestep [s].

        Returns
        -------
        np.ndarray
            Corrected cell-centred B, shape (3, nx, ny, nz).
        """
        from dpf.fluid.constrained_transport import (
            cell_centered_to_face,
            ct_update,
            emf_from_fluxes,
            face_to_cell_centered,
        )

        nx, ny, nz = self.grid_shape
        dx = self.dx

        # Skip CT if grid is too small for meaningful stencils
        if nx < 3 or ny < 3 or nz < 3:
            return B_new

        # Compute the effective induction dB/dt from the RK update
        dB_dt = (B_new - B_old) / max(dt, 1e-30)

        # Convert old B to face-centred
        staggered = cell_centered_to_face(
            B_old[0], B_old[1], B_old[2], dx, dx, dx,
        )

        # Build face-centred EMF contributions from dB/dt
        # The EMF at x-faces represents the induction flux for Bx, etc.
        # We construct face arrays from cell-centred dB_dt by averaging
        Fx = np.zeros((nx + 1, ny, nz))
        Fx[1:-1, :, :] = 0.5 * (dB_dt[0, :-1, :, :] + dB_dt[0, 1:, :, :])
        Fx[0, :, :] = dB_dt[0, 0, :, :]
        Fx[-1, :, :] = dB_dt[0, -1, :, :]

        Fy = np.zeros((nx, ny + 1, nz))
        Fy[:, 1:-1, :] = 0.5 * (dB_dt[1, :, :-1, :] + dB_dt[1, :, 1:, :])
        Fy[:, 0, :] = dB_dt[1, :, 0, :]
        Fy[:, -1, :] = dB_dt[1, :, -1, :]

        Fz = np.zeros((nx, ny, nz + 1))
        Fz[:, :, 1:-1] = 0.5 * (dB_dt[2, :, :, :-1] + dB_dt[2, :, :, 1:])
        Fz[:, :, 0] = dB_dt[2, :, :, 0]
        Fz[:, :, -1] = dB_dt[2, :, :, -1]

        # Build edge EMFs via simple CT averaging
        Ex_edge, Ey_edge, Ez_edge = emf_from_fluxes(Fx, Fy, Fz, dx, dx, dx)

        # Apply CT update to face-centred B
        staggered_new = ct_update(staggered, Ex_edge, Ey_edge, Ez_edge, dt)

        # Convert back to cell-centred
        Bx_cc, By_cc, Bz_cc = face_to_cell_centered(staggered_new)

        B_corrected = np.empty_like(B_new)
        B_corrected[0] = Bx_cc
        B_corrected[1] = By_cc
        B_corrected[2] = Bz_cc

        return B_corrected

    def _step_ssp_rk2_core(
        self,
        rho_n: np.ndarray,
        vel_n: np.ndarray,
        mom_n: np.ndarray,
        p_n: np.ndarray,
        B_n: np.ndarray,
        psi_n: np.ndarray,
        Te: np.ndarray,
        Ti: np.ndarray,
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None,
        apply_electrode_bc: bool,
        anode_radius: float,
        cathode_radius: float,
    ) -> tuple:
        """SSP-RK2 (Shu-Osher): 2 stages, 2nd-order.

        U^(1) = U^n + dt * L(U^n)
        U^(n+1) = 1/2*U^n + 1/2*(U^(1) + dt * L(U^(1)))
        """
        # Stage 1
        rho_1, mom_1, p_1, B_1, psi_1, rhs1 = self._euler_stage(
            rho_n, mom_n, p_n, B_n, psi_n, Te, Ti, dt, current, voltage, eta_field,
        )
        if self.use_ct:
            B_1 = self._apply_ct_correction_numpy(B_1, B_n, dt)
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(B_1, current, anode_radius, cathode_radius)

        # Stage 2
        rho_2, mom_2, p_2, B_2, psi_2, rhs2 = self._euler_stage(
            rho_1, mom_1, p_1, B_1, psi_1, Te, Ti, dt, current, voltage, eta_field,
        )

        rho_new = 0.5 * rho_n + 0.5 * rho_2
        rho_new = np.maximum(rho_new, 1e-20)
        mom_new = 0.5 * mom_n + 0.5 * mom_2
        vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :, :], 1e-30)
        p_new = 0.5 * p_n + 0.5 * p_2
        p_new = np.maximum(p_new, 1e-20)
        B_new = 0.5 * B_n + 0.5 * B_2
        if self.use_ct:
            B_new = self._apply_ct_correction_numpy(B_new, B_n, dt)
        psi_new = 0.5 * psi_n + 0.5 * psi_2

        return rho_new, mom_new, vel_new, p_new, B_new, psi_new, rhs1, rhs2

    def _step_ssp_rk3_core(
        self,
        rho_n: np.ndarray,
        vel_n: np.ndarray,
        mom_n: np.ndarray,
        p_n: np.ndarray,
        B_n: np.ndarray,
        psi_n: np.ndarray,
        Te: np.ndarray,
        Ti: np.ndarray,
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None,
        apply_electrode_bc: bool,
        anode_radius: float,
        cathode_radius: float,
    ) -> tuple:
        """SSP-RK3 (Shu-Osher 1988): 3 stages, 3rd-order.

        U^(1) = U^n + dt * L(U^n)
        U^(2) = 3/4*U^n + 1/4*(U^(1) + dt * L(U^(1)))
        U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt * L(U^(2)))

        References:
            Shu C.-W. & Osher S., J. Comput. Phys. 77, 439 (1988).
            Gottlieb S. et al., SIAM Rev. 43, 89-112 (2001).
        """
        # Stage 1: U^(1) = U^n + dt * L(U^n)
        rho_1, mom_1, p_1, B_1, psi_1, rhs1 = self._euler_stage(
            rho_n, mom_n, p_n, B_n, psi_n, Te, Ti, dt, current, voltage, eta_field,
        )
        if self.use_ct:
            B_1 = self._apply_ct_correction_numpy(B_1, B_n, dt)
        if apply_electrode_bc and cathode_radius > 0:
            B_1 = self.apply_electrode_bfield_bc(B_1, current, anode_radius, cathode_radius)

        # Stage 2: U^(2) = 3/4*U^n + 1/4*(U^(1) + dt * L(U^(1)))
        rho_2e, mom_2e, p_2e, B_2e, psi_2e, rhs2 = self._euler_stage(
            rho_1, mom_1, p_1, B_1, psi_1, Te, Ti, dt, current, voltage, eta_field,
        )
        rho_2 = 0.75 * rho_n + 0.25 * rho_2e
        rho_2 = np.maximum(rho_2, 1e-20)
        mom_2 = 0.75 * mom_n + 0.25 * mom_2e
        p_2 = 0.75 * p_n + 0.25 * p_2e
        p_2 = np.maximum(p_2, 1e-20)
        B_2 = 0.75 * B_n + 0.25 * B_2e
        if self.use_ct:
            B_2 = self._apply_ct_correction_numpy(B_2, B_n, dt)
        psi_2 = 0.75 * psi_n + 0.25 * psi_2e
        if apply_electrode_bc and cathode_radius > 0:
            B_2 = self.apply_electrode_bfield_bc(B_2, current, anode_radius, cathode_radius)

        # Stage 3: U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt * L(U^(2)))
        rho_3e, mom_3e, p_3e, B_3e, psi_3e, rhs3 = self._euler_stage(
            rho_2, mom_2, p_2, B_2, psi_2, Te, Ti, dt, current, voltage, eta_field,
        )
        rho_new = (1.0 / 3.0) * rho_n + (2.0 / 3.0) * rho_3e
        rho_new = np.maximum(rho_new, 1e-20)
        mom_new = (1.0 / 3.0) * mom_n + (2.0 / 3.0) * mom_3e
        vel_new = mom_new / np.maximum(rho_new[np.newaxis, :, :, :], 1e-30)
        p_new = (1.0 / 3.0) * p_n + (2.0 / 3.0) * p_3e
        p_new = np.maximum(p_new, 1e-20)
        B_new = (1.0 / 3.0) * B_n + (2.0 / 3.0) * B_3e
        if self.use_ct:
            B_new = self._apply_ct_correction_numpy(B_new, B_n, dt)
        psi_new = (1.0 / 3.0) * psi_n + (2.0 / 3.0) * psi_3e

        return rho_new, mom_new, vel_new, p_new, B_new, psi_new, rhs1, rhs3

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        eta_field: np.ndarray | None = None,
        anode_radius: float = 0.0,
        cathode_radius: float = 0.0,
        apply_electrode_bc: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Advance MHD state by one timestep using SSP-RK3 (default) or SSP-RK2.

        SSP-RK3 (Shu-Osher 1988, Gottlieb et al. 2001):
            U^(1) = U^n + dt * L(U^n)
            U^(2) = 3/4*U^n + 1/4*(U^(1) + dt * L(U^(1)))
            U^(n+1) = 1/3*U^n + 2/3*(U^(2) + dt * L(U^(2)))

        SSP-RK2 (Shu-Osher form):
            U^(1) = U^n + dt * L(U^n)
            U^(n+1) = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))

        After the RK step, applies:
        - Electrode B-field boundary conditions
        - Braginskii anisotropic heat flux (operator-split)
        - Two-temperature update (preserving Te ≠ Ti)
        - Circuit coupling state update with dL_dt

        Args:
            state: Dictionary with keys rho, velocity, pressure, B, Te, Ti, psi.
            dt: Timestep [s].
            current: Circuit current [A].
            voltage: Circuit voltage [V].
            eta_field: Spatially-resolved resistivity [Ohm*m], shape (nx,ny,nz).
            anode_radius: Anode radius [m] for electrode BC.
            cathode_radius: Cathode radius [m] for electrode BC.
            apply_electrode_bc: Whether to apply electrode B-field BC.

        Returns:
            Updated state dictionary.
        """
        rho = state["rho"]
        vel = state["velocity"]
        p = state["pressure"]
        B = state["B"]
        Te = state.get("Te", np.full_like(rho, 1e4))
        Ti = state.get("Ti", np.full_like(rho, 1e4))
        psi = state.get("psi", np.zeros_like(rho))
        e_electron = state.get("e_electron")

        # Save U^n
        rho_n = rho.copy()
        vel_n = vel.copy()
        p_n = p.copy()
        B_n = B.copy()
        psi_n = psi.copy()
        mom_n = rho_n[np.newaxis, :, :, :] * vel_n

        if self.time_integrator == "ssp_rk3":
            rho_new, mom_new, vel_new, p_new, B_new, psi_new, rhs1, rhs_last = (
                self._step_ssp_rk3_core(
                    rho_n, vel_n, mom_n, p_n, B_n, psi_n, Te, Ti,
                    dt, current, voltage, eta_field,
                    apply_electrode_bc, anode_radius, cathode_radius,
                )
            )
        else:
            rho_new, mom_new, vel_new, p_new, B_new, psi_new, rhs1, rhs_last = (
                self._step_ssp_rk2_core(
                    rho_n, vel_n, mom_n, p_n, B_n, psi_n, Te, Ti,
                    dt, current, voltage, eta_field,
                    apply_electrode_bc, anode_radius, cathode_radius,
                )
            )

        # Apply electrode BC after stage 2
        if apply_electrode_bc and cathode_radius > 0:
            B_new = self.apply_electrode_bfield_bc(
                B_new, current, anode_radius, cathode_radius,
            )

        # --- Two-temperature update (preserve Te ≠ Ti) ---
        # Compute total pressure change ratio to scale temperatures
        n_i = rho_new / self.ion_mass
        n_i_safe = np.maximum(n_i, 1e-30)

        if e_electron is not None:
            # We have a separate electron energy — use it to get Te
            # Advect electron energy: de_e/dt = -div(e_e * v) - p_e * div(v) + Q_ohm_e
            div_v = (
                np.gradient(vel_new[0], self.dx, axis=0)
                + np.gradient(vel_new[1], self.dx, axis=1)
                + np.gradient(vel_new[2], self.dx, axis=2)
            )
            p_e_old = n_i_safe * k_B * Te
            # Simple advection + compression
            de_e_dt = -p_e_old * div_v
            # Ohmic heating goes primarily to electrons
            ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs_last["ohmic_heating"])
            de_e_dt += ohmic_avg
            e_electron_new = e_electron + dt * de_e_dt
            e_electron_new = np.maximum(e_electron_new, n_i_safe * k_B * 1.0)  # Floor 1 K
            Te_new = (2.0 / 3.0) * e_electron_new / (n_i_safe * k_B)
            # Ion temperature from total pressure minus electron pressure
            p_e_new = n_i_safe * k_B * Te_new
            p_i_new = np.maximum(p_new - p_e_new, 1e-20)
            Ti_new = p_i_new / (n_i_safe * k_B)
        else:
            # No separate electron energy tracked — recover Te, Ti from
            # total pressure split.  T_total = p/(n_i*k_B) = Te + Ti.
            # Preserve the Te/(Te+Ti) fraction from the previous step.
            Te_old = Te
            Ti_old = Ti
            T_sum_old = np.maximum(Te_old + Ti_old, 1.0)
            f_e = Te_old / T_sum_old  # Electron fraction of total temperature

            # Total temperature from new pressure: T_total = p_new / (n_i * k_B)
            T_total_new = p_new / np.maximum(n_i_safe * k_B, 1e-30)
            Te_new = f_e * T_total_new
            Ti_new = (1.0 - f_e) * T_total_new

            # Add Ohmic heating preferentially to electrons
            ohmic_avg = 0.5 * (rhs1["ohmic_heating"] + rhs_last["ohmic_heating"])
            dTe_ohmic = (2.0 / 3.0) * ohmic_avg * dt / np.maximum(n_i_safe * k_B, 1e-30)
            Te_new = Te_new + dTe_ohmic

        Te_new = np.maximum(Te_new, 1.0)
        Ti_new = np.maximum(Ti_new, 1.0)

        # --- Braginskii anisotropic heat flux (operator-split) ---
        if self.enable_braginskii:
            ne = rho_new / self.ion_mass  # Z=1
            Te_new = _braginskii_heat_flux(Te_new, ne, B_new, self.dx, dt)

        # --- Update coupling for circuit ---
        # Lp from magnetic energy: Lp = 2*W_mag/I² = ∫B²/µ₀ dV / I²
        # Standard energy-based inductance formula. For Cartesian grids,
        # cell volume is dx³ (uniform spacing).
        if current > 0:
            B_sq = np.sum(B_new**2, axis=0)
            dV = self.dx**3
            Lp_est = float(np.sum(B_sq / mu_0 * dV)) / (current**2 + 1e-30)
        else:
            Lp_est = 0.0

        # Compute dL_dt from previous Lp
        if self._prev_Lp is not None and dt > 0:
            dL_dt = (Lp_est - self._prev_Lp) / dt
        else:
            dL_dt = 0.0
        self._prev_Lp = Lp_est

        self._coupling = CouplingState(
            Lp=Lp_est,
            current=current,
            voltage=voltage,
            dL_dt=dL_dt,
        )

        result = {
            "rho": rho_new,
            "velocity": vel_new,
            "pressure": p_new,
            "B": B_new,
            "Te": Te_new,
            "Ti": Ti_new,
            "psi": psi_new,
        }
        if e_electron is not None:
            result["e_electron"] = e_electron_new
        return result

    def coupling_interface(self) -> CouplingState:
        return self._coupling
