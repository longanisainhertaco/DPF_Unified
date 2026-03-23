"""HLL and HLLD Riemann solvers for ideal MHD on Metal/CPU.

Functions:
    hll_flux_mps   -- HLL (Harten-Lax-van Leer) approximate Riemann solver.
    hlld_flux_mps  -- HLLD (Miyoshi & Kusano 2005) 4-wave Riemann solver.
"""

from __future__ import annotations

import logging

import torch

from dpf.metal._riemann_constants import (
    IB1,
    IDN,
    IEE,
    IEN,
    IM1,
    NVAR,
    P_FLOOR,
    RHO_FLOOR,
)
from dpf.metal._riemann_nan_safety import _should_check_nan
from dpf.metal._riemann_primitives import (
    _cons_to_prim_mps,
    _fast_magnetosonic_mps,
    _physical_flux_mps,
)
from dpf.metal._utils import _ensure_mps

logger = logging.getLogger(__name__)


# ============================================================
# HLL Riemann solver (8-component, fully vectorized)
# ============================================================


def hll_flux_mps(
    UL: torch.Tensor,
    UR: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """HLL (Harten-Lax-van Leer) approximate Riemann solver for ideal MHD.

    Computes the numerical flux at cell interfaces given left and right
    reconstructed states.  The HLL flux is:

        F_HLL = (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)

    where SL and SR are the left and right wave speed estimates (Davis bounds):

        SL = min(vn_L - cf_L,  vn_R - cf_R)
        SR = max(vn_L + cf_L,  vn_R + cf_R)

    and cf is the fast magnetosonic speed.

    Args:
        UL: Left state at interfaces, shape (8, ...), float32, MPS.
        UR: Right state at interfaces, shape (8, ...), float32, MPS.
        gamma: Adiabatic index.
        dim: Normal direction (0=x, 1=y, 2=z).

    Returns:
        HLL numerical flux, shape (8, ...), float32, MPS.
    """
    _ensure_mps(UL, "UL")
    _ensure_mps(UR, "UR")

    UL = torch.where(torch.isnan(UL), torch.zeros_like(UL) + RHO_FLOOR, UL)
    UR = torch.where(torch.isnan(UR), torch.zeros_like(UR) + RHO_FLOOR, UR)

    UL_clean = UL.clone()
    UR_clean = UR.clone()
    UL_clean[IDN] = torch.clamp(UL[IDN], min=RHO_FLOOR)
    UR_clean[IDN] = torch.clamp(UR[IDN], min=RHO_FLOOR)
    UL_clean[IEN] = torch.clamp(UL[IEN], min=P_FLOOR)
    UR_clean[IEN] = torch.clamp(UR[IEN], min=P_FLOOR)
    if UL.shape[0] > NVAR:
        UL_clean[IEE] = torch.clamp(UL[IEE], min=0.0)
        UR_clean[IEE] = torch.clamp(UR[IEE], min=0.0)

    rho_L, vel_L, p_L, B_L = _cons_to_prim_mps(UL_clean, gamma)
    rho_R, vel_R, p_R, B_R = _cons_to_prim_mps(UR_clean, gamma)

    _V_MAX = 1e6
    vel_L = torch.clamp(vel_L, min=-_V_MAX, max=_V_MAX)
    vel_R = torch.clamp(vel_R, min=-_V_MAX, max=_V_MAX)

    vn_L = vel_L[dim]
    vn_R = vel_R[dim]

    cf_L = _fast_magnetosonic_mps(rho_L, p_L, B_L, gamma, dim)
    cf_R = _fast_magnetosonic_mps(rho_R, p_R, B_R, gamma, dim)

    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = torch.maximum(SR, SL + 1e-10)

    FL = _physical_flux_mps(UL_clean, gamma, dim)
    FR = _physical_flux_mps(UR_clean, gamma, dim)

    FL = torch.where(torch.isnan(FL), torch.zeros_like(FL), FL)
    FR = torch.where(torch.isnan(FR), torch.zeros_like(FR), FR)

    denom = SR - SL
    denom = torch.clamp(denom, min=1e-20)

    SL_8 = SL.unsqueeze(0)
    SR_8 = SR.unsqueeze(0)
    denom_8 = denom.unsqueeze(0)

    F_HLL = (SR_8 * FL - SL_8 * FR + SL_8 * SR_8 * (UR_clean - UL_clean)) / denom_8

    all_right = SL.unsqueeze(0) >= 0.0
    all_left = SR.unsqueeze(0) <= 0.0

    F_HLL = torch.where(all_right, FL, F_HLL)
    F_HLL = torch.where(all_left, FR, F_HLL)

    if _should_check_nan():
        has_nan = torch.isnan(F_HLL)
        if has_nan.any():
            F_LF = 0.5 * (FL + FR)
            F_HLL = torch.where(has_nan, F_LF, F_HLL)
            logger.warning("HLL flux dim=%d: %d NaN values replaced with LF flux",
                            dim, int(has_nan.sum().item()))

    return F_HLL


# ============================================================
# HLLD Riemann solver (8-component, fully vectorized)
# ============================================================


def hlld_flux_mps(
    UL: torch.Tensor,
    UR: torch.Tensor,
    gamma: float,
    dim: int,
) -> torch.Tensor:
    """HLLD (Harten-Lax-van Leer-Discontinuities) Riemann solver for MHD.

    Fully vectorized 8-component HLLD solver that resolves four intermediate
    states: two outer fast magnetosonic shocks and two inner Alfven/rotational
    discontinuities separated by a contact surface.

    The HLLD flux is selected from five regions::

        SL    SL*    SM    SR*    SR
        |------|------|------|------|
        F_L   F*_L  F**_L  F**_R  F*_R   F_R

    Following Miyoshi & Kusano, JCP 208, 315 (2005).

    Args:
        UL: Left state at interfaces, shape (8, ...), float32.
        UR: Right state at interfaces, shape (8, ...), float32.
        gamma: Adiabatic index.
        dim: Normal direction (0=x, 1=y, 2=z).

    Returns:
        HLLD numerical flux, shape (8, ...), float32.
    """
    _ensure_mps(UL, "UL")
    _ensure_mps(UR, "UR")

    im_n = IM1 + dim
    im_t1 = IM1 + (dim + 1) % 3
    im_t2 = IM1 + (dim + 2) % 3
    ib_n = IB1 + dim
    ib_t1 = IB1 + (dim + 1) % 3
    ib_t2 = IB1 + (dim + 2) % 3

    UL = torch.where(torch.isnan(UL), torch.zeros_like(UL) + RHO_FLOOR, UL)
    UR = torch.where(torch.isnan(UR), torch.zeros_like(UR) + RHO_FLOOR, UR)
    UL = UL.clone()
    UR = UR.clone()
    UL[IDN] = torch.clamp(UL[IDN], min=RHO_FLOOR)
    UR[IDN] = torch.clamp(UR[IDN], min=RHO_FLOOR)
    UL[IEN] = torch.clamp(UL[IEN], min=P_FLOOR)
    UR[IEN] = torch.clamp(UR[IEN], min=P_FLOOR)

    rho_L, vel_L, p_L, B_L = _cons_to_prim_mps(UL, gamma)
    rho_R, vel_R, p_R, B_R = _cons_to_prim_mps(UR, gamma)

    vn_L = vel_L[dim]
    vn_R = vel_R[dim]

    Bn_L = B_L[dim]
    Bn_R = B_R[dim]
    Bn = 0.5 * (Bn_L + Bn_R)

    cf_L = _fast_magnetosonic_mps(rho_L, p_L, B_L, gamma, dim)
    cf_R = _fast_magnetosonic_mps(rho_R, p_R, B_R, gamma, dim)

    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = torch.maximum(SR, SL + 1e-10)

    B_sq_L = B_L[0] ** 2 + B_L[1] ** 2 + B_L[2] ** 2
    B_sq_R = B_R[0] ** 2 + B_R[1] ** 2 + B_R[2] ** 2
    pt_L = p_L + 0.5 * B_sq_L
    pt_R = p_R + 0.5 * B_sq_R

    denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L)
    denom_SM = torch.where(
        torch.abs(denom_SM) < 1e-20,
        torch.full_like(denom_SM, 1e-20) * torch.sign(denom_SM + 1e-30),
        denom_SM,
    )
    SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM

    pt_star = pt_L + rho_L * (SL - vn_L) * (SM - vn_L)
    pt_star = torch.clamp(pt_star, min=P_FLOOR)

    denom_L = torch.clamp(torch.abs(SL - SM), min=1e-20) * torch.sign(SL - SM + 1e-30)
    denom_R = torch.clamp(torch.abs(SR - SM), min=1e-20) * torch.sign(SR - SM + 1e-30)
    rho_sL = torch.clamp(rho_L * (SL - vn_L) / denom_L, min=RHO_FLOOR)
    rho_sR = torch.clamp(rho_R * (SR - vn_R) / denom_R, min=RHO_FLOOR)

    D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn ** 2
    safe_D_L = torch.where(torch.abs(D_L) < 1e-20, torch.full_like(D_L, 1e-20), D_L)
    inv_rhoL_dSL = 1.0 / safe_D_L

    D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn ** 2
    safe_D_R = torch.where(torch.abs(D_R) < 1e-20, torch.full_like(D_R, 1e-20), D_R)
    inv_rhoR_dSR = 1.0 / safe_D_R

    Bn_small = torch.abs(Bn) < 1e-10

    vt1_sL = vel_L[(dim + 1) % 3] - Bn * B_L[(dim + 1) % 3] * (SM - vn_L) * inv_rhoL_dSL
    vt2_sL = vel_L[(dim + 2) % 3] - Bn * B_L[(dim + 2) % 3] * (SM - vn_L) * inv_rhoL_dSL
    vt1_sL = torch.where(Bn_small, vel_L[(dim + 1) % 3], vt1_sL)
    vt2_sL = torch.where(Bn_small, vel_L[(dim + 2) % 3], vt2_sL)

    vt1_sR = vel_R[(dim + 1) % 3] - Bn * B_R[(dim + 1) % 3] * (SM - vn_R) * inv_rhoR_dSR
    vt2_sR = vel_R[(dim + 2) % 3] - Bn * B_R[(dim + 2) % 3] * (SM - vn_R) * inv_rhoR_dSR
    vt1_sR = torch.where(Bn_small, vel_R[(dim + 1) % 3], vt1_sR)
    vt2_sR = torch.where(Bn_small, vel_R[(dim + 2) % 3], vt2_sR)

    Bt1_sL = B_L[(dim + 1) % 3] * (rho_L * (SL - vn_L) ** 2 - Bn ** 2) * inv_rhoL_dSL
    Bt2_sL = B_L[(dim + 2) % 3] * (rho_L * (SL - vn_L) ** 2 - Bn ** 2) * inv_rhoL_dSL
    Bt1_sL = torch.where(Bn_small, B_L[(dim + 1) % 3], Bt1_sL)
    Bt2_sL = torch.where(Bn_small, B_L[(dim + 2) % 3], Bt2_sL)

    Bt1_sR = B_R[(dim + 1) % 3] * (rho_R * (SR - vn_R) ** 2 - Bn ** 2) * inv_rhoR_dSR
    Bt2_sR = B_R[(dim + 2) % 3] * (rho_R * (SR - vn_R) ** 2 - Bn ** 2) * inv_rhoR_dSR
    Bt1_sR = torch.where(Bn_small, B_R[(dim + 1) % 3], Bt1_sR)
    Bt2_sR = torch.where(Bn_small, B_R[(dim + 2) % 3], Bt2_sR)

    vB_sL = SM * Bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL
    vB_L = (vn_L * Bn_L + vel_L[(dim + 1) % 3] * B_L[(dim + 1) % 3]
            + vel_L[(dim + 2) % 3] * B_L[(dim + 2) % 3])
    e_sL = ((SL - vn_L) * UL[IEN] - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L

    vB_sR = SM * Bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR
    vB_R = (vn_R * Bn_R + vel_R[(dim + 1) % 3] * B_R[(dim + 1) % 3]
            + vel_R[(dim + 2) % 3] * B_R[(dim + 2) % 3])
    e_sR = ((SR - vn_R) * UR[IEN] - pt_R * vn_R + pt_star * SM + Bn * (vB_R - vB_sR)) / denom_R

    U_sL = torch.zeros_like(UL)
    U_sL[IDN] = rho_sL
    U_sL[im_n] = rho_sL * SM
    U_sL[im_t1] = rho_sL * vt1_sL
    U_sL[im_t2] = rho_sL * vt2_sL
    U_sL[IEN] = e_sL
    U_sL[ib_n] = Bn
    U_sL[ib_t1] = Bt1_sL
    U_sL[ib_t2] = Bt2_sL

    U_sR = torch.zeros_like(UR)
    U_sR[IDN] = rho_sR
    U_sR[im_n] = rho_sR * SM
    U_sR[im_t1] = rho_sR * vt1_sR
    U_sR[im_t2] = rho_sR * vt2_sR
    U_sR[IEN] = e_sR
    U_sR[ib_n] = Bn
    U_sR[ib_t1] = Bt1_sR
    U_sR[ib_t2] = Bt2_sR

    has_ee = UL.shape[0] > NVAR
    if has_ee:
        denom_ee_L = torch.where(
            torch.abs(SL - SM) < 1e-20, torch.full_like(SM, 1e-20), SL - SM,
        )
        denom_ee_R = torch.where(
            torch.abs(SR - SM) < 1e-20, torch.full_like(SM, 1e-20), SR - SM,
        )
        U_sL[IEE] = torch.clamp(UL[IEE] * (SL - vn_L) / denom_ee_L, min=0.0)
        U_sR[IEE] = torch.clamp(UR[IEE] * (SR - vn_R) / denom_ee_R, min=0.0)

    FL = _physical_flux_mps(UL, gamma, dim)
    FR = _physical_flux_mps(UR, gamma, dim)
    FL = torch.where(torch.isnan(FL), torch.zeros_like(FL), FL)
    FR = torch.where(torch.isnan(FR), torch.zeros_like(FR), FR)

    SL_8 = SL.unsqueeze(0)
    SR_8 = SR.unsqueeze(0)
    F_sL = FL + SL_8 * (U_sL - UL)
    F_sR = FR + SR_8 * (U_sR - UR)

    sqrt_rho_sL = torch.sqrt(torch.clamp(rho_sL, min=RHO_FLOOR))
    sqrt_rho_sR = torch.sqrt(torch.clamp(rho_sR, min=RHO_FLOOR))
    SL_star = SM - torch.abs(Bn) / sqrt_rho_sL
    SR_star = SM + torch.abs(Bn) / sqrt_rho_sR

    sign_Bn = torch.sign(Bn + 1e-30)
    denom_ds = torch.clamp(sqrt_rho_sL + sqrt_rho_sR, min=1e-20)

    vt1_ds = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR
              + (Bt1_sR - Bt1_sL) * sign_Bn) / denom_ds
    vt2_ds = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR
              + (Bt2_sR - Bt2_sL) * sign_Bn) / denom_ds

    Bt1_ds = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL
              + sqrt_rho_sL * sqrt_rho_sR * (vt1_sR - vt1_sL) * sign_Bn) / denom_ds
    Bt2_ds = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL
              + sqrt_rho_sL * sqrt_rho_sR * (vt2_sR - vt2_sL) * sign_Bn) / denom_ds

    vt1_dsL = torch.where(Bn_small, vt1_sL, vt1_ds)
    vt2_dsL = torch.where(Bn_small, vt2_sL, vt2_ds)
    vt1_dsR = torch.where(Bn_small, vt1_sR, vt1_ds)
    vt2_dsR = torch.where(Bn_small, vt2_sR, vt2_ds)
    Bt1_ds_L = torch.where(Bn_small, Bt1_sL, Bt1_ds)
    Bt2_ds_L = torch.where(Bn_small, Bt2_sL, Bt2_ds)
    Bt1_ds_R = torch.where(Bn_small, Bt1_sR, Bt1_ds)
    Bt2_ds_R = torch.where(Bn_small, Bt2_sR, Bt2_ds)

    vB_dsL = SM * Bn + vt1_dsL * Bt1_ds_L + vt2_dsL * Bt2_ds_L
    e_dsL = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn

    vB_dsR = SM * Bn + vt1_dsR * Bt1_ds_R + vt2_dsR * Bt2_ds_R
    e_dsR = e_sR + sqrt_rho_sR * (vB_sR - vB_dsR) * sign_Bn

    U_dsL = torch.zeros_like(UL)
    U_dsL[IDN] = rho_sL
    U_dsL[im_n] = rho_sL * SM
    U_dsL[im_t1] = rho_sL * vt1_dsL
    U_dsL[im_t2] = rho_sL * vt2_dsL
    U_dsL[IEN] = e_dsL
    U_dsL[ib_n] = Bn
    U_dsL[ib_t1] = Bt1_ds_L
    U_dsL[ib_t2] = Bt2_ds_L

    U_dsR = torch.zeros_like(UR)
    U_dsR[IDN] = rho_sR
    U_dsR[im_n] = rho_sR * SM
    U_dsR[im_t1] = rho_sR * vt1_dsR
    U_dsR[im_t2] = rho_sR * vt2_dsR
    U_dsR[IEN] = e_dsR
    U_dsR[ib_n] = Bn
    U_dsR[ib_t1] = Bt1_ds_R
    U_dsR[ib_t2] = Bt2_ds_R

    if has_ee:
        U_dsL[IEE] = U_sL[IEE]
        U_dsR[IEE] = U_sR[IEE]

    SL_star_8 = SL_star.unsqueeze(0)
    SR_star_8 = SR_star.unsqueeze(0)
    F_dsL = F_sL + SL_star_8 * (U_dsL - U_sL)
    F_dsR = F_sR + SR_star_8 * (U_dsR - U_sR)

    SM_8 = SM.unsqueeze(0)

    F_HLLD = FR.clone()

    mask_sR = (SR_star_8 <= 0.0) & (SR_8 > 0.0)
    F_HLLD = torch.where(mask_sR, F_sR, F_HLLD)

    mask_dsR = (SM_8 <= 0.0) & (SR_star_8 > 0.0)
    F_HLLD = torch.where(mask_dsR, F_dsR, F_HLLD)

    mask_dsL = (SL_star_8 <= 0.0) & (SM_8 > 0.0)
    F_HLLD = torch.where(mask_dsL, F_dsL, F_HLLD)

    mask_sL = (SL_8 <= 0.0) & (SL_star_8 > 0.0)
    F_HLLD = torch.where(mask_sL, F_sL, F_HLLD)

    mask_L = SL_8 > 0.0
    F_HLLD = torch.where(mask_L, FL, F_HLLD)

    if _should_check_nan():
        has_nan = torch.isnan(F_HLLD)
        if has_nan.any():
            denom_hll = torch.clamp(SR - SL, min=1e-20).unsqueeze(0)
            F_HLL_fallback = (SR_8 * FL - SL_8 * FR + SL_8 * SR_8 * (UR - UL)) / denom_hll
            F_HLL_fallback = torch.where(torch.isnan(F_HLL_fallback), 0.5 * (FL + FR), F_HLL_fallback)
            F_HLLD = torch.where(has_nan, F_HLL_fallback, F_HLLD)
            logger.warning("HLLD flux dim=%d: %d NaN values replaced with HLL fallback",
                            dim, int(has_nan.sum().item()))

    return F_HLLD
