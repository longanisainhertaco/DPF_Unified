#!/usr/bin/env python3
"""Reproduce the HLL NaN at DPF electrode conditions (beta ~ 7e-7).

Root cause: float32 overflow in `_fast_magnetosonic_mps` discriminant.
At electrode boundaries: rho ~ RHO_FLOOR (1e-12), B_HL ~ 21,409 (Heaviside-Lorentz).
  va_sq = B_sq / rho = 4.58e8 / 1e-12 = 4.58e20
  diff  = a_sq - va_sq ~ -4.58e20
  diff^2 = 2.1e41  -->  OVERFLOW (float32 max = 3.4e38)  -->  inf
  cf = inf  -->  SL = -inf, SR = inf
  HLL numerator: SR*FL - SL*FR + SL*SR*(UR-UL) contains inf*0 = NaN and inf-inf = NaN

Fix: _CF_SQ_MAX = 9e16 clamp on va_sq and a_sq (commit a5b3728).
"""

import numpy as np
import torch

RHO_FLOOR = 1e-12
P_FLOOR = 1e-12
GAMMA = 5.0 / 3.0


def fast_magnetosonic_UNFIXED(
    rho: torch.Tensor, p: torch.Tensor, B: torch.Tensor, gamma: float, dim: int
) -> torch.Tensor:
    """Original (pre-fix) fast magnetosonic speed -- overflows at electrode."""
    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    p_safe = torch.clamp(p, min=P_FLOOR)
    inv_rho = 1.0 / rho_safe

    a_sq = gamma * p_safe * inv_rho
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    Bn_sq = B[dim] ** 2
    Bt_sq = torch.clamp(B_sq - Bn_sq, min=0.0)
    va_sq = B_sq * inv_rho

    diff = a_sq - va_sq
    discriminant = diff * diff + 4.0 * a_sq * Bt_sq * inv_rho
    discriminant = torch.clamp(discriminant, min=0.0)

    sum_sq = a_sq + va_sq
    cf_sq = 0.5 * (sum_sq + torch.sqrt(discriminant))
    cf_sq = torch.clamp(cf_sq, min=0.0)
    return torch.sqrt(cf_sq)


def fast_magnetosonic_FIXED(
    rho: torch.Tensor, p: torch.Tensor, B: torch.Tensor, gamma: float, dim: int
) -> torch.Tensor:
    """Fixed version with _CF_SQ_MAX clamp (commit a5b3728)."""
    _CF_SQ_MAX = torch.tensor(9.0e16, dtype=rho.dtype, device=rho.device)

    rho_safe = torch.clamp(rho, min=RHO_FLOOR)
    p_safe = torch.clamp(p, min=P_FLOOR)
    inv_rho = 1.0 / rho_safe

    a_sq = torch.clamp(gamma * p_safe * inv_rho, max=_CF_SQ_MAX)
    B_sq = B[0] ** 2 + B[1] ** 2 + B[2] ** 2
    Bn_sq = B[dim] ** 2
    Bt_sq = torch.clamp(B_sq - Bn_sq, min=0.0)
    va_sq = torch.clamp(B_sq * inv_rho, max=_CF_SQ_MAX)

    diff = a_sq - va_sq
    vat_sq = torch.clamp(Bt_sq * inv_rho, max=_CF_SQ_MAX)
    discriminant = diff * diff + 4.0 * a_sq * vat_sq
    discriminant = torch.clamp(discriminant, min=0.0)

    sum_sq = a_sq + va_sq
    cf_sq = 0.5 * (sum_sq + torch.sqrt(discriminant))
    cf_sq = torch.clamp(cf_sq, min=0.0, max=_CF_SQ_MAX)
    return torch.sqrt(cf_sq)


def hll_flux_manual(
    UL: torch.Tensor, UR: torch.Tensor, gamma: float, dim: int,
    cf_func,
) -> torch.Tensor:
    """Minimal HLL flux to demonstrate NaN propagation."""
    rho_L = torch.clamp(UL[0], min=RHO_FLOOR)
    rho_R = torch.clamp(UR[0], min=RHO_FLOOR)
    inv_rho_L, inv_rho_R = 1.0 / rho_L, 1.0 / rho_R

    vn_L = UL[1 + dim] * inv_rho_L
    vn_R = UR[1 + dim] * inv_rho_R
    vn_L = torch.clamp(vn_L, min=-1e6, max=1e6)
    vn_R = torch.clamp(vn_R, min=-1e6, max=1e6)

    B_L = UL[5:8]
    B_R = UR[5:8]

    KE_L = 0.5 * rho_L * vn_L ** 2
    ME_L = 0.5 * (B_L[0] ** 2 + B_L[1] ** 2 + B_L[2] ** 2)
    p_L = torch.clamp((gamma - 1) * (UL[4] - KE_L - ME_L), min=P_FLOOR)

    KE_R = 0.5 * rho_R * vn_R ** 2
    ME_R = 0.5 * (B_R[0] ** 2 + B_R[1] ** 2 + B_R[2] ** 2)
    p_R = torch.clamp((gamma - 1) * (UR[4] - KE_R - ME_R), min=P_FLOOR)

    cf_L = cf_func(rho_L, p_L, B_L, gamma, dim)
    cf_R = cf_func(rho_R, p_R, B_R, gamma, dim)

    SL = torch.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = torch.maximum(vn_L + cf_L, vn_R + cf_R)

    # Physical fluxes (simplified: only density and momentum-x for dim=0)
    p_total_L = p_L + ME_L
    p_total_R = p_R + ME_R
    FL = torch.stack([rho_L * vn_L, rho_L * vn_L ** 2 + p_total_L])
    FR = torch.stack([rho_R * vn_R, rho_R * vn_R ** 2 + p_total_R])

    denom = torch.clamp(SR - SL, min=1e-20)
    F_HLL = (SR * FL - SL * FR + SL * SR * (UR[:2] - UL[:2])) / denom
    return F_HLL, SL, SR, cf_L, cf_R


def main() -> None:
    mu0 = 4.0 * np.pi * 1e-7
    B_HL = float(np.float32(24.0 / np.sqrt(mu0)))

    print("=" * 72)
    print("HLL NaN Reproducer: DPF electrode conditions (beta ~ 7e-7)")
    print("=" * 72)
    print("B_SI  = 24.0 T")
    print(f"B_HL  = {B_HL:.1f}  (Heaviside-Lorentz code units)")
    print(f"B_sq  = {B_HL**2:.3e}")
    print(f"ME    = 0.5 * B_sq = {0.5*B_HL**2:.3e}")
    print("p     = 160 Pa  (D2 fill, 1.2 Torr)")
    print(f"beta  = p / ME = {160/(0.5*B_HL**2):.3e}")
    print(f"float32 max = {np.finfo(np.float32).max:.3e}")
    print()

    # Conservative states
    E_plasma = 160.0 / (GAMMA - 1) + 0.5 * B_HL ** 2
    E_electrode = P_FLOOR / (GAMMA - 1) + 0.5 * B_HL ** 2

    UL = torch.tensor(
        [[1e-3], [0.0], [0.0], [0.0], [E_plasma], [0.0], [B_HL], [0.0]],
        dtype=torch.float32,
    )
    UR = torch.tensor(
        [[RHO_FLOOR], [0.0], [0.0], [0.0], [E_electrode], [0.0], [B_HL], [0.0]],
        dtype=torch.float32,
    )

    # ---- UNFIXED (original code) ----
    print("--- UNFIXED (no _CF_SQ_MAX clamp) ---")
    rho_R = torch.tensor([RHO_FLOOR], dtype=torch.float32)
    p_R = torch.tensor([P_FLOOR], dtype=torch.float32)
    B_R = torch.stack(
        [torch.zeros(1), torch.tensor([B_HL], dtype=torch.float32), torch.zeros(1)]
    )
    cf_unfixed = fast_magnetosonic_UNFIXED(rho_R, p_R, B_R, GAMMA, 0)
    print(f"cf (unfixed) = {cf_unfixed.item():.3e}  [inf = {torch.isinf(cf_unfixed).item()}]")

    inv_rho = 1.0 / rho_R
    va_sq = B_HL ** 2 * inv_rho
    diff = GAMMA * P_FLOOR * inv_rho - va_sq
    diff_sq = diff * diff
    print(f"  va_sq = {va_sq.item():.3e}")
    print(f"  diff  = {diff.item():.3e}")
    print(f"  diff^2 = {diff_sq.item():.3e}  [OVERFLOW: {torch.isinf(diff_sq).item()}]")
    print("  --> cf = inf --> SL = -inf, SR = inf")

    F_unfixed, SL_u, SR_u, _, _ = hll_flux_manual(UL, UR, GAMMA, 0, fast_magnetosonic_UNFIXED)
    print(f"  SL = {SL_u.item():.3e}, SR = {SR_u.item():.3e}")
    print(f"  SL*SR = {(SL_u * SR_u).item():.3e}")
    print(f"  HLL flux = {F_unfixed.squeeze().tolist()}")
    print(f"  NaN count = {torch.isnan(F_unfixed).sum().item()} / {F_unfixed.numel()}")
    print("  ROOT CAUSE: inf*0 = NaN, inf - inf = NaN in HLL numerator")
    print()

    # ---- FIXED (with _CF_SQ_MAX clamp) ----
    print("--- FIXED (_CF_SQ_MAX = 9e16, cf_max = 3e8) ---")
    cf_fixed = fast_magnetosonic_FIXED(rho_R, p_R, B_R, GAMMA, 0)
    print(f"cf (fixed)   = {cf_fixed.item():.3e}  [inf = {torch.isinf(cf_fixed).item()}]")

    F_fixed, SL_f, SR_f, _, _ = hll_flux_manual(UL, UR, GAMMA, 0, fast_magnetosonic_FIXED)
    print(f"  SL = {SL_f.item():.3e}, SR = {SR_f.item():.3e}")
    print(f"  SL*SR = {(SL_f * SR_f).item():.3e}")
    print(f"  HLL flux = {F_fixed.squeeze().tolist()}")
    print(f"  NaN count = {torch.isnan(F_fixed).sum().item()} / {F_fixed.numel()}")
    print()

    # ---- Verify with actual solver code ----
    print("--- Verification: actual hll_flux_mps / hlld_flux_mps ---")
    import sys
    sys.path.insert(0, "/Users/anthonyzamora/dpf-unified/src")
    from dpf.metal._riemann_solvers import hll_flux_mps, hlld_flux_mps

    F_real_hll = hll_flux_mps(UL, UR, GAMMA, 0)
    F_real_hlld = hlld_flux_mps(UL, UR, GAMMA, 0)
    print(f"  hll_flux_mps  NaN = {torch.isnan(F_real_hll).any().item()}")
    print(f"  hlld_flux_mps NaN = {torch.isnan(F_real_hlld).any().item()}")
    print("  (Both should be False after commit a5b3728)")


if __name__ == "__main__":
    main()
