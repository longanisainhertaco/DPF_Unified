"""Kernel 2: HLLD Riemann Solver (Miyoshi & Kusano 2005).

MLX Metal kernel for the 4-wave HLLD approximate Riemann solver.
Resolves fast magnetosonic, Alfven, and contact waves.

State layout (10 conserved variables per cell):
  0: rho          1: rho*vr       2: rho*vz       3: rho*vtheta
  4: E_total      5: S*rho        6: Br           7: Bz
  8: Btheta       9: e_electron

For radial fluxes (dim=0): normal=r, t1=z, t2=theta
For axial fluxes  (dim=2): normal=z, t1=theta, t2=r

Key numerical safety features:
  - NaN-safe fast magnetosonic speed via (a^2-va^2)^2 + 4*a^2*Bt^2/rho
  - Lax-Friedrichs fallback where NaN detected
  - Floor clamping on density, pressure, and denominators
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

NVAR = 10
GAMMA = 5.0 / 3.0

# Variable indices
IDN = 0
IMR = 1
IMZ = 2
IMT = 3
IEN = 4
ISR = 5   # entropy tracer
IBR = 6
IBZ = 7
IBT = 8
IEE = 9

RHO_FLOOR = 1.0e-12
P_FLOOR = 1.0e-12
V_MAX = 1.0e6

# ============================================================
# MSL Kernel Source
# ============================================================

_HLLD_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

constant float RHO_FLOOR = 1.0e-12f;
constant float P_FLOOR   = 1.0e-12f;
constant float V_MAX     = 1.0e6f;
constant float TINY      = 1.0e-20f;
constant float TINY_BN   = 1.0e-10f;
constant int NVAR = 10;

// Variable indices for radial normal (dim=0):
//   normal=r(1), t1=z(2), t2=theta(3), Bn=Br(6), Bt1=Bz(7), Bt2=Btheta(8)
// For axial normal (dim=2):
//   normal=z(2), t1=theta(3), t2=r(1), Bn=Bz(7), Bt1=Btheta(8), Bt2=Br(6)

struct DimMap {
    int im_n;   // momentum component for normal direction
    int im_t1;  // momentum component for tangent-1
    int im_t2;  // momentum component for tangent-2
    int ib_n;   // B component for normal
    int ib_t1;  // B tangent-1
    int ib_t2;  // B tangent-2
};

// Fast magnetosonic speed with NaN-safe discriminant.
// Uses (a^2 - va^2)^2 + 4*a^2*(Bt1^2+Bt2^2)/rho to avoid cancellation.
inline float fast_magnetosonic(float rho, float p, float Bn, float Bt1, float Bt2, float gamma) {
    rho = max(rho, RHO_FLOOR);
    p = max(p, P_FLOOR);
    float inv_rho = 1.0f / rho;
    float a2 = gamma * p * inv_rho;
    float B2 = Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2;
    float va2 = B2 * inv_rho;
    float Bt2_sum = Bt1 * Bt1 + Bt2 * Bt2;

    // Stable discriminant: (a2 - va2)^2 + 4*a2*Bt2_sum/rho
    float diff = a2 - va2;
    float disc = diff * diff + 4.0f * a2 * Bt2_sum * inv_rho;
    disc = max(disc, 0.0f);

    float cf2 = 0.5f * (a2 + va2 + sqrt(disc));
    cf2 = max(cf2, 0.0f);
    return sqrt(cf2);
}

// Conservative → primitive conversion for a single cell
inline void cons_to_prim(
    const device float* U, uint stride, uint idx,
    int im_n, int im_t1, int im_t2,
    int ib_n, int ib_t1, int ib_t2,
    float gamma,
    thread float& rho, thread float& vn, thread float& vt1, thread float& vt2,
    thread float& p, thread float& Bn, thread float& Bt1, thread float& Bt2
) {
    rho = max(U[0 * stride + idx], RHO_FLOOR);
    float inv_rho = 1.0f / rho;
    float mn  = U[im_n  * stride + idx];
    float mt1 = U[im_t1 * stride + idx];
    float mt2 = U[im_t2 * stride + idx];
    vn  = clamp(mn  * inv_rho, -V_MAX, V_MAX);
    vt1 = clamp(mt1 * inv_rho, -V_MAX, V_MAX);
    vt2 = clamp(mt2 * inv_rho, -V_MAX, V_MAX);
    Bn  = U[ib_n  * stride + idx];
    Bt1 = U[ib_t1 * stride + idx];
    Bt2 = U[ib_t2 * stride + idx];
    float E = max(U[4 * stride + idx], P_FLOOR);
    float ke = 0.5f * rho * (vn*vn + vt1*vt1 + vt2*vt2);
    float mag = 0.5f * (Bn*Bn + Bt1*Bt1 + Bt2*Bt2);
    p = max((gamma - 1.0f) * (E - ke - mag), P_FLOOR);
}

// Physical flux in normal direction for a single cell
inline void physical_flux(
    float rho, float vn, float vt1, float vt2,
    float p, float Bn, float Bt1, float Bt2,
    float E, float gamma,
    thread float F[NVAR],
    int im_n, int im_t1, int im_t2,
    int ib_n, int ib_t1, int ib_t2
) {
    float B2 = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    float pt = p + 0.5f * B2;

    F[0]     = rho * vn;                         // mass flux
    F[im_n]  = rho * vn * vn + pt - Bn * Bn;     // normal momentum
    F[im_t1] = rho * vn * vt1 - Bn * Bt1;        // tangent-1 momentum
    F[im_t2] = rho * vn * vt2 - Bn * Bt2;        // tangent-2 momentum
    F[4]     = (E + pt) * vn - Bn * (vn*Bn + vt1*Bt1 + vt2*Bt2);  // energy
    // Entropy tracer: passive scalar, physical flux = Srho * vn
    // Srho is passed in separately; caller fills F[5] after this function.
    F[5]     = 0.0f;  // placeholder — filled by caller with Srho * vn
    F[ib_n]  = 0.0f;                              // Bn is constant (CT handles)
    F[ib_t1] = vn * Bt1 - vt1 * Bn;              // induction
    F[ib_t2] = vn * Bt2 - vt2 * Bn;              // induction
    F[9]     = 0.0f;                              // e_electron flux handled separately
}
"""

_HLLD_SOURCE = r"""
    uint r = thread_position_in_grid.x;
    uint z = thread_position_in_grid.y;

    // UL, UR: shape (NVAR, n_interfaces_r, nz) or (NVAR, nr, n_interfaces_z)
    uint n_iface = UL_shape[1];
    uint nz = UL_shape[2];
    if (r >= n_iface || z >= nz) return;

    uint stride = n_iface * nz;
    uint idx = r * nz + z;

    // Read dim parameter (0=radial, 2=axial)
    int dim = (int)dim_param[0];

    // Set up dimension mapping
    int im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2;
    if (dim == 0) {
        im_n=1; im_t1=2; im_t2=3; ib_n=6; ib_t1=7; ib_t2=8;
    } else {
        im_n=2; im_t1=3; im_t2=1; ib_n=7; ib_t1=8; ib_t2=6;
    }

    float gamma = gamma_param[0];

    // Reconstruct primitives from left/right states
    float rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L;
    float rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R;

    cons_to_prim(UL, stride, idx, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2, gamma,
                 rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L);
    cons_to_prim(UR, stride, idx, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2, gamma,
                 rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R);

    // Average normal B
    float Bn = 0.5f * (Bn_L + Bn_R);

    // Fast magnetosonic speeds
    float cf_L = fast_magnetosonic(rho_L, p_L, Bn, Bt1_L, Bt2_L, gamma);
    float cf_R = fast_magnetosonic(rho_R, p_R, Bn, Bt1_R, Bt2_R, gamma);

    // Wave speed estimates (Davis bounds)
    float SL = min(vn_L - cf_L, vn_R - cf_R);
    float SR = max(vn_L + cf_L, vn_R + cf_R);
    SR = max(SR, SL + 1.0e-10f);

    // Total pressure
    float B2_L = Bn_L*Bn_L + Bt1_L*Bt1_L + Bt2_L*Bt2_L;
    float B2_R = Bn_R*Bn_R + Bt1_R*Bt1_R + Bt2_R*Bt2_R;
    float pt_L = p_L + 0.5f * B2_L;
    float pt_R = p_R + 0.5f * B2_R;

    float E_L = max(UL[4 * stride + idx], P_FLOOR);
    float E_R = max(UR[4 * stride + idx], P_FLOOR);

    // Contact speed SM
    float denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L);
    if (abs(denom_SM) < TINY) denom_SM = TINY * (denom_SM >= 0.0f ? 1.0f : -1.0f);
    float SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM;

    // Total pressure in star region
    float pt_star = max(pt_L + rho_L * (SL - vn_L) * (SM - vn_L), P_FLOOR);

    // Star-state densities
    float denom_L = (SL - SM);
    if (abs(denom_L) < TINY) denom_L = TINY * (denom_L >= 0.0f ? 1.0f : -1.0f);
    float denom_R = (SR - SM);
    if (abs(denom_R) < TINY) denom_R = TINY * (denom_R >= 0.0f ? 1.0f : -1.0f);
    float rho_sL = max(rho_L * (SL - vn_L) / denom_L, RHO_FLOOR);
    float rho_sR = max(rho_R * (SR - vn_R) / denom_R, RHO_FLOOR);

    // Tangential velocities and B in star region
    float D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn * Bn;
    float D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn * Bn;
    bool bn_small = abs(Bn) < TINY_BN;

    float inv_DL = (abs(D_L) < TINY) ? 0.0f : (1.0f / D_L);
    float inv_DR = (abs(D_R) < TINY) ? 0.0f : (1.0f / D_R);

    float vt1_sL = bn_small ? vt1_L : vt1_L - Bn * Bt1_L * (SM - vn_L) * inv_DL;
    float vt2_sL = bn_small ? vt2_L : vt2_L - Bn * Bt2_L * (SM - vn_L) * inv_DL;
    float vt1_sR = bn_small ? vt1_R : vt1_R - Bn * Bt1_R * (SM - vn_R) * inv_DR;
    float vt2_sR = bn_small ? vt2_R : vt2_R - Bn * Bt2_R * (SM - vn_R) * inv_DR;

    float factor_L = rho_L * (SL - vn_L) * (SL - vn_L) - Bn * Bn;
    float factor_R = rho_R * (SR - vn_R) * (SR - vn_R) - Bn * Bn;
    float inv_fL = (abs(factor_L) < TINY) ? 0.0f : (1.0f / factor_L);
    float inv_fR = (abs(factor_R) < TINY) ? 0.0f : (1.0f / factor_R);

    // Note: we reuse inv_DL (same as inv_fL, D_L == factor_L)
    float Bt1_sL = bn_small ? Bt1_L : Bt1_L * factor_L * inv_DL;
    float Bt2_sL = bn_small ? Bt2_L : Bt2_L * factor_L * inv_DL;
    float Bt1_sR = bn_small ? Bt1_R : Bt1_R * factor_R * inv_DR;
    float Bt2_sR = bn_small ? Bt2_R : Bt2_R * factor_R * inv_DR;

    // Star-state energies
    float vB_L = vn_L * Bn_L + vt1_L * Bt1_L + vt2_L * Bt2_L;
    float vB_sL = SM * Bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL;
    float e_sL = ((SL - vn_L) * E_L - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L;

    float vB_R = vn_R * Bn_R + vt1_R * Bt1_R + vt2_R * Bt2_R;
    float vB_sR = SM * Bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR;
    float e_sR = ((SR - vn_R) * E_R - pt_R * vn_R + pt_star * SM + Bn * (vB_R - vB_sR)) / denom_R;

    // --- Double-star (Alfven) states ---
    float sqrt_rho_sL = sqrt(max(rho_sL, RHO_FLOOR));
    float sqrt_rho_sR = sqrt(max(rho_sR, RHO_FLOOR));
    float SL_star = SM - abs(Bn) / sqrt_rho_sL;
    float SR_star = SM + abs(Bn) / sqrt_rho_sR;

    float sign_Bn = (Bn >= 0.0f) ? 1.0f : -1.0f;
    if (abs(Bn) < TINY_BN) sign_Bn = 0.0f;

    float denom_ds = max(sqrt_rho_sL + sqrt_rho_sR, TINY);
    float vt1_ds = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR + (Bt1_sR - Bt1_sL) * sign_Bn) / denom_ds;
    float vt2_ds = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR + (Bt2_sR - Bt2_sL) * sign_Bn) / denom_ds;
    float Bt1_ds = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL + sqrt_rho_sL * sqrt_rho_sR * (vt1_sR - vt1_sL) * sign_Bn) / denom_ds;
    float Bt2_ds = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL + sqrt_rho_sL * sqrt_rho_sR * (vt2_sR - vt2_sL) * sign_Bn) / denom_ds;

    // When Bn is small, double-star = single-star
    float vt1_dsL = bn_small ? vt1_sL : vt1_ds;
    float vt2_dsL = bn_small ? vt2_sL : vt2_ds;
    float vt1_dsR = bn_small ? vt1_sR : vt1_ds;
    float vt2_dsR = bn_small ? vt2_sR : vt2_ds;
    float Bt1_dsL = bn_small ? Bt1_sL : Bt1_ds;
    float Bt2_dsL = bn_small ? Bt2_sL : Bt2_ds;
    float Bt1_dsR = bn_small ? Bt1_sR : Bt1_ds;
    float Bt2_dsR = bn_small ? Bt2_sR : Bt2_ds;

    float vB_dsL = SM * Bn + vt1_dsL * Bt1_dsL + vt2_dsL * Bt2_dsL;
    float e_dsL = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn;
    float vB_dsR = SM * Bn + vt1_dsR * Bt1_dsR + vt2_dsR * Bt2_dsR;
    float e_dsR = e_sR + sqrt_rho_sR * (vB_sR - vB_dsR) * sign_Bn;

    // --- Physical fluxes ---
    float FL[NVAR], FR[NVAR];
    physical_flux(rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L, E_L, gamma,
                  FL, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2);
    physical_flux(rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R, E_R, gamma,
                  FR, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2);

    // Entropy tracer: passive scalar, physical flux = Srho * vn
    float Srho_L = UL[5 * stride + idx];
    float Srho_R = UR[5 * stride + idx];
    FL[5] = Srho_L * vn_L;
    FR[5] = Srho_R * vn_R;

    // Electron energy
    float ee_L = UL[9 * stride + idx];
    float ee_R = UR[9 * stride + idx];

    // --- Assemble star-state conservative vectors ---
    float UsL[NVAR], UsR[NVAR];
    UsL[0] = rho_sL; UsR[0] = rho_sR;
    UsL[im_n] = rho_sL * SM;       UsR[im_n] = rho_sR * SM;
    UsL[im_t1] = rho_sL * vt1_sL;  UsR[im_t1] = rho_sR * vt1_sR;
    UsL[im_t2] = rho_sL * vt2_sL;  UsR[im_t2] = rho_sR * vt2_sR;
    UsL[4] = e_sL;                  UsR[4] = e_sR;
    UsL[5] = Srho_L * (SL - vn_L) / denom_L;  // entropy tracer star state
    UsR[5] = Srho_R * (SR - vn_R) / denom_R;
    UsL[ib_n] = Bn;                 UsR[ib_n] = Bn;
    UsL[ib_t1] = Bt1_sL;           UsR[ib_t1] = Bt1_sR;
    UsL[ib_t2] = Bt2_sL;           UsR[ib_t2] = Bt2_sR;
    UsL[9] = max(ee_L * (SL - vn_L) / denom_L, 0.0f);
    UsR[9] = max(ee_R * (SR - vn_R) / denom_R, 0.0f);

    // Double-star states
    float UdsL[NVAR], UdsR[NVAR];
    UdsL[0] = rho_sL; UdsR[0] = rho_sR;
    UdsL[im_n] = rho_sL * SM;       UdsR[im_n] = rho_sR * SM;
    UdsL[im_t1] = rho_sL * vt1_dsL; UdsR[im_t1] = rho_sR * vt1_dsR;
    UdsL[im_t2] = rho_sL * vt2_dsL; UdsR[im_t2] = rho_sR * vt2_dsR;
    UdsL[4] = e_dsL;                UdsR[4] = e_dsR;
    UdsL[5] = UsL[5];               UdsR[5] = UsR[5];  // entropy same as star
    UdsL[ib_n] = Bn;                UdsR[ib_n] = Bn;
    UdsL[ib_t1] = Bt1_dsL;          UdsR[ib_t1] = Bt1_dsR;
    UdsL[ib_t2] = Bt2_dsL;          UdsR[ib_t2] = Bt2_dsR;
    UdsL[9] = UsL[9];               UdsR[9] = UsR[9];

    // --- Star-region fluxes: F* = F + S*(U* - U) ---
    float FsL[NVAR], FsR[NVAR], FdsL[NVAR], FdsR[NVAR];
    for (int v = 0; v < NVAR; v++) {
        float uL_v = UL[v * stride + idx];
        float uR_v = UR[v * stride + idx];
        FsL[v] = FL[v] + SL * (UsL[v] - uL_v);
        FsR[v] = FR[v] + SR * (UsR[v] - uR_v);
        FdsL[v] = FsL[v] + SL_star * (UdsL[v] - UsL[v]);
        FdsR[v] = FsR[v] + SR_star * (UdsR[v] - UsR[v]);
    }

    // --- Select flux based on wave structure ---
    float F_out[NVAR];
    if (SL > 0.0f) {
        for (int v = 0; v < NVAR; v++) F_out[v] = FL[v];
    } else if (SL_star > 0.0f) {
        for (int v = 0; v < NVAR; v++) F_out[v] = FsL[v];
    } else if (SM > 0.0f) {
        for (int v = 0; v < NVAR; v++) F_out[v] = FdsL[v];
    } else if (SR_star > 0.0f) {
        for (int v = 0; v < NVAR; v++) F_out[v] = FdsR[v];
    } else if (SR > 0.0f) {
        for (int v = 0; v < NVAR; v++) F_out[v] = FsR[v];
    } else {
        for (int v = 0; v < NVAR; v++) F_out[v] = FR[v];
    }

    // --- NaN check with Lax-Friedrichs fallback ---
    bool has_nan = false;
    for (int v = 0; v < NVAR; v++) {
        if (isnan(F_out[v]) || isinf(F_out[v])) { has_nan = true; break; }
    }
    if (has_nan) {
        float S_max = max(abs(SL), abs(SR));
        for (int v = 0; v < NVAR; v++) {
            float uL_v = UL[v * stride + idx];
            float uR_v = UR[v * stride + idx];
            F_out[v] = 0.5f * (FL[v] + FR[v]) - 0.5f * S_max * (uR_v - uL_v);
        }
    }

    // --- Write output ---
    for (int v = 0; v < NVAR; v++) {
        flux[v * stride + idx] = F_out[v];
    }
"""


def _build_hlld_kernel():
    """Build and cache the HLLD Riemann solver kernel."""
    return mx.fast.metal_kernel(
        name="hlld_riemann",
        input_names=["UL", "UR", "gamma_param", "dim_param"],
        output_names=["flux"],
        source=_HLLD_SOURCE,
        header=_HLLD_HEADER,
        ensure_row_contiguous=True,
    )


_hlld_kernel = None


def hlld_flux_mlx(
    UL: mx.array,
    UR: mx.array,
    gamma: float = GAMMA,
    dim: int = 0,
) -> mx.array:
    """HLLD Riemann solver flux using Metal kernel.

    Args:
        UL: Left reconstructed state, shape (10, n_ifaces, nz), float32.
        UR: Right reconstructed state, shape (10, n_ifaces, nz), float32.
        gamma: Adiabatic index (default 5/3).
        dim: Normal direction (0=radial, 2=axial).

    Returns:
        Numerical flux, shape (10, n_ifaces, nz), float32.
    """
    global _hlld_kernel
    if _hlld_kernel is None:
        _hlld_kernel = _build_hlld_kernel()

    nvar, n_ifaces, nz = UL.shape
    gamma_param = mx.array([gamma], dtype=mx.float32)
    dim_param = mx.array([float(dim)], dtype=mx.float32)

    # Thread group: 32x8 = 256 threads
    tg_r = min(32, n_ifaces)
    tg_z = min(8, nz)
    grid_r = ((n_ifaces + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    outputs = _hlld_kernel(
        inputs=[UL, UR, gamma_param, dim_param],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, n_ifaces, nz)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ============================================================
# NumPy Reference Implementation
# ============================================================


def _cons_to_prim_np(
    U: np.ndarray, gamma: float, dim: int
) -> tuple[np.ndarray, ...]:
    """Conservative to primitive for numpy arrays."""
    im_n = 1 + dim if dim < 2 else 2
    im_t1 = 1 + (dim + 1) % 3
    im_t2 = 1 + (dim + 2) % 3
    ib_n = 6 + dim if dim < 2 else 7
    ib_t1 = 6 + (dim + 1) % 3
    ib_t2 = 6 + (dim + 2) % 3

    rho = np.maximum(U[0], RHO_FLOOR)
    inv_rho = 1.0 / rho
    vn = np.clip(U[im_n] * inv_rho, -V_MAX, V_MAX)
    vt1 = np.clip(U[im_t1] * inv_rho, -V_MAX, V_MAX)
    vt2 = np.clip(U[im_t2] * inv_rho, -V_MAX, V_MAX)
    Bn = U[ib_n]
    Bt1 = U[ib_t1]
    Bt2 = U[ib_t2]
    E = np.maximum(U[4], P_FLOOR)
    ke = 0.5 * rho * (vn**2 + vt1**2 + vt2**2)
    mag = 0.5 * (Bn**2 + Bt1**2 + Bt2**2)
    p = np.maximum((gamma - 1.0) * (E - ke - mag), P_FLOOR)

    # Map back to (im_n, im_t1, im_t2) → actual indices
    return rho, vn, vt1, vt2, p, Bn, Bt1, Bt2, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2


def _fast_ms_np(rho, p, Bn, Bt1, Bt2, gamma):
    """Fast magnetosonic speed with safe discriminant."""
    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    inv_rho = 1.0 / rho
    a2 = gamma * p * inv_rho
    B2 = Bn**2 + Bt1**2 + Bt2**2
    va2 = B2 * inv_rho
    Bt2_sum = Bt1**2 + Bt2**2
    diff = a2 - va2
    disc = diff**2 + 4.0 * a2 * Bt2_sum * inv_rho
    disc = np.maximum(disc, 0.0)
    cf2 = 0.5 * (a2 + va2 + np.sqrt(disc))
    return np.sqrt(np.maximum(cf2, 0.0))


def hlld_flux_numpy(
    UL: np.ndarray,
    UR: np.ndarray,
    gamma: float = GAMMA,
    dim: int = 0,
) -> np.ndarray:
    """Reference NumPy HLLD Riemann solver (vectorized).

    Args:
        UL: Left state, shape (10, n_ifaces, nz), float32.
        UR: Right state, shape (10, n_ifaces, nz), float32.
        gamma: Adiabatic index.
        dim: Normal direction (0 or 2).

    Returns:
        Numerical flux, shape (10, n_ifaces, nz), float32.
    """
    UL = UL.astype(np.float64)
    UR = UR.astype(np.float64)
    TINY = 1e-20
    TINY_BN = 1e-10
    eps = 1e-30

    (rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L,
     im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2) = _cons_to_prim_np(UL, gamma, dim)
    (rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R,
     *_) = _cons_to_prim_np(UR, gamma, dim)

    Bn = 0.5 * (Bn_L + Bn_R)

    cf_L = _fast_ms_np(rho_L, p_L, Bn, Bt1_L, Bt2_L, gamma)
    cf_R = _fast_ms_np(rho_R, p_R, Bn, Bt1_R, Bt2_R, gamma)

    SL = np.minimum(vn_L - cf_L, vn_R - cf_R)
    SR = np.maximum(vn_L + cf_L, vn_R + cf_R)
    SR = np.maximum(SR, SL + 1e-10)

    B2_L = Bn_L**2 + Bt1_L**2 + Bt2_L**2
    B2_R = Bn_R**2 + Bt1_R**2 + Bt2_R**2
    pt_L = p_L + 0.5 * B2_L
    pt_R = p_R + 0.5 * B2_R

    E_L = np.maximum(UL[4], P_FLOOR)
    E_R = np.maximum(UR[4], P_FLOOR)

    denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L)
    denom_SM = np.where(np.abs(denom_SM) < TINY,
                        TINY * np.sign(denom_SM + eps), denom_SM)
    SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM

    pt_star = np.maximum(pt_L + rho_L * (SL - vn_L) * (SM - vn_L), P_FLOOR)

    denom_L = SL - SM
    denom_L = np.where(np.abs(denom_L) < TINY, TINY * np.sign(denom_L + eps), denom_L)
    denom_R = SR - SM
    denom_R = np.where(np.abs(denom_R) < TINY, TINY * np.sign(denom_R + eps), denom_R)

    rho_sL = np.maximum(rho_L * (SL - vn_L) / denom_L, RHO_FLOOR)
    rho_sR = np.maximum(rho_R * (SR - vn_R) / denom_R, RHO_FLOOR)

    D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn**2
    D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn**2
    bn_small = np.abs(Bn) < TINY_BN
    inv_DL = np.where(np.abs(D_L) < TINY, 0.0, 1.0 / np.where(np.abs(D_L) < TINY, 1.0, D_L))
    inv_DR = np.where(np.abs(D_R) < TINY, 0.0, 1.0 / np.where(np.abs(D_R) < TINY, 1.0, D_R))

    vt1_sL = np.where(bn_small, vt1_L, vt1_L - Bn * Bt1_L * (SM - vn_L) * inv_DL)
    vt2_sL = np.where(bn_small, vt2_L, vt2_L - Bn * Bt2_L * (SM - vn_L) * inv_DL)
    vt1_sR = np.where(bn_small, vt1_R, vt1_R - Bn * Bt1_R * (SM - vn_R) * inv_DR)
    vt2_sR = np.where(bn_small, vt2_R, vt2_R - Bn * Bt2_R * (SM - vn_R) * inv_DR)

    fL = rho_L * (SL - vn_L)**2 - Bn**2
    fR = rho_R * (SR - vn_R)**2 - Bn**2
    Bt1_sL = np.where(bn_small, Bt1_L, Bt1_L * fL * inv_DL)
    Bt2_sL = np.where(bn_small, Bt2_L, Bt2_L * fL * inv_DL)
    Bt1_sR = np.where(bn_small, Bt1_R, Bt1_R * fR * inv_DR)
    Bt2_sR = np.where(bn_small, Bt2_R, Bt2_R * fR * inv_DR)

    vB_L = vn_L * Bn_L + vt1_L * Bt1_L + vt2_L * Bt2_L
    vB_sL = SM * Bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL
    e_sL = ((SL - vn_L) * E_L - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L

    vB_R = vn_R * Bn_R + vt1_R * Bt1_R + vt2_R * Bt2_R
    vB_sR = SM * Bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR
    e_sR = ((SR - vn_R) * E_R - pt_R * vn_R + pt_star * SM + Bn * (vB_R - vB_sR)) / denom_R

    # Double-star states
    sqrt_rho_sL = np.sqrt(np.maximum(rho_sL, RHO_FLOOR))
    sqrt_rho_sR = np.sqrt(np.maximum(rho_sR, RHO_FLOOR))
    SL_star = SM - np.abs(Bn) / sqrt_rho_sL
    SR_star = SM + np.abs(Bn) / sqrt_rho_sR

    sign_Bn = np.where(np.abs(Bn) < TINY_BN, 0.0, np.sign(Bn + eps))
    denom_ds = np.maximum(sqrt_rho_sL + sqrt_rho_sR, TINY)
    vt1_ds = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR + (Bt1_sR - Bt1_sL) * sign_Bn) / denom_ds
    vt2_ds = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR + (Bt2_sR - Bt2_sL) * sign_Bn) / denom_ds
    Bt1_ds = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL + sqrt_rho_sL * sqrt_rho_sR * (vt1_sR - vt1_sL) * sign_Bn) / denom_ds
    Bt2_ds = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL + sqrt_rho_sL * sqrt_rho_sR * (vt2_sR - vt2_sL) * sign_Bn) / denom_ds

    vt1_dsL = np.where(bn_small, vt1_sL, vt1_ds)
    vt2_dsL = np.where(bn_small, vt2_sL, vt2_ds)
    vt1_dsR = np.where(bn_small, vt1_sR, vt1_ds)
    vt2_dsR = np.where(bn_small, vt2_sR, vt2_ds)
    Bt1_dsL = np.where(bn_small, Bt1_sL, Bt1_ds)
    Bt2_dsL = np.where(bn_small, Bt2_sL, Bt2_ds)
    Bt1_dsR = np.where(bn_small, Bt1_sR, Bt1_ds)
    Bt2_dsR = np.where(bn_small, Bt2_sR, Bt2_ds)

    vB_dsL = SM * Bn + vt1_dsL * Bt1_dsL + vt2_dsL * Bt2_dsL
    e_dsL = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn
    vB_dsR = SM * Bn + vt1_dsR * Bt1_dsR + vt2_dsR * Bt2_dsR
    e_dsR = e_sR + sqrt_rho_sR * (vB_sR - vB_dsR) * sign_Bn

    # Build star/double-star conservative vectors

    def _build_state(rho_s, vt1_s, vt2_s, e_s, Bt1_s, Bt2_s, Srho_src, ee_src, S_wave, vn_src):
        U_s = np.zeros_like(UL)
        U_s[0] = rho_s
        U_s[im_n] = rho_s * SM
        U_s[im_t1] = rho_s * vt1_s
        U_s[im_t2] = rho_s * vt2_s
        U_s[4] = e_s
        denom_s = np.where(np.abs(S_wave - SM) < TINY, TINY, S_wave - SM)
        U_s[5] = Srho_src * (S_wave - vn_src) / denom_s
        U_s[ib_n] = Bn
        U_s[ib_t1] = Bt1_s
        U_s[ib_t2] = Bt2_s
        U_s[9] = np.maximum(ee_src * (S_wave - vn_src) / denom_s, 0.0)
        return U_s

    U_sL = _build_state(rho_sL, vt1_sL, vt2_sL, e_sL, Bt1_sL, Bt2_sL, UL[5], UL[9], SL, vn_L)
    U_sR = _build_state(rho_sR, vt1_sR, vt2_sR, e_sR, Bt1_sR, Bt2_sR, UR[5], UR[9], SR, vn_R)

    U_dsL = _build_state(rho_sL, vt1_dsL, vt2_dsL, e_dsL, Bt1_dsL, Bt2_dsL, UL[5], UL[9], SL, vn_L)
    U_dsR = _build_state(rho_sR, vt1_dsR, vt2_dsR, e_dsR, Bt1_dsR, Bt2_dsR, UR[5], UR[9], SR, vn_R)

    # Physical fluxes
    def _phys_flux(rho, vn, vt1, vt2, p, Bn_v, Bt1_v, Bt2_v, E):
        F = np.zeros_like(UL)
        B2 = Bn_v**2 + Bt1_v**2 + Bt2_v**2
        pt = p + 0.5 * B2
        F[0] = rho * vn
        F[im_n] = rho * vn * vn + pt - Bn_v * Bn_v
        F[im_t1] = rho * vn * vt1 - Bn_v * Bt1_v
        F[im_t2] = rho * vn * vt2 - Bn_v * Bt2_v
        F[4] = (E + pt) * vn - Bn_v * (vn * Bn_v + vt1 * Bt1_v + vt2 * Bt2_v)
        F[5] = 0.0  # placeholder, filled after
        F[ib_n] = 0.0
        F[ib_t1] = vn * Bt1_v - vt1 * Bn_v
        F[ib_t2] = vn * Bt2_v - vt2 * Bn_v
        F[9] = 0.0
        return F

    FL = _phys_flux(rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L, E_L)
    FR = _phys_flux(rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R, E_R)
    FL[5] = UL[5] * vn_L  # entropy tracer flux = Srho * vn
    FR[5] = UR[5] * vn_R

    F_sL = FL + SL[np.newaxis] * (U_sL - UL)
    F_sR = FR + SR[np.newaxis] * (U_sR - UR)
    F_dsL = F_sL + SL_star[np.newaxis] * (U_dsL - U_sL)
    F_dsR = F_sR + SR_star[np.newaxis] * (U_dsR - U_sR)

    # Region selection
    F_out = FR.copy()
    F_out = np.where(SR[np.newaxis] <= 0, FR, F_out)
    mask = (SR_star[np.newaxis] <= 0) & (SR[np.newaxis] > 0)
    F_out = np.where(mask, F_sR, F_out)
    mask = (SM[np.newaxis] <= 0) & (SR_star[np.newaxis] > 0)
    F_out = np.where(mask, F_dsR, F_out)
    mask = (SL_star[np.newaxis] <= 0) & (SM[np.newaxis] > 0)
    F_out = np.where(mask, F_dsL, F_out)
    mask = (SL[np.newaxis] <= 0) & (SL_star[np.newaxis] > 0)
    F_out = np.where(mask, F_sL, F_out)
    mask = SL[np.newaxis] > 0
    F_out = np.where(mask, FL, F_out)

    # NaN fallback: Lax-Friedrichs
    has_nan = np.isnan(F_out) | np.isinf(F_out)
    if np.any(has_nan):
        S_max = np.maximum(np.abs(SL), np.abs(SR))
        F_LF = 0.5 * (FL + FR) - 0.5 * S_max[np.newaxis] * (UR - UL)
        F_out = np.where(has_nan, F_LF, F_out)

    return F_out.astype(np.float32)
