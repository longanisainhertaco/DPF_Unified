"""Metal compute kernels for the MLX MHD solver.

Three kernels handle the compute-intensive operations:
1. Ghost cell padding with electrode boundary conditions
2. HLLD Riemann solver (Miyoshi & Kusano 2005)
3. Cylindrical geometric source terms with L'Hopital at axis

Each kernel has a NumPy reference implementation for testing
and a Metal GPU implementation via mx.fast.metal_kernel().
MLX Metal kernels fall back to NumPy if compilation fails.
"""

from __future__ import annotations

import math

import numpy as np

# --- Constants ---
GAMMA = 5.0 / 3.0
NVAR = 10
IDN, IMR, IMZ, IMT, IEN, ISR, IBR, IBZ, IBT, IEE = range(10)

# HLLD numerical floors
RHO_FLOOR = 1.0e-12
P_FLOOR = 1.0e-12
V_MAX = 1.0e6
MU0 = 4.0 * math.pi * 1e-7

# ──────────────────────────────────────────────────────────────
# MLX availability guard
# ──────────────────────────────────────────────────────────────
HAS_MLX_KERNELS = False
try:
    import mlx.core as mx  # noqa: E402

    # Probe that Metal kernel compilation works by building a trivial kernel.
    _probe = mx.fast.metal_kernel(
        name="_dpf_probe",
        input_names=["x"],
        output_names=["y"],
        source="uint i=thread_position_in_grid.x; y[i]=x[i];",
        header="#include <metal_stdlib>\nusing namespace metal;",
        ensure_row_contiguous=True,
    )
    _probe(
        inputs=[mx.array([1.0], dtype=mx.float32)],
        template=[],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.float32],
    )
    HAS_MLX_KERNELS = True
except Exception:
    HAS_MLX_KERNELS = False


# ══════════════════════════════════════════════════════════════
# 1. Ghost Cell Padding
# ══════════════════════════════════════════════════════════════

_GHOST_HEADER = """
#include <metal_stdlib>
using namespace metal;

constant float MU0 = 1.2566370614359173e-6f;
constant float GAMMA_GHOST = 5.0f / 3.0f;
constant float P_FLOOR_GHOST = 1.0e-12f;
constant float BETA_FLOOR = 1.0e-4f;
constant float RHO_FLOOR_GHOST = 1.0e-4f;
constant int NG_MAX = 8;
constant int NVAR = 10;
constant int IDN = 0;
constant int IMR = 1;
constant int IMZ = 2;
constant int IMT = 3;
constant int IEN = 4;
constant int IBR = 6;
constant int IBZ = 7;
constant int IBT = 8;
"""

_GHOST_SOURCE = """
    uint r_out = thread_position_in_grid.x;
    uint z_out = thread_position_in_grid.y;

    uint nr    = state_shape[1];
    uint nz    = state_shape[2];
    int  ng    = (int)params[3];
    uint nr_g  = nr + 2 * (uint)ng;

    if (r_out >= nr_g || z_out >= nz) return;

    uint in_stride_var  = nr * nz;
    uint out_stride_var = nr_g * nz;

    int r_interior = (int)r_out - ng;

    // First pass: copy all variables with standard BCs
    for (int v = 0; v < NVAR; v++) {
        float val = 0.0f;

        if (r_interior >= 0 && r_interior < (int)nr) {
            val = state[v * in_stride_var + (uint)r_interior * nz + z_out];
        } else if (r_interior < 0) {
            int mirror = -r_interior - 1;
            if (mirror >= (int)nr) mirror = (int)nr - 1;
            val = state[v * in_stride_var + (uint)mirror * nz + z_out];
            if (v == IMR || v == IBR || v == IBT || v == IMT) val = -val;
        } else {
            int src = (int)nr - 1;
            val = state[v * in_stride_var + (uint)src * nz + z_out];
            if (v == IMR || v == IBR) val = 0.0f;
            if (v == IBT) {
                float current = params[0];
                if (metal::abs(current) > 1.0e-10f) {
                    float r_inner = params[1];
                    float dr      = params[2];
                    float r_pos   = r_inner + ((float)r_out - (float)ng + 0.5f) * dr;
                    r_pos = metal::max(r_pos, 1.0e-10f);
                    val = MU0 * current / (2.0f * M_PI_F * r_pos);
                }
            }
        }

        padded[v * out_stride_var + r_out * nz + z_out] = val;
    }

    // Second pass: fix energy consistency in outer electrode ghost cells.
    // The first pass copied E from the last interior cell but injected a
    // new B_theta.  Without updating E, pressure = (gamma-1)(E - KE - B^2/2)
    // goes negative, causing NaN in the HLLD Riemann solver.
    if (r_interior >= (int)nr) {
        float current = params[0];
        if (metal::abs(current) > 1.0e-10f) {
            uint oidx = r_out * nz + z_out;
            float Br  = padded[IBR * out_stride_var + oidx];
            float Bz  = padded[IBZ * out_stride_var + oidx];
            float Bt  = padded[IBT * out_stride_var + oidx];
            float B2_new = Br*Br + Bz*Bz + Bt*Bt;

            // B^2 from the source cell (last interior) before electrode injection
            int src = (int)nr - 1;
            uint sidx = (uint)src * nz + z_out;
            float Br_old = 0.0f;  // was zeroed
            float Bz_old = state[IBZ * in_stride_var + sidx];
            float Bt_old = state[IBT * in_stride_var + sidx];
            float B2_old = Br_old*Br_old + Bz_old*Bz_old + Bt_old*Bt_old;

            // Update total energy: add magnetic energy difference
            float E_val = padded[IEN * out_stride_var + oidx];
            E_val += 0.5f * (B2_new - B2_old);

            // Enforce minimum plasma beta
            float p_mag = 0.5f * B2_new;
            float p_min = BETA_FLOOR * metal::max(p_mag, P_FLOOR_GHOST);
            float E_floor = p_min / (GAMMA_GHOST - 1.0f) + 0.5f * B2_new;
            E_val = metal::max(E_val, E_floor);
            padded[IEN * out_stride_var + oidx] = E_val;

            // Density floor: prevent extreme Alfven speed
            float rho_val = padded[IDN * out_stride_var + oidx];
            rho_val = metal::max(rho_val, RHO_FLOOR_GHOST);
            padded[IDN * out_stride_var + oidx] = rho_val;
        }
    }
"""

_ghost_kernel_cache: object = None


def _get_ghost_kernel() -> object:
    global _ghost_kernel_cache
    if _ghost_kernel_cache is None:
        _ghost_kernel_cache = mx.fast.metal_kernel(
            name="dpf_ghost_pad",
            input_names=["state", "params"],
            output_names=["padded"],
            source=_GHOST_SOURCE,
            header=_GHOST_HEADER,
            ensure_row_contiguous=True,
        )
    return _ghost_kernel_cache


def ghost_pad_numpy(
    Q: np.ndarray,
    ng: int,
    bc_type: str,
    current: float = 0.0,
    r_face: np.ndarray | None = None,
    mu0: float = MU0,
) -> np.ndarray:
    """NumPy reference: pad Q[NVAR, nr, nz] with ng ghost cells.

    Args:
        Q: State array, shape (NVAR, nr, nz), float32.
        ng: Number of ghost cells.
        bc_type: One of "outflow", "reflecting", "electrode".
        current: Circuit current [A] (electrode BC only).
        r_face: Cell-centre radii, shape (nr + 2*ng,) (electrode BC only).
        mu0: Permeability of free space [H/m].

    Returns:
        Padded array, shape (NVAR, nr + 2*ng, nz), float32.
    """
    nvar, nr, nz = Q.shape
    nr_g = nr + 2 * ng
    padded = np.zeros((nvar, nr_g, nz), dtype=np.float32)

    padded[:, ng : ng + nr, :] = Q

    # Inner ghosts (axis): reflecting
    for ig in range(ng):
        mirror = ng - 1 - ig
        src_idx = min(mirror, nr - 1)
        padded[:, ig, :] = Q[:, src_idx, :]
        for v in [IMR, IBR, IBT, IMT]:
            padded[v, ig, :] = -padded[v, ig, :]

    # Outer ghosts
    for ig in range(ng):
        out_idx = ng + nr + ig
        if bc_type == "reflecting":
            mirror = nr - 1 - ig
            src_idx = max(mirror, 0)
            padded[:, out_idx, :] = Q[:, src_idx, :]
            for v in [IMR, IBR]:
                padded[v, out_idx, :] = -padded[v, out_idx, :]
        else:
            # outflow or electrode: zero-gradient base
            padded[:, out_idx, :] = Q[:, nr - 1, :]
            padded[IMR, out_idx, :] = 0.0
            padded[IBR, out_idx, :] = 0.0
            if bc_type == "electrode" and abs(current) > 1e-10:
                if r_face is not None:
                    r_pos = float(r_face[out_idx])
                else:
                    r_pos = max(out_idx * 1e-3, 1e-10)
                r_pos = max(r_pos, 1e-10)
                # Old B^2 before electrode injection
                B2_old = (padded[IBR, out_idx, :] ** 2
                          + padded[IBZ, out_idx, :] ** 2
                          + padded[IBT, out_idx, :] ** 2)
                # Inject electrode B_theta
                padded[IBT, out_idx, :] = mu0 * current / (2.0 * math.pi * r_pos)
                # New B^2 after electrode injection
                B2_new = (padded[IBR, out_idx, :] ** 2
                          + padded[IBZ, out_idx, :] ** 2
                          + padded[IBT, out_idx, :] ** 2)
                # Update total energy to account for magnetic energy change.
                # Without this, p = (gamma-1)(E - KE - 0.5*B^2) goes negative
                # because E was copied from fill gas but B^2 is now dominated
                # by the electrode field.
                padded[IEN, out_idx, :] += 0.5 * (B2_new - B2_old)
                # Enforce minimum plasma beta to prevent extreme wavespeeds
                p_mag = 0.5 * B2_new
                beta_floor = 1e-4
                p_min = beta_floor * np.maximum(p_mag, P_FLOOR)
                E_floor = p_min / (GAMMA - 1.0) + 0.5 * B2_new
                padded[IEN, out_idx, :] = np.maximum(
                    padded[IEN, out_idx, :], E_floor
                )
                # Density floor: prevent extreme Alfven speed in ghost cells
                rho_floor = np.maximum(Q[IDN, nr - 1, :], 1e-4)
                padded[IDN, out_idx, :] = np.maximum(
                    padded[IDN, out_idx, :], rho_floor
                )

    return padded


def ghost_pad_mlx(
    Q: mx.array,
    ng: int,
    bc_type: str,
    current: float = 0.0,
    r_face: np.ndarray | None = None,
    mu0: float = MU0,
) -> mx.array:
    """MLX Metal kernel: ghost cell padding on GPU.

    Falls back to NumPy reference if MLX kernels unavailable.

    Args:
        Q: State array, shape (NVAR, nr, nz), float32 mx.array.
        ng: Number of ghost cells.
        bc_type: One of "outflow", "reflecting", "electrode".
        current: Circuit current [A].
        r_face: Cell-centre radii, shape (nr + 2*ng,).
        mu0: Permeability of free space [H/m].

    Returns:
        Padded mx.array, shape (NVAR, nr + 2*ng, nz), float32.
    """
    if not HAS_MLX_KERNELS:
        result = ghost_pad_numpy(np.asarray(Q), ng, bc_type, current, r_face, mu0)
        return mx.array(result)

    # Electrode BC: derive r_inner and dr from r_face if provided
    r_inner = 0.0
    dr = 1e-3
    if r_face is not None and len(r_face) >= ng + 1:
        # r_face covers the full padded grid; interior starts at index ng
        r_inner_interior = float(r_face[ng])
        dr = float(r_face[ng + 1] - r_face[ng]) if len(r_face) > ng + 1 else 1e-3
        # r_inner for ghost formula: outer ghost at out_idx = ng+nr+ig
        # r_pos = r_inner + (out_idx - ng + 0.5)*dr
        # With r_inner_interior = r_face[ng] = (0+0.5)*dr → r_inner_interior = 0.5*dr
        r_inner = r_inner_interior - 0.5 * dr

    nvar, nr, nz = Q.shape
    nr_g = nr + 2 * ng

    params = mx.array([current, r_inner, dr, float(ng)], dtype=mx.float32)

    tg_r = min(32, nr_g)
    tg_z = min(8, nz)
    grid_r = ((nr_g + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    kernel = _get_ghost_kernel()
    outputs = kernel(
        inputs=[Q, params],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, nr_g, nz)],
        output_dtypes=[mx.float32],
    )
    result = outputs[0]

    # Reflecting outer BC not handled by MSL kernel; patch via NumPy if needed
    if bc_type == "reflecting":
        arr = np.asarray(result)
        Q_np = np.asarray(Q)
        for ig in range(ng):
            out_idx = ng + nr + ig
            mirror = nr - 1 - ig
            src_idx = max(mirror, 0)
            arr[:, out_idx, :] = Q_np[:, src_idx, :]
            arr[IMR, out_idx, :] = -arr[IMR, out_idx, :]
            arr[IBR, out_idx, :] = -arr[IBR, out_idx, :]
        result = mx.array(arr)

    return result


# ══════════════════════════════════════════════════════════════
# 2. HLLD Riemann Solver
# ══════════════════════════════════════════════════════════════

_HLLD_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

constant float RHO_FLOOR = 1.0e-12f;
constant float P_FLOOR   = 1.0e-12f;
constant float V_MAX     = 1.0e6f;
constant float TINY      = 1.0e-20f;
constant float TINY_BN   = 1.0e-10f;
constant int NVAR = 10;

inline float fast_magnetosonic(float rho, float p, float Bn, float Bt1, float Bt2, float gamma) {
    rho = max(rho, RHO_FLOOR);
    p   = max(p,   P_FLOOR);
    float inv_rho  = 1.0f / rho;
    float a2       = gamma * p * inv_rho;
    float B2       = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    float va2      = B2 * inv_rho;
    float Bt2_sum  = Bt1*Bt1 + Bt2*Bt2;
    float diff     = a2 - va2;
    float disc     = diff*diff + 4.0f * a2 * Bt2_sum * inv_rho;
    disc = max(disc, 0.0f);
    float cf2 = 0.5f * (a2 + va2 + sqrt(disc));
    return sqrt(max(cf2, 0.0f));
}

inline void cons_to_prim(
    const device float* U, uint stride, uint idx,
    int im_n, int im_t1, int im_t2, int ib_n, int ib_t1, int ib_t2,
    float gamma,
    thread float& rho, thread float& vn, thread float& vt1, thread float& vt2,
    thread float& p, thread float& Bn, thread float& Bt1, thread float& Bt2
) {
    rho = max(U[0 * stride + idx], RHO_FLOOR);
    float inv_rho = 1.0f / rho;
    vn  = clamp(U[im_n  * stride + idx] * inv_rho, -V_MAX, V_MAX);
    vt1 = clamp(U[im_t1 * stride + idx] * inv_rho, -V_MAX, V_MAX);
    vt2 = clamp(U[im_t2 * stride + idx] * inv_rho, -V_MAX, V_MAX);
    Bn  = U[ib_n  * stride + idx];
    Bt1 = U[ib_t1 * stride + idx];
    Bt2 = U[ib_t2 * stride + idx];
    float E  = max(U[4 * stride + idx], P_FLOOR);
    float ke = 0.5f * rho * (vn*vn + vt1*vt1 + vt2*vt2);
    float mag = 0.5f * (Bn*Bn + Bt1*Bt1 + Bt2*Bt2);
    p = max((gamma - 1.0f) * (E - ke - mag), P_FLOOR);
}

inline void physical_flux(
    float rho, float vn, float vt1, float vt2,
    float p, float Bn, float Bt1, float Bt2, float E, float gamma,
    thread float F[10],
    int im_n, int im_t1, int im_t2, int ib_n, int ib_t1, int ib_t2
) {
    float B2 = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    float pt = p + 0.5f * B2;
    F[0]     = rho * vn;
    F[im_n]  = rho * vn * vn + pt - Bn * Bn;
    F[im_t1] = rho * vn * vt1 - Bn * Bt1;
    F[im_t2] = rho * vn * vt2 - Bn * Bt2;
    F[4]     = (E + pt) * vn - Bn * (vn*Bn + vt1*Bt1 + vt2*Bt2);
    F[5]     = 0.0f;
    F[ib_n]  = 0.0f;
    F[ib_t1] = vn * Bt1 - vt1 * Bn;
    F[ib_t2] = vn * Bt2 - vt2 * Bn;
    F[9]     = 0.0f;
}
"""

_HLLD_SOURCE = r"""
    uint r = thread_position_in_grid.x;
    uint z = thread_position_in_grid.y;

    uint n_iface = UL_shape[1];
    uint nz      = UL_shape[2];
    if (r >= n_iface || z >= nz) return;

    uint stride = n_iface * nz;
    uint idx    = r * nz + z;

    int dim = (int)dim_param[0];
    int im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2;
    if (dim == 0) {
        im_n=1; im_t1=2; im_t2=3; ib_n=6; ib_t1=7; ib_t2=8;
    } else {
        im_n=2; im_t1=3; im_t2=1; ib_n=7; ib_t1=8; ib_t2=6;
    }
    float gamma = gamma_param[0];

    float rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L;
    float rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R;

    cons_to_prim(UL, stride, idx, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2, gamma,
                 rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L);
    cons_to_prim(UR, stride, idx, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2, gamma,
                 rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R);

    float Bn = 0.5f * (Bn_L + Bn_R);

    float cf_L = fast_magnetosonic(rho_L, p_L, Bn, Bt1_L, Bt2_L, gamma);
    float cf_R = fast_magnetosonic(rho_R, p_R, Bn, Bt1_R, Bt2_R, gamma);

    float SL = min(vn_L - cf_L, vn_R - cf_R);
    float SR = max(vn_L + cf_L, vn_R + cf_R);
    SR = max(SR, SL + 1.0e-10f);

    float B2_L = Bn_L*Bn_L + Bt1_L*Bt1_L + Bt2_L*Bt2_L;
    float B2_R = Bn_R*Bn_R + Bt1_R*Bt1_R + Bt2_R*Bt2_R;
    float pt_L = p_L + 0.5f * B2_L;
    float pt_R = p_R + 0.5f * B2_R;

    float E_L = max(UL[4 * stride + idx], P_FLOOR);
    float E_R = max(UR[4 * stride + idx], P_FLOOR);

    float denom_SM = rho_R * (SR - vn_R) - rho_L * (SL - vn_L);
    if (metal::abs(denom_SM) < TINY)
        denom_SM = TINY * (denom_SM >= 0.0f ? 1.0f : -1.0f);
    float SM = (rho_R * vn_R * (SR - vn_R) - rho_L * vn_L * (SL - vn_L) + pt_L - pt_R) / denom_SM;

    float pt_star = max(pt_L + rho_L * (SL - vn_L) * (SM - vn_L), P_FLOOR);

    float denom_L = SL - SM;
    if (metal::abs(denom_L) < TINY) denom_L = TINY * (denom_L >= 0.0f ? 1.0f : -1.0f);
    float denom_R = SR - SM;
    if (metal::abs(denom_R) < TINY) denom_R = TINY * (denom_R >= 0.0f ? 1.0f : -1.0f);

    float rho_sL = max(rho_L * (SL - vn_L) / denom_L, RHO_FLOOR);
    float rho_sR = max(rho_R * (SR - vn_R) / denom_R, RHO_FLOOR);

    float D_L = rho_L * (SL - vn_L) * (SL - SM) - Bn * Bn;
    float D_R = rho_R * (SR - vn_R) * (SR - SM) - Bn * Bn;
    bool bn_small = metal::abs(Bn) < TINY_BN;
    float inv_DL = (metal::abs(D_L) < TINY) ? 0.0f : (1.0f / D_L);
    float inv_DR = (metal::abs(D_R) < TINY) ? 0.0f : (1.0f / D_R);

    float vt1_sL = bn_small ? vt1_L : vt1_L - Bn * Bt1_L * (SM - vn_L) * inv_DL;
    float vt2_sL = bn_small ? vt2_L : vt2_L - Bn * Bt2_L * (SM - vn_L) * inv_DL;
    float vt1_sR = bn_small ? vt1_R : vt1_R - Bn * Bt1_R * (SM - vn_R) * inv_DR;
    float vt2_sR = bn_small ? vt2_R : vt2_R - Bn * Bt2_R * (SM - vn_R) * inv_DR;

    float fL = rho_L * (SL - vn_L) * (SL - vn_L) - Bn * Bn;
    float fR = rho_R * (SR - vn_R) * (SR - vn_R) - Bn * Bn;
    float Bt1_sL = bn_small ? Bt1_L : Bt1_L * fL * inv_DL;
    float Bt2_sL = bn_small ? Bt2_L : Bt2_L * fL * inv_DL;
    float Bt1_sR = bn_small ? Bt1_R : Bt1_R * fR * inv_DR;
    float Bt2_sR = bn_small ? Bt2_R : Bt2_R * fR * inv_DR;

    float vB_L  = vn_L * Bn_L + vt1_L * Bt1_L + vt2_L * Bt2_L;
    float vB_sL = SM * Bn + vt1_sL * Bt1_sL + vt2_sL * Bt2_sL;
    float e_sL  = ((SL - vn_L) * E_L - pt_L * vn_L + pt_star * SM + Bn * (vB_L - vB_sL)) / denom_L;

    float vB_R  = vn_R * Bn_R + vt1_R * Bt1_R + vt2_R * Bt2_R;
    float vB_sR = SM * Bn + vt1_sR * Bt1_sR + vt2_sR * Bt2_sR;
    float e_sR  = ((SR - vn_R) * E_R - pt_R * vn_R + pt_star * SM + Bn * (vB_R - vB_sR)) / denom_R;

    float sqrt_rho_sL = sqrt(max(rho_sL, RHO_FLOOR));
    float sqrt_rho_sR = sqrt(max(rho_sR, RHO_FLOOR));
    float SL_star = SM - metal::abs(Bn) / sqrt_rho_sL;
    float SR_star = SM + metal::abs(Bn) / sqrt_rho_sR;

    float sign_Bn = (Bn >= 0.0f) ? 1.0f : -1.0f;
    if (metal::abs(Bn) < TINY_BN) sign_Bn = 0.0f;

    float denom_ds = max(sqrt_rho_sL + sqrt_rho_sR, TINY);
    float vt1_ds = (sqrt_rho_sL * vt1_sL + sqrt_rho_sR * vt1_sR + (Bt1_sR - Bt1_sL) * sign_Bn) / denom_ds;
    float vt2_ds = (sqrt_rho_sL * vt2_sL + sqrt_rho_sR * vt2_sR + (Bt2_sR - Bt2_sL) * sign_Bn) / denom_ds;
    float Bt1_ds = (sqrt_rho_sL * Bt1_sR + sqrt_rho_sR * Bt1_sL + sqrt_rho_sL * sqrt_rho_sR * (vt1_sR - vt1_sL) * sign_Bn) / denom_ds;
    float Bt2_ds = (sqrt_rho_sL * Bt2_sR + sqrt_rho_sR * Bt2_sL + sqrt_rho_sL * sqrt_rho_sR * (vt2_sR - vt2_sL) * sign_Bn) / denom_ds;

    float vt1_dsL = bn_small ? vt1_sL : vt1_ds;
    float vt2_dsL = bn_small ? vt2_sL : vt2_ds;
    float vt1_dsR = bn_small ? vt1_sR : vt1_ds;
    float vt2_dsR = bn_small ? vt2_sR : vt2_ds;
    float Bt1_dsL = bn_small ? Bt1_sL : Bt1_ds;
    float Bt2_dsL = bn_small ? Bt2_sL : Bt2_ds;
    float Bt1_dsR = bn_small ? Bt1_sR : Bt1_ds;
    float Bt2_dsR = bn_small ? Bt2_sR : Bt2_ds;

    float vB_dsL = SM * Bn + vt1_dsL * Bt1_dsL + vt2_dsL * Bt2_dsL;
    float e_dsL  = e_sL - sqrt_rho_sL * (vB_sL - vB_dsL) * sign_Bn;
    float vB_dsR = SM * Bn + vt1_dsR * Bt1_dsR + vt2_dsR * Bt2_dsR;
    float e_dsR  = e_sR + sqrt_rho_sR * (vB_sR - vB_dsR) * sign_Bn;

    float FL[10], FR[10];
    physical_flux(rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L, E_L, gamma,
                  FL, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2);
    physical_flux(rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R, E_R, gamma,
                  FR, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2);

    float Srho_L = UL[5 * stride + idx];
    float Srho_R = UR[5 * stride + idx];
    FL[5] = Srho_L * vn_L;
    FR[5] = Srho_R * vn_R;

    float ee_L = UL[9 * stride + idx];
    float ee_R = UR[9 * stride + idx];

    // Star states
    float UsL[10], UsR[10], UdsL[10], UdsR[10];
    UsL[0]=rho_sL; UsR[0]=rho_sR;
    UsL[im_n]=rho_sL*SM;       UsR[im_n]=rho_sR*SM;
    UsL[im_t1]=rho_sL*vt1_sL; UsR[im_t1]=rho_sR*vt1_sR;
    UsL[im_t2]=rho_sL*vt2_sL; UsR[im_t2]=rho_sR*vt2_sR;
    UsL[4]=e_sL;               UsR[4]=e_sR;
    UsL[5]=Srho_L*(SL-vn_L)/denom_L; UsR[5]=Srho_R*(SR-vn_R)/denom_R;
    UsL[ib_n]=Bn;              UsR[ib_n]=Bn;
    UsL[ib_t1]=Bt1_sL;        UsR[ib_t1]=Bt1_sR;
    UsL[ib_t2]=Bt2_sL;        UsR[ib_t2]=Bt2_sR;
    UsL[9]=max(ee_L*(SL-vn_L)/denom_L,0.0f);
    UsR[9]=max(ee_R*(SR-vn_R)/denom_R,0.0f);

    UdsL[0]=rho_sL; UdsR[0]=rho_sR;
    UdsL[im_n]=rho_sL*SM;        UdsR[im_n]=rho_sR*SM;
    UdsL[im_t1]=rho_sL*vt1_dsL;  UdsR[im_t1]=rho_sR*vt1_dsR;
    UdsL[im_t2]=rho_sL*vt2_dsL;  UdsR[im_t2]=rho_sR*vt2_dsR;
    UdsL[4]=e_dsL;                UdsR[4]=e_dsR;
    UdsL[5]=UsL[5];               UdsR[5]=UsR[5];
    UdsL[ib_n]=Bn;                UdsR[ib_n]=Bn;
    UdsL[ib_t1]=Bt1_dsL;          UdsR[ib_t1]=Bt1_dsR;
    UdsL[ib_t2]=Bt2_dsL;          UdsR[ib_t2]=Bt2_dsR;
    UdsL[9]=UsL[9];               UdsR[9]=UsR[9];

    float FsL[10], FsR[10], FdsL[10], FdsR[10];
    for (int v = 0; v < NVAR; v++) {
        float uLv = UL[v * stride + idx];
        float uRv = UR[v * stride + idx];
        FsL[v]  = FL[v]  + SL      * (UsL[v]  - uLv);
        FsR[v]  = FR[v]  + SR      * (UsR[v]  - uRv);
        FdsL[v] = FsL[v] + SL_star * (UdsL[v] - UsL[v]);
        FdsR[v] = FsR[v] + SR_star * (UdsR[v] - UsR[v]);
    }

    float F_out[10];
    if      (SL     > 0.0f) { for (int v=0;v<NVAR;v++) F_out[v]=FL[v];   }
    else if (SL_star> 0.0f) { for (int v=0;v<NVAR;v++) F_out[v]=FsL[v];  }
    else if (SM     > 0.0f) { for (int v=0;v<NVAR;v++) F_out[v]=FdsL[v]; }
    else if (SR_star> 0.0f) { for (int v=0;v<NVAR;v++) F_out[v]=FdsR[v]; }
    else if (SR     > 0.0f) { for (int v=0;v<NVAR;v++) F_out[v]=FsR[v];  }
    else                    { for (int v=0;v<NVAR;v++) F_out[v]=FR[v];   }

    bool has_nan = false;
    for (int v = 0; v < NVAR; v++) {
        if (isnan(F_out[v]) || isinf(F_out[v])) { has_nan = true; break; }
    }
    if (has_nan) {
        float S_max = max(metal::abs(SL), metal::abs(SR));
        for (int v = 0; v < NVAR; v++) {
            float uLv = UL[v * stride + idx];
            float uRv = UR[v * stride + idx];
            F_out[v] = 0.5f * (FL[v] + FR[v]) - 0.5f * S_max * (uRv - uLv);
        }
    }

    for (int v = 0; v < NVAR; v++) flux[v * stride + idx] = F_out[v];
"""

_hlld_kernel_cache: object = None


def _get_hlld_kernel() -> object:
    global _hlld_kernel_cache
    if _hlld_kernel_cache is None:
        _hlld_kernel_cache = mx.fast.metal_kernel(
            name="dpf_hlld",
            input_names=["UL", "UR", "gamma_param", "dim_param"],
            output_names=["flux"],
            source=_HLLD_SOURCE,
            header=_HLLD_HEADER,
            ensure_row_contiguous=True,
        )
    return _hlld_kernel_cache


def _cons_to_prim_np(
    U: np.ndarray, gamma: float, dim: int
) -> tuple:
    im_n = 1 if dim == 0 else 2
    im_t1 = 2 if dim == 0 else 3
    im_t2 = 3 if dim == 0 else 1
    ib_n = 6 if dim == 0 else 7
    ib_t1 = 7 if dim == 0 else 8
    ib_t2 = 8 if dim == 0 else 6

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
    return rho, vn, vt1, vt2, p, Bn, Bt1, Bt2, im_n, im_t1, im_t2, ib_n, ib_t1, ib_t2


def _fast_ms_np(rho, p, Bn, Bt1, Bt2, gamma):
    rho = np.maximum(rho, RHO_FLOOR)
    p = np.maximum(p, P_FLOOR)
    inv_rho = 1.0 / rho
    a2 = gamma * p * inv_rho
    B2 = Bn**2 + Bt1**2 + Bt2**2
    va2 = B2 * inv_rho
    Bt2_sum = Bt1**2 + Bt2**2
    diff = a2 - va2
    disc = diff**2 + 4.0 * a2 * Bt2_sum * inv_rho
    cf2 = 0.5 * (a2 + va2 + np.sqrt(np.maximum(disc, 0.0)))
    return np.sqrt(np.maximum(cf2, 0.0))


def hlld_flux_numpy(
    QL: np.ndarray,
    QR: np.ndarray,
    gamma: float = GAMMA,
    dim: int = 0,
) -> np.ndarray:
    """NumPy reference: HLLD flux from left/right reconstructed states.

    Args:
        QL: Left state, shape (NVAR, n_ifaces, nz), float32.
        QR: Right state, shape (NVAR, n_ifaces, nz), float32.
        gamma: Adiabatic index.
        dim: Normal direction (0=radial, 1=axial).

    Returns:
        Numerical flux, shape (NVAR, n_ifaces, nz), float32.
    """
    UL = QL.astype(np.float64)
    UR = QR.astype(np.float64)
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
    denom_SM = np.where(np.abs(denom_SM) < TINY, TINY * np.sign(denom_SM + eps), denom_SM)
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

    fL = rho_L * (SL - vn_L) ** 2 - Bn**2
    fR = rho_R * (SR - vn_R) ** 2 - Bn**2
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

    def _build_state(rho_s, vt1_s, vt2_s, e_s, Bt1_s, Bt2_s, Srho_src, ee_src, S_wave, vn_src):
        Us = np.zeros_like(UL)
        Us[0] = rho_s
        Us[im_n] = rho_s * SM
        Us[im_t1] = rho_s * vt1_s
        Us[im_t2] = rho_s * vt2_s
        Us[4] = e_s
        denom_s = np.where(np.abs(S_wave - SM) < TINY, TINY, S_wave - SM)
        Us[5] = Srho_src * (S_wave - vn_src) / denom_s
        Us[ib_n] = Bn
        Us[ib_t1] = Bt1_s
        Us[ib_t2] = Bt2_s
        Us[9] = np.maximum(ee_src * (S_wave - vn_src) / denom_s, 0.0)
        return Us

    U_sL = _build_state(rho_sL, vt1_sL, vt2_sL, e_sL, Bt1_sL, Bt2_sL, UL[5], UL[9], SL, vn_L)
    U_sR = _build_state(rho_sR, vt1_sR, vt2_sR, e_sR, Bt1_sR, Bt2_sR, UR[5], UR[9], SR, vn_R)
    U_dsL = _build_state(rho_sL, vt1_dsL, vt2_dsL, e_dsL, Bt1_dsL, Bt2_dsL, UL[5], UL[9], SL, vn_L)
    U_dsR = _build_state(rho_sR, vt1_dsR, vt2_dsR, e_dsR, Bt1_dsR, Bt2_dsR, UR[5], UR[9], SR, vn_R)

    def _phys_flux(rho, vn, vt1, vt2, p, Bn_v, Bt1_v, Bt2_v, E):
        F = np.zeros_like(UL)
        B2 = Bn_v**2 + Bt1_v**2 + Bt2_v**2
        pt = p + 0.5 * B2
        F[0] = rho * vn
        F[im_n] = rho * vn * vn + pt - Bn_v * Bn_v
        F[im_t1] = rho * vn * vt1 - Bn_v * Bt1_v
        F[im_t2] = rho * vn * vt2 - Bn_v * Bt2_v
        F[4] = (E + pt) * vn - Bn_v * (vn * Bn_v + vt1 * Bt1_v + vt2 * Bt2_v)
        F[ib_n] = 0.0
        F[ib_t1] = vn * Bt1_v - vt1 * Bn_v
        F[ib_t2] = vn * Bt2_v - vt2 * Bn_v
        return F

    FL = _phys_flux(rho_L, vn_L, vt1_L, vt2_L, p_L, Bn_L, Bt1_L, Bt2_L, E_L)
    FR = _phys_flux(rho_R, vn_R, vt1_R, vt2_R, p_R, Bn_R, Bt1_R, Bt2_R, E_R)
    FL[5] = UL[5] * vn_L
    FR[5] = UR[5] * vn_R

    F_sL = FL + SL[np.newaxis] * (U_sL - UL)
    F_sR = FR + SR[np.newaxis] * (U_sR - UR)
    F_dsL = F_sL + SL_star[np.newaxis] * (U_dsL - U_sL)
    F_dsR = F_sR + SR_star[np.newaxis] * (U_dsR - U_sR)

    F_out = FR.copy()
    F_out = np.where(SR[np.newaxis] > 0, F_sR, F_out)
    mask = (SR_star[np.newaxis] <= 0) & (SR[np.newaxis] > 0)
    F_out = np.where(mask, F_sR, F_out)
    mask = (SM[np.newaxis] <= 0) & (SR_star[np.newaxis] > 0)
    F_out = np.where(mask, F_dsR, F_out)
    mask = (SL_star[np.newaxis] <= 0) & (SM[np.newaxis] > 0)
    F_out = np.where(mask, F_dsL, F_out)
    mask = (SL[np.newaxis] <= 0) & (SL_star[np.newaxis] > 0)
    F_out = np.where(mask, F_sL, F_out)
    F_out = np.where(SL[np.newaxis] > 0, FL, F_out)

    has_nan = np.isnan(F_out) | np.isinf(F_out)
    if np.any(has_nan):
        S_max = np.maximum(np.abs(SL), np.abs(SR))
        F_LF = 0.5 * (FL + FR) - 0.5 * S_max[np.newaxis] * (UR - UL)
        F_out = np.where(has_nan, F_LF, F_out)

    return F_out.astype(np.float32)


def hlld_flux_mlx(
    QL: mx.array,
    QR: mx.array,
    gamma: float = GAMMA,
    dim: int = 0,
) -> mx.array:
    """MLX Metal kernel: HLLD Riemann solver on GPU.

    Falls back to NumPy reference if MLX kernels unavailable.

    Args:
        QL: Left state, shape (NVAR, n_ifaces, nz), float32 mx.array.
        QR: Right state, shape (NVAR, n_ifaces, nz), float32 mx.array.
        gamma: Adiabatic index.
        dim: Normal direction (0=radial, 1=axial).

    Returns:
        Numerical flux mx.array, shape (NVAR, n_ifaces, nz), float32.
    """
    if not HAS_MLX_KERNELS:
        result = hlld_flux_numpy(np.asarray(QL), np.asarray(QR), gamma, dim)
        return mx.array(result)

    nvar, n_ifaces, nz = QL.shape
    gamma_param = mx.array([gamma], dtype=mx.float32)
    dim_param = mx.array([float(dim)], dtype=mx.float32)

    tg_r = min(32, n_ifaces)
    tg_z = min(8, nz)
    grid_r = ((n_ifaces + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    kernel = _get_hlld_kernel()
    outputs = kernel(
        inputs=[QL, QR, gamma_param, dim_param],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, n_ifaces, nz)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ══════════════════════════════════════════════════════════════
# 3. Cylindrical Geometric Source Terms
# ══════════════════════════════════════════════════════════════

# Primitive variable indices for cylindrical source (input is primitive state)
_IVR = 1  # vr
_IVT = 3  # vtheta
_IPR = 4  # pressure

_CYL_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

constant float TINY_R = 1.0e-30f;
constant int NVAR = 10;
constant int IDN = 0;
constant int IVR = 1;
constant int IVZ = 2;
constant int IVT = 3;
constant int IPR = 4;
constant int ISR = 5;
constant int IBR = 6;
constant int IBZ = 7;
constant int IBT = 8;
constant int IEE = 9;
"""

_CYL_SOURCE = r"""
    uint ir = thread_position_in_grid.x;
    uint iz = thread_position_in_grid.y;

    uint nr = prim_shape[1];
    uint nz = prim_shape[2];
    if (ir >= nr || iz >= nz) return;

    uint stride = nr * nz;
    uint idx    = ir * nz + iz;

    float rho    = prim[IDN * stride + idx];
    float vr     = prim[IVR * stride + idx];
    float vtheta = prim[IVT * stride + idx];
    float p      = prim[IPR * stride + idx];
    float Br     = prim[IBR * stride + idx];
    float Bz     = prim[IBZ * stride + idx];
    float Btheta = prim[IBT * stride + idx];

    float r  = r_cell[ir];
    float dr = grid_params[0];

    float B2    = Br*Br + Bz*Bz + Btheta*Btheta;
    float p_tot = p + 0.5f * B2;
    float inv_r = 1.0f / max(r, TINY_R);

    float S_mr = (p_tot - Btheta * Btheta) * inv_r + rho * vtheta * vtheta * inv_r;
    float S_mt = -(rho * vr * vtheta - Br * Btheta) * inv_r;
    float S_Bt = -(vr * Btheta - Br * vtheta) * inv_r;

    if (ir == 0 && nr > 1) {
        uint idx1 = 1 * nz + iz;
        float p_next  = prim[IPR * stride + idx1];
        float Br_next = prim[IBR * stride + idx1];
        float Bz_next = prim[IBZ * stride + idx1];
        float Bt_next = prim[IBT * stride + idx1];
        float B2_next = Br_next*Br_next + Bz_next*Bz_next + Bt_next*Bt_next;
        float pt_next = p_next + 0.5f * B2_next;
        S_mr = (pt_next - p_tot) / dr;
        S_mt = 0.0f;
        S_Bt = 0.0f;
    }

    src[IDN * stride + idx] = 0.0f;
    src[IVR * stride + idx] = S_mr;
    src[IVZ * stride + idx] = 0.0f;
    src[IVT * stride + idx] = S_mt;
    src[IPR * stride + idx] = 0.0f;
    src[ISR * stride + idx] = 0.0f;
    src[IBR * stride + idx] = 0.0f;
    src[IBZ * stride + idx] = 0.0f;
    src[IBT * stride + idx] = S_Bt;
    src[IEE * stride + idx] = 0.0f;
"""

_cyl_kernel_cache: object = None


def _get_cyl_kernel() -> object:
    global _cyl_kernel_cache
    if _cyl_kernel_cache is None:
        _cyl_kernel_cache = mx.fast.metal_kernel(
            name="dpf_cyl_source",
            input_names=["prim", "r_cell", "grid_params"],
            output_names=["src"],
            source=_CYL_SOURCE,
            header=_CYL_HEADER,
            ensure_row_contiguous=True,
        )
    return _cyl_kernel_cache


def cylindrical_source_numpy(
    Q: np.ndarray,
    r_cell: np.ndarray,
    inv_r: np.ndarray | None = None,
    gamma: float = GAMMA,
) -> np.ndarray:
    """NumPy reference: geometric source terms for cylindrical MHD.

    Expects a primitive state array with layout:
      [rho, vr, vz, vtheta, p, S, Br, Bz, Btheta, e_electron]

    Args:
        Q: Primitive state, shape (NVAR, nr, nz), float32.
        r_cell: Cell-centre radii, shape (nr,), float32.
        inv_r: Pre-computed 1/r, shape (nr,) (optional; computed if None).
        gamma: Adiabatic index (unused; kept for API symmetry).

    Returns:
        Source terms, shape (NVAR, nr, nz), float32.
    """
    nvar, nr, nz = Q.shape
    src = np.zeros_like(Q)

    rho = Q[IDN]
    vr = Q[_IVR]
    vtheta = Q[_IVT]
    p = Q[_IPR]
    Br = Q[IBR]
    Bz = Q[IBZ]
    Btheta = Q[IBT]

    B2 = Br**2 + Bz**2 + Btheta**2
    p_tot = p + 0.5 * B2

    r = r_cell[:, np.newaxis]
    if inv_r is not None:
        _inv_r = inv_r[:, np.newaxis]
    else:
        _inv_r = 1.0 / np.maximum(r, 1e-30)

    S_mr = (p_tot - Btheta**2) * _inv_r + rho * vtheta**2 * _inv_r
    S_mt = -(rho * vr * vtheta - Br * Btheta) * _inv_r
    S_Bt = -(vr * Btheta - Br * vtheta) * _inv_r

    # L'Hopital at ir=0: replace p_tot/r with dp_tot/dr
    if nr > 1:
        dr = float(r_cell[1] - r_cell[0])
        dpt_dr = (p_tot[1, :] - p_tot[0, :]) / dr
        S_mr[0, :] = dpt_dr
        S_mt[0, :] = 0.0
        S_Bt[0, :] = 0.0

    src[_IVR] = S_mr
    src[_IVT] = S_mt
    src[IBT] = S_Bt

    return src.astype(np.float32)


def cylindrical_source_mlx(
    Q: mx.array,
    r_cell: mx.array,
    inv_r: mx.array | None = None,
    gamma: float = GAMMA,
) -> mx.array:
    """MLX Metal kernel: cylindrical source on GPU.

    Falls back to NumPy reference if MLX kernels unavailable.

    Args:
        Q: Primitive state, shape (NVAR, nr, nz), float32 mx.array.
        r_cell: Cell-centre radii, shape (nr,), float32 mx.array.
        inv_r: Pre-computed 1/r (unused in Metal path; for API compat).
        gamma: Adiabatic index (unused in Metal path).

    Returns:
        Source terms mx.array, shape (NVAR, nr, nz), float32.
    """
    if not HAS_MLX_KERNELS:
        r_np = np.asarray(r_cell)
        inv_r_np = np.asarray(inv_r) if inv_r is not None else None
        result = cylindrical_source_numpy(np.asarray(Q), r_np, inv_r_np, gamma)
        return mx.array(result)

    nvar, nr, nz = Q.shape
    dr = float(r_cell[1] - r_cell[0]) if nr > 1 else 1e-3
    grid_params = mx.array([dr], dtype=mx.float32)

    tg_r = min(32, nr)
    tg_z = min(8, nz)
    grid_r = ((nr + tg_r - 1) // tg_r) * tg_r
    grid_z = ((nz + tg_z - 1) // tg_z) * tg_z

    kernel = _get_cyl_kernel()
    outputs = kernel(
        inputs=[Q, r_cell, grid_params],
        template=[],
        grid=(grid_r, grid_z, 1),
        threadgroup=(tg_r, tg_z, 1),
        output_shapes=[(nvar, nr, nz)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
