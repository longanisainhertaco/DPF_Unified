//
//  hll_flux.metal
//  DPF Unified - Metal Backend
//
//  Harten-Lax-van Leer (HLL) Riemann Solver.
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

// ============================================================================
// HLL Flux Kernel (Generic Dimension)
// ============================================================================

// Compute flux from UL, UR interface states.
// Thread dispatch: (n_interface, ny, nz)
// dim_idx: 0=x, 1=y, 2=z
kernel void hll_flux(
    device const float*  UL_in [[ buffer(0) ]],
    device const float*  UR_in [[ buffer(1) ]],
    device       float*  F_out [[ buffer(2) ]],
    constant     int3&   dims  [[ buffer(3) ]], // (n_interface, ny, nz)
    constant     float&  gamma [[ buffer(4) ]],
    constant     int&    dim   [[ buffer(5) ]], // 0, 1, 2
    uint3                gid   [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    if (gid.x >= uint(nx) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    // Flat index
    auto idx = [&](int _i, int _j, int _k, int _v) {
        return ((_k * ny + _j) * nx + _i) * NVAR + _v;
    };
    
    // Load Left and Right states
    ConsState U_L, U_R;
    for (int v=0; v<NVAR; ++v) {
        U_L.u[v] = UL_in[idx(gid.x, gid.y, gid.z, v)];
        U_R.u[v] = UR_in[idx(gid.x, gid.y, gid.z, v)];
    }
    
    // To Primitive
    PrimState W_L = cons_to_prim(U_L, gamma);
    PrimState W_R = cons_to_prim(U_R, gamma);
    
    // Compute Wave Speeds (Davis bounds)
    // We need fast magnetosonic speed c_f
    // c_f^2 = 0.5 * (a^2 + v_a^2 + sqrt( (a^2 + v_a^2)^2 - 4 a^2 v_an^2 ))
    
    auto fast_magnetosonic = [&](PrimState W) {
        float rho = W.rho;
        float p   = W.p;
        float B2  = W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2];
        float Bn2 = W.B[dim]*W.B[dim];
        
        float a2  = gamma * p / rho;        // sound speed^2
        float va2 = B2 / rho;               // Alfven speed^2
        float van2= Bn2 / rho;              // Normal Alfven speed^2
        
        float term1 = a2 + va2;
        float term2 = sqrt(max(term1*term1 - 4.0f*a2*van2, 0.0f));
        
        return sqrt(0.5f * (term1 + term2));
    };
    
    float cf_L = fast_magnetosonic(W_L);
    float cf_R = fast_magnetosonic(W_R);
    
    float vn_L = W_L.v[dim];
    float vn_R = W_R.v[dim];
    
    float SL = min(vn_L - cf_L, vn_R - cf_R);
    float SR = max(vn_L + cf_L, vn_R + cf_R);
    
    // Enforce SR > SL
    SR = max(SR, SL + 1e-10f);
    
    // Compute Physical Fluxes F(U)
    auto physical_flux = [&](ConsState U, PrimState W) {
        ConsState F;
        
        float vn = W.v[dim];
        float Bn = W.B[dim];
        float v_dot_B = W.v[0]*W.B[0] + W.v[1]*W.B[1] + W.v[2]*W.B[2];
        float B2 = W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2];
        float p_tot = W.p + 0.5f * B2;
        float E = U.u[IEN];
        
        // Mass Flux
        F.u[IDN] = U.u[IDN] * vn;
        
        // Momentum Flux
        // F_mi = rho*vi*vn - Bi*Bn + delta*p_tot
        F.u[IM1] = U.u[IM1] * vn - W.B[0] * Bn;
        F.u[IM2] = U.u[IM2] * vn - W.B[1] * Bn;
        F.u[IM3] = U.u[IM3] * vn - W.B[2] * Bn;
        
        if (dim == 0) F.u[IM1] += p_tot;
        if (dim == 1) F.u[IM2] += p_tot;
        if (dim == 2) F.u[IM3] += p_tot;
        
        // Energy Flux
        F.u[IEN] = (E + p_tot) * vn - Bn * v_dot_B;
        
        // Induction Flux
        // F_Bi = Bi*vn - vi*Bn
        // Note: For dim component, this is 0 (div B = 0 constraint)
        F.u[IB1] = W.B[0] * vn - W.v[0] * Bn;
        F.u[IB2] = W.B[1] * vn - W.v[1] * Bn;
        F.u[IB3] = W.B[2] * vn - W.v[2] * Bn;
        
        return F;
    };
    
    ConsState FL = physical_flux(U_L, W_L);
    ConsState FR = physical_flux(U_R, W_R);
    
    // Compute HLL Flux
    // F_hll = (SR*FL - SL*FR + SL*SR*(UR - UL)) / (SR - SL)
    
    ConsState F_hll;
    
    if (SL >= 0.0f) {
        F_hll = FL;
    } else if (SR <= 0.0f) {
        F_hll = FR;
    } else {
        float inv_denom = 1.0f / (SR - SL);
        for (int v=0; v<NVAR; ++v) {
            F_hll.u[v] = (SR * FL.u[v] - SL * FR.u[v] + SL * SR * (U_R.u[v] - U_L.u[v])) * inv_denom;
        }
    }
    
    // Write out
    for (int v=0; v<NVAR; ++v) {
        F_out[idx(gid.x, gid.y, gid.z, v)] = F_hll.u[v];
    }
}
