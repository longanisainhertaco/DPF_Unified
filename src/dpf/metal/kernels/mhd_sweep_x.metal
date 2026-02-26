//
//  mhd_sweep_x.metal
//  DPF Unified - Metal Backend
//
//  Monolithic compute kernel for X-direction MHD sweep.
//  Fuses Load -> Reconstruct -> Flux -> Update (Partial)
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

// Forward declarations
void plm_reconstruct_x_cell(int i, int j, int k, 
                            device const float* U_in, 
                            int nx, int ny, int nz, 
                            int limiter_type, 
                            thread ConsState& UL, 
                            thread ConsState& UR);

ConsState hll_flux_interface(ConsState UL, ConsState UR, float gamma, int dim);

// ============================================================================
// MHD Sweep X
// ============================================================================

kernel void mhd_sweep_x(
    device const float*  U_in  [[ buffer(0) ]],
    device       float*  F_out [[ buffer(1) ]],
    constant     int3&   dims  [[ buffer(2) ]],
    constant     float&  gamma [[ buffer(3) ]],
    constant     int&    limiter_type [[ buffer(4) ]],
    uint3                gid   [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    // We compute fluxes at interfaces i+1/2 for i=0..nx-2
    if (gid.x >= uint(nx - 1) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    
    // Reconstruct Left State (UL at i+1/2) and Right State (UR at i+1/2)
    ConsState UL_interface, UR_interface;
    
    // Helper to read U
    auto get_U = [&](int _i, int _j, int _k) {
        ConsState U;
        int base_idx = ((_k * ny + _j) * nx + _i) * NVAR;
        for(int v=0; v<NVAR; ++v) U.u[v] = U_in[base_idx + v];
        return U;
    };
    
    // Manual PLM inline logic for performance
    // Need neighborhood: i-1, i, i+1, i+2
    
    // Slope limiter
    auto limiter = [&](float a, float b) {
        if (limiter_type == 1) { // MC
             float c1 = 2.0f * a;
             float c2 = 0.5f * (a + b);
             float c3 = 2.0f * b;
             float max_v = max(max(c1, c2), c3);
             float min_v = min(min(c1, c2), c3);
             float med = c1 + c2 + c3 - max_v - min_v;
             return (a * b > 0.0f) ? med : 0.0f;
        } else { // Minmod
             return (a * b > 0.0f) ? sign(a) * min(abs(a), abs(b)) : 0.0f;
        }
    };
    
    // 1. Reconstruct UL at i+1/2 (Right face of cell i)
    // Needs U[i] and Slope[i]. Slope[i] needs U[i-1], U[i], U[i+1]
    
    ConsState U_i = get_U(i, j, k);
    ConsState Slope_i;
    
    if (i > 0) {
        ConsState U_m1 = get_U(i-1, j, k);
        ConsState U_p1 = get_U(i+1, j, k);
        for(int v=0; v<NVAR; ++v) {
            float dL = U_i.u[v] - U_m1.u[v];
            float dR = U_p1.u[v] - U_i.u[v];
            Slope_i.u[v] = limiter(dL, dR);
        }
    } else {
        for(int v=0; v<NVAR; ++v) Slope_i.u[v] = 0.0f;
    }
    
    for(int v=0; v<NVAR; ++v) {
        UL_interface.u[v] = U_i.u[v] + 0.5f * Slope_i.u[v];
    }
    
    // 2. Reconstruct UR at i+1/2 (Left face of cell i+1)
    // Needs U[i+1] and Slope[i+1]. Slope[i+1] needs U[i], U[i+1], U[i+2]
    
    ConsState U_p1 = get_U(i+1, j, k);
    ConsState Slope_ip1;
    
    if (i+1 < nx - 1) {
        ConsState U_p2 = get_U(i+2, j, k);
        for(int v=0; v<NVAR; ++v) {
            float dL = U_p1.u[v] - U_i.u[v]; // U[i+1]-U[i]
            float dR = U_p2.u[v] - U_p1.u[v];
            Slope_ip1.u[v] = limiter(dL, dR);
        }
    } else {
        for(int v=0; v<NVAR; ++v) Slope_ip1.u[v] = 0.0f;
    }
    
    for(int v=0; v<NVAR; ++v) {
        UR_interface.u[v] = U_p1.u[v] - 0.5f * Slope_ip1.u[v];
    }
    
    // Enforce floors
    UL_interface.u[IDN] = max(UL_interface.u[IDN], RHO_FLOOR);
    UR_interface.u[IDN] = max(UR_interface.u[IDN], RHO_FLOOR);
    
    // 3. Compute HLL Flux
    // Inline HLL logic to avoid function call overhead if compiler doesn't inline
    // Re-use logic from hll_flux.metal but adapting to ConsState struct
    
    PrimState W_L = cons_to_prim(UL_interface, gamma);
    PrimState W_R = cons_to_prim(UR_interface, gamma);
    
    // Fast Magnetosonic Speed (X-dim specific dim=0)
    auto fast_mag = [&](PrimState W) {
        float rho = W.rho;
        float p   = W.p;
        float B2  = W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2];
        float Bn2 = W.B[0]*W.B[0]; // dim=0
        
        float a2  = gamma * p / rho;
        float va2 = B2 / rho;
        float van2= Bn2 / rho;
        
        float term1 = a2 + va2;
        float term2 = sqrt(max(term1*term1 - 4.0f*a2*van2, 0.0f));
        return sqrt(0.5f * (term1 + term2));
    };
    
    float cf_L = fast_mag(W_L);
    float cf_R = fast_mag(W_R);
    
    float vn_L = W_L.v[0];
    float vn_R = W_R.v[0];
    
    float SL = min(vn_L - cf_L, vn_R - cf_R);
    float SR = max(vn_L + cf_L, vn_R + cf_R);
    SR = max(SR, SL + 1e-10f);
    
    // Physical Flux X-dim
    auto phys_flux = [&](ConsState U, PrimState W) {
        ConsState F;
        float vn = W.v[0];
        float Bn = W.B[0];
        float v_dot_B = W.v[0]*W.B[0] + W.v[1]*W.B[1] + W.v[2]*W.B[2];
        float B2 = W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2];
        float p_tot = W.p + 0.5f * B2;
        float E = U.u[IEN];
        
        F.u[IDN] = U.u[IDN] * vn;
        F.u[IM1] = U.u[IM1] * vn - W.B[0] * Bn + p_tot;
        F.u[IM2] = U.u[IM2] * vn - W.B[1] * Bn;
        F.u[IM3] = U.u[IM3] * vn - W.B[2] * Bn;
        F.u[IEN] = (E + p_tot) * vn - Bn * v_dot_B;
        F.u[IB1] = 0.0f; // Div constraint
        F.u[IB2] = W.B[1] * vn - W.v[1] * Bn;
        F.u[IB3] = W.B[2] * vn - W.v[2] * Bn;
        return F;
    };
    
    ConsState FL = phys_flux(UL_interface, W_L);
    ConsState FR = phys_flux(UR_interface, W_R);
    ConsState F_hll;
    
    if (SL >= 0.0f) {
        F_hll = FL;
    } else if (SR <= 0.0f) {
        F_hll = FR;
    } else {
        float inv_denom = 1.0f / (SR - SL);
        for(int v=0; v<NVAR; ++v) {
            F_hll.u[v] = (SR * FL.u[v] - SL * FR.u[v] + SL * SR * (UR_interface.u[v] - UL_interface.u[v])) * inv_denom;
        }
    }
    
    // Write Flux to Output
    int out_base_idx = ((k * ny + j) * (nx - 1) + i) * NVAR;
    for(int v=0; v<NVAR; ++v) {
        F_out[out_base_idx + v] = F_hll.u[v];
    }
}
