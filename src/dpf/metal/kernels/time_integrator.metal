//
//  time_integrator.metal
//  DPF Unified - Metal Backend
//
//  Kernels for explicit time integration (SSP-RK2).
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

// General Linear Combination Kernel for RK Stages
// Computes: U_out = c1 * U_1 + c2 * U_2 + c3 * dt * RHS
//
// Stage 1: U_star = 1.0 * U_n + 0.0 * U_star + 1.0 * dt * RHS_1
// Stage 2: U_new  = 0.5 * U_n + 0.5 * U_star + 0.5 * dt * RHS_2

kernel void rk_update(
    device       float*  U_out [[ buffer(0) ]],
    device const float*  U_1   [[ buffer(1) ]],
    device const float*  U_2   [[ buffer(2) ]],
    device const float*  RHS   [[ buffer(3) ]],
    constant     int3&   dims  [[ buffer(4) ]],
    constant     float&  c1    [[ buffer(5) ]],
    constant     float&  c2    [[ buffer(6) ]],
    constant     float&  c3    [[ buffer(7) ]],
    constant     float&  dt    [[ buffer(8) ]],
    uint3                gid   [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    if (gid.x >= uint(nx) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    // Linear index
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    int idx = ((k * ny + j) * nx + i) * NVAR;
    
    // Vectorized update ideal, but NVAR=8 is not a standard vector type.
    // Loop over variables.
    for (int v = 0; v < NVAR; v++) {
        float u1_val = U_1[idx + v];
        float u2_val = U_2[idx + v];
        float rhs_val = RHS[idx + v];
        
        // Physics checks / floors could be applied here if needed,
        // but typically done after reconstruction or end of step.
        // For strict conservation, we just update.
        
        float res = c1 * u1_val + c2 * u2_val + c3 * dt * rhs_val;
        
        if (v == IDN || v == IEN) {
             // Density and Energy floors
             if (v == IDN) res = max(res, RHO_FLOOR);
             if (v == IEN) res = max(res, P_FLOOR / (5.0/3.0 - 1.0)); // Approx check
        }
        
        U_out[idx + v] = res;
    }
}
