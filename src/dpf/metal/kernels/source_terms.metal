//
//  source_terms.metal
//  DPF Unified - Metal Backend
//
//  Compute kernel for extended MHD source terms:
//  1. Hall Term: E_hall = (J x B) / (ne * e)
//  2. Anomalous Resistivity: E_res = eta * J, Q_ohm = eta * J^2
//
//  Updates B and Energy/Pressure in-place or via delta.
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

// ============================================================================
// Source Terms Kernel
// ============================================================================

kernel void add_source_terms(
    device       float*  U     [[ buffer(0) ]], // Conservative state (Read/Write)
    constant     int3&   dims  [[ buffer(1) ]],
    constant     float&  gamma [[ buffer(2) ]],
    constant     float&  dt    [[ buffer(3) ]],
    constant     float&  dx    [[ buffer(4) ]],
    constant     float&  dy    [[ buffer(5) ]],
    constant     float&  dz    [[ buffer(6) ]],
    constant     int&    enable_hall [[ buffer(7) ]],
    constant     int&    enable_res  [[ buffer(8) ]],
    constant     float&  ion_mass    [[ buffer(9) ]],
    constant     float&  anom_alpha  [[ buffer(10) ]],
    uint3                gid   [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    if (gid.x >= uint(nx) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    
    // Indices for central differences (handle boundaries by clamping)
    auto idx = [&](int _i, int _j, int _k) {
        int ii = clamp(_i, 0, nx-1);
        int jj = clamp(_j, 0, ny-1);
        int kk = clamp(_k, 0, nz-1);
        return ((kk * ny + jj) * nx + ii) * NVAR;
    };
    
    int c_idx = idx(i, j, k);
    
    // Read local state
    float rho = U[c_idx + IDN];
    float Bx  = U[c_idx + IB1];
    float By  = U[c_idx + IB2];
    float Bz  = U[c_idx + IB3];
    float p_curr = 0.0f; // Need to compute from E? Or just update E directly?
    // E = p/(g-1) + 0.5*rho*v^2 + 0.5*B^2
    // We only need to update E for Ohmic heating.
    
    // ------------------------------------------------------------------------
    // 1. Compute Current Density J = curl(B)
    // ------------------------------------------------------------------------
    // Central difference
    
    float Bx_dy_p = U[idx(i, j+1, k) + IB1];
    float Bx_dy_m = U[idx(i, j-1, k) + IB1];
    float dBx_dy = (Bx_dy_p - Bx_dy_m) / (2.0f * dy);
    
    float Bx_dz_p = U[idx(i, j, k+1) + IB1];
    float Bx_dz_m = U[idx(i, j, k-1) + IB1];
    float dBx_dz = (Bx_dz_p - Bx_dz_m) / (2.0f * dz);
    
    float By_dx_p = U[idx(i+1, j, k) + IB2];
    float By_dx_m = U[idx(i-1, j, k) + IB2];
    float dBy_dx = (By_dx_p - By_dx_m) / (2.0f * dx);
    
    float By_dz_p = U[idx(i, j, k+1) + IB2];
    float By_dz_m = U[idx(i, j, k-1) + IB2];
    float dBy_dz = (By_dz_p - By_dz_m) / (2.0f * dz);
    
    float Bz_dx_p = U[idx(i+1, j, k) + IB3];
    float Bz_dx_m = U[idx(i-1, j, k) + IB3];
    float Bz_dx = (Bz_dx_p - Bz_dx_m) / (2.0f * dx);
    
    float Bz_dy_p = U[idx(i, j+1, k) + IB3];
    float Bz_dy_m = U[idx(i, j-1, k) + IB3];
    float Bz_dy = (Bz_dy_p - Bz_dy_m) / (2.0f * dy);
    
    float3 J = float3(Bz_dy - dBy_dz, dBx_dz - Bz_dx, dBy_dx - dBx_dy);
    
    // ------------------------------------------------------------------------
    // 2. Physics Terms
    // ------------------------------------------------------------------------
    
    float3 E_total = float3(0.0f);
    float Q_ohm = 0.0f;
    float ne = rho / ion_mass;
    float ne_safe = max(ne, 1e10f); // Floor
    float e_charge = 1.60217663e-19f;
    
    // A. Hall Term: E = (J x B) / (ne * e)
    if (enable_hall) {
        float3 B_vec = float3(Bx, By, Bz);
        float3 JxB = cross(J, B_vec);
        E_total += JxB / (ne_safe * e_charge);
    }
    
    // B. Anomalous Resistivity
    if (enable_res) {
        // Simplified Threshold: v_d > v_ti
        // Need Pressure for v_ti. 
        // E = p/(g-1) + K + M.
        // p = (g-1)*(E - K - M)
        float E_int = U[c_idx + IEN];
        float vx = U[c_idx + IM1] / rho;
        float vy = U[c_idx + IM2] / rho;
        float vz = U[c_idx + IM3] / rho;
        float v2 = vx*vx + vy*vy + vz*vz;
        float B2 = Bx*Bx + By*By + Bz*Bz;
        float p = (gamma - 1.0f) * (E_int - 0.5f*rho*v2 - 0.5f*B2);
        p = max(p, 1e-5f);
        
        float v_ti = sqrt(0.5f * p / rho);
        float J_mag = length(J);
        float v_d = J_mag / (ne_safe * e_charge);
        
        if (v_d > v_ti) {
            // eta = alpha * me * w_pe / (ne * e^2)
            // w_pe = sqrt(ne*e^2 / (eps0 * me))
            float me = 9.109e-31f;
            float eps0 = 8.854e-12f;
            float w_pe = sqrt(ne_safe * e_charge * e_charge / (eps0 * me));
            float eta = anom_alpha * me * w_pe / (ne_safe * e_charge * e_charge);
            
            E_total += eta * J;
            Q_ohm += eta * dot(J, J);
        }
    }
    
    // ------------------------------------------------------------------------
    // 3. Update B-field: dB/dt = -curl(E)
    // ------------------------------------------------------------------------
    // Need curl(E). But E is computed locally!
    // We cannot compute curl(E) in the same kernel pass unless we use shared mem or 2 passes.
    // 
    // CHANGE OF PLAN:
    // This kernel should compute E_total and store it in an auxiliary buffer?
    // Or we compute E at neighbors? That's expensive (recomputing J everywhere).
    //
    // Simplified Staggered approach or Operator Splitting:
    // For now, let's just store E_total in a buffer, then verify. 
    // But implementation plan said "Update B... in-place".
    //
    // If we want single kernel, we can't do gradients of locally computed values easily.
    //
    // Correction:
    // We will make this kernel return the explicit Source Updates dU/dt?
    // Or we split it: 1. Compute E_hall + E_res. 2. Compute Curl(E).
    //
    // Let's modify the kernel to OUTPUT 'E_eff' (Effective E-field from sources).
    // Then Python (or another kernel) takes curl.
    //
    // Wait, reusing `metal_solver.py` simple differences?
    // If we want performance, we should do it all on GPU.
    //
    // Let's define TWO kernels in this file:
    // 1. compute_extended_E ( Computes E_hall + E_res -> E_out )
    // 2. update_from_E ( Computes -curl(E_out) -> B, and Q_ohm -> E )
}

// ----------------------------------------------------------------------------
// Revised Kernels
// ----------------------------------------------------------------------------

kernel void compute_extended_e(
    device const float*  U     [[ buffer(0) ]],
    device       float*  E_out [[ buffer(1) ]], // Store E (3 components) + Q_ohm (1 comp) = 4 floats/cell?
    constant     int3&   dims  [[ buffer(2) ]],
    constant     float&  gamma [[ buffer(3) ]],
    constant     float&  dx    [[ buffer(4) ]],
    constant     float&  dy    [[ buffer(5) ]],
    constant     float&  dz    [[ buffer(6) ]],
    constant     int&    enable_hall [[ buffer(7) ]],
    constant     int&    enable_res  [[ buffer(8) ]],
    constant     float&  ion_mass    [[ buffer(9) ]],
    constant     float&  anom_alpha  [[ buffer(10) ]],
    uint3                gid   [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    if (gid.x >= uint(nx) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    
    // Helper indices
    auto idx = [&](int _i, int _j, int _k) {
        int ii = clamp(_i, 0, nx-1);
        int jj = clamp(_j, 0, ny-1);
        int kk = clamp(_k, 0, nz-1);
        return ((kk * ny + jj) * nx + ii) * NVAR;
    };
    
    int c_idx = idx(i, j, k);
    int e_idx = ((k * ny + j) * nx + i) * 4; // E_x, E_y, E_z, Q_ohm
    
    // Read State
    float rho = U[c_idx + IDN];
    float Bx  = U[c_idx + IB1];
    float By  = U[c_idx + IB2];
    float Bz  = U[c_idx + IB3];
    
    // Calc J via Central Diff
     float Bx_dy_p = U[idx(i, j+1, k) + IB1];
    float Bx_dy_m = U[idx(i, j-1, k) + IB1];
    float dBx_dy = (Bx_dy_p - Bx_dy_m) / (2.0f * dy);
    
    float Bx_dz_p = U[idx(i, j, k+1) + IB1];
    float Bx_dz_m = U[idx(i, j, k-1) + IB1];
    float dBx_dz = (Bx_dz_p - Bx_dz_m) / (2.0f * dz);
    
    float By_dx_p = U[idx(i+1, j, k) + IB2];
    float By_dx_m = U[idx(i-1, j, k) + IB2];
    float dBy_dx = (By_dx_p - By_dx_m) / (2.0f * dx);
    
    float By_dz_p = U[idx(i, j, k+1) + IB2];
    float By_dz_m = U[idx(i, j, k-1) + IB2];
    float dBy_dz = (By_dz_p - By_dz_m) / (2.0f * dz);
    
    float Bz_dx_p = U[idx(i+1, j, k) + IB3];
    float Bz_dx_m = U[idx(i-1, j, k) + IB3];
    float Bz_dx = (Bz_dx_p - Bz_dx_m) / (2.0f * dx);
    
    float Bz_dy_p = U[idx(i, j+1, k) + IB3];
    float Bz_dy_m = U[idx(i, j-1, k) + IB3];
    float Bz_dy = (Bz_dy_p - Bz_dy_m) / (2.0f * dy);
    
    float3 J = float3(Bz_dy - dBy_dz, dBx_dz - Bz_dx, dBy_dx - dBx_dy);
    
    float3 E_total = float3(0.0f);
    float Q_ohm = 0.0f;
    float ne = rho / ion_mass;
    float ne_safe = max(ne, 1e10f); 
    float e_charge = 1.60217663e-19f;
    
    if (enable_hall) {
         float3 B_vec = float3(Bx, By, Bz);
         float3 JxB = cross(J, B_vec);
         E_total += JxB / (ne_safe * e_charge);
    }
    
    if (enable_res) {
        float E_int = U[c_idx + IEN];
        float vx = U[c_idx + IM1] / rho;
        float vy = U[c_idx + IM2] / rho;
        float vz = U[c_idx + IM3] / rho;
        float v2 = vx*vx + vy*vy + vz*vz;
        float B2 = Bx*Bx + By*By + Bz*Bz;
        float p = (gamma - 1.0f) * (E_int - 0.5f*rho*v2 - 0.5f*B2);
        p = max(p, 1e-5f);
        
        float v_ti = sqrt(0.5f * p / rho);
        float J_mag = length(J);
        float v_d = J_mag / (ne_safe * e_charge);
        
        if (v_d > v_ti) {
             float me = 9.109e-31f;
             float eps0 = 8.854e-12f;
             float w_pe = sqrt(ne_safe * e_charge * e_charge / (eps0 * me));
             float eta = anom_alpha * me * w_pe / (ne_safe * e_charge * e_charge);
             
             E_total += eta * J;
             Q_ohm += eta * dot(J, J);
        }
    }
    
    E_out[e_idx + 0] = E_total.x;
    E_out[e_idx + 1] = E_total.y;
    E_out[e_idx + 2] = E_total.z;
    E_out[e_idx + 3] = Q_ohm;
}

// Compute Source Terms contribution to RHS (dU/dt)
// RHS_out = [0, 0, 0, 0, Q_ohm, -curlE_x, -curlE_y, -curlE_z]
kernel void compute_source_rhs(
    device       float*  RHS_out [[ buffer(0) ]],
    device const float*  E_in    [[ buffer(1) ]],
    constant     int3&   dims    [[ buffer(2) ]],
    constant     float&  dx      [[ buffer(3) ]],
    constant     float&  dy      [[ buffer(4) ]],
    constant     float&  dz      [[ buffer(5) ]],
    uint3                gid     [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    if (gid.x >= uint(nx) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    
    int c_idx = ((k * ny + j) * nx + i) * NVAR;
    
    // Initialize all to 0.0
    for(int v=0; v<NVAR; ++v) {
        RHS_out[c_idx + v] = 0.0f;
    }
    
    // Helper to read E
    auto get_E = [&](int _i, int _j, int _k) {
         int ii = clamp(_i, 0, nx-1);
         int jj = clamp(_j, 0, ny-1);
         int kk = clamp(_k, 0, nz-1);
         int e_idx = ((kk * ny + jj) * nx + ii) * 4;
         return float3(E_in[e_idx+0], E_in[e_idx+1], E_in[e_idx+2]);
    };
    
    float Q_ohm = E_in[((k * ny + j) * nx + i) * 4 + 3];
    
    // Curl(E)
    float3 E_dy_p = get_E(i, j+1, k);
    float3 E_dy_m = get_E(i, j-1, k);
    float3 dE_dy = (E_dy_p - E_dy_m) / (2.0f * dy);
    
    float3 E_dz_p = get_E(i, j, k+1);
    float3 E_dz_m = get_E(i, j, k-1);
    float3 dE_dz = (E_dz_p - E_dz_m) / (2.0f * dz);
    
    float3 E_dx_p = get_E(i+1, j, k);
    float3 E_dx_m = get_E(i-1, j, k);
    float3 dE_dx = (E_dx_p - E_dx_m) / (2.0f * dx);
    
    float curlE_x = dE_dy.z - dE_dz.y;
    float curlE_y = dE_dz.x - dE_dx.z;
    float curlE_z = dE_dx.y - dE_dy.x;
    
    // Update B: dB/dt = -curl(E)
    RHS_out[c_idx + IB1] = -curlE_x;
    RHS_out[c_idx + IB2] = -curlE_y;
    RHS_out[c_idx + IB3] = -curlE_z;
    
    // Update Energy: dE/dt = Q_ohm
    RHS_out[c_idx + IEN] = Q_ohm;
}
