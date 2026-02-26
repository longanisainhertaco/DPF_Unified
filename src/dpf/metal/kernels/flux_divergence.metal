//
//  flux_divergence.metal
//  DPF Unified - Metal Backend
//
//  Computes the divergence of fluxes to get the RHS:
//  dU/dt = - (dF/dx + dG/dy + dH/dz) + Source
//
//  This kernel only handles the flux divergence part.
//  Source terms are handled separately.
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

kernel void compute_flux_divergence(
    device       float*  RHS   [[ buffer(0) ]],
    device const float*  F_x   [[ buffer(1) ]],
    device const float*  F_y   [[ buffer(2) ]],
    device const float*  F_z   [[ buffer(3) ]], // Optional, can be null for 2D
    constant     int3&   dims  [[ buffer(4) ]],
    constant     float&  dx    [[ buffer(5) ]],
    constant     float&  dy    [[ buffer(6) ]],
    constant     float&  dz    [[ buffer(7) ]],
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
    
    // Linear index for cell center
    int idx = ((k * ny + j) * nx + i) * NVAR;
    
    // Initialize RHS with zero (or accumulate if needed? Standard RK usually overwrites)
    // We overwrite here as this is the primary dU/dt term.
    for (int v = 0; v < NVAR; v++) {
        RHS[idx + v] = 0.0f;
    }
    
    float inv_dx = 1.0f / dx;
    float inv_dy = 1.0f / dy;
    float inv_dz = 1.0f / dz;
    
    // ----------------------------------------------------------------
    // X-Flux Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
    // F_x stores fluxes at i+1/2.
    // Index mapping: F_x[i] is at i+1/2. F_x[i-1] is at i-1/2.
    // ----------------------------------------------------------------
    
    // F_i_plus: Flux at i+1/2 (Right face)
    // F_i_minus: Flux at i-1/2 (Left face)
    
    // Boundary conditions for flux array access need care.
    // mhd_sweep_x computes fluxes for i=0..nx-2 (interior interfaces).
    // so F_x has size (nx-1)*ny*nz usually, or we use a full grid and mask.
    // Let's assume F_x has shape (nx+1, ny, nz) to handle all interfaces,
    // or standard Athena layout (nx+1).
    // The previous mhd_sweep_x wrote to index corresponding to i (interface i+1/2).
    
    // Check bounds for flux reading
    // IF i=0, we need F_{-1/2} -> Boundary condition? 
    // Usually solvers populate ghost zones or we assume F buffer has ghosts.
    // For this implementation, we assume F_x covers interfaces from 0 to nx.
    // Let's assume F_x is size (nx+1) * ny * nz * NVAR.
    
    // Simplification: In metal_solver, we might just compute internal fluxes.
    // If i=0 or i=nx-1, we might just set RHS=0 (ghost cells).
    
    if (i > 0 && i < nx - 1) {
        int idx_f_p = ((k * ny + j) * (nx + 1) + i) * NVAR;      // i+1/2
        int idx_f_m = ((k * ny + j) * (nx + 1) + (i - 1)) * NVAR; // i-1/2
        
        // Wait, mhd_sweep_x outputted to `(nx-1)` size grid?
        // Let's check mhd_sweep_x.metal matches:
        // `int out_base_idx = ((k * ny + j) * (nx - 1) + i) * NVAR;`
        // It writes i=0..nx-2. So i corresponds to interface i+1/2.
        
        // So for cell i:
        // Right flux (i+1/2) is at index i.
        // Left flux (i-1/2) is at index i-1.
        
        int idx_plus  = ((k * ny + j) * (nx - 1) + i) * NVAR;
        int idx_minus = ((k * ny + j) * (nx - 1) + (i - 1)) * NVAR;
        
        for (int v = 0; v < NVAR; v++) {
            float f_plus  = F_x[idx_plus + v];
            float f_minus = F_x[idx_minus + v];
            RHS[idx + v] -= (f_plus - f_minus) * inv_dx;
        }
    }
    
    // ----------------------------------------------------------------
    // Y-Flux Divergence
    // ----------------------------------------------------------------
    if (ny > 1 && j > 0 && j < ny - 1) {
        // F_y buffer assumed similar layout: (nx, ny-1, nz)
        int idx_plus  = ((k * (ny - 1) + j) * nx + i) * NVAR;
        int idx_minus = ((k * (ny - 1) + (j - 1)) * nx + i) * NVAR;
        
        for (int v = 0; v < NVAR; v++) {
            float f_plus  = F_y[idx_plus + v];
            float f_minus = F_y[idx_minus + v];
            RHS[idx + v] -= (f_plus - f_minus) * inv_dy;
        }
    }
    
    // ----------------------------------------------------------------
    // Z-Flux Divergence
    // ----------------------------------------------------------------
    if (nz > 1 && k > 0 && k < nz - 1) {
        // F_z buffer: (nx, ny, nz-1)
        int idx_plus  = ((k * ny + j) * nx + i) * NVAR;       // This indexing is tricky if flat
        // Proper striding for Z-faces usually:
        // k * (ny * nx) + ... 
        
        int stride_z = nx * ny * NVAR;
        int idx_plus_z  = k * stride_z + (j * nx + i) * NVAR;
        int idx_minus_z = (k - 1) * stride_z + (j * nx + i) * NVAR;
        
        for (int v = 0; v < NVAR; v++) {
            float f_plus  = F_z[idx_plus_z + v];
            float f_minus = F_z[idx_minus_z + v];
            RHS[idx + v] -= (f_plus - f_minus) * inv_dz;
        }
    }
}
