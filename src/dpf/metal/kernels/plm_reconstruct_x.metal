//
//  plm_reconstruct.metal
//  DPF Unified - Metal Backend
//
//  Piecewise Linear Method (PLM) reconstruction.
//

#include <metal_stdlib>
#include "common.metal"

using namespace metal;

// ============================================================================
// Slope Limiters
// ============================================================================

// Minmod limiter
inline float minmod(float a, float b) {
    if (a * b > 0.0f) {
        return sign(a) * min(abs(a), abs(b));
    }
    return 0.0f;
}

// Monotonized Central (MC) limiter
inline float mc_limiter(float a, float b) {
    float c1 = 2.0f * a;
    float c2 = 0.5f * (a + b);
    float c3 = 2.0f * b;
    
    // Median of (2a, (a+b)/2, 2b) logic
    float max_val = max(max(c1, c2), c3);
    float min_val = min(min(c1, c2), c3);
    float med = c1 + c2 + c3 - max_val - min_val;
    
    if (a * b > 0.0f) {
        return med;
    }
    return 0.0f;
}

// ============================================================================
// PLM Kernel (X-Direction)
// ============================================================================

// Reconstruct U -> UL, UR along the X-axis (first dimension)
// Thread dispatch: (nx-1, ny, nz)
kernel void plm_reconstruct_x(
    device const float*  U_in  [[ buffer(0) ]],
    device       float*  UL_out [[ buffer(1) ]],
    device       float*  UR_out [[ buffer(2) ]],
    constant     int3&   dims   [[ buffer(3) ]], // (nx, ny, nz)
    constant     int&    limiter_type [[ buffer(4) ]], // 0=minmod, 1=mc
    uint3                gid    [[ thread_position_in_grid ]]
) {
    int nx = dims.x;
    int ny = dims.y;
    int nz = dims.z;
    
    // Grid bounds check (we compute interfaces i+1/2 for i=0..nx-2)
    if (gid.x >= uint(nx - 1) || gid.y >= uint(ny) || gid.z >= uint(nz)) {
        return;
    }
    
    int i = gid.x;
    int j = gid.y;
    int k = gid.z;
    
    // Stride for 8-component array
    // U is layout [NVAR, nx, ny, nz] or [nx, ny, nz, NVAR]?
    // Typically Athena uses struct-of-arrays or NVAR-last.
    // Let's assume [nx, ny, nz, NVAR] for coalescing.
    // Wait, common.metal structs suggest we might process component-wise?
    // Let's stick to standard layout: U[k][j][i][v] 
    
    // Flat index helper
    auto idx = [&](int _i, int _j, int _k, int _v) {
        return ((_k * ny + _j) * nx + _i) * NVAR + _v;
    };
    
    // We need 4 points for slope at i and i+1: i-1, i, i+1, i+2
    // But PLM only strictly needs i-1, i, i+1 for slope at i
    // and i, i+1, i+2 for slope at i+1.
    // Interface i+1/2 connects cell i (Right Face) and cell i+1 (Left Face).
    
    // Left State (UL) at i+1/2: U[i] + 0.5 * slope[i]
    // Right State (UR) at i+1/2: U[i+1] - 0.5 * slope[i+1]
    
    // Load cell i and i+1
    float u_i[8];
    float u_ip1[8];
    
    for (int v=0; v<NVAR; ++v) {
        u_i[v]   = U_in[idx(i, j, k, v)];
        u_ip1[v] = U_in[idx(i+1, j, k, v)];
    }
    
    // Calculate slope at i
    // needs i-1. If boundary, slope=0
    float slope_i[8];
    if (i > 0) {
        for (int v=0; v<NVAR; ++v) {
            float val_m1 = U_in[idx(i-1, j, k, v)];
            float dL = u_i[v] - val_m1;
            float dR = u_ip1[v] - u_i[v]; // This is actually U[i+1]-U[i] which can be used for both
            
            if (limiter_type == 1) {
                slope_i[v] = mc_limiter(dL, dR);
            } else {
                slope_i[v] = minmod(dL, dR);
            }
        }
    } else {
        for (int v=0; v<NVAR; ++v) slope_i[v] = 0.0f;
    }
    
    // Calculate slope at i+1
    // needs i+2. If boundary (i+1 == nx-1), e.g. i=nx-2, then i+1 is last cell.
    float slope_ip1[8];
    if (i + 1 < nx - 1) {
        for (int v=0; v<NVAR; ++v) {
            float val_ip2 = U_in[idx(i+2, j, k, v)];
            float dL = u_ip1[v] - u_i[v]; // U[i+1]-U[i]
            float dR = val_ip2 - u_ip1[v];
            
            if (limiter_type == 1) {
                slope_ip1[v] = mc_limiter(dL, dR);
            } else {
                slope_ip1[v] = minmod(dL, dR);
            }
        }
    } else {
        for (int v=0; v<NVAR; ++v) slope_ip1[v] = 0.0f;
    }
    
    // Reconstruct
    // Interface index is just i
    // Output dimensions are (nx-1, ny, nz)
    auto out_idx = [&](int _i, int _j, int _k, int _v) {
        return ((_k * ny + _j) * (nx-1) + _i) * NVAR + _v;
    };
    
    for (int v=0; v<NVAR; ++v) {
        float val_UL = u_i[v] + 0.5f * slope_i[v];
        float val_UR = u_ip1[v] - 0.5f * slope_ip1[v];
        
        // Enforce floors (Density & Pressure/Energy)?
        // Usually done after reconstruction but before flux.
        // For simplicity, we just write out.
        
        UL_out[out_idx(i, j, k, v)] = val_UL;
        UR_out[out_idx(i, j, k, v)] = val_UR;
    }
}
