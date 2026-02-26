//
//  common.metal
//  DPF Unified - Metal Backend
//
//  Shared definitions for MHD kernels.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants & Configuration
// ============================================================================

// Number of conservative variables per cell
constant int NVAR = 8;

// Conservative variable indices
constant int IDN = 0;   // Density
constant int IM1 = 1;   // Momentum X
constant int IM2 = 2;   // Momentum Y
constant int IM3 = 3;   // Momentum Z
constant int IEN = 4;   // Total Energy
constant int IB1 = 5;   // B_x
constant int IB2 = 6;   // B_y
constant int IB3 = 7;   // B_z

// Floors for physical realizability
constant float RHO_FLOOR = 1e-12f;
constant float P_FLOOR   = 1e-12f;
constant float V_MAX     = 1e6f;

// Threadgroup block size for reconstruction
constant int BLOCK_SIZE_X = 4;
constant int BLOCK_SIZE_Y = 4;
constant int BLOCK_SIZE_Z = 4;

// ============================================================================
// Data Structures
// ============================================================================

// Conservative state vector (8 components)
// Using float array inside struct to allow passing between helper functions
struct ConsState {
    float u[8];
};

struct PrimState {
    float rho;
    float v[3];
    float p;
    float B[3];
};

// ============================================================================
// Helper Functions: Primitive <-> Conservative
// ============================================================================

// Convert Conservative -> Primitive
// U: [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]
inline PrimState cons_to_prim(ConsState U, float gamma) {
    PrimState W;
    
    W.rho = max(U.u[IDN], RHO_FLOOR);
    float inv_rho = 1.0f / W.rho;
    
    W.v[0] = U.u[IM1] * inv_rho;
    W.v[1] = U.u[IM2] * inv_rho;
    W.v[2] = U.u[IM3] * inv_rho;
    
    W.B[0] = U.u[IB1];
    W.B[1] = U.u[IB2];
    W.B[2] = U.u[IB3];
    
    float E = U.u[IEN];
    float KE = 0.5f * W.rho * (W.v[0]*W.v[0] + W.v[1]*W.v[1] + W.v[2]*W.v[2]);
    float ME = 0.5f * (W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2]);
    
    W.p = (gamma - 1.0f) * (E - KE - ME);
    W.p = max(W.p, P_FLOOR);
    
    return W;
}

// Convert Primitive -> Conservative
inline ConsState prim_to_cons(PrimState W, float gamma) {
    ConsState U;
    
    float rho = max(W.rho, RHO_FLOOR);
    float p   = max(W.p, P_FLOOR);
    
    float KE = 0.5f * rho * (W.v[0]*W.v[0] + W.v[1]*W.v[1] + W.v[2]*W.v[2]);
    float ME = 0.5f * (W.B[0]*W.B[0] + W.B[1]*W.B[1] + W.B[2]*W.B[2]);
    float E_total = p / (gamma - 1.0f) + KE + ME;
    
    U.u[IDN] = rho;
    U.u[IM1] = rho * W.v[0];
    U.u[IM2] = rho * W.v[1];
    U.u[IM3] = rho * W.v[2];
    U.u[IEN] = E_total;
    U.u[IB1] = W.B[0];
    U.u[IB2] = W.B[1];
    U.u[IB3] = W.B[2];
    
    return U;
}
