"""Apple Metal GPU acceleration for DPF.

Provides Metal-accelerated solvers and inference via:
- PyTorch MPS backend for MHD physics kernels
- Smart device routing based on operation type and grid size

Modules:
    device          — Hardware detection and device management
    device_router   — Per-operation CPU/GPU routing decisions
    metal_stencil   — PyTorch MPS stencil operations (CT, divergence, gradient)
    metal_riemann   — Vectorized HLL & HLLD Riemann solvers on MPS
    metal_solver    — MetalMHDSolver (PlasmaSolverBase implementation)
"""

from __future__ import annotations
