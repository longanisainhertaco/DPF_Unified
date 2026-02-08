"""Experimental modules — code exists but is NOT integrated into the simulation engine.

These modules are complete or near-complete implementations that have not yet been
wired into engine.py. They are kept here as building blocks for future development.

Modules:
    amr/        - Adaptive mesh refinement (block-structured, gradient tagging)
    pic/        - Hybrid particle-in-cell (Boris pusher, CIC deposition)
    species.py  - Multi-species tracking (SpeciesMixture, advection)
    gpu_backend.py - CuPy GPU detection stub (no actual kernels)
"""
