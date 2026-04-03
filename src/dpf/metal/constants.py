"""Physics constants — single source of truth for all DPF-Unified MLX modules.

Every Python file in src/dpf/metal/ MUST import constants from here.
Do NOT define P_FLOOR, RHO_FLOOR, C_BORIS, MU_0, or GAMMA inline.

The Metal MSL shader (mlx_kernels.py) cannot import Python — its constants
are hardcoded in the shader source. test_constants_consistency.py verifies
they match these values.

References:
    NIST CODATA 2018 for fundamental constants.
    Gombosi et al. 2002, JCP 177:176 for Boris correction.
"""

from __future__ import annotations

import math

# ── Fundamental Constants (SI) ────────────────────────────────────

MU_0: float = 4.0 * math.pi * 1e-7       # Permeability of free space [H/m]
SQRT_MU0: float = math.sqrt(MU_0)         # For HL ↔ SI B-field conversion
K_B: float = 1.380649e-23                  # Boltzmann constant [J/K]
E_CHARGE: float = 1.602176634e-19          # Elementary charge [C]
EPS_0: float = 8.854187817e-12             # Permittivity of free space [F/m]
M_ELECTRON: float = 9.10938e-31            # Electron mass [kg]
M_DEUTERIUM: float = 3.34358377e-27        # Deuterium ion mass [kg]
M_PROTON: float = 1.67262192e-27           # Proton mass [kg]

# ── Numerical Floors ──────────────────────────────────────────────

RHO_FLOOR: float = 1e-12                   # Minimum density [kg/m^3]
P_FLOOR: float = 1e-12                     # Minimum pressure [Pa]

# ── Default Adiabatic Index ───────────────────────────────────────

GAMMA: float = 5.0 / 3.0                   # Monatomic ideal gas

# ── Boris Correction ──────────────────────────────────────────────
# Gombosi et al. 2002, JCP 177:176; Matsumoto, Miyoshi & Takasao 2019, ApJ 874:37.
# c_boris should be 3-10x the maximum physical flow speed.
# For DPF: max sheath velocity ~200 km/s, so c_boris = 500 km/s gives 2.5x margin.

C_BORIS: float = 5e5                       # Reduced speed of light [m/s]
C_BORIS_SQ: float = C_BORIS * C_BORIS      # (500 km/s)^2 = 2.5e11

# ── Velocity Clamping ─────────────────────────────────────────────

V_MAX: float = 1e6                         # Maximum velocity for clamping [m/s]
V_CLAMP_FACTOR: float = 10.0               # Velocity clamp = factor * fast magnetosonic

# ── Transport ─────────────────────────────────────────────────────

COULOMB_LOG_DEFAULT: float = 10.0           # Default Coulomb logarithm

# ── Variable Indices (conserved state layout) ─────────────────────
# These are duplicated in mlx_kernels.py for the Metal shader.
# test_constants_consistency.py verifies they match.

IDN: int = 0    # Density
IMR: int = 1    # Radial momentum
IMZ: int = 2    # Axial momentum
IMT: int = 3    # Azimuthal momentum
IEN: int = 4    # Total energy
ISR: int = 5    # Entropy tracer
IBR: int = 6    # Radial B-field
IBZ: int = 7    # Axial B-field
IBT: int = 8    # Azimuthal B-field
IEE: int = 9    # Electron energy
NVAR: int = 10  # Total number of conserved variables
