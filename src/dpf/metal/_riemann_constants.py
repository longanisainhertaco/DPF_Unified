"""Conservative MHD variable constants for the Metal Riemann solver.

Shared by all submodules in the metal_riemann split.
"""

from __future__ import annotations

# Number of conservative MHD variables:
#   [rho, rho*vx, rho*vy, rho*vz, E_total, Bx, By, Bz]
NVAR: int = 8

# Index aliases for the conservative state vector U[8, ...]
IDN: int = 0   # density
IM1: int = 1   # x-momentum  (rho * vx)
IM2: int = 2   # y-momentum  (rho * vy)
IM3: int = 3   # z-momentum  (rho * vz)
IEN: int = 4   # total energy
IB1: int = 5   # Bx
IB2: int = 6   # By
IB3: int = 7   # Bz
IEE: int = 8   # electron energy density (optional 9th variable)

# Density and pressure floors
RHO_FLOOR: float = 1e-12
P_FLOOR: float = 1e-12
