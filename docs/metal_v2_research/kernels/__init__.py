"""MLX Metal MHD Kernels for DPF-Unified.

Three custom MSL kernels for cylindrical MHD on Apple Silicon:
  1. ghost_cell_kernel  - Ghost cell padding with electrode BCs
  2. hlld_kernel        - HLLD Riemann solver (Miyoshi & Kusano 2005)
  3. cylindrical_source_kernel - Cylindrical geometric source terms
"""

from .cylindrical_source_kernel import cylindrical_source_mlx, cylindrical_source_numpy
from .ghost_cell_kernel import ghost_cell_pad_mlx, ghost_cell_pad_numpy
from .hlld_kernel import hlld_flux_mlx, hlld_flux_numpy

__all__ = [
    "ghost_cell_pad_mlx",
    "ghost_cell_pad_numpy",
    "hlld_flux_mlx",
    "hlld_flux_numpy",
    "cylindrical_source_mlx",
    "cylindrical_source_numpy",
]
