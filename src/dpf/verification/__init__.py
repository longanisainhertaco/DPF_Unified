"""Verification test problems for the DPF solver."""

from dpf.verification.diffusion_convergence import (
    DiffusionConvergenceResult,
    gaussian_B_analytical,
    run_diffusion_convergence,
)
from dpf.verification.orszag_tang import (
    OrszagTangResult,
    run_orszag_tang,
)
from dpf.verification.sedov_cylindrical import (
    SedovCylindricalResult,
    run_sedov_cylindrical,
    sedov_shock_radius_cylindrical,
)

__all__ = [
    "DiffusionConvergenceResult",
    "OrszagTangResult",
    "SedovCylindricalResult",
    "gaussian_B_analytical",
    "run_diffusion_convergence",
    "run_orszag_tang",
    "run_sedov_cylindrical",
    "sedov_shock_radius_cylindrical",
]
