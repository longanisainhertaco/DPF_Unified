"""Verification test problems for the DPF solver."""

from dpf.verification.cylindrical_convergence import (
    check_equilibrium_preservation,
    run_convergence_test,
    setup_zpinch_equilibrium,
)
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
    "check_equilibrium_preservation",
    "gaussian_B_analytical",
    "run_convergence_test",
    "run_diffusion_convergence",
    "run_orszag_tang",
    "run_sedov_cylindrical",
    "sedov_shock_radius_cylindrical",
    "setup_zpinch_equilibrium",
]
