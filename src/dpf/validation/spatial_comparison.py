"""Spatial validation: compare MHD density profiles to experimental data.

Provides functions to compare simulated electron density profiles against
line-integrated density measurements from laser interferometry.

The workflow:
1. Extract ne(r) from MHD state at a given axial position
2. Abel-transform to get line-integrated density N_L(y)
3. Compare N_L(y) to experimental interferometry data
4. Report NRMSE and profile-resolved metrics

References:
    Malir et al. (2024), Phys. Plasmas, DOI 10.1063/5.0193268
    Hutchinson, "Principles of Plasma Diagnostics" (2002)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpf.diagnostics.interferometry import abel_transform
from dpf.metal.floor_telemetry import apply_floor


@dataclass
class SpatialComparisonResult:
    """Result of a density profile comparison.

    Attributes:
        nrmse: Normalized root-mean-square error (peak-normalized).
        r_sim: Radial coordinates of simulation profile [m].
        ne_sim: Simulated electron density profile [m^-3].
        NL_sim: Abel-transformed line-integrated density from simulation [m^-2].
        r_exp: Radial coordinates of experimental data [m].
        NL_exp: Experimental line-integrated density [m^-2].
        peak_ratio: Ratio of simulation peak to experimental peak NL.
        fwhm_sim: Full width at half maximum of simulation NL profile [m].
        fwhm_exp: Full width at half maximum of experimental NL profile [m].
    """

    nrmse: float
    r_sim: np.ndarray
    ne_sim: np.ndarray
    NL_sim: np.ndarray
    r_exp: np.ndarray
    NL_exp: np.ndarray
    peak_ratio: float
    fwhm_sim: float
    fwhm_exp: float


def _fwhm(r: np.ndarray, profile: np.ndarray) -> float:
    """Compute full width at half maximum of a radial profile."""
    half_max = 0.5 * np.max(profile)
    above = profile >= half_max
    if not np.any(above):
        return 0.0
    r_above = r[above]
    return float(r_above[-1] - r_above[0]) if len(r_above) > 1 else 0.0


def extract_density_profile(
    state: dict[str, np.ndarray],
    z_index: int | None = None,
    z_position: float | None = None,
    dz: float | None = None,
    ion_mass: float = 3.34358377e-27,
    Z_bar: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract electron density profile ne(r) from MHD state at a given z.

    Args:
        state: MHD state dict with 'rho' array of shape (nr, nz) or (nr, 1, nz).
        z_index: Axial cell index. If None, uses z_position.
        z_position: Axial position [m]. Requires dz. Ignored if z_index given.
        dz: Axial cell size [m].
        ion_mass: Ion mass [kg] (default deuterium).
        Z_bar: Mean ionization state (default 1 = fully ionized).

    Returns:
        Tuple (r, ne) where r is the radial coordinate array [m] and ne is
        the electron density [m^-3].
    """
    rho = state["rho"]
    if rho.ndim == 3:
        rho = rho[:, 0, :]  # (nr, nz)

    nr, nz = rho.shape

    if z_index is None:
        if z_position is not None and dz is not None:
            z_index = min(int(z_position / dz), nz - 1)
        else:
            z_index = nz // 2  # default: midplane

    rho_slice = rho[:, z_index]
    n_i = rho_slice / ion_mass
    ne = Z_bar * n_i

    # Radial coordinate (cell-centered, assume uniform grid)
    dx = float(state.get("dx", 0.01)) if isinstance(state, dict) else 0.01
    r = (np.arange(nr) + 0.5) * dx

    return r, ne


def compare_density_profile(
    state: dict[str, np.ndarray],
    r_exp: np.ndarray,
    NL_exp: np.ndarray,
    z_index: int | None = None,
    dx: float = 0.01,
    ion_mass: float = 3.34358377e-27,
    Z_bar: float = 1.0,
) -> SpatialComparisonResult:
    """Compare simulated density profile to experimental interferometry data.

    Computes the Abel transform of the simulated ne(r) to produce
    line-integrated density N_L(y), then compares to experimental N_L(y)
    via NRMSE.

    Args:
        state: MHD state dict with 'rho' array.
        r_exp: Experimental radial/impact parameter array [m].
        NL_exp: Experimental line-integrated density [m^-2].
        z_index: Axial slice index (default: midplane).
        dx: Radial cell size [m].
        ion_mass: Ion mass [kg].
        Z_bar: Mean ionization state.

    Returns:
        SpatialComparisonResult with NRMSE and profiles.
    """
    state_with_dx = dict(state)
    state_with_dx["dx"] = dx

    r_sim, ne_sim = extract_density_profile(
        state_with_dx, z_index=z_index, ion_mass=ion_mass, Z_bar=Z_bar,
    )

    # Abel transform: ne(r) -> N_L(y)
    NL_sim = abel_transform(ne_sim, r_sim)

    # Interpolate simulation onto experimental radial grid
    NL_sim_interp = np.interp(r_exp, r_sim, NL_sim, left=NL_sim[0], right=0.0)

    # Peak-normalized NRMSE
    NL_peak = max(float(np.max(np.abs(NL_exp))), 1e-30)
    residuals = NL_sim_interp - NL_exp
    nrmse = float(np.sqrt(np.mean(residuals**2)) / NL_peak)

    peak_ratio = float(np.max(NL_sim)) / NL_peak

    return SpatialComparisonResult(
        nrmse=nrmse,
        r_sim=r_sim,
        ne_sim=ne_sim,
        NL_sim=NL_sim,
        r_exp=r_exp,
        NL_exp=NL_exp,
        peak_ratio=peak_ratio,
        fwhm_sim=_fwhm(r_sim, NL_sim),
        fwhm_exp=_fwhm(r_exp, NL_exp),
    )


def spatial_nrmse_multi(
    results: list[SpatialComparisonResult],
) -> dict[str, float]:
    """Aggregate statistics from multiple time-slice comparisons.

    Args:
        results: List of SpatialComparisonResult from different times.

    Returns:
        Dict with mean_nrmse, max_nrmse, mean_peak_ratio, fwhm_ratio.
    """
    nrmses = [r.nrmse for r in results]
    peak_ratios = [r.peak_ratio for r in results]
    fwhm_ratios = [
        r.fwhm_sim / apply_floor(
            r.fwhm_exp, 1e-10, "spatial_comparison/fwhm_exp_div_guard",
        )
        for r in results
    ]
    return {
        "mean_nrmse": float(np.mean(nrmses)),
        "max_nrmse": float(np.max(nrmses)),
        "std_nrmse": float(np.std(nrmses)),
        "mean_peak_ratio": float(np.mean(peak_ratios)),
        "mean_fwhm_ratio": float(np.mean(fwhm_ratios)),
        "n_slices": len(results),
    }
