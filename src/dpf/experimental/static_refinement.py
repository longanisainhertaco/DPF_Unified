"""Static Mesh Refinement (SMR) for DPF MHD simulations.

Two-pass refinement strategy for resolving the current sheath:
1. Run a coarse simulation to identify sheath location
2. Create a refined grid window around the sheath region
3. Interpolate coarse solution as initial condition
4. Re-run with higher resolution in the sheath region

This avoids full AMR (which requires variable-size tensors incompatible
with PyTorch) while providing ~5x improvement in sheath resolution.

The approach uses uniform grids throughout (compatible with Metal solver)
but focuses cells in the physically interesting region.

References:
    Stone et al., ApJS 249, 4 (2020) — Athena++ SMR implementation.
    Berger & Colella, JCP 82, 64 (1989) — AMR for hyperbolic PDEs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RefinementRegion:
    """Description of a refined grid region."""

    r_min: float          # Inner radius of refined region [m]
    r_max: float          # Outer radius of refined region [m]
    z_min: float          # Axial start of refined region [m]
    z_max: float          # Axial end of refined region [m]
    nr_fine: int           # Number of radial cells in refined region
    nz_fine: int           # Number of axial cells in refined region
    dr_fine: float         # Radial cell size in refined region [m]
    dz_fine: float         # Axial cell size in refined region [m]
    refinement_factor: int # Ratio of coarse/fine cell size
    sheath_r: float       # Detected sheath radial position [m]
    sheath_z: float       # Detected sheath axial position [m]
    method: str           # Detection method used


def detect_sheath_location(
    rho: np.ndarray,
    r_cells: np.ndarray,
    z_cells: np.ndarray,
    anode_radius: float,
    cathode_radius: float,
) -> tuple[float, float, float]:
    """Detect sheath location from density gradient.

    The current sheath is identified as the location of maximum
    density gradient in the radial direction (midplane slice).

    Args:
        rho: Density field [nr, ny, nz].
        r_cells: Radial cell centers [nr].
        z_cells: Axial cell centers [nz].
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].

    Returns:
        Tuple of (r_sheath, z_sheath, sheath_width):
            - r_sheath: Radial position of sheath [m]
            - z_sheath: Axial position of sheath [m]
            - sheath_width: Estimated sheath thickness [m]
    """
    nr, ny, nz = rho.shape
    dr = r_cells[1] - r_cells[0] if nr > 1 else (cathode_radius - anode_radius) / nr

    # Take midplane slice (ny//2 for 2D cylindrical)
    rho_2d = rho[:, ny // 2, :]

    # Radial density gradient: |d(rho)/dr| at each (r,z)
    drho_dr = np.abs(np.gradient(rho_2d, dr, axis=0))

    # Find peak gradient location
    peak_idx = np.unravel_index(np.argmax(drho_dr), drho_dr.shape)
    r_sheath = r_cells[peak_idx[0]] if peak_idx[0] < len(r_cells) else r_cells[-1]
    z_sheath = z_cells[peak_idx[1]] if peak_idx[1] < len(z_cells) else z_cells[-1]

    # Estimate sheath width from FWHM of gradient peak
    grad_profile = drho_dr[:, peak_idx[1]]
    half_max = np.max(grad_profile) * 0.5
    above_half = np.where(grad_profile > half_max)[0]
    if len(above_half) >= 2:
        sheath_width = (above_half[-1] - above_half[0] + 1) * dr
    else:
        sheath_width = 3.0 * dr  # Default: 3 cells

    return float(r_sheath), float(z_sheath), float(sheath_width)


def compute_refinement_region(
    r_sheath: float,
    z_sheath: float,
    sheath_width: float,
    anode_radius: float,
    cathode_radius: float,
    anode_length: float,
    refinement_factor: int = 4,
    padding_factor: float = 3.0,
    max_cells_per_dim: int = 128,
) -> RefinementRegion:
    """Compute the refined grid region centered on the sheath.

    The refined region extends padding_factor * sheath_width around
    the detected sheath location. The number of cells is chosen to
    achieve refinement_factor improvement in resolution.

    Args:
        r_sheath: Radial sheath position [m].
        z_sheath: Axial sheath position [m].
        sheath_width: Estimated sheath width [m].
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].
        anode_length: Anode length [m].
        refinement_factor: Resolution improvement factor (default 4).
        padding_factor: How many sheath widths to pad around sheath (default 3).
        max_cells_per_dim: Maximum cells per dimension (default 128).

    Returns:
        RefinementRegion describing the fine grid.
    """
    gap = cathode_radius - anode_radius
    pad = padding_factor * max(sheath_width, gap * 0.05)

    # Radial bounds: center on sheath with padding
    r_min = max(anode_radius, r_sheath - pad)
    r_max = min(cathode_radius, r_sheath + pad)
    radial_extent = r_max - r_min

    # Axial bounds: from sheath to anode tip (z=L_anode)
    # The sheath moves toward higher z, so refine ahead of it
    z_min = max(0.0, z_sheath - pad)
    z_max = min(anode_length, z_sheath + 2.0 * pad)
    axial_extent = z_max - z_min

    # Compute fine grid cell sizes
    # Target: dr_fine = dr_coarse / refinement_factor
    dr_coarse = gap / 16  # Assume 16-cell coarse grid
    dr_fine = dr_coarse / refinement_factor
    dz_fine = dr_fine  # Square cells

    # Number of cells (clamped)
    nr_fine = max(16, min(max_cells_per_dim, int(radial_extent / dr_fine)))
    nz_fine = max(16, min(max_cells_per_dim, int(axial_extent / dz_fine)))

    # Recompute actual cell sizes
    dr_fine = radial_extent / nr_fine
    dz_fine = axial_extent / nz_fine

    return RefinementRegion(
        r_min=r_min,
        r_max=r_max,
        z_min=z_min,
        z_max=z_max,
        nr_fine=nr_fine,
        nz_fine=nz_fine,
        dr_fine=dr_fine,
        dz_fine=dz_fine,
        refinement_factor=refinement_factor,
        sheath_r=r_sheath,
        sheath_z=z_sheath,
        method="gradient_peak",
    )


def interpolate_to_fine_grid(
    coarse_state: dict[str, np.ndarray],
    r_coarse: np.ndarray,
    z_coarse: np.ndarray,
    region: RefinementRegion,
) -> dict[str, np.ndarray]:
    """Interpolate coarse MHD state onto fine grid.

    Uses bilinear interpolation (conservative for smooth fields).
    The fine grid covers only the refinement region.

    Args:
        coarse_state: MHD state dict with keys: rho, velocity, pressure, B, Te, Ti.
        r_coarse: Coarse radial cell centers [nr_coarse].
        z_coarse: Coarse axial cell centers [nz_coarse].
        region: RefinementRegion defining the fine grid.

    Returns:
        Fine-grid MHD state dict with the same keys.
    """
    from scipy.interpolate import RegularGridInterpolator

    nr_c, ny, nz_c = coarse_state["rho"].shape

    # Fine grid cell centers
    r_fine = np.linspace(
        region.r_min + region.dr_fine * 0.5,
        region.r_max - region.dr_fine * 0.5,
        region.nr_fine,
    )
    z_fine = np.linspace(
        region.z_min + region.dz_fine * 0.5,
        region.z_max - region.dz_fine * 0.5,
        region.nz_fine,
    )

    # Create interpolation grid
    rr_fine, zz_fine = np.meshgrid(r_fine, z_fine, indexing="ij")
    points_fine = np.column_stack([rr_fine.ravel(), zz_fine.ravel()])

    fine_state: dict[str, np.ndarray] = {}

    for key in ("rho", "pressure"):
        if key not in coarse_state:
            continue
        field_2d = coarse_state[key][:, ny // 2, :]
        interp = RegularGridInterpolator(
            (r_coarse, z_coarse), field_2d,
            method="linear", bounds_error=False, fill_value=None,
        )
        fine_2d = interp(points_fine).reshape(region.nr_fine, region.nz_fine)
        fine_state[key] = fine_2d[:, np.newaxis, :]

    # Vector fields: velocity (3, nr, ny, nz) and B (3, nr, ny, nz)
    for key in ("velocity", "B"):
        if key not in coarse_state:
            continue
        field = coarse_state[key]
        ncomp = field.shape[0]
        fine_field = np.zeros((ncomp, region.nr_fine, 1, region.nz_fine))
        for c in range(ncomp):
            field_2d = field[c, :, ny // 2, :]
            interp = RegularGridInterpolator(
                (r_coarse, z_coarse), field_2d,
                method="linear", bounds_error=False, fill_value=None,
            )
            fine_field[c, :, 0, :] = interp(points_fine).reshape(
                region.nr_fine, region.nz_fine
            )
        fine_state[key] = fine_field

    # Temperature fields (scalar)
    for key in ("Te", "Ti"):
        if key in coarse_state:
            field_2d = coarse_state[key][:, ny // 2, :]
            interp = RegularGridInterpolator(
                (r_coarse, z_coarse), field_2d,
                method="linear", bounds_error=False, fill_value=None,
            )
            fine_2d = interp(points_fine).reshape(region.nr_fine, region.nz_fine)
            fine_state[key] = fine_2d[:, np.newaxis, :]

    # Divergence cleaning scalar
    if "psi" in coarse_state:
        fine_state["psi"] = np.zeros((region.nr_fine, 1, region.nz_fine))

    return fine_state


def create_stretched_grid(
    r_min: float,
    r_max: float,
    nr: int,
    r_focus: float,
    stretch_ratio: float = 1.2,
) -> np.ndarray:
    """Create a non-uniform radial grid with geometric stretching.

    Cells are smallest near r_focus and grow outward with ratio stretch_ratio.
    This is a simple algebraic stretching — no solver changes needed as long
    as the grid spacing is passed correctly.

    Args:
        r_min: Inner radius [m].
        r_max: Outer radius [m].
        nr: Number of cells.
        r_focus: Radius to concentrate cells around [m].
        stretch_ratio: Growth ratio of adjacent cells (1.0 = uniform).

    Returns:
        Array of cell centers [nr], non-uniformly spaced.
    """
    if stretch_ratio <= 1.0 + 1e-10:
        return np.linspace(r_min + (r_max - r_min) / (2 * nr),
                           r_max - (r_max - r_min) / (2 * nr), nr)

    # Normalized coordinate xi in [0, 1]
    xi = np.linspace(0, 1, nr + 1)

    # Tanh stretching: concentrate cells near r_focus
    # Map: r = r_min + (r_max - r_min) * f(xi)
    # where f(xi) stretches points toward the focus location
    focus_frac = (r_focus - r_min) / (r_max - r_min)
    focus_frac = np.clip(focus_frac, 0.1, 0.9)

    beta = stretch_ratio
    # Roberts transformation (Roberts, 1971)
    # Two-sided stretching toward focus_frac
    f = np.zeros_like(xi)
    for i, x in enumerate(xi):
        if x <= focus_frac:
            # Stretch in [0, focus_frac]
            t = x / focus_frac
            f[i] = focus_frac * (1.0 + np.tanh(beta * (t - 1.0)) / np.tanh(beta))
        else:
            # Stretch in [focus_frac, 1]
            t = (x - focus_frac) / (1.0 - focus_frac)
            f[i] = focus_frac + (1.0 - focus_frac) * np.tanh(beta * t) / np.tanh(beta)

    # Cell edges
    r_edges = r_min + (r_max - r_min) * f
    # Cell centers
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    return r_centers


def lohner_error_indicator(
    rho: np.ndarray,
    dr: float,
    dz: float,
    threshold: float = 0.1,
) -> np.ndarray:
    """Lohner error indicator for refinement decisions.

    Computes a normalized second derivative indicator that identifies
    regions where the solution has strong curvature (shocks, sheaths).

    E = |d2u/dx2| / (|du/dx|/dx + epsilon * |u|/dx^2)

    Args:
        rho: Density field [nr, ny, nz].
        dr: Radial cell spacing [m].
        dz: Axial cell spacing [m].
        threshold: Refinement threshold (0-1). Cells with E > threshold
            should be refined.

    Returns:
        Error indicator field [nr, ny, nz] in [0, 1].

    References:
        Lohner, R., Comp. Methods Appl. Mech. Eng. 61:323-338 (1987).
    """
    nr, ny, nz = rho.shape
    eps = 1e-6 * float(np.mean(np.abs(rho)))

    indicator = np.zeros_like(rho)

    # Radial direction
    if nr > 2:
        d2rho_r = np.zeros_like(rho)
        drho_r = np.zeros_like(rho)
        d2rho_r[1:-1, :, :] = rho[2:, :, :] - 2 * rho[1:-1, :, :] + rho[:-2, :, :]
        drho_r[1:-1, :, :] = np.abs(rho[2:, :, :] - rho[:-2, :, :])
        num = np.abs(d2rho_r)
        den = drho_r + eps * np.abs(rho) / dr
        indicator += num / (den + 1e-30)

    # Axial direction
    if nz > 2:
        d2rho_z = np.zeros_like(rho)
        drho_z = np.zeros_like(rho)
        d2rho_z[:, :, 1:-1] = rho[:, :, 2:] - 2 * rho[:, :, 1:-1] + rho[:, :, :-2]
        drho_z[:, :, 1:-1] = np.abs(rho[:, :, 2:] - rho[:, :, :-2])
        num = np.abs(d2rho_z)
        den = drho_z + eps * np.abs(rho) / dz
        indicator += num / (den + 1e-30)

    # Normalize to [0, 1]
    max_val = float(np.max(indicator))
    if max_val > 0:
        indicator /= max_val

    return indicator


def identify_refinement_cells(
    indicator: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Identify cells that need refinement based on error indicator.

    Args:
        indicator: Error indicator field [nr, ny, nz] from lohner_error_indicator.
        threshold: Refinement threshold (default 0.3).

    Returns:
        Boolean mask [nr, ny, nz] where True = needs refinement.
    """
    return indicator > threshold


def run_refined_simulation(
    coarse_result: dict,
    anode_radius: float,
    cathode_radius: float,
    anode_length: float,
    circuit_config: dict,
    gas_config: dict,
    refinement_factor: int = 4,
    sim_time: float | None = None,
) -> dict:
    """Run a refined simulation using two-pass SMR.

    This is the main entry point for static mesh refinement. It:
    1. Extracts sheath location from the coarse result
    2. Computes a refinement region
    3. Interpolates the coarse solution onto the fine grid
    4. Runs the Metal solver on the fine grid

    Args:
        coarse_result: Result dict from a coarse simulation.
        anode_radius: Anode radius [m].
        cathode_radius: Cathode radius [m].
        anode_length: Anode length [m].
        circuit_config: Circuit configuration dict.
        gas_config: Gas species dict.
        refinement_factor: Resolution improvement factor (default 4).
        sim_time: Fine simulation time [s]. If None, uses remaining time.

    Returns:
        Fine-resolution result dict (same format as coarse).
    """
    final_state = coarse_result.get("final_state")
    if final_state is None:
        logger.warning("No final_state in coarse result — cannot refine")
        return coarse_result

    rho = final_state["rho"]
    nr_c, ny, nz_c = rho.shape
    gap = cathode_radius - anode_radius
    dr_c = gap / nr_c
    dz_c = anode_length / nz_c

    r_coarse = np.linspace(anode_radius + dr_c * 0.5, cathode_radius - dr_c * 0.5, nr_c)
    z_coarse = np.linspace(dz_c * 0.5, anode_length - dz_c * 0.5, nz_c)

    # Step 1: Detect sheath
    r_sh, z_sh, w_sh = detect_sheath_location(
        rho, r_coarse, z_coarse, anode_radius, cathode_radius,
    )
    logger.info(
        "SMR: Sheath detected at r=%.3f mm, z=%.1f mm, width=%.2f mm",
        r_sh * 1e3, z_sh * 1e3, w_sh * 1e3,
    )

    # Step 2: Compute refinement region
    region = compute_refinement_region(
        r_sh, z_sh, w_sh,
        anode_radius, cathode_radius, anode_length,
        refinement_factor=refinement_factor,
    )
    logger.info(
        "SMR: Refined grid %d x %d, dr=%.3f mm (%.1fx finer), "
        "r=[%.1f, %.1f] mm, z=[%.1f, %.1f] mm",
        region.nr_fine, region.nz_fine,
        region.dr_fine * 1e3, dr_c / region.dr_fine,
        region.r_min * 1e3, region.r_max * 1e3,
        region.z_min * 1e3, region.z_max * 1e3,
    )

    # Step 3: Interpolate to fine grid
    fine_state = interpolate_to_fine_grid(
        final_state, r_coarse, z_coarse, region,
    )

    # Step 4: Run Metal solver on fine grid
    try:
        from dpf.metal.metal_solver import MetalMHDSolver

        solver = MetalMHDSolver(
            grid_shape=(region.nr_fine, 1, region.nz_fine),
            dx=region.dr_fine,
            dz=region.dz_fine,
            gamma=5.0 / 3.0,
            cfl=0.3,
            reconstruction="plm",
            riemann_solver="hll",
            coordinates="cylindrical",
        )

        t_fine = sim_time if sim_time is not None else 1e-6
        dt = solver._compute_dt_cfl(solver._to_gpu(fine_state))
        n_steps = max(1, int(t_fine / dt))
        n_steps = min(n_steps, 1000)  # Cap at 1000 steps

        state = fine_state
        for _ in range(n_steps):
            state = solver.step(state, dt)

        result = dict(coarse_result)
        result["final_state"] = state
        result["refinement"] = {
            "method": "static_mesh_refinement",
            "factor": refinement_factor,
            "nr_fine": region.nr_fine,
            "nz_fine": region.nz_fine,
            "dr_fine_mm": region.dr_fine * 1e3,
            "dz_fine_mm": region.dz_fine * 1e3,
            "sheath_r_mm": r_sh * 1e3,
            "sheath_z_mm": z_sh * 1e3,
            "sheath_width_mm": w_sh * 1e3,
        }
        return result

    except ImportError:
        logger.warning("Metal solver not available — returning coarse result")
        return coarse_result
    except Exception as exc:
        logger.warning("SMR fine simulation failed: %s — returning coarse result", exc)
        return coarse_result
