"""Patch-based adaptive mesh refinement for 2D axisymmetric (r, z) grids.

A Dense Plasma Focus device has extreme resolution requirements: the pinch
column is ~1 mm in radius inside a ~10 cm electrode assembly, demanding a
100:1 resolution ratio. Uniform grids are prohibitively expensive.

This module implements a pure-Python patch-based AMR framework:
- Gradient-based cell tagging on density (|grad rho| / rho > threshold)
- Clustered patch creation at successive refinement levels
- Bilinear prolongation (coarse -> fine) and volume-weighted restriction
  (fine -> coarse) appropriate for cylindrical (r, z) geometry
- Buffer cells around tagged regions to avoid immediate re-regridding

The grid is cell-centered with:
    r[i] = (i + 0.5) * dr  for i = 0, ..., nr-1
    z[j] = (j + 0.5) * dz  for j = 0, ..., nz-1

Reference:
    Berger & Colella, "Local adaptive mesh refinement for shock
    hydrodynamics", JCP 82, 64-84 (1989).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


class AMRConfig(BaseModel):
    """Adaptive mesh refinement parameters.

    Attributes:
        max_levels: Maximum number of refinement levels (0 = base only).
        refinement_ratio: Integer ratio between successive levels.
        regrid_interval: Timesteps between regridding operations.
        gradient_threshold: Relative gradient threshold |grad rho|/rho for tagging.
        buffer_cells: Extra cells around tagged regions to avoid re-tagging.
    """

    max_levels: int = Field(3, ge=1, le=6)
    refinement_ratio: int = Field(2, ge=2, le=4)
    regrid_interval: int = Field(10, ge=1)
    gradient_threshold: float = Field(0.3, gt=0)
    buffer_cells: int = Field(2, ge=1)


# ============================================================
# AMR Patch
# ============================================================


@dataclass
class AMRPatch:
    """A rectangular sub-region at a single refinement level.

    Coordinates are in the level's own index space. Cell-centered positions:
        r[i] = (i_start + i + 0.5) * dx
        z[j] = (j_start + j + 0.5) * dz

    Attributes:
        level: Refinement level (0 = coarsest).
        i_start: Starting radial index in this level's index space.
        j_start: Starting axial index in this level's index space.
        ni: Number of radial cells.
        nj: Number of axial cells.
        dx: Radial grid spacing at this level [m].
        dz: Axial grid spacing at this level [m].
        data: Dictionary of field arrays, each shape (ni, nj).
    """

    level: int
    i_start: int
    j_start: int
    ni: int
    nj: int
    dx: float
    dz: float
    data: dict[str, np.ndarray] = field(default_factory=dict)

    def cell_centers_r(self) -> np.ndarray:
        """Return radial cell-center coordinates [m].

        Returns:
            1D array of shape (ni,) with r positions.
        """
        return (self.i_start + np.arange(self.ni) + 0.5) * self.dx

    def cell_centers_z(self) -> np.ndarray:
        """Return axial cell-center coordinates [m].

        Returns:
            1D array of shape (nj,) with z positions.
        """
        return (self.j_start + np.arange(self.nj) + 0.5) * self.dz


# ============================================================
# Numba-accelerated helper functions
# ============================================================


@njit(cache=True)
def tag_cells_gradient(
    rho: np.ndarray,
    dx: float,
    dz: float,
    threshold: float,
) -> np.ndarray:
    """Tag cells where the relative density gradient exceeds a threshold.

    Computes |grad rho| / rho using centered finite differences and marks
    cells where the ratio exceeds ``threshold``. This identifies the pinch
    column (sharp density gradients) and current sheet regions.

    Args:
        rho: Density field, shape (nr, nz).
        dx: Radial grid spacing [m].
        dz: Axial grid spacing [m].
        threshold: Tag if |grad rho| / rho > threshold.

    Returns:
        Boolean mask of shape (nr, nz), True where refinement is needed.
    """
    nr, nz = rho.shape
    tagged = np.zeros((nr, nz), dtype=np.bool_)
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz

    for i in range(1, nr - 1):
        for j in range(1, nz - 1):
            rho_c = rho[i, j]
            if rho_c < 1e-30:
                continue

            grad_r = 0.5 * (rho[i + 1, j] - rho[i - 1, j]) * inv_dx
            grad_z = 0.5 * (rho[i, j + 1] - rho[i, j - 1]) * inv_dz
            grad_mag = (grad_r * grad_r + grad_z * grad_z) ** 0.5

            if grad_mag / rho_c > threshold:
                tagged[i, j] = True

    return tagged


@njit(cache=True)
def restrict_patch(fine_data: np.ndarray, ratio: int) -> np.ndarray:
    """Volume-weighted restriction from fine to coarse grid.

    For cylindrical geometry the cell volume is proportional to
    r * dr * dz (annular ring). We weight each fine cell by its radial
    coordinate relative to the coarse cell center.

    In practice, for a uniform sub-grid within a single patch, the
    radial variation across a small number of fine cells is modest, so
    we use a simple volume-weighted average:

        V_fine[i] ~ (i_fine + 0.5) * dx_fine

    The coarse cell value is the volume-weighted mean of the fine cells
    it covers.

    Args:
        fine_data: Fine-level field data, shape (ni_fine, nj_fine).
        ratio: Refinement ratio (fine cells per coarse cell in each direction).

    Returns:
        Coarse-level data, shape (ni_fine // ratio, nj_fine // ratio).
    """
    ni_fine, nj_fine = fine_data.shape
    ni_coarse = ni_fine // ratio
    nj_coarse = nj_fine // ratio
    coarse = np.zeros((ni_coarse, nj_coarse))

    for ic in range(ni_coarse):
        for jc in range(nj_coarse):
            total_val = 0.0
            total_weight = 0.0
            for di in range(ratio):
                i_f = ic * ratio + di
                # Weight proportional to radial position (volume ~ r)
                r_weight = i_f + 0.5  # in fine-grid index units
                for dj in range(ratio):
                    j_f = jc * ratio + dj
                    w = max(r_weight, 1e-30)
                    total_val += w * fine_data[i_f, j_f]
                    total_weight += w
            if total_weight > 0.0:
                coarse[ic, jc] = total_val / total_weight

    return coarse


@njit(cache=True)
def prolong_patch(coarse_data: np.ndarray, ratio: int) -> np.ndarray:
    """Bilinear prolongation from coarse to fine grid.

    Interpolates coarse cell-centered values to fine cell centers using
    bilinear interpolation. Fine cells outside the coarse stencil are
    filled with nearest-neighbor extrapolation.

    Args:
        coarse_data: Coarse-level field data, shape (ni_coarse, nj_coarse).
        ratio: Refinement ratio.

    Returns:
        Fine-level data, shape (ni_coarse * ratio, nj_coarse * ratio).
    """
    ni_c, nj_c = coarse_data.shape
    ni_f = ni_c * ratio
    nj_f = nj_c * ratio
    fine = np.zeros((ni_f, nj_f))

    for i_f in range(ni_f):
        for j_f in range(nj_f):
            # Fine cell center in coarse-cell coordinates
            # Fine cell i_f corresponds to coarse fractional index:
            #   x_c = (i_f + 0.5) / ratio - 0.5
            x_c = (i_f + 0.5) / ratio - 0.5
            y_c = (j_f + 0.5) / ratio - 0.5

            # Coarse cell indices for bilinear stencil
            ic0 = int(x_c)
            jc0 = int(y_c)

            # Clamp to valid range
            ic0 = max(0, min(ic0, ni_c - 2))
            jc0 = max(0, min(jc0, nj_c - 2))
            ic1 = ic0 + 1
            jc1 = jc0 + 1

            # Fractional offsets within the coarse cell
            fx = x_c - ic0
            fy = y_c - jc0
            fx = max(0.0, min(fx, 1.0))
            fy = max(0.0, min(fy, 1.0))

            # Bilinear interpolation
            fine[i_f, j_f] = (
                (1.0 - fx) * (1.0 - fy) * coarse_data[ic0, jc0]
                + fx * (1.0 - fy) * coarse_data[ic1, jc0]
                + (1.0 - fx) * fy * coarse_data[ic0, jc1]
                + fx * fy * coarse_data[ic1, jc1]
            )

    return fine


# ============================================================
# Helper: add buffer cells to tagged mask
# ============================================================


def _add_buffer(tagged: np.ndarray, buffer_cells: int) -> np.ndarray:
    """Expand tagged region by buffer_cells in all directions.

    Args:
        tagged: Boolean mask, shape (nr, nz).
        buffer_cells: Number of cells to expand.

    Returns:
        Expanded boolean mask, same shape.
    """
    result = tagged.copy()
    nr, nz = tagged.shape
    for _ in range(buffer_cells):
        expanded = result.copy()
        for i in range(nr):
            for j in range(nz):
                if result[i, j]:
                    # Expand to neighbors
                    if i > 0:
                        expanded[i - 1, j] = True
                    if i < nr - 1:
                        expanded[i + 1, j] = True
                    if j > 0:
                        expanded[i, j - 1] = True
                    if j < nz - 1:
                        expanded[i, j + 1] = True
        result = expanded
    return result


# ============================================================
# Helper: find bounding box of tagged clusters
# ============================================================


def _find_tagged_bbox(
    tagged: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Find the bounding box of contiguous tagged regions.

    Returns a list of (i_start, j_start, ni, nj) tuples, one per
    connected cluster. Uses a simple single-pass bounding box (no
    connected-component analysis) for efficiency.

    For the DPF use case, the pinch column is typically a single
    connected region, so a single bounding box is usually sufficient.

    Args:
        tagged: Boolean mask, shape (nr, nz).

    Returns:
        List of (i_start, j_start, ni, nj) bounding boxes.
    """
    if not np.any(tagged):
        return []

    # Find global bounding box of all tagged cells
    rows = np.any(tagged, axis=1)
    cols = np.any(tagged, axis=0)

    i_min = int(np.argmax(rows))
    i_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    j_min = int(np.argmax(cols))
    j_max = int(len(cols) - 1 - np.argmax(cols[::-1]))

    ni = i_max - i_min + 1
    nj = j_max - j_min + 1

    if ni <= 0 or nj <= 0:
        return []

    return [(i_min, j_min, ni, nj)]


# ============================================================
# AMR Grid
# ============================================================


class AMRGrid:
    """Patch-based AMR hierarchy for 2D axisymmetric (r, z) DPF simulations.

    Manages a hierarchy of refinement levels. Level 0 is the base (coarsest)
    grid covering the full domain. Higher levels contain patches with finer
    resolution, concentrated where density gradients are steepest (the pinch
    column).

    Args:
        nr_base: Number of radial cells at base level.
        nz_base: Number of axial cells at base level.
        r_max: Radial extent of the domain [m].
        z_max: Axial extent of the domain [m].
        config: AMR configuration parameters.
    """

    def __init__(
        self,
        nr_base: int,
        nz_base: int,
        r_max: float,
        z_max: float,
        config: AMRConfig,
    ) -> None:
        self.nr_base = nr_base
        self.nz_base = nz_base
        self.r_max = r_max
        self.z_max = z_max
        self.config = config

        # Base-level grid spacing
        self.dx_base = r_max / nr_base
        self.dz_base = z_max / nz_base

        # patches[level] = list of AMRPatch at that level
        self.patches: list[list[AMRPatch]] = []

        # Create base level (level 0) as a single patch covering the full domain
        base_patch = AMRPatch(
            level=0,
            i_start=0,
            j_start=0,
            ni=nr_base,
            nj=nz_base,
            dx=self.dx_base,
            dz=self.dz_base,
            data={},
        )
        self.patches.append([base_patch])

        # Initialize empty higher levels up to max_levels
        for _ in range(config.max_levels):
            self.patches.append([])

        logger.info(
            "AMRGrid initialized: base %dx%d, domain (%.3e x %.3e) m, "
            "max_levels=%d, ratio=%d",
            nr_base,
            nz_base,
            r_max,
            z_max,
            config.max_levels,
            config.refinement_ratio,
        )

    # ----------------------------------------------------------
    # Cell tagging
    # ----------------------------------------------------------

    def tag_cells(self, field_name: str = "rho") -> np.ndarray:
        """Tag base-level cells that need refinement.

        Uses gradient-based tagging on the specified field and adds a
        buffer region around tagged cells.

        Args:
            field_name: Field to compute gradients on (default: 'rho').

        Returns:
            Boolean mask of shape (nr_base, nz_base), True where
            refinement is needed.
        """
        base_patch = self.patches[0][0]
        if field_name not in base_patch.data:
            logger.warning(
                "Field '%s' not found in base patch; returning empty tags",
                field_name,
            )
            return np.zeros((self.nr_base, self.nz_base), dtype=bool)

        rho = base_patch.data[field_name]

        tagged = tag_cells_gradient(
            rho,
            self.dx_base,
            self.dz_base,
            self.config.gradient_threshold,
        )

        # Add buffer cells around tagged regions
        tagged = _add_buffer(tagged, self.config.buffer_cells)

        n_tagged = int(np.sum(tagged))
        logger.debug(
            "Tagged %d / %d base cells (%.1f%%)",
            n_tagged,
            self.nr_base * self.nz_base,
            100.0 * n_tagged / (self.nr_base * self.nz_base),
        )

        return tagged

    # ----------------------------------------------------------
    # Regridding
    # ----------------------------------------------------------

    def regrid(self, tagged: np.ndarray) -> None:
        """Create or destroy patches at higher levels based on tagged cells.

        For each cluster of tagged cells at level L, creates a new patch
        at level L+1 with refined grid spacing. New patch data is filled
        by prolongation from the coarser level.

        Args:
            tagged: Boolean mask at base level, shape (nr_base, nz_base).
        """
        ratio = self.config.refinement_ratio

        # Work level by level, starting from the base
        current_tagged = tagged.copy()

        for lev in range(self.config.max_levels):
            bboxes = _find_tagged_bbox(current_tagged)

            if not bboxes:
                # No tagged cells at this level; clear all higher levels
                for higher in range(lev + 1, len(self.patches)):
                    self.patches[higher] = []
                break

            # Grid spacing at the new (finer) level
            coarse_dx = self.dx_base / (ratio**lev)
            coarse_dz = self.dz_base / (ratio**lev)
            fine_dx = coarse_dx / ratio
            fine_dz = coarse_dz / ratio

            new_patches: list[AMRPatch] = []
            for i_start, j_start, ni_c, nj_c in bboxes:
                ni_fine = ni_c * ratio
                nj_fine = nj_c * ratio

                patch = AMRPatch(
                    level=lev + 1,
                    i_start=i_start * ratio,
                    j_start=j_start * ratio,
                    ni=ni_fine,
                    nj=nj_fine,
                    dx=fine_dx,
                    dz=fine_dz,
                    data={},
                )

                # Prolongate data from coarser level
                coarse_patches = self.patches[lev]
                for cp in coarse_patches:
                    for name, coarse_field in cp.data.items():
                        # Extract the sub-region of the coarse field that
                        # corresponds to this fine patch
                        ci_start = i_start - cp.i_start
                        cj_start = j_start - cp.j_start
                        ci_end = ci_start + ni_c
                        cj_end = cj_start + nj_c

                        # Clip to coarse patch bounds
                        ci_start_clip = max(0, ci_start)
                        cj_start_clip = max(0, cj_start)
                        ci_end_clip = min(cp.ni, ci_end)
                        cj_end_clip = min(cp.nj, cj_end)

                        if ci_start_clip >= ci_end_clip or cj_start_clip >= cj_end_clip:
                            continue

                        coarse_sub = coarse_field[
                            ci_start_clip:ci_end_clip,
                            cj_start_clip:cj_end_clip,
                        ]

                        fine_sub = prolong_patch(coarse_sub, ratio)

                        # Place into the full fine patch data array
                        if name not in patch.data:
                            patch.data[name] = np.zeros((ni_fine, nj_fine))

                        # Offset in fine patch coordinates
                        fi_start = (ci_start_clip - ci_start) * ratio
                        fj_start = (cj_start_clip - cj_start) * ratio
                        fi_end = fi_start + fine_sub.shape[0]
                        fj_end = fj_start + fine_sub.shape[1]

                        patch.data[name][fi_start:fi_end, fj_start:fj_end] = fine_sub

                new_patches.append(patch)

            self.patches[lev + 1] = new_patches

            # For next level, tag the fine patches using the same gradient
            # criterion at finer resolution
            if lev + 1 < self.config.max_levels and new_patches:
                # Build a combined tagged mask at the fine level
                p = new_patches[0]
                if "rho" in p.data:
                    fine_tagged = tag_cells_gradient(
                        p.data["rho"],
                        fine_dx,
                        fine_dz,
                        self.config.gradient_threshold,
                    )
                    fine_tagged = _add_buffer(fine_tagged, self.config.buffer_cells)
                    current_tagged = fine_tagged
                else:
                    break
            else:
                break

        n_patches = sum(len(lev_patches) for lev_patches in self.patches)
        logger.info(
            "Regridded: %d total patches across %d levels, %d total cells",
            n_patches,
            sum(1 for lp in self.patches if lp),
            self.total_cells(),
        )

    # ----------------------------------------------------------
    # Restriction (fine -> coarse)
    # ----------------------------------------------------------

    def restrict(self) -> None:
        """Average fine-level data back to coarser levels.

        Walks from the finest level down to level 1, averaging each fine
        patch's data into the corresponding coarse-level cells using
        volume-weighted restriction (appropriate for cylindrical geometry).
        """
        ratio = self.config.refinement_ratio

        for lev in range(len(self.patches) - 1, 0, -1):
            for fine_patch in self.patches[lev]:
                if not fine_patch.data:
                    continue

                # Find the coarse patch(es) that overlap
                for coarse_patch in self.patches[lev - 1]:
                    for name, fine_field in fine_patch.data.items():
                        if name not in coarse_patch.data:
                            continue

                        restricted = restrict_patch(fine_field, ratio)

                        # Map restricted data back to coarse patch coordinates
                        ci_start = fine_patch.i_start // ratio - coarse_patch.i_start
                        cj_start = fine_patch.j_start // ratio - coarse_patch.j_start

                        ni_r, nj_r = restricted.shape
                        ci_end = ci_start + ni_r
                        cj_end = cj_start + nj_r

                        # Clip to coarse patch bounds
                        ci_start_clip = max(0, ci_start)
                        cj_start_clip = max(0, cj_start)
                        ci_end_clip = min(coarse_patch.ni, ci_end)
                        cj_end_clip = min(coarse_patch.nj, cj_end)

                        if ci_start_clip >= ci_end_clip or cj_start_clip >= cj_end_clip:
                            continue

                        ri_start = ci_start_clip - ci_start
                        rj_start = cj_start_clip - cj_start
                        ri_end = ri_start + (ci_end_clip - ci_start_clip)
                        rj_end = rj_start + (cj_end_clip - cj_start_clip)

                        coarse_patch.data[name][
                            ci_start_clip:ci_end_clip,
                            cj_start_clip:cj_end_clip,
                        ] = restricted[ri_start:ri_end, rj_start:rj_end]

    # ----------------------------------------------------------
    # Prolongation (coarse -> fine)
    # ----------------------------------------------------------

    def prolong(self, level: int) -> None:
        """Interpolate coarse data to fill fine patches at the given level.

        Uses bilinear interpolation from level-1 to populate ghost/boundary
        regions of patches at ``level``.

        Args:
            level: Target level to fill (must be >= 1).
        """
        if level < 1 or level >= len(self.patches):
            return

        ratio = self.config.refinement_ratio

        for fine_patch in self.patches[level]:
            for coarse_patch in self.patches[level - 1]:
                for name, coarse_field in coarse_patch.data.items():
                    # Determine the coarse sub-region that covers this fine patch
                    ci_start = fine_patch.i_start // ratio - coarse_patch.i_start
                    cj_start = fine_patch.j_start // ratio - coarse_patch.j_start
                    ni_c = fine_patch.ni // ratio
                    nj_c = fine_patch.nj // ratio
                    ci_end = ci_start + ni_c
                    cj_end = cj_start + nj_c

                    # Clip to coarse patch bounds
                    ci_start_clip = max(0, ci_start)
                    cj_start_clip = max(0, cj_start)
                    ci_end_clip = min(coarse_patch.ni, ci_end)
                    cj_end_clip = min(coarse_patch.nj, cj_end)

                    if ci_start_clip >= ci_end_clip or cj_start_clip >= cj_end_clip:
                        continue

                    coarse_sub = coarse_field[
                        ci_start_clip:ci_end_clip,
                        cj_start_clip:cj_end_clip,
                    ]
                    fine_sub = prolong_patch(coarse_sub, ratio)

                    if name not in fine_patch.data:
                        fine_patch.data[name] = np.zeros(
                            (fine_patch.ni, fine_patch.nj),
                        )

                    fi_start = (ci_start_clip - ci_start) * ratio
                    fj_start = (cj_start_clip - cj_start) * ratio
                    fi_end = fi_start + fine_sub.shape[0]
                    fj_end = fj_start + fine_sub.shape[1]

                    # Clip to fine patch bounds
                    fi_end = min(fi_end, fine_patch.ni)
                    fj_end = min(fj_end, fine_patch.nj)
                    s0 = fi_end - fi_start
                    s1 = fj_end - fj_start

                    fine_patch.data[name][fi_start:fi_end, fj_start:fj_end] = (
                        fine_sub[:s0, :s1]
                    )

    # ----------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------

    def total_cells(self) -> int:
        """Return total number of cells across all levels and patches.

        Returns:
            Integer cell count.
        """
        total = 0
        for lev_patches in self.patches:
            for p in lev_patches:
                total += p.ni * p.nj
        return total

    def get_level_data(self, level: int, field_name: str) -> np.ndarray:
        """Assemble a complete field array at the given refinement level.

        For level 0, returns the base patch data directly. For higher levels,
        assembles data from all patches at that level into a single array
        covering the full level domain (unrefined regions are filled with
        zeros).

        Args:
            level: Refinement level.
            field_name: Name of the field to retrieve.

        Returns:
            2D array at the level's resolution. For level 0, shape is
            (nr_base, nz_base). For higher levels, shape is
            (nr_base * ratio^level, nz_base * ratio^level).

        Raises:
            IndexError: If level is out of range.
        """
        if level < 0 or level >= len(self.patches):
            msg = f"Level {level} out of range [0, {len(self.patches) - 1}]"
            raise IndexError(msg)

        ratio = self.config.refinement_ratio
        nr_lev = self.nr_base * (ratio**level)
        nz_lev = self.nz_base * (ratio**level)

        result = np.zeros((nr_lev, nz_lev))

        for p in self.patches[level]:
            if field_name not in p.data:
                continue

            i_end = p.i_start + p.ni
            j_end = p.j_start + p.nj

            # Clip to domain bounds
            i_end = min(i_end, nr_lev)
            j_end = min(j_end, nz_lev)
            ni_actual = i_end - p.i_start
            nj_actual = j_end - p.j_start

            result[p.i_start:i_end, p.j_start:j_end] = (
                p.data[field_name][:ni_actual, :nj_actual]
            )

        return result
