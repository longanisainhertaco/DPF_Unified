"""Tests for static mesh refinement (Phase 3A).

Tests cover:
- Sheath detection from density gradients
- Refinement region computation
- Grid interpolation
- Stretched grid generation
- Lohner error indicator
"""

from __future__ import annotations

import numpy as np

from dpf.experimental.static_refinement import (
    RefinementRegion,
    compute_refinement_region,
    create_stretched_grid,
    detect_sheath_location,
    identify_refinement_cells,
    interpolate_to_fine_grid,
    lohner_error_indicator,
)

# --- Test fixtures ---

def _make_sheath_density(nr: int = 32, nz: int = 64, sheath_r_idx: int = 16) -> tuple:
    """Create a synthetic density field with a sharp sheath."""
    a = 0.01   # anode radius 10 mm
    b = 0.04   # cathode radius 40 mm
    L = 0.16   # anode length 160 mm
    dr = (b - a) / nr
    dz = L / nz
    r_cells = np.linspace(a + dr * 0.5, b - dr * 0.5, nr)
    z_cells = np.linspace(dz * 0.5, L - dz * 0.5, nz)

    rho = np.ones((nr, 1, nz)) * 1e-4  # Background density
    # Sharp sheath at sheath_r_idx: density jump
    rho[sheath_r_idx:sheath_r_idx + 3, :, nz // 3:2 * nz // 3] = 1e-2
    return rho, r_cells, z_cells, a, b, L, dr, dz


# --- Sheath detection tests ---


class TestSheathDetection:
    def test_detects_sheath_position(self):
        rho, r_cells, z_cells, a, b, *_ = _make_sheath_density()
        r_sh, z_sh, w_sh = detect_sheath_location(rho, r_cells, z_cells, a, b)
        # Sheath should be near index 16 → r ~ 0.025 m
        expected_r = r_cells[16]
        assert abs(r_sh - expected_r) < 0.005  # Within 5 mm

    def test_sheath_width_positive(self):
        rho, r_cells, z_cells, a, b, *_ = _make_sheath_density()
        _, _, w_sh = detect_sheath_location(rho, r_cells, z_cells, a, b)
        assert w_sh > 0

    def test_detects_at_different_positions(self):
        """Sheath at different radial positions should be detected."""
        for idx in [8, 16, 24]:
            rho, r_cells, z_cells, a, b, *_ = _make_sheath_density(sheath_r_idx=idx)
            r_sh, _, _ = detect_sheath_location(rho, r_cells, z_cells, a, b)
            expected_r = r_cells[min(idx, len(r_cells) - 1)]
            assert abs(r_sh - expected_r) < 0.005

    def test_uniform_density_returns_valid(self):
        """Uniform density should still return a valid position."""
        rho = np.ones((32, 1, 64)) * 1e-4
        r_cells = np.linspace(0.01, 0.04, 32)
        z_cells = np.linspace(0.001, 0.16, 64)
        r_sh, z_sh, w_sh = detect_sheath_location(rho, r_cells, z_cells, 0.01, 0.04)
        assert 0.01 <= r_sh <= 0.04
        assert w_sh > 0


# --- Refinement region tests ---


class TestRefinementRegion:
    def test_region_within_bounds(self):
        region = compute_refinement_region(
            r_sheath=0.025, z_sheath=0.08, sheath_width=0.002,
            anode_radius=0.01, cathode_radius=0.04, anode_length=0.16,
        )
        assert region.r_min >= 0.01
        assert region.r_max <= 0.04
        assert region.z_min >= 0.0
        assert region.z_max <= 0.16

    def test_fine_grid_smaller_than_coarse(self):
        region = compute_refinement_region(
            r_sheath=0.025, z_sheath=0.08, sheath_width=0.002,
            anode_radius=0.01, cathode_radius=0.04, anode_length=0.16,
            refinement_factor=4,
        )
        coarse_dr = 0.03 / 16  # gap / nr_coarse
        assert region.dr_fine < coarse_dr

    def test_refinement_factor_stored(self):
        region = compute_refinement_region(
            r_sheath=0.025, z_sheath=0.08, sheath_width=0.002,
            anode_radius=0.01, cathode_radius=0.04, anode_length=0.16,
            refinement_factor=8,
        )
        assert region.refinement_factor == 8

    def test_minimum_cells(self):
        region = compute_refinement_region(
            r_sheath=0.025, z_sheath=0.08, sheath_width=0.0001,
            anode_radius=0.01, cathode_radius=0.04, anode_length=0.16,
        )
        assert region.nr_fine >= 16
        assert region.nz_fine >= 16


# --- Interpolation tests ---


class TestInterpolation:
    def test_interpolation_preserves_uniform(self):
        """Uniform field should remain uniform after interpolation."""
        nr_c, nz_c = 16, 32
        a, b, L = 0.01, 0.04, 0.16
        dr_c = (b - a) / nr_c
        dz_c = L / nz_c
        r_coarse = np.linspace(a + dr_c * 0.5, b - dr_c * 0.5, nr_c)
        z_coarse = np.linspace(dz_c * 0.5, L - dz_c * 0.5, nz_c)

        coarse_state = {
            "rho": np.ones((nr_c, 1, nz_c)) * 1e-4,
            "pressure": np.ones((nr_c, 1, nz_c)) * 100.0,
            "velocity": np.zeros((3, nr_c, 1, nz_c)),
            "B": np.ones((3, nr_c, 1, nz_c)) * 0.1,
        }

        region = RefinementRegion(
            r_min=0.015, r_max=0.035, z_min=0.04, z_max=0.12,
            nr_fine=32, nz_fine=64,
            dr_fine=(0.035 - 0.015) / 32, dz_fine=(0.12 - 0.04) / 64,
            refinement_factor=4, sheath_r=0.025, sheath_z=0.08,
            method="test",
        )

        fine_state = interpolate_to_fine_grid(coarse_state, r_coarse, z_coarse, region)

        assert fine_state["rho"].shape == (32, 1, 64)
        assert np.allclose(fine_state["rho"], 1e-4, rtol=0.01)
        assert np.allclose(fine_state["pressure"], 100.0, rtol=0.01)

    def test_interpolation_output_shape(self):
        nr_c, nz_c = 16, 32
        a, b, L = 0.01, 0.04, 0.16
        dr_c = (b - a) / nr_c
        dz_c = L / nz_c
        r_coarse = np.linspace(a + dr_c * 0.5, b - dr_c * 0.5, nr_c)
        z_coarse = np.linspace(dz_c * 0.5, L - dz_c * 0.5, nz_c)

        coarse_state = {
            "rho": np.ones((nr_c, 1, nz_c)) * 1e-4,
            "pressure": np.ones((nr_c, 1, nz_c)) * 100.0,
            "velocity": np.zeros((3, nr_c, 1, nz_c)),
            "B": np.zeros((3, nr_c, 1, nz_c)),
            "Te": np.ones((nr_c, 1, nz_c)) * 1000.0,
        }

        region = RefinementRegion(
            r_min=0.015, r_max=0.035, z_min=0.04, z_max=0.12,
            nr_fine=24, nz_fine=48,
            dr_fine=0.02 / 24, dz_fine=0.08 / 48,
            refinement_factor=4, sheath_r=0.025, sheath_z=0.08,
            method="test",
        )

        fine_state = interpolate_to_fine_grid(coarse_state, r_coarse, z_coarse, region)

        assert fine_state["rho"].shape == (24, 1, 48)
        assert fine_state["velocity"].shape == (3, 24, 1, 48)
        assert fine_state["B"].shape == (3, 24, 1, 48)
        assert fine_state["Te"].shape == (24, 1, 48)


# --- Stretched grid tests ---


class TestStretchedGrid:
    def test_uniform_when_ratio_one(self):
        r = create_stretched_grid(0.01, 0.04, 32, r_focus=0.025, stretch_ratio=1.0)
        assert len(r) == 32
        dr = np.diff(r)
        assert np.allclose(dr, dr[0], rtol=0.01)

    def test_non_uniform_spacing(self):
        r = create_stretched_grid(0.01, 0.04, 32, r_focus=0.025, stretch_ratio=2.0)
        assert len(r) == 32
        dr = np.diff(r)
        # Non-uniform grid: min and max spacing should differ
        assert np.max(dr) / np.min(dr) > 1.5  # At least 1.5x variation

    def test_covers_full_range(self):
        r = create_stretched_grid(0.01, 0.04, 32, r_focus=0.025, stretch_ratio=1.5)
        assert r[0] > 0.01
        assert r[-1] < 0.04
        assert r[0] < 0.015  # Near inner boundary
        assert r[-1] > 0.035  # Near outer boundary


# --- Lohner error indicator tests ---


class TestLohnerIndicator:
    def test_smooth_field_low_indicator(self):
        """Smooth density should have low error indicator."""
        nr, nz = 32, 64
        r = np.linspace(0.01, 0.04, nr)
        z = np.linspace(0, 0.16, nz)
        rr, zz = np.meshgrid(r, z, indexing="ij")
        rho = np.sin(np.pi * rr / 0.03)[:, np.newaxis, :]  # Smooth
        indicator = lohner_error_indicator(rho, dr=0.001, dz=0.0025)
        assert indicator.shape == rho.shape
        # Smooth field: most indicator values should be moderate
        assert np.median(indicator) < 0.5

    def test_sharp_jump_high_indicator(self):
        """Sharp density jump should produce high error indicator."""
        rho = np.ones((32, 1, 64)) * 1e-4
        rho[15:17, :, :] = 1e-2  # Sharp jump
        indicator = lohner_error_indicator(rho, dr=0.001, dz=0.0025)
        # Near the jump, indicator should be high
        assert np.max(indicator[14:18, :, :]) > 0.5

    def test_indicator_bounded(self):
        """Indicator should be in [0, 1]."""
        rho = np.random.rand(32, 1, 64) * 1e-3
        indicator = lohner_error_indicator(rho, dr=0.001, dz=0.0025)
        assert np.all(indicator >= 0)
        assert np.all(indicator <= 1.0 + 1e-10)


# --- Refinement cell identification ---


class TestRefinementCells:
    def test_threshold_filters(self):
        indicator = np.array([[[0.1, 0.5, 0.9, 0.3]]])
        mask = identify_refinement_cells(indicator, threshold=0.4)
        assert mask[0, 0, 0] is np.False_
        assert mask[0, 0, 1] is np.True_
        assert mask[0, 0, 2] is np.True_
        assert mask[0, 0, 3] is np.False_

    def test_all_below_threshold(self):
        indicator = np.ones((8, 1, 16)) * 0.1
        mask = identify_refinement_cells(indicator, threshold=0.5)
        assert not np.any(mask)
