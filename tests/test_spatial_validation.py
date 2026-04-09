"""Tests for spatial validation comparison module.

Validates the pipeline: MHD state -> ne(r) -> Abel transform -> N_L(y) -> NRMSE.
"""

import numpy as np
import pytest

from dpf.validation.spatial_comparison import (
    SpatialComparisonResult,
    compare_density_profile,
    extract_density_profile,
    spatial_nrmse_multi,
)


class TestExtractDensityProfile:
    """Test density profile extraction from MHD state."""

    def test_basic_extraction(self):
        nr, nz = 32, 64
        rho = np.ones((nr, nz)) * 1e-3  # kg/m^3
        state = {"rho": rho, "dx": 0.005}
        r, ne = extract_density_profile(state, z_index=32)
        assert r.shape == (nr,)
        assert ne.shape == (nr,)
        assert np.all(ne > 0)

    def test_3d_state_squeezed(self):
        """3D state (nr, 1, nz) should be handled correctly."""
        nr, nz = 16, 32
        rho = np.ones((nr, 1, nz)) * 5e-4
        state = {"rho": rho, "dx": 0.01}
        r, ne = extract_density_profile(state, z_index=16)
        assert r.shape == (nr,)

    def test_z_index_default_midplane(self):
        nr, nz = 16, 32
        rho = np.random.rand(nr, nz) * 1e-3 + 1e-6
        state = {"rho": rho, "dx": 0.01}
        r, ne = extract_density_profile(state)
        # Default z_index should be nz//2 = 16
        expected_ne = rho[:, 16] / 3.34358377e-27
        np.testing.assert_allclose(ne, expected_ne, rtol=1e-10)


class TestCompareDensityProfile:
    """Test the full comparison pipeline."""

    def _make_gaussian_state(self, nr: int = 64, nz: int = 128, dx: float = 0.005):
        """Create a Gaussian density profile for testing."""
        r = (np.arange(nr) + 0.5) * dx
        sigma = 0.02  # 2 cm width
        rho_r = 1e-3 * np.exp(-r**2 / (2 * sigma**2))
        rho = np.outer(rho_r, np.ones(nz))
        return {"rho": rho, "dx": dx}, r

    def test_perfect_match_gives_low_nrmse(self):
        """When sim matches experiment exactly, NRMSE should be near zero."""
        state, r_sim = self._make_gaussian_state()
        # Extract and Abel-transform to create "experimental" data
        from dpf.diagnostics.interferometry import abel_transform

        _, ne = extract_density_profile(state, z_index=64, ion_mass=3.34358377e-27)
        NL_exp = abel_transform(ne, r_sim)

        result = compare_density_profile(
            state, r_exp=r_sim, NL_exp=NL_exp, z_index=64, dx=0.005,
        )
        assert result.nrmse < 0.01, f"Self-comparison NRMSE should be ~0, got {result.nrmse}"
        assert abs(result.peak_ratio - 1.0) < 0.01

    def test_shifted_profile_gives_nonzero_nrmse(self):
        """A shifted profile should produce measurable NRMSE."""
        state, r_sim = self._make_gaussian_state()
        from dpf.diagnostics.interferometry import abel_transform

        _, ne = extract_density_profile(state, z_index=64)
        NL_exp = abel_transform(ne, r_sim)
        # Shift experimental profile by 20%
        NL_exp_shifted = NL_exp * 1.2

        result = compare_density_profile(
            state, r_exp=r_sim, NL_exp=NL_exp_shifted, z_index=64, dx=0.005,
        )
        assert result.nrmse > 0.01, f"20% shift should give NRMSE > 0.01, got {result.nrmse}"

    def test_result_has_all_fields(self):
        state, r_sim = self._make_gaussian_state()
        NL_exp = np.ones_like(r_sim) * 1e20
        result = compare_density_profile(
            state, r_exp=r_sim, NL_exp=NL_exp, z_index=64, dx=0.005,
        )
        assert isinstance(result, SpatialComparisonResult)
        assert result.r_sim is not None
        assert result.ne_sim is not None
        assert result.NL_sim is not None
        assert result.fwhm_sim >= 0
        assert result.fwhm_exp >= 0


class TestSpatialNRMSEMulti:
    """Test multi-slice aggregation."""

    def test_aggregation(self):
        results = [
            SpatialComparisonResult(
                nrmse=0.1, r_sim=np.array([0]), ne_sim=np.array([0]),
                NL_sim=np.array([0]), r_exp=np.array([0]), NL_exp=np.array([0]),
                peak_ratio=1.1, fwhm_sim=0.02, fwhm_exp=0.025,
            ),
            SpatialComparisonResult(
                nrmse=0.2, r_sim=np.array([0]), ne_sim=np.array([0]),
                NL_sim=np.array([0]), r_exp=np.array([0]), NL_exp=np.array([0]),
                peak_ratio=0.9, fwhm_sim=0.03, fwhm_exp=0.025,
            ),
        ]
        agg = spatial_nrmse_multi(results)
        assert agg["mean_nrmse"] == pytest.approx(0.15)
        assert agg["max_nrmse"] == pytest.approx(0.2)
        assert agg["n_slices"] == 2
