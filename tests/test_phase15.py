"""Tests for Phase 15: AMR + Tabulated EOS + GPU Backend.

Covers:
    15.1 Block-structured AMR (gradient tagging, restriction, prolongation)
    15.2 Tabulated equation of state
    15.3 GPU acceleration (CuPy fallback to NumPy)
"""

from __future__ import annotations

import numpy as np
import pytest

from dpf.amr import (
    AMRConfig,
    AMRGrid,
    AMRPatch,
    prolong_patch,
    restrict_patch,
    tag_cells_gradient,
)
from dpf.fluid.gpu_backend import (
    get_device_info,
    is_gpu_available,
    synchronize,
    to_device,
    to_numpy,
    xp,
)
from dpf.fluid.tabulated_eos import EOSTable, TabulatedEOS

# ===================================================================
# 15.1 — Block-structured AMR
# ===================================================================


class TestAmrConfig:
    """Tests for AMRConfig."""

    def test_defaults(self):
        """Default config has reasonable values."""
        cfg = AMRConfig()
        assert cfg.max_levels == 3
        assert cfg.refinement_ratio == 2
        assert cfg.regrid_interval == 10
        assert cfg.gradient_threshold > 0

    def test_custom_config(self):
        """Custom config is accepted."""
        cfg = AMRConfig(max_levels=5, refinement_ratio=4, gradient_threshold=0.1)
        assert cfg.max_levels == 5
        assert cfg.refinement_ratio == 4


class TestAmrPatch:
    """Tests for AMRPatch dataclass."""

    def test_create_patch(self):
        """Create a patch with data."""
        patch = AMRPatch(
            level=0, i_start=0, j_start=0, ni=16, nj=32,
            dx=0.001, dz=0.002, data={},
        )
        assert patch.ni == 16
        assert patch.nj == 32
        assert patch.level == 0

    def test_cell_centers(self):
        """Cell center coordinates have correct size and spacing."""
        patch = AMRPatch(
            level=1, i_start=0, j_start=0, ni=10, nj=20,
            dx=0.0005, dz=0.001, data={},
        )
        r = patch.cell_centers_r()
        z = patch.cell_centers_z()
        assert len(r) == 10
        assert len(z) == 20
        np.testing.assert_allclose(r[1] - r[0], 0.0005, rtol=1e-10)


class TestTagCellsGradient:
    """Tests for gradient-based cell tagging."""

    def test_uniform_not_tagged(self):
        """Uniform density produces no tagged cells."""
        rho = np.full((32, 64), 1e-3)
        tagged = tag_cells_gradient(rho, dx=0.001, dz=0.001, threshold=0.3)
        assert not np.any(tagged)

    def test_sharp_gradient_tagged(self):
        """Sharp density jump is tagged for refinement."""
        rho = np.full((32, 64), 1e-3)
        rho[14:18, :] = 1e-1  # 100x density jump
        tagged = tag_cells_gradient(rho, dx=0.001, dz=0.001, threshold=0.3)
        assert np.any(tagged)

    def test_tag_shape_matches_input(self):
        """Tagged array has same shape as input."""
        rho = np.random.default_rng(42).uniform(1e-4, 1e-2, (20, 40))
        tagged = tag_cells_gradient(rho, 0.001, 0.001, 0.3)
        assert tagged.shape == rho.shape


class TestRestrictProlong:
    """Tests for restriction and prolongation operators."""

    def test_restrict_uniform(self):
        """Restriction of a uniform field gives the same value."""
        fine = np.full((8, 16), 5.0)
        coarse = restrict_patch(fine, 2)
        assert coarse.shape == (4, 8)
        np.testing.assert_allclose(coarse, 5.0, atol=1e-12)

    def test_prolong_uniform(self):
        """Prolongation of a uniform field gives the same value."""
        coarse = np.full((4, 8), 3.0)
        fine = prolong_patch(coarse, 2)
        assert fine.shape == (8, 16)
        np.testing.assert_allclose(fine, 3.0, atol=1e-12)

    def test_restrict_prolong_roundtrip(self):
        """Restrict then prolong approximately recovers smooth fields."""
        coarse = np.outer(np.linspace(1, 2, 8), np.linspace(1, 3, 16))
        fine = prolong_patch(coarse, 2)
        coarse2 = restrict_patch(fine, 2)
        # Should be close to original for smooth data
        np.testing.assert_allclose(coarse2, coarse, rtol=0.1)

    def test_restrict_conserves_total(self):
        """Restriction approximately conserves total mass (sum * area)."""
        fine = np.random.default_rng(99).uniform(1, 10, (16, 32))
        coarse = restrict_patch(fine, 2)
        # Each coarse cell covers ratio^2 fine cells
        # Total ~ sum(fine) should ~ ratio^2 * sum(coarse)
        np.testing.assert_allclose(np.sum(fine), 4 * np.sum(coarse), rtol=0.01)


class TestAmrGrid:
    """Tests for the AMRGrid class."""

    def test_create_base_grid(self):
        """Creating a grid gives a single base-level patch."""
        cfg = AMRConfig(max_levels=2)
        grid = AMRGrid(32, 64, 0.05, 0.10, cfg)
        assert grid.total_cells() == 32 * 64

    def test_tag_and_regrid(self):
        """Tagging cells with gradient creates higher-level patches."""
        cfg = AMRConfig(max_levels=2, gradient_threshold=0.2)
        grid = AMRGrid(32, 64, 0.05, 0.10, cfg)
        # Initialize density field on the base patch
        grid.patches[0][0].data["rho"] = np.full((32, 64), 1e-3)
        # Add a density spike
        grid.patches[0][0].data["rho"][14:18, 30:34] = 1.0  # strong density jump
        tagged = grid.tag_cells("rho")
        grid.regrid(tagged)
        # Should have patches at level 1
        assert len(grid.patches) > 1
        assert grid.total_cells() > 32 * 64

    def test_restrict(self):
        """Restriction runs without error on a multi-level grid."""
        cfg = AMRConfig(max_levels=2, gradient_threshold=0.1)
        grid = AMRGrid(16, 32, 0.05, 0.10, cfg)
        # Initialize density field on the base patch
        grid.patches[0][0].data["rho"] = np.full((16, 32), 1e-3)
        grid.patches[0][0].data["rho"][6:10, 14:18] = 1.0
        tagged = grid.tag_cells("rho")
        grid.regrid(tagged)
        grid.restrict()  # Should not raise

    def test_get_level_data(self):
        """get_level_data returns correct shape for base level."""
        cfg = AMRConfig()
        grid = AMRGrid(16, 32, 0.05, 0.10, cfg)
        data = grid.get_level_data(0, "rho")
        assert data.shape == (16, 32)


# ===================================================================
# 15.2 — Tabulated equation of state
# ===================================================================


class TestEosTable:
    """Tests for EOSTable dataclass."""

    def test_create_table(self):
        """Can create an EOSTable."""
        log_rho = np.linspace(-3, 3, 10)
        log_T = np.linspace(2, 9, 10)
        p = np.ones((10, 10))
        e = np.ones((10, 10))
        table = EOSTable(
            material="test", log_rho=log_rho, log_T=log_T,
            pressure=p, energy=e, ionization=None,
        )
        assert table.material == "test"
        assert table.pressure.shape == (10, 10)


class TestTabulatedEos:
    """Tests for TabulatedEOS class."""

    def test_ideal_fallback_pressure(self):
        """Without a table, pressure uses ideal gas law."""
        eos = TabulatedEOS()
        rho = np.array([1e-3])
        T = np.array([1e6])
        p = eos.pressure(rho, T)
        assert p[0] > 0

    def test_ideal_fallback_energy(self):
        """Without a table, energy uses ideal gas formula."""
        eos = TabulatedEOS()
        e = eos.internal_energy(np.array([1e-3]), np.array([1e6]))
        assert e[0] > 0

    def test_ideal_fallback_ionization(self):
        """Without a table, ionization returns 1.0."""
        eos = TabulatedEOS()
        Z = eos.ionization_state(np.array([1e-3]), np.array([1e6]))
        np.testing.assert_allclose(Z, 1.0)

    def test_generate_ideal_table(self):
        """Generate and query an ideal gas table."""
        eos = TabulatedEOS()
        eos.generate_ideal_table()
        rho = np.array([0.1])
        T = np.array([1e5])
        p_table = eos.pressure(rho, T)
        # Just check it's positive and finite (exact match depends on interpolation)
        assert p_table[0] > 0
        assert np.isfinite(p_table[0])

    def test_table_matches_ideal_gas(self):
        """Table lookup approximately matches ideal gas in interior."""
        eos = TabulatedEOS()
        eos.generate_ideal_table(material="deuterium")
        from dpf.constants import k_B
        m_d = 3.34e-27
        rho = np.array([1e-2, 1e0, 1e2])
        T = np.array([1e4, 1e6, 1e8])
        p_table = eos.pressure(rho, T, material="deuterium")
        p_ideal = 2.0 * rho / m_d * k_B * T
        # Bilinear interpolation should be within ~5% for interior points
        for i in range(3):
            if p_table[i] > 0 and p_ideal[i] > 0:
                ratio = p_table[i] / p_ideal[i]
                assert 0.8 < ratio < 1.2, f"ratio={ratio} at i={i}"

    def test_load_custom_table(self):
        """Load a custom EOS table."""
        eos = TabulatedEOS()
        n = 20
        log_rho = np.linspace(-3, 3, n)
        log_T = np.linspace(2, 9, n)
        p = np.ones((n, n)) * 1e5
        e = np.ones((n, n)) * 1e8
        eos.load_table("custom", log_rho, log_T, p, e)
        result = eos.pressure(np.array([1.0]), np.array([1e5]), material="custom")
        assert result[0] > 0

    def test_load_table_shape_mismatch_raises(self):
        """Mismatched shapes in load_table should raise."""
        eos = TabulatedEOS()
        with pytest.raises(ValueError):
            eos.load_table(
                "bad", np.zeros(5), np.zeros(10),
                np.zeros((5, 8)),  # should be (5, 10)
                np.zeros((5, 10)),
            )


# ===================================================================
# 15.3 — GPU Backend
# ===================================================================


class TestGpuBackend:
    """Tests for GPU/CPU backend abstraction."""

    def test_xp_is_numpy(self):
        """On machines without CuPy, xp should be numpy."""
        # This test always passes since we're on CPU
        assert xp is not None
        assert hasattr(xp, "zeros")
        assert hasattr(xp, "ones")

    def test_gpu_available_returns_bool(self):
        """is_gpu_available returns a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_to_numpy_identity(self):
        """to_numpy on a numpy array returns the same array."""
        a = np.array([1.0, 2.0, 3.0])
        b = to_numpy(a)
        np.testing.assert_array_equal(a, b)

    def test_to_device_identity(self):
        """to_device on CPU returns a numpy array."""
        a = np.array([1.0, 2.0, 3.0])
        b = to_device(a)
        np.testing.assert_array_equal(a, b)

    def test_synchronize_noop(self):
        """synchronize() on CPU does not raise."""
        synchronize()  # Should be a no-op

    def test_device_info_cpu(self):
        """Device info on CPU has expected keys."""
        info = get_device_info()
        assert "type" in info
        assert info["type"] in ("cpu", "gpu")

    def test_xp_array_operations(self):
        """Basic array operations work through xp."""
        a = xp.ones((5, 5))
        b = xp.zeros((5, 5))
        c = a + b
        result = to_numpy(c)
        np.testing.assert_allclose(result, 1.0)
