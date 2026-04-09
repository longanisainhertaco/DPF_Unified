"""Tests for MLX-native Saha EOS (src/dpf/metal/mlx_eos.py).

Validates:
1. Saha Z_bar limits: Z~0 at low T, Z~1 at high T
2. Transition temperature for hydrogen (~1-3 eV)
3. Table interpolation accuracy vs direct computation
4. Temperature recovery with Saha correction
5. Integration with MLX solver constructor
"""

import numpy as np
import pytest

from dpf.metal.mlx_eos import SahaEOS, _saha_zbar_numpy


class TestSahaZbarNumpy:
    """Test the raw Saha computation."""

    def test_neutral_at_low_temperature(self):
        """Below 0.5 eV (~5800 K), deuterium should be mostly neutral."""
        T = np.array([300.0, 1000.0, 5000.0])
        Z = _saha_zbar_numpy(T, n_e=1e22)
        assert np.all(Z < 0.1), f"Expected Z < 0.1 at low T, got {Z}"

    def test_ionized_at_high_temperature(self):
        """Above 5 eV (~58,000 K), deuterium should be fully ionized."""
        T = np.array([60000.0, 100000.0, 1e6])
        Z = _saha_zbar_numpy(T, n_e=1e22)
        assert np.all(Z > 0.9), f"Expected Z > 0.9 at high T, got {Z}"

    def test_transition_region(self):
        """Z should transition between 0.1 and 0.9 in the 1-4 eV range."""
        # 1 eV = 11604 K, 4 eV = 46416 K
        T = np.array([11604.0, 23208.0, 46416.0])
        Z = _saha_zbar_numpy(T, n_e=1e22)
        # At least one point should be in transition
        assert np.any((Z > 0.1) & (Z < 0.9)), f"No transition point found: {Z}"

    def test_density_dependence(self):
        """Higher density shifts ionization to higher temperature (Saha)."""
        T = np.array([20000.0])
        Z_low_ne = _saha_zbar_numpy(T, n_e=1e20)
        Z_high_ne = _saha_zbar_numpy(T, n_e=1e24)
        assert Z_low_ne[0] > Z_high_ne[0], (
            f"Higher n_e should reduce Z: Z(1e20)={Z_low_ne[0]}, Z(1e24)={Z_high_ne[0]}"
        )

    def test_monotonic_in_temperature(self):
        """Z_bar should monotonically increase with temperature."""
        T = np.logspace(2, 7, 100)
        Z = _saha_zbar_numpy(T, n_e=1e22)
        dZ = np.diff(Z)
        assert np.all(dZ >= -1e-10), "Z_bar is not monotonic in T"


class TestSahaEOS:
    """Test the SahaEOS lookup table class."""

    @pytest.fixture()
    def eos(self):
        return SahaEOS(n_e_ref=1e22)

    def test_numpy_lookup_matches_direct(self, eos):
        """Table lookup should match direct computation within interpolation error."""
        T = np.logspace(3, 6, 50)
        Z_direct = _saha_zbar_numpy(T, n_e=1e22)
        Z_table = eos.zbar_numpy(T)
        np.testing.assert_allclose(Z_table, Z_direct, atol=0.01)

    def test_numpy_limits(self, eos):
        Z_cold = eos.zbar_numpy(np.array([300.0]))
        Z_hot = eos.zbar_numpy(np.array([1e6]))
        assert Z_cold[0] < 0.01
        assert Z_hot[0] > 0.99

    @pytest.mark.skipif(
        not pytest.importorskip("mlx.core", reason="MLX not available"),
        reason="MLX required",
    )
    def test_mlx_lookup_matches_numpy(self, eos):
        import mlx.core as mx
        T_np = np.logspace(3, 6, 50).astype(np.float32)
        T_mx = mx.array(T_np)
        Z_np = eos.zbar_numpy(T_np)
        Z_mx = np.array(eos.zbar_mlx(T_mx))
        np.testing.assert_allclose(Z_mx, Z_np, atol=0.02)

    def test_temperature_from_pressure_z1_limit(self, eos):
        """At high T (Z~1), Saha temperature should match Z=1 formula."""
        pytest.importorskip("mlx.core")
        import mlx.core as mx
        rho = mx.array([[1e-3]], dtype=mx.float32)
        # p chosen to give T ~ 100 eV = 1.16e6 K (fully ionized)
        m_d = 3.34358377e-27
        k_B = 1.380649e-23
        T_target = 1.16e6  # K
        p = mx.array([[2.0 * (rho[0, 0].item() / m_d) * k_B * T_target]], dtype=mx.float32)
        T_saha, Z = eos.temperature_from_pressure(rho, p)
        T_val = T_saha[0, 0].item()
        Z_val = Z[0, 0].item()
        assert Z_val > 0.99, f"Expected Z~1 at 100 eV, got {Z_val}"
        assert abs(T_val - T_target) / T_target < 0.05, (
            f"T mismatch: {T_val:.0f} vs {T_target:.0f}"
        )


class TestSolverIntegration:
    """Test that MLX solver accepts enable_saha_eos flag."""

    def test_solver_constructs_with_saha(self):
        """MLXMHDSolver should accept enable_saha_eos=True without error."""
        pytest.importorskip("mlx.core")
        from dpf.metal.mlx_solver import MLXMHDSolver
        solver = MLXMHDSolver(
            grid_shape=(16, 1, 32),
            dx=0.01,
            enable_saha_eos=True,
        )
        assert solver._saha_eos is not None
        assert solver._enable_saha_eos is True

    def test_solver_default_no_saha(self):
        """Default solver should not have Saha EOS (backward compatible)."""
        pytest.importorskip("mlx.core")
        from dpf.metal.mlx_solver import MLXMHDSolver
        solver = MLXMHDSolver(
            grid_shape=(16, 1, 32),
            dx=0.01,
        )
        assert solver._saha_eos is None
        assert solver._enable_saha_eos is False
