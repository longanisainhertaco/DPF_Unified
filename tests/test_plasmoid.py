"""Tests for plasmoid detection and force-free diagnostic (Campaign 2F)."""
import numpy as np
import pytest


class TestFluxFunction:
    def test_flux_from_uniform_Bz(self):
        from dpf.diagnostics.plasmoid import compute_flux_function
        B = np.zeros((3, 8, 8, 16))
        B[2] = 1.0  # uniform Bz
        psi = compute_flux_function(B, dr=0.001, dz=0.001)
        # psi should be monotonically increasing in x (cumsum of Bz)
        assert psi.shape == (8, 16)
        assert psi[-1, 0] > psi[0, 0]

    def test_zero_B_gives_zero_psi(self):
        from dpf.diagnostics.plasmoid import compute_flux_function
        B = np.zeros((3, 8, 8, 16))
        psi = compute_flux_function(B, dr=0.001, dz=0.001)
        np.testing.assert_array_equal(psi, 0)


class TestCriticalPoints:
    def test_single_island(self):
        from dpf.diagnostics.plasmoid import find_critical_points
        # Create psi with a single magnetic island (one O-point)
        nx, nz = 32, 32
        x = np.linspace(-1, 1, nx)
        z = np.linspace(-1, 1, nz)
        X, Z = np.meshgrid(x, z, indexing="ij")
        psi = -(X**2 + Z**2)  # single maximum at origin → O-point
        o_pts, x_pts = find_critical_points(psi)
        assert len(o_pts) >= 1

    def test_saddle_point(self):
        from dpf.diagnostics.plasmoid import find_critical_points
        nx, nz = 32, 32
        x = np.linspace(-1, 1, nx)
        z = np.linspace(-1, 1, nz)
        X, Z = np.meshgrid(x, z, indexing="ij")
        psi = X**2 - Z**2  # saddle at origin → X-point
        o_pts, x_pts = find_critical_points(psi)
        assert len(x_pts) >= 1


class TestDetectPlasmoids:
    def test_laminar_field(self):
        from dpf.diagnostics.plasmoid import detect_plasmoids
        B = np.zeros((3, 8, 8, 16))
        B[2] = 1.0
        rho = np.full((8, 8, 16), 0.084)
        result = detect_plasmoids(B, rho, dr=0.001, dz=0.001)
        assert result["topology"] == "laminar"
        assert result["magnetic_energy_J"] > 0

    def test_result_keys(self):
        from dpf.diagnostics.plasmoid import detect_plasmoids
        B = np.zeros((3, 8, 8, 16))
        B[1] = 0.5
        rho = np.full((8, 8, 16), 0.084)
        result = detect_plasmoids(B, rho, dr=0.001, dz=0.001)
        assert "n_plasmoids" in result
        assert "n_o_points" in result
        assert "n_x_points" in result
        assert "topology" in result
        assert "magnetic_energy_J" in result


class TestForceFreeDiagnostic:
    def test_uniform_B_is_force_free(self):
        from dpf.diagnostics.plasmoid import force_free_diagnostic
        B = np.zeros((3, 16, 16, 16))
        B[2] = 1.0  # uniform Bz → J = 0 everywhere → trivially force-free
        ff = force_free_diagnostic(B, dx=0.001, dz=0.001)
        assert ff.alpha_ff == pytest.approx(0.0, abs=1e-10)

    def test_nonuniform_B_has_current(self):
        from dpf.diagnostics.plasmoid import force_free_diagnostic
        B = np.zeros((3, 16, 16, 16))
        # B_theta profile (like DPF pinch)
        r = np.linspace(0.001, 0.03, 16)
        B[1] = (1.0 / r[:, np.newaxis, np.newaxis]) * np.ones((16, 16, 16))
        ff = force_free_diagnostic(B, dx=0.002, dz=0.002)
        # Should have nonzero force-free error (not force-free)
        assert ff.force_free_error > 0

    def test_force_free_error_bounded(self):
        from dpf.diagnostics.plasmoid import force_free_diagnostic
        B = np.random.RandomState(42).randn(3, 8, 8, 8) * 0.1
        ff = force_free_diagnostic(B, dx=0.001, dz=0.001)
        assert 0 <= ff.force_free_error <= 1.0
        assert 0 <= ff.j_parallel_frac <= 1.0
