"""Tests for filamentation diagnostic (Campaign 2G)."""
import numpy as np
import pytest

from dpf.diagnostics.filamentation import FilamentResult, detect_filaments


class TestFilamentDetection:
    def test_uniform_no_filaments(self):
        rho = np.full((16, 16, 16), 1.0)
        result = detect_filaments(rho, dx=0.001)
        assert result.n_filaments == 0
        assert result.density_contrast == pytest.approx(1.0, abs=0.01)
        assert not result.is_filamented

    def test_m4_perturbation_detected(self):
        nx, ny, nz = 32, 32, 16
        dx = 0.001
        x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        y = (np.arange(ny) - ny / 2.0 + 0.5) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        theta = np.arctan2(Y, X)
        rho = np.ones((nx, ny, nz))
        rho += 0.5 * np.cos(4 * theta)[:, :, np.newaxis]  # 50% m=4 perturbation
        result = detect_filaments(rho, dx=dx)
        assert result.dominant_m == 4
        assert result.density_contrast > 1.5
        assert result.is_filamented

    def test_m1_kink(self):
        nx, ny, nz = 32, 32, 16
        dx = 0.001
        x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        y = (np.arange(ny) - ny / 2.0 + 0.5) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        theta = np.arctan2(Y, X)
        rho = np.ones((nx, ny, nz))
        rho += 0.3 * np.cos(theta)[:, :, np.newaxis]  # m=1 kink
        result = detect_filaments(rho, dx=dx)
        assert result.dominant_m == 1

    def test_small_grid_returns_zero(self):
        rho = np.full((4, 4, 4), 1.0)
        result = detect_filaments(rho, dx=0.01)
        assert result.n_filaments == 0
        assert not result.is_filamented

    def test_filament_width_physical(self):
        nx, ny, nz = 32, 32, 16
        dx = 0.001
        x = (np.arange(nx) - nx / 2.0 + 0.5) * dx
        y = (np.arange(ny) - ny / 2.0 + 0.5) * dx
        X, Y = np.meshgrid(x, y, indexing="ij")
        theta = np.arctan2(Y, X)
        rho = np.ones((nx, ny, nz))
        rho += 0.5 * np.cos(8 * theta)[:, :, np.newaxis]
        result = detect_filaments(rho, dx=dx)
        assert result.filament_width_mm > 0
        assert isinstance(result, FilamentResult)
