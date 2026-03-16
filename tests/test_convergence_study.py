"""Tests for grid convergence study utilities.

Tests the mathematical functions (convergence order, Richardson
extrapolation, GCI) without running actual simulations.
"""

from __future__ import annotations

import pytest

from dpf.validation.convergence_study import (
    ConvergenceResult,
    compute_convergence_order,
    grid_convergence_index,
    richardson_extrapolation,
)


class TestConvergenceOrder:
    def test_second_order(self):
        """f(h) = 1 + h^2: should give p=2."""
        # f(h) = 1 + h^2 with h = dx
        # h1=0.25, h2=0.5, h3=1.0, r=2
        f1 = 1 + 0.25**2  # 1.0625
        f2 = 1 + 0.5**2   # 1.25
        f3 = 1 + 1.0**2   # 2.0
        p = compute_convergence_order(f1, f2, f3, r=2.0)
        assert p == pytest.approx(2.0, abs=0.01)

    def test_first_order(self):
        """f(h) = 1 + h: should give p=1."""
        f1 = 1 + 0.25
        f2 = 1 + 0.5
        f3 = 1 + 1.0
        p = compute_convergence_order(f1, f2, f3, r=2.0)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_converged_returns_zero(self):
        """If all solutions are identical, order should be 0."""
        p = compute_convergence_order(1.0, 1.0, 1.0, r=2.0)
        assert p == 0.0

    def test_oscillatory_returns_zero(self):
        """Oscillatory convergence should return 0."""
        p = compute_convergence_order(1.0, 1.1, 1.05, r=2.0)
        assert p == 0.0  # (1.05-1.1)/(1.1-1.0) = -0.5 → negative ratio


class TestRichardsonExtrapolation:
    def test_second_order(self):
        """Richardson extrapolation of f(h)=1+h^2 should give 1.0."""
        f1 = 1 + 0.25**2  # 1.0625
        f2 = 1 + 0.5**2   # 1.25
        f_exact = richardson_extrapolation(f1, f2, p=2.0, r=2.0)
        assert f_exact == pytest.approx(1.0, abs=0.01)

    def test_first_order(self):
        """Richardson extrapolation of f(h)=1+h should give 1.0."""
        f1 = 1 + 0.25
        f2 = 1 + 0.5
        f_exact = richardson_extrapolation(f1, f2, p=1.0, r=2.0)
        assert f_exact == pytest.approx(1.0, abs=0.01)

    def test_zero_order_returns_f1(self):
        """If p=0, can't extrapolate — return finest grid value."""
        f = richardson_extrapolation(1.5, 2.0, p=0.0)
        assert f == 1.5


class TestGCI:
    def test_small_error_small_gci(self):
        """Small relative error should give small GCI."""
        gci = grid_convergence_index(1.000, 1.001, p=2.0, r=2.0)
        assert gci < 0.01  # < 1%

    def test_large_error_large_gci(self):
        """Large relative error should give larger GCI."""
        gci = grid_convergence_index(1.0, 1.5, p=1.0, r=2.0)
        assert gci > 0.1  # > 10%

    def test_higher_order_reduces_gci(self):
        """Higher convergence order should reduce GCI for same error."""
        gci_p1 = grid_convergence_index(1.0, 1.1, p=1.0, r=2.0)
        gci_p2 = grid_convergence_index(1.0, 1.1, p=2.0, r=2.0)
        assert gci_p2 < gci_p1

    def test_zero_f1_returns_one(self):
        """Zero finest solution should return 100% uncertainty."""
        gci = grid_convergence_index(0.0, 1.0, p=2.0, r=2.0)
        assert gci == 1.0

    def test_safety_factor(self):
        """Higher safety factor should increase GCI."""
        gci_125 = grid_convergence_index(1.0, 1.1, p=2.0, r=2.0, Fs=1.25)
        gci_300 = grid_convergence_index(1.0, 1.1, p=2.0, r=2.0, Fs=3.0)
        assert gci_300 > gci_125


class TestConvergenceResult:
    def test_dataclass_creation(self):
        result = ConvergenceResult(
            resolutions=[(16, 1, 32), (32, 1, 64)],
            dx_values=[0.002, 0.001],
            I_peak_values=[1.5, 1.55],
            t_peak_values=[5.0, 5.1],
            B_max_values=[10.0, 12.0],
            rho_max_values=[1e-3, 2e-3],
            wall_times=[1.0, 4.0],
            convergence_order=1.5,
            richardson_I_peak=1.58,
            gci_fine=0.03,
            is_converged=True,
            preset="pf1000",
            summary="test summary",
        )
        assert result.is_converged is True
        assert len(result.resolutions) == 2
        assert result.convergence_order == 1.5
