"""Tests for the MHD solver."""

from __future__ import annotations

import numpy as np
import pytest

from dpf.core.field_manager import FieldManager
from dpf.fluid.mhd_solver import MHDSolver, _dedner_source


class TestFieldManager:
    """Verify field manager vector calculus operations."""

    def test_divergence_zero_for_curl(self):
        """div(curl(F)) = 0 for any smooth field."""
        fm = FieldManager((16, 16, 16), dx=0.1)
        # Create a smooth vector field
        x = np.linspace(0, 1, 16)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        F = np.array([np.sin(2 * np.pi * Y), np.sin(2 * np.pi * Z), np.sin(2 * np.pi * X)])
        curl_F = fm.curl(F)
        div_curl = fm.divergence(curl_F)
        # Should be close to zero (finite-difference discretization error)
        assert np.max(np.abs(div_curl)) < 1.0  # Loose bound for np.gradient

    def test_neumann_bc_shape(self):
        """Neumann BC preserves array shape."""
        fm = FieldManager((10, 10, 10), dx=0.1)
        field = np.random.rand(3, 10, 10, 10)
        result = fm.apply_neumann_bc(field, ng=2)
        assert result.shape == field.shape

    def test_max_div_B_initial(self):
        """Initial B=0 has zero divergence."""
        fm = FieldManager((8, 8, 8), dx=0.1)
        assert fm.max_div_B() == 0.0


class TestDednerCleaning:
    """Verify Dedner divergence cleaning."""

    def test_reduces_div_B(self):
        """Dedner source term opposes existing div(B)."""
        nx = 16
        dx = 0.1
        B = np.zeros((3, nx, nx, nx))
        # Add a non-solenoidal component
        x = np.linspace(0, 1, nx)
        X, _, _ = np.meshgrid(x, x, x, indexing="ij")
        B[0] = X  # div(B) = 1 everywhere (non-zero)

        psi = np.zeros((nx, nx, nx))
        ch = 1.0
        cp = 1.0

        dpsi_dt, dB_dt = _dedner_source(psi, B, ch, cp, dx)

        # dpsi_dt should be negative (opposing positive div B)
        # dB_dt should correct B to reduce divergence
        assert np.mean(dpsi_dt) < 0


class TestMHDSolver:
    """Integration tests for the MHD solver."""

    def test_step_returns_correct_keys(self):
        """MHD step returns state with all expected keys."""
        solver = MHDSolver((8, 8, 8), dx=0.01)
        state = {
            "rho": np.full((8, 8, 8), 1e-4),
            "velocity": np.zeros((3, 8, 8, 8)),
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-10, current=0.0, voltage=0.0)

        for key in ["rho", "velocity", "pressure", "B", "Te", "Ti", "psi"]:
            assert key in result

    def test_density_floor(self):
        """Density never goes negative."""
        solver = MHDSolver((8, 8, 8), dx=0.01)
        state = {
            "rho": np.full((8, 8, 8), 1e-30),  # Very low density
            "velocity": np.random.randn(3, 8, 8, 8) * 1e3,
            "pressure": np.full((8, 8, 8), 1.0),
            "B": np.zeros((3, 8, 8, 8)),
            "Te": np.full((8, 8, 8), 1e4),
            "Ti": np.full((8, 8, 8), 1e4),
            "psi": np.zeros((8, 8, 8)),
        }
        result = solver.step(state, dt=1e-12, current=0.0, voltage=0.0)
        assert np.all(result["rho"] > 0)

    def test_coupling_interface(self):
        """Coupling interface returns a CouplingState."""
        from dpf.core.bases import CouplingState

        solver = MHDSolver((8, 8, 8), dx=0.01)
        cs = solver.coupling_interface()
        assert isinstance(cs, CouplingState)
