"""Tests for the Athena++ wrapper layer (Sprint F.1).

These tests validate:
1. athinput generation from SimulationConfig
2. AthenaPPSolver instantiation and interface compliance
3. Subprocess mode execution (if binary available)
4. HDF5 output reading
5. State dict shape and key conventions
6. Linked mode (if C++ extension compiled)

Tests are organized by dependency — pure-Python tests first,
then integration tests that need the binary or extension.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dpf.config import SimulationConfig
from dpf.core.bases import CouplingState, PlasmaSolverBase

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def dpf_config(default_circuit_params):
    """Minimal DPF config for Athena++ wrapper tests."""
    return SimulationConfig(
        grid_shape=[16, 1, 32],
        dx=1e-3,
        sim_time=1e-7,
        circuit=default_circuit_params,
        geometry={"type": "cylindrical"},
    )


@pytest.fixture
def cartesian_config(default_circuit_params):
    """Minimal Cartesian DPF config."""
    return SimulationConfig(
        grid_shape=[8, 8, 8],
        dx=1e-2,
        sim_time=1e-6,
        circuit=default_circuit_params,
        geometry={"type": "cartesian"},
    )


@pytest.fixture
def athena_binary():
    """Path to compiled Athena++ binary, or None if not available."""
    candidates = [
        Path(__file__).parent.parent / "external" / "athena" / "bin" / "athena",
        Path.cwd() / "external" / "athena" / "bin" / "athena",
    ]
    for p in candidates:
        if p.is_file() and os.access(str(p), os.X_OK):
            return str(p)
    return None


# ============================================================
# Test: athinput generation
# ============================================================


class TestAthinputGeneration:
    """Tests for athena_config.py: SimulationConfig -> athinput."""

    def test_generate_athinput_returns_string(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_athinput_has_required_blocks(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert "<job>" in text
        assert "<time>" in text
        assert "<mesh>" in text
        assert "<hydro>" in text
        assert "<problem>" in text

    def test_athinput_cylindrical_coordinates(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        # Cylindrical: should have reflecting inner R BC
        assert "ix1_bc     = reflecting" in text
        assert "ox1_bc     = outflow" in text
        # nx3=1 for axisymmetric
        assert "nx3        = 1" in text

    def test_athinput_cartesian_coordinates(self, cartesian_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(cartesian_config)
        assert "coord=cartesian" in text

    def test_athinput_preserves_circuit_params(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert "V0" in text
        assert "C " in text or "capacitance" in text.lower()

    def test_athinput_gamma(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        # Default gamma = 5/3 ≈ 1.6667
        assert "1.6667" in text

    def test_athinput_cfl_number(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert "cfl_number" in text
        assert "0.4" in text

    def test_athinput_grid_dimensions(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        # nr=16 in grid_shape[0]
        assert "nx1        = 16" in text
        # nz=32 in grid_shape[2]
        assert "nx2        = 32" in text

    def test_write_athinput_creates_file(self, dpf_config, tmp_path):
        from dpf.athena_wrapper.athena_config import write_athinput
        path = str(tmp_path / "test_athinput.dpf")
        text = write_athinput(dpf_config, path)
        assert os.path.isfile(path)
        with open(path) as f:
            contents = f.read()
        assert contents == text

    def test_athinput_custom_problem_id(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config, problem_id="my_test")
        assert "problem_id  = my_test" in text


# ============================================================
# Test: AthenaPPSolver interface
# ============================================================


class TestAthenaPPSolverInterface:
    """Tests for athena_engine.py: AthenaPPSolver class."""

    def test_solver_is_plasma_solver_base(self):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        assert issubclass(AthenaPPSolver, PlasmaSolverBase)

    def test_solver_instantiation_subprocess(self, dpf_config, athena_binary):
        """AthenaPPSolver can be created in subprocess mode."""
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        assert solver.mode == "subprocess"

    def test_solver_has_step_method(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        assert callable(getattr(solver, "step", None))

    def test_solver_has_coupling_interface(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        coupling = solver.coupling_interface()
        assert isinstance(coupling, CouplingState)

    def test_initial_state_shapes_cylindrical(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        nx, ny, nz = dpf_config.grid_shape
        assert state["rho"].shape == (nx, ny, nz)
        assert state["velocity"].shape == (3, nx, ny, nz)
        assert state["pressure"].shape == (nx, ny, nz)
        assert state["B"].shape == (3, nx, ny, nz)
        assert state["Te"].shape == (nx, ny, nz)
        assert state["Ti"].shape == (nx, ny, nz)
        assert state["psi"].shape == (nx, ny, nz)

    def test_initial_state_shapes_cartesian(self, cartesian_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            cartesian_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        nx, ny, nz = cartesian_config.grid_shape
        assert state["rho"].shape == (nx, ny, nz)
        assert state["velocity"].shape == (3, nx, ny, nz)

    def test_initial_state_values(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        # Density should be uniform rho0
        np.testing.assert_allclose(state["rho"], dpf_config.rho0)
        # Velocity should be zero
        np.testing.assert_allclose(state["velocity"], 0.0)
        # B-field should be zero
        np.testing.assert_allclose(state["B"], 0.0)
        # Temperature should be T0
        np.testing.assert_allclose(state["Te"], dpf_config.T0)

    def test_compute_dt_positive(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        dt = solver._compute_dt(state)
        assert dt > 0
        assert np.isfinite(dt)

    def test_time_and_cycle_properties(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        assert solver.time == 0.0
        assert solver.cycle == 0


# ============================================================
# Test: State dictionary conventions
# ============================================================


class TestStateDictConventions:
    """Verify state dict matches DPF conventions exactly."""

    REQUIRED_KEYS = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}

    def test_initial_state_has_all_keys(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        assert set(state.keys()) == self.REQUIRED_KEYS

    def test_state_values_are_numpy(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        for key, arr in state.items():
            assert isinstance(arr, np.ndarray), f"{key} is not ndarray"

    def test_state_values_are_finite(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        for key, arr in state.items():
            assert np.all(np.isfinite(arr)), f"{key} has non-finite values"


# ============================================================
# Test: Subprocess execution (integration)
# ============================================================


class TestSubprocessMode:
    """Integration tests for subprocess mode execution."""

    @pytest.mark.slow
    def test_subprocess_single_step(self, dpf_config, athena_binary):
        """Run a single timestep via subprocess and verify output."""
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        dt = 1e-10  # Very small timestep
        new_state = solver.step(state, dt, current=0.0, voltage=0.0)

        # State should be returned (may be unchanged for zero forcing)
        assert set(new_state.keys()) == set(state.keys())
        for key in state:
            assert new_state[key].shape == state[key].shape

    @pytest.mark.slow
    def test_subprocess_updates_time(self, dpf_config, athena_binary):
        """Verify time advances after a step."""
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        dt = 1e-10
        solver.step(state, dt, current=0.0, voltage=0.0)
        assert solver.time == pytest.approx(dt)
        assert solver.cycle == 1


# ============================================================
# Test: athinput parameter override
# ============================================================


class TestAthinputOverride:
    """Tests for the _override_athinput_param utility."""

    def test_override_existing_param(self):
        from dpf.athena_wrapper.athena_engine import _override_athinput_param
        text = "<time>\ncfl_number  = 0.4\ntlim        = 1.0e-6\n"
        result = _override_athinput_param(text, "time", "tlim", "2.0e-6")
        assert "2.0e-6" in result
        assert "1.0e-6" not in result

    def test_override_preserves_comment(self):
        from dpf.athena_wrapper.athena_engine import _override_athinput_param
        text = "<time>\ntlim        = 1.0e-6  # time limit\n"
        result = _override_athinput_param(text, "time", "tlim", "2.0e-6")
        assert "2.0e-6" in result
        assert "time limit" in result

    def test_override_only_target_block(self):
        from dpf.athena_wrapper.athena_engine import _override_athinput_param
        text = "<time>\ndt = 1e-9\n<mesh>\ndt = 5e-8\n"
        result = _override_athinput_param(text, "time", "dt", "2e-9")
        lines = result.split("\n")
        # Only the <time> block's dt should change
        assert any("2e-9" in line for line in lines)
        assert any("5e-8" in line for line in lines)


# ============================================================
# Test: C++ extension availability check
# ============================================================


class TestExtensionAvailability:
    """Tests for the is_available() check."""

    def test_is_available_returns_bool(self):
        from dpf.athena_wrapper import is_available
        result = is_available()
        assert isinstance(result, bool)

    def test_import_always_works(self):
        """The wrapper package should always be importable."""
        from dpf.athena_wrapper import AthenaPPSolver, generate_athinput
        assert AthenaPPSolver is not None
        assert generate_athinput is not None


# ============================================================
# Test: HDF5 output reading
# ============================================================


class TestHDF5Reading:
    """Tests for reading Athena++ HDF5 output files."""

    def test_read_existing_athdf(self, dpf_config, athena_binary):
        """Read the smoke test output from Sprint F.0."""
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver

        if athena_binary is None:
            pytest.skip("Athena++ binary not available")

        # Check if smoke test output exists
        smoke_dir = "/tmp/athena_smoke_test"
        if not os.path.isdir(smoke_dir):
            pytest.skip("Smoke test output not found at /tmp/athena_smoke_test")

        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()  # For fallback
        result = solver._read_hdf5_output(smoke_dir, state)

        # Should get valid arrays
        assert "rho" in result
        assert "velocity" in result
        assert "B" in result
        assert result["rho"].ndim == 3
        assert result["velocity"].ndim == 4
        assert result["velocity"].shape[0] == 3

    def test_read_missing_dir_returns_fallback(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        result = solver._read_hdf5_output("/nonexistent/path", state)
        # Should return fallback state unchanged
        assert result is state


# ============================================================
# Test: Temperature estimation
# ============================================================


class TestTemperatureEstimation:
    """Tests for Te/Ti estimation from single-fluid variables."""

    def test_estimate_Te_positive(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        rho = np.full((4, 1, 8), 1e-4)
        pressure = np.full((4, 1, 8), 1.0)
        Te = solver._estimate_Te(rho, pressure)
        assert np.all(Te > 0)
        assert np.all(np.isfinite(Te))

    def test_estimate_Te_equals_Ti(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        rho = np.full((4, 1, 8), 1e-4)
        pressure = np.full((4, 1, 8), 100.0)
        Te = solver._estimate_Te(rho, pressure)
        Ti = solver._estimate_Ti(rho, pressure)
        np.testing.assert_array_equal(Te, Ti)

    def test_estimate_Te_zero_density_safe(self, dpf_config, athena_binary):
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver
        if athena_binary is None:
            pytest.skip("Athena++ binary not available")
        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        rho = np.zeros((4, 1, 8))
        pressure = np.full((4, 1, 8), 1.0)
        Te = solver._estimate_Te(rho, pressure)
        assert np.all(Te >= 1.0)
        assert np.all(np.isfinite(Te))
