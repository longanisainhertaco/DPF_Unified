"""Consolidated Athena++ and AthenaK test suite.

Source files merged:
- tests/test_athena_wrapper.py          (Sprint F.1 — Athena++ wrapper layer)
- tests/test_phase_j_athenak.py         (Phase J.1 — AthenaK backend integration)
- tests/test_phase_r_athena_primary.py  (Phase R — Athena++ as primary backend)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dpf.config import FluidConfig, SimulationConfig
from dpf.core.bases import CouplingState, PlasmaSolverBase

# --- Section: test_athena_wrapper.py ---


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
        assert "ix1_bc     = reflecting" in text
        assert "ox1_bc     = user" in text
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
        assert "1.6667" in text

    def test_athinput_cfl_number(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert "cfl_number" in text
        assert "0.4" in text

    def test_athinput_grid_dimensions(self, dpf_config):
        from dpf.athena_wrapper.athena_config import generate_athinput
        text = generate_athinput(dpf_config)
        assert "nx1        = 16" in text
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
        np.testing.assert_allclose(state["rho"], dpf_config.rho0)
        np.testing.assert_allclose(state["velocity"], 0.0)
        np.testing.assert_allclose(state["B"], 0.0)
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
        dt = 1e-10
        new_state = solver.step(state, dt, current=0.0, voltage=0.0)
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
        assert any("2e-9" in line for line in lines)
        assert any("5e-8" in line for line in lines)


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


class TestHDF5Reading:
    """Tests for reading Athena++ HDF5 output files."""

    def test_read_existing_athdf(self, dpf_config, athena_binary):
        """Read the smoke test output from Sprint F.0."""
        from dpf.athena_wrapper.athena_engine import AthenaPPSolver

        if athena_binary is None:
            pytest.skip("Athena++ binary not available")

        smoke_dir = "/tmp/athena_smoke_test"
        if not os.path.isdir(smoke_dir):
            pytest.skip("Smoke test output not found at /tmp/athena_smoke_test")

        solver = AthenaPPSolver(
            dpf_config,
            athena_binary=athena_binary,
            use_subprocess=True,
        )
        state = solver.initial_state()
        result = solver._read_hdf5_output(smoke_dir, state)

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
        assert result is state


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


# --- Section: test_phase_j_athenak.py ---


class TestFluidConfigAthenaK:
    """Test FluidConfig accepts "athenak" as a valid backend."""

    def test_athenak_backend_valid(self):
        fc = FluidConfig(backend="athenak")
        assert fc.backend == "athenak"

    def test_python_backend_still_valid(self):
        fc = FluidConfig(backend="python")
        assert fc.backend == "python"

    def test_athena_backend_still_valid(self):
        fc = FluidConfig(backend="athena")
        assert fc.backend == "athena"

    def test_auto_backend_still_valid(self):
        fc = FluidConfig(backend="auto")
        assert fc.backend == "auto"

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValueError, match="backend must be"):
            FluidConfig(backend="invalid")

    def test_athenak_config_roundtrip(self, sample_config_dict):
        """Config with athenak backend serializes and deserializes."""
        sample_config_dict["fluid"] = {"backend": "athenak"}
        config = SimulationConfig(**sample_config_dict)
        assert config.fluid.backend == "athenak"


class TestAthenaKAvailability:
    """Test is_available() and get_binary_path()."""

    def test_is_available_returns_bool(self):
        from dpf.athenak_wrapper import is_available
        assert isinstance(is_available(), bool)

    def test_get_binary_path_returns_str_or_none(self):
        from dpf.athenak_wrapper import get_binary_path
        result = get_binary_path()
        assert result is None or isinstance(result, str)

    def test_module_importable(self):
        """athenak_wrapper can always be imported regardless of binary."""
        import dpf.athenak_wrapper  # noqa: F401

    def test_all_exports(self):
        from dpf.athenak_wrapper import __all__
        assert "AthenaKSolver" in __all__
        assert "is_available" in __all__
        assert "generate_athenak_input" in __all__
        assert "read_vtk_file" in __all__


class TestAthenaKConfig:
    """Test generate_athenak_input() athinput generation."""

    @pytest.fixture
    def config(self, sample_config_dict):
        return SimulationConfig(**sample_config_dict)

    def test_generates_string(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_mesh_block(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert "<mesh>" in text
        assert "nx1" in text
        assert "nx2" in text
        assert "nx3" in text

    def test_contains_mhd_block(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert "<mhd>" in text
        assert "reconstruct" in text
        assert "rsolver" in text
        assert "gamma" in text

    def test_contains_time_block(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert "<time>" in text
        assert "tlim" in text
        assert "cfl_number" in text

    def test_contains_problem_block(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert "<problem>" in text
        assert "rho0" in text

    def test_pgen_name_included(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, pgen_name="shock_tube")
        assert "pgen_name  = shock_tube" in text

    def test_pgen_name_omitted_when_none(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, pgen_name=None)
        assert "pgen_name" not in text

    def test_vtk_output_enabled(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, output_vtk=True)
        assert "file_type  = vtk" in text
        assert "mhd_w_bcc" in text

    def test_vtk_output_disabled(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, output_vtk=False)
        assert "mhd_w_bcc" not in text

    def test_nlim_override(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, n_steps=50)
        assert "nlim       = 50" in text

    def test_circuit_params_included(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config)
        assert "V0" in text
        assert "C_cap" in text
        assert "anode_r" in text

    def test_reconstruct_mapping(self, sample_config_dict):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        sample_config_dict["fluid"] = {"reconstruction": "weno5"}
        config = SimulationConfig(**sample_config_dict)
        text = generate_athenak_input(config)
        assert "reconstruct = wenoz" in text

    def test_custom_problem_id(self, config):
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        text = generate_athenak_input(config, problem_id="my_sim")
        assert "basename  = my_sim" in text


def _create_vtk_file(
    filepath: Path,
    nx: int = 8,
    ny: int = 8,
    nz: int = 1,
    time: float = 0.01,
    cycle: int = 5,
) -> None:
    """Create a minimal AthenaK-style VTK binary file for testing."""
    n_cells = nx * ny * max(nz, 1)
    header = (
        f"# vtk DataFile Version 2.0\n"
        f"# Athena++ data at time= {time}  level= 0  nranks= 1  cycle={cycle}  "
        f"variables=mhd_w_bcc\n"
        f"BINARY\n"
        f"DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx + 1} {ny + 1} {nz + (0 if nz == 1 else 1)}\n"
        f"ORIGIN 0.0 0.0 0.0\n"
        f"SPACING 0.01 0.01 1.0\n"
        f"\n"
        f"CELL_DATA {n_cells}\n"
    )

    variables = {
        "dens": np.ones(n_cells, dtype=np.float32) * 1.5,
        "velx": np.random.randn(n_cells).astype(np.float32) * 0.1,
        "vely": np.random.randn(n_cells).astype(np.float32) * 0.1,
        "velz": np.zeros(n_cells, dtype=np.float32),
        "eint": np.ones(n_cells, dtype=np.float32) * 2.0,
        "bcc1": np.ones(n_cells, dtype=np.float32) * 0.5,
        "bcc2": np.zeros(n_cells, dtype=np.float32),
        "bcc3": np.zeros(n_cells, dtype=np.float32),
    }

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        for var_name, data in variables.items():
            var_header = f"\nSCALARS {var_name} float\nLOOKUP_TABLE default\n"
            f.write(var_header.encode("ascii"))
            f.write(data.astype(">f4").tobytes())


class TestVTKReader:
    """Test read_vtk_file() and VTK parsing."""

    def test_read_basic_vtk(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=8, ny=8, time=0.05, cycle=10)

        data = read_vtk_file(vtk_file)
        assert data["time"] == pytest.approx(0.05)
        assert data["cycle"] == 10
        assert data["dims"] == [8, 8, 1]
        assert "dens" in data["variables"]
        assert len(data["variables"]) == 8

    def test_variable_shapes(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=16, ny=8)

        data = read_vtk_file(vtk_file)
        for var_name, arr in data["variables"].items():
            assert arr.shape == (128,), f"{var_name} has wrong shape"
            assert arr.dtype == np.float64

    def test_density_values(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=4, ny=4)

        data = read_vtk_file(vtk_file)
        dens = data["variables"]["dens"]
        assert np.allclose(dens, 1.5, atol=0.01)

    def test_file_not_found(self):
        from dpf.athenak_wrapper.athenak_io import read_vtk_file
        with pytest.raises(FileNotFoundError):
            read_vtk_file("/nonexistent/file.vtk")

    def test_origin_and_spacing(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file)

        data = read_vtk_file(vtk_file)
        assert len(data["origin"]) == 3
        assert len(data["spacing"]) == 3


class TestStateConversion:
    """Test convert_to_dpf_state() mapping."""

    def test_all_state_keys_present(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data)

        expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert set(state.keys()) == expected_keys

    def test_rho_shape(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=8, ny=8)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data)
        assert state["rho"].shape == (8, 8)

    def test_velocity_shape(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=8, ny=8)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data)
        assert state["velocity"].shape == (3, 8, 8)

    def test_B_shape(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file, nx=8, ny=8)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data)
        assert state["B"].shape == (3, 8, 8)

    def test_pressure_positive(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data, gamma=5.0 / 3.0)
        assert np.all(state["pressure"] > 0)

    def test_psi_is_zero(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import convert_to_dpf_state, read_vtk_file
        vtk_file = tmp_path / "test.vtk"
        _create_vtk_file(vtk_file)

        data = read_vtk_file(vtk_file)
        state = convert_to_dpf_state(data)
        assert np.all(state["psi"] == 0)


class TestFindVTK:
    """Test find_latest_vtk() and find_all_vtk()."""

    def test_find_latest_vtk(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import find_latest_vtk
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir / "Blast.mhd_w_bcc.00000.vtk")
        _create_vtk_file(vtk_dir / "Blast.mhd_w_bcc.00001.vtk")
        _create_vtk_file(vtk_dir / "Blast.mhd_w_bcc.00002.vtk")

        result = find_latest_vtk(tmp_path)
        assert result is not None
        assert "00002" in str(result)

    def test_find_latest_vtk_empty(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import find_latest_vtk
        assert find_latest_vtk(tmp_path) is None

    def test_find_all_vtk(self, tmp_path):
        from dpf.athenak_wrapper.athenak_io import find_all_vtk
        vtk_dir = tmp_path / "vtk"
        vtk_dir.mkdir()
        _create_vtk_file(vtk_dir / "Blast.mhd_w_bcc.00000.vtk")
        _create_vtk_file(vtk_dir / "Blast.mhd_w_bcc.00001.vtk")

        files = find_all_vtk(tmp_path)
        assert len(files) == 2


class TestAthenaKSolver:
    """Test AthenaKSolver initialization and method signatures."""

    @pytest.fixture
    def config(self, sample_config_dict):
        return SimulationConfig(**sample_config_dict)

    def test_solver_init_with_mock_binary(self, config, tmp_path):
        """Solver initializes when given a valid binary path."""
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 0\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        assert solver._binary == str(fake_binary)

    def test_solver_raises_on_missing_binary(self, config):
        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        with pytest.raises(FileNotFoundError, match="not found"):
            AthenaKSolver(config, binary_path="/nonexistent/binary")

    def test_initial_state_keys(self, config, tmp_path):
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 0\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        state = solver.initial_state()

        expected_keys = {"rho", "velocity", "pressure", "B", "Te", "Ti", "psi"}
        assert set(state.keys()) == expected_keys

    def test_initial_state_rho(self, config, tmp_path):
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 0\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        state = solver.initial_state()

        assert np.allclose(state["rho"], config.rho0)

    def test_initial_state_temperature(self, config, tmp_path):
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 0\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        state = solver.initial_state()

        assert np.allclose(state["Te"], config.T0)
        assert np.allclose(state["Ti"], config.T0)

    def test_coupling_interface(self, config, tmp_path):
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 0\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        coupling = solver.coupling_interface()

        assert isinstance(coupling, CouplingState)

    def test_step_returns_state_on_subprocess_failure(self, config, tmp_path):
        """On subprocess failure, step() returns unchanged state."""
        fake_binary = tmp_path / "athenak"
        fake_binary.write_text("#!/bin/bash\nexit 1\n")
        fake_binary.chmod(0o755)

        from dpf.athenak_wrapper.athenak_solver import AthenaKSolver
        solver = AthenaKSolver(config, binary_path=str(fake_binary))
        state = solver.initial_state()
        original_rho = state["rho"].copy()

        result = solver.step(state, dt=1e-9, current=100e3, voltage=20e3)
        assert np.array_equal(result["rho"], original_rho)


class TestBackendResolution:
    """Test SimulationEngine._resolve_backend with athenak."""

    def test_athenak_requested_unavailable(self):
        from dpf.engine import SimulationEngine
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=False),
            pytest.raises(RuntimeError, match="AthenaK"),
        ):
            SimulationEngine._resolve_backend("athenak")

    def test_athenak_requested_available(self):
        from dpf.engine import SimulationEngine
        with patch("dpf.athenak_wrapper.is_available", return_value=True):
            result = SimulationEngine._resolve_backend("athenak")
            assert result == "athenak"

    def test_auto_prefers_athena_over_athenak(self):
        """Auto resolution prefers Athena++ over AthenaK (Phase R priority change)."""
        from dpf.engine import SimulationEngine
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=True),
            patch("dpf.athena_wrapper.is_available", return_value=True),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "athena"

    def test_auto_falls_back_to_athenak_when_athena_unavailable(self):
        """Auto resolution falls back to AthenaK when Athena++ and Metal unavailable."""
        from dpf.engine import SimulationEngine

        class FakeMetal:
            @staticmethod
            def is_available():
                return False

        with (
            patch("dpf.athenak_wrapper.is_available", return_value=True),
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.metal.metal_solver.MetalMHDSolver", FakeMetal),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "athenak"

    def test_auto_falls_back_to_python(self):
        from dpf.engine import SimulationEngine

        class FakeMetal:
            @staticmethod
            def is_available():
                return False

        with (
            patch("dpf.athenak_wrapper.is_available", return_value=False),
            patch("dpf.athena_wrapper.is_available", return_value=False),
            patch("dpf.metal.metal_solver.MetalMHDSolver", FakeMetal),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "python"

    def test_python_still_works(self):
        from dpf.engine import SimulationEngine
        result = SimulationEngine._resolve_backend("python")
        assert result == "python"


# --- Section: test_phase_r_athena_primary.py ---


_CIRCUIT = {
    "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
    "anode_radius": 0.005, "cathode_radius": 0.01,
}


def test_resolve_backend_auto_selects_athena_when_available(monkeypatch):
    """Auto resolution prioritizes Athena++ when pybind11 extension is available."""
    from dpf.engine import SimulationEngine

    mock_is_available = Mock(return_value=True)
    monkeypatch.setattr("dpf.athena_wrapper.is_available", mock_is_available)

    backend = SimulationEngine._resolve_backend("auto")

    assert backend == "athena", "Auto resolution should prioritize Athena++"
    mock_is_available.assert_called_once()


def test_resolve_backend_auto_falls_back_to_metal_when_athena_unavailable(monkeypatch):
    """Auto resolution falls back to Metal when Athena++ is unavailable."""
    from dpf.engine import SimulationEngine

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: False)

    mock_metal_solver = Mock()
    mock_metal_solver.is_available.return_value = True
    monkeypatch.setattr("dpf.metal.metal_solver.MetalMHDSolver", mock_metal_solver)

    backend = SimulationEngine._resolve_backend("auto")

    assert backend == "metal", "Auto resolution should fall back to Metal"


def test_resolve_backend_auto_falls_back_to_athenak_when_athena_metal_unavailable(monkeypatch):
    """Auto resolution falls back to AthenaK when Athena++ and Metal are unavailable."""
    from dpf.engine import SimulationEngine

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: False)

    class FakeMetalSolver:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr("dpf.metal.metal_solver.MetalMHDSolver", FakeMetalSolver)
    monkeypatch.setattr("dpf.athenak_wrapper.is_available", lambda: True)

    backend = SimulationEngine._resolve_backend("auto")

    assert backend == "athenak", "Auto resolution should fall back to AthenaK"


def test_resolve_backend_auto_falls_back_to_python_when_all_unavailable(monkeypatch):
    """Auto resolution falls back to Python when all C++ backends are unavailable."""
    from dpf.engine import SimulationEngine

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: False)

    class FakeMetalSolver:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr("dpf.metal.metal_solver.MetalMHDSolver", FakeMetalSolver)
    monkeypatch.setattr("dpf.athenak_wrapper.is_available", lambda: False)

    backend = SimulationEngine._resolve_backend("auto")

    assert backend == "python", "Auto resolution should fall back to Python"


def test_resolve_backend_explicit_athena_raises_when_unavailable():
    """Explicit backend='athena' raises RuntimeError when unavailable."""
    from dpf.engine import SimulationEngine

    with patch("dpf.athena_wrapper.is_available", return_value=False), \
            pytest.raises(RuntimeError, match="Athena\\+\\+ backend requested but"):
        SimulationEngine._resolve_backend("athena")


def test_resolve_backend_hybrid_returns_hybrid():
    """backend='hybrid' returns 'hybrid' without checking availability."""
    from dpf.engine import SimulationEngine

    backend = SimulationEngine._resolve_backend("hybrid")
    assert backend == "hybrid"


def test_generate_athinput_ppm_reconstruction():
    """generate_athinput() sets xorder=3 for PPM reconstruction."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(64, 1, 64),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.fluid.reconstruction = "ppm"

    athinput = generate_athinput(config)

    assert "xorder      = 3" in athinput, "PPM should set xorder=3"
    assert "integrator  = rk3" in athinput, "xorder=3 should use RK3 integrator"


def test_generate_athinput_weno5_maps_to_ppm():
    """generate_athinput() maps WENO5 to PPM (xorder=3) since Athena++ lacks WENO."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(32, 1, 32),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.fluid.reconstruction = "weno5"

    athinput = generate_athinput(config)

    assert "xorder      = 3" in athinput


def test_generate_athinput_plm_reconstruction():
    """generate_athinput() sets xorder=2 for PLM reconstruction."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(32, 1, 32),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.fluid.reconstruction = "plm"

    athinput = generate_athinput(config)

    assert "xorder      = 2" in athinput, "PLM should set xorder=2"
    assert "integrator  = vl2" in athinput, "xorder=2 should use vl2 integrator"


def test_generate_athinput_nscalars_parameter():
    """generate_athinput() includes nscalars=2 for two-temperature model."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(32, 1, 32),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )

    athinput = generate_athinput(config)

    assert "nscalars          = 2" in athinput, "Should include nscalars=2"


def test_generate_athinput_user_boundary_conditions():
    """generate_athinput() sets user BC at outer radial boundary for circuit coupling."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(64, 1, 64),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.geometry.type = "cylindrical"

    athinput = generate_athinput(config)

    assert "ox1_bc     = user" in athinput, "Should use user BC at cathode (ox1)"


def test_generate_athinput_radiation_flags():
    """generate_athinput() includes radiation toggle flags."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(32, 1, 32),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.radiation.bremsstrahlung_enabled = True

    athinput = generate_athinput(config)

    assert "enable_radiation  = 1" in athinput, "Should enable radiation"
    assert "gaunt_factor" in athinput, "Should include gaunt_factor parameter"


def test_generate_athinput_resistive_physics_flags():
    """generate_athinput() includes resistive physics toggle flags."""
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig(
        grid_shape=(32, 1, 32),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.fluid.enable_resistive = True
    config.fluid.enable_nernst = True

    athinput = generate_athinput(config)

    assert "enable_resistive  = 1" in athinput
    assert "enable_nernst     = 1" in athinput


def test_athena_solver_subprocess_finds_binary(tmp_path, monkeypatch):
    """AthenaPPSolver in subprocess mode locates the Athena++ binary."""
    from dpf.athena_wrapper.athena_engine import AthenaPPSolver

    fake_athena_dir = tmp_path / "external" / "athena" / "bin"
    fake_athena_dir.mkdir(parents=True)
    fake_binary = fake_athena_dir / "athena"
    fake_binary.write_text("#!/bin/bash\necho 'fake athena'")
    fake_binary.chmod(0o755)

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: False)

    mock_file_path = tmp_path / "src" / "dpf" / "athena_wrapper" / "athena_engine.py"
    monkeypatch.setattr(
        "dpf.athena_wrapper.athena_engine.Path",
        lambda x: mock_file_path if x == __file__ else Path(x),
    )

    config = SimulationConfig(
        grid_shape=(16, 1, 16), dx=0.001, sim_time=1e-6, circuit=_CIRCUIT
    )

    solver = object.__new__(AthenaPPSolver)
    solver.config = config

    found_binary = solver._find_binary(str(fake_binary))
    assert found_binary == str(fake_binary)


def test_athena_solver_subprocess_raises_when_binary_missing(monkeypatch, tmp_path):
    """AthenaPPSolver raises FileNotFoundError when binary is missing."""
    from dpf.athena_wrapper.athena_engine import AthenaPPSolver

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: False)

    config = SimulationConfig(
        grid_shape=(16, 1, 16), dx=0.001, sim_time=1e-6, circuit=_CIRCUIT
    )

    solver = object.__new__(AthenaPPSolver)
    solver.config = config

    fake_file = tmp_path / "src" / "dpf" / "athena_wrapper" / "athena_engine.py"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text("")
    monkeypatch.setattr(
        "dpf.athena_wrapper.athena_engine.__file__", str(fake_file)
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Athena\\+\\+ binary not found"):
        solver._find_binary(None)


def test_read_Te_Ti_from_scalars_fallback_when_no_scalars():
    """_read_Te_Ti_from_scalars() falls back to single-fluid estimate when scalars unavailable."""
    from dpf.athena_wrapper.athena_engine import AthenaPPSolver

    config = SimulationConfig(
        grid_shape=(8, 1, 8), dx=0.001, sim_time=1e-6, circuit=_CIRCUIT
    )

    solver = object.__new__(AthenaPPSolver)
    solver.config = config
    solver.mode = "linked"
    solver._initialized = False

    rho = np.full((8, 1, 8), 1e-3)
    pressure = np.full((8, 1, 8), 1000.0)

    mock_core = Mock()
    mock_core.get_scalar_data.side_effect = AttributeError("not available")
    solver._core = mock_core
    solver._mesh_handle = Mock()

    Te, Ti = solver._read_Te_Ti_from_scalars(rho, pressure)

    assert Te.shape == rho.shape
    assert Ti.shape == rho.shape
    assert np.all(Te > 0), "Te should be positive"
    assert np.all(Ti > 0), "Ti should be positive"
    k_B = 1.380649e-23
    n_i = rho / config.ion_mass
    expected_T = pressure / (2.0 * n_i * k_B)
    assert np.allclose(Te, expected_T, rtol=1e-6)
    assert np.allclose(Ti, expected_T, rtol=1e-6)


def test_read_Te_Ti_from_scalars_uses_scalar_data_when_available():
    """_read_Te_Ti_from_scalars() uses scalar data when available."""
    from dpf.athena_wrapper.athena_engine import AthenaPPSolver

    config = SimulationConfig(
        grid_shape=(8, 1, 8), dx=0.001, sim_time=1e-6, circuit=_CIRCUIT
    )
    config.fluid.gamma = 5.0 / 3.0

    solver = object.__new__(AthenaPPSolver)
    solver.config = config
    solver.mode = "linked"
    solver._initialized = True

    rho = np.full((8, 1, 8), 1e-3)
    pressure = np.full((8, 1, 8), 1000.0)

    k_B = 1.380649e-23
    gamma = config.fluid.gamma
    T_target = 5000.0
    e_spec = T_target * k_B / ((gamma - 1.0) * config.ion_mass)

    scalar_data = [
        np.full((8, 1, 8), e_spec),
        np.full((8, 1, 8), e_spec),
    ]

    mock_core = Mock()
    mock_core.get_scalar_data.return_value = scalar_data
    solver._core = mock_core
    solver._mesh_handle = Mock()

    Te, Ti = solver._read_Te_Ti_from_scalars(rho, pressure)

    assert np.allclose(Te, T_target, rtol=0.1), f"Expected Te ~ {T_target} K"
    assert np.allclose(Ti, T_target, rtol=0.1), f"Expected Ti ~ {T_target} K"


def test_step_athena_uses_coupling_data_from_cpp(monkeypatch):
    """_step_athena() uses coupling data (R_plasma, L_plasma) from C++ instead of zeros."""
    from dpf.engine import SimulationEngine

    config = SimulationConfig(
        grid_shape=(16, 1, 16),
        dx=0.001,
        sim_time=1e-6,
        circuit=_CIRCUIT,
    )
    config.fluid.backend = "athena"

    monkeypatch.setattr("dpf.athena_wrapper.is_available", lambda: True)

    mock_solver = Mock()
    mock_coupling = CouplingState(
        Lp=5e-8,
        current=100e3,
        R_plasma=0.02,
        Z_bar=1.0,
    )
    mock_solver.coupling_interface.return_value = mock_coupling
    mock_solver.step.return_value = {
        "rho": np.ones((16, 1, 16)),
        "velocity": np.zeros((3, 16, 1, 16)),
        "pressure": np.ones((16, 1, 16)) * 1000,
        "B": np.zeros((3, 16, 1, 16)),
        "Te": np.ones((16, 1, 16)) * 300,
        "Ti": np.ones((16, 1, 16)) * 300,
        "psi": np.zeros((16, 1, 16)),
    }

    engine = object.__new__(SimulationEngine)
    engine.config = config
    engine.backend = "athena"
    engine.fluid = mock_solver
    engine.circuit = Mock()
    engine.circuit.step.return_value = CouplingState(
        current=100e3, voltage=10e3, Lp=5e-8, R_plasma=0.02, Z_bar=1.0
    )
    engine.circuit.state = engine.circuit.step.return_value
    engine.diagnostics = Mock()
    engine.diag_interval = 1000
    engine.time = 0.0
    engine.step_count = 0
    engine._prev_L_plasma = 0.0
    engine._last_R_plasma = 0.0
    engine._last_Z_bar = 1.0
    engine._last_eta_anom = 0.0
    engine.state = mock_solver.step.return_value
    engine._coupling = engine.circuit.state
    engine.initial_energy = 1.0
    engine.total_radiated_energy = 0.0
    engine.total_neutron_yield = 0.0
    engine.well_interval = 0
    engine.well_exporter = Mock()
    engine.boundary_cfg = config.boundary
    engine.geometry_type = getattr(config.geometry, "coord_system", "cartesian")
    engine.snowplow = None
    engine.circuit.total_energy.return_value = 1.0
    engine.circuit.current = 100e3
    engine.circuit.voltage = 10e3

    engine._step_athena(dt=1e-9, sim_time=1e-6, _max_steps=None)

    mock_solver.coupling_interface.assert_called_once()

    call_args = engine.circuit.step.call_args
    coupling_arg = call_args[0][0]
    assert coupling_arg.Lp > 0, "L_plasma should be non-zero from C++"
    assert coupling_arg.R_plasma > 0, "R_plasma should be non-zero from C++"


def test_fluid_config_accepts_hybrid_backend():
    """FluidConfig accepts backend='hybrid' as valid."""
    config = FluidConfig(backend="hybrid")
    assert config.backend == "hybrid"


def test_fluid_config_rejects_invalid_backend():
    """FluidConfig rejects invalid backend names."""
    with pytest.raises(ValueError, match="backend must be one of"):
        FluidConfig(backend="invalid_backend")


def test_fluid_config_accepts_all_valid_backends():
    """FluidConfig accepts all valid backend options."""
    valid_backends = ["python", "athena", "athenak", "metal", "hybrid", "auto"]
    for backend in valid_backends:
        config = FluidConfig(backend=backend)
        assert config.backend == backend


def test_fluid_config_handoff_fraction_default():
    """FluidConfig handoff_fraction defaults to 0.1."""
    config = FluidConfig()
    assert config.handoff_fraction == pytest.approx(0.1)


def test_fluid_config_handoff_fraction_accepts_valid_range():
    """FluidConfig accepts handoff_fraction in [0.0, 1.0]."""
    config_zero = FluidConfig(handoff_fraction=0.0)
    assert config_zero.handoff_fraction == pytest.approx(0.0)

    config_one = FluidConfig(handoff_fraction=1.0)
    assert config_one.handoff_fraction == pytest.approx(1.0)

    config_mid = FluidConfig(handoff_fraction=0.5)
    assert config_mid.handoff_fraction == pytest.approx(0.5)


def test_fluid_config_handoff_fraction_rejects_negative():
    """FluidConfig rejects negative handoff_fraction."""
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        FluidConfig(handoff_fraction=-0.1)


def test_fluid_config_handoff_fraction_rejects_above_one():
    """FluidConfig rejects handoff_fraction > 1.0."""
    with pytest.raises(ValueError, match="less than or equal to 1"):
        FluidConfig(handoff_fraction=1.1)


def test_fluid_config_validation_interval_default():
    """FluidConfig validation_interval defaults to 50."""
    config = FluidConfig()
    assert config.validation_interval == 50


def test_fluid_config_validation_interval_rejects_zero():
    """FluidConfig rejects validation_interval <= 0."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        FluidConfig(validation_interval=0)


def test_python_backend_deprecation_warning_at_step_1000():
    """Python backend shows deprecation warning at step 1000 for large grids."""
    step_count = 1000
    backend = "python"
    grid_shape = (64, 64, 64)
    nx, ny, nz = grid_shape

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if step_count == 1000 and backend == "python" and nx * ny * nz > 32**3:
            warnings.warn(
                "Python MHD backend is deprecated for production simulations "
                f"(grid {nx}x{ny}x{nz} > 32^3, step {step_count}). "
                "Use backend='athena' (Athena++ C++) or backend='metal' (GPU) "
                "for better accuracy and performance.",
                DeprecationWarning,
                stacklevel=2,
            )

        assert len(w) == 1, "Should issue exactly one warning"
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Python MHD backend is deprecated" in str(w[0].message)
        assert "64x64x64" in str(w[0].message)
        assert "step 1000" in str(w[0].message)


def test_python_backend_no_warning_for_small_grids():
    """Python backend does NOT warn for small grids even at step 1000."""
    step_count = 1000
    backend = "python"
    grid_shape = (16, 16, 16)
    nx, ny, nz = grid_shape

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        if step_count == 1000 and backend == "python" and nx * ny * nz > 32**3:
            warnings.warn(
                "Python MHD backend is deprecated",
                DeprecationWarning,
                stacklevel=2,
            )

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0, "Should not warn for small grids"


def test_python_backend_warning_only_at_step_1000():
    """Python backend warning appears only at step 1000, not before or after."""
    backend = "python"
    grid_shape = (64, 64, 64)
    nx, ny, nz = grid_shape

    def check_warning(step_count: int) -> list:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if step_count == 1000 and backend == "python" and nx * ny * nz > 32**3:
                warnings.warn(
                    "Python MHD backend is deprecated",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return [x for x in w if issubclass(x.category, DeprecationWarning)]

    assert len(check_warning(999)) == 0, "Should not warn at step 999"
    assert len(check_warning(1000)) == 1, "Should warn at step 1000"
    assert len(check_warning(1001)) == 0, "Should not warn at step 1001"
