"""Phase J.1 — AthenaK backend integration tests.

Tests cover:
- FluidConfig validation with "athenak" backend
- AthenaK availability detection
- Config → athinput generation
- VTK file reading and parsing
- State conversion (AthenaK → DPF)
- AthenaKSolver initialization and subprocess mocking
- Backend resolution in SimulationEngine
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from dpf.config import FluidConfig, SimulationConfig

# ── FluidConfig validation ──────────────────────────────────────


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


# ── AthenaK availability ────────────────────────────────────────


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


# ── Config generation ───────────────────────────────────────────


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


# ── VTK I/O ─────────────────────────────────────────────────────


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
            # Big-endian float32
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


# ── AthenaKSolver ───────────────────────────────────────────────


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

        from dpf.core.bases import CouplingState
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


# ── Backend resolution ──────────────────────────────────────────


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

    def test_auto_prefers_athenak(self):
        from dpf.engine import SimulationEngine
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=True),
            patch("dpf.athena_wrapper.is_available", return_value=True),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "athenak"

    def test_auto_falls_back_to_athena(self):
        from dpf.engine import SimulationEngine
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=False),
            patch("dpf.athena_wrapper.is_available", return_value=True),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "athena"

    def test_auto_falls_back_to_python(self):
        from dpf.engine import SimulationEngine
        with (
            patch("dpf.athenak_wrapper.is_available", return_value=False),
            patch("dpf.athena_wrapper.is_available", return_value=False),
        ):
            result = SimulationEngine._resolve_backend("auto")
            assert result == "python"

    def test_python_still_works(self):
        from dpf.engine import SimulationEngine
        result = SimulationEngine._resolve_backend("python")
        assert result == "python"
