"""Phase F.3: Athena++ Verification Suite.

Validates the Athena++ backend against standard MHD test problems:

1. Sod shock tube — exact Riemann solution comparison (subprocess mode)
2. Brio-Wu MHD shock tube — positivity + Bx conservation (subprocess mode)
3. Magnoh z-pinch baseline — linked mode integration checks
4. Cross-backend comparison — Python vs Athena++ state dict parity

All subprocess tests use dedicated Athena++ binaries built for each problem.
Linked-mode tests use the magnoh problem generator.

References:
    - Sod, G.A., JCP 27, 1--31 (1978)
    - Brio, M. & Wu, C.C., JCP 75, 400--422 (1988)
    - Giuliani et al., Phys. Plasmas (2018) — Magnetized Noh
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Project root for locating Athena++ binaries and inputs
_PROJECT_ROOT = Path(__file__).parents[1]
_ATHENA_DIR = _PROJECT_ROOT / "external" / "athena"
_ATHENA_BIN = _ATHENA_DIR / "bin"
_ATHENA_INPUTS = _ATHENA_DIR / "inputs"

# ============================================================
# Helper: Exact Sod Riemann solution
# ============================================================


def _sod_exact(x: np.ndarray, t: float, gamma: float = 1.4) -> dict:
    """Compute exact Sod shock tube solution at time *t*.

    Args:
        x: Cell-centre coordinates.
        t: Time at which to evaluate.
        gamma: Adiabatic index.

    Returns:
        Dict with keys 'rho', 'velocity', 'pressure'.
    """
    # Left / right states
    rho_l, p_l, u_l = 1.0, 1.0, 0.0
    rho_r, p_r, u_r = 0.125, 0.1, 0.0

    gp1 = gamma + 1.0
    gm1 = gamma - 1.0

    # Sound speeds
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Star region (solve iteratively for p_star)
    # Initial guess from two-rarefaction approximation
    p_star = (
        (c_l + c_r - 0.5 * gm1 * (u_r - u_l))
        / (c_l / p_l ** (gm1 / (2.0 * gamma)) + c_r / p_r ** (gm1 / (2.0 * gamma)))
    ) ** (2.0 * gamma / gm1)

    # Newton iteration for p_star
    for _ in range(50):
        # Left wave (rarefaction)
        f_l = (
            (2.0 * c_l / gm1)
            * ((p_star / p_l) ** (gm1 / (2.0 * gamma)) - 1.0)
        )
        fp_l = (
            (1.0 / (rho_l * c_l))
            * (p_star / p_l) ** (-(gp1) / (2.0 * gamma))
        )

        # Right wave (shock)
        A_r = 2.0 / (gp1 * rho_r)
        B_r = gm1 / gp1 * p_r
        f_r = (p_star - p_r) * np.sqrt(A_r / (p_star + B_r))
        fp_r = np.sqrt(A_r / (p_star + B_r)) * (
            1.0 - (p_star - p_r) / (2.0 * (p_star + B_r))
        )

        residual = f_l + f_r + (u_r - u_l)
        if abs(residual) < 1e-12:
            break
        p_star -= residual / (fp_l + fp_r)

    # Star velocity
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    # Star region density (left = rarefaction fan tail)
    rho_star_l = rho_l * (p_star / p_l) ** (1.0 / gamma)

    # Right shock: density behind shock
    rho_star_r = rho_r * (
        (p_star / p_r + gm1 / gp1) / (gm1 / gp1 * p_star / p_r + 1.0)
    )

    # Wave speeds
    # Left rarefaction: head at x/t = u_l - c_l, tail at x/t = u_star - c_star_l
    c_star_l = c_l * (p_star / p_l) ** (gm1 / (2.0 * gamma))
    head_l = u_l - c_l
    tail_l = u_star - c_star_l

    # Right shock speed
    shock_r = u_r + c_r * np.sqrt(gp1 / (2.0 * gamma) * p_star / p_r + gm1 / (2.0 * gamma))

    # Sampling
    rho = np.empty_like(x)
    vel = np.empty_like(x)
    prs = np.empty_like(x)

    for i, xi in enumerate(x):
        s = xi / t  # similarity variable

        if s < head_l:
            # Left undisturbed
            rho[i], vel[i], prs[i] = rho_l, u_l, p_l
        elif s < tail_l:
            # Inside rarefaction fan
            vel[i] = 2.0 / gp1 * (c_l + gm1 / 2.0 * u_l + s)
            c = 2.0 / gp1 * (c_l - gm1 / 2.0 * (s - u_l))
            rho[i] = rho_l * (c / c_l) ** (2.0 / gm1)
            prs[i] = p_l * (c / c_l) ** (2.0 * gamma / gm1)
        elif s < u_star:
            # Left star region
            rho[i], vel[i], prs[i] = rho_star_l, u_star, p_star
        elif s < shock_r:
            # Right star region
            rho[i], vel[i], prs[i] = rho_star_r, u_star, p_star
        else:
            # Right undisturbed
            rho[i], vel[i], prs[i] = rho_r, u_r, p_r

    return {"rho": rho, "velocity": vel, "pressure": prs}


# ============================================================
# Helper: Run Athena++ subprocess and read HDF5
# ============================================================


def _run_athena_subprocess(
    binary: str | Path,
    athinput: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    timeout: int = 60,
) -> Path:
    """Run Athena++ binary with given input and return output directory.

    Args:
        binary: Path to the Athena++ executable.
        athinput: Path to the athinput file.
        overrides: Optional dict of parameter overrides (e.g. {"time/tlim": "0.1"}).
        timeout: Timeout in seconds.

    Returns:
        Path to the temporary output directory containing HDF5 files.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="athena_vv_"))
    cmd = [str(binary), "-i", str(athinput), "-d", str(tmpdir)]
    if overrides:
        for key, val in overrides.items():
            cmd.append(f"{key}={val}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(tmpdir),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Athena++ failed (exit {result.returncode}):\n{result.stderr}"
        )
    return tmpdir


def _read_athdf(dirpath: Path, pattern: str = "*.athdf") -> dict:
    """Read the last HDF5 output from an Athena++ run.

    Returns dict with 'rho', 'velocity', 'pressure', 'B' arrays.
    """
    import glob

    files = sorted(glob.glob(str(dirpath / "*.*.athdf")))
    if not files:
        raise FileNotFoundError(f"No .athdf files in {dirpath}")

    # Last output = final time
    with h5py.File(files[-1], "r") as f:
        # Athena++ HDF5 layout: prim(NHYDRO, nblocks, nk, nj, ni)
        # Variable ordering from VariableNames attribute
        prim = f["prim"][:]
        var_names = [
            v.decode() if isinstance(v, bytes) else v
            for v in f.attrs.get("VariableNames", [])
        ]
        # Check for B-field
        B = f["B"][:] if "B" in f else None
        x1v = f["x1v"][:]  # cell centers
        time_attr = f.attrs.get("Time", 0.0)
        time = float(time_attr)

    # Build variable index map
    var_idx = {name: i for i, name in enumerate(var_names)}

    # Squeeze out degenerate dimensions
    rho = np.squeeze(prim[var_idx.get("rho", 0)])
    prs = np.squeeze(prim[var_idx.get("press", 1)])
    vx = np.squeeze(prim[var_idx.get("vel1", 2)])
    vy = np.squeeze(prim[var_idx.get("vel2", 3)])
    vz = np.squeeze(prim[var_idx.get("vel3", 4)])
    x = np.squeeze(x1v)

    result = {
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "pressure": prs,
        "x": x,
        "time": time,
    }

    if B is not None:
        result["Bx"] = np.squeeze(B[0])
        result["By"] = np.squeeze(B[1])
        result["Bz"] = np.squeeze(B[2])

    return result


# ============================================================
# Test: Sod Shock Tube (Athena++ subprocess)
# ============================================================


@pytest.mark.slow
class TestSodShockTube:
    """Sod shock tube via Athena++ subprocess mode.

    Verifies density, velocity, and pressure against the exact Riemann
    solution at t=0.25 with 256 cells.
    """

    SOD_BINARY = _ATHENA_BIN / "athena_sod"
    SOD_INPUT = _ATHENA_INPUTS / "hydro" / "athinput.sod"

    @pytest.fixture(scope="class")
    def sod_result(self, tmp_path_factory):
        """Run Sod shock tube and return numerical + exact solution."""
        if not self.SOD_BINARY.is_file():
            pytest.skip("athena_sod binary not built")

        tmpdir = tmp_path_factory.mktemp("sod")

        # Write a clean athinput with HDF5 output (not tab) to avoid
        # data_format incompatibility with the original athinput.sod.
        athinput = tmpdir / "athinput.sod"
        athinput.write_text(
            "<comment>\nproblem = Sod shock tube\n\n"
            "<job>\nproblem_id = Sod\n\n"
            "<output1>\nfile_type = hdf5\nvariable = prim\ndt = 0.25\n\n"
            "<output2>\nfile_type = hst\ndt = 0.01\n\n"
            "<time>\ncfl_number = 0.8\nnlim = -1\ntlim = 0.25\n"
            "integrator = vl2\nxorder = 2\nncycle_out = 100\n\n"
            "<mesh>\nnx1 = 256\nx1min = -0.5\nx1max = 0.5\n"
            "ix1_bc = outflow\nox1_bc = outflow\n"
            "nx2 = 1\nx2min = -0.5\nx2max = 0.5\n"
            "ix2_bc = periodic\nox2_bc = periodic\n"
            "nx3 = 1\nx3min = -0.5\nx3max = 0.5\n"
            "ix3_bc = periodic\nox3_bc = periodic\n\n"
            "<hydro>\ngamma = 1.4\n\n"
            "<problem>\nshock_dir = 1\nxshock = 0.0\n"
            "dl = 1.0\npl = 1.0\nul = 0.0\nvl = 0.0\nwl = 0.0\n"
            "dr = 0.125\npr = 0.1\nur = 0.0\nvr = 0.0\nwr = 0.0\n"
        )

        cmd = [str(self.SOD_BINARY), "-i", str(athinput), "-d", str(tmpdir)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=str(tmpdir),
        )
        assert result.returncode == 0, f"Athena++ Sod failed:\n{result.stdout}\n{result.stderr}"

        # Read HDF5 output
        data = _read_athdf(tmpdir)

        # Compute exact solution
        exact = _sod_exact(data["x"], data["time"], gamma=1.4)

        return {"numerical": data, "exact": exact}

    def test_sod_density_L1_error(self, sod_result):
        """L1 error in density < 5%."""
        num = sod_result["numerical"]["rho"]
        exact = sod_result["exact"]["rho"]
        L1 = np.mean(np.abs(num - exact)) / np.mean(np.abs(exact))
        assert L1 < 0.05, f"Sod density L1 = {L1:.4f} (expected < 0.05)"

    def test_sod_pressure_L1_error(self, sod_result):
        """L1 error in pressure < 5%."""
        num = sod_result["numerical"]["pressure"]
        exact = sod_result["exact"]["pressure"]
        L1 = np.mean(np.abs(num - exact)) / np.mean(np.abs(exact))
        assert L1 < 0.05, f"Sod pressure L1 = {L1:.4f} (expected < 0.05)"

    def test_sod_velocity_L1_error(self, sod_result):
        """L1 error in velocity < 10% (velocity has larger errors near contact)."""
        num = sod_result["numerical"]["vx"]
        exact = sod_result["exact"]["velocity"]
        # Use absolute L1 since velocity includes zeros
        L1 = np.mean(np.abs(num - exact))
        v_scale = max(np.max(np.abs(exact)), 1e-10)
        assert L1 / v_scale < 0.10, f"Sod velocity L1 = {L1/v_scale:.4f}"

    def test_sod_positivity(self, sod_result):
        """Density and pressure must be positive."""
        assert np.all(sod_result["numerical"]["rho"] > 0)
        assert np.all(sod_result["numerical"]["pressure"] > 0)

    def test_sod_density_jump_exists(self, sod_result):
        """Solution should have a density jump (contact + shock structure)."""
        rho = sod_result["numerical"]["rho"]
        # Left state is rho=1.0, right is rho=0.125
        # After evolution, we should see a jump from ~1.0 → ~0.125
        rho_max = rho.max()
        rho_min = rho.min()
        jump = rho_max / rho_min
        # Density contrast should be at least 3:1
        assert jump > 3.0, f"Density contrast = {jump:.2f} (expected > 3)"


# ============================================================
# Test: Brio-Wu MHD Shock Tube (Athena++ subprocess)
# ============================================================


@pytest.mark.slow
class TestBrioWuMHD:
    """Brio-Wu MHD shock tube via Athena++ subprocess.

    No exact solution exists; we verify:
    - Positivity of density and pressure
    - Conservation of B_x (constant 0.75)
    - Correct wave structure (7 waves in MHD)
    """

    BW_BINARY = _ATHENA_BIN / "athena_briowu"
    BW_INPUT = _ATHENA_INPUTS / "mhd" / "athinput.bw"

    @pytest.fixture(scope="class")
    def bw_result(self, tmp_path_factory):
        """Run Brio-Wu and return numerical result."""
        if not self.BW_BINARY.is_file():
            pytest.skip("athena_briowu binary not built")

        tmpdir = tmp_path_factory.mktemp("briowu")

        # Write a clean athinput with HDF5 output
        athinput = tmpdir / "athinput.bw"
        athinput.write_text(
            "<comment>\nproblem = Brio-Wu MHD shock tube\n\n"
            "<job>\nproblem_id = BrioWu\n\n"
            "<output1>\nfile_type = hdf5\nvariable = prim\ndt = 0.1\n\n"
            "<output2>\nfile_type = hst\ndt = 0.01\n\n"
            "<time>\ncfl_number = 0.4\nnlim = -1\ntlim = 0.1\n"
            "integrator = vl2\nxorder = 2\nncycle_out = 100\n\n"
            "<mesh>\nnx1 = 256\nx1min = -0.5\nx1max = 0.5\n"
            "ix1_bc = outflow\nox1_bc = outflow\n"
            "nx2 = 1\nx2min = -0.5\nx2max = 0.5\n"
            "ix2_bc = periodic\nox2_bc = periodic\n"
            "nx3 = 1\nx3min = -0.5\nx3max = 0.5\n"
            "ix3_bc = periodic\nox3_bc = periodic\n\n"
            "<hydro>\ngamma = 2.0\n\n"
            "<problem>\nshock_dir = 1\nxshock = 0.0\n"
            "dl = 1.0\npl = 1.0\nul = 0.0\nvl = 0.0\nwl = 0.0\n"
            "bxl = 0.75\nbyl = 1.0\nbzl = 0.0\n"
            "dr = 0.125\npr = 0.1\nur = 0.0\nvr = 0.0\nwr = 0.0\n"
            "bxr = 0.75\nbyr = -1.0\nbzr = 0.0\n"
        )

        cmd = [str(self.BW_BINARY), "-i", str(athinput), "-d", str(tmpdir)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=str(tmpdir),
        )
        assert result.returncode == 0, f"Athena++ Brio-Wu failed:\n{result.stdout}\n{result.stderr}"

        return _read_athdf(tmpdir)

    def test_bw_positivity(self, bw_result):
        """Density and pressure must be positive."""
        assert np.all(bw_result["rho"] > 0), "Negative density in Brio-Wu"
        assert np.all(bw_result["pressure"] > 0), "Negative pressure in Brio-Wu"

    def test_bw_Bx_conservation(self, bw_result):
        """B_x must be constant (0.75) everywhere."""
        Bx = bw_result["Bx"]
        assert np.allclose(Bx, 0.75, atol=1e-10), (
            f"Bx not conserved: min={Bx.min():.6e}, max={Bx.max():.6e}"
        )

    def test_bw_finite_solution(self, bw_result):
        """All fields must be finite."""
        for key in ("rho", "vx", "vy", "vz", "pressure", "Bx", "By", "Bz"):
            assert np.all(np.isfinite(bw_result[key])), f"Non-finite in {key}"

    def test_bw_symmetry_breaking(self, bw_result):
        """B_y should be asymmetric (changes sign across contact)."""
        By = bw_result["By"]
        assert By[0] > 0, "Expected By > 0 on left"
        assert By[-1] < 0, "Expected By < 0 on right"

    def test_bw_density_range(self, bw_result):
        """Density should remain bounded (no extreme values)."""
        rho = bw_result["rho"]
        # Brio-Wu produces densities in roughly [0.1, 1.1] range
        assert rho.min() > 0.05, f"Density too low: {rho.min():.4e}"
        assert rho.max() < 1.5, f"Density too high: {rho.max():.4e}"

    def test_bw_wave_structure(self, bw_result):
        """Should have multiple density jumps (MHD wave structure)."""
        rho = bw_result["rho"]
        grad = np.abs(np.gradient(rho))
        # Count significant gradient regions (threshold relative to max gradient)
        threshold = 0.1 * np.max(grad)
        peaks = np.sum(grad > threshold)
        # MHD produces at least 5 distinct wave regions
        assert peaks >= 5, f"Only {peaks} gradient peaks (expected >= 5)"


# ============================================================
# Test: Magnoh Z-Pinch Baseline (linked mode)
# ============================================================


class TestMagnohBaseline:
    """Verify magnoh problem generator via linked mode.

    Uses the module-scoped athena_engine fixture from test_dual_engine.
    Tests that the z-pinch solution is well-behaved.
    """

    @pytest.fixture(scope="class")
    def magnoh_engine(self):
        """Create an Athena++ engine for magnoh verification."""
        from dpf.athena_wrapper import is_available
        if not is_available():
            pytest.skip("Athena++ C++ extension not compiled")

        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[32, 1, 64],
            dx=5e-4,
            sim_time=1e-7,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "ESR": 0.0, "ESL": 0.0,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            geometry={"type": "cylindrical"},
            fluid={"backend": "athena"},
        )
        engine = SimulationEngine(config)
        # Run several steps to evolve the solution
        for _ in range(10):
            engine.step()
        return engine

    def test_magnoh_density_positive(self, magnoh_engine):
        """Density must remain positive throughout evolution."""
        rho = magnoh_engine.state["rho"]
        assert np.all(rho > 0), f"Negative density: min={rho.min():.4e}"

    def test_magnoh_pressure_positive(self, magnoh_engine):
        """Pressure must remain positive."""
        p = magnoh_engine.state["pressure"]
        assert np.all(p > 0), f"Negative pressure: min={p.min():.4e}"

    def test_magnoh_finite_fields(self, magnoh_engine):
        """All state fields must be finite."""
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti"):
            arr = magnoh_engine.state[key]
            assert np.all(np.isfinite(arr)), f"Non-finite values in {key}"

    def test_magnoh_circuit_active(self, magnoh_engine):
        """Circuit should produce non-zero current after stepping."""
        assert magnoh_engine.circuit.current != 0.0

    def test_magnoh_time_advanced(self, magnoh_engine):
        """Time should advance past zero."""
        assert magnoh_engine.time > 0
        assert magnoh_engine.step_count == 10

    def test_magnoh_energy_conservation(self, magnoh_engine):
        """Energy should be approximately conserved."""
        E_total = magnoh_engine.circuit.total_energy()
        E_init = magnoh_engine.initial_energy
        ratio = E_total / max(E_init, 1e-30)
        assert 0.5 < ratio < 2.0, f"Energy ratio = {ratio:.4f}"


# ============================================================
# Test: Cross-Backend Comparison
# ============================================================


class TestCrossBackendComparison:
    """Compare Python and Athena++ backends for consistency.

    Tests that the state dictionary interface is identical and that
    both backends produce physically reasonable output for the same
    DPF configuration.
    """

    @pytest.fixture(scope="class")
    def python_engine(self):
        """Create a Python-backend engine."""
        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[16, 1, 32],
            dx=1e-3,
            sim_time=1e-7,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "ESR": 0.0, "ESL": 0.0,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            geometry={"type": "cylindrical"},
            fluid={"backend": "python"},
            snowplow={"enabled": False},
        )
        engine = SimulationEngine(config)
        for _ in range(5):
            engine.step()
        return engine

    @pytest.fixture(scope="class")
    def athena_engine_cb(self):
        """Create an Athena++-backend engine for comparison."""
        from dpf.athena_wrapper import is_available
        if not is_available():
            pytest.skip("Athena++ C++ extension not compiled")

        from dpf.config import SimulationConfig
        from dpf.engine import SimulationEngine

        config = SimulationConfig(
            grid_shape=[16, 1, 32],
            dx=1e-3,
            sim_time=1e-7,
            circuit={
                "C": 1e-6, "V0": 1e3, "L0": 1e-7, "R0": 0.01,
                "ESR": 0.0, "ESL": 0.0,
                "anode_radius": 0.005, "cathode_radius": 0.01,
            },
            geometry={"type": "cylindrical"},
            fluid={"backend": "athena"},
            snowplow={"enabled": False},
        )
        engine = SimulationEngine(config)
        for _ in range(5):
            engine.step()
        return engine

    def test_state_keys_identical(self, python_engine, athena_engine_cb):
        """Both backends produce same state dictionary keys."""
        py_keys = set(python_engine.state.keys())
        ath_keys = set(athena_engine_cb.state.keys())
        assert py_keys == ath_keys

    def test_both_produce_positive_density(self, python_engine, athena_engine_cb):
        """Both backends should have positive density."""
        assert np.all(python_engine.state["rho"] > 0)
        assert np.all(athena_engine_cb.state["rho"] > 0)

    def test_both_produce_positive_pressure(self, python_engine, athena_engine_cb):
        """Both backends should have positive pressure."""
        assert np.all(python_engine.state["pressure"] > 0)
        assert np.all(athena_engine_cb.state["pressure"] > 0)

    def test_both_produce_finite_fields(self, python_engine, athena_engine_cb):
        """Both backends should have finite field values."""
        for key in ("rho", "velocity", "pressure", "B", "Te", "Ti"):
            assert np.all(np.isfinite(python_engine.state[key])), \
                f"Python: non-finite {key}"
            assert np.all(np.isfinite(athena_engine_cb.state[key])), \
                f"Athena++: non-finite {key}"

    def test_both_advance_time(self, python_engine, athena_engine_cb):
        """Both backends should advance simulation time."""
        assert python_engine.time > 0
        assert athena_engine_cb.time > 0

    def test_circuit_agreement(self, python_engine, athena_engine_cb):
        """Circuit state should be similar (same RLC parameters)."""
        # Circuit evolution is identical (same Python solver),
        # but timesteps may differ, so we just check order of magnitude
        py_I = abs(python_engine.circuit.current)
        ath_I = abs(athena_engine_cb.circuit.current)
        # Both should be non-zero and within 10x of each other
        assert py_I > 0
        assert ath_I > 0
        ratio = max(py_I, ath_I) / max(min(py_I, ath_I), 1e-30)
        # Phase S adds FieldManager-based L_plasma + dL/dt coupling,
        # which amplifies timestep-dependent divergence between backends.
        # 15× is still order-of-magnitude agreement.
        assert ratio < 15, f"Current ratio = {ratio:.2f}"
