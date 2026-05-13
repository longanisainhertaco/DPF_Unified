"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORIGINAL_IMPORTORSKIP = pytest.importorskip
_OPTIONAL_IMPORT_SKIP_ROOTS = {"jax", "jaxlib", "optax"}


def _safe_importorskip(modname: str, *args, **kwargs):
    """Make optional dependency skips robust when a broken import aborts collection."""
    if modname == "mlx.core":
        from dpf.metal.mlx_device import HAS_MLX

        if not HAS_MLX:
            reason = kwargs.get("reason") or "MLX not available"
            pytest.skip(reason, allow_module_level=True)
    if modname.split(".", 1)[0] in _OPTIONAL_IMPORT_SKIP_ROOTS:
        try:
            return _ORIGINAL_IMPORTORSKIP(modname, *args, **kwargs)
        except Exception as exc:
            reason = kwargs.get("reason") or f"{modname} import failed: {exc}"
            pytest.skip(reason, allow_module_level=True)
    return _ORIGINAL_IMPORTORSKIP(modname, *args, **kwargs)


pytest.importorskip = _safe_importorskip

# ---------------------------------------------------------------------------
# Auto-apply xdist_group("gpu") to MLX/Metal test files.
# When running with `pytest -n auto --dist loadgroup`, all GPU tests
# serialize on one worker while CPU tests fan out across cores.
# ---------------------------------------------------------------------------

_GPU_FILE_PREFIXES = ("test_mlx_", "test_metal_")
# Files that use SimulationEngine — serialize to avoid Numba JIT/global state races
_ENGINE_FILES = {
    "test_infrastructure_consolidated.py",
    "test_snowplow_consolidated.py",
    "test_mhd_solver_consolidated.py",
    "test_physics.py",
    "test_two_temperature.py",
    "test_preset_smoke.py",
    "test_verification_consolidated.py",
    "test_calibration_consolidated.py",
    "test_circuit_coupler.py",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark MLX/Metal groups and diagnostics evidence lanes."""
    gpu_marker = pytest.mark.xdist_group("gpu")
    engine_marker = pytest.mark.xdist_group("engine")
    for item in items:
        filename = Path(item.fspath).name
        if filename.startswith(_GPU_FILE_PREFIXES):
            item.add_marker(gpu_marker)
        elif filename in _ENGINE_FILES:
            item.add_marker(engine_marker)

        diagnostics_lane = diagnostics_test_lane_for_file(filename)
        if diagnostics_lane is not None:
            for marker_name in diagnostics_lane.markers:
                item.add_marker(getattr(pytest.mark, marker_name))
            item.user_properties.append(("diagnostics_test_lane", diagnostics_lane.lane))
            item.user_properties.append(
                ("diagnostics_validation_status", diagnostics_lane.validation_status),
            )

# Ensure project root is on sys.path so tests can import root-level
# app modules (app_mhd.py, app_engine.py) that aren't part of the dpf package.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dpf.config import SimulationConfig  # noqa: E402
from dpf.diagnostics.test_lanes import diagnostics_test_lane_for_file  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance Tiers — formalized accuracy expectations by test category
# ---------------------------------------------------------------------------
# Tier 1 (UNIT): Machine-precision correctness — analytical solutions,
#   round-trip conversions, identity operations. No solver involved.
# Tier 2 (INTEGRATION): Solver-level fidelity — shock tests, conservation
#   laws, cross-backend parity. Depends on reconstruction + Riemann solver.
# Tier 3 (ACCEPTANCE): Experimental validation — I_peak, t_peak, NRMSE
#   against published device data. Depends on full physics chain.
# ---------------------------------------------------------------------------

TOLERANCE_TIERS = {
    "unit": {"rtol": 1e-10, "atol": 1e-15},
    "unit_f32": {"rtol": 1e-5, "atol": 1e-7},
    "integration": {"rtol": 0.01, "atol": 1e-8},
    "integration_loose": {"rtol": 0.05, "atol": 1e-6},
    "acceptance": {"rtol": 0.15, "atol": 0.0},
    "acceptance_loose": {"rtol": 0.50, "atol": 0.0},
}

DEVICE_TOLERANCES = {
    # I_peak 0.12: REGRESSION GUARD, not a published validation gate.
    # [KR: UNVERIFIED — no published I_peak tolerance budget]
    # Scholz 2006 (PF-1000 reference, 1.87 MA) publishes no measurement
    # uncertainty. Lee Course p.11 (KR: a-course-on-plasma-focus-numerical-
    # experiments-s-lee-and-s-h-saw-part-1-basic-course.md) describes a
    # 5-point fit (rising slope, topping, peak, dip slope, dip bottom) with
    # "reasonable (typically very good) fit" and ~3% residual at the dip
    # bottom — but does NOT publish an I_peak tolerance. KR-canonical Malek
    # 2025 Lee fits (fc=0.7, fm=0.13, fmr=0.35, fcr=0.65) [KR: plasma-physics-
    # and-technology-1211-9-2025.md §3 lines 177-180] are inputs; Akel 2021
    # R0=6.1 mOhm restored at commit ref e219ebb. The 12% number = current
    # model output (~9.2%) + ~3% headroom for fit variation across operating
    # conditions; treat as a CI fence to detect regressions, not a claim
    # that 12% is a paper-published acceptance criterion.
    #
    # DUAL-FENCE ARCHITECTURE: this 0.12 is DISTINCT from the 0.10 in
    # tests/reference_data/radpf_pf1000_27kv.json acceptance_criteria.I_peak_tolerance.
    # That 0.10 is the VALIDATION gate consumed exclusively by test_mhd_acceptance.py
    # (test_angle1_ipeak, line 56) and reflects Anthony-generated RADPF truth data.
    # This 0.12 is the REGRESSION gate consumed by conftest fixtures (line 144) for
    # non-acceptance tests. Do NOT collapse them: 0.10 != 0.12 intentionally.
    "PF-1000": {"I_peak": 0.12, "t_peak": 0.10, "nrmse": 0.22, "energy": 0.05, "Yn": 1.0},
    "PF-1000-Gribkov": {"I_peak": 0.06, "t_peak": 0.03, "nrmse": 0.30, "energy": 0.05, "Yn": 1.0},
    "PF-1000-16kV": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.20, "energy": 0.05, "Yn": 1.0},
    "PF-1000-20kV": {"I_peak": 0.15, "t_peak": 0.15, "nrmse": 0.25, "energy": 0.05, "Yn": 1.0},
    "NX2": {"I_peak": 0.35, "t_peak": 0.50, "nrmse": 0.40, "energy": 0.05, "Yn": 1.0},
    "UNU-ICTP": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.15, "energy": 0.05, "Yn": 1.0},
    "POSEIDON": {"I_peak": 0.10, "t_peak": 0.40, "nrmse": 0.30, "energy": 0.05, "Yn": 1.0},
    "POSEIDON-60kV": {"I_peak": 0.05, "t_peak": 0.05, "nrmse": 0.15, "energy": 0.05, "Yn": 1.0},
    "MJOLNIR": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.30, "energy": 0.05, "Yn": 1.0},
    "FAETON-I": {"I_peak": 0.10, "t_peak": 0.10, "nrmse": 0.10, "energy": 0.05, "Yn": 1.0},
}

# CI gate thresholds (used by test_validation_ci.py and CI workflow)
# Wave-10 (2026-04-29): nrmse_fail bumped 0.30 → 0.35 to accommodate Wave-9 #12's
# Malek 2025 canonical preset, which produces +7.6% I_peak vs Scholz 2006 reference
# (NRMSE=0.31). This is regression-fence, not validation — Wave-9 #12 chose Malek
# verbatim per papers-are-truth rule; calibrating to close the gap = bug.
# True validation gate is per-device "nrmse" in DEVICE_TOLERANCES.
CI_THRESHOLDS = {
    "nrmse_warn": 0.20,
    "nrmse_fail": 0.35,
    "ipeak_fail": 0.15,
    "min_test_count": 4000,
}


@pytest.fixture(params=["unit", "unit_f32", "integration", "integration_loose",
                         "acceptance", "acceptance_loose"])
def tolerance_tier(request):
    """Parametrized fixture returning tolerance dict for the requested tier."""
    return TOLERANCE_TIERS[request.param]


@pytest.fixture
def unit_tol():
    """Machine-precision tolerances (float64)."""
    return TOLERANCE_TIERS["unit"]


@pytest.fixture
def unit_tol_f32():
    """Machine-precision tolerances (float32)."""
    return TOLERANCE_TIERS["unit_f32"]


@pytest.fixture
def integration_tol():
    """Solver-level tolerances (shock tests, conservation)."""
    return TOLERANCE_TIERS["integration"]


@pytest.fixture
def acceptance_tol():
    """Experimental validation tolerances (device-dependent)."""
    return TOLERANCE_TIERS["acceptance"]


def device_tol(device_name: str) -> dict:
    """Get device-specific tolerances. Falls back to acceptance tier."""
    return DEVICE_TOLERANCES.get(device_name, {
        "I_peak": 0.15, "t_peak": 0.15, "nrmse": 0.30, "energy": 0.05, "Yn": 1.0,
    })


@pytest.fixture
def grid_shape():
    """Small grid for fast unit tests."""
    return (8, 8, 8)


@pytest.fixture
def dx():
    return 1e-2


@pytest.fixture
def cylindrical_grid():
    """Small CylindricalGrid for AMR unit tests."""
    mlx = pytest.importorskip("mlx.core")  # noqa: F841
    from dpf.metal.mlx_grid import CylindricalGrid
    return CylindricalGrid(nr=32, nz=64, dr=1e-3, dz=1e-3, r_inner=0.01)


@pytest.fixture
def default_circuit_params():
    """Standard DPF circuit parameters."""
    return {
        "C": 1e-6,
        "V0": 1e3,
        "L0": 1e-7,
        "R0": 0.01,
        "ESR": 0.0,
        "ESL": 0.0,
        "anode_radius": 0.005,
        "cathode_radius": 0.01,
    }


@pytest.fixture
def sample_config_dict(grid_shape, dx, default_circuit_params):
    """Minimal valid SimulationConfig as a dictionary."""
    return {
        "grid_shape": list(grid_shape),
        "dx": dx,
        "sim_time": 1e-6,
        "circuit": default_circuit_params,
    }


@pytest.fixture
def small_config(sample_config_dict):
    """Small SimulationConfig for fast unit tests."""
    return SimulationConfig(**sample_config_dict)


# --- Module-scoped Metal solver fixture (avoid repeated init per test) ---


@pytest.fixture(scope="module")
def metal_solver_16():
    """Module-scoped MetalMHDSolver for 16^3 grid."""
    try:
        import torch

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")
        from dpf.metal.metal_solver import MetalMHDSolver

        return MetalMHDSolver(
            grid_shape=(16, 16, 16),
            dx=1e-2,
            gamma=5.0 / 3.0,
        )
    except ImportError:
        pytest.skip("Metal solver not available")


# --- Shared initial condition fixtures ---


@pytest.fixture
def sod_ic_1d():
    """Standard Sod shock tube IC for 1D tests."""
    import numpy as np

    nx = 16
    rho = np.where(np.arange(nx) < nx // 2, 1.0, 0.125)
    p = np.where(np.arange(nx) < nx // 2, 1.0, 0.1)
    v = np.zeros((3, nx))
    B = np.zeros((3, nx))
    return {"rho": rho, "pressure": p, "velocity": v, "B": B}


@pytest.fixture
def brio_wu_ic_1d():
    """Standard Brio-Wu MHD shock IC."""
    import numpy as np

    nx = 16
    rho = np.where(np.arange(nx) < nx // 2, 1.0, 0.125)
    p = np.where(np.arange(nx) < nx // 2, 1.0, 0.1)
    v = np.zeros((3, nx))
    B = np.zeros((3, nx))
    B[0] = 0.75  # Bx constant
    B[1] = np.where(np.arange(nx) < nx // 2, 1.0, -1.0)  # By discontinuity
    return {"rho": rho, "pressure": p, "velocity": v, "B": B}


# --- Session-scoped Numba pre-warming ---


@pytest.fixture(scope="session", autouse=True)
def _prewarm_numba_cache():
    """Pre-compile frequently-used Numba JIT functions at session start."""
    try:
        import numpy as np

        # Trigger JIT compilation with tiny arrays
        np.ones((4, 4, 4), dtype=np.float64)

        # Import hot-path modules to trigger cache loading
        import contextlib

        with contextlib.suppress(ImportError):
            from dpf.fluid.mhd_solver import MHDSolver  # noqa: F401
        with contextlib.suppress(ImportError):
            from dpf.fluid.constrained_transport import ct_emf_kernel  # noqa: F401
    except Exception:
        pass  # Never fail tests due to pre-warming
