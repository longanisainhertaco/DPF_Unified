"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from dpf.config import SimulationConfig


@pytest.fixture
def grid_shape():
    """Small grid for fast unit tests."""
    return (8, 8, 8)


@pytest.fixture
def dx():
    return 1e-2


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
