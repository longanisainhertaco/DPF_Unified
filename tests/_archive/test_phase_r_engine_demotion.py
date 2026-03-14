"""Phase R.2: Python engine demotion to teaching-only tier.

Tests verifying that the Python MHD engine is properly demoted to
teaching-only status, with appropriate warnings and the engine_tier
property reflecting the backend's production readiness.

The Python engine uses dp/dt (non-conservative pressure equation)
instead of dE/dt (total energy), violating Rankine-Hugoniot at shocks.
Production simulations should use Metal or Athena++ backends.
"""

from __future__ import annotations

import logging
import warnings

import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_python_config() -> dict:
    """Config dict for a small Python-backend simulation."""
    return {
        "grid_shape": [8, 8, 8],
        "dx": 1e-3,
        "sim_time": 1e-8,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 1e-6,
            "V0": 1e3,
            "L0": 1e-7,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        "fluid": {"backend": "python"},
        "diagnostics": {"hdf5_filename": ":memory:"},
    }


@pytest.fixture
def large_python_config() -> dict:
    """Config for Python backend exceeding the demotion threshold (grid > 16^3)."""
    return {
        "grid_shape": [32, 32, 32],
        "dx": 5e-4,
        "sim_time": 5e-7,
        "dt_init": 1e-10,
        "rho0": 1e-4,
        "T0": 300.0,
        "circuit": {
            "C": 5e-6,
            "V0": 5e3,
            "L0": 5e-8,
            "R0": 0.01,
            "anode_radius": 0.005,
            "cathode_radius": 0.01,
        },
        "fluid": {"backend": "python"},
        "diagnostics": {"hdf5_filename": ":memory:"},
    }


# ---------------------------------------------------------------------------
# Engine tier property tests
# ---------------------------------------------------------------------------

class TestEngineTier:
    """Tests for the engine_tier property."""

    def test_python_backend_is_teaching_tier(self, small_python_config: dict):
        """Python backend should report 'teaching' tier."""
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        assert engine.engine_tier == "teaching"

    def test_python_backend_name(self, small_python_config: dict):
        """Python backend should resolve correctly."""
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        assert engine.backend == "python"


# ---------------------------------------------------------------------------
# Deprecation warning tests
# ---------------------------------------------------------------------------

class TestPythonDeprecationWarnings:
    """Tests for deprecation warnings on Python backend."""

    def test_large_grid_deprecation_warning(self, large_python_config: dict):
        """Python backend on large grid should emit DeprecationWarning."""
        config = SimulationConfig(**large_python_config)
        engine = SimulationEngine(config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.step()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            msg = str(dep_warnings[0].message)
            assert "non-conservative" in msg or "dp/dt" in msg

    def test_small_grid_no_deprecation(self, small_python_config: dict):
        """Python backend on small grid with short sim_time should NOT warn."""
        # 8^3 = 512 < 16^3 = 4096 and sim_time=1e-8 < 1e-7
        config = SimulationConfig(**small_python_config)
        engine = SimulationEngine(config)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.step()
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# ---------------------------------------------------------------------------
# MHD solver docstring tests
# ---------------------------------------------------------------------------

class TestMHDSolverDocstring:
    """Tests for the MHD solver docstring warning."""

    def test_mhd_solver_docstring_warns_teaching(self):
        """MHDSolver module docstring should mention teaching/fallback."""
        from dpf.fluid import mhd_solver
        doc = mhd_solver.__doc__
        assert "teaching" in doc.lower() or "fallback" in doc.lower()
        assert "non-conservative" in doc.lower() or "dp/dt" in doc.lower()

    def test_mhd_solver_docstring_recommends_metal(self):
        """MHDSolver module docstring should recommend Metal/Athena++."""
        from dpf.fluid import mhd_solver
        doc = mhd_solver.__doc__
        assert "metal" in doc.lower()
        assert "athena" in doc.lower()


# ---------------------------------------------------------------------------
# Backend resolution tests
# ---------------------------------------------------------------------------

class TestBackendResolution:
    """Tests for backend auto-resolution and tier classification."""

    def test_resolve_python_explicit(self):
        """Explicitly requesting Python returns 'python'."""
        result = SimulationEngine._resolve_backend("python")
        assert result == "python"

    def test_resolve_unknown_raises(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            SimulationEngine._resolve_backend("nonexistent_backend")


# ---------------------------------------------------------------------------
# Metal backend physics coverage
# ---------------------------------------------------------------------------

class TestMetalPhysicsCoverage:
    """Tests verifying Metal solver supports key physics modules."""

    def test_metal_solver_has_transport_flags(self):
        """MetalMHDSolver should accept transport physics flags."""
        pytest.importorskip("torch")
        from dpf.metal.metal_solver import MetalMHDSolver
        # Instantiate with all physics enabled (may fail on non-MPS, use CPU)
        solver = MetalMHDSolver(
            grid_shape=(8, 8, 8),
            dx=1e-3,
            device="cpu",
            use_ct=False,
            enable_hall=True,
            enable_braginskii_conduction=True,
            enable_braginskii_viscosity=True,
            enable_nernst=True,
        )
        assert solver.enable_hall is True
        assert solver.enable_braginskii_conduction is True
        assert solver.enable_braginskii_viscosity is True
        assert solver.enable_nernst is True

    def test_metal_solver_conservative_formulation(self):
        """Metal solver uses conservative variables (8-component: rho, mom, E, B)."""
        from dpf.metal.metal_riemann import IEN, NVAR
        # Conservative variables should include total energy (not pressure)
        assert NVAR == 8
        assert IEN == 4  # Total energy is component 4


# ---------------------------------------------------------------------------
# Backend physics warning tests
# ---------------------------------------------------------------------------

class TestBackendPhysicsWarnings:
    """Tests for backend-specific physics warnings."""

    def test_python_engine_no_physics_skip_warning(
        self, small_python_config: dict, caplog: pytest.LogCaptureFixture,
    ):
        """Python engine should NOT warn about skipped physics (it has all)."""
        with caplog.at_level(logging.WARNING):
            config = SimulationConfig(**small_python_config)
            SimulationEngine(config)
        # Should not have "skips physics" in warnings for Python backend
        skip_msgs = [r for r in caplog.records if "skips physics" in r.message]
        assert len(skip_msgs) == 0


# ---------------------------------------------------------------------------
# R_plasma cap test
# ---------------------------------------------------------------------------

class TestRPlasmaCapIncrease:
    """Test that R_plasma cap was increased from 10 to 1000 Ohm."""

    def test_r_plasma_cap_is_1000(self):
        """The R_plasma cap in engine.py should allow up to 1000 Ohm."""
        # Search the source for the R_plasma cap
        import inspect
        source = inspect.getsource(SimulationEngine.step)
        # Should contain 1000.0, not 10.0 as the cap
        assert "1000.0" in source or "1000" in source
