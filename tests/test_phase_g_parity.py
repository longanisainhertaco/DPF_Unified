"""Phase G: Metal & Athena Physics Parity Tests.

Tests for:
1. Athena++ Operator Splitting:
   - Verify that Athena++ backend skips Python collision/radiation operators
     (these will be added as native C++ source terms in a future phase).

2. Metal Backend Physics:
   - Verify Hall term implementation (if enabled).
   - Verify Anomalous Resistivity implementation (if enabled).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.metal.metal_solver import MetalMHDSolver


class TestAthenaIntegration:
    """Tests for the Athena++ backend wrapper in SimulationEngine."""

    @patch("dpf.engine.SimulationEngine._apply_collision_radiation")
    def test_athena_skips_python_radiation(self, mock_col_rad):
        """Verify that Athena++ backend skips Python collision/radiation operators.

        The _step_athena() method deliberately bypasses the Python-side
        collision/radiation operators (Spitzer resistivity, Nernst, viscosity,
        bremsstrahlung).  These physics will be handled natively as Athena++
        source terms in a future phase.  The Python engine's Strang-split
        wrapper is NOT used for the Athena++ backend.
        """
        cfg = SimulationConfig(
            grid_shape=[10, 10, 10], dx=0.01, sim_time=1e-6,
            circuit={"C": 1e-6, "V0": 1e3, "L0": 1e-9, "anode_radius": 0.01, "cathode_radius": 0.02},
            fluid={"backend": "athena"},
            radiation={"bremsstrahlung_enabled": True},
        )
        engine = SimulationEngine(cfg)

        # Mock the fluid step to avoid needing actual Athena binaries
        engine.fluid.step = MagicMock(return_value=engine.state)
        engine.fluid.coupling_interface = MagicMock(return_value=engine._coupling)

        # Run one step
        engine.step()

        # Athena++ backend intentionally does NOT call Python collision/radiation
        # (these will be added as native C++ source terms in a future phase)
        assert mock_col_rad.call_count == 0, (
            "Athena++ backend should not call Python _apply_collision_radiation; "
            "physics operators are handled natively by Athena++"
        )


class TestMetalPhysics:
    """Tests for the Metal backend physics implementations."""

    def test_metal_solver_initialization(self):
        """Verify Metal solver initializes with new flags."""
        solver = MetalMHDSolver(
            grid_shape=(10, 10, 10), dx=0.01,
            enable_hall=True,
            device="cpu",
        )
        assert solver.enable_hall is True

    @pytest.mark.skipif(not MetalMHDSolver.is_available(), reason="Metal/MPS not available")
    def test_hall_term_effect(self):
        """Test that Hall term affects the field evolution (qualitative)."""
        # Setup a scenario where Hall term matters (e.g. current sheet)
        # TODO: Implement a specific test case (e.g. Whistler wave)
        pass
