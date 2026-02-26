
from unittest.mock import MagicMock

import numpy as np

from dpf.ai.hybrid_engine import HybridEngine
from dpf.config import SimulationConfig


class TestHybridValidation:

    def setup_method(self):
        self.config = SimulationConfig(
            grid_shape=(10, 10, 10),
            dx=0.1,
            sim_time=1e-6,
            circuit={
                "type": "rlc",
                "L": 1e-9, "C": 1e-6, "R": 0.0, "V0": 1.0,
                "L0": 10e-9, "anode_radius": 0.05, "cathode_radius": 0.1
            }
        )
        self.surrogate = MagicMock()
        self.surrogate.history_length = 2

        self.engine = HybridEngine(
            self.config,
            self.surrogate,
            handoff_fraction=0.1,
            validation_interval=1,
            max_l2_divergence=1e-2,
        )

    def test_surrogate_phase_runs(self):
        """Surrogate phase runs and returns predictions."""
        grid = (10, 10, 10)
        B = np.zeros((3, *grid))
        B[0] = 1.0
        B[1] = 2.0
        B[2] = 3.0

        state = {
            "B": B,
            "rho": np.ones(grid),
            "velocity": np.zeros((3, *grid)),
            "pressure": np.ones(grid),
        }
        self.surrogate.predict_next_step = MagicMock(return_value=state)

        history = [state, state]
        res = self.engine._run_surrogate_phase(history, n_steps=2)

        assert len(res) == 2

    def test_surrogate_nan_fallback(self):
        """Surrogate phase detects NaN and falls back."""
        grid = (10, 10, 10)
        nan_state = {
            "B": np.full((3, *grid), np.nan),
            "rho": np.full(grid, np.nan),
        }
        good_state = {
            "B": np.ones((3, *grid)),
            "rho": np.ones(grid),
        }
        self.surrogate.predict_next_step = MagicMock(return_value=nan_state)

        history = [good_state, good_state]

        # validation_interval=1 means it checks at step 1 and detects NaN
        res = self.engine._run_surrogate_phase(history, n_steps=5)

        # Should bail out after step 1 (appends NaN state, validates, breaks)
        assert len(res) == 1

    def test_dead_validate_step_removed(self):
        """_validate_step dead code has been removed from HybridEngine."""
        assert not hasattr(self.engine, "_validate_step")
