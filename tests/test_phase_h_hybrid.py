
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
            max_div_b_error=1e-2
        )

    def test_div_b_check_pass(self):
        """Validation should pass for divergence-free field."""
        # Create a B field with 0 divergence
        # B = (y, x, 0) -> div B = 0 + 0 + 0 = 0
        grid = (10, 10, 10)
        B = np.zeros((3, *grid))

        # Need coordinates to be perfect?
        # Just use constants: div const = 0.
        B[0] = 1.0
        B[1] = 2.0
        B[2] = 3.0

        state = {"B": B}
        self.surrogate.predict_next_step.return_value = state

        # History
        history = [state, state]

        # Run surrogate phase for 1 step
        # It calls _validate_step inside loop (at interval).
        # We need to ensure it DOES NOT break.
        # But wait, run_surrogate_phase has loop length check.
        # If we request 2 steps, and validation passes, we get 2 steps.

        # Mock predict_next_step to return consistent state
        self.surrogate.predict_next_step = MagicMock(return_value=state)

        res = self.engine._run_surrogate_phase(history, n_steps=2)

        # Should return full history (2 steps)
        assert len(res) == 2

    def test_div_b_check_fail(self):
        """Validation should detect Magnetic Monopole and stop/fallback."""
        # Create a B field with significant divergence
        # B_x = x -> div B = 1
        grid = (10, 10, 10)
        B = np.zeros((3, *grid))

        # Create coordinate grid indices
        # B_x varies with x
        for i in range(10):
            B[0, i, :, :] = float(i)

        # div B ~ dBx/dx = 1.
        # Mean |B| ~ 4.5.
        # Error ~ 1/4.5 = 0.22 > 0.01

        state = {"B": B}
        self.surrogate.predict_next_step = MagicMock(return_value=state)

        history = [state, state]

        # Run
        # We ask for 5 steps.
        # Step 1: Predict -> Validate -> Fail -> Break.
        # Returns [state_step1] (length 1).

        res = self.engine._run_surrogate_phase(history, n_steps=5)

        # Should bail out after step 1 (validation interval=1)
        # It appends the bad state, validates, fails, breaks.
        # So len should be 1.
        assert len(res) == 1

    def test_div_b_computation(self):
        """Test the _compute_div_b_error method directly."""
        # Case 1: Constant vector field -> Div 0
        grid = (5, 5, 5)
        B = np.zeros((3, *grid))
        B[0] = 1.0
        err = self.engine._compute_div_b_error(B)
        assert err == 0.0

        # Case 2: Monopole field -> High Div
        # Bx = x
        B = np.zeros((3, *grid))
        for i in range(5):
             B[0, i, :, :] = float(i) + 1.0 # Offset 1.0 to avoid huge relative error if mean is tiny

        err = self.engine._compute_div_b_error(B)
        # Mean B ~ 3.
        # Div B = 1 (central diff of linear 1,2,3,4,5 is 1.0 everywhere in interior)
        # Error = 1 / 3 = 0.33 > 0.1
        assert err > 0.1
