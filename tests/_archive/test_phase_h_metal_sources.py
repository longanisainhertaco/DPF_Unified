"""Phase H: Metal transport source term integration tests.

Tests that MetalMHDSolver correctly dispatches operator-split transport
physics (Hall, Braginskii conduction/viscosity, Nernst) via metal_transport.py.
"""

import numpy as np
import torch

from dpf.metal.metal_solver import MetalMHDSolver


class TestMetalSourceTerms:

    def setup_method(self):
        self.nx, self.ny, self.nz = 8, 8, 8
        self.solver = MetalMHDSolver(
            grid_shape=(self.nx, self.ny, self.nz),
            dx=0.1,
            enable_hall=True,
            enable_braginskii_conduction=False,
        )

    def test_hall_enabled_flag(self):
        """Verify that enable_hall flag is set correctly."""
        assert self.solver.enable_hall is True
        solver2 = MetalMHDSolver(
            grid_shape=(self.nx, self.ny, self.nz),
            dx=0.1,
            enable_hall=False,
        )
        assert solver2.enable_hall is False

    def test_step_runs_with_transport_enabled(self):
        """Verify step() runs without error when transport physics enabled."""
        state = {
            "rho": np.ones((self.nx, self.ny, self.nz)),
            "velocity": np.zeros((3, self.nx, self.ny, self.nz)),
            "pressure": np.ones((self.nx, self.ny, self.nz)),
            "B": np.zeros((3, self.nx, self.ny, self.nz)),
            "Te": np.ones((self.nx, self.ny, self.nz)) * 1e4,
            "Ti": np.ones((self.nx, self.ny, self.nz)) * 1e4,
            "psi": np.zeros((self.nx, self.ny, self.nz)),
        }
        state["B"][2] = 1.0  # Bz = 1

        dt = self.solver.compute_dt(state)
        result = self.solver.step(state, dt, current=0.0, voltage=0.0)

        assert "rho" in result
        rho = result["rho"]
        if isinstance(rho, torch.Tensor):
            rho = rho.cpu().numpy()
        assert rho.shape[-3:] == (self.nx, self.ny, self.nz)
        assert np.all(np.isfinite(rho))
