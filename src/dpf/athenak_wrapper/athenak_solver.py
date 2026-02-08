"""AthenaK subprocess solver implementing PlasmaSolverBase.

:class:`AthenaKSolver` runs AthenaK as a subprocess, communicating via
input files (athinput) and VTK output.  This avoids the need for in-process
linking while still leveraging AthenaK's Kokkos-accelerated C++ MHD solver.

The solver operates in batch mode: each subprocess call runs multiple
timesteps (controlled by ``nlim`` or ``tlim``), amortizing the subprocess
spawn overhead.

Example::

    from dpf.config import SimulationConfig
    from dpf.athenak_wrapper import AthenaKSolver

    config = SimulationConfig.from_file("pf1000.json")
    solver = AthenaKSolver(config)
    state = solver.initial_state()
    state = solver.step(state, dt=1e-9, current=100e3, voltage=20e3)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dpf.core.bases import CouplingState, PlasmaSolverBase

if TYPE_CHECKING:
    from dpf.config import SimulationConfig

logger = logging.getLogger(__name__)


class AthenaKSolver(PlasmaSolverBase):
    """AthenaK MHD solver via subprocess execution.

    Runs AthenaK as a child process, writing athinput files and reading
    VTK output.  Supports both built-in problem generators (via
    ``pgen_name``) and custom compiled ones.

    Args:
        config: DPF simulation configuration.
        binary_path: Path to the AthenaK binary. If None, auto-detects.
        batch_steps: Number of timesteps per subprocess call.
        timeout: Maximum seconds to wait for subprocess completion.
        pgen_name: Built-in problem generator name (e.g., "shock_tube").

    Raises:
        FileNotFoundError: If AthenaK binary not found.
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        binary_path: str | None = None,
        batch_steps: int = 100,
        timeout: float = 300.0,
        pgen_name: str | None = None,
    ) -> None:
        self.config = config
        self._coupling = CouplingState()
        self._time = 0.0
        self._cycle = 0

        # Grid dimensions
        nx, ny, nz = config.grid_shape
        self._nx = nx
        self._ny = ny
        self._nz = nz

        self._batch_steps = batch_steps
        self._timeout = timeout
        self._pgen_name = pgen_name

        # Find binary
        self._binary = self._find_binary(binary_path)

        logger.info(
            "AthenaKSolver initialized: binary=%s, grid=(%d,%d,%d), "
            "batch_steps=%d, pgen=%s",
            self._binary, nx, ny, nz, batch_steps,
            pgen_name or "custom",
        )

    @staticmethod
    def _find_binary(binary_path: str | None) -> str:
        """Locate the AthenaK binary."""
        if binary_path is not None:
            if os.path.isfile(binary_path) and os.access(binary_path, os.X_OK):
                return binary_path
            raise FileNotFoundError(f"AthenaK binary not found: {binary_path}")

        from dpf.athenak_wrapper import get_binary_path

        path = get_binary_path()
        if path is not None:
            return path

        raise FileNotFoundError(
            "AthenaK binary not found. Build with:\n"
            "  bash scripts/setup_athenak.sh\n"
            "  bash scripts/build_athenak.sh"
        )

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Advance the plasma state by one batch of timesteps.

        In subprocess mode, each call runs ``batch_steps`` AthenaK cycles.
        The ``dt`` parameter sets the total time limit for the batch rather
        than a single step size (AthenaK computes its own CFL-limited dt).

        Args:
            state: Current DPF state dictionary.
            dt: Total time to advance (AthenaK manages sub-stepping).
            current: Circuit current [A] for boundary forcing.
            voltage: Capacitor voltage [V].

        Returns:
            Updated state dictionary from AthenaK's final output.
        """
        from dpf.athenak_wrapper.athenak_config import generate_athenak_input
        from dpf.athenak_wrapper.athenak_io import (
            convert_to_dpf_state,
            find_latest_vtk,
            read_vtk_file,
        )

        # Create temporary directory for this run
        with tempfile.TemporaryDirectory(prefix="athenak_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Generate input file with time limit = dt
            # Override sim_time temporarily
            original_sim_time = self.config.sim_time
            self.config.sim_time = dt  # type: ignore[misc]

            athinput = generate_athenak_input(
                self.config,
                problem_id="dpf_step",
                pgen_name=self._pgen_name,
                n_steps=self._batch_steps,
                output_vtk=True,
                vtk_dt=dt,  # Single VTK output at end
            )

            self.config.sim_time = original_sim_time  # type: ignore[misc]

            # Write input file
            input_path = tmpdir_path / "athinput"
            input_path.write_text(athinput)

            # Run AthenaK
            result = self._run_subprocess(str(input_path), tmpdir)

            if result.returncode != 0:
                logger.error(
                    "AthenaK failed (exit %d): %s",
                    result.returncode,
                    result.stderr[:500] if result.stderr else "no stderr",
                )
                return state  # Return unchanged state on failure

            # Parse output
            vtk_path = find_latest_vtk(tmpdir_path)
            if vtk_path is None:
                logger.warning("No VTK output from AthenaK — returning unchanged state")
                return state

            vtk_data = read_vtk_file(vtk_path)
            new_state = convert_to_dpf_state(vtk_data, gamma=self.config.fluid.gamma)

            self._time += dt
            self._cycle += vtk_data.get("cycle", self._batch_steps)

            logger.debug(
                "AthenaK step complete: time=%.4e, cycle=%d",
                self._time, self._cycle,
            )

            return new_state

    def _run_subprocess(
        self,
        input_path: str,
        output_dir: str,
    ) -> subprocess.CompletedProcess[str]:
        """Execute AthenaK as a subprocess.

        Args:
            input_path: Path to athinput file.
            output_dir: Directory for output files.

        Returns:
            CompletedProcess with stdout, stderr, returncode.
        """
        cmd = [self._binary, "-i", input_path, "-d", output_dir]

        logger.debug("Running AthenaK: %s", " ".join(cmd))

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            cwd=output_dir,
        )

    def coupling_interface(self) -> CouplingState:
        """Return coupling quantities for the circuit solver."""
        return self._coupling

    def initial_state(self) -> dict[str, np.ndarray]:
        """Generate initial state arrays.

        Returns:
            DPF state dictionary with uniform initial conditions.
        """
        shape = (self._nz, self._ny, self._nx) if self._nz > 1 else (
            (self._ny, self._nx) if self._ny > 1 else (self._nx,)
        )

        rho = np.full(shape, self.config.rho0)
        velocity = np.zeros((3, *shape))
        pressure = np.full(shape, self.config.rho0 * 1.380649e-23 * self.config.T0 / 3.34358377e-27)
        B = np.zeros((3, *shape))
        Te = np.full(shape, self.config.T0)
        Ti = np.full(shape, self.config.T0)
        psi = np.zeros(shape)

        return {
            "rho": rho,
            "velocity": velocity,
            "pressure": pressure,
            "B": B,
            "Te": Te,
            "Ti": Ti,
            "psi": psi,
        }
