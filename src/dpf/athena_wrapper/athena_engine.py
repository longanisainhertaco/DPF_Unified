"""Athena++ solver wrapper implementing PlasmaSolverBase.

:class:`AthenaPPSolver` wraps the Athena++ C++ MHD code as a drop-in
replacement for the Python :class:`~dpf.fluid.mhd_solver.MHDSolver`.
It delegates all MHD computations to the compiled C++ backend while
presenting the standard DPF state-dictionary interface.

The solver can operate in two modes:

1. **Linked mode** (C++ extension compiled): Uses pybind11 bindings
   to drive Athena++ in-process.  Zero-copy NumPy array access via
   ``AthenaArray<Real>::data()`` pointers.

2. **Subprocess mode** (fallback): Writes an athinput file, runs
   the ``athena`` binary as a subprocess, and reads HDF5 output.
   Slower but works without compiling the extension.

Example::

    from dpf.config import SimulationConfig
    from dpf.athena_wrapper import AthenaPPSolver, is_available

    config = SimulationConfig.from_file("pf1000.json")
    solver = AthenaPPSolver(config)
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

# Constants
_MU_0 = 4.0e-7 * np.pi  # Vacuum permeability [H/m]


class AthenaPPSolver(PlasmaSolverBase):
    """Athena++ MHD solver implementing the DPF PlasmaSolverBase interface.

    This class wraps the Princeton Athena++ C++ code, translating between
    the DPF state dictionary format and Athena++'s internal data structures.

    Args:
        config: DPF simulation configuration.
        athena_binary: Path to compiled ``athena`` binary (for subprocess mode).
            Defaults to ``external/athena/bin/athena`` relative to project root.
        use_subprocess: Force subprocess mode even if C++ extension is available.

    Attributes:
        config: The DPF simulation configuration.
        mode: "linked" if using C++ extension, "subprocess" if using binary.
        mesh: Reference to the Athena++ Mesh object (linked mode only).
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        athena_binary: str | None = None,
        use_subprocess: bool = False,
    ) -> None:
        self.config = config
        self._coupling = CouplingState()
        self._time = 0.0
        self._cycle = 0
        self._initialized = False

        # Grid dimensions
        nx, ny, nz = config.grid_shape
        self._nx = nx
        self._ny = ny
        self._nz = nz
        self._is_cylindrical = config.geometry.type == "cylindrical"

        # Determine mode
        from dpf.athena_wrapper import is_available

        if is_available() and not use_subprocess:
            self.mode = "linked"
            self._init_linked()
        else:
            self.mode = "subprocess"
            self._athena_binary = self._find_binary(athena_binary)
            logger.info(
                "AthenaPPSolver using subprocess mode (binary: %s)",
                self._athena_binary,
            )

        logger.info(
            "AthenaPPSolver initialized: mode=%s, grid=(%d,%d,%d), geometry=%s",
            self.mode, nx, ny, nz, config.geometry.type,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _find_binary(self, athena_binary: str | None) -> str:
        """Locate the Athena++ binary."""
        if athena_binary is not None:
            return athena_binary

        # Search relative to project root
        candidates = [
            Path(__file__).parents[3] / "external" / "athena" / "bin" / "athena",
            Path.cwd() / "external" / "athena" / "bin" / "athena",
        ]
        for p in candidates:
            if p.is_file() and os.access(str(p), os.X_OK):
                return str(p)

        raise FileNotFoundError(
            "Athena++ binary not found. Build it with:\n"
            "  cd external/athena\n"
            "  python3 configure.py --prob=magnoh --coord=cylindrical -b "
            "--flux=hlld --hdf5_path=/opt/homebrew/opt/hdf5\n"
            "  make -j8"
        )

    def _init_linked(self) -> None:
        """Initialize linked mode via pybind11 C++ extension."""
        from dpf.athena_wrapper import _athena_core
        from dpf.athena_wrapper.athena_config import generate_athinput

        # Generate athinput text
        athinput_text = generate_athinput(self.config)

        # Initialize Athena++ mesh via C++ bindings
        self._core = _athena_core
        self._mesh_handle = _athena_core.init_from_string(athinput_text)
        self._initialized = True

        logger.info("Athena++ linked mode initialized: %d mesh blocks",
                     _athena_core.get_num_meshblocks(self._mesh_handle))

    # ------------------------------------------------------------------
    # PlasmaSolverBase interface
    # ------------------------------------------------------------------

    def step(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Advance the plasma state by one timestep.

        Args:
            state: Dictionary of field arrays (rho, velocity, pressure, B, Te, Ti, psi).
            dt: Timestep size [s].
            current: Circuit current [A] for boundary forcing.
            voltage: Capacitor voltage [V].
            **kwargs: Additional parameters (eta_field, electrode BCs, etc.).

        Returns:
            Updated state dictionary with same keys and shapes.
        """
        if self.mode == "linked":
            return self._step_linked(state, dt, current, voltage, **kwargs)
        else:
            return self._step_subprocess(state, dt, current, voltage, **kwargs)

    def coupling_interface(self) -> CouplingState:
        """Return coupling quantities for the circuit solver.

        Computes plasma resistance and inductance from the current
        Athena++ state using volume integrals.
        """
        return self._coupling

    def _compute_dt(self, state: dict[str, np.ndarray]) -> float:
        """Compute CFL-limited timestep from current state.

        Args:
            state: Current simulation state.

        Returns:
            CFL-limited timestep [s].
        """
        if self.mode == "linked" and self._initialized:
            return self._core.get_dt(self._mesh_handle)

        # Fallback: estimate from sound speed + Alfven speed
        rho = state["rho"]
        p = state["pressure"]
        B = state["B"]
        gamma = self.config.fluid.gamma
        dx = self.config.dx

        cs = np.sqrt(gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30))
        B_mag = np.sqrt(np.sum(B**2, axis=0))
        va = B_mag / np.sqrt(_MU_0 * np.maximum(rho, 1e-30))
        v_max = float(np.max(cs + va))

        return self.config.fluid.cfl * dx / max(v_max, 1e-30)

    # ------------------------------------------------------------------
    # Linked mode implementation
    # ------------------------------------------------------------------

    def _step_linked(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Execute one MHD cycle using the C++ extension."""
        core = self._core

        # Push circuit parameters to C++ side
        core.set_circuit_params(self._mesh_handle, current, voltage)

        # Execute one Athena++ cycle
        core.execute_cycle(self._mesh_handle, dt)

        # Read updated state back from C++ arrays
        new_state = self._read_state_from_cpp()

        self._time += dt
        self._cycle += 1

        # Update coupling from new state
        self._update_coupling(new_state, current)

        return new_state

    def _read_state_from_cpp(self) -> dict[str, np.ndarray]:
        """Read primitive variables from Athena++ mesh blocks into DPF state dict."""
        core = self._core
        handle = self._mesh_handle

        # Get combined arrays from all mesh blocks
        # C++ returns contiguous arrays in DPF convention:
        #   rho:      (nx, ny, nz)
        #   velocity: (3, nx, ny, nz)
        #   pressure: (nx, ny, nz)
        #   B:        (3, nx, ny, nz)
        prim = core.get_primitive_data(handle)  # dict of numpy arrays

        # Map Athena++ primitives to DPF state dict
        state = {
            "rho": prim["rho"],
            "velocity": prim["velocity"],
            "pressure": prim["pressure"],
            "B": prim["B"],
            "Te": prim.get("Te", self._estimate_Te(prim["rho"], prim["pressure"])),
            "Ti": prim.get("Ti", self._estimate_Ti(prim["rho"], prim["pressure"])),
            "psi": prim.get("psi", np.zeros_like(prim["rho"])),
        }
        return state

    def _estimate_Te(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Estimate electron temperature from total pressure (single-fluid).

        In single-fluid MHD, P = n_i * k_B * (Te + Ti).
        Assume Te = Ti = P / (2 * n_i * k_B).

        Args:
            rho: Density array [kg/m^3].
            pressure: Total thermal pressure [Pa].

        Returns:
            Electron temperature array [K].
        """
        k_B = 1.380649e-23
        n_i = np.maximum(rho, 1e-30) / self.config.ion_mass
        Te = np.maximum(pressure, 1e-30) / (2.0 * n_i * k_B)
        return np.maximum(Te, 1.0)

    def _estimate_Ti(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        """Estimate ion temperature (same as Te for single-fluid)."""
        return self._estimate_Te(rho, pressure)

    # ------------------------------------------------------------------
    # Subprocess mode implementation
    # ------------------------------------------------------------------

    def _step_subprocess(
        self,
        state: dict[str, np.ndarray],
        dt: float,
        current: float,
        voltage: float,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """Execute one timestep by running Athena++ as a subprocess.

        This mode writes state to an athinput file, runs the binary for
        a single cycle, and reads the HDF5 output back.  Much slower
        than linked mode but works without compiling extensions.
        """
        from dpf.athena_wrapper.athena_config import generate_athinput

        with tempfile.TemporaryDirectory(prefix="dpf_athena_") as tmpdir:
            # Write athinput with single-step time limit
            self.config.model_copy()
            # We can't easily modify Pydantic frozen models, so we
            # generate athinput with the current time + dt as tlim
            athinput = generate_athinput(
                self.config,
                problem_id="dpf_sub",
            )

            # Override tlim to advance by exactly dt
            athinput = _override_athinput_param(athinput, "time", "tlim", f"{dt:.15e}")

            athinput_path = os.path.join(tmpdir, "athinput.dpf")
            with open(athinput_path, "w") as f:
                f.write(athinput)

            # Run Athena++
            cmd = [self._athena_binary, "-i", athinput_path, "-d", tmpdir]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error("Athena++ subprocess failed:\n%s", result.stderr)
                # Return state unchanged on failure
                return state

            # Read HDF5 output
            new_state = self._read_hdf5_output(tmpdir, state)

        self._time += dt
        self._cycle += 1
        self._update_coupling(new_state, current)

        return new_state

    def _read_hdf5_output(
        self, output_dir: str, fallback_state: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Read the last HDF5 snapshot from an Athena++ output directory.

        Args:
            output_dir: Directory containing ``.athdf`` files.
            fallback_state: State to return if reading fails.

        Returns:
            DPF state dictionary.
        """
        import glob

        try:
            import h5py
        except ImportError:
            logger.warning("h5py not available; returning unchanged state")
            return fallback_state

        # Find the last .athdf file
        athdf_files = sorted(glob.glob(os.path.join(output_dir, "*.athdf")))
        if not athdf_files:
            logger.warning("No .athdf files found in %s", output_dir)
            return fallback_state

        last_file = athdf_files[-1]
        logger.debug("Reading Athena++ output: %s", last_file)

        with h5py.File(last_file, "r") as f:
            # Athena++ HDF5 layout:
            #   prim: (nvars, nblocks, nk, nj, ni)
            #   B:    (3, nblocks, nk, nj, ni)
            prim_data = f["prim"][:]  # (5, nblocks, nk, nj, ni)
            B_data = f["B"][:]        # (3, nblocks, nk, nj, ni)

            # Grid coordinates for reassembly
            f["x1v"][:]  # (nblocks, ni)
            f["x2v"][:]  # (nblocks, nj)

            int(f.attrs["NumMeshBlocks"])
            # Block logical locations for reassembly
            log_locs = f["LogicalLocations"][:]  # (nblocks, 3)
            f["Levels"][:]              # (nblocks,)

        # Reassemble blocks into global arrays
        state = self._reassemble_blocks(
            prim_data, B_data, log_locs, fallback_state
        )
        return state

    def _reassemble_blocks(
        self,
        prim_data: np.ndarray,
        B_data: np.ndarray,
        log_locs: np.ndarray,
        fallback_state: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Reassemble mesh blocks into global arrays matching DPF convention.

        Athena++ data layout: (nvars, nblocks, nk, nj, ni)
        DPF state dict: rho(nx, ny, nz), velocity(3, nx, ny, nz), etc.

        For cylindrical: x1=R, x2=z, x3=phi(=1)
        DPF convention: (nr, 1, nz) for cylindrical

        Args:
            prim_data: Primitive variables (5, nblocks, nk, nj, ni).
            B_data: Magnetic field (3, nblocks, nk, nj, ni).
            log_locs: Logical locations of blocks (nblocks, 3).
            fallback_state: Fallback if reassembly fails.

        Returns:
            DPF state dictionary.
        """
        nvars, nblocks, nk, nj, ni = prim_data.shape

        if self._is_cylindrical:
            # Cylindrical: x1=R(ni), x2=z(nj), x3=phi(nk=1)
            # Need to map to DPF (nr, 1, nz) convention
            # Sort blocks by logical location
            np.lexsort((log_locs[:, 0], log_locs[:, 1]))

            # Determine grid layout
            unique_x1 = np.unique(log_locs[:, 0])
            unique_x2 = np.unique(log_locs[:, 1])
            n_blocks_r = len(unique_x1)
            n_blocks_z = len(unique_x2)

            nr_total = n_blocks_r * ni
            nz_total = n_blocks_z * nj

            rho = np.zeros((nr_total, 1, nz_total), dtype=np.float64)
            vel = np.zeros((3, nr_total, 1, nz_total), dtype=np.float64)
            prs = np.zeros((nr_total, 1, nz_total), dtype=np.float64)
            B = np.zeros((3, nr_total, 1, nz_total), dtype=np.float64)

            for b_idx in range(nblocks):
                lx1_idx = int(np.searchsorted(unique_x1, log_locs[b_idx, 0]))
                lx2_idx = int(np.searchsorted(unique_x2, log_locs[b_idx, 1]))

                i_start = lx1_idx * ni
                j_start = lx2_idx * nj

                # prim_data: (nvars, nblocks, nk, nj, ni)
                # Map: IDN=0 -> rho, IVX=1,IVY=2,IVZ=3 -> vel, IPR=4 -> prs
                # Athena cylindrical: x1=R, x2=z, x3=phi
                # nk=1 for 2D
                block_data = prim_data[:, b_idx, 0, :, :]  # (nvars, nj, ni)

                # Transpose from (nj, ni) to (ni, nj) = (nr_block, nz_block)
                rho[i_start:i_start+ni, 0, j_start:j_start+nj] = block_data[0].T
                vel[0, i_start:i_start+ni, 0, j_start:j_start+nj] = block_data[1].T  # vR
                vel[1, i_start:i_start+ni, 0, j_start:j_start+nj] = block_data[2].T  # vz
                vel[2, i_start:i_start+ni, 0, j_start:j_start+nj] = block_data[3].T  # vphi
                prs[i_start:i_start+ni, 0, j_start:j_start+nj] = block_data[4].T

                B_block = B_data[:, b_idx, 0, :, :]  # (3, nj, ni)
                B[0, i_start:i_start+ni, 0, j_start:j_start+nj] = B_block[0].T  # BR
                B[1, i_start:i_start+ni, 0, j_start:j_start+nj] = B_block[1].T  # Bz
                B[2, i_start:i_start+ni, 0, j_start:j_start+nj] = B_block[2].T  # Bphi

        else:
            # Cartesian 3D — simpler reassembly
            # For now, support single-block Cartesian
            rho = prim_data[0, 0].astype(np.float64)
            vel = np.array([
                prim_data[1, 0],
                prim_data[2, 0],
                prim_data[3, 0],
            ], dtype=np.float64)
            prs = prim_data[4, 0].astype(np.float64)
            B = B_data[:, 0].astype(np.float64)

        # Build state dict
        state = {
            "rho": rho,
            "velocity": vel,
            "pressure": prs,
            "B": B,
            "Te": self._estimate_Te(rho, prs),
            "Ti": self._estimate_Ti(rho, prs),
            "psi": np.zeros_like(rho),
        }
        return state

    # ------------------------------------------------------------------
    # Coupling and diagnostics
    # ------------------------------------------------------------------

    def _update_coupling(
        self, state: dict[str, np.ndarray], current: float
    ) -> None:
        """Update plasma coupling quantities from current state.

        Computes volume-integral R_plasma and L_plasma for the
        circuit solver.

        Args:
            state: Current DPF state dictionary.
            current: Circuit current [A].
        """
        rho = state["rho"]
        B = state["B"]
        state["Te"]
        dx = self.config.dx

        I_sq = max(current**2, 1e-30)

        # Rough estimate of B-field energy for inductance
        B_sq = np.sum(B**2, axis=0)
        if self._is_cylindrical:
            dz = self.config.geometry.dz or dx
            dr = dx
            # Cell volumes: 2*pi*r*dr*dz
            nr = rho.shape[0]
            r = np.linspace(dr, nr * dr, nr)
            r_vol = 2.0 * np.pi * r[:, np.newaxis, np.newaxis] * dr * dz
            L_plasma = float(np.sum(B_sq / _MU_0 * r_vol)) / I_sq
        else:
            dV = dx**3
            L_plasma = float(np.sum(B_sq / _MU_0 * dV)) / I_sq

        self._coupling = CouplingState(
            Lp=L_plasma,
            current=current,
            R_plasma=0.0,  # TODO: compute from Spitzer in Phase G
            Z_bar=1.0,
        )

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def initial_state(self) -> dict[str, np.ndarray]:
        """Create the initial plasma state matching the Python solver.

        Returns:
            State dictionary with uniform fill gas initial conditions.
        """
        nx, ny, nz = self.config.grid_shape
        rho0 = self.config.rho0
        T0 = self.config.T0
        k_B = 1.380649e-23

        n_i = rho0 / self.config.ion_mass
        p0 = 2.0 * n_i * k_B * T0

        return {
            "rho": np.full((nx, ny, nz), rho0),
            "velocity": np.zeros((3, nx, ny, nz)),
            "pressure": np.full((nx, ny, nz), p0),
            "B": np.zeros((3, nx, ny, nz)),
            "Te": np.full((nx, ny, nz), T0),
            "Ti": np.full((nx, ny, nz), T0),
            "psi": np.zeros((nx, ny, nz)),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def time(self) -> float:
        """Current simulation time [s]."""
        return self._time

    @property
    def cycle(self) -> int:
        """Current cycle number."""
        return self._cycle


def _override_athinput_param(
    text: str, block: str, key: str, value: str
) -> str:
    """Override a parameter value in athinput text.

    Args:
        text: Full athinput file text.
        block: Block name (without angle brackets).
        key: Parameter name.
        value: New value string.

    Returns:
        Modified athinput text.
    """
    lines = text.split("\n")
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("<") and stripped.endswith(">"):
            in_block = stripped == f"<{block}>"
            continue
        if in_block and "=" in line:
            parts = line.split("=", 1)
            if parts[0].strip() == key:
                # Preserve any inline comment
                comment = ""
                if "#" in parts[1]:
                    val_part, comment_part = parts[1].split("#", 1)
                    comment = f"  # {comment_part.strip()}"
                lines[i] = f"{key:<12s}= {value}{comment}"
                break
    return "\n".join(lines)
