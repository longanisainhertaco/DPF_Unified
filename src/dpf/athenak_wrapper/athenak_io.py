"""AthenaK VTK output reader and state conversion.

Reads VTK legacy binary files produced by AthenaK and converts them
to the DPF state dictionary format used by all solvers.

AthenaK VTK format:
- VTK DataFile Version 2.0
- Binary, big-endian float32
- STRUCTURED_POINTS dataset
- Variables: dens, velx, vely, velz, eint, bcc1, bcc2, bcc3
- Dimensions are cell+1 in each direction

Example::

    from dpf.athenak_wrapper.athenak_io import read_vtk_file, convert_to_dpf_state

    data = read_vtk_file("output/vtk/Blast.mhd_w_bcc.00005.vtk")
    state = convert_to_dpf_state(data, gamma=5.0/3.0)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def read_vtk_file(filepath: str | Path) -> dict[str, Any]:
    """Read an AthenaK VTK binary output file.

    Args:
        filepath: Path to .vtk file.

    Returns:
        Dictionary with keys:
        - ``"time"``: Simulation time (float)
        - ``"cycle"``: Cycle number (int)
        - ``"dims"``: Grid dimensions [nx, ny, nz] (cell counts)
        - ``"origin"``: Grid origin [x1min, x2min, x3min]
        - ``"spacing"``: Cell spacing [dx1, dx2, dx3]
        - ``"variables"``: Dict mapping variable names to flat numpy arrays

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file format is unexpected.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"VTK file not found: {filepath}")

    with open(filepath, "rb") as f:
        content = f.read()

    # Decode header as ASCII (binary data may contain non-ASCII bytes)
    text = content.decode("ascii", errors="replace")

    # Parse header metadata
    header_match = re.search(
        r"time=\s*([\d.eE+-]+)\s+level=\s*(\d+)\s+nranks=\s*(\d+)\s+cycle=(\d+)",
        text,
    )
    sim_time = float(header_match.group(1)) if header_match else 0.0
    cycle = int(header_match.group(4)) if header_match else 0

    # Parse grid dimensions
    dims_match = re.search(r"DIMENSIONS\s+(\d+)\s+(\d+)\s+(\d+)", text)
    if dims_match is None:
        raise ValueError(f"DIMENSIONS not found in VTK file: {filepath}")
    # VTK DIMENSIONS are vertex counts — cells = vertices - 1
    vertex_dims = [int(dims_match.group(i)) for i in range(1, 4)]
    cell_dims = [max(d - 1, 1) for d in vertex_dims]

    # Parse origin and spacing
    origin_match = re.search(
        r"ORIGIN\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", text
    )
    origin = [float(origin_match.group(i)) for i in range(1, 4)] if origin_match else [0.0] * 3

    spacing_match = re.search(
        r"SPACING\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", text
    )
    spacing = [float(spacing_match.group(i)) for i in range(1, 4)] if spacing_match else [1.0] * 3

    # Parse CELL_DATA count
    cell_data_match = re.search(r"CELL_DATA\s+(\d+)", text)
    if cell_data_match is None:
        raise ValueError(f"CELL_DATA not found in VTK file: {filepath}")
    n_cells = int(cell_data_match.group(1))

    # Find all SCALARS declarations and read binary data
    scalars_pattern = re.compile(r"SCALARS\s+(\w+)\s+float")
    lookup_pattern = b"LOOKUP_TABLE default\n"

    variables: dict[str, np.ndarray] = {}
    for match in scalars_pattern.finditer(text):
        var_name = match.group(1)
        # Find the LOOKUP_TABLE default line after this SCALARS declaration
        pos = match.start()
        lt_idx = content.index(lookup_pattern, pos)
        data_start = lt_idx + len(lookup_pattern)
        data_end = data_start + n_cells * 4
        raw = content[data_start:data_end]
        arr = np.frombuffer(raw, dtype=">f4").astype(np.float64)
        variables[var_name] = arr

    logger.debug(
        "Read VTK: time=%.4e, cycle=%d, dims=%s, variables=%s",
        sim_time, cycle, cell_dims, list(variables.keys()),
    )

    return {
        "time": sim_time,
        "cycle": cycle,
        "dims": cell_dims,
        "origin": origin,
        "spacing": spacing,
        "variables": variables,
    }


def convert_to_dpf_state(
    vtk_data: dict[str, Any],
    gamma: float = 5.0 / 3.0,
) -> dict[str, np.ndarray]:
    """Convert AthenaK VTK data to DPF state dictionary.

    Maps AthenaK variable names to DPF state keys:
    - dens -> rho
    - velx, vely, velz -> velocity (3, nx, ny, nz)
    - eint -> pressure (via p = (gamma-1) * rho * eint)
    - bcc1, bcc2, bcc3 -> B (3, nx, ny, nz)
    - Te, Ti computed from pressure and density

    Args:
        vtk_data: Output from :func:`read_vtk_file`.
        gamma: Adiabatic index for pressure computation.

    Returns:
        DPF state dict with keys: rho, velocity, pressure, B, Te, Ti, psi
    """
    variables = vtk_data["variables"]
    dims = vtk_data["dims"]
    nx, ny, nz = dims

    # Reshape: VTK stores in (nz, ny, nx) order for STRUCTURED_POINTS
    # but for 2D (nz=1), it's just (ny, nx)
    shape_3d = (nz, ny, nx) if nz > 1 else (ny, nx) if ny > 1 else (nx,)

    def _get_field(name: str, default: float = 0.0) -> np.ndarray:
        if name in variables:
            return variables[name].reshape(shape_3d)
        return np.full(shape_3d, default)

    rho = _get_field("dens", 1.0)
    velx = _get_field("velx")
    vely = _get_field("vely")
    velz = _get_field("velz")
    eint = _get_field("eint")
    bcc1 = _get_field("bcc1")
    bcc2 = _get_field("bcc2")
    bcc3 = _get_field("bcc3")

    # Pressure from internal energy: p = (gamma - 1) * rho * eint
    pressure = (gamma - 1.0) * rho * eint

    # Stack vector fields
    velocity = np.stack([velx, vely, velz], axis=0)
    B = np.stack([bcc1, bcc2, bcc3], axis=0)

    # Temperature from ideal gas: T = p / (n * k_B)
    # Using T = p * m_i / (rho * k_B) for single species
    k_B = 1.380649e-23  # Boltzmann constant
    m_D = 3.34358377e-27  # Deuterium mass
    Te = np.where(rho > 0, pressure * m_D / (rho * k_B), 0.0)
    Ti = Te.copy()

    # Divergence cleaning scalar (not available in AthenaK VTK output)
    psi = np.zeros_like(rho)

    return {
        "rho": rho,
        "velocity": velocity,
        "pressure": pressure,
        "B": B,
        "Te": Te,
        "Ti": Ti,
        "psi": psi,
    }


def find_latest_vtk(output_dir: str | Path) -> Path | None:
    """Find the VTK file with the highest snapshot number.

    Args:
        output_dir: Directory containing VTK output files.

    Returns:
        Path to the latest VTK file, or None if none found.
    """
    output_dir = Path(output_dir)
    vtk_dir = output_dir / "vtk"
    if not vtk_dir.exists():
        vtk_dir = output_dir

    vtk_files = sorted(vtk_dir.glob("*.vtk"))
    return vtk_files[-1] if vtk_files else None


def find_all_vtk(output_dir: str | Path) -> list[Path]:
    """Find all VTK files in output directory, sorted by snapshot number.

    Args:
        output_dir: Directory containing VTK output files.

    Returns:
        List of Paths to VTK files, sorted by snapshot number.
    """
    output_dir = Path(output_dir)
    vtk_dir = output_dir / "vtk"
    if not vtk_dir.exists():
        vtk_dir = output_dir

    return sorted(vtk_dir.glob("*.vtk"))
