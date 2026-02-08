"""Translate DPF SimulationConfig to AthenaK athinput format.

AthenaK uses the same INI-style input file format as Athena++, but physics
parameters (reconstruction, Riemann solver, ghost zones) are runtime rather
than compile-time.  This simplifies input generation significantly.

Key differences from Athena++ athinput:
- ``<mhd>`` block replaces ``<hydro>`` (reconstruction, rsolver here)
- ``ohmic_resistivity`` is a runtime parameter in ``<mhd>``
- ``nghost`` is runtime in ``<mesh>``
- Problem generator selected via ``pgen_name`` in ``<problem>`` for built-in
  generators, or compiled in via ``-D PROBLEM=name`` for custom ones.
- No native cylindrical coordinates — uses Cartesian mesh only.

Example::

    from dpf.config import SimulationConfig
    from dpf.athenak_wrapper.athenak_config import generate_athenak_input

    config = SimulationConfig.from_file("pf1000.json")
    text = generate_athenak_input(config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dpf.config import SimulationConfig


def generate_athenak_input(
    config: SimulationConfig,
    *,
    problem_id: str = "dpf_sim",
    pgen_name: str | None = None,
    n_steps: int | None = None,
    output_vtk: bool = True,
    output_hst: bool = True,
    vtk_dt: float | None = None,
) -> str:
    """Generate AthenaK input file text from a DPF SimulationConfig.

    Args:
        config: Validated DPF simulation configuration.
        problem_id: Basename for output files.
        pgen_name: Built-in problem generator name (e.g. "shock_tube").
            If None, assumes a custom pgen was compiled in.
        n_steps: Override maximum cycle count (``nlim``).
        output_vtk: Enable VTK output for field data.
        output_hst: Enable history (.hst) output for scalar diagnostics.
        vtk_dt: Time between VTK snapshots (default: sim_time / 10).

    Returns:
        Complete AthenaK athinput file content as a string.
    """
    cc = config.circuit
    fc = config.fluid
    nx, ny, nz = config.grid_shape
    dx = config.dx

    # Reconstruct mapping: DPF names -> AthenaK names
    recon_map = {
        "plm": "plm",
        "ppm": "ppm4",
        "weno5": "wenoz",
    }
    reconstruct = recon_map.get(fc.reconstruction, "ppm4")

    # Riemann solver mapping
    rsolver_map = {
        "hll": "hll",
        "hlld": "hlld",
        "roe": "roe",
        "llf": "llf",
    }
    rsolver = rsolver_map.get(fc.riemann_solver, "hlld")

    # Ghost zones: PPM4/WENOZ need >= 4
    nghost = 4 if reconstruct in ("ppm4", "wenoz") else 2

    # Grid extents
    x1max = nx * dx
    x2max = ny * dx
    x3max = nz * dx

    # Time control
    nlim = n_steps if n_steps is not None else -1
    tlim = config.sim_time

    # VTK output interval
    if vtk_dt is None:
        vtk_dt = tlim / 10.0

    lines = []

    # Job block
    lines.append("<job>")
    lines.append(f"basename  = {problem_id}")
    lines.append("")

    # Mesh block
    lines.append("<mesh>")
    lines.append(f"nghost    = {nghost}")
    lines.append(f"nx1       = {nx}")
    lines.append("x1min     = 0.0")
    lines.append(f"x1max     = {x1max:.10e}")
    lines.append("ix1_bc    = outflow")
    lines.append("ox1_bc    = outflow")
    lines.append("")
    lines.append(f"nx2       = {ny}")
    lines.append("x2min     = 0.0")
    lines.append(f"x2max     = {x2max:.10e}")
    if ny == 1:
        lines.append("ix2_bc    = periodic")
        lines.append("ox2_bc    = periodic")
    else:
        lines.append("ix2_bc    = outflow")
        lines.append("ox2_bc    = outflow")
    lines.append("")
    lines.append(f"nx3       = {nz}")
    lines.append("x3min     = 0.0")
    lines.append(f"x3max     = {x3max:.10e}")
    if nz == 1:
        lines.append("ix3_bc    = periodic")
        lines.append("ox3_bc    = periodic")
    else:
        lines.append("ix3_bc    = outflow")
        lines.append("ox3_bc    = outflow")
    lines.append("")

    # Meshblock (single block for subprocess mode)
    lines.append("<meshblock>")
    lines.append(f"nx1       = {nx}")
    lines.append(f"nx2       = {ny}")
    lines.append(f"nx3       = {nz}")
    lines.append("")

    # Time block
    lines.append("<time>")
    lines.append("evolution  = dynamic")
    lines.append("integrator = rk2")
    lines.append(f"cfl_number = {fc.cfl}")
    lines.append(f"nlim       = {nlim}")
    lines.append(f"tlim       = {tlim:.10e}")
    lines.append("ndiag      = 1")
    lines.append("")

    # MHD block
    lines.append("<mhd>")
    lines.append("eos         = ideal")
    lines.append(f"reconstruct = {reconstruct}")
    lines.append(f"rsolver     = {rsolver}")
    lines.append(f"gamma       = {fc.gamma:.10f}")
    if fc.enable_resistive:
        # Default Spitzer-like resistivity — actual value set at runtime
        lines.append("ohmic_resistivity = 1.0e-6")
    lines.append("")

    # Problem block
    lines.append("<problem>")
    if pgen_name is not None:
        lines.append(f"pgen_name  = {pgen_name}")

    # DPF-specific parameters (available to custom problem generators)
    lines.append(f"rho0       = {config.rho0:.10e}")
    lines.append(f"T0         = {config.T0:.6e}")
    lines.append(f"V0         = {cc.V0:.6e}")
    lines.append(f"C_cap      = {cc.C:.10e}")
    lines.append(f"L0         = {cc.L0:.10e}")
    lines.append(f"R0         = {cc.R0:.10e}")
    lines.append(f"anode_r    = {cc.anode_radius:.10e}")
    lines.append(f"cathode_r  = {cc.cathode_radius:.10e}")
    lines.append(f"gamma      = {fc.gamma:.10f}")
    lines.append("")

    # Output blocks
    output_idx = 1
    if output_hst:
        lines.append(f"<output{output_idx}>")
        lines.append("file_type  = hst")
        lines.append(f"dt         = {vtk_dt:.10e}")
        lines.append("")
        output_idx += 1

    if output_vtk:
        lines.append(f"<output{output_idx}>")
        lines.append("file_type  = vtk")
        lines.append("variable   = mhd_w_bcc")
        lines.append(f"dt         = {vtk_dt:.10e}")
        lines.append("")

    return "\n".join(lines) + "\n"
