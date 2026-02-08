"""Translate DPF SimulationConfig to Athena++ athinput format.

The Athena++ input file format is INI-style with ``<block>`` headers
and ``key = value`` entries.  This module converts DPF's Pydantic
:class:`~dpf.config.SimulationConfig` into the corresponding athinput
text that Athena++ can parse via :class:`ParameterInput`.

The mapping is designed to preserve DPF semantics while exploiting
Athena++'s native cylindrical coordinate support, HLLD Riemann solver,
and constrained transport.

Example::

    from dpf.config import SimulationConfig
    from dpf.athena_wrapper.athena_config import generate_athinput

    config = SimulationConfig.from_file("pf1000.json")
    athinput_text = generate_athinput(config)
    # Write to file or pass directly to C++ ParameterInput
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dpf.config import SimulationConfig


def generate_athinput(
    config: SimulationConfig,
    *,
    problem_id: str = "dpf_sim",
    output_dir: str | None = None,
) -> str:
    """Generate Athena++ input file text from a DPF SimulationConfig.

    Args:
        config: Validated DPF simulation configuration.
        problem_id: Problem identifier for output filenames.
        output_dir: Optional output directory path (``-d`` flag).

    Returns:
        Complete athinput file content as a string.
    """
    cc = config.circuit
    fc = config.fluid
    gc = config.geometry
    dc = config.diagnostics

    nx, ny, nz = config.grid_shape
    dx = config.dx

    # Determine coordinate system and grid parameters
    is_cylindrical = gc.type == "cylindrical"

    if is_cylindrical:
        # For cylindrical: x1=R, x2=z, nx3=1 (axisymmetric)
        nr, nz_grid = nx, nz
        dr = dx
        dz = gc.dz if gc.dz is not None else dx
        r_min = dr  # Avoid axis singularity
        r_max = cc.cathode_radius
        z_min = 0.0
        z_max = nz_grid * dz
        coord_system = "cylindrical"
    else:
        # Cartesian 3D
        nr = nx
        nz_grid = nz
        r_min = 0.0
        r_max = nx * dx
        z_min = 0.0
        z_max = nz * dx
        coord_system = "cartesian"

    # Map DPF Riemann solver names to Athena++
    riemann_map = {
        "hll": "hll",
        "hllc": "hllc",
        "hlld": "hlld",
        "roe": "roe",
        "llf": "llf",
    }
    flux = riemann_map.get(fc.riemann_solver, "hlld")

    # Map reconstruction
    recon_map = {
        "plm": 2,
        "ppm": 3,
        "weno5": 3,  # Athena++ uses xorder=3 for PPM/WENO
    }
    xorder = recon_map.get(fc.reconstruction, 2)

    # History output interval — every N steps or dt-based
    hst_dt = config.sim_time / max(dc.output_interval, 1)

    # HDF5 output interval
    if dc.field_output_interval > 0:
        hdf5_dt = config.sim_time / max(dc.field_output_interval, 1)
    else:
        hdf5_dt = config.sim_time  # Single output at end

    lines = []

    # Comment header
    lines.append("<comment>")
    lines.append("problem = DPF simulation via dpf-unified (Athena++ backend)")
    lines.append(f"configure = --prob=dpf_zpinch --coord={coord_system} -b --flux={flux}")
    lines.append("")

    # Job block
    lines.append("<job>")
    lines.append(f"problem_id  = {problem_id}")
    lines.append("")

    # History output
    lines.append("<output1>")
    lines.append("file_type   = hst")
    lines.append(f"dt          = {hst_dt:.6e}")
    lines.append("")

    # HDF5 output
    lines.append("<output2>")
    lines.append("file_type = hdf5")
    lines.append("variable  = prim")
    lines.append(f"dt        = {hdf5_dt:.6e}")
    lines.append("")

    # Time block
    lines.append("<time>")
    lines.append(f"cfl_number  = {fc.cfl}")
    lines.append("nlim        = -1")
    lines.append(f"tlim        = {config.sim_time:.6e}")
    lines.append("integrator  = vl2")
    lines.append(f"xorder      = {xorder}")
    lines.append("ncycle_out  = 100")
    if config.dt_init is not None:
        lines.append(f"dt          = {config.dt_init:.6e}")
    lines.append("")

    # Mesh block
    lines.append("<mesh>")
    if is_cylindrical:
        lines.append(f"nx1        = {nr}")
        lines.append(f"x1min      = {r_min:.6e}")
        lines.append(f"x1max      = {r_max:.6e}")
        lines.append("ix1_bc     = reflecting")
        lines.append("ox1_bc     = outflow")
        lines.append("")
        lines.append(f"nx2        = {nz_grid}")
        lines.append(f"x2min      = {z_min:.6e}")
        lines.append(f"x2max      = {z_max:.6e}")
        lines.append("ix2_bc     = reflecting")
        lines.append("ox2_bc     = outflow")
        lines.append("")
        lines.append("nx3        = 1")
        lines.append("x3min      = 0.0")
        lines.append("x3max      = 6.283185307")
        lines.append("ix3_bc     = periodic")
        lines.append("ox3_bc     = periodic")
    else:
        lines.append(f"nx1        = {nx}")
        lines.append(f"x1min      = {0.0:.6e}")
        lines.append(f"x1max      = {r_max:.6e}")
        lines.append("ix1_bc     = reflecting")
        lines.append("ox1_bc     = outflow")
        lines.append("")
        lines.append(f"nx2        = {ny}")
        lines.append(f"x2min      = {0.0:.6e}")
        lines.append(f"x2max      = {ny * dx:.6e}")
        lines.append("ix2_bc     = periodic")
        lines.append("ox2_bc     = periodic")
        lines.append("")
        lines.append(f"nx3        = {nz}")
        lines.append(f"x3min      = {z_min:.6e}")
        lines.append(f"x3max      = {z_max:.6e}")
        lines.append("ix3_bc     = periodic")
        lines.append("ox3_bc     = periodic")
    lines.append("")

    # Meshblock decomposition (try to keep blocks manageable)
    mb_nx1 = min(nr, 64)
    mb_nx2 = min(nz_grid, 64) if is_cylindrical else min(ny, 64)
    mb_nx3 = 1 if is_cylindrical else min(nz, 64)
    lines.append("<meshblock>")
    lines.append(f"nx1        = {mb_nx1}")
    lines.append(f"nx2        = {mb_nx2}")
    lines.append(f"nx3        = {mb_nx3}")
    lines.append("")

    # Hydro block
    lines.append("<hydro>")
    lines.append(f"gamma      = {fc.gamma:.4f}")
    lines.append("")

    # Problem block — DPF-specific parameters
    lines.append("<problem>")
    lines.append("# DPF simulation parameters")
    lines.append(f"d          = {config.rho0:.6e}    # initial density [kg/m^3]")
    lines.append(f"T0         = {config.T0:.2f}       # initial temperature [K]")
    lines.append(f"ion_mass   = {config.ion_mass:.6e}  # ion mass [kg]")
    lines.append("# Circuit parameters")
    lines.append(f"V0         = {cc.V0:.2f}           # initial voltage [V]")
    lines.append(f"C          = {cc.C:.6e}            # capacitance [F]")
    lines.append(f"L0         = {cc.L0:.6e}           # external inductance [H]")
    lines.append(f"R0         = {cc.R0:.6e}           # external resistance [Ohm]")
    lines.append(f"anode_r    = {cc.anode_radius:.6e}  # anode radius [m]")
    lines.append(f"cathode_r  = {cc.cathode_radius:.6e}# cathode radius [m]")
    lines.append("# Physics toggles")
    lines.append(f"enable_resistive = {int(fc.enable_resistive)}")
    lines.append(f"enable_nernst    = {int(fc.enable_nernst)}")
    lines.append(f"enable_viscosity = {int(fc.enable_viscosity)}")
    lines.append(f"anomalous_alpha  = {config.anomalous_alpha:.4f}")
    lines.append("")

    return "\n".join(lines)


def write_athinput(
    config: SimulationConfig,
    path: str,
    **kwargs,
) -> str:
    """Generate and write athinput file to disk.

    Args:
        config: DPF simulation configuration.
        path: Output file path.
        **kwargs: Passed to :func:`generate_athinput`.

    Returns:
        The generated athinput text.
    """
    text = generate_athinput(config, **kwargs)
    with open(path, "w") as f:
        f.write(text)
    return text
