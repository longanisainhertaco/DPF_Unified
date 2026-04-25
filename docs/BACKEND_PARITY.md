# Backend Physics Parity Matrix

This document tracks which physics modules are available on each solver backend.
Updated 2026-04-08 by grep audit of solver source files.

## Legend

- **ACTIVE**: Called by default in the solver loop
- **AVAILABLE**: Implemented, behind a config flag (default off)
- **CIRCUIT**: Handled by the circuit solver, not the MHD backend
- **NOT IMPL**: Not implemented in this backend

## Matrix

| # | Physics Module | Python | MLX | Athena++ | AthenaK | Config Flag |
|---|----------------|--------|-----|----------|---------|-------------|
| 1 | Resistive MHD (Spitzer/Lee-More) | ACTIVE | ACTIVE | AVAILABLE | AVAILABLE | `enable_resistive` |
| 2 | Braginskii anisotropic viscosity | AVAILABLE | AVAILABLE | NOT IMPL | NOT IMPL | `enable_braginskii_viscosity` |
| 3 | Braginskii anisotropic conduction | AVAILABLE | ACTIVE | NOT IMPL | NOT IMPL | `enable_anisotropic_conduction` |
| 4 | Hall MHD term | ACTIVE | ACTIVE (cyl only) | AVAILABLE | AVAILABLE | `enable_hall` |
| 5 | Nernst effect | NOT IMPL | AVAILABLE (no-op) | NOT IMPL | NOT IMPL | `enable_nernst` |
| 6 | Two-temperature (Te/Ti) | AVAILABLE | AVAILABLE | NOT IMPL | NOT IMPL | `two_temperature` |
| 7 | Bremsstrahlung radiation | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `bremsstrahlung_enabled` |
| 8 | Line radiation | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `line_radiation_enabled` |
| 9 | Constrained transport (div B) | AVAILABLE | ACTIVE | ACTIVE | AVAILABLE | `use_ct` |
| 10 | Dedner GLM (div B) | ACTIVE | AVAILABLE | NOT IMPL | AVAILABLE | `enable_powell` |
| 11 | WENO5-Z reconstruction | ACTIVE | ACTIVE | AVAILABLE (PPM) | AVAILABLE | `reconstruction` |
| 12 | HLLD Riemann solver | AVAILABLE | AVAILABLE | ACTIVE | AVAILABLE | `riemann_solver` |
| 13 | HLLS Riemann solver | NOT IMPL | ACTIVE | NOT IMPL | NOT IMPL | `riemann_solver` |
| 14 | SSP-RK3 time integration | ACTIVE | ACTIVE | NOT IMPL | NOT IMPL | `time_integrator` |
| 15 | Dual-energy entropy tracer | NOT IMPL | ACTIVE | AVAILABLE | AVAILABLE | `use_dual_energy` |
| 16 | RKL2 super time-stepping | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `diffusion_method` |
| 17 | Implicit resistive diffusion | AVAILABLE | AVAILABLE | NOT IMPL | NOT IMPL | `diffusion_method` |
| 18 | Multi-species advection | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `species_config` |
| 19 | PIC hybrid coupling | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `enable_mhd_coupling` |
| 20 | Anomalous resistivity | NOT IMPL | AVAILABLE | NOT IMPL | NOT IMPL | `anomalous_resistivity` |
| 21 | Electrode boundary conditions | AVAILABLE | AVAILABLE | AVAILABLE | NOT IMPL | `electrode_bc` |
| 22 | Crowbar switch model | CIRCUIT | CIRCUIT | CIRCUIT | CIRCUIT | `crowbar_*` |
| 23 | Cylindrical coordinates | ACTIVE | ACTIVE | ACTIVE | NOT IMPL | `coordinates` |

## Backend Summary

### MLX (`src/dpf/metal/mlx_solver.py`)
- **Most complete backend** for DPF simulation
- 9 active modules, 8 available behind flags
- Conservative formulation (total energy), dual-energy entropy tracer
- Float32 on GPU with entropy switching for robustness
- Cylindrical (r,z) only; 3D Cartesian not yet implemented in MLX

### Python (`src/dpf/fluid/mhd_solver.py`)
- **Teaching and fallback** backend
- Non-conservative pressure equation (dp/dt, NOT dE/dt) — violates
  Rankine-Hugoniot at shocks. See warning at line 1 of mhd_solver.py.
- Useful for: prototyping, debugging, small-grid parameter scans
- Not recommended for production accuracy

### Athena++ (`src/dpf/athena_wrapper/athena_engine.py`)
- **C++ reference** backend via pybind11
- Physics selected at compile time (`configure.py --flux=hlld --coord=cylindrical`)
- PPM + HLLD + CT by default
- Limited Python-side physics: resistivity and electrode BCs only
- Electrode BC path has a known segfault issue (pybind11 read-only array views)

### AthenaK (`src/dpf/athenak_wrapper/athenak_solver.py`)
- **Kokkos C++** backend via subprocess
- Cartesian mesh only (no cylindrical coordinates)
- Runtime physics selection via athinput file
- Cannot run standard DPF cylindrical simulations
- Useful for: Cartesian blast/shock verification, GPU portability testing

## Behavior When Unsupported Physics Is Requested

**Current behavior: silent skip.** If a user enables a physics module via config
that is not implemented in their selected backend, the backend ignores the flag
without warning. For example:

- Setting `enable_braginskii_viscosity=True` with `backend="athenak"` silently
  runs without viscosity
- Setting `two_temperature=True` with `backend="athena"` silently runs
  single-temperature

**This is a known usability gap.** Future work: add explicit warnings in
`engine/core.py` at backend initialization when unsupported flags are enabled.

## Recommended Backend by Use Case

| Use Case | Recommended Backend | Why |
|----------|-------------------|-----|
| Production DPF simulation | MLX | Most complete physics, GPU-accelerated |
| Teaching / algorithm study | Python | Human-readable, no compilation |
| Cross-validation / reference | Athena++ | Independent C++ implementation |
| Cartesian verification | AthenaK | Stock Kokkos test problems |
| Maximum accuracy (float64) | Python or Athena++ | Metal has no float64 |
