# DPF Unified

Dense Plasma Focus multi-physics simulator combining validated physics kernels with modern software infrastructure.

## Quick Start

```bash
pip install -e ".[dev]"
dpf simulate config.json --steps=100
dpf verify config.json
```

## Architecture

- **Circuit**: Implicit midpoint RLC solver with plasma inductance coupling
- **Fluid/MHD**: WENO5 reconstruction, HLL Riemann, Dedner div-cleaning, Braginskii transport
- **Collisions**: Spitzer frequencies, dynamic Coulomb log, implicit temperature relaxation
- **Diagnostics**: HDF5 time-series output

## Testing

```bash
pytest tests/ -v
```

## Project Layout

```
src/dpf/
  config.py          # Pydantic v2 configuration
  constants.py       # Physical constants (scipy)
  engine.py          # Simulation orchestrator
  core/bases.py      # ABC interfaces (CouplingState, PlasmaSolverBase, etc.)
  circuit/           # RLC circuit solver
  fluid/             # Hall MHD solver + EOS
  collision/         # Spitzer + Braginskii
  diagnostics/       # HDF5 writer
  cli/               # Click CLI
```
