# DPF Unified

[![CI](https://github.com/longanisainhertaco/DPF_Unified/actions/workflows/ci.yml/badge.svg)](https://github.com/longanisainhertaco/DPF_Unified/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dense Plasma Focus (DPF) multi-physics simulator** — a comprehensive, modular simulation framework that couples validated physics kernels (circuit dynamics, magnetohydrodynamics, collisions, radiation transport, neutron production) with modern software infrastructure (REST/WebSocket API, real-time streaming, checkpoint/restart, validation against experimental devices).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Command-Line Interface](#command-line-interface)
- [Configuration Reference](#configuration-reference)
- [Presets](#presets)
- [Server & API](#server--api)
- [Physics Modules](#physics-modules)
- [Validation Suite](#validation-suite)
- [Project Layout](#project-layout)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

Dense Plasma Focus (DPF) devices are compact pulsed-power machines that produce high-temperature, high-density plasmas by magnetically compressing a current-carrying plasma sheath. DPF devices are used for:

- **Neutron production** (DD fusion yields of 10^8–10^11 neutrons/shot)
- **X-ray generation** (for lithography, radiography)
- **Ion beam acceleration**
- **Fundamental plasma physics research**

**DPF Unified** provides a fully-coupled simulation environment that models:

1. **Circuit dynamics** — capacitor discharge, plasma inductance, resistive losses
2. **Plasma fluid dynamics** — compressible MHD with Hall term, resistive diffusion
3. **Collisions & transport** — Spitzer resistivity, temperature relaxation, Braginskii coefficients
4. **Radiation** — Bremsstrahlung cooling, optional flux-limited diffusion
5. **Atomic physics** — Saha ionization equilibrium, ablation models
6. **Diagnostics** — neutron yield, synthetic interferometry, HDF5 time-series output

The code supports both **3D Cartesian** and **2D axisymmetric (r,z) cylindrical** geometries with appropriate metric terms.

---

## Features

### Physics

| Module | Description |
|--------|-------------|
| **Circuit** | Implicit midpoint RLC solver with dynamic plasma inductance and resistance coupling |
| **Fluid/MHD** | WENO5 reconstruction, HLL/HLLC Riemann solvers, Dedner divergence cleaning, Hall MHD |
| **Collisions** | Spitzer electron-ion frequencies, dynamic Coulomb logarithm, implicit temperature relaxation |
| **Radiation** | Bremsstrahlung with Gaunt factor, flux-limited diffusion (FLD) radiation transport |
| **Turbulence** | Buneman anomalous resistivity model with threshold-based activation |
| **Sheath** | Bohm sheath boundary conditions at electrodes |
| **Atomic** | Saha ionization equilibrium, electrode ablation models |
| **Neutron yield** | DD thermonuclear reaction rate integration |

### Numerical Methods

- **Temporal**: Strang operator splitting (collision+radiation ↔ MHD ↔ circuit)
- **Spatial**: 5th-order WENO reconstruction, HLL/HLLC approximate Riemann solvers
- **Diffusion**: Explicit, super-time-stepping (RKL2), or implicit (Crank-Nicolson ADI)
- **Div-B cleaning**: Dedner mixed hyperbolic-parabolic cleaning
- **CFL-adaptive timestep** with circuit and diffusion constraints

### Software Infrastructure

- **CLI** via [Click](https://click.palletsprojects.com/) — `dpf simulate`, `dpf verify`, `dpf serve`
- **REST + WebSocket API** via [FastAPI](https://fastapi.tiangolo.com/) for real-time GUI integration
- **Checkpoint/restart** in HDF5 format
- **Pydantic v2** configuration with JSON Schema export and validation
- **Device presets** for well-known DPF machines (PF-1000, NX2, LLNL-DPF)
- **Validation suite** comparing against published experimental data

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/longanisainhertaco/DPF_Unified.git
cd DPF_Unified

# Install with development dependencies
pip install -e ".[dev]"

# Run a quick simulation (10 steps)
dpf simulate config.json --steps=10

# Verify a configuration file
dpf verify config.json

# Run the full test suite
pytest tests/ -v
```

---

## Installation

### Requirements

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- Core dependencies: NumPy, SciPy, Pydantic v2, Numba, h5py, Click, tqdm, Matplotlib

### Basic Installation

```bash
pip install -e .
```

### Installation with Extras

```bash
# Development tools (pytest, ruff, mypy, coverage)
pip install -e ".[dev]"

# Server/API support (FastAPI, uvicorn, websockets)
pip install -e ".[server]"

# GPU acceleration (CuPy)
pip install -e ".[gpu]"

# MPI parallelization (mpi4py)
pip install -e ".[mpi]"

# Machine learning extensions (PyTorch, transformers)
pip install -e ".[ml]"

# All extras
pip install -e ".[dev,server,gpu,mpi,ml]"
```

### Verifying Installation

```bash
# Check the CLI is accessible
dpf --help

# Verify a config file
dpf verify config.json

# Run unit tests
pytest tests/ -v --tb=short
```

---

## Command-Line Interface

DPF Unified provides a command-line interface via the `dpf` command:

### `dpf simulate` — Run a Simulation

```bash
dpf simulate <config_file> [OPTIONS]

Options:
  --steps INTEGER          Maximum timesteps (default: run to sim_time)
  -o, --output TEXT        Override output HDF5 filename
  --restart PATH           Restart from checkpoint file
  --checkpoint-interval N  Auto-checkpoint every N steps (0=off)
  -v, --verbose            Enable debug logging

Examples:
  # Run 100 steps
  dpf simulate config.json --steps=100

  # Run to completion with custom output
  dpf simulate config.json -o my_run.h5

  # Restart from checkpoint
  dpf simulate config.json --restart=checkpoint.h5

  # Enable auto-checkpointing every 1000 steps
  dpf simulate config.json --checkpoint-interval=1000
```

### `dpf verify` — Validate Configuration

```bash
dpf verify <config_file>

Example:
  dpf verify config.json
  # Output:
  # Configuration is valid:
  #   Grid: [16, 16, 16]
  #   dx: 1.00e-03 m
  #   sim_time: 1.00e-06 s
  #   Circuit: C=1.00e-06 F, V0=15000.0 V
  #   Fluid: weno5, CFL=0.4
```

### `dpf serve` — Start the API Server

```bash
dpf serve [OPTIONS]

Options:
  --host TEXT      Bind address (default: 127.0.0.1)
  --port INTEGER   Port number (default: 8765)
  --reload         Auto-reload on code changes (dev only)

Example:
  dpf serve --host=0.0.0.0 --port=8000
  # Output:
  # Starting DPF server on 0.0.0.0:8000
  #   REST API: http://0.0.0.0:8000/api/health
  #   WebSocket: ws://0.0.0.0:8000/ws/{sim_id}
  #   Docs: http://0.0.0.0:8000/docs
```

---

## Configuration Reference

Configuration files are JSON format validated by Pydantic. All physical units are SI.

### Minimal Configuration

```json
{
  "grid_shape": [16, 16, 16],
  "dx": 1e-3,
  "sim_time": 1e-6,
  "circuit": {
    "C": 1e-6,
    "V0": 15000,
    "L0": 1e-7,
    "anode_radius": 0.005,
    "cathode_radius": 0.01
  }
}
```

### Full Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Top-level** ||||
| `grid_shape` | `[int, int, int]` | *required* | Grid dimensions (nx, ny, nz). For cylindrical: (nr, 1, nz) |
| `dx` | `float` | *required* | Grid spacing [m] |
| `sim_time` | `float` | *required* | Total simulation time [s] |
| `dt_init` | `float` | `null` | Initial timestep [s] (null = auto) |
| `rho0` | `float` | `1e-4` | Initial fill gas density [kg/m³] |
| `T0` | `float` | `300.0` | Initial temperature [K] |
| `anomalous_alpha` | `float` | `0.05` | Buneman anomalous resistivity coefficient |
| `ion_mass` | `float` | `3.34e-27` | Ion mass [kg] (default: deuterium) |
| **circuit** ||||
| `C` | `float` | *required* | Capacitance [F] |
| `V0` | `float` | *required* | Initial voltage [V] |
| `L0` | `float` | *required* | External inductance [H] |
| `R0` | `float` | `0.0` | External resistance [Ω] |
| `anode_radius` | `float` | *required* | Anode radius [m] |
| `cathode_radius` | `float` | *required* | Cathode radius [m] (must be > anode_radius) |
| `ESR` | `float` | `0.0` | Equivalent series resistance [Ω] |
| `ESL` | `float` | `0.0` | Equivalent series inductance [H] |
| **geometry** ||||
| `type` | `str` | `"cartesian"` | Coordinate system: `"cartesian"` or `"cylindrical"` |
| `dz` | `float` | `null` | Axial spacing for cylindrical [m] (default: use dx) |
| **fluid** ||||
| `reconstruction` | `str` | `"weno5"` | Reconstruction scheme |
| `riemann_solver` | `str` | `"hll"` | Riemann solver: `"hll"`, `"hllc"` |
| `cfl` | `float` | `0.4` | CFL number |
| `dedner_ch` | `float` | `0.0` | Dedner cleaning speed (0 = auto) |
| `gamma` | `float` | `1.6667` | Adiabatic index |
| `enable_resistive` | `bool` | `true` | Enable resistive MHD |
| `enable_energy_equation` | `bool` | `true` | Use conservative energy equation |
| `diffusion_method` | `str` | `"explicit"` | `"explicit"`, `"sts"` (RKL2), or `"implicit"` (ADI) |
| `sts_stages` | `int` | `8` | RKL2 super-time-stepping stages (2-32) |
| `implicit_tol` | `float` | `1e-8` | Implicit solver tolerance |
| **collision** ||||
| `coulomb_log` | `float` | `10.0` | Coulomb logarithm (fixed or initial) |
| `dynamic_coulomb_log` | `bool` | `true` | Compute Coulomb log dynamically |
| `sigma_en` | `float` | `1e-19` | Electron-neutral cross-section [m²] |
| **radiation** ||||
| `bremsstrahlung_enabled` | `bool` | `true` | Enable bremsstrahlung cooling |
| `gaunt_factor` | `float` | `1.2` | Gaunt factor |
| `fld_enabled` | `bool` | `false` | Enable flux-limited diffusion transport |
| `flux_limiter` | `float` | `0.333` | Flux limiter λ (0 < λ ≤ 1) |
| **sheath** ||||
| `enabled` | `bool` | `false` | Enable sheath boundary conditions |
| `boundary` | `str` | `"z_high"` | Boundary to apply sheath (`"z_high"`, `"z_low"`) |
| `V_sheath` | `float` | `0.0` | Sheath voltage drop [V] (0 = auto from Te) |
| **boundary** ||||
| `electrode_bc` | `bool` | `false` | Apply electrode B-field BC |
| `axis_bc` | `bool` | `true` | Enforce symmetry at r=0 (cylindrical) |
| **diagnostics** ||||
| `hdf5_filename` | `str` | `"diagnostics.h5"` | Output HDF5 file |
| `output_interval` | `int` | `10` | Steps between scalar outputs |
| `field_output_interval` | `int` | `0` | Steps between field snapshots (0 = off) |

### Example: Cylindrical Configuration

```json
{
  "grid_shape": [32, 1, 64],
  "dx": 5e-4,
  "sim_time": 1e-6,
  "dt_init": 1e-11,
  "geometry": {
    "type": "cylindrical",
    "dz": 1e-3
  },
  "circuit": {
    "C": 1e-6,
    "V0": 15000,
    "L0": 1e-7,
    "R0": 0.01,
    "anode_radius": 0.005,
    "cathode_radius": 0.01
  },
  "radiation": {
    "bremsstrahlung_enabled": true,
    "fld_enabled": true
  },
  "sheath": {
    "enabled": true,
    "boundary": "z_high"
  }
}
```

---

## Presets

DPF Unified includes configuration presets for well-known DPF devices:

| Preset | Device | Energy | Description |
|--------|--------|--------|-------------|
| `tutorial` | Generic | — | Minimal 8³ Cartesian grid for quick tests |
| `pf1000` | PF-1000 (IPPLM Warsaw) | 1 MJ | Largest DPF in Europe, deuterium fill |
| `nx2` | NX2 (NIE Singapore) | 3 kJ | Compact Mather-type DPF |
| `llnl_dpf` | LLNL-DPF | 100 kJ | Research device |
| `cartesian_demo` | Generic | — | 32³ Cartesian with all physics enabled |

### Using Presets

**CLI:**
```bash
# Presets are accessed via the server API or Python
```

**Python:**
```python
from dpf.presets import get_preset, list_presets
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

# List available presets
for p in list_presets():
    print(f"{p['name']}: {p['description']}")

# Load a preset
config = SimulationConfig(**get_preset("pf1000"))
engine = SimulationEngine(config)
summary = engine.run(max_steps=100)
```

**Server API:**
```bash
# List presets
curl http://localhost:8765/api/presets

# Create simulation from preset
curl -X POST http://localhost:8765/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"preset": "tutorial", "max_steps": 100}'
```

---

## Server & API

DPF Unified includes a FastAPI server for real-time simulation control and GUI integration.

### Starting the Server

```bash
dpf serve --host=0.0.0.0 --port=8765
```

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/presets` | List available presets |
| `GET` | `/api/config/schema` | JSON Schema for configuration |
| `POST` | `/api/config/validate` | Validate configuration JSON |
| `POST` | `/api/simulations` | Create a new simulation |
| `GET` | `/api/simulations/{id}` | Get simulation status |
| `POST` | `/api/simulations/{id}/start` | Start simulation |
| `POST` | `/api/simulations/{id}/pause` | Pause simulation |
| `POST` | `/api/simulations/{id}/resume` | Resume simulation |
| `POST` | `/api/simulations/{id}/stop` | Stop simulation |
| `GET` | `/api/simulations/{id}/fields` | Get field data metadata |

### WebSocket Streaming

Connect to `ws://host:port/ws/{sim_id}` for real-time updates:

**Server → Client (JSON):**
```json
{
  "step": 100,
  "time": 1.5e-7,
  "dt": 1.2e-11,
  "current": 45000.0,
  "voltage": 12000.0,
  "energy_conservation": 0.9995,
  "max_Te": 1.5e6,
  "max_rho": 0.002,
  "Z_bar": 1.0,
  "R_plasma": 0.001,
  "eta_anomalous": 0.0,
  "total_radiated_energy": 1.2,
  "neutron_rate": 1e8,
  "total_neutron_yield": 1e6,
  "finished": false
}
```

**Client → Server (Field Request):**
```json
{
  "type": "request_fields",
  "fields": ["rho", "Te", "B"],
  "downsample": 2
}
```

### Interactive API Docs

When the server is running, visit:
- **Swagger UI**: `http://localhost:8765/docs`
- **ReDoc**: `http://localhost:8765/redoc`

---

## Physics Modules

### Circuit Module (`dpf.circuit`)

Implicit midpoint RLC solver with dynamic plasma coupling:

```
L_total = L0 + L_plasma(t)
R_total = R0 + R_plasma(Te, ne, Z)

L * dI/dt + R * I = V_cap - V_back_emf
C * dV/dt = -I
```

- **Plasma inductance**: Computed from magnetic field energy integral
- **Plasma resistance**: Spitzer + anomalous resistivity

### Fluid/MHD Module (`dpf.fluid`)

Compressible Hall MHD equations:

```
∂ρ/∂t + ∇·(ρv) = 0
∂(ρv)/∂t + ∇·(ρvv + P*I - BB/μ₀) = J×B
∂E/∂t + ∇·((E+P)v - (v·B)B/μ₀) = -Q_rad
∂B/∂t + ∇×E = 0   with E = -v×B + ηJ + J×B/(en_e)
```

Features:
- 5th-order WENO reconstruction for shock capturing
- HLL/HLLC approximate Riemann solvers
- Dedner divergence cleaning (∇·B control)
- Super-time-stepping for diffusion stability
- Cylindrical (r,z) axisymmetric geometry with proper metric terms

### Collision Module (`dpf.collision`)

- **Spitzer resistivity**: η = 5.2×10⁻⁵ Z ln(Λ) / Te^(3/2) [Ω·m]
- **Dynamic Coulomb log**: ln(Λ) = 23 - ln(ne^(1/2) / Te^(3/2))
- **Electron-ion relaxation**: Implicit temperature equilibration
- **Braginskii transport**: Thermal conductivity, viscosity coefficients

### Radiation Module (`dpf.radiation`)

- **Bremsstrahlung**: P_ff = 1.69×10⁻³² g_ff Z² n_e n_i √Te [W/m³]
- **Flux-limited diffusion**: ∂E_rad/∂t = ∇·(D ∇E_rad) - κ(E_rad - aT⁴)
- **Line radiation**: Coronal equilibrium for impurity cooling (optional)

### Turbulence Module (`dpf.turbulence`)

Buneman anomalous resistivity model for current-driven instabilities:

```
η_anom = α * m_e * v_d / (e² n_e)   when v_d > v_critical
```

where v_d = J/(e n_e) is the electron drift velocity.

### Atomic Module (`dpf.atomic`)

- **Saha ionization**: Temperature-dependent ionization equilibrium
- **Ablation**: Electrode material injection into plasma

### Diagnostics Module (`dpf.diagnostics`)

- **HDF5 output**: Time-series scalars + optional field snapshots
- **Neutron yield**: DD thermonuclear reaction rate integration
- **Synthetic interferometry**: Abel transform for line-integrated density
- **Beam-target reactions**: Accelerated ion fusion contributions
- **Checkpointing**: Full state save/restore in HDF5

---

## Validation Suite

DPF Unified includes a validation framework comparing simulations to published experimental data.

### Supported Devices

| Device | Institution | Energy | Peak Current | Neutron Yield | Reference |
|--------|-------------|--------|--------------|---------------|-----------|
| PF-1000 | IPPLM Warsaw | 1 MJ | 2.5 MA | 10^11 | Scholz et al., Nukleonika 51 (2006) |
| NX2 | NIE Singapore | 3 kJ | 400 kA | 10^8 | Lee & Saw, J. Fusion Energy 27 (2008) |
| UNU-ICTP | UNU-ICTP PFF | 3 kJ | 170 kA | 10^8 | Lee et al., Am. J. Phys. 56 (1988) |

### Usage

```python
from dpf.validation.suite import ValidationSuite

suite = ValidationSuite()

# Run simulation and get summary
sim_summary = {
    "peak_current_A": 2.3e6,
    "peak_current_time_s": 5.2e-6,
    "energy_conservation": 0.98,
    "neutron_yield": 8e10,
}

# Validate against PF-1000
result = suite.validate_full("PF-1000", sim_summary)
print(f"Score: {result.overall_score:.1%}")
print(f"Passed: {result.passed}")

# Generate report
report = suite.report({"PF-1000": result})
print(report)
```

### Validation Metrics

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Peak current | 15-20% | Maximum discharge current |
| Current timing | 20-25% | Time to peak current |
| Energy conservation | 5% | E_total / E_initial at end |
| Neutron yield | 1 decade | Order-of-magnitude comparison |
| Peak n_e | 50% | Maximum electron density |
| Peak T_e | 50% | Maximum electron temperature |

---

## Project Layout

```
DPF_Unified/
├── README.md                 # This file
├── pyproject.toml            # Package metadata, dependencies
├── config.json               # Example Cartesian configuration
├── config_cylindrical.json   # Example cylindrical configuration
├── .github/
│   └── workflows/
│       └── ci.yml            # CI pipeline (lint, test, smoke)
├── src/
│   └── dpf/
│       ├── __init__.py
│       ├── config.py         # Pydantic v2 configuration models
│       ├── constants.py      # Physical constants (from scipy)
│       ├── engine.py         # Simulation orchestrator
│       ├── presets.py        # Named device presets
│       ├── species.py        # Ion species definitions
│       │
│       ├── core/
│       │   ├── bases.py      # ABC interfaces (CouplingState, etc.)
│       │   └── field_manager.py
│       │
│       ├── circuit/
│       │   └── rlc_solver.py # Implicit RLC circuit solver
│       │
│       ├── fluid/
│       │   ├── mhd_solver.py        # 3D Cartesian MHD
│       │   ├── cylindrical_mhd.py   # 2D axisymmetric MHD
│       │   ├── eos.py               # Ideal gas EOS
│       │   ├── tabulated_eos.py     # Tabulated EOS
│       │   ├── constrained_transport.py
│       │   ├── super_time_step.py   # RKL2 diffusion
│       │   ├── implicit_diffusion.py
│       │   ├── nernst.py            # Nernst effect
│       │   ├── viscosity.py         # Braginskii viscosity
│       │   └── gpu_backend.py       # CuPy acceleration
│       │
│       ├── collision/
│       │   └── spitzer.py    # Spitzer resistivity, relaxation
│       │
│       ├── radiation/
│       │   ├── bremsstrahlung.py
│       │   ├── transport.py  # Flux-limited diffusion
│       │   └── line_radiation.py
│       │
│       ├── turbulence/
│       │   └── anomalous.py  # Buneman anomalous resistivity
│       │
│       ├── sheath/
│       │   └── bohm.py       # Bohm sheath BCs
│       │
│       ├── atomic/
│       │   ├── ionization.py # Saha equilibrium
│       │   └── ablation.py   # Electrode ablation
│       │
│       ├── geometry/
│       │   └── cylindrical.py # Metric tensors, operators
│       │
│       ├── amr/
│       │   └── grid.py       # Adaptive mesh refinement
│       │
│       ├── pic/
│       │   └── hybrid.py     # Particle-in-cell hybrid
│       │
│       ├── ml/
│       │   └── __init__.py   # Machine learning extensions
│       │
│       ├── diagnostics/
│       │   ├── hdf5_writer.py
│       │   ├── checkpoint.py
│       │   ├── neutron_yield.py
│       │   ├── interferometry.py
│       │   ├── beam_target.py
│       │   └── derived.py
│       │
│       ├── validation/
│       │   ├── suite.py      # Validation framework
│       │   └── experimental.py # Device reference data
│       │
│       ├── server/
│       │   ├── app.py        # FastAPI application
│       │   ├── models.py     # Pydantic API models
│       │   ├── simulation.py # SimulationManager
│       │   └── encoding.py   # Binary field encoding
│       │
│       └── cli/
│           └── main.py       # Click CLI
│
└── tests/
    ├── conftest.py
    ├── test_circuit.py
    ├── test_collision.py
    ├── test_fluid.py
    ├── test_radiation.py
    ├── test_engine_step.py
    ├── test_integration.py
    ├── test_validation.py
    ├── test_server.py
    └── ... (590+ tests total)
```

---

## Testing

DPF Unified has comprehensive test coverage with 590+ tests.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=dpf --cov-report=term-missing

# Run specific test module
pytest tests/test_circuit.py -v

# Run tests matching a pattern
pytest tests/ -k "test_spitzer" -v

# Skip slow integration tests
pytest tests/ -v -m "not slow"
```

### Test Categories

| Category | Description |
|----------|-------------|
| `test_circuit.py` | RLC solver, energy conservation |
| `test_collision.py` | Spitzer resistivity, temperature relaxation |
| `test_fluid.py` | MHD solver, WENO reconstruction, Riemann |
| `test_radiation.py` | Bremsstrahlung, FLD transport |
| `test_cylindrical.py` | 2D axisymmetric geometry |
| `test_engine_step.py` | Full engine timestep |
| `test_integration.py` | End-to-end simulations |
| `test_validation.py` | Experimental comparison |
| `test_server.py` | REST/WebSocket API |
| `test_e2e_server.py` | Server integration tests |

### Linting

```bash
# Run ruff linter
ruff check src/ tests/

# Run type checker
mypy src/dpf/
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Run the test suite** to ensure all tests pass
4. **Run the linter** (`ruff check src/ tests/`)
5. **Submit a pull request** with a clear description

### Code Style

- Follow PEP 8 with 100-character line limit
- Use type hints for all public functions
- Physics variable names (B, Te, Lp, etc.) are exempt from lowercase conventions
- Docstrings follow NumPy style

### Physics Contributions

When adding new physics modules:
1. Implement the appropriate base class from `dpf.core.bases`
2. Add unit tests with known analytical solutions
3. Add validation against published experimental data if applicable
4. Update this README with the new capability

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## References

### Dense Plasma Focus Physics

1. J.W. Mather, "Formation of a High-Density Deuterium Plasma Focus," *Phys. Fluids* 8, 366 (1965)
2. N.V. Filippov et al., "Dense, high-temperature plasma in a noncylindrical z-pinch compression," *Nucl. Fusion* Suppl. 2, 577 (1962)
3. S. Lee & S.H. Saw, "Numerical experiments on plasma focus neutron yield," *J. Fusion Energy* 27, 292 (2008)
4. M. Scholz et al., "Compression of plasma by plasma in the PF-1000 device," *Nukleonika* 51(1), 79 (2006)

### Numerical Methods

5. C.-W. Shu, "Essentially non-oscillatory and weighted essentially non-oscillatory schemes," *ICASE Report* 97-65 (1997)
6. A. Dedner et al., "Hyperbolic divergence cleaning for the MHD equations," *J. Comput. Phys.* 175, 645 (2002)
7. L. Braginskii, "Transport processes in a plasma," *Rev. Plasma Phys.* 1, 205 (1965)

### DPF Simulation Codes

8. S. Lee, "Radiative Dense Plasma Focus Model," *IEEE Trans. Plasma Sci.* 19(6), 912 (1991)
9. M. Liu, "Soft X-rays from compact plasma focus," PhD Thesis, NIE Singapore (1996)
