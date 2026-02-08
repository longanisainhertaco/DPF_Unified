# DPF Unified

[![CI](https://github.com/longanisainhertaco/DPF_Unified/actions/workflows/ci.yml/badge.svg)](https://github.com/longanisainhertaco/DPF_Unified/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A modern dense plasma focus (DPF) simulator** — built from scratch in Python, targeting high-fidelity multi-physics simulation of plasma focus devices on local hardware (Apple Silicon) and eventually HPC clusters.

---

## Vision

DPF Unified is being built as a complete simulation platform for dense plasma focus research and engineering:

| Layer | Description | Status |
|-------|-------------|--------|
| **Simulation Backend** | Dual-engine MHD solver — Python (NumPy/Numba) fallback + Athena++ C++ primary backend via pybind11. Circuit coupling, radiation, collisions, neutron production | **Active development** |
| **Unity Frontend** | Two-mode UI — *Teaching Mode* (educational visualization) and *Engineering Mode* (parameter sweeps, optimization) | **Planned** |
| **AI Integration** | Surrogate models via [Polymathic.ai](https://polymathic-ai.org/) for fast estimates and inverse design ("what config yields X neutrons?") | **Planned** |
| **HPC Backend** | MPI-parallel and GPU-accelerated solvers for production-grade fidelity | **Planned** |

**Current MVP focus**: Get the simulation backend to the highest fidelity possible, running locally on Apple Silicon (M3 Ultra MacBook Pro / Mac Studio). The Unity frontend and HPC support come after the physics is right.

---

## Current State — Honest Assessment

### Fidelity Grade: 5–6 / 10

> **Grading scale**: Sandia National Laboratories production codes (e.g., ALEGRA, HYDRA) = 8/10. Established open-source codes (Athena++, FLASH, PLUTO) = 6-7/10. Our target for this development cycle = 6/10.

The simulation backend now has a complete V&V (Verification & Validation) framework, full Braginskii anisotropic transport, Powell + Dedner div(B) control, and Numba-parallelized kernels for Apple Silicon. 745+ tests pass with 0 failures. Phases A–E are complete, and Phase F (Athena++ integration) has successfully delivered a dual-engine architecture with backend selection.

### Active Modules (What Actually Runs)

These modules are wired into `engine.py` and execute during every simulation:

| Module | Implementation | Quality |
|--------|----------------|---------|
| **Circuit RLC** | Implicit midpoint solver with dynamic plasma inductance/resistance | Solid — energy conservation to 1% |
| **MHD Solver** | WENO5 reconstruction + HLL Riemann solver, Numba-accelerated | Good — 5th-order convergence verified on smooth data |
| **Two-Temperature Plasma** | Separate Te, Ti with implicit relaxation via Spitzer collision rates | Strong — matches NRL Plasma Formulary |
| **Spitzer Collisions** | Quantum-corrected Coulomb logarithm (Gericke-Murillo-Schlanges), nu_ei, resistivity | Strong — analytically verified |
| **Bremsstrahlung** | Backward Euler cooling with Gaunt factor, stable for large dt | Good |
| **Saha Ionization** | Temperature-dependent Z_bar from tabulated data | Basic but functional |
| **DD Neutron Yield** | Thermonuclear cross-section integration <sigma*v>(Ti) | Implemented |
| **Nernst Effect** | First-order upwind advection of B by temperature gradient | Simplified — operator-split, no gyropolarization |
| **Braginskii Viscosity** | Full anisotropic tensor: eta_0 (parallel) + eta_1, eta_2 (perpendicular) with field-aligned decomposition | Complete — Phase D |
| **Anomalous Resistivity** | Buneman threshold model: eta_anom when v_drift > v_crit | Phenomenological |
| **Cylindrical Geometry** | 2D (r,z) axisymmetric with proper 1/r metric, axis protection at r=0 | Well-implemented |
| **Strang Splitting** | collision/radiation <-> MHD <-> circuit, 2nd-order | Correct |
| **REST API + WebSocket** | FastAPI server with binary field encoding, pause/resume control | Functional, tested |
| **Diagnostics** | HDF5 time-series output, checkpoint/restart framework | Working |

### Recently Integrated (Phases B–D — completed)

These modules were dormant or newly implemented and are now wired into the engine:

| Module | Status | How Activated |
|--------|--------|---------------|
| **Implicit Diffusion (ADI)** | ✅ Active | `fluid.diffusion_method: "implicit"` in config |
| **Super Time-Stepping (RKL2)** | ✅ Active | `fluid.diffusion_method: "sts"` in config |
| **Line Radiation** | ✅ Active | `radiation.line_radiation_enabled: true` + `impurity_fraction > 0` |
| **Constrained Transport** | ✅ Active | Default ON in cylindrical solver |
| **Anisotropic Thermal Conduction** | ✅ Active | `fluid.enable_anisotropic_conduction: true` — Sharma-Hammett slope-limited, field-aligned |
| **Full Braginskii Viscosity** | ✅ Active | `fluid.full_braginskii_viscosity: true` — eta_0 + eta_1 + eta_2 tensor decomposition |
| **Powell 8-wave div(B)** | ✅ Active | `fluid.enable_powell: true` — non-conservative source terms for div(B) control |
| **Dedner GLM Tuning** | ✅ Active | `fluid.dedner_cr` — Mignone-Tzeferacos (2010) optimal ch/cp prescription |

### Dormant Modules (Code Exists, Not Integrated)

These live in `src/dpf/experimental/` to clearly communicate their status:

| Module | Lines | Completeness | Why Dormant |
|--------|-------|--------------|-------------|
| **Adaptive Mesh Refinement** | 755 | Code complete | MHD solvers assume uniform grids; needs solver refactoring |
| **GPU Backend** | ~100 | CuPy detection stub only | No actual GPU kernels; Apple Silicon needs MLX, not CUDA |
| **Hybrid PIC** | 978 | Boris pusher + CIC deposition complete | Never instantiated; kinetic effects are fidelity-6+ |
| **Multi-Species** | 409 | SpeciesMixture class complete | Will be integrated in Phase D (after line radiation validation) |

**Bottom line**: After Phase B integration, ~20-25% of source code remains dormant (AMR, PIC, GPU, multi-species). The core physics pipeline is now substantially complete.

### Verification & Validation (Phase C — completed)

| Benchmark | Status |
|-----------|--------|
| **Resistive diffusion convergence** | ✅ Explicit, ADI, RKL2 — Gaussian B-field vs analytical solution |
| **Orszag-Tang vortex** | ✅ Canonical 2D MHD benchmark (Cartesian) |
| **Cylindrical Sedov blast** | ✅ Best-effort — analytical similarity solution, documents solver limitations |
| **Lee Model comparison** | ✅ 2-phase snowplow model for PF-1000 and NX2 device validation |
| **Sod / Brio-Wu shock tubes** | ✅ Correct wave structure, L1 errors verified |

### Testing Reality

| Category | Status |
|----------|--------|
| **Unit physics** (collision, EOS, circuit, radiation) | **Strong** — verified against analytical formulas |
| **Shock tubes** (Sod, Brio-Wu) | **Good** — correct wave structure, L1 errors reasonable |
| **Convergence studies** | **Good** — diffusion convergence (3 methods), Orszag-Tang, Sedov |
| **Experimental validation** | **Improved** — Lee Model comparison for PF-1000 and NX2 |
| **Braginskii / anisotropic transport** | **Good** — 14 tests covering limits, backward compatibility, field alignment |
| **Dormant module tests** | **Missing** — AMR, GPU, PIC, multi-species have zero coverage |
| **Turbulence/sheath tests** | **Empty** — stub files with no actual tests |

---

## Reference Codes

We study these established MHD codes to guide our development:

### Top 3

| Code | Institution | Why It Matters | What We Learn |
|------|-------------|----------------|---------------|
| **[OpenMHD](https://github.com/zenitani/OpenMHD)** | JAXA (Zenitani) | Compact resistive MHD with CUDA GPU. Excellent for magnetic reconnection — directly relevant to DPF pinch dynamics. | Resistive MHD patterns, HLLD solver reference, GPU kernel design |
| **[FLASH](http://flash.uchicago.edu/)** | U. Chicago | Proven in High Energy Density Physics (HEDP) and laboratory plasma experiments. Multi-physics coupling closest to DPF needs. | Multi-physics architecture, radiation MHD, experimental validation methodology |
| **[MPI-AMRVAC](https://amrvac.org/)** | KU Leuven | Best-in-class div(B) control (Powell + Dedner GLM), excellent shock handling, mature block-structured AMR. | Powell source terms, Dedner tuning, divergence control, AMR patterns |

### Honorable Mentions

- **[Athena++ / AthenaK](https://www.athena-astro.app/)** (Princeton) — Best architecture, Kokkos GPU portability, now integrated as primary C++ backend via pybind11
- **[PLUTO / gPLUTO](https://plutocode.ph.unito.it/)** (Torino) — Hall MHD, new GPU implementation via OpenACC, strong astrophysical MHD
- **[Lee Model](http://plasmafocus.net/)** — DPF-specific semi-empirical code, gold standard for circuit-level validation of plasma focus devices

---

## Roadmap

| Phase | Goal | Target Fidelity | Key Work | Status |
|-------|------|-----------------|----------|--------|
| ~~Phase A~~ | ~~Honest documentation~~ | — | ~~README rewrite, dormant code triage~~ | ✅ Done |
| ~~Phase B~~ | ~~Wire dormant physics~~ | 4/10 | ~~ADI/RKL2 diffusion, line radiation, CT default on~~ | ✅ Done |
| ~~Phase C~~ | ~~Verification & validation~~ | 5/10 | ~~Diffusion convergence, Orszag-Tang, Sedov, Lee Model~~ | ✅ Done |
| ~~Phase D~~ | ~~Physics improvements~~ | 6/10 | ~~Full Braginskii, Powell div-B, anisotropic conduction, Dedner GLM~~ | ✅ Done |
| ~~Phase E~~ | ~~Apple Silicon optimization~~ | 6/10 (faster) | ~~Numba prange in viscosity, CT, Nernst; benchmark suite~~ | ✅ Done |
| ~~Phase F~~ | ~~Athena++ integration~~ | — | ~~Submodule, pybind11, dual-engine, verification, CLI/server~~ | ✅ Done |
| **Phase G** (next) | Athena++ DPF physics | — | Circuit coupling C++, Spitzer η, two-temp, radiation, Braginskii | 🔜 |
| **Phase H** | WALRUS data pipeline | — | Well exporter, batch runner, dataset validator | |
| **Phase I** | WALRUS fine-tuning + AI | — | Surrogate, inverse design, real-time server | |
| **Phase J** | Unity frontend + HPC | — | Teaching/Engineering mode, AthenaK GPU | |

> **AI Integration**: Phases H-I use [Polymathic AI WALRUS](https://huggingface.co/polymathic-ai/walrus) — a 1.3B-parameter foundation model pretrained on 19 physical systems including MHD. We fine-tune it on DPF simulation data to create fast surrogate models for parameter sweeps, inverse design ("what config yields X neutrons?"), and real-time Unity visualization. See the [forward plan](docs/PLAN.md) for full WALRUS integration architecture.

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

# Athena++ C++ backend (requires building from source)
pip install -e ".[dev,server,athena]"
# See docs/ATHENA_BUILD.md for Athena++ compilation instructions

# All currently useful extras
pip install -e ".[dev,server]"
```

> **Note**: The `pyproject.toml` also lists `gpu` (CuPy), `mpi` (mpi4py), and `ml` (PyTorch) extras. These are placeholders for future work — no GPU kernels, MPI decomposition, or ML models exist yet.

---

## Command-Line Interface

### `dpf simulate` — Run a Simulation

```bash
dpf simulate <config_file> [OPTIONS]

Options:
  --steps INTEGER          Maximum timesteps (default: run to sim_time)
  -o, --output TEXT        Override output HDF5 filename
  --restart PATH           Restart from checkpoint file
  --checkpoint-interval N  Auto-checkpoint every N steps (0=off)
  --backend [python|athena|auto]  MHD solver backend (default: from config)
  -v, --verbose            Enable debug logging

Examples:
  dpf simulate config.json --steps=100
  dpf simulate config.json -o my_run.h5
  dpf simulate config.json --restart=checkpoint.h5
  dpf simulate config.json --backend=athena
```

### `dpf verify` — Validate Configuration

```bash
dpf verify <config_file>
```

### `dpf backends` — Show Available Backends

```bash
dpf backends
```

### `dpf serve` — Start the API Server

```bash
dpf serve [OPTIONS]

Options:
  --host TEXT      Bind address (default: 127.0.0.1)
  --port INTEGER   Port number (default: 8765)
  --reload         Auto-reload on code changes (dev only)
```

---

## Server & API

DPF Unified includes a FastAPI server for real-time simulation control and future Unity frontend integration.

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + backend availability |
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

Connect to `ws://host:port/ws/{sim_id}` for real-time step-by-step updates. Binary field encoding supported with configurable downsampling.

Interactive docs at `http://localhost:8765/docs` when the server is running.

---

## Configuration

Configuration files are JSON, validated by Pydantic v2. All physical units are SI.

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

### Cylindrical Configuration (Recommended for DPF)

```json
{
  "grid_shape": [32, 1, 64],
  "dx": 5e-4,
  "sim_time": 1e-6,
  "dt_init": 1e-11,
  "geometry": { "type": "cylindrical", "dz": 1e-3 },
  "circuit": {
    "C": 1e-6, "V0": 15000, "L0": 1e-7,
    "R0": 0.01, "anode_radius": 0.005, "cathode_radius": 0.01
  },
  "radiation": { "bremsstrahlung_enabled": true, "fld_enabled": true },
  "sheath": { "enabled": true, "boundary": "z_high" }
}
```

Full configuration reference: see `dpf verify <config_file>` for all available fields and defaults.

---

## Presets

| Preset | Device | Energy | Description |
|--------|--------|--------|-------------|
| `tutorial` | Generic | — | Minimal 8x8x8 Cartesian grid for quick tests |
| `pf1000` | PF-1000 (IPPLM Warsaw) | 1 MJ | Largest DPF in Europe |
| `nx2` | NX2 (NIE Singapore) | 3 kJ | Compact Mather-type DPF |
| `llnl_dpf` | LLNL-DPF | 100 kJ | Research device |
| `cartesian_demo` | Generic | — | 32x32x32 with all active physics |

```python
from dpf.presets import get_preset
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

config = SimulationConfig(**get_preset("pf1000"))
engine = SimulationEngine(config)
summary = engine.run(max_steps=100)
```

---

## Project Layout

```
DPF_Unified/
├── README.md
├── pyproject.toml
├── config.json                    # Example Cartesian config
├── config_cylindrical.json        # Example cylindrical config
│
├── src/dpf/
│   ├── engine.py                  # Simulation orchestrator (central loop)
│   ├── config.py                  # Pydantic v2 configuration
│   ├── constants.py               # Physical constants
│   ├── presets.py                 # Device presets
│   │
│   ├── circuit/                   # [ACTIVE] RLC circuit solver
│   ├── fluid/                     # [ACTIVE] MHD solvers, EOS, viscosity, Nernst
│   ├── collision/                 # [ACTIVE] Spitzer resistivity, temperature relaxation
│   ├── radiation/                 # [ACTIVE] Bremsstrahlung, FLD transport
│   ├── turbulence/                # [ACTIVE] Anomalous resistivity (Buneman)
│   ├── sheath/                    # [ACTIVE] Bohm sheath BCs
│   ├── atomic/                    # [ACTIVE] Saha ionization
│   ├── geometry/                  # [ACTIVE] Cylindrical metric operators
│   ├── diagnostics/               # [ACTIVE] HDF5, neutron yield, interferometry
│   ├── validation/                # [ACTIVE] Experimental comparison suite
│   ├── verification/              # [ACTIVE] Shock tubes, convergence tests
│   ├── server/                    # [ACTIVE] FastAPI REST + WebSocket
│   ├── cli/                       # [ACTIVE] Click CLI
│   ├── athena_wrapper/            # [ACTIVE] Athena++ C++ pybind11 wrapper
│   ├── core/                      # [ACTIVE] Base classes, field manager
│   │
│   └── experimental/              # [DORMANT] Code exists but not integrated
│       ├── amr/                   #   Adaptive mesh refinement
│       ├── pic/                   #   Hybrid particle-in-cell
│       ├── species.py             #   Multi-species tracking
│       └── gpu_backend.py         #   CuPy detection stub
│
│   ├── benchmarks/                # [ACTIVE] Apple Silicon performance benchmarks
│
├── external/
│   └── athena/                    # Athena++ git submodule (Princeton MHD code)
│
└── tests/                         # 745+ tests (pytest)
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=dpf --cov-report=term-missing

# Run specific module
pytest tests/test_circuit.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Coverage by Module

| Module | Tests | Quality |
|--------|-------|---------|
| Circuit | 20+ | Strong — frequency + energy conservation |
| Collision/Spitzer | 15+ | Strong — matches NRL formulary |
| EOS | 10+ | Strong — numerical roundtrip verified |
| MHD/Fluid | 100+ | Good — WENO5 convergence, shock tubes |
| Radiation | 40+ | Good — scaling laws verified |
| Braginskii/Anisotropic | 14 | Good — limits, backward compat, field alignment |
| V&V Benchmarks | 40+ | Good — diffusion, Orszag-Tang, Sedov, Lee Model |
| Athena++ / dual-engine | 70+ | Good — Sod, Brio-Wu, magnoh, cross-backend, CLI |
| Server/API | 60+ | Good — REST + WebSocket functional |
| Integration | 50+ | Moderate — pipeline runs, peak-value validation |
| Dormant modules | 0 | Missing |

---

## Contributing

Contributions are welcome. When adding physics:

1. Implement against the base classes in `dpf.core.bases`
2. Add unit tests with known analytical solutions
3. Wire into `engine.py` (don't create dormant code)
4. Validate against published data where applicable
5. Run `pytest tests/ -v` and `ruff check src/ tests/`

---

## License

MIT License — see [LICENSE](LICENSE).

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
7. S.I. Braginskii, "Transport processes in a plasma," *Rev. Plasma Phys.* 1, 205 (1965)
8. T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal MHD," *J. Comput. Phys.* 208, 315 (2005)

### Reference Codes

9. S. Zenitani, "OpenMHD: Open-source magnetohydrodynamics code," [github.com/zenitani/OpenMHD](https://github.com/zenitani/OpenMHD)
10. FLASH Center, "FLASH User's Guide," University of Chicago, [flash.uchicago.edu](http://flash.uchicago.edu/)
11. R. Keppens et al., "MPI-AMRVAC 3.0," *Astron. Astrophys.* 673, A66 (2023)
12. S. Lee, "Radiative Dense Plasma Focus Model," *IEEE Trans. Plasma Sci.* 19(6), 912 (1991)

### DPF Simulation Codes

13. S. Lee, "Radiative Dense Plasma Focus Model," *IEEE Trans. Plasma Sci.* 19(6), 912 (1991)
14. M. Liu, "Soft X-rays from compact plasma focus," PhD Thesis, NIE Singapore (1996)
