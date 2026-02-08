# DPF Unified — Claude Code Project Configuration

## Project Overview
Dense Plasma Focus (DPF) multi-physics MHD simulator with dual-engine architecture:
- **Python engine**: NumPy/Numba MHD solver (src/dpf/fluid/, 11 modules, ~5,300 LOC) — fallback
- **Athena++ engine**: Princeton C++ MHD code (external/athena/ git submodule) — primary
- **Orchestration**: Python (engine.py, config.py, server, diagnostics, AI/ML)
- **Hardware**: M3 Pro MacBook Pro, 36GB unified memory, Python 3.11

## Architecture

engine.py selects backend via config.fluid.backend:
  backend="python"  → MHDSolver / CylindricalMHDSolver (src/dpf/fluid/)
  backend="athena"  → AthenaPPSolver (src/dpf/athena_wrapper/)
                        → pybind11 → Athena++ C++ (external/athena/)

Both backends implement PlasmaSolverBase (src/dpf/core/bases.py).
State dict: {rho, velocity, pressure, B, Te, Ti, psi} as NumPy arrays.

## Model Assignments for Agents

| Task | Model | Rationale |
|------|-------|-----------|
| Test scaffolding, boilerplate, file creation, monitoring | haiku | Fast, cheap, sufficient for templates |
| Implementation (Python, C++ source, tests) | sonnet | Balance of speed + capability |
| Physics design, architecture, V&V analysis, debugging | opus | Deep reasoning for complex physics |
| Athena++ C++ problem generators, source terms | opus | Physics + systems programming expertise |
| Documentation, README, plan updates | sonnet | Clear technical writing |
| Code review, refactoring | sonnet | Pattern recognition |

## Coding Conventions

### Python
- Physics names: Te, Ti, B, rho, eta, Z_bar, nu_ei (ruff ignores N802/N803/N806/N815/N816)
- Line length: 100 chars. Type hints on all public functions
- Docstrings: NumPy style (Args/Returns/References)
- Imports: absolute only (from dpf.fluid.mhd_solver import ...)
- Tests: pytest.approx() with explicit tolerances, @pytest.mark.slow for >1s

### C++ (Athena++ extensions)
- 2-space indent, PascalCase classes, snake_case functions
- Problem generators: external/athena/src/pgen/
- pybind11 bindings: src/dpf/athena_wrapper/cpp/
- Use Athena++ ParameterInput API: pin->GetReal("problem", "key")
- Register functions in Mesh::InitUserMeshData()

### Phase Numbering
Completed: A (docs), B (wire physics), C (V&V), D (Braginskii), E (Apple Silicon), F (Athena++ integration)
Active: G (Athena++ DPF physics)
Planned: H (WALRUS pipeline), I (AI features), J (backlog)

### Test Patterns
- Phase tests: test_phase_{letter}_{topic}.py
- Module tests: test_{module}.py
- Use conftest.py fixtures (8x8x8 grid, default_circuit_params)
- @pytest.mark.slow for anything > 1 second
- CI gate: >= 745 tests (745 non-slow pass after Phase F)

## Key File Paths

| Purpose | Path |
|---------|------|
| Engine orchestration | src/dpf/engine.py |
| Configuration (Pydantic) | src/dpf/config.py |
| Solver base class | src/dpf/core/bases.py |
| Python MHD solver | src/dpf/fluid/mhd_solver.py |
| Cylindrical solver | src/dpf/fluid/cylindrical_mhd.py |
| Athena++ wrapper | src/dpf/athena_wrapper/ |
| Athena++ submodule | external/athena/ |
| DPF input files | external/athinput/ |
| CLI entry point | src/dpf/cli/main.py |
| CLI/server tests (F.4) | tests/test_phase_f_cli_server.py |
| Dual-engine tests (F.2) | tests/test_dual_engine.py |
| Athena++ verification (F.3) | tests/test_phase_f_verification.py |
| Athena++ binaries | external/athena/bin/ |
| AI/ML modules | src/dpf/ai/ |
| Server API | src/dpf/server/app.py |
| Circuit solver | src/dpf/circuit/rlc_solver.py |
| Device presets | src/dpf/presets.py |
| Tests | tests/ |
| Benchmarks | src/dpf/benchmarks/ |
| Forward plan | docs/PLAN.md |

## Workflow Patterns

1. Before any work: `pytest tests/ -x -q` to verify baseline
2. After C++ changes: rebuild with `make -j8` in external/athena/
3. After pybind11 changes: `pip install -e ".[dev,athena]"`
4. After any change: `ruff check src/ tests/`
5. Kill stale processes before heavy compute: `pkill -f "pytest|python.*dpf"`
6. Commit naming: "Phase X.Y: description" (e.g., "Phase F.4: CLI and server backend integration")

## Lessons Learned (Phases A-E)

1. **Stale processes**: Python/Numba processes from interrupted runs consume all CPU. Always kill before starting: `pkill -f "pytest|python.*dpf"`
2. **Slow tests**: Mark with @pytest.mark.slow. Use `pytest -m "not slow"` for rapid iteration. Full slow suite takes 30+ min on M3 Pro.
3. **Agent permissions**: Background agents can't prompt for Bash approval — run compute-heavy commands from main session, not background agents.
4. **CI gating**: CI requires >= 590 tests. Always add tests with new code. Never delete without replacing.
5. **Numba JIT latency**: First call to @njit takes 1-5s. Use cache=True. Athena++ eliminates this entirely.
6. **Memory on M3 Pro (36GB)**: Use small grids for unit tests (8x8x8). Multiple parallel pytest processes can exhaust memory.
7. **HDF5 conflicts**: Use ":memory:" for diagnostics in test configs. Use tmp_path fixture for file-writing tests.
8. **Physics naming**: Ruff must ignore N802/N803/N806 for physics variables (Te, Ti, B, Z_bar).
9. **Athena++ global state**: Athena++ cannot be re-initialized in the same process (signal handlers, global vars). Use module-scoped test fixtures and singleton pattern for linked-mode engines.
10. **Athena++ nghost**: Default compile uses nghost=2, which only supports xorder<=2 (PLM). PPM (xorder=3) and WENO5 require nghost>=3.
11. **ruser_mesh_data**: Stock Athena++ problem generators (magnoh.cpp, etc.) do NOT allocate ruser_mesh_data. Accessing it will segfault. Only custom problem generators that call AllocateRealUserMeshDataField() can use it.
12. **Separate Athena++ binaries**: Different problem generators require separate compiled binaries (e.g., athena_sod for shock_tube, athena_briowu for MHD shock_tube, athena for magnoh).
13. **Athena++ HDF5 format**: Variable ordering in .athdf files is NOT fixed — read VariableNames attribute to build var_idx map. Time attribute is scalar, not array.

## Working with Athena++ C++

### Building
```bash
cd external/athena
python configure.py --prob=dpf_zpinch --coord=cylindrical -b --flux=hlld \
    --cxx=clang++-apple -omp -hdf5
make clean && make -j8
```

### Key Athena++ Extension Points
- `EnrollUserExplicitSourceFunction()` — circuit coupling, radiation
- `EnrollUserBoundaryFunction()` — electrode BCs
- `EnrollUserTimeStepFunction()` — circuit timestep constraint
- `Mesh::UserWorkInLoop()` — volume-integral diagnostics (R_plasma, L_plasma)
- `CalcMagDiffCoeff_` — custom resistivity (Spitzer, anomalous)
- `ThermalFluxAniso()` — Braginskii anisotropic conduction
- `ViscousFluxAniso()` — Braginskii anisotropic viscosity

### Key Athena++ Source Files
- `src/pgen/magnoh.cpp` — z-pinch reference problem
- `src/pgen/resist.cpp` — resistive diffusion validation
- `src/field/field_diffusion/` — Ohmic/Hall/Ambipolar
- `src/hydro/hydro_diffusion/` — viscosity/conduction
- `src/field/ct.cpp` — constrained transport
- `src/eos/general/` — tabulated/general EOS
