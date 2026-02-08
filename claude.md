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

## Agent Usage Patterns

Use parallel agents (Task tool) when work can be split into independent units. Key scenarios:

### When to Use Agents
- **Writing test files**: Launch one agent per test file (e.g., 9 agents for 10 test files). Each agent writes a complete test file independently. Use `subagent_type=Bash` or `subagent_type=general-purpose`.
- **Exploring codebase**: Use `subagent_type=Explore` for broad searches across multiple modules.
- **Independent file creation**: When creating multiple unrelated source files simultaneously.

### When NOT to Use Agents
- Sequential edits to the same file (agents can't coordinate on shared files)
- Simple single-file searches (use Glob/Grep directly)
- Tasks requiring context from previous agent results (run sequentially instead)

### Agent Best Practices
- Provide complete context in the prompt: file paths, function signatures, mock strategies, coding conventions
- For test writing agents, include: module under test path, key classes/functions, mock patterns (especially lazy imports), existing fixture names
- Check agent results with TaskOutput before proceeding to verification
- Run `ruff check` and `pytest` from main session after agents complete (agents can't prompt for Bash approval)

### Mock Patterns for AI Tests
- Torch/WALRUS/optuna: Use `pytest.importorskip()` at module level + `monkeypatch.setattr` for lazy imports
- Lazy imports inside methods: monkeypatch the *original* module path (e.g., `dpf.engine.SimulationEngine`), NOT the importing module
- Heavy class `__init__`: Use `object.__new__(ClassName)` to bypass `__init__`, then set attributes manually
- FastAPI endpoints: Use `TestClient` from `fastapi.testclient`, send body as JSON and query params in URL

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
Completed: A (docs), B (wire physics), C (V&V), D (Braginskii), E (Apple Silicon), F (Athena++ integration), G (Athena++ DPF physics), H (WALRUS pipeline), I (AI features)
Planned: J (backlog)

### Test Patterns
- Phase tests: test_phase_{letter}_{topic}.py
- Module tests: test_{module}.py
- Use conftest.py fixtures (8x8x8 grid, default_circuit_params)
- @pytest.mark.slow for anything > 1 second
- CI gate: >= 745 tests (currently 1129 total, 1103 non-slow)

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
| AI/ML modules | src/dpf/ai/ (10 files) |
| AI server (FastAPI router) | src/dpf/ai/realtime_server.py |
| AI surrogate model | src/dpf/ai/surrogate.py |
| AI inverse design | src/dpf/ai/inverse_design.py |
| AI hybrid engine | src/dpf/ai/hybrid_engine.py |
| AI confidence/ensemble | src/dpf/ai/confidence.py |
| AI instability detector | src/dpf/ai/instability_detector.py |
| WALRUS field mapping | src/dpf/ai/field_mapping.py |
| WALRUS Well exporter | src/dpf/ai/well_exporter.py |
| WALRUS batch runner | src/dpf/ai/batch_runner.py |
| WALRUS dataset validator | src/dpf/ai/dataset_validator.py |
| Server API | src/dpf/server/app.py |
| Circuit solver | src/dpf/circuit/rlc_solver.py |
| Device presets | src/dpf/presets.py |
| Tests | tests/ |
| Benchmarks | src/dpf/benchmarks/ |
| Forward plan | docs/PLAN.md |
| Phase H tests (WALRUS pipeline) | tests/test_phase_h_*.py (4 files, ~90 tests) |
| Phase I tests (AI features) | tests/test_phase_i_*.py (6 files, ~140 tests) |
| Slash commands (AI) | .claude/commands/ (7 AI commands) |

## Workflow Patterns

1. Before any work: `pytest tests/ -x -q` to verify baseline
2. After C++ changes: rebuild with `make -j8` in external/athena/
3. After pybind11 changes: `pip install -e ".[dev,athena]"`
4. After any change: `ruff check src/ tests/`
5. Kill stale processes before heavy compute: `pkill -f "pytest|python.*dpf"`
6. Commit naming: "Phase X.Y: description" (e.g., "Phase F.4: CLI and server backend integration")

## Lessons Learned (Phases A-I)

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

### Lessons Learned (Phases H-I)

14. **Lazy import monkeypatching**: When modules use lazy imports inside methods (e.g., `from dpf.engine import SimulationEngine` inside a method body), monkeypatch must target the original module path (`dpf.engine.SimulationEngine`), NOT the importing module.
15. **Pydantic v2 access**: `SimulationConfig` uses Pydantic v2. Access fields via attributes (`config.circuit.V0`), NOT subscript (`config.circuit["V0"]`).
16. **FastAPI endpoint signatures**: When endpoints have `(body_param: list[...], query_param: int = 10)`, tests must send body as JSON and query params in URL: `client.post("/path?n_steps=5", json=list_data)`.
17. **NumPy array gotchas**: `np.array(["a", "b", "c"])` succeeds (creates string dtype array). `np.array([{"k": "v"}])` succeeds (creates object dtype array). Don't assume these raise errors.
18. **Ruff import ordering**: When `pytest.importorskip()` calls must precede conditional imports, use `# noqa: E402, I001`.
19. **B023 lambda loop variable**: Don't use `lambda: func(var)` inside loops where `var` changes — the lambda captures by reference.
20. **Parallel test agents**: Writing 10 test files with 9 parallel agents (231 tests) completes in minutes. Always provide full context in agent prompts.

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
