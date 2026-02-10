# DPF Unified — Claude Code Project Configuration

## Project Overview
Dense Plasma Focus (DPF) multi-physics MHD simulator with tri-engine architecture:
- **Python engine**: NumPy/Numba MHD solver (src/dpf/fluid/, 11 modules, ~5,300 LOC) — fallback
- **Athena++ engine**: Princeton C++ MHD code (external/athena/ git submodule, pybind11) — primary
- **AthenaK engine**: Kokkos C++ MHD code (external/athenak/ submodule, subprocess) — GPU-ready
- **Orchestration**: Python (engine.py, config.py, server, diagnostics, AI/ML)
- **Hardware**: M3 Pro MacBook Pro, 36GB unified memory, Python 3.11

## Architecture

engine.py selects backend via config.fluid.backend:
  backend="python"  → MHDSolver / CylindricalMHDSolver (src/dpf/fluid/)
  backend="athena"  → AthenaPPSolver (src/dpf/athena_wrapper/)
                        → pybind11 → Athena++ C++ (external/athena/)
  backend="athenak" → AthenaKSolver (src/dpf/athenak_wrapper/)
                        → subprocess → AthenaK binary (external/athenak/)

All three backends implement PlasmaSolverBase (src/dpf/core/bases.py).
Auto-resolution priority: athenak > athena > python.
State dict: {rho, velocity, pressure, B, Te, Ti, psi} as NumPy arrays.

## Model Assignments for Agents

| Task | Model | Rationale |
|------|-------|-----------|
| Test scaffolding, boilerplate, file creation, monitoring | haiku | Fast, cheap, sufficient for templates |
| Implementation (Python, C++ source, tests) | sonnet | Balance of speed + capability |
| Physics design, architecture, V&V analysis, debugging | opus | Deep reasoning for complex physics |
| Athena++ / AthenaK C++ problem generators, source terms | opus | Physics + systems programming expertise |
| WALRUS integration (surrogate.py, Hydra configs, tensor shapes) | opus | Complex API mapping, tensor math, ML architecture |
| WALRUS fine-tuning setup, hyperparameter selection | opus | ML training expertise, hardware memory planning |
| WALRUS data pipeline (Well export, batch runner, validation) | sonnet | Data engineering, HDF5 schema compliance |
| Hardware compatibility checks (MPS, MLX, memory estimation) | sonnet | Systems profiling, dependency resolution |
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
Completed: A (docs), B (wire physics), C (V&V), D (Braginskii), E (Apple Silicon), F (Athena++ integration), G (Athena++ DPF physics), H (WALRUS pipeline), I (AI features), J.1 (AthenaK integration), M (Metal GPU optimization), N (hardening & cross-backend V&V), O (physics accuracy), P (engine accuracy: WENO-Z, SSP-RK3, HLLD defaults, Metal resistive MHD)
Planned: J.2+ (backlog)

### Test Patterns
- Phase tests: test_phase_{letter}_{topic}.py
- Module tests: test_{module}.py
- Use conftest.py fixtures (8x8x8 grid, default_circuit_params)
- @pytest.mark.slow for anything > 1 second
- CI gate: >= 745 tests (currently 1475 total, 1353 non-slow, 122 slow, 45 Phase O + 23 Phase P)

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
| AthenaK wrapper | src/dpf/athenak_wrapper/ (4 files) |
| AthenaK submodule | external/athenak/ |
| AthenaK setup/build scripts | scripts/setup_athenak.sh, scripts/build_athenak.sh |
| AthenaK research docs | docs/ATHENAK_RESEARCH.md |
| Phase J tests (AthenaK) | tests/test_phase_j_athenak.py, tests/test_phase_j_cli_server.py (57 tests) |
| Metal GPU package | src/dpf/metal/ (6 files) |
| Metal device manager | src/dpf/metal/device.py |
| Metal stencil ops | src/dpf/metal/metal_stencil.py |
| Metal Riemann solver | src/dpf/metal/metal_riemann.py |
| Metal MHD solver | src/dpf/metal/metal_solver.py |
| MLX surrogate | src/dpf/metal/mlx_surrogate.py |
| Metal benchmarks | src/dpf/benchmarks/metal_benchmark.py |
| Metal production tests | tests/test_metal_production.py (35 tests) |
| Phase N cross-backend tests | tests/test_phase_n_cross_backend.py (17 tests) |
| Phase O physics accuracy tests | tests/test_phase_o_physics_accuracy.py (45 tests) |
| Phase P engine accuracy tests | tests/test_phase_p_accuracy.py (22 non-slow + 1 slow) |
| Slash commands | .claude/commands/ (18 commands) |

## Iterative Accuracy Workflow

This is the core development loop for physics accuracy improvement. Follow this cycle for every physics feature, solver enhancement, or accuracy improvement:

### The Cycle: Create → Test → Rate → Research → Improve → Repeat

1. **Create**: Implement the physics feature or solver improvement
   - Follow established coding conventions (NumPy style docstrings, type hints, 100-char lines)
   - Use reference literature (Miyoshi & Kusano for HLLD, Borges et al. for WENO-Z, Shu-Osher for SSP-RK3)
   - Write clean, vectorized code (prefer torch tensor ops for Metal, numpy for Python engine)

2. **Test**: Write comprehensive tests covering the new feature
   - Unit tests: instantiation, single step, uniform state preservation
   - Shock tests: Sod, Brio-Wu stability (no NaN, no blowup)
   - Conservation tests: energy, mass, momentum conservation
   - Convergence tests: measure order of accuracy on smooth problems
   - Cross-backend parity: compare against Python engine or Athena++
   - Mark slow tests (>1s) with `@pytest.mark.slow`
   - Run full suite: `pytest tests/ -x -q` (non-slow) and `pytest tests/ -x -q -m slow` (slow)

3. **Rate Accuracy**: Assess current fidelity grade (scale of 1-10)
   - Consider: reconstruction order, Riemann solver sophistication, time integrator accuracy, precision mode
   - Reference scale: Sandia production codes = 8/10, established open-source (Athena++, FLASH) = 6-7/10
   - Document the rating and what limits it

4. **Research**: Investigate what would improve accuracy further
   - Use opus agents for deep physics research (characteristic decomposition, higher-order methods)
   - Consult reference implementations (OpenMHD, MPI-AMRVAC, Athena++)
   - Evaluate cost vs. benefit (e.g., characteristic WENO5 = ~500 LOC for marginal gain → not worth it)
   - Check academic literature (arXiv, JCP, ApJS) for modern best practices

5. **Improve**: Implement the most impactful improvement identified by research
   - Prioritize: correctness > accuracy > performance
   - Prefer well-tested algorithms from the literature over novel approaches
   - When in doubt, use float64 mode for maximum accuracy

6. **Repeat**: Go back to step 2 and verify the improvement didn't regress anything

### Accuracy Milestones

| Fidelity | What It Takes | Status |
|----------|---------------|--------|
| 6.5-7/10 | PLM + HLL + SSP-RK2 + float32 + CT | Phase M/N |
| 8.0/10 | WENO5 + HLLD + SSP-RK3 + float32 | Phase O (interim) |
| 8.7/10 | WENO5-Z + HLLD + SSP-RK3 + float64 + CT + MC limiter | Phase O |
| **8.9/10** | **+ Python WENO-Z + SSP-RK3 + HLLD defaults + Metal resistive MHD** | **Phase P (current)** |
| 9.0/10 | + characteristic decomposition, or Athena++ PPM+characteristic | Future |
| 9.5/10 | + AMR, higher-order CT, production HPC scaling | Future |

### Maximum Accuracy Configuration (Phase O)
```python
MetalMHDSolver(
    reconstruction="weno5",      # 5th-order WENO-Z (Borges et al. 2008)
    riemann_solver="hlld",       # Miyoshi & Kusano (2005) 4-wave solver
    time_integrator="ssp_rk3",   # Shu-Osher (1988) 3rd-order SSP
    precision="float64",         # CPU float64 for maximum accuracy
    use_ct=True,                 # Constrained transport for div(B)=0
    limiter="mc",                # Monotonized Central slope limiter
)
```

## Workflow Patterns

1. Before any work: `pytest tests/ -x -q` to verify baseline
2. After C++ changes: rebuild with `make -j8` in external/athena/
3. After pybind11 changes: `pip install -e ".[dev,athena]"`
4. After any change: `ruff check src/ tests/`
5. Kill stale processes before heavy compute: `pkill -f "pytest|python.*dpf"`
6. Commit naming: "Phase X.Y: description" (e.g., "Phase F.4: CLI and server backend integration")
7. After AthenaK source changes: `bash scripts/build_athenak.sh`
8. AthenaK uses subprocess mode only (no pybind11 linking)
9. WALRUS training/fine-tuning: use separate venv due to pinned torch==2.5.1
10. WALRUS inference: load checkpoint → instantiate IsotropicModel → RevIN normalize → forward → denormalize delta → add residual
11. Well format export: always use `grid_type="cartesian"`, axis order `[x, y, z]`, float32
12. Follow the iterative accuracy workflow: Create → Test → Rate → Research → Improve → Repeat

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

### Lessons Learned (Phase J.1 — AthenaK)

21. **Kokkos NO Metal backend**: AthenaK uses Kokkos for performance portability. Kokkos supports Serial, OpenMP, CUDA, HIP, SYCL — but NOT Apple Metal. On M3 Pro, use Serial or OpenMP only.
22. **OpenMP sysroot fix**: Homebrew LLVM 19+ on macOS requires `CMAKE_CXX_FLAGS="--sysroot=$(xcrun --show-sdk-path)"` to find system headers. Without this, `<cstdlib>` includes fail.
23. **VTK format**: AthenaK outputs VTK legacy binary files — big-endian float32 (`>f4` NumPy dtype), STRUCTURED_POINTS, CELL_DATA. DIMENSIONS are vertex counts (cells = dims - 1). Variables: dens, velx, vely, velz, eint, bcc1, bcc2, bcc3.
24. **Built-in vs custom pgens**: AthenaK has built-in problem generators selected via `pgen_name` in `<problem>` block at runtime. Custom pgens require `-D PROBLEM=name` at CMake compile time.
25. **No native cylindrical coordinates**: AthenaK uses Cartesian mesh only (DynGR supports curvilinear but only for GR). This is a limitation for DPF cylindrical geometry.
26. **Runtime physics params**: Unlike Athena++ (compile-time `--flux`, `--coord`), AthenaK sets reconstruction, Riemann solver, and ghost zones at runtime via the athinput file.
27. **Shell CWD destruction**: NEVER `rm -rf` the Bash tool's current working directory or any parent of it. The CWD persists across calls, and if deleted, ALL subsequent Bash commands fail with exit code 1 and no output. Fix by using Write tool to recreate the directory, then `cd` to a valid path.

### Lessons Learned (WALRUS Integration)

28. **WALRUS pinned torch version**: WALRUS requires `torch==2.5.1` (exact pin). This can conflict with other ML tools. Use a separate venv for WALRUS training/fine-tuning.
29. **RevIN normalization required**: WALRUS uses Reversible Instance Normalization (RMS-based, sample-wise). Skipping RevIN or using wrong stats produces garbage predictions. Always compute stats per-sample, normalize before model, denormalize after.
30. **Delta prediction mode**: WALRUS predicts state *changes* (`Δu`), NOT absolute states. Reconstruct: `u(t+1) = u(t) + denormalize(model_output)`. Forgetting the residual connection produces zero-mean noise.
31. **Hydra config system**: All WALRUS configuration is via Hydra YAML overrides on the command line. Do NOT modify Python defaults — use `key=value` CLI args or create override YAML files.
32. **Well grid_type must be "cartesian"**: The Well spec uses `grid_type="cartesian"` for uniform grids, NOT `"uniform"`. Our `well_exporter.py` currently uses `"uniform"` — needs fixing.
33. **Well axis ordering**: Well uses `[x, y, z]` spatial axis order. NumPy image-style `[y, x]` or DPF's `[r, z]` must be mapped correctly when creating Well datasets.
34. **WALRUS checkpoint format**: Checkpoints contain `model_state_dict`, `optimizer_state_dict`, and `config`. Load with `model.load_state_dict(ckpt["model_state_dict"])`, NOT `torch.load()` directly into model.
35. **Apple Silicon AMP incompatibility**: PyTorch AMP (automatic mixed precision) does not work reliably on MPS backend. Always set `trainer.enable_amp=False` for Apple Silicon training.
36. **surrogate.py is FULLY IMPLEMENTED** (as of Phase J.2): `DPFSurrogate._load_walrus_model()` instantiates a real `IsotropicModel` from Hydra config, loads weights via `model.load_state_dict()`, sets up RevIN normalization and `ChannelsFirstWithTimeFormatter`. `_walrus_predict()` runs the full inference pipeline. A 4.8GB pretrained checkpoint exists at `models/walrus-pretrained/walrus.pt`. Minimum grid size for WALRUS is 16×16×16 (3D). CPU inference takes ~58s per step.

### Lessons Learned (Phase M — Metal GPU)

37. **Metal float32 only**: Apple Metal GPU has no float64 support. All physics kernels must enforce `torch.float32`. Energy conservation holds to ~1e-7 relative error with float32 (acceptable for MHD stencils with CT).
38. **Numba has no Metal backend**: Confirmed via Numba GitHub Issue #5706. Will never support Apple Metal. Use PyTorch MPS tensor operations instead of custom kernels.
39. **Unified memory ≠ zero-copy for PyTorch**: `torch.from_numpy()` on MPS still copies data to GPU-visible memory. True zero-copy only with MLX (`mx.array(np_data)` shares memory on Apple Silicon).
40. **MPS launch overhead**: For small grids (<32³), MPS kernel launch overhead exceeds the compute benefit. Elementwise ops on 16³ are ~30× slower on MPS vs NumPy. Only use Metal for grids >64³ or compute-bound operations (AI inference).
41. **PLM over WENO5 on Metal**: WENO5's 5-point stencil with nonlinear weights is hard to vectorize efficiently on GPU. PLM with minmod/MC limiter is trivially vectorizable and sufficient for production use at higher resolution.
42. **HLL over HLLD on Metal**: HLLD requires complex conditional branching (4 intermediate states). HLL is 2-wave, fully vectorizable, and works well with PLM reconstruction.
43. **MetalMHDSolver interface alignment**: Engine calls `_compute_dt()` (private) and passes `**kwargs` to `step()`. Metal solver needs `_compute_dt = compute_dt` alias and `**kwargs` in step signature.
44. **WALRUS MPS speedup**: Real benchmark shows 1.57× speedup for WALRUS inference on MPS vs CPU (38s vs 60s per step). MLX should improve this further.
45. **GeometryConfig has no `dy`**: Only `dz` is configurable. For Metal solver, use `dx` for all directions unless `dz` is set.

### Lessons Learned (Phase N — Cross-Backend V&V)

46. **Metal cross-backend parity**: Sod shock L1(rho) < 15% between Python engine (WENO5+HLL) and Metal (PLM+HLL). Different reconstruction orders explain the gap.
47. **Metal test grids must be cubic-ish**: Non-cubic grids with very thin transverse dimensions (e.g., 32×4×4) cause spurious 3D HLL flux behavior. Use 16×16×16 or similar.

### Lessons Learned (Phase O — Physics Accuracy)

48. **HLLD NaN for strong By discontinuities**: Metal HLL produced NaN for Brio-Wu due to catastrophic cancellation in float32 discriminant `(a²+va²)²-4a²van²`. Fix: numerically stable form `(a²-va²)²+4a²Bt²/rho` + NaN guards + velocity clamping + Lax-Friedrichs fallback.
49. **HLLD 8-component solver**: Full HLLD (Miyoshi & Kusano 2005) resolves contact + Alfven waves. Less diffusive than HLL. Select with `riemann_solver="hlld"`. Falls back to HLL where NaN detected.
50. **WENO5-Z over WENO-JS**: WENO-Z (Borges et al. 2008) uses global smoothness indicator `tau5=|beta0-beta2|` for better accuracy at critical points. Weight formula: `alpha_k = d_k * (1 + (tau5/(eps+beta_k))^2)`.
51. **WENO5 FV vs FD formula pitfall**: Jiang-Shu (1996) coefficients (2/6, -7/6, 11/6) and ideal weights (0.1, 0.6, 0.3) are for finite-volume cell-average reconstruction. DPF's cell-centered code uses point values, requiring different coefficients: (3/8, -10/8, 15/8), (-1/8, 6/8, 3/8), (3/8, 6/8, -1/8) with ideal weights d0=1/16, d1=10/16, d2=5/16. Using FV formulas on point values gives ~3rd order instead of 5th.
52. **SSP-RK3 time integration**: 3-stage SSP-RK3 (Shu-Osher 1988): U1=Un+dt*L(Un), U2=3/4*Un+1/4*(U1+dt*L(U1)), Un+1=1/3*Un+2/3*(U2+dt*L(U2)). Verified lower error than SSP-RK2 on smooth problems. Overall solver order ~1.86 (limited by MHD nonlinearity, not temporal).
53. **Float64 precision mode**: `MetalMHDSolver(precision="float64")` forces CPU + float64 for maximum accuracy. MPS only supports float32. Eliminates round-off accumulation. Use for production V&V runs.
54. **Characteristic WENO5 not worth it for Metal**: Requires ~500 LOC of MHD eigenvector computation (L/R matrices), 3 degenerate case handlers, complex torch.where branching. Cost too high vs benefit. Better to use float64 mode or route through Athena++ PPM+characteristic.
55. **Energy floor after WENO5 reconstruction**: Clamp total energy (IEN component) above P_FLOOR after reconstruction to prevent negative energy states reaching Riemann solver. Guard with `if UL_out.shape[0] > IEN:` since convergence tests may pass 1-component tensors.

### Lessons Learned (Phase P — Engine Accuracy)

56. **Python hybrid WENO5 is inconsistent**: The Python MHD solver's WENO5 flux divergence only updates density/momentum in interior cells [2, N-3], while induction, pressure gradient, and other terms use np.gradient on all cells. This boundary mismatch causes instability under sound wave propagation with both RK2 and RK3. For full WENO5+HLLD+SSP-RK3 fidelity, use the Metal engine instead.
57. **CT requires MPS device**: The `emf_from_fluxes_mps()` function in `metal_stencil.py` requires tensors on MPS device. Tests using Metal solver on CPU must set `use_ct=False`.
58. **Resistive diffusion CFL**: The explicit resistive diffusion operator-split step has CFL limit `dt < dx^2 * mu_0 / (2*eta)`. For typical MHD CFL timesteps (~1e-3 s) with dx=0.01, this requires `eta < ~3e-8 Ohm·m`. Large resistivities (>1e-4) cause blowup without sub-cycling.
59. **WENO-Z weights in Python engine**: The Python engine uses FV (Jiang-Shu 1996) candidate polynomials with WENO-Z (Borges 2008) nonlinear weights `alpha_k = d_k * (1 + (tau5/(eps+beta_k))^2)`. FD point-value formulas are unstable in the hybrid scheme. The Metal engine uses FD formulas because it has a fully conservative formulation.
60. **Velocity clamping for hybrid WENO5**: When WENO5 is active in the Python engine, boundary cells get zero `drho_dt` but non-zero `dmom_dt`, causing extreme velocities. The `_euler_stage` method clamps velocity to 10× the local fast magnetosonic speed to prevent multi-stage RK methods from amplifying these boundary artifacts.

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

## Working with AthenaK (Kokkos)

### Setup & Building
```bash
# Clone submodule + Kokkos
bash scripts/setup_athenak.sh

# Build (auto-detects OpenMP vs Serial)
bash scripts/build_athenak.sh          # default: auto-detect
bash scripts/build_athenak.sh serial   # force serial
bash scripts/build_athenak.sh openmp   # force OpenMP
bash scripts/build_athenak.sh blast    # build stock blast problem
```

### CMake Recipes
```bash
# Serial build
cmake -S . -B build_serial -D Kokkos_ENABLE_SERIAL=ON -D PROBLEM=blast

# OpenMP build (M3 Pro)
cmake -S . -B build_omp -D Kokkos_ENABLE_OPENMP=ON \
  -D CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
  -D CMAKE_CXX_FLAGS="--sysroot=$(xcrun --show-sdk-path)" \
  -D PROBLEM=blast
```

### Key Differences from Athena++

| Feature | Athena++ | AthenaK |
|---------|----------|---------|
| Build system | `configure.py` + `make` | CMake + Kokkos |
| GPU support | None (CPU only) | CUDA, HIP, SYCL via Kokkos |
| Physics config | Compile-time (`--flux`, `--coord`) | Runtime (athinput `<mhd>` block) |
| Output format | HDF5 (`.athdf`) | VTK legacy binary |
| Coordinates | Cartesian, cylindrical, spherical | Cartesian only (DynGR for curvilinear) |
| Python binding | pybind11 (linked mode) | Subprocess only |
| Problem generators | Compile separate binaries | Built-in `pgen_name` (runtime) or `-D PROBLEM` |
| Ghost zones | Compile-time `nghost` | Runtime `nghost` in `<mesh>` |

### Variable Mapping (VTK → DPF State)

| AthenaK VTK | DPF State Key | Notes |
|-------------|---------------|-------|
| `dens` | `rho` | Mass density |
| `velx`, `vely`, `velz` | `velocity[0,1,2]` | Stacked as (3, nx, ny, nz) |
| `eint` | → `pressure` | p = (γ-1) × ρ × eint |
| `bcc1`, `bcc2`, `bcc3` | `B[0,1,2]` | Cell-centered B fields |
| — | `Te`, `Ti` | Derived: T = p × m_D / (ρ × k_B) |
| — | `psi` | Not in AthenaK output (set to 0) |

### Stock Problem Generators
- `blast` — spherical blast wave (MHD, standard benchmark)
- `shock_tube` — 1D Riemann problems (Sod, Brio-Wu)
- `resist` — resistive diffusion (tests ohmic_resistivity)
- `linear_wave` — linear MHD wave convergence tests
- `kh` — Kelvin-Helmholtz instability
- `orszag_tang` — Orszag-Tang MHD vortex

### Performance (M3 Pro)
- Serial: ~2.6M zone-cycles/sec (200×200 blast)
- OpenMP 8T: ~4.2M zone-cycles/sec (1.63× speedup)

## Working with WALRUS (Polymathic AI)

WALRUS is a 1.3B-parameter Encoder-Processor-Decoder Transformer for continuum dynamical systems (MIT license, github.com/PolymathicAI/walrus). DPF uses it as a surrogate model for fast parameter sweeps, inverse design, and real-time prediction.

### Installation & Dependencies

WALRUS has **pinned** dependency versions — install in an isolated environment or use extras carefully:

```bash
# WALRUS core (from git — not on PyPI)
pip install git+https://github.com/PolymathicAI/walrus.git

# The Well dataset tools (required by WALRUS)
pip install "the_well[benchmark] @ git+https://github.com/PolymathicAI/the_well@master"

# Key pinned versions (WALRUS requirements):
# torch==2.5.1, numpy==1.26.4, einops~=0.8, h5py>=3.9.0,<4
# hydra-core>=1.3, timm>=1.0, wandb>=0.17.9
```

**Warning**: WALRUS pins `torch==2.5.1` and `numpy==1.26.4`. These may conflict with DPF's broader dependency set. Use a separate venv or conda environment for WALRUS fine-tuning.

### Architecture Overview

```
IsotropicModel (walrus.models)
├── Encoder: SpaceBagAdaptiveDVstrideEncoder (strided conv + stride modulation)
├── Processor: 12-40 SpaceTimeSplitBlocks (axial or full attention)
└── Decoder: AdaptiveDVstrideDecoder (transposed conv, state-dependent)
```

- **Pretrain config**: 768 hidden, 12 blocks (~300M params)
- **Finetune config**: 1408 hidden, 40 blocks (1.3B params)
- **Prediction mode**: Delta — `u(t+1) = u(t) + model(U(t))`
- **Normalization**: RevIN (RMS-based, sample-wise)
- **Loss**: Per-field normalized MAE (L1)
- **Config system**: Hydra (`@hydra.main()`) — all config via YAML overrides

### WALRUS Inference API

```python
from walrus.models import IsotropicModel
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter

# Load model
model = instantiate(config.model, n_states=total_input_fields)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()

with torch.no_grad():
    inputs, y_ref = formatter.process_input(batch, causal_in_time=True, predict_delta=True)
    stats = revin.compute_stats(inputs[0], metadata, epsilon=1e-5)
    normalized_x = revin.normalize_stdmean(inputs[0], stats)
    y_pred = model(normalized_x, inputs[1], inputs[2].tolist(), metadata=metadata)
    y_pred = inputs[0][-y_pred.shape[0]:].float() + revin.denormalize_delta(y_pred, stats)
```

### WALRUS Training (Local, Single GPU / Apple Silicon)

```bash
python train.py \
    distribution=local \
    model=isotropic_model \
    finetune=True \
    optimizer=adam optimizer.lr=1.e-4 \
    trainer.enable_amp=False \
    model.gradient_checkpointing_freq=1 \
    data.module_parameters.batch_size=1 \
    data.module_parameters.n_steps_input=6 \
    data.module_parameters.n_steps_output=1 \
    trainer.prediction_type="delta" \
    model.causal_in_time=True
```

### WALRUS Batch Dict (DataLoader Output)

```python
batch = {
    "input_fields":       Tensor,   # [B, T_in, H, W, (D), C]
    "output_fields":      Tensor,   # [B, T_out, H, W, (D), C]
    "constant_fields":    Tensor,   # [B, H, W, (D), C_const]
    "boundary_conditions": list,    # [[bc_dim0_lo, bc_dim0_hi], ...]
    "padded_field_mask":  Tensor,   # [C] bool
    "field_indices":      dict,     # field_name -> index
    "metadata":           object,
}
```

### The Well HDF5 Format

```
Root attributes: dataset_name, grid_type ("cartesian"), n_spatial_dims,
                 n_trajectories, simulation_parameters
/dimensions/            → spatial coords + time array
/boundary_conditions/   → "WALL"=0, "OPEN"=1, "PERIODIC"=2 with masks
/t0_fields/             → scalars: shape (n_traj, n_steps, nx, ny [,nz]), float32
/t1_fields/             → vectors: shape (n_traj, n_steps, nx, ny [,nz], D), float32
/t2_fields/             → tensors: shape (n_traj, n_steps, nx, ny [,nz], D²), float32
```

**Important**: Well uses axis ordering `[x, y, z]` (NOT image-style `[y, x]`). Each field dataset should have `dim_varying`, `sample_varying`, `time_varying` boolean attributes. Directory layout: `dataset_name/{train,valid,test}/sample_NNNN.hdf5`.

### Apple Silicon Compatibility (M3 Pro, 36GB)

| Scenario | Memory | Feasibility |
|----------|--------|-------------|
| Float16 inference (1.3B) | ~2.6 GB weights | ✅ Easy |
| LoRA fine-tuning (batch=1, grad ckpt) | ~19-25 GB total | ✅ Feasible |
| Full fine-tuning (batch=1, grad ckpt) | ~30-35 GB total | ⚠️ Tight, may OOM |
| MLX inference | ~2.6 GB + faster than MPS | ✅ Recommended path |

- **Gradient checkpointing is essential** for fine-tuning: `model.gradient_checkpointing_freq=1`
- **AMP disabled** on MPS: set `trainer.enable_amp=False`
- **Batch size 1-2**, gradient accumulation 4-8 steps
- **MLX** is preferred over PyTorch MPS for Apple Silicon (native Metal, lower overhead)

### DPF Module Readiness for WALRUS

| Module | Status | Notes |
|--------|--------|-------|
| `field_mapping.py` | ✅ Ready | Bidirectional DPF ↔ Well transforms |
| `well_exporter.py` | ✅ Ready | `grid_type` fixed to `"cartesian"` (was `"uniform"`) |
| `dataset_validator.py` | ✅ Ready | NaN/Inf, schema, energy conservation checks |
| `batch_runner.py` | ✅ Ready | WellExporter API mismatch fixed |
| `surrogate.py` | ✅ **Implemented** | Real WALRUS IsotropicModel loading + RevIN + inference pipeline. 4.8GB checkpoint at `models/walrus-pretrained/walrus.pt`. |
| `confidence.py` | ✅ Ready | Uses `DPFSurrogate()` constructor (`.load()` bug fixed) |
| `realtime_server.py` | ✅ Ready | All 3 API calls fixed (parameter_sweep, inverse, confidence) |

**Status**: WALRUS inference in `surrogate.py` is now fully implemented using IsotropicModel + RevIN + ChannelsFirstWithTimeFormatter. Next step: fine-tune on DPF-specific training data for improved predictions.

## Project Health Notes

### Dormant Modules (No Tests)
| Module | LOC | Status |
|--------|-----|--------|
| AMR (`src/dpf/experimental/amr/`) | 756 | Code complete, needs solver refactoring for non-uniform grids |
| PIC (`src/dpf/experimental/pic/`) | 979 | Boris pusher + CIC deposition complete, never instantiated |
| Multi-species (`src/dpf/experimental/species.py`) | 410 | SpeciesMixture class complete, not integrated |
| GPU stub (`src/dpf/experimental/gpu_backend.py`) | 119 | CuPy detection only, no actual kernels |

### Unused pyproject.toml Dependency Groups
- `[ml]` — placeholder for MLX / Apple Metal ML
- `[gpu]` — CuPy (not useful on Apple Silicon)
- `[mpi]` — mpi4py (no MPI parallelism implemented)

### Config Validator Gaps
These Pydantic models exist in `config.py` but have minimal or no validation:
`CollisionConfig`, `RadiationConfig`, `SheathConfig`, `BoundaryConfig`

### Backend Physics Parity
Physics implemented only in the Python engine (not in Athena++ or AthenaK):
Braginskii viscosity (Python), Nernst effect, RKL2 super time-stepping, ADI implicit diffusion, Hall MHD
