# DPF Unified — Developer Workflow Guide

A practical, actionable guide for developing, testing, and deploying the DPF-Unified dense plasma focus simulator.

---

## Table of Contents

1. [Development Workflow](#1-development-workflow)
2. [Testing Strategy](#2-testing-strategy)
3. [Git Workflow](#3-git-workflow)
4. [Agent Commands Reference](#4-agent-commands-reference)
5. [Troubleshooting Common Issues](#5-troubleshooting-common-issues)

---

## 1. Development Workflow

### 1.1 Phase-Based Development

DPF-Unified follows a lettered phase system (A through AB+). Each phase focuses on a specific area:

| Phase Range | Focus Area | Examples |
|-------------|-----------|----------|
| A-E | Core physics & optimization | Docs, wire physics, V&V, Braginskii, Apple Silicon |
| F-G | Athena++ C++ backend | Submodule, pybind11, DPF physics in C++ |
| H-I | AI/ML integration | WALRUS pipeline, surrogate, inverse design |
| J | AthenaK + WALRUS live | Kokkos wrapper, real model inference |
| M-N | Metal GPU & hardening | PyTorch MPS kernels, cross-backend V&V |
| O-P | Physics accuracy | WENO5-Z, HLLD, SSP-RK3, float64 |
| S-Z, AA-AB | DPF-specific physics | Snowplow, crowbar, radial compression, calibration |

**Starting a new phase:**

1. Verify baseline: `pytest tests/ -x -q` (all non-slow tests pass)
2. Create phase tests first (TDD): `tests/test_phase_{letter}_{topic}.py`
3. Implement the feature in source files
4. Run linter: `ruff check src/ tests/`
5. Run full suite: `pytest tests/ -x -q`
6. Commit: `git commit -m "Phase X.Y: description"`

### 1.2 The Iterative Accuracy Workflow

This is the core development loop for physics improvements:

```
Create → Test → Rate → Research → Improve → Repeat
```

| Step | What You Do |
|------|-------------|
| **Create** | Implement the physics feature using reference literature |
| **Test** | Write unit, shock, conservation, convergence, and cross-backend tests |
| **Rate** | Assess fidelity grade (1-10 scale; Sandia=8, Athena++=6-7) |
| **Research** | Investigate improvements via opus agents and academic literature |
| **Improve** | Implement the highest-impact improvement found |
| **Repeat** | Go back to Test — verify no regressions |

**Accuracy milestones:**

| Fidelity | Configuration |
|----------|--------------|
| 6.5-7/10 | PLM + HLL + SSP-RK2 + float32 + CT |
| 8.0/10 | WENO5 + HLLD + SSP-RK3 + float32 |
| 8.7/10 | WENO5-Z + HLLD + SSP-RK3 + float64 + CT + MC limiter |
| **8.9/10** | + Python WENO-Z + SSP-RK3 + HLLD defaults + Metal resistive MHD |

### 1.3 Working with Claude Code Agents

DPF-Unified is designed to be developed with Claude Code agent assistance. The workflow:

```
Research → Plan → Verify → Approve → Build → Test → Troubleshoot → Document → Commit
```

**Agent model assignments:**

| Task Type | Model | Why |
|-----------|-------|-----|
| Test scaffolding, boilerplate | haiku | Fast, cheap |
| Implementation (Python, C++, tests) | sonnet | Speed + capability balance |
| Physics design, V&V, debugging | opus | Deep reasoning |
| WALRUS integration, tensor math | opus | Complex ML architecture |

**Parallel agent usage:** Launch multiple agents for independent work:
- One agent per test file (up to 9 parallel agents for 10 test files)
- Use `subagent_type=Explore` for broad codebase searches
- Never use agents for sequential edits to the same file

### 1.4 Environment Setup

```bash
# Python (required: 3.11)
/opt/homebrew/Cellar/python@3.11/3.11.11/bin/python3.11

# Install in dev mode
pip install -e ".[dev]"

# With Athena++ support
pip install -e ".[dev,athena]"

# Verify installation
dpf backends
```

**Backend builds:**

```bash
# Athena++ (pybind11)
cd external/athena
python configure.py --prob=dpf_zpinch --coord=cylindrical -b --flux=hlld \
    --cxx=clang++-apple -omp -hdf5
make clean && make -j8

# AthenaK (Kokkos)
bash scripts/setup_athenak.sh
bash scripts/build_athenak.sh          # auto-detect
bash scripts/build_athenak.sh openmp   # force OpenMP

# After C++ changes
make -j8                               # in external/athena/
bash scripts/build_athenak.sh          # for AthenaK
pip install -e ".[dev,athena]"         # after pybind11 changes
```

---

## 2. Testing Strategy

### 2.1 Test Categories

| Category | Purpose | Convention | Example |
|----------|---------|-----------|---------|
| Phase tests | Feature development | `test_phase_{letter}_{topic}.py` | `test_phase_o_physics_accuracy.py` |
| Module tests | Unit coverage | `test_{module}.py` | `test_metal_production.py` |
| Integration | Cross-module | Specific naming | `test_dual_engine.py` |
| Slow tests | Long-running physics | `@pytest.mark.slow` | Convergence studies |

### 2.2 Running Tests

```bash
# Kill stale processes first (always do this)
pkill -f "pytest|python.*dpf"

# Full non-slow suite (primary CI gate)
pytest tests/ -x -q

# Specific phase
pytest tests/test_phase_o_physics_accuracy.py -v

# Specific test by name pattern
pytest tests/ -k "test_sod_shock" -v

# Slow tests only (30+ minutes)
pytest tests/ -m slow -v --tb=short

# Count all tests (validate CI gate)
pytest tests/ --collect-only -q

# With coverage
pytest tests/ -x -q --cov=src/dpf --cov-report=term-missing
```

### 2.3 CI Gate Requirements

- **Minimum tests:** >= 745 (currently ~2417 non-slow)
- **Zero failures** required on non-slow suite
- Always add tests with new code; never delete without replacing

### 2.4 Writing New Tests

**File naming:**
```
tests/test_phase_{letter}_{topic}.py   # Phase work
tests/test_{module}.py                 # Module tests
```

**Standard patterns:**

```python
import numpy as np
import pytest

from dpf.config import SimulationConfig


class TestMyFeature:
    """Test suite for new feature."""

    def test_instantiation(self, small_config):
        """Basic creation test."""
        # Use conftest.py fixtures: small_config, grid_shape, dx, default_circuit_params
        engine = SimulationEngine(small_config)
        assert engine.backend == "python"

    def test_physics_result(self):
        """Test with explicit tolerance."""
        result = compute_something(1.0, 2.0)
        assert result == pytest.approx(3.14159, rel=1e-5)

    @pytest.mark.slow
    def test_convergence(self):
        """Long-running convergence test (>1s)."""
        # Mark with @pytest.mark.slow
        errors = []
        for n in [32, 64, 128, 256]:
            errors.append(run_convergence_test(n))
        order = np.log2(errors[-2] / errors[-1])
        assert order > 1.8
```

**Mock patterns for AI/ML tests:**

```python
# Torch/WALRUS: use importorskip + monkeypatch
torch = pytest.importorskip("torch")  # noqa: E402, I001

def test_surrogate(monkeypatch):
    # Monkeypatch lazy imports at the ORIGINAL module path
    monkeypatch.setattr("dpf.engine.SimulationEngine", MockEngine)

    # Bypass heavy __init__ with object.__new__
    obj = object.__new__(DPFSurrogate)
    obj.model = mock_model

# FastAPI endpoints: use TestClient
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post("/api/path?n_steps=5", json=body_data)
```

**Key fixtures from `conftest.py`:**

| Fixture | Value | Use |
|---------|-------|-----|
| `grid_shape` | `(8, 8, 8)` | Small grid for fast tests |
| `dx` | `1e-2` | Grid spacing |
| `default_circuit_params` | Standard DPF params | Circuit config dict |
| `sample_config_dict` | Minimal valid config | SimulationConfig input |
| `small_config` | `SimulationConfig` instance | Ready-to-use config |

### 2.5 Test Performance Tips

- Use 8x8x8 grids for unit tests (larger grids are slow)
- Use `:memory:` for HDF5 diagnostics in test configs
- Use `tmp_path` fixture for file-writing tests
- Mark anything > 1 second with `@pytest.mark.slow`
- Multiple parallel pytest processes can exhaust 36GB memory — be careful

---

## 3. Git Workflow

### 3.1 Branch Strategy

DPF-Unified uses **main-based development**. All work happens on `main` with frequent commits.

### 3.2 Commit Naming Convention

```
Phase X.Y: description

# Examples:
Phase F.4: CLI and server backend integration
Phase O.1: HLLD Riemann solver with NaN-safe discriminant
Phase AA.0: Fix PF-1000 electrode dimensions (D3 bug)
Strike Team: Fix C2/C3/M9 bugs, resolve test debt
```

Format: `Phase {LETTER}.{NUMBER}: {imperative description}`

For non-phase work (bug fixes, cleanup): use descriptive imperative sentences.

### 3.3 When to Commit

- **End of each phase step** (e.g., after Phase O.1 is complete and tests pass)
- **After fixing bugs** (especially critical ones tracked in `memory/bugs.md`)
- **After strike-team sessions** (multi-bug fix batches)
- **Never commit** with failing tests

### 3.4 Pre-Commit Checklist

```bash
# 1. Kill stale processes
pkill -f "pytest|python.*dpf"

# 2. Lint check
ruff check src/ tests/

# 3. Run full non-slow suite
pytest tests/ -x -q

# 4. Verify test count meets CI gate
pytest tests/ --collect-only -q | tail -1

# 5. Stage and commit
git add <specific files>
git commit -m "Phase X.Y: description"
```

### 3.5 What NOT to Commit

- `.env` files or credentials
- Large binary files (checkpoints, training data)
- `__pycache__/` directories
- IDE-specific files (`.vscode/`, `.idea/`)
- Temporary output files (`diagnostics.h5`, `*.vtk`)

---

## 4. Agent Commands Reference

### 4.1 Slash Commands

DPF-Unified includes 23 slash commands in `.claude/commands/`. Invoke them with `/command-name`.

#### Simulation & Physics

| Command | Description |
|---------|-------------|
| `/physics` | DPF physics expert — MHD equations, V&V analysis |
| `/run-tests` | Run pytest suites, validate CI gate, diagnose failures |
| `/review-dpf` | Code review for correctness, style, physics accuracy |
| `/sweep` | Run parameter sweep to generate WALRUS training data |
| `/inverse-design` | Find DPF configs matching target outputs |

#### Build & Backend

| Command | Description |
|---------|-------------|
| `/build-athena` | Compile/configure Athena++ C++ backend |
| `/build-athenak` | Build AthenaK Kokkos backend |
| `/verify-athenak` | Run stock AthenaK problems, parse VTK output |

#### Metal GPU

| Command | Description |
|---------|-------------|
| `/metal-info` | Report Apple Silicon GPU capabilities |
| `/metal-benchmark` | Profile DPF performance on Metal GPU |
| `/metal-optimize` | Implement/tune Metal GPU kernels |
| `/metal-profile` | Profile memory usage and compute utilization |

#### WALRUS / AI

| Command | Description |
|---------|-------------|
| `/walrus-predict` | Run WALRUS surrogate inference |
| `/walrus-metal` | Run WALRUS inference on Apple Metal |
| `/walrus-train` | Fine-tune WALRUS on custom datasets |
| `/well-export` | Convert DPF simulation data to Well HDF5 format |
| `/validate-dataset` | Validate a Well-format training dataset |
| `/benchmark-ai` | Benchmark AI surrogate vs full physics |

#### Memory System

| Command | Description |
|---------|-------------|
| `/session-save` | Save current session state for later resumption |
| `/session-resume` | Resume from last saved session checkpoint |
| `/remember` | Save a fact or finding to the appropriate topic file |
| `/recall` | Search project memory for relevant context |
| `/memory-status` | Report on memory system health |

### 4.2 Memory System

The memory system persists knowledge across Claude Code sessions. Files live in `~/.claude/projects/-Users-anthonyzamora-dpf-unified/memory/`.

**Core files:**

| File | Purpose | Auto-loaded? |
|------|---------|-------------|
| `MEMORY.md` | Ground truth: project state, active bugs, quick reference | Yes (always) |
| `session.md` | Session checkpoint for resumption | On session start |
| `bugs.md` | Bug tracker: active, fixed, retracted | On demand |
| `patterns.md` | Coding patterns, gotchas, test techniques | On demand |
| `phases.md` | Phase implementation history | On demand |
| `metal.md` | Metal GPU lessons and capabilities | On demand |
| `walrus.md` | WALRUS integration status | On demand |
| `debates.md` | PhD debate scores and verdicts | On demand |

**Workflow:**

```
# Save a discovery
/remember "HLLD requires NaN guards in float32 discriminant"

# Search for past context
/recall "HLLD NaN"

# Save session before leaving
/session-save

# Resume next time
/session-resume
```

**Anti-hallucination rules** (enforced by the memory protocol):
1. Never claim a bug is fixed without checking `memory/bugs.md`
2. Never quote a PhD debate score without reading `memory/debates.md`
3. Never state WALRUS module status without reading `memory/walrus.md`
4. Never claim Metal solver capabilities without reading `memory/metal.md`

### 4.3 CLI Commands

The DPF CLI is accessed via the `dpf` command:

```bash
# Run a simulation
dpf simulate config.json --steps=100 --backend=metal

# Verify a config file
dpf verify config.json

# List available backends
dpf backends

# Show Metal GPU capabilities
dpf metal-info

# Start the simulation server (FastAPI + WebSocket)
dpf serve --host 127.0.0.1 --port 8765

# Start AI inference server
dpf serve-ai --checkpoint models/walrus.pt --device mps

# Export simulation to Well format (for WALRUS training)
dpf export-well config.json -o output.h5 --field-interval 10

# Run parameter sweep
dpf sweep sweep_config.json -o sweep_output/ -w 4

# Validate training dataset
dpf validate-dataset ./training_data/

# Run WALRUS prediction
dpf predict config.json --checkpoint models/walrus.pt --steps 100

# Run inverse design
dpf inverse targets.json --checkpoint models/walrus.pt --n-trials 100
```

---

## 5. Troubleshooting Common Issues

### 5.1 Stale Processes

**Symptom:** CPU pegged at 100%, tests hang, system becomes unresponsive.

**Cause:** Python/Numba processes from interrupted test runs or simulations persist.

**Fix:**
```bash
pkill -f "pytest|python.*dpf"
```

**Prevention:** Always kill stale processes before starting work:
```bash
pkill -f "pytest|python.*dpf"   # Do this first, every time
```

### 5.2 Numba JIT Latency

**Symptom:** First call to a physics function takes 1-5 seconds.

**Cause:** Numba compiles `@njit` functions to machine code on first invocation.

**Fix:** Numba functions use `cache=True` — second invocation is fast. The Athena++ backend eliminates this entirely.

### 5.3 Memory Issues on M3 Pro (36GB)

**Symptom:** Tests fail with `MemoryError` or system becomes sluggish.

**Cause:** Multiple parallel pytest processes or large grid simulations exhaust unified memory.

**Fixes:**
- Use small grids (8x8x8) for unit tests
- Don't run parallel pytest workers (`-n auto` is risky)
- Kill stale processes before testing
- For WALRUS fine-tuning: use `gradient_checkpointing_freq=1` and `batch_size=1`

### 5.4 Athena++ Global State / Segfaults

**Symptom:** Segfault when creating multiple Athena++ solver instances.

**Cause:** Athena++ uses global state (signal handlers, MeshBlock arrays) that cannot be re-initialized.

**Fixes:**
- Use `module`-scoped test fixtures for Athena++ tests (not `function`-scoped)
- Use singleton pattern for linked-mode engines
- Never access `ruser_mesh_data` in stock problem generators (they don't allocate it)
- Different problem generators require separate compiled binaries

### 5.5 Metal float32 Limitations

**Symptom:** Loss of precision, energy conservation drift, or unexpected NaN in Metal solver.

**Cause:** Apple Metal GPU has **no float64 support**. All physics runs in float32.

**Fixes:**
- Use `MetalMHDSolver(precision="float64")` to force CPU + float64 for maximum accuracy
- Energy conservation holds to ~1e-7 relative error in float32 (acceptable for most MHD)
- HLLD uses numerically stable discriminant form to avoid catastrophic cancellation
- For production V&V runs, always use float64 mode

### 5.6 AthenaK Build Failures

**Symptom:** CMake errors about missing headers or Kokkos configuration.

**Cause:** OpenMP + macOS sysroot incompatibility with Homebrew LLVM.

**Fix:**
```bash
# The build script handles this, but manually:
cmake -S . -B build_omp \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
  -D CMAKE_CXX_FLAGS="--sysroot=$(xcrun --show-sdk-path)" \
  -D PROBLEM=blast
```

**Note:** Kokkos does NOT support Apple Metal. On M3 Pro, use Serial or OpenMP only.

### 5.7 Import Errors After C++ Rebuild

**Symptom:** `ImportError: No module named '_athena_core'` after rebuilding.

**Fix:**
```bash
pip install -e ".[dev,athena]"
```

### 5.8 WALRUS Dependency Conflicts

**Symptom:** Version conflicts when installing WALRUS alongside DPF.

**Cause:** WALRUS pins `torch==2.5.1` and `numpy==1.26.4`, which may conflict.

**Fix:** Use a separate virtual environment for WALRUS training:
```bash
python -m venv walrus-env
source walrus-env/bin/activate
pip install git+https://github.com/PolymathicAI/walrus.git
```

DPF inference can use the main venv if torch version is compatible.

### 5.9 HDF5 File Conflicts in Tests

**Symptom:** Tests fail with HDF5 locking or file-in-use errors.

**Fix:**
- Use `":memory:"` for diagnostics in test configs
- Use `tmp_path` pytest fixture for file-writing tests
- Never use hardcoded paths for test output

### 5.10 Ruff Linting Errors

**Symptom:** Ruff complains about physics variable names like `Te`, `Ti`, `B`.

**Cause:** Physics convention names violate PEP 8 naming rules.

**Fix:** The project ignores N802/N803/N806/N815/N816 for physics variables. If you see these warnings, they are already suppressed in `pyproject.toml`.

**Common ruff issues:**
```bash
# Check for issues
ruff check src/ tests/

# Auto-fix safe issues
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/
```

---

## Quick Reference Card

```
# Daily workflow
pkill -f "pytest|python.*dpf"        # Kill stale processes
ruff check src/ tests/                # Lint
pytest tests/ -x -q                   # Test (non-slow)
pytest tests/ --collect-only -q       # Count tests (CI gate >= 745)

# Run specific phase tests
pytest tests/test_phase_o_*.py -v     # Phase O tests
pytest tests/ -k "metal" -v           # All Metal tests

# Backend management
dpf backends                          # List available backends
dpf simulate config.json --backend=metal  # Use Metal GPU

# Memory system
/session-resume                       # Start of session
/remember "important finding"         # Save a fact
/recall "keyword"                     # Search memory
/session-save                         # End of session

# Key file locations
src/dpf/engine.py                     # Main orchestration
src/dpf/config.py                     # Pydantic config
src/dpf/cli/main.py                   # CLI entry point
src/dpf/fluid/mhd_solver.py           # Python MHD solver
src/dpf/metal/metal_solver.py         # Metal GPU solver
src/dpf/ai/surrogate.py               # WALRUS surrogate
tests/conftest.py                     # Shared test fixtures
```
