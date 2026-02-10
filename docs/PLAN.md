# DPF Unified — Forward Development Plan (v3)

## Context & What's Done

We are building a modern dense plasma focus (DPF) simulator with:
- **Backend**: Tri-engine MHD solver — Python (NumPy/Numba) fallback + Athena++ C++ (pybind11) + AthenaK Kokkos (subprocess)
- **Frontend**: Unity UI with Teaching Mode + Engineering Mode (future)
- **Platform**: Apple Silicon local execution (MVP), HPC cluster (future)
- **AI**: Polymathic AI WALRUS integration for surrogate models and inverse design

### Completed Phases
- ✅ **Phase A**: README rewrite — Honest assessment with fidelity grade
- ✅ **Phase B**: Wire dormant physics — ADI/RKL2, line radiation, CT default ON
- ✅ **Phase C**: Verification & Validation — Diffusion convergence, Orszag-Tang, Sedov, Lee Model, shock tubes
- ✅ **Phase D**: Physics improvements — Full Braginskii, Powell div-B, anisotropic conduction, Dedner GLM
- ✅ **Phase E**: Apple Silicon optimization — Numba prange, benchmarks
- ✅ **Phase F**: Athena++ integration — Submodule, pybind11, dual-engine, verification, CLI/server
  - F.0: Athena++ submodule setup and M3 Pro build (magnoh, resist, shock_tube)
  - F.1: pybind11 wrapper layer (linked-mode via `_athena_core.so`)
  - F.2: Dual-engine architecture (backend="python"/"athena"/"auto" in config)
  - F.3: Athena++ verification suite (Sod, Brio-Wu, magnoh, cross-backend)
  - F.4: CLI `--backend` flag, server health backend reporting, `dpf backends` command
  - F.5: Documentation (CLAUDE.md, README, ATHENA_BUILD.md, CI, PLAN.md)

- ✅ **Phase G**: Athena++ DPF physics — Circuit coupling C++, Spitzer resistivity, two-temperature, bremsstrahlung, Braginskii transport
- ✅ **Phase H**: WALRUS training data pipeline — Field mapping, Well HDF5 exporter, dataset validator, batch trajectory runner
- ✅ **Phase I**: AI features — Surrogate inference, inverse design, hybrid engine, instability detector, confidence/ensemble, real-time server, CLI + config extensions

- ✅ **Phase J.1**: AthenaK integration — Kokkos subprocess wrapper (4 modules), VTK I/O, build scripts, CLI/server backend, 57 tests
  - J.1.a: Research spike (build Serial + OpenMP, benchmark, VTK format analysis, coordinate/physics survey)
  - J.1.b: Build infrastructure (setup_athenak.sh, build_athenak.sh, athinput.athenak_blast)
  - J.1.c: Subprocess wrapper (athenak_config, athenak_io, athenak_solver, __init__)
  - J.1.d: Integration (config.py athenak backend, engine.py dispatch, CLI --backend=athenak, server /api/health)

- ✅ **Phase M**: Metal GPU optimization — Production MetalMHDSolver (SSP-RK2 + HLL + PLM + CT), PyTorch MPS stencil kernels, MLX surrogate inference, device management, 35 production tests, 8 benchmarks

- ✅ **Phase N**: Hardening & cross-backend verification — Metal cross-backend parity (Sod shock), AthenaK cross-backend blast parity, Metal long-run energy conservation, coverage gate, doc updates

- ✅ **Phase O**: Physics accuracy — WENO5-Z reconstruction (Borges et al. 2008, point-value FD formulas), HLLD Riemann solver (Miyoshi & Kusano 2005, 8-component), SSP-RK3 time integration (Shu-Osher 1988), float64 precision mode, energy floor guards, 45 accuracy tests
  - O.1: HLLD solver (NaN-safe, Lax-Friedrichs fallback, numerically stable discriminant)
  - O.2: WENO5 with correct FD polynomial coefficients (NOT Jiang-Shu FV formulas)
  - O.3: SSP-RK3 (3-stage, 3rd-order, verified lower error than RK2)
  - O.4: WENO-Z weights (global smoothness indicator tau5=|beta0-beta2|)
  - O.5: Float64 precision mode (CPU float64 for production V&V)
  - O.6: Convergence order verification (5.47-5.79 interior, ~1.86 overall solver)

**1452 total tests** (1331 non-slow, 121 slow), 0 failures.
**Current fidelity: 8.7/10** — WENO5-Z + HLLD + SSP-RK3 + float64 = production-grade.

---

## What Is WALRUS?

[Polymathic AI WALRUS](https://huggingface.co/polymathic-ai/walrus) is a **1.3B-parameter space-time Transformer** foundation model for continuum dynamical systems. MIT licensed. Built by the Simons Foundation / Flatiron Institute.

### Key Specs
| Property | Value |
|----------|-------|
| Parameters | 1.3 billion |
| Architecture | Encoder-processor-decoder Transformer with factorized space-time attention |
| Training data | 19 physical scenarios, 63 fields (acoustics, fluids, plasma, astrophysics, rheology) |
| Training compute | 96× NVIDIA H100 (HSDP, 4 GPU/shard) |
| License | MIT |
| Paper | arXiv:2511.15684 |
| Code | github.com/PolymathicAI/walrus |

### How It Works
```
Input:  U(t) = [u(t-τ+1), ..., u(t)]    ← short history of field snapshots
Output: Δu(t+1) = M(U(t))                ← predicted state change
Next:   u(t+1) = u(t) + Δu(t+1)          ← reconstructed next state
```

### Architectural Innovations
1. **Adaptive-compute patch embedding** — balances token count across resolutions, mixes 2D/3D
2. **Patch jittering** — harmonic-analysis-based augmentation that reduces aliasing and improves long-horizon stability
3. **Tensor-law-aware augmentation** — 2D data embedded into 3D via plane rotations, vector/tensor fields rotated correctly
4. **Asymmetric normalization** — input: RMS over space-time; output: RMS of Δu

### Pretraining Datasets Relevant to DPF
| Dataset | Domain | Dims | Resolution | Size |
|---------|--------|------|------------|------|
| **MHD_64** | Magnetohydrodynamics | 3D | 64³ | 72 GB |
| **MHD_256** | Magnetohydrodynamics | 3D | 256³ | 4.5 TB |
| **euler_multi_quadrants** | Compressible flow | 2D | 512² | 5.1 TB |
| **supernova_explosion_64/128** | Astrophysics (MHD+rad) | 3D | 64³/128³ | 1 TB |
| **rayleigh_taylor_instability** | Fluid dynamics | 3D | 128³ | 256 GB |

**WALRUS has already seen MHD data.** It understands magnetic field evolution, shock dynamics, and compressible plasma behavior. This makes fine-tuning for DPF much more tractable than training from scratch.

### The Well Data Format (for WALRUS)
HDF5 with structure:
```
Root attributes: simulation_parameters, dataset_name, grid_type, n_spatial_dims, n_trajectories
/dimensions/     → spatial coords (x, y, z) + time array
/boundary_conditions/ → periodic/wall/open with masks
/t0_fields/      → scalar fields: shape (n_traj, n_steps, nx, ny, nz)
/t1_fields/      → vector fields: shape (n_traj, n_steps, nx, ny, nz, D)
/t2_fields/      → tensor fields: shape (n_traj, n_steps, nx, ny, nz, D²)
```
Arrays: float32, uniform grids, constant time intervals.

---

## Where & Why WALRUS Fits Into DPF Unified

### The Problem WALRUS Solves

A single DPF simulation at moderate resolution (128×1×256 cylindrical) takes **minutes to hours** depending on physics enabled. Parameter sweeps over 100+ configurations become intractable on a single M3 Ultra. For the Unity Engineering Mode use case — where an engineer asks "what fill pressure and charging voltage give me 10^10 neutrons?" — we need:

1. **Fast surrogate predictions** (~100ms per configuration) for the parameter explorer
2. **Inverse design capability** — given a target output, find input configurations
3. **Uncertainty quantification** — how confident is the prediction?

### Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Unity Frontend                       │
│  ┌──────────────┐  ┌───────────────────────────────────┐│
│  │ Teaching Mode │  │        Engineering Mode            ││
│  │ (visualize)   │  │ ┌─────────────┐ ┌──────────────┐ ││
│  │               │  │ │ Quick Sweep  │ │ Inverse      │ ││
│  │               │  │ │ (WALRUS)     │ │ Design       │ ││
│  │               │  │ └──────┬──────┘ │ (WALRUS)     │ ││
│  │               │  │        │        └──────┬───────┘ ││
│  └──────┬───────┘  └────────┼───────────────┼─────────┘│
│         │                   │               │           │
└─────────┼───────────────────┼───────────────┼───────────┘
          │                   │               │
          ▼                   ▼               ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│  DPF Engine     │  │ WALRUS       │  │ WALRUS       │
│  (full physics) │  │ Surrogate    │  │ Inverse      │
│  Minutes-hours  │  │ ~100ms/pred  │  │ Optimizer    │
│                 │  │              │  │              │
│  • Verify       │  │ • Predict    │  │ • Target Y   │
│  • Generate     │  │ • Sweep      │  │ • Find X     │
│  • Validate     │  │ • Rank       │  │ • Verify     │
└────────┬────────┘  └──────┬───────┘  └──────┬───────┘
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────────────────────────────────────────────┐
│              Training Data Pipeline                     │
│  DPF Engine → Well-format HDF5 → Fine-tune WALRUS     │
│  (generate trajectories)          (on DPF-specific     │
│                                    MHD + circuit)      │
└────────────────────────────────────────────────────────┘
```

### Five Use Cases for WALRUS in DPF Unified

#### Use Case 1: Fast Parameter Sweep (Engineering Mode)
- **What**: Engineer sets ranges for V0, C, fill pressure, electrode geometry → WALRUS predicts I(t), Te(t), neutron yield for 1000+ configurations in seconds
- **Where**: `src/dpf/ai/surrogate.py` → called by Unity Engineering Mode
- **Why**: Full simulation of 1000 configs = days. WALRUS = minutes.

#### Use Case 2: Inverse Design
- **What**: Engineer specifies target (e.g., "10^10 DD neutrons, < 2 MA peak current") → optimizer searches WALRUS latent space → returns top-N configurations → DPF engine verifies best candidates
- **Where**: `src/dpf/ai/inverse_design.py`
- **Why**: This is the "what configurations do I need to reach X yields?" question. Impossible without a fast surrogate.

#### Use Case 3: Real-Time Teaching Mode Visualization
- **What**: Student adjusts sliders (voltage, pressure) → WALRUS predicts next 10 frames in ~200ms → Unity renders smoothly
- **Where**: `src/dpf/ai/realtime_predictor.py` → WebSocket stream to Unity
- **Why**: Full MHD is too slow for interactive visualization. WALRUS provides "physics-plausible" frames in real time.

#### Use Case 4: Simulation Acceleration (Hybrid)
- **What**: Run full DPF engine for first 20% of simulation (rundown phase) → hand off to WALRUS for remaining 80% (pinch + decay) → validate key frames with full engine
- **Where**: `src/dpf/ai/hybrid_engine.py`
- **Why**: Most compute time is in the stiff pinch phase. WALRUS can skip ahead, with periodic validation checkpoints.

#### Use Case 5: Anomaly Detection / Instability Prediction
- **What**: Feed live simulation state to WALRUS → compare WALRUS prediction to actual next step → large divergence = instability onset (m=0 sausage, m=1 kink)
- **Where**: `src/dpf/ai/instability_detector.py`
- **Why**: Foundation model has seen thousands of MHD trajectories. Deviation from its prediction means something unusual (instability, numerical artifact) is happening.

---

## Forward Plan: Steps 3-7

### Step 3: Verification & Validation (Fidelity 4→5)

This is the next immediate work. Without V&V, we can't trust the physics enough to generate meaningful training data for WALRUS.

#### 3a. Convergence Studies
- **Sod shock tube**: Run at N = 64, 128, 256, 512, 1024 → compute L1 error vs exact solution → verify 2nd-order convergence (SSP-RK2)
- **Brio-Wu MHD**: Self-convergence at N = 128, 256, 512 using Richardson extrapolation
- **Resistive diffusion**: NEW — uniform B-field with constant eta, compare to exact Gaussian solution for all three methods (explicit, ADI, RKL2)
- **Files**: Extend `src/dpf/verification/shock_tubes.py`, new `src/dpf/verification/diffusion_convergence.py`

#### 3b. Standard MHD Benchmarks
- **Orszag-Tang vortex** (2D, Cartesian): The canonical MHD benchmark. Compare density contours at t=0.5 to Athena++ reference images.
- **Cylindrical Sedov blast**: Tests geometric source terms in cylindrical coords. Exact similarity solution exists.
- **Files**: New `src/dpf/verification/orszag_tang.py`, `src/dpf/verification/sedov_cylindrical.py`

#### 3c. Experimental Validation Upgrade
- **PF-1000 waveform comparison**: Digitize I(t) from Scholz et al. (2006). Compute RMSE over full waveform, not just peak current matching.
- **Lee Model cross-check**: Implement simplified Lee Model in Python. Compare I(t), dI/dt, pinch time, neutron yield vs our MHD engine for same device parameters.
- **Files**: Modify `src/dpf/validation/suite.py`, new `src/dpf/validation/lee_model_comparison.py`

#### 3d. CT & Diffusion Verification
- Run cylindrical MHD for 1000+ steps → verify div(B) < 1e-12 with CT enabled
- Run resistive diffusion (ADI + RKL2) → compare to analytical Gaussian at t=0.1, 0.5, 1.0
- Compare cooling curves for Cu impurity at 1-100 eV against published atomic data

**Deliverables**: Convergence plots, benchmark comparison images, validation RMSE numbers
**Effort**: 4-6 days
**Target fidelity after**: 5/10

---

### Step 4: Physics Fidelity Improvements (Fidelity 5→6)

#### 4a. Complete Braginskii Transport
- **Perpendicular viscosity (eta_1, eta_2)**: Add Braginskii polynomial fits. Currently only parallel eta_0.
- **Gyroviscosity (eta_3)**: Already computed in `viscosity.py` but never applied to stress tensor. Wire it in.
- **Anisotropic thermal conduction**: Replace isotropic kappa with field-aligned Braginskii kappa_parallel + kappa_perp using Sharma-Hammett slope limiter method.
- **Files**: `src/dpf/fluid/viscosity.py`, new `src/dpf/fluid/anisotropic_conduction.py`

#### 4b. Improved div(B) Control
- **Powell 8-wave source terms**: Add non-conservative source terms as complement to Dedner (following MPI-AMRVAC approach)
- **Dedner GLM tuning**: Implement Mignone & Tzeferacos (2010) optimal ch/cp prescription
- **Files**: `src/dpf/fluid/mhd_solver.py`, `src/dpf/fluid/cylindrical_mhd.py`

#### 4c. Multi-Species Integration
- Move `experimental/species.py` back to `src/dpf/species.py` and wire `SpeciesMixture` into engine
- Each species gets its own density field; Z_eff becomes spatially varying
- Electrode ablation as mass/energy source at boundaries (wire `atomic/ablation.py`)
- **Files**: `src/dpf/engine.py`, `src/dpf/species.py`, `src/dpf/atomic/ablation.py`

**Deliverables**: Full Braginskii tests, Powell+Dedner benchmark, multi-species run
**Effort**: 5-7 days
**Target fidelity after**: 6/10

---

### Step 5: WALRUS Training Data Pipeline

This is the bridge between our physics engine and AI. We need to generate hundreds of DPF simulation trajectories in The Well format.

#### 5a. Well-Format Exporter (`src/dpf/ai/well_exporter.py`)

Convert DPF simulation output to The Well HDF5 schema:

```python
# Mapping: DPF state → Well fields
FIELD_MAP = {
    # Scalar fields (t0_fields)
    "rho":      "density",         # (nx, ny, nz) → (n_traj, n_steps, nr, 1, nz)
    "Te":       "electron_temp",   # (nx, ny, nz)
    "Ti":       "ion_temp",        # (nx, ny, nz)
    "pressure": "pressure",        # (nx, ny, nz)
    "psi":      "dedner_psi",      # (nx, ny, nz)

    # Vector fields (t1_fields)
    "B":        "magnetic_field",  # (3, nx, ny, nz) → (n_traj, n_steps, nr, 1, nz, 3)
    "velocity": "velocity",        # (3, nx, ny, nz) → (n_traj, n_steps, nr, 1, nz, 3)

    # Scalars (non-spatial, time-varying)
    "current":  "circuit_current", # scalar per timestep
    "voltage":  "circuit_voltage", # scalar per timestep
}
```

**Implementation details**:
- Read from DPF HDF5 field snapshots or checkpoint files
- Reshape from DPF layout `(3, nx, ny, nz)` to Well layout `(n_traj, n_steps, nx, ny, nz, 3)`
- Cast to float32 (Well standard)
- Attach metadata: grid spacing (dx, dz), boundary conditions (wall at r=0, open at r_max, etc.)
- Include simulation parameters (V0, C, L0, fill_pressure, electrode_geometry) as root attributes
- Handle cylindrical geometry: store as 2D with `n_spatial_dims=2`, dims = [r, z]

#### 5b. Batch Trajectory Generator (`src/dpf/ai/batch_runner.py`)

Automated parameter sweep runner that:
1. Generates Latin Hypercube Sample (LHS) over parameter space
2. Runs N simulations in parallel (multiprocessing on M3 Ultra, 24 cores)
3. Exports each trajectory to Well format
4. Produces a combined training dataset

**Parameter space for DPF training data**:
| Parameter | Range | Units | Justification |
|-----------|-------|-------|---------------|
| V0 (voltage) | 10-40 kV | V | Typical DPF range |
| C (capacitance) | 5-100 μF | F | Small to large banks |
| L0 (inductance) | 10-200 nH | H | Connection + collector plate |
| fill_pressure | 1-20 Torr | Pa | Deuterium fill |
| anode_radius | 5-25 mm | m | Mather-type geometry |
| cathode_radius | 25-60 mm | m | Must be > anode |
| rho0 (density) | 1e-5 to 1e-2 | kg/m³ | Derived from fill pressure |

**Target**: 500-1000 trajectories at 64×1×128 (fast, ~2min each) + 50-100 at 128×1×256 (high-res, ~30min each)

#### 5c. Dataset Validation (`src/dpf/ai/dataset_validator.py`)

Before training, verify dataset quality:
- Check for NaN/Inf in any field
- Verify energy conservation (< 5% drift per trajectory)
- Flag trajectories with numerical instabilities
- Compute field statistics (mean, std, min, max per field per timestep)
- Verify Well format compliance (field shapes, metadata, boundary conditions)

**Deliverables**: Well-format exporter, batch runner, validated training dataset
**Effort**: 4-5 days
**Target**: 500+ valid trajectories ready for WALRUS fine-tuning

---

### Step 6: WALRUS Fine-Tuning & Inference Integration

#### 6a. Fine-Tune WALRUS on DPF Data

```bash
# Install WALRUS
git clone git@github.com:PolymathicAI/walrus.git
cd walrus && pip install .

# Fine-tune from pretrained checkpoint
# Uses Hydra config — create dpf-specific config
python -m walrus.train \
    model=walrus_1.3b \
    model.checkpoint=polymathic-ai/walrus \
    data.dataset_path=/path/to/dpf_well_dataset/ \
    training.epochs=50 \
    training.lr=1e-5 \
    training.batch_size=4
```

**Fine-tuning strategy**:
- Start from the pretrained 1.3B checkpoint (already understands MHD)
- Low learning rate (1e-5 to 5e-5) to preserve pretrained knowledge
- Train on DPF-specific fields: rho, B, Te, Ti, velocity, pressure + circuit scalars
- History length τ = 4-8 timesteps (captures wave propagation + circuit dynamics)
- Validate on held-out 20% of trajectories
- Target: < 5% normalized L1 error on next-step prediction

**Hardware requirements for fine-tuning**:
- M3 Ultra (192GB unified memory) can fit 1.3B model in float16 → ~2.6GB weights
- Fine-tuning with gradient checkpointing: feasible on M3 Ultra with small batch (1-2)
- Faster option: Use cloud GPU (A100/H100) for training, deploy locally for inference
- Inference: ~100ms per prediction on M3 Ultra GPU (Metal via PyTorch MPS backend)

#### 6b. DPF Surrogate Server (`src/dpf/ai/surrogate.py`)

Python class that wraps fine-tuned WALRUS for DPF predictions:

```python
class DPFSurrogate:
    """Fast surrogate model using fine-tuned WALRUS."""

    def __init__(self, checkpoint_path: str, device: str = "mps"):
        """Load fine-tuned WALRUS model."""
        self.model = load_walrus_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

    def predict_next_step(self, history: list[dict]) -> dict:
        """Given τ previous states, predict next state."""
        # Convert DPF state dicts to WALRUS input tensor
        # Returns predicted field dict

    def rollout(self, initial_state: dict, n_steps: int) -> list[dict]:
        """Autoregressive rollout for n_steps."""
        # Iteratively predict, feeding predictions back as input

    def parameter_sweep(self, param_grid: dict) -> pd.DataFrame:
        """Predict scalar outputs (I_peak, Te_max, Y_neutron) for grid of configs."""
        # Batch inference over parameter combinations

    def predict_waveform(self, config: SimulationConfig) -> dict:
        """Predict full I(t), Te(t) waveforms for a given configuration."""
        # Initialize from config → rollout → extract time series
```

#### 6c. Inference Optimization for Apple Silicon

- **PyTorch MPS backend**: Native Metal GPU acceleration on M3 Ultra
- **Model quantization**: INT8 quantization via `torch.quantization` → 4x memory reduction, ~2x inference speedup
- **CoreML export** (optional): Convert to CoreML for maximum Apple Silicon efficiency
- **Batch prediction**: Process multiple configurations simultaneously using batch dimension

**Deliverables**: Fine-tuned WALRUS checkpoint, DPFSurrogate class, Apple Silicon inference benchmark
**Effort**: 5-7 days (including training time)
**Target**: < 200ms per full trajectory prediction on M3 Ultra

---

### Step 7: AI-Powered Features

#### 7a. Inverse Design Engine (`src/dpf/ai/inverse_design.py`)

Given target outputs, find input configurations:

```python
class InverseDesigner:
    """Find DPF configurations that achieve target performance."""

    def __init__(self, surrogate: DPFSurrogate):
        self.surrogate = surrogate

    def find_config(
        self,
        targets: dict,           # e.g., {"neutron_yield": 1e10, "I_peak_max": 2e6}
        constraints: dict,       # e.g., {"V0_max": 30e3, "C_max": 50e-6}
        method: str = "bayesian" # "bayesian", "evolutionary", "gradient"
    ) -> list[SimulationConfig]:
        """Return top-N configurations ranked by proximity to targets."""
```

**Methods**:
1. **Bayesian optimization** (primary): Use `optuna` or `botorch` with WALRUS as objective function. Efficient for expensive-to-evaluate functions.
2. **Evolutionary** (fallback): CMA-ES or differential evolution over parameter space.
3. **Gradient-based** (experimental): Backpropagate through WALRUS to find input sensitivities.

**Workflow**: Inverse design → top 5 configs → full DPF engine verification → present to user

#### 7b. Real-Time Prediction Server (`src/dpf/ai/realtime_server.py`)

WebSocket endpoint for Unity Teaching Mode:

```
POST /ai/predict     → single next-step prediction
POST /ai/rollout     → multi-step autoregressive rollout
POST /ai/sweep       → parameter sweep (batch)
POST /ai/inverse     → inverse design query
WS   /ai/stream      → real-time frame streaming for Unity
```

#### 7c. Confidence & Uncertainty Estimation

WALRUS predictions need uncertainty bounds:
- **Ensemble approach**: Fine-tune 3-5 WALRUS models with different random seeds → prediction variance = uncertainty
- **Residual monitoring**: Track |WALRUS prediction - DPF engine| on validation set → build error model
- **Out-of-distribution detection**: Flag predictions where input state is far from training distribution (Mahalanobis distance in latent space)

**Deliverables**: Inverse design engine, real-time server, uncertainty estimation
**Effort**: 4-6 days
**Target**: Full Unity integration API ready

---

## Apple Silicon Optimization (Parallel Track)

### Quick Wins (Step 5 from v1 plan)
- **Numba `parallel=True` + `prange`**: Triple-nested loops in CT, viscosity, Nernst are trivially parallelizable → expect 8-16x on M3 Ultra (24 performance cores)
- **NumPy on Accelerate**: Verify `numpy.__config__.show()` links to Apple Accelerate BLAS/LAPACK for ADI tridiagonal solves
- **Profile first**: `py-spy record -o profile.svg -- python -m dpf simulate config.json` → identify actual hotspots before optimizing

### MLX for GPU Kernels (Future)
- Apple Metal GPU via `mlx` is the correct Apple Silicon GPU path (NOT CuPy/CUDA)
- Port WENO5+HLL flux sweep to MLX → benchmark against Numba CPU
- MLX is also the best path for WALRUS inference on Apple Silicon (if PyTorch MPS is insufficient)

---

## Updated Roadmap

| Phase | Goal | Fidelity | Key Work | Status |
|-------|------|----------|----------|--------|
| ~~A~~ | ~~Honest documentation~~ | — | ~~README rewrite, dormant code triage~~ | ✅ Done |
| ~~B~~ | ~~Wire dormant physics~~ | 4/10 | ~~ADI/RKL2 diffusion, line radiation, CT default on~~ | ✅ Done |
| ~~C~~ | ~~Verification & validation~~ | 5/10 | ~~Diffusion convergence, Orszag-Tang, Sedov, Lee Model~~ | ✅ Done |
| ~~D~~ | ~~Physics improvements~~ | 6/10 | ~~Full Braginskii, Powell div-B, anisotropic conduction~~ | ✅ Done |
| ~~E~~ | ~~Apple Silicon optimization~~ | 6/10 | ~~Numba prange, Accelerate BLAS, benchmarks~~ | ✅ Done |
| ~~F~~ | ~~Athena++ integration~~ | — | ~~Submodule, pybind11, dual-engine, verification, CLI/server~~ | ✅ Done |
| ~~G~~ | ~~Athena++ DPF physics~~ | 7/10 | ~~Circuit coupling C++, Spitzer η, two-temp, radiation, Braginskii~~ | ✅ Done |
| ~~H~~ | ~~WALRUS data pipeline~~ | — | ~~Field mapping, Well exporter, batch runner, dataset validator~~ | ✅ Done |
| ~~I~~ | ~~AI features~~ | — | ~~Surrogate, inverse design, hybrid engine, instability, confidence, server~~ | ✅ Done |
| ~~J.1~~ | ~~AthenaK integration~~ | — | ~~Kokkos subprocess wrapper, VTK I/O, build scripts, 57 tests~~ | ✅ Done |
| ~~M~~ | ~~Metal GPU optimization~~ | — | ~~MetalMHDSolver, MPS stencils, MLX surrogate, 35 tests, 8 benchmarks~~ | ✅ Done |
| ~~N~~ | ~~Hardening & cross-backend V&V~~ | 7/10 | ~~Metal parity test, AthenaK parity, energy conservation, coverage gate~~ | ✅ Done |
| ~~O~~ | ~~Physics accuracy~~ | 8.7/10 | ~~WENO5-Z, HLLD, SSP-RK3, float64, 45 accuracy tests~~ | ✅ Done |
| **J.2** (next) | WALRUS live integration | — | Real IsotropicModel inference in surrogate.py, fix Well exporter, fix API mismatches | 🔜 |
| **J.3** | Unity frontend | — | Teaching/Engineering mode (greenfield, same repo) | 🔜 |
| **J.4** | HPC scaling | 9/10 | MPI via AthenaK, cloud GPU (CUDA Kokkos) | 🔜 |
| **J.5** | Advanced AthenaK physics | — | Custom DPF z-pinch pgen, circuit coupling, resistive MHD | 🔜 |

**Phases A-O complete**: 1452 tests (1331 non-slow + 121 slow), tri-engine + Metal GPU (WENO5-Z + HLLD + SSP-RK3 + float64) + full AI/ML integration. Fidelity: 8.7/10.
**Next**: Phase J.2 implements real WALRUS model loading and inference.

---

## Critical Files

### Existing (to modify)
| File | Changes |
|------|---------|
| `src/dpf/engine.py` | Powell sources, multi-species, instability hooks |
| `src/dpf/config.py` | Powell/Braginskii/multi-species config fields |
| `src/dpf/fluid/mhd_solver.py` | Powell sources, Dedner tuning, Numba prange |
| `src/dpf/fluid/cylindrical_mhd.py` | Powell sources, axis BC improvements |
| `src/dpf/fluid/viscosity.py` | Complete Braginskii (eta_1, eta_2, eta_3) |
| `src/dpf/validation/suite.py` | Full waveform RMSE comparison |
| `README.md` | Update roadmap, add WALRUS section |

### New Files to Create (for Phases J.2-J.4)
| File | Purpose |
|------|---------|
| TBD | J.2: Unity frontend integration |
| TBD | J.3: MPI scaling, cloud GPU deployment |
| `external/athenak/src/pgen/dpf_zpinch_k.cpp` | J.4: Custom AthenaK DPF z-pinch problem generator |

### Files Created in Phases A-I
| File | Purpose | Phase |
|------|---------|-------|
| `src/dpf/verification/diffusion_convergence.py` | Resistive diffusion convergence test | C |
| `src/dpf/verification/orszag_tang.py` | Orszag-Tang vortex benchmark | C |
| `src/dpf/verification/sedov_cylindrical.py` | Cylindrical Sedov blast | C |
| `src/dpf/validation/lee_model_comparison.py` | Lee Model cross-check | C |
| `src/dpf/fluid/anisotropic_conduction.py` | Field-aligned thermal conduction | D |
| `src/dpf/ai/__init__.py` | AI module init (HAS_TORCH, HAS_WALRUS guards) | H |
| `src/dpf/ai/field_mapping.py` | DPF ↔ Well field name/shape mapping | H |
| `src/dpf/ai/well_exporter.py` | DPF → Well HDF5 format converter | H |
| `src/dpf/ai/batch_runner.py` | Parallel trajectory generation (LHS sampling) | H |
| `src/dpf/ai/dataset_validator.py` | Training data QA (NaN/Inf, schema, energy) | H |
| `src/dpf/ai/surrogate.py` | WALRUS inference wrapper (predict, rollout, sweep) | I |
| `src/dpf/ai/inverse_design.py` | Target → config optimizer (Bayesian/evolutionary) | I |
| `src/dpf/ai/hybrid_engine.py` | DPF engine + WALRUS handoff with validation | I |
| `src/dpf/ai/instability_detector.py` | Anomaly detection via WALRUS divergence | I |
| `src/dpf/ai/confidence.py` | Ensemble prediction with uncertainty quantification | I |
| `src/dpf/ai/realtime_server.py` | FastAPI router for AI endpoints + WebSocket | I |
| `src/dpf/athenak_wrapper/__init__.py` | AthenaK module init, binary detection, is_available() | J.1 |
| `src/dpf/athenak_wrapper/athenak_config.py` | SimulationConfig → AthenaK athinput translation | J.1 |
| `src/dpf/athenak_wrapper/athenak_io.py` | VTK binary reader + DPF state conversion | J.1 |
| `src/dpf/athenak_wrapper/athenak_solver.py` | AthenaKSolver subprocess wrapper | J.1 |
| `scripts/setup_athenak.sh` | AthenaK submodule + Kokkos setup script | J.1 |
| `scripts/build_athenak.sh` | Platform-detecting AthenaK build script | J.1 |
| `docs/ATHENAK_RESEARCH.md` | AthenaK research spike findings and recommendations | J.1 |

### Test Files (Phases H-I, J.1)
| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_phase_h_field_mapping.py` | ~15 | Field name mapping, shape transforms, geometry inference |
| `tests/test_phase_h_well_exporter.py` | ~30 | HDF5 Well format export, metadata, cylindrical geometry |
| `tests/test_phase_h_dataset_validator.py` | ~20 | NaN/Inf checks, schema validation, energy conservation |
| `tests/test_phase_h_batch_runner.py` | ~27 | LHS sampling, config building, parallel execution |
| `tests/test_phase_i_surrogate.py` | ~25 | Load/predict/rollout/sweep, mock torch |
| `tests/test_phase_i_inverse_design.py` | ~20 | Bayesian/evolutionary search, constraints |
| `tests/test_phase_i_hybrid_engine.py` | ~15 | Physics→surrogate handoff, validation, fallback |
| `tests/test_phase_i_instability.py` | ~15 | Divergence detection, severity classification |
| `tests/test_phase_i_confidence.py` | ~15 | Ensemble mean/std, OOD detection, confidence scoring |
| `tests/test_phase_i_ai_server.py` | ~25 | REST endpoints, WebSocket, surrogate loading |
| `tests/test_phase_j_athenak.py` | 50 | AthenaK config, VTK I/O, solver, backend resolution |
| `tests/test_phase_j_cli_server.py` | 7 | CLI --backend=athenak, server health AthenaK |

---

## Verification Checkpoints

| Check | Pass Criteria |
|-------|---------------|
| Sod shock tube convergence | L1 error ∝ dx², order > 1.8 |
| Brio-Wu self-convergence | Richardson order > 1.5 |
| Resistive diffusion (all methods) | L2 error vs exact < 1e-4 at t=1.0 |
| Orszag-Tang density @ t=0.5 | Visual match to Athena++ reference |
| Cylindrical Sedov blast | Peak density within 5% of similarity solution |
| CT div(B) | < 1e-12 after 1000+ steps |
| PF-1000 waveform | RMSE(I(t)) < 20% over full waveform |
| Lee Model cross-check | Peak current within 10%, pinch time within 20% |
| WALRUS next-step error | Normalized L1 < 5% on held-out DPF trajectories |
| WALRUS rollout stability | No blowup for 100+ autoregressive steps |
| Inverse design accuracy | Top-5 configs within 20% of target when verified by full engine |
| Apple Silicon inference | < 200ms per prediction on M3 Ultra |

---

## Dependencies & Installation for AI Integration

```toml
# pyproject.toml [project.optional-dependencies]
ai = [
    "torch>=2.1",           # PyTorch with MPS backend for Apple Silicon
    "optuna",               # Bayesian optimization for inverse design
    "pyDOE2",               # Latin Hypercube Sampling for training data generation
]

# WALRUS requires separate installation due to pinned versions:
# pip install git+https://github.com/PolymathicAI/walrus.git
# pip install "the_well[benchmark] @ git+https://github.com/PolymathicAI/the_well@master"
#
# WALRUS pins: torch==2.5.1, numpy==1.26.4, einops~=0.8,
#              hydra-core>=1.3, timm>=1.0, wandb>=0.17.9, h5py>=3.9.0,<4
#
# IMPORTANT: Use a separate venv for WALRUS training to avoid version conflicts.
# DPF inference can use the main venv if torch version is compatible.
```

### WALRUS Real API Reference

The `surrogate.py` module is **fully implemented** with real IsotropicModel instantiation + RevIN normalization. The real inference pipeline uses:

```python
# Real WALRUS inference pattern (to replace surrogate.py stubs)
from walrus.models import IsotropicModel
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from hydra.utils import instantiate

# 1. Load checkpoint
ckpt = torch.load(checkpoint_path, map_location=device)
config = ckpt["config"]  # Hydra config stored in checkpoint

# 2. Instantiate model
model = instantiate(config.model, n_states=total_input_fields)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(device)

# 3. Inference (delta prediction + RevIN normalization)
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()

with torch.no_grad():
    inputs, y_ref = formatter.process_input(batch, causal_in_time=True, predict_delta=True)
    stats = revin.compute_stats(inputs[0], metadata, epsilon=1e-5)
    normalized_x = revin.normalize_stdmean(inputs[0], stats)
    y_pred = model(normalized_x, inputs[1], inputs[2].tolist(), metadata=metadata)
    y_pred = inputs[0][-y_pred.shape[0]:].float() + revin.denormalize_delta(y_pred, stats)
```

### WALRUS Model Configs
| Config | Hidden Dim | Blocks | Params | Use |
|--------|-----------|--------|--------|-----|
| Pretrain | 768 | 12 | ~300M | Base model training |
| Finetune | 1408 | 40 | 1.3B | Fine-tuning from checkpoint |

### Apple Silicon Hardware Budget (M3 Pro, 36GB)
| Workload | Memory | Status |
|----------|--------|--------|
| Float16 inference | ~2.6 GB | ✅ Easy |
| LoRA fine-tuning (batch=1, grad ckpt) | ~19-25 GB | ✅ Feasible |
| Full fine-tuning (batch=1, grad ckpt) | ~30-35 GB | ⚠️ Tight |
| MLX inference | ~2.6 GB + lower latency | ✅ Recommended |

### WALRUS Integration Gaps (Action Items)
1. **surrogate.py**: Replace stub `_load_model()` with real `IsotropicModel` instantiation + RevIN
2. **surrogate.py**: Replace stub `predict_next_step()` with actual forward pass + delta prediction
3. **well_exporter.py**: Change `grid_type="uniform"` → `"cartesian"`
4. **well_exporter.py**: Add `dim_varying`, `sample_varying`, `time_varying` attributes
5. **batch_runner.py**: Fix WellExporter API call mismatch (lines 199-219)
6. **confidence.py**: Fix `DPFSurrogate.load()` → use constructor instead
7. **realtime_server.py**: Fix 3 wrong API calls (parameter_sweep, optimize, field access)
8. **pyproject.toml**: Document WALRUS installation separately (pinned deps conflict)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| WALRUS fine-tuning fails on DPF data | High | DPF is MHD + circuit (novel coupling). Start with MHD-only fine-tuning, add circuit as auxiliary loss. |
| Not enough training trajectories | Medium | Start with 500 low-res (64×128). Can augment with time-reversal, mirroring, noise injection. |
| Apple Silicon inference too slow | Medium | INT8 quantization, CoreML export, or offload to cloud. |
| WALRUS can't handle cylindrical geometry | Medium | Convert to Cartesian 3D for WALRUS input (embed 2D axisymmetric as thin 3D slice — WALRUS already supports this via tensor-law-aware augmentation). |
| Inverse design finds unphysical configs | Low | Always verify top-N with full DPF engine. Add physics-based constraints to optimizer. |
| Pretrained weights don't transfer to DPF | Low | MHD_64 and MHD_256 are in pretraining data. Compressible MHD transfer is strong. |

---

## Top 3 Reference Codes (unchanged)

| Code | Why It Matters | What We Take |
|------|----------------|-------------|
| **OpenMHD** (Zenitani/JAXA) | Compact resistive MHD, CUDA GPU, reconnection physics | MHD patterns, HLLD reference, GPU design |
| **FLASH** (U. Chicago) | HEDP multi-physics, radiation MHD, experimental validation | Architecture, rad-MHD, validation methodology |
| **MPI-AMRVAC** (KU Leuven) | Powell + Dedner div-B, shock handling, block AMR | Powell sources, Dedner tuning, AMR patterns |

## Key AI Reference

| Tool | Why It Matters | What We Use |
|------|----------------|-------------|
| **WALRUS** (Polymathic AI / Simons Foundation) | 1.3B foundation model pretrained on MHD + 18 other physics domains. MIT licensed. First physics foundation model large enough and general enough to serve as DPF surrogate. | Fine-tune for DPF → fast surrogate predictions → inverse design → real-time Unity visualization |
