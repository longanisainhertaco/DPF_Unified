# DPF Unified — Forward Development Plan (v2)

## Context & What's Done

We are building a modern dense plasma focus (DPF) simulator with:
- **Backend**: Multi-physics MHD solver in Python (NumPy/Numba) with circuit coupling
- **Frontend**: Unity UI with Teaching Mode + Engineering Mode (future)
- **Platform**: Apple Silicon local execution (MVP), HPC cluster (future)
- **AI**: Polymathic AI WALRUS integration for surrogate models and inverse design

### Completed (Steps 1-2 from v1 plan)
- ✅ **README rewrite** — Honest assessment with fidelity grade 3-4/10
- ✅ **Dormant code moved** to `src/dpf/experimental/` (AMR, PIC, GPU, multi-species, ML deleted)
- ✅ **Implicit diffusion (ADI + RKL2)** wired into engine via `_apply_diffusion()`
- ✅ **Line radiation** wired into `_apply_collision_radiation()` with impurity config
- ✅ **Constrained transport** default flipped to ON in cylindrical solver
- ✅ **All 623 tests passing**, 0 failures

**Current fidelity: ~4/10** (up from 3-4/10 before wiring dormant physics)

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

| Phase | Goal | Fidelity | Key Work | Effort |
|-------|------|----------|----------|--------|
| ~~A~~ | ~~README + triage~~ | — | ✅ Done | ✅ Done |
| ~~B~~ | ~~Wire dormant physics~~ | 4/10 | ✅ Done (ADI, RKL2, line rad, CT) | ✅ Done |
| **C** (next) | Verification & validation | 5/10 | Convergence studies, Orszag-Tang, PF-1000 waveform, Lee Model | 4-6 days |
| **D** | Physics improvements | 6/10 | Full Braginskii, Powell div-B, multi-species, aniso conduction | 5-7 days |
| **E** | Apple Silicon optimization | 6/10 fast | Numba prange, Accelerate BLAS, profiling | 2-3 days |
| **F** | WALRUS data pipeline | — | Well exporter, batch trajectory generator, dataset validation | 4-5 days |
| **G** | WALRUS fine-tuning | — | Fine-tune on DPF data, surrogate server, Apple Silicon inference | 5-7 days |
| **H** | AI-powered features | — | Inverse design, real-time server, uncertainty estimation | 4-6 days |
| **I** | Unity frontend | — | Teaching Mode + Engineering Mode, WebSocket integration | TBD |
| **J** | HPC backend | 7-8/10 | MPI decomposition, GPU kernels, production runs | TBD |

**Total to 6/10 fidelity**: ~11-16 days (Steps C-E)
**Total to AI integration**: ~24-37 days (Steps C-H)

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

### New Files to Create
| File | Purpose |
|------|---------|
| `src/dpf/verification/diffusion_convergence.py` | Resistive diffusion convergence test |
| `src/dpf/verification/orszag_tang.py` | Orszag-Tang vortex benchmark |
| `src/dpf/verification/sedov_cylindrical.py` | Cylindrical Sedov blast |
| `src/dpf/validation/lee_model_comparison.py` | Lee Model cross-check |
| `src/dpf/fluid/anisotropic_conduction.py` | Field-aligned thermal conduction |
| `src/dpf/ai/__init__.py` | AI module init |
| `src/dpf/ai/well_exporter.py` | DPF → Well HDF5 format converter |
| `src/dpf/ai/batch_runner.py` | Parallel trajectory generation |
| `src/dpf/ai/dataset_validator.py` | Training data QA |
| `src/dpf/ai/surrogate.py` | WALRUS inference wrapper |
| `src/dpf/ai/inverse_design.py` | Target → config optimizer |
| `src/dpf/ai/realtime_server.py` | WebSocket endpoint for Unity |
| `src/dpf/ai/hybrid_engine.py` | DPF engine + WALRUS handoff |
| `src/dpf/ai/instability_detector.py` | Anomaly detection via WALRUS |

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
# Add to pyproject.toml [project.optional-dependencies]
ai = [
    "torch>=2.1",           # PyTorch with MPS backend for Apple Silicon
    "walrus",               # Polymathic AI WALRUS (pip install from git)
    "the-well",             # Well dataset tools
    "optuna",               # Bayesian optimization for inverse design
    "pyDOE2",               # Latin Hypercube Sampling for training data generation
]
```

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
