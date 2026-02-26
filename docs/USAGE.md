# DPF-Unified Usage Guide

## 1. What is DPF-Unified?

DPF-Unified is a multi-physics magnetohydrodynamics (MHD) simulator for **Dense Plasma Focus** (DPF) devices. It models the complete lifecycle of a DPF discharge: capacitor bank discharge through an RLC circuit, snowplow sheath dynamics, radial plasma compression, and thermonuclear neutron production. The simulator couples a time-dependent circuit solver with a 3D MHD fluid solver, supporting resistive MHD, Braginskii transport, radiation losses, and anomalous resistivity.

**What it models:**
- Magnetohydrodynamic plasma evolution (density, velocity, pressure, magnetic field)
- RLC circuit coupling with plasma load (current, voltage, inductance, resistance)
- Snowplow sheath dynamics (axial rundown, radial implosion, pinch)
- DD thermonuclear and beam-target neutron yield
- Braginskii electron-ion temperature relaxation and transport
- Bremsstrahlung and line radiation losses
- Saha ionization equilibrium
- Anomalous resistivity (Buneman, LHDI, ion-acoustic thresholds)

**Who it's for:** Plasma physics researchers, pulsed power engineers, graduate students studying Z-pinch / DPF physics, and anyone exploring AI-accelerated surrogate models for plasma simulation.

---

## 2. Quick Start

### Installation

```bash
git clone https://github.com/your-org/dpf-unified.git
cd dpf-unified
pip install -e ".[dev]"
```

### Minimal Simulation (5 Lines of Python)

```python
from dpf.config import SimulationConfig
from dpf.presets import get_preset

config = SimulationConfig(**get_preset("tutorial"))
from dpf.engine import SimulationEngine
summary = SimulationEngine(config).run()
```

### Run Your First Simulation

```python
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine
from dpf.presets import get_preset

# Load a preset configuration
config = SimulationConfig(**get_preset("tutorial"))

# Create and run engine
engine = SimulationEngine(config)
summary = engine.run()

# View results
print(f"Steps: {summary['steps']}")
print(f"Peak current: {summary['peak_current_A']:.1f} A")
print(f"Energy conservation: {summary['energy_conservation']:.6f}")
print(f"Neutron yield: {summary['total_neutron_yield']:.2e}")
```

### Step-by-Step Control

```python
engine = SimulationEngine(config)

while True:
    result = engine.step()

    if result.step % 100 == 0:
        print(f"Step {result.step}: t={result.time:.2e} s, I={result.current:.1f} A")

    if result.finished:
        break
```

---

## 3. Backend Selection Guide

DPF-Unified supports four MHD solver backends. Each has different strengths:

| Backend | Speed | Accuracy | GPU | Physics Coverage | Best For |
|---------|-------|----------|-----|------------------|----------|
| `python` | Slow | Teaching-tier | No | Full (non-conservative dE) | Teaching, prototyping, debugging |
| `metal` | Fast | Production-tier | Apple GPU | Conservative + Hall + Braginskii | Production on Apple Silicon |
| `athena` | Fast | Production-tier | No | Conservative (C++ Godunov) | Production on CPU, highest accuracy |
| `athenak` | Fast | Production-tier | CUDA/HIP | Conservative (Kokkos) | HPC GPU clusters |

### Python Backend (default)
- Always available, no compilation needed
- Full physics operator coverage (Nernst, viscosity, radiation, STS, implicit diffusion)
- Uses non-conservative pressure equation (dp/dt) — not suitable for production shock-capturing
- Supports WENO5-Z + HLLD + SSP-RK3 reconstruction

### Metal Backend (Apple Silicon GPU)
- Requires Apple Silicon Mac with PyTorch MPS
- Conservative energy equation, constrained transport for div(B)=0
- Supports Hall MHD, Braginskii conduction/viscosity, Nernst effect, resistive MHD
- Float32 only on GPU; float64 mode forces CPU execution
- Best for grids > 64^3; small grids are faster on CPU

### Athena++ Backend
- Requires building the C++ code with pybind11
- Full Godunov scheme: PPM, HLLD, CT, AMR
- Highest accuracy for shock-capturing MHD
- CPU only (no GPU support)

### AthenaK Backend
- Requires building with CMake + Kokkos
- Runtime physics selection (reconstruction, Riemann solver, ghost zones)
- GPU support via CUDA, HIP, SYCL through Kokkos
- Cartesian mesh only (no native cylindrical coordinates)

### Selecting a Backend

```python
from dpf.config import SimulationConfig

# Explicit backend selection
config = SimulationConfig(
    ...,
    fluid={"backend": "metal"},  # or "python", "athena", "athenak"
)

# Auto-select best available (priority: athena > metal > athenak > python)
config = SimulationConfig(
    ...,
    fluid={"backend": "auto"},
)
```

Check available backends:
```bash
dpf backends
```

---

## 4. Configuration Reference

All configuration is managed through Pydantic v2 models. The top-level model is `SimulationConfig`.

### SimulationConfig (Top-Level)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_shape` | `list[int]` | **required** | Grid dimensions `[nx, ny, nz]` |
| `dx` | `float` | **required** | Grid spacing [m] |
| `sim_time` | `float` | **required** | Total simulation time [s] |
| `dt_init` | `float \| None` | `None` | Initial timestep [s] (auto if None) |
| `rho0` | `float` | `1e-4` | Initial fill gas density [kg/m^3] |
| `T0` | `float` | `300.0` | Initial temperature [K] |
| `anomalous_alpha` | `float` | `0.05` | Anomalous resistivity alpha parameter |
| `anomalous_threshold_model` | `str` | `"ion_acoustic"` | Threshold: `"ion_acoustic"`, `"lhdi"`, `"buneman_classic"` |
| `ion_mass` | `float` | `3.34e-27` | Ion mass [kg] (default: deuterium) |
| `circuit` | `CircuitConfig` | **required** | Circuit parameters |
| `fluid` | `FluidConfig` | defaults | MHD solver parameters |
| `geometry` | `GeometryConfig` | defaults | Coordinate system |
| `diagnostics` | `DiagnosticsConfig` | defaults | Output settings |
| `snowplow` | `SnowplowConfig` | defaults | Sheath dynamics |
| `radiation` | `RadiationConfig` | defaults | Radiation transport |
| `collision` | `CollisionConfig` | defaults | Collision model |
| `sheath` | `SheathConfig` | defaults | Sheath boundary conditions |
| `boundary` | `BoundaryConfig` | defaults | Boundary conditions |
| `ablation` | `AblationConfig` | defaults | Electrode ablation |
| `kinetic` | `KineticConfig` | defaults | PIC kinetic module |
| `ai` | `AIConfig \| None` | `None` | AI/ML surrogate settings |

### CircuitConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `C` | `float` | **required** | Capacitance [F] |
| `V0` | `float` | **required** | Initial voltage [V] |
| `L0` | `float` | **required** | External inductance [H] |
| `R0` | `float` | `0.0` | External resistance [Ohm] |
| `anode_radius` | `float` | **required** | Anode radius [m] |
| `cathode_radius` | `float` | **required** | Cathode radius [m] |
| `ESR` | `float` | `0.0` | Equivalent series resistance [Ohm] |
| `ESL` | `float` | `0.0` | Equivalent series inductance [H] |
| `crowbar_enabled` | `bool` | `False` | Enable crowbar switch |
| `crowbar_mode` | `str` | `"voltage_zero"` | Trigger mode: `"voltage_zero"` or `"fixed_time"` |
| `crowbar_time` | `float` | `0.0` | Trigger time [s] (for `fixed_time` mode) |
| `crowbar_resistance` | `float` | `0.0` | Crowbar switch resistance [Ohm] |

**Validation:** `anode_radius` must be less than `cathode_radius`.

### FluidConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"python"` | MHD backend: `"python"`, `"athena"`, `"athenak"`, `"metal"`, `"hybrid"`, `"auto"` |
| `reconstruction` | `str` | `"weno5"` | Reconstruction scheme |
| `riemann_solver` | `str` | `"hlld"` | Riemann solver: `"hll"`, `"hlld"` |
| `time_integrator` | `str` | `"ssp_rk3"` | Time integrator: `"ssp_rk2"`, `"ssp_rk3"` |
| `precision` | `str` | `"float32"` | Precision: `"float32"` (GPU), `"float64"` (CPU only) |
| `cfl` | `float` | `0.4` | CFL number |
| `gamma` | `float` | `5/3` | Adiabatic index |
| `enable_resistive` | `bool` | `True` | Enable resistive MHD |
| `enable_energy_equation` | `bool` | `True` | Conservative total energy equation |
| `enable_hall` | `bool` | `True` | Enable Hall term |
| `enable_nernst` | `bool` | `False` | Enable Nernst B-field advection |
| `enable_viscosity` | `bool` | `False` | Enable Braginskii ion viscosity |
| `enable_anisotropic_conduction` | `bool` | `False` | Field-aligned thermal conduction |
| `full_braginskii_viscosity` | `bool` | `False` | Full anisotropic Braginskii stress |
| `enable_powell` | `bool` | `False` | Powell 8-wave div(B) source terms |
| `use_ct` | `bool` | `False` | Constrained transport for div(B)=0 |
| `diffusion_method` | `str` | `"explicit"` | `"explicit"`, `"sts"` (RKL2), `"implicit"` (ADI) |
| `sts_stages` | `int` | `8` | RKL2 super time-stepping stages |
| `handoff_fraction` | `float` | `0.1` | Physics fraction before surrogate handoff (hybrid only) |
| `validation_interval` | `int` | `50` | Surrogate validation frequency in steps (hybrid only) |

### GeometryConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | `str` | `"cartesian"` | `"cartesian"` (3D) or `"cylindrical"` (2D axisymmetric) |
| `dz` | `float \| None` | `None` | Axial grid spacing [m] (cylindrical; defaults to dx) |

**Note:** For cylindrical geometry, `grid_shape` must be `[nr, 1, nz]` (ny=1 for axisymmetry).

### DiagnosticsConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hdf5_filename` | `str` | `"diagnostics.h5"` | Output HDF5 file |
| `output_interval` | `int` | `10` | Steps between scalar outputs |
| `field_output_interval` | `int` | `0` | Steps between field snapshots (0 = off) |
| `well_output_interval` | `int` | `0` | Steps between Well-format snapshots (0 = off) |
| `well_filename_prefix` | `str` | `"well_output"` | Prefix for Well-format output files |

### SnowplowConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable snowplow sheath dynamics |
| `mass_fraction` | `float` | `0.3` | Fraction of fill gas swept (f_m) |
| `fill_pressure_Pa` | `float` | `400.0` | Fill gas pressure [Pa] |
| `anode_length` | `float` | `0.16` | Anode length / rundown distance [m] |
| `current_fraction` | `float` | `0.7` | Fraction of current in sheath (f_c) |
| `radial_mass_fraction` | `float \| None` | `None` | Radial sweep fraction (defaults to mass_fraction) |

### RadiationConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bremsstrahlung_enabled` | `bool` | `True` | Enable bremsstrahlung radiation |
| `gaunt_factor` | `float` | `1.2` | Gaunt factor |
| `fld_enabled` | `bool` | `False` | Enable flux-limited diffusion |
| `flux_limiter` | `float` | `1/3` | Flux limiter lambda |
| `line_radiation_enabled` | `bool` | `False` | Enable impurity line radiation |
| `impurity_Z` | `float` | `29.0` | Impurity atomic number (default: copper) |
| `impurity_fraction` | `float` | `0.0` | Impurity density fraction |

### AIConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surrogate_checkpoint` | `str \| None` | `None` | Path to WALRUS checkpoint |
| `device` | `str` | `"cpu"` | Inference device: `"cpu"`, `"mps"`, `"cuda"` |
| `history_length` | `int` | `4` | Timestep history window for WALRUS |
| `ensemble_size` | `int` | `1` | Number of ensemble models |
| `confidence_threshold` | `float` | `0.8` | Minimum prediction confidence |
| `sweep` | `SweepConfig` | defaults | Parameter sweep settings |
| `inverse` | `InverseConfig` | defaults | Inverse design settings |

### Loading Config from File

```python
# From JSON file
config = SimulationConfig.from_file("my_config.json")

# Save to JSON
config.to_json("output_config.json")
```

---

## 5. Running Simulations

### CLI Usage

```bash
# Run a simulation
dpf simulate config.json

# Override backend
dpf simulate config.json --backend metal

# Limit steps
dpf simulate config.json --steps 1000

# Override output file
dpf simulate config.json -o results.h5

# Restart from checkpoint
dpf simulate config.json --restart checkpoint.h5

# Auto-checkpoint every 500 steps
dpf simulate config.json --checkpoint-interval 500

# Validate config file
dpf verify config.json

# Show available backends
dpf backends

# Show Metal GPU info
dpf metal-info
```

### Python API

```python
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

# Create config
config = SimulationConfig(
    grid_shape=[64, 1, 128],
    dx=5e-4,
    sim_time=3e-6,
    circuit={
        "C": 1e-3,
        "V0": 20e3,
        "L0": 30e-9,
        "R0": 5e-3,
        "anode_radius": 0.01,
        "cathode_radius": 0.03,
    },
    geometry={"type": "cylindrical"},
    fluid={"backend": "python"},
)

# Run to completion
engine = SimulationEngine(config)
summary = engine.run()

# Or step-by-step with custom logic
engine = SimulationEngine(config)
while True:
    result = engine.step()

    # Access field data
    snapshot = engine.get_field_snapshot()
    # snapshot keys: rho, velocity, pressure, B, Te, Ti, psi

    # Access circuit state
    print(f"Current: {result.current:.1f} A")

    if result.finished:
        break

# Checkpointing
engine.save_checkpoint("my_checkpoint.h5")
engine.load_from_checkpoint("my_checkpoint.h5")
```

### REST API

Start the server:
```bash
dpf serve --host 0.0.0.0 --port 8765
```

Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check with backend availability |
| POST | `/api/simulations` | Create a simulation |
| GET | `/api/simulations/{id}` | Get simulation status |
| POST | `/api/simulations/{id}/start` | Start running |
| POST | `/api/simulations/{id}/pause` | Pause simulation |
| POST | `/api/simulations/{id}/resume` | Resume simulation |
| POST | `/api/simulations/{id}/stop` | Stop simulation |
| GET | `/api/simulations/{id}/fields` | Get field snapshot metadata |
| GET | `/api/config/schema` | JSON Schema for SimulationConfig |
| POST | `/api/config/validate` | Validate a config dict |
| GET | `/api/presets` | List device presets |
| WS | `/ws/{sim_id}` | Real-time scalar streaming + field requests |

Example: Create and run a simulation via REST:

```bash
# Create from preset
curl -X POST http://localhost:8765/api/simulations \
  -H "Content-Type: application/json" \
  -d '{"preset": "tutorial"}'

# Start it
curl -X POST http://localhost:8765/api/simulations/{sim_id}/start
```

Interactive API docs are available at `http://localhost:8765/docs`.

### Parameter Sweeps

```bash
# Via CLI
dpf sweep sweep_config.json -o sweep_output/ -w 4
```

Sweep config file format:

```json
{
  "base_config": {
    "grid_shape": [64, 1, 128],
    "dx": 5e-4,
    "sim_time": 3e-6,
    "circuit": {
      "C": 1e-3, "V0": 20e3, "L0": 30e-9, "R0": 5e-3,
      "anode_radius": 0.01, "cathode_radius": 0.03
    },
    "geometry": {"type": "cylindrical"}
  },
  "parameter_ranges": [
    {"name": "circuit.V0", "low": 10000, "high": 40000},
    {"name": "circuit.C", "low": 0.0005, "high": 0.005, "log_scale": true}
  ],
  "n_samples": 50
}
```

Via Python:

```python
from dpf.ai.batch_runner import BatchRunner, ParameterRange
from dpf.config import SimulationConfig
from dpf.presets import get_preset

base_config = SimulationConfig(**get_preset("tutorial"))
ranges = [
    ParameterRange("circuit.V0", low=1e3, high=10e3),
    ParameterRange("circuit.C", low=0.5e-6, high=5e-6, log_scale=True),
]

runner = BatchRunner(
    base_config=base_config,
    parameter_ranges=ranges,
    n_samples=20,
    output_dir="sweep_data",
    workers=4,
)
result = runner.run()
print(f"Success: {result.n_success}/{result.n_total}")
```

---

## 6. AI/WALRUS Surrogate

DPF-Unified integrates [WALRUS](https://github.com/PolymathicAI/walrus) (Polymathic AI), a 1.3B-parameter Transformer foundation model for continuum dynamics, as a surrogate for fast plasma predictions.

### What It Does

- **Fast predictions:** Replace expensive MHD time-stepping with neural network inference (~100x faster)
- **Parameter sweeps:** Evaluate hundreds of configurations in minutes instead of hours
- **Inverse design:** Find device configurations that achieve target plasma performance
- **Uncertainty quantification:** Ensemble predictions with confidence estimation

### Loading a Checkpoint

```python
from dpf.ai.surrogate import DPFSurrogate

# From explicit path
surrogate = DPFSurrogate("models/walrus-pretrained/walrus.pt", device="cpu")

# Auto-discover (checks WALRUS_CHECKPOINT env var, models/, ~/.dpf/)
surrogate = DPFSurrogate(device="mps")  # Apple Silicon GPU

# Check status
print(surrogate.is_loaded)        # True if model ready
print(surrogate.history_length)   # Number of input timesteps needed (default: 4)
```

### Running Inference

```python
from dpf.engine import SimulationEngine
from dpf.config import SimulationConfig
from dpf.presets import get_preset

# Generate initial states from physics
config = SimulationConfig(**get_preset("tutorial"))
engine = SimulationEngine(config)
history = []
for _ in range(surrogate.history_length):
    engine.step()
    history.append(engine.get_field_snapshot())

# Single-step prediction
next_state = surrogate.predict_next_step(history)

# Multi-step autoregressive rollout
trajectory = surrogate.rollout(history, n_steps=100)
print(f"Predicted {len(trajectory)} steps")
```

### Running Inverse Design

```python
from dpf.ai.inverse_design import InverseDesigner

designer = InverseDesigner(
    surrogate=surrogate,
    parameter_ranges={
        "V0": (5e3, 50e3),
        "C": (1e-6, 100e-6),
        "rho0": (1e-5, 1e-3),
    },
)

result = designer.find_config(
    targets={"max_Te": 1e7, "max_rho": 1e-2},
    method="bayesian",   # or "evolutionary"
    n_trials=100,
)

print(f"Best score: {result.best_score:.4e}")
print(f"Best params: {result.best_params}")
```

### CLI Commands

```bash
# Surrogate prediction
dpf predict config.json --checkpoint models/walrus-pretrained/ --steps 100 --device mps

# Inverse design
dpf inverse targets.json --checkpoint models/walrus-pretrained/ --method bayesian --n-trials 200

# Start AI inference server
dpf serve-ai --checkpoint models/walrus-pretrained/ --device mps --port 8766

# Export training data in Well format
dpf export-well config.json -o training_data.h5 --field-interval 10

# Validate training dataset
dpf validate-dataset training_data/
```

### AI REST API Endpoints

When running with `dpf serve --checkpoint ...` or `dpf serve-ai`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/ai/status` | Model status and device info |
| POST | `/api/ai/predict` | Single-step prediction from history |
| POST | `/api/ai/rollout` | Multi-step autoregressive rollout |
| POST | `/api/ai/sweep` | Surrogate-accelerated parameter sweep |
| POST | `/api/ai/inverse` | Inverse design optimization |
| POST | `/api/ai/confidence` | Ensemble prediction with uncertainty |
| POST | `/api/ai/validate` | Cross-validate surrogate vs physics trajectory |
| POST | `/api/ai/chat` | Natural language query interface |
| WS | `/api/ai/ws/stream` | WebSocket streaming rollout |

### Training Data Pipeline

1. **Generate trajectories:** Use parameter sweeps to create diverse training data
2. **Export to Well format:** `dpf export-well` creates HDF5 files compatible with The Well
3. **Validate:** `dpf validate-dataset` checks for NaN/Inf, schema compliance, energy conservation
4. **Fine-tune WALRUS:** Use the WALRUS training script with DPF-specific data

```bash
# Step 1: Generate training data
dpf sweep sweep_config.json -o training_data/ -w 4

# Step 2: Validate
dpf validate-dataset training_data/

# Step 3: Fine-tune (requires walrus package + separate venv)
python train.py \
    distribution=local \
    model=isotropic_model \
    finetune=True \
    optimizer=adam optimizer.lr=1.e-4 \
    trainer.enable_amp=False \
    trainer.prediction_type="delta" \
    model.causal_in_time=True
```

---

## 7. Diagnostics & Output

### HDF5 Output

Simulations produce an HDF5 file (default: `diagnostics.h5`) containing:
- **Scalar time series:** current, voltage, energy components, R_plasma, Z_bar, neutron yield
- **Field snapshots** (if `field_output_interval > 0`): rho, velocity, pressure, B, Te, Ti

### Key Metrics

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `current` | Circuit current [A] | 0 to ~2 MA (PF-1000) |
| `voltage` | Capacitor voltage [V] | V0 down to 0 |
| `energy_conservation` | E_total / E_initial | 0.95-1.05 (good), >1.1 (problem) |
| `R_plasma` | Plasma resistance [Ohm] | 1e-3 to 10 |
| `Z_bar` | Average ionization state | 1 to ~20+ |
| `max_Te` | Peak electron temperature [K] | 300 to ~1e8 |
| `total_neutron_yield` | Cumulative DD neutrons | 0 to ~1e11 |
| `neutron_rate` | Instantaneous production [1/s] | 0 to ~1e18 |

### StepResult Fields

Every call to `engine.step()` returns a `StepResult` dataclass:

```python
result = engine.step()
result.time                  # Simulation time [s]
result.step                  # Step number
result.dt                    # Timestep used [s]
result.current               # Circuit current [A]
result.voltage               # Capacitor voltage [V]
result.energy_conservation   # E_total / E_initial
result.max_Te                # Peak electron temperature [K]
result.max_rho               # Peak density [kg/m^3]
result.Z_bar                 # Average ionization
result.R_plasma              # Plasma resistance [Ohm]
result.eta_anomalous         # Anomalous resistivity [Ohm*m]
result.total_radiated_energy # Cumulative radiated energy [J]
result.neutron_rate          # DD neutron rate [1/s]
result.total_neutron_yield   # Cumulative neutron yield
result.finished              # True when complete
```

### Checkpointing

```python
# Save checkpoint (includes full state + circuit + snowplow)
engine.save_checkpoint("checkpoint.h5")

# Set auto-checkpoint interval
engine.checkpoint_interval = 500  # every 500 steps

# Load and resume
engine.load_from_checkpoint("checkpoint.h5")
summary = engine.run()
```

---

## 8. Example Configurations

### PF-1000 (Large Research Device, 1 MJ)

```python
from dpf.config import SimulationConfig
from dpf.presets import get_preset

config = SimulationConfig(**get_preset("pf1000"))
# Grid: 128x1x256 cylindrical
# Circuit: C=1.332 mF, V0=27 kV, L0=33.5 nH
# Crowbar enabled, electrode BCs, bremsstrahlung + FLD radiation
```

### NX2 (Compact DPF, 3 kJ)

```python
config = SimulationConfig(**get_preset("nx2"))
# Grid: 192x1x384 cylindrical
# Circuit: C=28 uF, V0=14 kV, L0=20 nH
# Fine grid (dx=0.25 mm) for the small device
```

### Quick Tutorial (8^3 Cartesian)

```python
config = SimulationConfig(**get_preset("tutorial"))
# Grid: 8x8x8 Cartesian
# Minimal circuit parameters
# Runs in seconds
```

### High-Fidelity Configuration (Phase P, 8.9/10 Accuracy)

```python
config = SimulationConfig(**get_preset("phase_p_fidelity"))
# WENO5-Z + HLLD + SSP-RK3 + float64
# Maximum accuracy for verification & validation
```

### Custom Metal GPU Configuration

```python
config = SimulationConfig(
    grid_shape=[64, 64, 64],
    dx=5e-4,
    sim_time=1e-6,
    circuit={
        "C": 5e-6, "V0": 10e3, "L0": 50e-9, "R0": 0.01,
        "anode_radius": 0.005, "cathode_radius": 0.01,
    },
    fluid={
        "backend": "metal",
        "reconstruction": "weno5",
        "riemann_solver": "hlld",
        "time_integrator": "ssp_rk3",
        "use_ct": True,
        "enable_hall": True,
        "enable_viscosity": True,
    },
    radiation={"bremsstrahlung_enabled": True},
)
```

### Hybrid Backend (Physics + AI Surrogate)

```python
config = SimulationConfig(
    grid_shape=[32, 32, 32],
    dx=5e-4,
    sim_time=5e-6,
    circuit={
        "C": 5e-6, "V0": 5e3, "L0": 50e-9, "R0": 0.01,
        "anode_radius": 0.005, "cathode_radius": 0.01,
    },
    fluid={
        "backend": "hybrid",
        "handoff_fraction": 0.1,      # 10% physics, 90% surrogate
        "validation_interval": 50,     # Validate every 50 steps
    },
    ai={
        "surrogate_checkpoint": "models/walrus-pretrained/walrus.pt",
        "device": "mps",
    },
)
```

### Available Presets

| Preset | Device | Grid | Geometry | Description |
|--------|--------|------|----------|-------------|
| `tutorial` | Generic | 8x8x8 | Cartesian | Quick test, runs in seconds |
| `pf1000` | PF-1000 | 128x1x256 | Cylindrical | 1 MJ Warsaw DPF |
| `nx2` | NX2 | 192x1x384 | Cylindrical | 3 kJ Singapore DPF |
| `llnl_dpf` | LLNL | 64x1x128 | Cylindrical | 4 kJ compact DPF |
| `cartesian_demo` | Generic | 32x32x32 | Cartesian | All physics enabled |
| `phase_p_fidelity` | Generic | 32x32x32 | Cartesian | Max accuracy (WENO5-Z + HLLD + SSP-RK3 + float64) |

List all presets programmatically:

```python
from dpf.presets import list_presets
for p in list_presets():
    print(f"{p['name']:20s} {p['description']}")
```
