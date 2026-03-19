# Getting Started

## Installation

```bash
pip install dpf-unified

# With Metal GPU support (Apple Silicon):
pip install "dpf-unified[metal]"

# With AI/surrogate support:
pip install "dpf-unified[ai]"

# Full development install:
git clone https://github.com/anthonyzamora/dpf-unified
cd dpf-unified
pip install -e ".[dev,server,metal]"
```

## First Simulation

```python
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

config = SimulationConfig.from_preset("pf1000")
engine = SimulationEngine(config)
results = engine.run()
print(f"Peak current: {results['I_peak']:.2f} MA")
print(f"Neutron yield: {results['neutron_yield']:.2e}")
```

## Using Presets

```python
from dpf.presets import get_preset, get_preset_names

print(get_preset_names())
# ['pf1000', 'nx2', 'unu_ictp', 'llnl_dpf', 'mjolnir', 'faeton_i', 'poseidon']

config = get_preset("mjolnir")
```

## Web UI

```bash
python3 app.py
# open http://localhost:7860
```

Features: device presets, real-time parameter sweeps, 3D animated playback,
experimental data overlay, auto-calibration.

## CLI

```bash
dpf run --preset pf1000 --backend python --output results.h5
dpf sweep --param pressure --range 1.0 10.0 --steps 20
dpf validate --preset pf1000 --experimental data/pf1000_akel.csv
```

## Backends

| Backend | Selection | Notes |
|---------|-----------|-------|
| `python` | `config.fluid.backend = "python"` | NumPy/Numba, always available |
| `metal` | `config.fluid.backend = "metal"` | Apple Silicon only |
| `athena` | `config.fluid.backend = "athena"` | Requires compiled Athena++ |
| `athenak` | `config.fluid.backend = "athenak"` | Requires compiled AthenaK |
| `auto` | `config.fluid.backend = "auto"` | athenak > athena > metal > python |
