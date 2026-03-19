# Examples

## Basic Simulation

```python
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

config = SimulationConfig.from_preset("pf1000")
engine = SimulationEngine(config)
results = engine.run()
```

## Parameter Sweep

```python
import numpy as np
from dpf.config import SimulationConfig
from dpf.engine import SimulationEngine

pressures = np.linspace(1.0, 10.0, 20)
yields = []

for p in pressures:
    config = SimulationConfig.from_preset("pf1000")
    config.physics.fill_pressure = p
    engine = SimulationEngine(config)
    results = engine.run()
    yields.append(results["neutron_yield"])
```

## Custom Device

```python
from dpf.config import SimulationConfig, CircuitConfig, GeometryConfig

config = SimulationConfig(
    circuit=CircuitConfig(V0=45e3, C=120e-6, L0=100e-9, R0=10e-3),
    geometry=GeometryConfig(a=0.115, b=0.32, z0=0.6),
)
```

## Experimental Validation

```python
from dpf.validation.experimental import compare_to_experiment

result = compare_to_experiment(
    sim_results=results,
    experimental_csv="data/pf1000_akel.csv",
)
print(f"NRMSE: {result.nrmse:.3f}")
```

## Neutron Yield Calculation

```python
from dpf.diagnostics.neutron_yield import calculate_neutron_yield

yn = calculate_neutron_yield(
    Te=1e3,           # eV
    n_i=1e24,         # m^-3
    r_pinch=1e-3,     # m
    z_pinch=0.05,     # m
    tau=10e-9,        # s
)
```
