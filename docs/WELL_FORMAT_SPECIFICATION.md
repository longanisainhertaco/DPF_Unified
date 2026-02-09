# The Well Dataset Format Specification

Complete technical reference for PolymathicAI's WALRUS (Well-structured Archive for Learning and Research in Universal Science) dataset format.

**Sources:**
- Official documentation: https://polymathic-ai.org/the_well/
- GitHub repository: https://github.com/PolymathicAI/the_well
- API documentation: https://polymathic-ai.org/the_well/api/

---

## Overview

The Well is a large-scale scientific dataset format (15TB across 16 datasets) designed for training machine learning models on multi-physics simulations. It uses HDF5 (Hierarchical Data Format v5) with a standardized schema for uniform access across diverse scientific domains.

**Key characteristics:**
- Single-precision float32 throughout
- Uniform spatial grids with constant time intervals
- Batch dimension for multiple trajectories
- Tensor-rank-based field organization (scalars, vectors, tensors)
- Comprehensive metadata for physical units and boundary conditions

---

## HDF5 Schema Structure

### Root Attributes

Required attributes on the HDF5 file root:

| Attribute | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `@dataset_name` | `str` | Unique dataset identifier | `"dpf_simulation"` |
| `@grid_type` | `str` | Coordinate system type | `"cartesian"`, `"spherical"` |
| `@n_spatial_dims` | `int` | Number of spatial dimensions | `2`, `3` |
| `@n_trajectories` | `int` | Batch dimension size | `1`, `100` |
| `@simulation_parameters` | `list[str]` | List of parameter names | `["Re", "Pr", "Ma"]` |
| `@{ParamA}` | `float` | Individual parameter values | `@Re: 1000.0` |

**Notes:**
- At least one of `@dataset_name` or `@n_trajectories` must be present
- `@simulation_parameters` lists the names of parameters; their values are stored as separate root attributes
- Custom simulation parameters can be added as additional root attributes

---

### Group: `/dimensions/`

Stores spatial and temporal coordinate information.

**Required datasets:**

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `time` | `(T,)` | `float32` | Time values for all snapshots |
| Spatial coords | `(N,)` | `float32` | Grid coordinates (see below) |

**Spatial coordinate names depend on `@grid_type`:**

- **Cartesian** (`grid_type="cartesian"`): `x`, `y`, `z`
- **Cylindrical**: `r`, `theta`, `z`
- **Spherical** (`grid_type="spherical"`): `r`, `theta`, `phi`

**Attributes on each dimension dataset:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@sample_varying` | `bool` | Whether coordinate varies across trajectories |
| `@time_varying` | `bool` | Whether coordinate varies over time |

**Additional group attribute:**

- `@spatial_dims`: `list[str]` — ordered list of spatial dimension names (e.g., `['x', 'y', 'z']`)

**Example:**
```python
dimensions/
  @spatial_dims = ['x', 'y', 'z']
  time: shape (100,), dtype float32
    @sample_varying = False
    @time_varying = False
  x: shape (64,), dtype float32
    @sample_varying = False
    @time_varying = False
  y: shape (64,), dtype float32
  z: shape (64,), dtype float32
```

---

### Group: `/boundary_conditions/`

Specifies boundary condition type and location for each physical domain boundary.

**Structure:**
- Subgroup per boundary (e.g., `X_boundary`, `Y_boundary`)
- Each subgroup contains mask and value datasets plus metadata attributes

**Attributes on each boundary subgroup:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@associated_dims` | `list[str]` | Dimensions this BC applies to (e.g., `['x']`) |
| `@associated_fields` | `list[str]` | Fields this BC affects |
| `@bc_type` | `str` | Boundary condition type |

**Valid `@bc_type` values:**
- `"periodic"` — periodic wrapping
- `"wall"` — no-slip or reflective wall
- `"open"` — outflow/non-reflecting

**Datasets in each boundary subgroup:**

| Dataset | Dtype | Description |
|---------|-------|-------------|
| `mask` | `bool` | Boolean array marking boundary locations |
| `values` | `float32` | Field values at masked boundary points |

**Simplified alternative:**
For uniform BCs, attributes can be set directly on the `/boundary_conditions/` group:
```python
boundary_conditions/
  @geometry_type = "cartesian"
  @all = "periodic"
```

**Example:**
```python
boundary_conditions/
  X_boundary/
    @associated_dims = ['x']
    @associated_fields = ['density', 'velocity']
    @bc_type = "wall"
    mask: shape (64, 64, 64), dtype bool
    values: shape (64, 64, 64), dtype float32
```

---

### Group: `/scalars/`

Non-spatial scalar quantities that vary over time and/or across trajectories (e.g., integrated diagnostics, circuit parameters).

**Group attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@field_names` | `list[str]` | List of scalar field names in this group |

**Shape convention:**
- `(n_trajectories, n_steps)` for time-varying scalars

**Attributes on each scalar dataset:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@sample_varying` | `bool` | Whether scalar varies across trajectories |
| `@time_varying` | `bool` | Whether scalar varies over time |

**Example:**
```python
scalars/
  @field_names = ['current', 'voltage', 'energy_conservation']
  current: shape (1, 100), dtype float32
    @sample_varying = False
    @time_varying = True
  voltage: shape (1, 100), dtype float32
  energy_conservation: shape (1, 100), dtype float32
```

---

### Group: `/t0_fields/` (Scalar Fields)

Spatially-resolved scalar fields (e.g., density, temperature, pressure).

**Shape convention:**
```
(n_trajectories, n_steps, coord1, coord2, [coord3])
```

For example, with 1 trajectory, 100 timesteps, and a 64×64×64 grid:
```
(1, 100, 64, 64, 64)
```

**Attributes on each field dataset:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@dim_varying` | `list[bool]` | Which spatial dims vary (e.g., `[True, True, True]`) |
| `@sample_varying` | `bool` | Whether field varies across trajectories |
| `@time_varying` | `bool` | Whether field varies over time |
| `@units` | `str` | Optional physical units (e.g., `"kg/m^3"`) |

**Example:**
```python
t0_fields/
  density: shape (1, 100, 64, 64, 64), dtype float32
    @dim_varying = [True, True, True]
    @sample_varying = False
    @time_varying = True
    @units = "kg/m^3"
  electron_temp: shape (1, 100, 64, 64, 64), dtype float32
    @units = "K"
```

---

### Group: `/t1_fields/` (Vector Fields)

Spatially-resolved vector fields (e.g., velocity, magnetic field).

**Shape convention:**
```
(n_trajectories, n_steps, coord1, coord2, [coord3], D)
```

Where `D` is the vector dimension (typically 3 for 3D vectors). The **last axis** contains vector components.

For example:
```
(1, 100, 64, 64, 64, 3)
```

**Attributes:** Same as `t0_fields` (see above).

**Example:**
```python
t1_fields/
  velocity: shape (1, 100, 64, 64, 64, 3), dtype float32
    @dim_varying = [True, True, True]
    @sample_varying = False
    @time_varying = True
    @units = "m/s"
  magnetic_field: shape (1, 100, 64, 64, 64, 3), dtype float32
    @units = "T"
```

---

### Group: `/t2_fields/` (Tensor Fields)

Spatially-resolved tensor fields (e.g., stress tensor, magnetic pressure tensor).

**Shape convention:**
```
(n_trajectories, n_steps, coord1, coord2, [coord3], D²)
```

The **last axis** contains tensor components (flattened). For 3D tensors, `D² = 9`.

**Additional attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `@symmetric` | `bool` | Whether tensor is symmetric |
| `@antisymmetric` | `bool` | Whether tensor is antisymmetric |

**Example:**
```python
t2_fields/
  stress_tensor: shape (1, 100, 64, 64, 64, 9), dtype float32
    @dim_varying = [True, True, True]
    @sample_varying = False
    @time_varying = True
    @symmetric = True
    @antisymmetric = False
    @units = "Pa"
```

---

## Field Naming Conventions

**The Well format does not enforce specific field names**, but the following conventions are recommended for consistency:

### Recommended Scalar Field Names (`t0_fields`)

| Field Name | Description | Typical Units |
|------------|-------------|---------------|
| `density` | Mass density | `kg/m^3` |
| `pressure` | Thermodynamic pressure | `Pa` |
| `electron_temp` | Electron temperature | `K` or `eV` |
| `ion_temp` | Ion temperature | `K` or `eV` |
| `energy_density` | Total energy density | `J/m^3` |
| `magnetic_pressure` | Magnetic pressure (B²/2μ₀) | `Pa` |
| `vorticity_magnitude` | Magnitude of vorticity | `1/s` |

### Recommended Vector Field Names (`t1_fields`)

| Field Name | Description | Typical Units |
|------------|-------------|---------------|
| `velocity` | Fluid velocity | `m/s` |
| `magnetic_field` | Magnetic field | `T` (Tesla) |
| `current_density` | Electric current density | `A/m^2` |
| `electric_field` | Electric field | `V/m` |
| `momentum_density` | Momentum density | `kg/(m^2·s)` |

### Recommended Scalar Names (`scalars` group)

| Scalar Name | Description | Typical Units |
|-------------|-------------|---------------|
| `current` | Total circuit current | `A` |
| `voltage` | Circuit voltage | `V` |
| `energy_conservation` | Energy conservation ratio | dimensionless (≈1.0) |
| `total_mass` | Total system mass | `kg` |
| `total_energy` | Total system energy | `J` |
| `neutron_rate` | Neutron production rate | `1/s` |

---

## Python Tools and Utilities

### Official Well Library

The `the-well` Python package provides tools for **loading** existing Well datasets:

**Installation:**
```bash
pip install the-well
```

**Key classes:**

#### `WellDataset`
PyTorch-compatible dataset loader for Well HDF5 files.

```python
from the_well.data import WellDataset

dataset = WellDataset(
    path="/path/to/well/datasets",
    well_dataset_name="dpf_simulation",
    well_split_name="train",  # 'train', 'valid', or 'test'
    n_steps_input=10,         # Input sequence length
    n_steps_output=1,         # Output sequence length
    use_normalization=True,   # Enable data normalization
    boundary_return_type="padding"  # 'padding', 'mask', 'exact', 'none'
)

# Load a sample
sample = dataset[0]  # Returns dict with fields as torch.Tensor
```

**Key method:**
- `to_xarray(backend)`: Export dataset to Xarray for analysis

#### `WellDataModule`
PyTorch Lightning DataModule wrapper for distributed training.

```python
from the_well.data import WellDataModule

datamodule = WellDataModule(
    path="/path/to/datasets",
    dataset_name="dpf_simulation",
    batch_size=32,
    world_size=4,  # Number of distributed processes
    rank=0         # Process rank
)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
rollout_loader = datamodule.rollout_val_dataloader()  # For long rollouts
```

### Creating Well Datasets

**The official Well library does NOT provide utilities for creating new datasets.** Users must write HDF5 files directly using `h5py`.

**DPF-specific tools:**

The DPF project provides custom exporters in `/Users/anthonyzamora/dpf-unified/src/dpf/ai/`:

1. **`well_exporter.py`**: `WellExporter` class for converting DPF simulation output to Well format
2. **`field_mapping.py`**: Field name mappings and conversion utilities
3. **`dataset_validator.py`**: `DatasetValidator` for schema compliance checking
4. **`batch_runner.py`**: Batch simulation runner for generating training datasets

**Example usage:**
```python
from dpf.ai.well_exporter import WellExporter

exporter = WellExporter(
    output_path="output.h5",
    grid_shape=(64, 64, 64),
    dx=1e-4,  # Grid spacing [m]
    geometry="cartesian",
    sim_params={"V0": 50000.0, "pressure_Pa": 100.0}
)

# Add snapshots during simulation
exporter.add_snapshot(state, time=1e-6, circuit_scalars={"current": 1e5})

# Or load from existing DPF HDF5 file
exporter.add_from_dpf_hdf5("diagnostics.h5")

# Write to Well format
exporter.finalize(n_trajectories=1)
```

**Validation:**
```python
from dpf.ai.dataset_validator import DatasetValidator

validator = DatasetValidator(energy_drift_threshold=0.05)
result = validator.validate_file("output.h5")

if result.valid:
    print(f"Valid dataset: {result.n_trajectories} trajectories, {result.n_timesteps} timesteps")
else:
    print("Validation errors:", result.errors)
```

---

## Requirements and Constraints

### Grid Types

Supported grid types (set via `@grid_type` attribute):
- `"cartesian"`: Uniform Cartesian grids (x, y, z)
- `"spherical"`: Spherical coordinates (r, θ, φ)

**Custom grid types** can be defined but may not be compatible with official Well loaders.

### Data Type Requirements

- **All numeric data must be `float32`** (single precision)
- Coordinate arrays: `float32`
- Field arrays: `float32`
- Time arrays: `float32`
- Boolean masks: `bool` (for boundary conditions)

### Shape Requirements

- **Uniform grids**: All spatial dimensions must be uniformly spaced
- **Constant time intervals**: Time sampling must be uniform (constant Δt)
- **Consistent shapes**: All fields in a group must share the same trajectory and timestep dimensions

### Attribute Requirements

**Mandatory attributes:**
- Root: `@dataset_name` or `@n_trajectories` (at least one)
- Root: `@grid_type`, `@n_spatial_dims`
- `/dimensions/`: `@spatial_dims` on group
- `/dimensions/{coord}`: `@sample_varying`, `@time_varying` on each coordinate
- `/t0_fields/{field}`: `@dim_varying`, `@sample_varying`, `@time_varying`
- `/t1_fields/{field}`: Same as `t0_fields`
- `/t2_fields/{field}`: Same as `t0_fields`, plus optional `@symmetric`, `@antisymmetric`

**Recommended attributes:**
- `@units` on all physical field datasets
- `@simulation_parameters` listing parameter names
- Individual `@{ParamName}` attributes for parameter values

---

## Example Well File Structure

Complete example for a 2D MHD simulation:

```
output.h5
├── @dataset_name = "dpf_zpinch"
├── @grid_type = "cartesian"
├── @n_spatial_dims = 3
├── @n_trajectories = 1
├── @simulation_parameters = ["V0", "pressure_Pa", "fill_gas"]
├── @V0 = 50000.0
├── @pressure_Pa = 100.0
├── @fill_gas = "deuterium"
│
├── dimensions/
│   ├── @spatial_dims = ['x', 'y', 'z']
│   ├── time: (100,) float32
│   │   ├── @sample_varying = False
│   │   └── @time_varying = False
│   ├── x: (64,) float32
│   ├── y: (64,) float32
│   └── z: (64,) float32
│
├── boundary_conditions/
│   ├── @geometry_type = "cartesian"
│   └── @all = "periodic"
│
├── scalars/
│   ├── @field_names = ['current', 'voltage', 'energy_conservation']
│   ├── current: (1, 100) float32
│   ├── voltage: (1, 100) float32
│   └── energy_conservation: (1, 100) float32
│
├── t0_fields/
│   ├── density: (1, 100, 64, 64, 64) float32
│   │   ├── @dim_varying = [True, True, True]
│   │   ├── @sample_varying = False
│   │   ├── @time_varying = True
│   │   └── @units = "kg/m^3"
│   ├── electron_temp: (1, 100, 64, 64, 64) float32
│   │   └── @units = "K"
│   ├── ion_temp: (1, 100, 64, 64, 64) float32
│   └── pressure: (1, 100, 64, 64, 64) float32
│       └── @units = "Pa"
│
└── t1_fields/
    ├── velocity: (1, 100, 64, 64, 64, 3) float32
    │   ├── @dim_varying = [True, True, True]
    │   ├── @sample_varying = False
    │   ├── @time_varying = True
    │   └── @units = "m/s"
    └── magnetic_field: (1, 100, 64, 64, 64, 3) float32
        └── @units = "T"
```

---

## DPF-Specific Field Mappings

The DPF project uses the following mappings (defined in `/Users/anthonyzamora/dpf-unified/src/dpf/ai/field_mapping.py`):

### Scalar Fields (DPF → Well)

| DPF Name | Well Name | Units |
|----------|-----------|-------|
| `rho` | `density` | `kg/m^3` |
| `Te` | `electron_temp` | `K` |
| `Ti` | `ion_temp` | `K` |
| `pressure` | `pressure` | `Pa` |
| `psi` | `dedner_psi` | `T*m/s` |

### Vector Fields (DPF → Well)

| DPF Name | Well Name | Units | Layout |
|----------|-----------|-------|--------|
| `B` | `magnetic_field` | `T` | (3, nx, ny, nz) → (nx, ny, nz, 3) |
| `velocity` | `velocity` | `m/s` | (3, nx, ny, nz) → (nx, ny, nz, 3) |

**Note:** DPF uses component-first layout `(3, nx, ny, nz)`, while Well uses component-last `(nx, ny, nz, 3)`. The exporter automatically transposes.

### Circuit Scalars

| Scalar Name | Description | Units |
|-------------|-------------|-------|
| `current` | Circuit current | `A` |
| `voltage` | Circuit voltage | `V` |
| `energy_conservation` | Energy ratio (should ≈ 1.0) | dimensionless |
| `R_plasma` | Plasma resistance | `Ω` |
| `Z_bar` | Average ionization | dimensionless |
| `total_radiated_energy` | Cumulative radiated energy | `J` |
| `neutron_rate` | Neutron production rate | `1/s` |
| `total_neutron_yield` | Cumulative neutron yield | dimensionless |

---

## Validation Checklist

Use this checklist when creating Well-format datasets:

**Schema compliance:**
- [ ] Root attributes `@dataset_name`, `@grid_type`, `@n_spatial_dims`, `@n_trajectories` present
- [ ] `/dimensions/` group exists with time and spatial coordinates
- [ ] `/boundary_conditions/` group exists
- [ ] At least one of `/t0_fields/` or `/t1_fields/` groups present
- [ ] All dimension datasets have `@sample_varying` and `@time_varying` attributes
- [ ] All field datasets have `@dim_varying`, `@sample_varying`, `@time_varying` attributes

**Data type compliance:**
- [ ] All numeric data is `float32`
- [ ] No `float64` arrays (these will fail to load in official loaders)
- [ ] Boundary condition masks are `bool`

**Numerical validity:**
- [ ] No NaN values in any field dataset
- [ ] No Inf values in any field dataset
- [ ] Energy conservation (if present) stays within acceptable bounds (e.g., ± 5%)

**Shape consistency:**
- [ ] All `t0_fields` share the same `(n_traj, n_steps, ...)` prefix
- [ ] All `t1_fields` share the same `(n_traj, n_steps, ..., D)` shape
- [ ] Spatial dimensions match between coordinates and field arrays

**Recommended:**
- [ ] Include `@units` attributes on all physical fields
- [ ] Include `@simulation_parameters` and individual parameter attributes
- [ ] Include `/scalars/` group with integrated diagnostics (especially `energy_conservation`)

---

## References

- **The Well project**: https://polymathic-ai.org/the_well/
- **GitHub repository**: https://github.com/PolymathicAI/the_well
- **API documentation**: https://polymathic-ai.org/the_well/api/
- **Data format specification**: https://polymathic-ai.org/the_well/data_format/
- **arXiv paper**: https://arxiv.org/abs/2412.00568 (NeurIPS 2024)
- **Contact**: Ruben Ohana and Michael McCabe at {rohana,mmccabe}@flatironinstitute.org

---

## Frequently Asked Questions

### Can I use float64 instead of float32?

No. The Well format requires single-precision `float32` throughout. Using `float64` will cause loader failures and significantly increase file sizes. Convert all data to `float32` before writing.

### How do I handle non-uniform grids?

The Well format **only supports uniform grids**. For non-uniform grids, you must either:
1. Interpolate to a uniform grid before export
2. Use a custom format (but you won't be able to use official Well loaders)

### Can I store additional metadata?

Yes. You can add custom attributes at the root level or on individual datasets. The schema is extensible. However, custom attributes may be ignored by standard loaders.

### What if my simulation uses cylindrical coordinates?

For cylindrical coordinates with `(r, θ, z)`, you have two options:
1. Set `@grid_type = "spherical"` and use coordinate names `r`, `theta`, `z` (non-standard but works)
2. Interpolate to Cartesian coordinates (standard but potentially lossy)

The DPF exporter uses option 1 with `grid_type = "uniform"` as a custom extension.

### How do I handle time-dependent boundary conditions?

Use the `mask` and `values` datasets in the `/boundary_conditions/{boundary_name}/` subgroups. These can vary over time by including time as a leading dimension:
```
mask: shape (n_steps, nx, ny, nz), dtype bool
values: shape (n_steps, nx, ny, nz), dtype float32
```

However, this is an **extension** to the standard format and may not be supported by all loaders.

### Can I include multiple trajectories from different parameter sweeps?

Yes. Increment `@n_trajectories` and stack trajectories along the first axis. However, all trajectories must share the same grid shape and timestep count. For varying-length trajectories, create separate files.

### How do I convert DPF output to Well format?

Use the `WellExporter` class from `dpf.ai.well_exporter`:

```python
from dpf.ai.well_exporter import WellExporter

exporter = WellExporter(
    output_path="output.h5",
    grid_shape=(64, 1, 64),  # nx, ny, nz (ny=1 for cylindrical)
    dx=1e-4,
    dz=1e-4,
    geometry="cylindrical",
    sim_params={"V0": 50000.0}
)

# Load from existing DPF HDF5
exporter.add_from_dpf_hdf5("diagnostics.h5")

# Finalize and write
exporter.finalize(n_trajectories=1)
```

Then validate:
```python
from dpf.ai.dataset_validator import DatasetValidator

validator = DatasetValidator()
result = validator.validate_file("output.h5")
print(result.valid, result.errors)
```
