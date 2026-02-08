# Athena++ Build Instructions for DPF Unified

This document provides detailed instructions for building Athena++ (Princeton's C++ MHD code) as a high-performance backend for the Dense Plasma Focus (DPF) simulator.

## Prerequisites

### macOS (Apple Silicon)

- **Operating System**: macOS 12+ (Monterey or later) on Apple Silicon (M1/M2/M3)
- **Xcode Command Line Tools**: Required for clang++
  ```bash
  xcode-select --install
  ```
- **Python**: 3.11 or later
  ```bash
  python3 --version  # Should be 3.11+
  ```
- **HDF5**: Install via Homebrew
  ```bash
  brew install hdf5
  # Verify h5cc is in PATH
  which h5cc
  h5cc -showconfig
  ```
- **pybind11**: Installed via pip (for Python bindings)
  ```bash
  pip install pybind11
  ```

## Athena++ Submodule Setup

The Athena++ code is included as a git submodule in `external/athena/`. Initialize it:

```bash
cd /Users/anthonyzamora/dpf-unified
git submodule update --init --recursive
```

This will clone the Athena++ repository into `external/athena/`.

## Building the Main DPF Binary

The primary Athena++ binary uses the `magnoh` problem generator (cylindrical z-pinch reference problem) with MHD and the HLLD Riemann solver.

```bash
cd external/athena

python configure.py \
  --prob=magnoh \
  --coord=cylindrical \
  -b \
  --flux=hlld \
  --cxx=clang++-apple \
  -hdf5

make clean && make -j8
```

The compiled binary will be located at `bin/athena`.

### Configuration Flags Explained

- `--prob=magnoh`: Z-pinch reference problem generator (see `src/pgen/magnoh.cpp`)
- `--coord=cylindrical`: Cylindrical coordinate system (r, φ, z)
- `-b`: Enable magnetic fields (MHD)
- `--flux=hlld`: HLLD Riemann solver (Harten-Lax-van Leer-Discontinuities, best for MHD)
- `--cxx=clang++-apple`: Use Apple's clang++ compiler
- `-hdf5`: Enable HDF5 output format

### OpenMP Warning

**Do NOT use `-omp` on macOS unless you have explicitly installed OpenMP via Homebrew:**

```bash
brew install libomp
```

Without `libomp`, the `-omp` flag will cause compilation errors (`omp.h not found`). For most DPF use cases, the pybind11 wrapper does not benefit from OpenMP, so it is omitted.

## Building Verification Binaries

For verification and validation (V&V), build additional binaries for standard MHD test problems.

### Sod Shock Tube (Hydro-Only)

```bash
cd external/athena

python configure.py \
  --prob=shock_tube \
  --flux=hllc \
  --cxx=clang++-apple \
  -hdf5

make clean && make -j8
cp bin/athena bin/athena_sod
```

- **Problem**: Sod shock tube (1D Riemann problem)
- **Flux**: HLLC (for pure hydrodynamics)
- **Output**: `bin/athena_sod`

### Brio-Wu MHD Shock Tube

```bash
cd external/athena

python configure.py \
  --prob=shock_tube \
  -b \
  --flux=hlld \
  --cxx=clang++-apple \
  -hdf5

make clean && make -j8
cp bin/athena bin/athena_briowu
```

- **Problem**: Brio-Wu MHD shock tube
- **Flux**: HLLD (for MHD)
- **Output**: `bin/athena_briowu`

These binaries are invoked as subprocesses by the test suite (see `tests/test_phase_f_verification.py`).

## Building the pybind11 Extension

The pybind11 wrapper (`_athena_core`) allows Python code to directly link against Athena++ C++ classes and drive the simulation loop from Python.

### Build Steps

```bash
cd src/dpf/athena_wrapper/cpp
mkdir -p build && cd build

cmake .. -DATHENA_ROOT=../../../../../external/athena

make -j8
```

### Output

The compiled extension module will be:
```
_athena_core.cpython-311-darwin.so
```

This file is automatically placed in `src/dpf/athena_wrapper/` and can be imported from Python:

```python
from dpf.athena_wrapper import _athena_core
```

### CMake Configuration

The `CMakeLists.txt` in `src/dpf/athena_wrapper/cpp/` expects:
- `ATHENA_ROOT`: Path to the Athena++ repository (default: `../../../../../external/athena`)
- Python 3.11+ with pybind11 installed
- The same HDF5 library used to build Athena++

## Important Notes

### 1. Ghost Zones and Reconstruction Order

By default, Athena++ is built with `nghost=2`, which **only supports spatial reconstruction up to 2nd order (PLM)**. If you need higher-order schemes:

- **PPM (3rd order)**: Rebuild with `-nghost=3`
- **WENO5 (5th order)**: Rebuild with `-nghost=3`

Example:
```bash
python configure.py --prob=magnoh --coord=cylindrical -b --flux=hlld \
  --cxx=clang++-apple -hdf5 -nghost=3
```

### 2. Athena++ Global State

Athena++ uses global state internally (e.g., `Globals::my_rank`, static mesh pointers). This means:

- **Cannot create multiple Athena++ instances in the same Python process**
- Attempting to re-initialize `Mesh` or `ParameterInput` will cause segmentation faults
- For multi-run workflows, use separate subprocesses or restart the Python interpreter

### 3. Linked-Mode Initialization

The pybind11 wrapper (`AthenaPPSolver`) uses `init_from_string()` to initialize Athena++ from a dynamically generated athinput text string. This allows:

- Full Python control over simulation parameters
- Integration with Pydantic configuration (`FluidConfig`, `CircuitConfig`)
- No need for external `.athinput` files (though they can still be used for debugging)

### 4. Verification Binaries as Subprocesses

Verification tests (Sod, Brio-Wu, etc.) are run as **subprocesses** using `subprocess.run()`, not via the pybind11 wrapper. This avoids global state conflicts and allows parallel test execution.

## Verification

After building, verify the installation:

### 1. Check pybind11 Wrapper

```bash
python -c "from dpf.athena_wrapper import is_available; print(is_available())"
```

Expected output: `True`

### 2. Run Athena++ Verification Tests

```bash
# Fast verification tests (Sod, Brio-Wu, diffusion)
pytest tests/test_phase_f_verification.py -v -m "not slow"

# Dual-engine tests (Python vs Athena++)
pytest tests/test_dual_engine.py -v
```

### 3. Run a Simple Simulation

```python
from dpf import DPFConfig, DPFEngine

config = DPFConfig(
    fluid={"backend": "athena", "nx": 64, "ny": 32, "nz": 32},
    circuit={"capacitance": 30e-6, "V0": 30e3},
    time={"t_end": 1e-6, "dt": 1e-9}
)

engine = DPFEngine(config)
engine.run()

print(f"Final time: {engine.get_state()['time']:.2e} s")
```

## Troubleshooting

### "omp.h not found" Error

**Cause**: The `-omp` flag was used, but OpenMP is not installed.

**Solution**:
1. Remove `-omp` from the configure command (recommended for DPF)
2. OR install OpenMP: `brew install libomp`

### Segmentation Fault on Second Initialization

**Cause**: Athena++ global state prevents multiple initializations in the same process.

**Solution**:
- Restart the Python interpreter between runs
- Use `subprocess.run()` to launch Athena++ in a separate process
- Avoid creating multiple `AthenaPPSolver` instances in the same script

### "magnoh: missing required parameter"

**Cause**: The `magnoh` problem generator expects specific parameters in the athinput file.

**Required parameters** (in `<problem>` block):
- `alpha`: Density profile exponent
- `beta`: Plasma beta (thermal/magnetic pressure ratio)
- `pcoeff`: Pressure coefficient
- `d`: Initial density
- `vr`: Initial radial velocity
- `bphi`: Initial azimuthal magnetic field
- `bz`: Initial axial magnetic field

**Solution**: Ensure your `FluidConfig` or athinput file includes all required parameters. See `external/athinput/athinput.dpf_zpinch` for reference.

### HDF5 Not Found

**Cause**: HDF5 library is not installed or not in PATH.

**Solution**:
```bash
brew install hdf5
export PATH="/opt/homebrew/bin:$PATH"  # Ensure h5cc is found
h5cc -showconfig  # Verify installation
```

Then rebuild Athena++:
```bash
cd external/athena
make clean && make -j8
```

### CMake Cannot Find Athena++

**Cause**: The `ATHENA_ROOT` path in CMake is incorrect.

**Solution**: Explicitly set `ATHENA_ROOT`:
```bash
cd src/dpf/athena_wrapper/cpp/build
cmake .. -DATHENA_ROOT=$(pwd)/../../../../../external/athena
```

### pybind11 Import Error

**Cause**: The `_athena_core.so` module is not in the Python path.

**Solution**: Install the DPF package in editable mode:
```bash
cd /Users/anthonyzamora/dpf-unified
pip install -e ".[dev,athena]"
```

This ensures `src/dpf/athena_wrapper/` is on the Python path.

## Build Variants

### Minimal Build (No HDF5, Cartesian, Hydro-Only)

For testing or development without full MHD:

```bash
python configure.py --prob=shock_tube --flux=hllc --cxx=clang++-apple
make clean && make -j8
```

### Full-Featured Build (MHD, Cylindrical, HDF5, 3rd-Order)

For production DPF runs:

```bash
python configure.py \
  --prob=magnoh \
  --coord=cylindrical \
  -b \
  --flux=hlld \
  --cxx=clang++-apple \
  -hdf5 \
  -nghost=3

make clean && make -j8
```

## Next Steps

After successfully building Athena++:

1. **Run the test suite**: `pytest tests/ -v -m "not slow"`
2. **Review example configs**: See `external/athinput/` for reference input files
3. **Explore DPF presets**: `src/dpf/presets.py` contains device-specific configurations
4. **Read the forward plan**: `docs/PLAN.md` outlines Phase G (Athena++ DPF physics)

## References

- **Athena++ Documentation**: [https://github.com/PrincetonUniversity/athena](https://github.com/PrincetonUniversity/athena)
- **DPF Project Plan**: `docs/PLAN.md`
- **Athena++ Wrapper Source**: `src/dpf/athena_wrapper/`
- **pybind11 Documentation**: [https://pybind11.readthedocs.io](https://pybind11.readthedocs.io)
