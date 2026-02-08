# AthenaK Research Spike — Phase J.1.a Findings

## Overview

AthenaK is the Kokkos-based performance-portable rewrite of Athena++, developed at
IAS (Princeton). It replaces the original Athena++ build system with CMake + Kokkos,
enabling execution on CPUs (Serial, OpenMP) and GPUs (CUDA, HIP, SYCL).

Repository: https://github.com/IAS-Astrophysics/athenak

## Build Results (M3 Pro, macOS 15.3)

### Serial Build — SUCCESS
```
Compiler: Apple clang 17.0.0 (Xcode)
Kokkos: 4.4.0, Serial backend, ARMV81 architecture
Build time: ~2 minutes
Binary size: ~8 MB
Command:
  cmake -B build_serial -D Kokkos_ARCH_ARMV81=On -D PROBLEM=blast
  cmake --build build_serial -j8
```

### OpenMP Build — SUCCESS
```
Compiler: Homebrew LLVM clang 19.1.7 (required for OpenMP)
Kokkos: 4.4.0, OpenMP backend, ARMV81 architecture
Key fix: --sysroot=$(xcrun --show-sdk-path) needed to resolve header conflicts
Command:
  cmake -B build_omp \
    -D CMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
    -D CMAKE_CXX_FLAGS="--sysroot=$(xcrun --show-sdk-path)" \
    -D Kokkos_ARCH_ARMV81=On -D Kokkos_ENABLE_OPENMP=On -D PROBLEM=blast
  cmake --build build_omp -j8
```

### GPU Build — NOT POSSIBLE
Kokkos does NOT have an Apple Metal GPU backend. The only GPU backends are CUDA,
HIP (AMD), and SYCL (Intel). On M3 Pro, AthenaK runs on CPU only via Serial or
OpenMP (up to 11 performance cores).

## Performance (MHD Blast Wave 200x200)

| Backend | Zone-cycles/sec | Wall time | Speedup |
|---------|----------------|-----------|---------|
| Serial  | 2.59M          | 8.57s     | 1.0x    |
| OpenMP 8T | 4.24M        | 5.24s     | 1.63x   |

Note: 200x200 (40K cells) is too small for good OpenMP scaling. Larger 3D grids
(e.g., 128³ = 2M cells) would show better parallel efficiency.

## Key Differences from Athena++

| Feature | Athena++ | AthenaK |
|---------|----------|---------|
| Build system | Python configure.py + Make | CMake + Kokkos |
| Physics params | Compile-time | Runtime (input file) |
| Coordinates | Native cylindrical/spherical | Cartesian only (DynGR for curvilinear) |
| Output format | HDF5 (.athdf) | VTK (single-task) or binary (MPI) |
| Problem generators | Separate builds per pgen | Built-in dispatch + custom `-D PROBLEM=` |
| Ghost zones | Compile-time (nghost) | Runtime (`<mesh>/nghost`) |
| Reconstruction | Compile-time (xorder) | Runtime (`<mhd>/reconstruct`) |
| Riemann solver | Compile-time (flux) | Runtime (`<mhd>/rsolver`) |

## VTK Output Format

AthenaK produces standard VTK legacy format (Version 2.0):
- Binary, big-endian float32
- STRUCTURED_POINTS dataset
- Variables: dens, velx, vely, velz, eint, bcc1, bcc2, bcc3
- Dimensions are CELL+1 in each direction (e.g., 64 cells → DIMENSIONS 65)
- Variable `mhd_w_bcc` outputs primitive variables + cell-centered B fields
- Output goes to `vtk/` subdirectory

### Python VTK Reader

Binary data is read as big-endian float32 (`>f4` in NumPy):
```python
import numpy as np
data = np.frombuffer(raw_bytes, dtype='>f4')
```

## Problem Generator Architecture

AthenaK has two modes:
1. **Built-in pgens** — selected via `pgen_name` in `<problem>` block:
   advection, cpaw, linear_wave, shock_tube, orszag_tang, diffusion, etc.
   These all share a single binary (no recompilation needed).

2. **Custom pgens** — selected via `-D PROBLEM=name` at CMake time:
   blast, field_loop, kh, shu_osher, current_sheet, turb, etc.
   Each requires a separate build.

For DPF integration, we should build with a custom problem generator that supports
our z-pinch physics, but can also use built-in pgens for V&V tests.

## Resistive MHD Support

AthenaK supports Ohmic resistivity via runtime parameter:
```
<mhd>
ohmic_resistivity = 1.0e-3
```

This is critical for DPF simulations (Spitzer + anomalous resistivity).

## Coordinate System Limitation

AthenaK does NOT have native cylindrical coordinates like Athena++. It uses a
Cartesian mesh with optional coordinate transformations via DynGR (for general
relativistic problems). For DPF cylindrical (r,z) geometry, options are:

1. **3D Cartesian** — Run full 3D and extract axisymmetric data
2. **Future custom coordinate support** — Would require AthenaK source modification
3. **Accept Cartesian** — Use Cartesian geometry for initial integration

Recommendation: Start with Cartesian 2D/3D for stock problems. Cylindrical
support can be added later or via custom source terms.

## Variable Mapping (AthenaK → DPF)

| AthenaK VTK | DPF state key | Notes |
|-------------|---------------|-------|
| dens | rho | Direct mapping |
| velx, vely, velz | velocity[0:3] | Stack into (3, nx, ny, nz) |
| eint | pressure | p = (gamma-1) * rho * eint |
| bcc1, bcc2, bcc3 | B[0:3] | Stack into (3, nx, ny, nz) |
| — | Te | Compute from eint + EOS |
| — | Ti | = Te for single-temperature |
| — | psi | 0 (no divergence cleaning in VTK) |

## Submodule Setup

AthenaK includes Kokkos as a git submodule. After cloning, must run:
```
git submodule update --init --depth=1
```

## Recommendations for J.1.b-d

1. **Build script**: Detect compiler (Apple clang vs Homebrew LLVM), auto-select
   Serial or OpenMP, handle sysroot for OpenMP builds.
2. **Subprocess wrapper**: Use VTK output (single-task mode). Support both
   built-in and custom problem generators via input file `pgen_name`.
3. **Batch mode**: Run N timesteps per subprocess call to amortize process overhead.
   Use `nlim` to limit cycle count and `tlim` for time limit.
4. **Two binaries**: One generic (built-in pgens) + one with `-D PROBLEM=blast`
   (or future dpf_zpinch custom pgen).
