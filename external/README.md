# External Dependencies

## Athena++ (git submodule)

**Athena++** is a production-grade astrophysical MHD code from Princeton University.
- **Repository**: https://github.com/PrincetonUniversity/athena
- **License**: BSD-3-Clause
- **Paper**: Stone et al., ApJS 249, 4 (2020)

### Why Athena++?

DPF Unified uses Athena++ as the primary MHD solver backend, providing:
- 10-100x speedup over the Python+Numba solver via C++, AMR, MPI, OpenMP
- Production-grade constrained transport (exact div(B) = 0)
- HLLD Riemann solver, PPM/WENO reconstruction
- Native cylindrical coordinate support
- `magnoh.cpp` z-pinch problem generator (direct DPF starting point)

### Submodule Management

```bash
# Initial clone (with submodules)
git clone --recurse-submodules https://github.com/your-repo/dpf-unified.git

# If already cloned without submodules
git submodule update --init --recursive

# Update to latest Athena++ commit
git submodule update --remote external/athena
```

### Building for DPF

```bash
cd external/athena
python configure.py --prob=magnoh --coord=cylindrical -b --flux=hlld \
    --cxx=clang++-apple -omp -hdf5
make -j8
```

### DPF Input Files

Custom Athena++ input files for DPF simulations are stored in `external/athinput/`.
