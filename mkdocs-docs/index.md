# DPF Unified

**A modern dense plasma focus (DPF) simulator** — tri-engine architecture with a Python (NumPy/Numba) fallback, Athena++ C++ primary backend (pybind11), and AthenaK Kokkos GPU-ready backend (subprocess), targeting high-fidelity multi-physics simulation of plasma focus devices on local hardware (Apple Silicon) and HPC clusters.

---

## Quick Links

- [Getting Started](getting-started.md) — install, first simulation, web UI
- [API Reference](api/index.md) — full module documentation
- [Examples](examples.md) — runnable code examples
- [Contributing](contributing.md) — development workflow

## Architecture

| Layer | Description | Status |
|-------|-------------|--------|
| **Python engine** | NumPy/Numba MHD solver (`src/dpf/fluid/`) | Production |
| **Athena++ engine** | Princeton C++ MHD (pybind11) | Production |
| **AthenaK engine** | Kokkos C++, GPU-ready (subprocess) | Production |
| **Metal GPU** | PyTorch MPS — WENO5-Z + HLLD + SSP-RK3 | Production |
| **AI surrogate** | WALRUS 1.3B IsotropicModel + RevIN | Phase I complete |

## Quick Start

```bash
pip install dpf-unified
dpf run --preset pf1000 --backend python
```

Or launch the web UI:

```bash
python3 app.py
# open http://localhost:7860
```
