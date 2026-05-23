# DPF-Unified

Dense plasma focus simulation workbench. The project contains Lee/snowplow,
resistive-MHD, circuit-coupling, diagnostics, and source-gated validation
infrastructure, but it is not yet an end-to-end predictive DPF simulator.

MIT License. Python 3.10+. Runs on Apple Silicon (MLX Metal) or CPU.

## Release posture

Current release decision: **HONEST-BLOCKED / SOURCE-GATED PREVIEW**. See
[`docs/SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md`](docs/SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md).

The first-principles and full-3D surfaces remain engineering probe paths with
fail-closed acceptance metadata; engineering probe outputs are not validation evidence:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

No production first-principles, full-3D, or end-to-end predictive DPF acceptance
claim is authorized by the current evidence stack.

## What it solves

The resistive MHD system in conservative form:

```
dU/dt + dF(U)/dr + dG(U)/dz = S(U)

U = [rho, rho*v_r, rho*v_z, rho*v_theta, E, S*rho, B_r, B_z, B_theta, e_electron]
```

Coupled to an RLC circuit ODE:

```
L(t) * dI/dt + I * dL/dt + R(t) * I = V_cap(t)
dV_cap/dt = -I / C
```

where L(t) includes a plasma inductance L_p computed from the MHD density field via the Lee formula:

```
L_p = (mu_0 / 2*pi) * z_sheath * ln(b / r_eff)
```

with r_eff density-weighted from the MHD state.

## Numerical methods

| Component | Method | Reference |
|-----------|--------|-----------|
| Riemann solver | HLLS (entropy-based two-wave) | Popovas et al. 2025, arXiv:2211.02438 |
| Riemann solver | HLLD (four-wave, CPU fallback) | Miyoshi & Kusano 2005, JCP 208:315 |
| Reconstruction | WENO5-Z (5th order) | Borges et al. 2008, JCP 227:3191 |
| Reconstruction | PLM (2nd order, MC limiter) | van Leer 1977, JCP 23:276 |
| Reconstruction | PPM (3rd order) | Colella & Woodward 1984, JCP 54:174 |
| Time integration | SSP-RK3 (3rd order) | Shu & Osher 1988, JCP 77:439 |
| Time integration | SSP-RK2 (2nd order) | Shu & Osher 1988 |
| Divergence-free B | Dedner GLM | Mignone & Tzeferacos 2010, JCP 229:5896 |
| Divergence-free B | Constrained transport (CT) | Evans & Hawley 1988, ApJ 332:659 |
| Pressure recovery | Dual-energy (entropy tracer) | Dispatch HLLS, Popovas et al. 2025 |
| Resistive diffusion | Implicit Thomas solver (operator-split) | Standard tridiagonal |
| Circuit integration | Implicit midpoint (2nd order, stiff-stable) | Standard |

Float32 throughout on GPU. Entropy tracer avoids catastrophic cancellation in pressure recovery. WENO5-Z epsilon = 1e-6 (not 1e-36) to prevent float32 underflow in smoothness indicators.

## Solver backends

| Backend | Location | Method | Hardware |
|---------|----------|--------|----------|
| MLX | `src/dpf/metal/mlx_*.py` (29 files, 12,961 LOC) | Native Metal kernels via MLX | Apple Silicon GPU |
| Metal/PyTorch | `src/dpf/metal/metal_solver.py` | PyTorch MPS tensors | Apple Silicon GPU |
| Python | `src/dpf/fluid/mhd_solver.py` | NumPy + Numba | CPU |
| Athena++ | `src/dpf/athena_wrapper/` + `external/athena/` | C++ pybind11 linked | CPU (OpenMP) |
| AthenaK | `src/dpf/athenak_wrapper/` + `external/athenak/` | Kokkos C++ subprocess | CPU (Serial/OpenMP) |

Backend selection: `config.fluid.backend` in `src/dpf/engine/core.py`.

## Physics implemented

All locations are file paths within `src/dpf/`.

| Physics | Location | Status |
|---------|----------|--------|
| Resistive MHD (ideal gas, gamma=5/3) | `metal/mlx_kernels.py`, `fluid/mhd_solver.py` | Tested |
| Cylindrical geometry (r,z) with geometric source terms | `metal/mlx_kernels.py:cylindrical_source_mlx` | Tested |
| Cartesian 3D | `metal/mlx_grid.py:CartesianGrid` | Tested |
| Electrode ghost-cell BCs (B_theta = mu0*I/2*pi*r) | `metal/mlx_bc.py` | Tested |
| RLC circuit with crowbar | `circuit/rlc_solver.py` | Tested |
| Lee snowplow model (axial + radial + pinch phases) | `metal/mlx_snowplow.py`, `fluid/snowplow.py` | Tested |
| MHD-circuit coupling (density-weighted L_p) | `metal/mlx_coupling.py`, `metal/mlx_engine.py` | Tested |
| Spitzer resistivity | `metal/mlx_transport.py` | Tested |
| Braginskii anisotropic viscosity | `metal/mlx_viscosity.py` | Tested |
| Braginskii anisotropic conduction | `fluid/anisotropic_conduction.py` | Python engine only |
| Bremsstrahlung radiation (log-space, float32-safe) | `metal/mlx_sources.py` | Tested |
| Line radiation (piecewise power-law) | `metal/mlx_line_radiation.py` | Tested |
| Two-temperature (electron-ion equilibration) | `fluid/two_temperature.py` | Python engine only |
| Multi-species advection | `metal/mlx_species.py` | Tested |
| Hybrid PIC (Boris push, CIC deposition, Nanbu collisions) | `experimental/pic/hybrid.py` | Active, wired into engine |
| AMR (block-structured) | `metal/mlx_amr.py` | Framework only |

**Not implemented**: Hall MHD in MLX, tabulated EOS, radiation transport, MPI parallelism.

## Validation status

Scientific validation evidence is source-gated against the local
`KnowledgeReference/` corpus. Runtime tests, reconstructed traces, external
waveform archives, and engineering campaigns are not promoted to validation
evidence unless they pass the explicit source-authority gates in
`dpf.validation.quality_assessment`.

**Predictive-readiness gate**: blocked by default until five evidence tiers pass:
KR-verified circuit waveform, KR-backed snowplow phase/timing, MHD analytic
verification, same-scope spatial DPF density/B-field/temperature validation, and
neutron timing/spectrum/anisotropy validation.

**Circuit-level (Lee snowplow model)**: source-gated waveform comparison exists.
Under the current KR-only rule, only the standard PF-1000 Scholz waveform record
is validation-ready. POSEIDON-60kV and UNU-ICTP have KR-supported parameter
tables, but their current waveform arrays come from IPFS/external archive traces
and are blocked by default.

**MHD-level (spatially-resolved solver)**: Verified against standard test problems
(Sod, Brio-Wu). Not yet validated against spatially-resolved experimental data
(density profiles, temperature maps). See [docs/SCOPE.md](docs/SCOPE.md) for full
regime-of-validity analysis, [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md)
for which physics runs on which backend. See `CodexFindings.md` for the current
KR-only review and ratchet log.

### PF-1000 circuit source record

The PF-1000 record is the current tier-1 source-authority anchor. It is not, by
itself, evidence for predictive readiness of pinch structure or neutron yield.

| Parameter | Value | Source |
|-----------|-------|--------|
| C | 1.332 mF | Scholz 2006 |
| L0 | 25 nH | Akel et al. 2021 table record in `KnowledgeReference/` |
| R0 | 2.3 mOhm | Scholz 2006 bare-bank short-circuit record |
| V0 | 27 kV | IFPiLM operating condition |
| fill pressure | 3.5 Torr D2 | Scholz 2006 |
| waveform source status | KR-verified measured trace | `PF1000_DATA.waveform_kr_status` |
| validation helper | `circuit_validation_evidence_from_waveform()` | strict tier-1 evidence producer |

### Engineering test campaign

| Test | Shots | Result |
|------|-------|--------|
| Grid convergence (16x32, 32x64, 64x128) | 3 | I_peak identical (circuit-dominated, not MHD-driven) |
| Deterministic reproducibility (50 identical) | 50 | std = 2.2e-16 (machine epsilon) |
| Sobol sensitivity (500 random) | 500 | V0 dominates (eta^2 = 0.94); fc, fm < 0.02 |
| Cross-device (4 presets) | 6 | 2/4 pass tolerance |
| Endurance (1,467 consecutive) | 1,467 | 0 failures, 6.8s +/- 0.3s per shot |

These are engineering and regression tests. They are not scientific validation
evidence unless converted into KR-sourced evidence objects and accepted by the
predictive-readiness gate.

### Known limitations

1. **Grid convergence test is misleading.** The convergence study shows identical I_peak across all grids because the circuit + snowplow ODE drives the current, not the MHD solver. MHD grid convergence for density/B-field profiles has not been validated.

2. **fc/fm insensitivity.** Sobol analysis shows fc and fm have <2% effect on I_peak. This is expected: the snowplow model determines sheath dynamics, and I_peak is primarily set by V0 and the LC period.

3. **MHD-circuit coupling is recent.** The MHD solver's density-weighted L_p now feeds back into the circuit ODE (as of v1.5.0+), but the blend only activates when the MHD sheath structure is resolved. During axial rundown, the snowplow ODE drives the circuit. Full MHD-driven current prediction requires initialized sheath structure, which is in progress.

4. **PF-1000 is only tier-1 evidence.** A passing PF-1000 waveform comparison can
support circuit-level evidence only. It does not validate spatial MHD fields,
late pinch disruption, or neutron production.

5. **External waveform traces are blocked by default.** POSEIDON-60kV,
UNU-ICTP, PF-1000-Gribkov, reconstructed, and reference-only traces cannot
support circuit-validation claims unless their waveform source is ingested into
`KnowledgeReference/` and marked verified.

## MHD solver verification

The MLX solver independently produces:

- **B_theta propagation**: Electrode BC injects B_theta = mu0*I/(2*pi*r) at cathode ghost cells. Verified to propagate inward via Alfven waves at correct speed.
- **J x B compression**: Radial density compression from magnetic pinch force. 15% density variation after 50 steps at 500 kA.
- **Inward radial velocity**: -92 m/s from J x B, directed inward as expected.
- **Sod shock tube**: L1(rho) < 0.02 (right plateau, N=256) with HLLS+PLM on standard 1D problem.
- **Brio-Wu MHD shock**: Completes without NaN in float32 with dual-energy.

Test: `pytest tests/test_mlx_circuit_coupling.py -v` (10 tests proving B-field propagation, density compression, coupling interface).

## Code metrics

| Metric | Value |
|--------|-------|
| Production code | 77,197 LOC (src/dpf/) |
| Test code | 159,722 LOC (tests/) |
| Tests collected | 5,261 (5,143 enabled) |
| MLX solver | 29 files, 12,961 LOC |
| Commits | 724 |
| Releases | 10 (v1.0 through v1.5.0) |

## Installation

```bash
git clone https://github.com/longanisainhertaco/dpf-unified.git
cd dpf-unified
pip install -e ".[dev]"
```

For MLX GPU (Apple Silicon):
```bash
pip install -e ".[dev,metal]"
```

For Athena++ (requires C++ compiler):
```bash
cd external/athena
python configure.py --prob=magnoh --coord=cylindrical -b --flux=hlld
make -j8
cd ../..
pip install -e ".[dev,athena]"
```

## Usage

### CLI
```bash
dpf simulate config.json --steps=1000
dpf simulate config_cylindrical.json --steps=1000

# PF-1000/Akel first-principles-only engineering candidate
dpf first-principles --sim-time-us=0.2 --output results/first_principles_pf1000_akel_smoke.json
```

The `first-principles` command is locked to the PF-1000/Akel demonstrator and
uses the field-coupled `first_principles_mhd` path. Its output is an engineering
probe artifact with fail-closed readiness metadata; it is not scientific
validation evidence and does not accept the Akel draft waveform packet.

### Python API
```python
from dpf.metal.mlx_engine import run_mlx_discharge

# Lee model only (circuit + snowplow, ~1ms per shot)
result = run_mlx_discharge("pf1000", mode="lee")

# Full MHD (circuit + snowplow + MLX MHD solver, ~7 min per shot)
result = run_mlx_discharge("pf1000", mode="mhd", grid_shape=(32, 1, 64))

print(f"I_peak = {result['I_peak_MA']:.3f} MA at t = {result['t_peak_us']:.1f} us")
```

### Run tests
```bash
python3 -m pytest tests/ -x -q                    # non-slow (~4200 tests, ~3 min)
python3 -m pytest tests/ -x -q -m slow            # slow tests (~600, ~30 min)
python3 -m pytest tests/ -x -q -k mlx             # MLX GPU tests (~430)
python3 -m pytest tests/test_mlx_circuit_coupling.py -v  # MHD coupling proof
```

## Project structure

```
src/dpf/
  circuit/          RLC solver, back-EMF
  core/             Base classes (PlasmaSolverBase, CouplingState)
  engine/           Backend dispatch, circuit-MHD coupling
  fluid/            Python MHD solver, snowplow, EOS, transport
  metal/            MLX + PyTorch Metal solvers (29 MLX modules)
  experimental/     PIC hybrid solver
  radiation/        Line radiation, bremsstrahlung
  validation/       Calibration, experimental data
  diagnostics/      27 diagnostic modules
  presets.py        Device configurations (PF-1000, UNU-ICTP, etc.)
tests/              92 test files
docs/               Design docs, calibration records, V&V reports
external/           Athena++ and AthenaK git submodules
```

## References

- Lee, S. & Saw, S.H. (2014). Phys. Plasmas 21, 072501. (Lee model)
- Akel, M. et al. (2021). J. Fusion Energy 40, 7. (PF-1000 operating data)
- Miyoshi, T. & Kusano, K. (2005). JCP 208, 315. (HLLD Riemann solver)
- Borges, R. et al. (2008). JCP 227, 3191. (WENO-Z weights)
- Shu, C.-W. & Osher, S. (1988). JCP 77, 439. (SSP-RK time integration)
- Popovas, A. et al. (2025). arXiv:2211.02438. (DISPATCH HLLS entropy switch)
- Mignone, A. & Tzeferacos, P. (2010). JCP 229, 5896. (Dedner GLM divergence cleaning)
- Colella, P. & Woodward, P. (1984). JCP 54, 174. (PPM reconstruction)
- Evans, C.R. & Hawley, J.F. (1988). ApJ 332, 659. (Constrained transport)

## License

MIT. Copyright 2024-2026 Anthony Zamora.

## Development

This project was developed with Claude Code (Anthropic). The AI wrote code, tests, and documentation. The physics design, validation methodology, experimental data selection, and architectural decisions are by the author.

All claims in this README can be verified by running the referenced tests or reading the referenced source files.
