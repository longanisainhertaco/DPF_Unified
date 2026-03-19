# Changelog

All notable changes to DPF-Unified are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Open-source release infrastructure: LICENSE (MIT), CITATION.cff, CONTRIBUTING, CODE_OF_CONDUCT, AI disclosure, JOSS draft
- `V_max` diagnostic + betatron heating operator-split
- 272 WALRUS training trajectories (Campaign 2J)

### Changed
- HuggingFace Spaces deployment updated for v1.4.1 release

### Fixed
- HF Space description truncated to 60-char limit
- Matplotlib dependency, package `__init__`, HF tags, and validation bugs
- 4 UI bugs: NaN warning, config round-trip, preset validation, backend help text
- CI: excluded slow tests, added `pytest-xdist`, fixed 30-min timeout
- Athena++ submodule pinned to public commit; CI checkout made resilient

---

## [1.4.1] — 2026-03-16

### Added
- 3D filamentation diagnostic + azimuthal IC seeding (Campaign 2G)
- Lightweight beam-ion tracker with Boris push (Campaign 2I)
- QMF diagnostic + beam tracker wired into MHD results

### Fixed
- `grid_shape`/`dx`/`sim_time` missing from custom preset (root cause of 4 test failures)
- Field shapes test uses actual preset `grid_shape` instead of hardcoded `8x8x8`
- 5 pre-existing test failures resolved
- `custom` preset correctly skipped in backend validation and E2E engine creation tests

### Changed
- Post-processing diagnostics extracted from `run_mhd_simulation` into dedicated module

---

## [1.4.0] — 2026-03-16

### Added
- Reconnection, force-free equilibrium, and sub-grid turbulence wired into physics narrative and summary (Campaigns 2E–2H)
- QMF bremsstrahlung suppression for p-B11 analysis (Campaign 2E)
- Plasmoid detection + force-free equilibrium diagnostic (Campaign 2F)
- Sub-grid turbulence module + reconnection diagnostics (Campaign 2H)
- Time-resolved neutron yield tracker wired into hybrid MHD loop (Campaign 2D)
- Cartesian 3D electrode BC with azimuthal decomposition (Campaign 2C)
- CR ionization + implicit cooling in multi-shot physics (Campaign 2B)
- 5 dormant physics modules wired into web UI (Campaign 2A)
- Quality assessment table in physics narrative

### Changed
- Version bumped to v1.3.1 in scaling law predictions → included in this release

---

## [1.3.1] — 2026-03-16

### Added
- Automated simulation quality assessment with A–F grading
- Quality grade shown in simulation metrics banner
- Plasma regime classifier: Lundquist, Rm, beta, Knudsen, Hall parameters
- DPF scaling law diagnostics (Yn~I^4, Yn~I^3.3, Yn~E^2)
- Scaling law predictions included in result dict

---

## [1.3] — 2026-03-15

### Added
- Full 15-challenge physics wired into web UI frontend
- Phase 1–3 complete: velocity shear stabilization, Auluck poloidal B-field, mesh refinement, improved radiation
- Grid convergence study with Richardson extrapolation + GCI
- Automated validation matrix across all device presets
- Gas species CIV display; CIV breakdown mechanism shown in metrics banner
- PF-400J preset (CCHEN Chile portable DPF)
- Energy balance tracker for conservation verification
- V&V summary report generator
- Time-resolved neutron yield tracker for MHD simulations
- Reproducibility package: export/import/verify simulation results
- Line radiation cooling wired into Metal MHD loop
- Radiation balance section in physics narrative
- Cross-discipline literature database: 731 papers, 325 formulas, 26 gaps, 246 findings

### Changed
- 24-shot PF-1000 validation: mean error 1.27% (down from 24.7%)
- Major UI/UX overhaul from UAT feedback (Verus Research panel)
- PhD panel corrections applied (B+ → A- essay); all 8 panel issues closed
- POSEIDON published I_peak corrected to 4.6 MA; PF-400J uses damped peak (104 kA)
- Backend names simplified: Quick / Standard / Detailed / High Accuracy / 3D / Reference

### Fixed
- CIV mechanism requires sufficient B-field (v_ExB < c clamp)
- NIST-validated ionization potentials for CIV gas properties

---

## [1.2] — 2026-03-15

### Added
- `metal_3d` backend: 3D Cartesian MHD for filamentation studies (Challenge 8)
- Multi-shot DPF simulator for high repetition rate (Challenge 10)
- Plasmoid detection diagnostic (Challenge 14)
- Neutron yield scaling validation script (Challenge 13)
- p-B11 fusion cross section, tearing mode diagnostic, stochastic IC (Challenges 7, 9, 11)
- Synthetic interferometry diagnostic in MHD output (Challenge 15)

---

## [1.1.1] — 2026-03-15

### Fixed
- Crowbar time correctly passed to `RLCSolver` in all MHD paths

### Added
- Electrode B-field boundary condition in Metal GPU solver

---

## [1.1] — 2026-03-15

### Added
- Hall MHD in Python solver for whistler-speed reconnection (Frontier F)
- Line + recombination radiation for high-Z fills (Frontier D.2)
- m=0 density perturbation seeding in Python MHD IC (Frontier C.2)
- Coulomb collision operator for PIC Boris pusher (Frontier E.3)
- PIC kinetic current J_kin wired into MHD induction equation (Frontier E.2)
- Bennett equilibrium diagnostic in MHD output
- Full bidirectional MHD-circuit back-EMF coupling
- Bremsstrahlung radiation cooling in Python MHD loop (Frontier D)
- m=0 instability timing diagnostic from Goyon 2025 (Frontier C)
- D-D neutron yield computed from MHD state (Frontier B)
- HuggingFace Spaces deployment script and requirements.txt
- 35 tests for dormant hybrid PIC module; 26 MHD physics integration tests

### Fixed
- Electrode B-field BC activated in MHD backends

---

## [1.0] — 2026-03-14

### Added
- Full MHD physics stack: resistive MHD, Braginskii transport, Hall MHD, two-temperature model, Powell div-B, constrained transport
- Cylindrical and Cartesian Metal GPU solver (Apple Silicon, MLX)
- Snowplow Lee model with auto-calibration, anomalous resistivity, beam-target neutrons, post-pinch disruption
- 9 simulation backends: Python, Metal, AthenaK, Hybrid, multi-shot, and reference variants
- PF-1000, POSEIDON, UNU-ICTP, FAETON-I, MJOLNIR, NX2, PF-400J device presets
- 24-shot PF-1000 statistical validation (Akel 2021)
- Leave-one-out cross-validation + ASME V&V 20 uncertainty budget
- Gradio web UI with animated MHD playback, experimental overlay, and parameter sweep
- Babylon.js 3D plasma renderer with 12 toggleable physics layers, AAA visual quality
- WALRUS 1.3B surrogate model integration + 272 training trajectories
- AthenaK backend via subprocess + VTK I/O
- Standard MHD benchmarks: Sod, Brio-Wu, Sedov, Orszag-Tang, linear wave convergence
- Pease-Braginskii current diagnostic, Bennett equilibrium, scaling law analysis
- Sub-cycled resistive diffusion for CFL stability
- 3 critical cylindrical MHD physics bug fixes (t_peak within 1.5% of published)
- 259 Metal GPU tests passing; ~1100+ total tests across framework
- Electron + React desktop GUI (Engineering Mode dashboard)
- CI pipeline on GitHub Actions

---

## Link Diffs

[Unreleased]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.4.1...HEAD
[1.4.1]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.3...v1.3.1
[1.3]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.2...v1.3
[1.2]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.1.1...v1.2
[1.1.1]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.1...v1.1.1
[1.1]: https://github.com/longanisainhertaco/DPF_Unified/compare/v1.0...v1.1
[1.0]: https://github.com/longanisainhertaco/DPF_Unified/releases/tag/v1.0
