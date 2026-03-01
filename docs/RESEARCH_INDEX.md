# DPF-Unified Research Document Index

Organized by field of study. Use this as the master reference for locating documentation, research findings, and knowledge sources across the project.

**Last updated**: 2026-03-01

---

## 1. Dense Plasma Focus Physics & Device Engineering

Core DPF physics: snowplow dynamics, pinch formation, neutron yield, circuit coupling, device scaling.

| Document | Location | Summary |
|----------|----------|---------|
| PF-1000 Experimental Validation | `docs/PF1000_EXPERIMENTAL_VALIDATION.md` | First I(t) waveform comparison vs Scholz (2006), fc=0.650, fm=0.178, NRMSE=0.192 |
| Phase Z Physics Assessment | `docs/phase_z_physics_assessment.md` | Z.1-Z.3 gaps (neutron yield, calibration, B-field init), score trajectory toward 7.0 |
| Phase 1 Assessment Report | `docs/PHASE1_ASSESSMENT_REPORT.md` | 17-agent hyper-critical audit, downgraded 8.9→3.0/10, 7 CRITICAL deficiencies |
| MJOLNIR DPF (Goyon 2025) | `memory/research_papers.md` §HIGH #1 | 2 MJ LLNL device, 4.5 MA, I_stag controls pinch, v_imp/T_st/tau_exp formulas |
| PFZ-200 Czech DPF (Novotny 2026) | `memory/research_papers.md` §HIGHEST | 3 kJ, 200+ kA, Y_n~1e8, need Ref. [25] for circuit params |
| Poloidal B_z Dynamo (Auluck 2024) | `memory/research_papers.md` §MED-HIGH #4 | f_c < 1 encodes azimuthal current loss to B_z generation |
| Collisional/Collisionless Shock (Kindi 2026) | `memory/research_papers.md` §MED-HIGH #5 | ND parameter for MHD validity, Daligault MFP formula |
| MA-Class Device Comparison | `memory/research_papers.md` §table | LANL DPF-6, Poseidon, Verus, PF-1000, Gemini, MJOLNIR |
| Phase Z Physics (memory) | `memory/phase_z_physics.md` | Z.1-Z.3 implementation details, calibration anomaly (fc=0.5, fm=0.95) |
| Bug Tracker | `memory/bugs.md` | Active/fixed DPF physics bugs (D1 double circuit step, frozen L_plasma, etc.) |

**Key references**: Lee & Saw (2008, 2014), Scholz et al. (2006), Lee & Serban (1996), Bernard & Sadowski (2006), Goyon et al. (2025), Novotny et al. (2026), Auluck (2024)

---

## 2. Computational MHD & Numerical Methods

Riemann solvers, reconstruction schemes, time integration, constrained transport, coordinate systems.

| Document | Location | Summary |
|----------|----------|---------|
| CESE MHD Feasibility | `docs/research_cese_mhd_feasibility.md` | CESE method analysis: 4th-order extension, cylindrical geometry, resistive MHD, prototype recommendation |
| Bilyeu CESE & QSSS Research | `docs/research_bilyeu_ol2_qsss.md` | O(L^2) CESE method (Bilyeu PhD 2014), 4x resolution at 5% cost, no Riemann solver |
| AthenaK Research | `docs/ATHENAK_RESEARCH.md` | Kokkos-based MHD, Serial/OpenMP builds, VTK format, runtime physics config |
| Athena++ Build Guide | `docs/ATHENA_BUILD.md` | Build instructions, pybind11, verification binaries, nghost/xorder constraints |
| Architecture | `docs/ARCHITECTURE.md` | Tri-engine data flow, Metal GPU stack, iterative accuracy workflow |
| Metal GPU Lessons (memory) | `memory/metal.md` | HLLD NaN fix, WENO-Z FV vs FD, SSP-RK3, float64 mode, CT MPS requirement |
| Magneto-RT Instability (Bian 2026) | `memory/research_papers.md` §HIGH #2 | MRT growth rate with B-field, critical B formula, pinch survival mechanism |
| Phase History S-AB (memory) | `memory/phases.md` | All phase implementations including solver improvements |

**Key references**: Miyoshi & Kusano (2005) HLLD, Borges et al. (2008) WENO-Z, Shu-Osher (1988) SSP-RK3, Jiang & Shu (1996) WENO-JS, Bilyeu et al. (2014) CESE, Jiang & Zhang (2025) CESE-MHD

---

## 3. Calibration, V&V & Experimental Validation

Lee model calibration, ASME V&V 20, cross-device prediction, uncertainty quantification, timing/amplitude decomposition.

| Document | Location | Summary |
|----------|----------|---------|
| PF-1000 Experimental Validation | `docs/PF1000_EXPERIMENTAL_VALIDATION.md` | First validation: 26-point NRMSE=0.192, fc/fm in published ranges |
| Verification & UAT Plan | `docs/VERIFICATION_AND_UAT_PLAN.md` | 5-tier verification pyramid, 2 UAT personas, physics test catalog |
| UAT-A (PhD Physicist) | `docs/UAT_A_ENHANCED.md` | Dr. Vasquez persona, 8 scenarios: RLC, I(t), mesh convergence, energy, scaling |
| UAT-B (M.S. Student) | `docs/UAT_B_ENHANCED.md` | Alex Chen persona, 14 scenarios: learnability, tooltips, error handling, chat |
| PhD Debate History (memory) | `memory/debates.md` | 42 debates, score 4.5→6.5/10, ASME V&V 20, fm non-physicality, fc^2/fm degeneracy |

**PhD Debate Verdicts by V&V Topic** (in `docs/`):

| Debate | Score | Topic |
|--------|-------|-------|
| #10 | 6.2 | PF-1000 calibration, I(t) vs Scholz 2006 |
| #11 | 6.3 | Cross-verification, crowbar model, Pease-Braginskii gap |
| #18 | 6.6 | f_mr pipeline fix (zero physics impact), 21% timing error |
| #20 | 6.2 | D1/D2 calibration, cross-device, reflected shock |
| #26 | 6.4 | Blind NX2 prediction, fc^2/fm degeneracy |
| #27 | 6.5 | Three-device cross-prediction, NX2/UNU data corrections |
| #28 | 6.5 | Timing-based validation, ASME V&V 20 FAIL |
| #29 | 6.7 | Peak-finder fix, ASME correction, L_p/L0 diagnostic |
| #30 | 6.8 | L_p/L0 diagnostic, PF-1000 16 kV blind prediction |
| #35 | 6.8 | ASME delta_model, bare RLC dimension bug, POSEIDON geometry |
| #36 | 6.5 | POSEIDON-60kV: NRMSE fitted 0.079, blind 0.250, ASME ratio 2.22 |
| #37-38 | 6.5 | Bootstrap CI, Bennett equilibrium, non-tautological checks |
| #39 | 6.5 | Circuit-only calibration, NRMSE timing/amplitude decomposition |
| #40 | 6.5 | 3-parameter liftoff delay, fc bound asymmetry confound |
| #41 | 6.5 | Constrained-fc liftoff, 28% NRMSE reduction, fm non-physical |
| #42 | 6.5 | fm-constrained liftoff, NRMSE robust, double boundary-trapping |

**Key references**: ASME V&V 20-2009, Lee & Saw (2014) published ranges, Scholz et al. (2006), Sadowski et al. (2004)

---

## 4. AI/ML Surrogate Modeling & Data Infrastructure

WALRUS integration, Well HDF5 format, surrogate inference, inverse design, MLX acceleration.

| Document | Location | Summary |
|----------|----------|---------|
| Well Format Specification | `docs/WELL_FORMAT_SPECIFICATION.md` | HDF5 schema: root attrs, /dimensions/, /boundary_conditions/, /t0-t2_fields/, axis ordering |
| WALRUS Integration (memory) | `memory/walrus.md` | 1.3B model, delta prediction, RevIN normalization, all 12 DPF modules working |
| WALRUS Improvements (memory) | `memory/walrus_improvements.md` | Phase Z changes, 12 AI modules (4200 LOC), environment status, technical debt |
| ML Plasma Acceleration (Powis 2026) | `memory/research_papers.md` §HIGH #3 | CNN IC generator, 17.1x speedup, 250 training sims, A100 |

**PhD Debate Verdicts touching AI/ML**:

| Debate | Score | Topic |
|--------|-------|-------|
| #12 | 6.4 | Metal GPU vs WALRUS surrogate vs Athena++ routing strategy |
| #15 | 6.7 | Architecture migration: Metal→MLX, WALRUS→FNO, Athena++ primary |

**Key references**: PolymathicAI WALRUS (github), The Well dataset format, Hydra config system, torch==2.5.1

---

## 5. Radiation & Material Physics

Bremsstrahlung, line radiation, collisional-radiative models, electrode ablation, Nernst effect.

| Document | Location | Summary |
|----------|----------|---------|
| Lederman NLR Panel Study | `docs/LEDERMAN_NLR_PANEL_STUDY.md` | Non-Linear Repartitioning for Cu electrode CR, tabulated Cu cooling recommendation |
| Lederman Implicit Schemes | `memory/research_papers.md` §MED #7 | ReLU positivity, O(L^2) vs O(L^3) implicit kinetics, temporal multiscale |

**Key references**: Lederman & Bilyeu (2024, NMPDE), Post et al. (1977), Griem (1964), Summers (2004) ADAS

---

## 6. Software Architecture & Project Management

Tri-engine design, phase methodology, testing strategy, CI/CD, developer workflow.

| Document | Location | Summary |
|----------|----------|---------|
| Forward Plan | `docs/PLAN.md` | Completed phases A-P, future steps 3-7, accuracy milestones to 9.5/10 |
| Architecture | `docs/ARCHITECTURE.md` | Tri-engine data flow, AI layer, iterative accuracy cycle |
| Usage Guide | `docs/USAGE.md` | Quick-start examples, backend selection, config reference |
| Developer Workflow | `docs/WORKFLOW.md` | Phase methodology, test strategy, CI gate (>=745), agent assignments |
| TODO Audit | `docs/todo_audit.md` | 3 CRITICAL (resolved), 9 MEDIUM, 5 LOW placeholder/stubs |
| Phase 1 Agent Dialogue | `docs/phase1_agent_dialogue.md` | 17-agent cross-domain audit log, critical gap discovery |
| Slow Test Optimization | `docs/slow_test_optimization_research.md` | 139 slow tests, bottleneck analysis, pytest-xdist/MPS recommendations |
| Workflow Efficiency (memory) | `memory/workflow.md` | Parallelism tips, lint-first, session memory, debate batching |
| Performance (memory) | `memory/performance.md` | Numba JIT, kernel panic lesson, Metal grid thresholds |
| Coding Patterns (memory) | `memory/patterns.md` | Monkeypatching, Pydantic v2, FastAPI, snowplow defaults |
| TODOs (memory) | `memory/todos.md` | Placeholder inventory, completion tracking |
| Fix Agent Post-mortem (memory) | `memory/fix_agent_deployment.md` | 11 regressions from bulk fix, lesson: always full test suite |

---

## 7. PhD Assessment & Debate History

Score progression, panel consensus, concessions, path to 7.0.

| Document | Location | Summary |
|----------|----------|---------|
| Debate Scores (memory) | `memory/debates.md` | Full 42-debate history, score 4.5→6.5/10, concession tracking |

**All PhD Debate Verdicts** (`docs/PHD_DEBATE_*_VERDICT.md`):

| Era | Debates | Score Range | Primary Focus |
|-----|---------|-------------|---------------|
| Foundation | R, 3 | 4.5-5.0 | Initial audit, 7 CRITICAL deficiencies |
| Physics Build | 4-7 | 5.2-6.5 | Snowplow, validation fixes, calibration framework |
| Bug Discovery | 8-9 | 5.8-6.1 | D1/D2 bugs, Bennett equilibrium, strike-team fixes |
| Calibration | 10-11 | 6.2-6.3 | PF-1000 I(t), cross-verification, crowbar model |
| Architecture | 12, 15 | 6.4-6.7 | Metal vs WALRUS routing, migration proposals |
| MHD Accuracy | 13-14 | 6.5-6.7 | HLLD momentum fix, Sod/Brio-Wu/Sedov/linear wave |
| Device Validation | 16-17 | 6.8-6.9 | PF-1000 I(t) tests, Tier 3 Metal+circuit |
| Calibration Refinement | 18-21 | 6.2-6.6 | f_mr fix, pcf, D1/D2, preset completeness |
| Solver Robustness | 22-24 | 6.3 | Metal NaN fix, grid convergence, shock convergence |
| Cross-Device | 25-27 | 6.4-6.5 | Metal I(t), blind NX2, three-device prediction |
| ASME V&V 20 | 28-31 | 6.5-6.8 | Timing validation, peak-finder, L_p/L0, crowbar |
| Advanced Calibration | 35-42 | 6.5-6.8 | POSEIDON, bootstrap, liftoff delay, fm constraints |

**Key finding**: Score has plateaued at 6.5/10 since Debate #36. Path to 7.0 requires: third independent waveform (+0.1-0.2), physical liftoff model (+0.05-0.10), ASME Section 5.3 compliance (+0.05).

---

## 8. Pulsed Power & Hardware Engineering

Capacitor banks, circuit modeling, parasitic inductance, jitter.

| Document | Location | Summary |
|----------|----------|---------|
| Pulsed Power Reliability (Zhao 2026) | `memory/research_papers.md` §MED #6 | Capacitor lifetime model, jitter in multi-module, ESL budget |

**Key references**: Eyring-Arrhenius lifetime model

---

## Cross-Reference: Memory Files → Fields

| Memory File | Primary Field(s) |
|-------------|-----------------|
| `session.md` | Project Management |
| `bugs.md` | DPF Physics, Software Quality |
| `patterns.md` | Software Architecture |
| `phases.md` | All fields (phase history) |
| `metal.md` | Computational MHD, Numerical Methods |
| `walrus.md` | AI/ML Surrogate Modeling |
| `debates.md` | PhD Assessment, V&V Methodology |
| `bilyeu_research.md` | Numerical Methods (CESE) |
| `phase_z_physics.md` | DPF Physics, Calibration |
| `walrus_improvements.md` | AI/ML, Software Architecture |
| `todos.md` | Software Quality |
| `performance.md` | Software Architecture, Hardware |
| `workflow.md` | Project Management |
| `research_papers.md` | DPF Physics, Numerical Methods, AI/ML |
| `fix_agent_deployment.md` | Software Quality |
