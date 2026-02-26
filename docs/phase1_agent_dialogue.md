# Phase 1 Assessment — Inter-Agent Dialogue Log

**Coordinator:** Agent 0 (Coordinator)
**Started:** 2026-02-14
**Principal Agents:** pulsed-power (1), plasma-physics (2), hpc-engineer (3), vv-engineer (4), godot-viz (5), architect (6), domain-science (7)
**Verification Agents:** circuit-theory-verifier (8), circuit-exp-verifier (9), mhd-theory-verifier (10), kinetic-verifier (11), numerics-fv-verifier (12), numerics-stability-verifier (13), diagnostics-verifier (14), neutron-rad-verifier (15), walrus-verifier (16)

---

## Round 1: Initial Cross-Domain Challenges

*Seeded by Coordinator at Round 1 start. Agents prompted to share findings and challenge each other.*

### Coordinator → pulsed-power
**Topic:** Circuit-Plasma Coupling Validation
**Message:** Share your circuit model findings with plasma-physics. Ask: do the circuit model limitations (simple RLC, no transmission line effects, no parasitic inductance modeling) affect MHD accuracy? Also respond to domain-science about whether the circuit model can reproduce real DPF current waveforms (derivative shape, peak timing, quarter-period match).

### Coordinator → plasma-physics
**Topic:** Numerical Method Feasibility
**Message:** Share your numerical method findings (WENO-Z, HLLD, SSP-RK3, reconstruction order, convergence rates) with hpc-engineer. Ask: are these methods computationally feasible at the resolutions needed for DPF pinch dynamics on Apple Silicon (M3 Pro, 36GB)? What grid sizes can we actually run in real-time or near-real-time?

### Coordinator → vv-engineer
**Topic:** Convergence & Validation Evidence Gaps
**Message:** Challenge plasma-physics on convergence evidence — are the measured convergence rates (order ~1.86) sufficient, or do we need to demonstrate higher order on smooth MHD problems? Also challenge domain-science on experimental validation gaps — what experimental data are we comparing against, and are there published DPF datasets we should be using?

### Coordinator → domain-science
**Topic:** Circuit Waveform Realism
**Message:** Challenge pulsed-power on whether the circuit model can reproduce real DPF current waveforms. Key questions: Does the RLC model capture the current derivative dI/dt correctly during the rundown phase? Can it reproduce the characteristic current dip at pinch time? Are there published experimental waveforms (Mather, Lee, Filippov types) we should benchmark against?

### Coordinator → architect
**Topic:** System Integration Requirements
**Message:** Ask ALL other agents about their API/interface requirements for the integrated system. Key questions: What data flows between circuit solver → MHD engine → diagnostics → visualization? What state dict format changes are needed? Are there latency or throughput requirements that constrain the architecture? How should the tri-engine backend selection (Python/Athena++/AthenaK) be exposed to the frontend?

### Coordinator → godot-viz
**Topic:** Data Pipeline Requirements
**Message:** Ask plasma-physics and hpc-engineer what data formats and update rates the visualization must handle. Key questions: What is the state dict size for typical grids (64³, 128³, 256³)? How frequently can the solver produce output frames? What fields are essential to visualize (rho, B, T, current density)? Is there a streaming protocol preference (WebSocket, SharedArrayBuffer, file-based)?

### Coordinator → hpc-engineer
**Topic:** Physics Feasibility on Apple Silicon
**Message:** Challenge plasma-physics on whether the requested physics extensions (Hall MHD, PIC coupling, resistive MHD with sub-cycling) are computationally feasible on Apple Silicon M3 Pro with 36GB. What grid sizes are practical? What are the memory bandwidth bottlenecks? Is Metal GPU actually faster than CPU for the stencil sizes used?

---

## Round 1 Responses

### godot-viz → Coordinator
**Topic:** Top 5 Findings & 3 Concerns from Visualization Audit

**Findings:**
1. **Zero Godot, zero 3D** — No .tscn, .gd, .gdshader, project.godot files. GUI is Electron + React + ECharts (2D charts only). No spatial visualization exists.
2. **Binary field pipeline is a dead end** — Server-side `encode_fields()` and WebSocket `field_header` protocol exist. But `websocket.ts:117` discards all binary data (`if (event.data instanceof Blob || ArrayBuffer) { return; }`). Pipeline terminates at a no-op.
3. **3 dead components** — `MultiTraceView`, `InductancePlot`, `TraceConfig` are fully written but never mounted.
4. **AI Co-Pilot solid but incomplete** — Sweep/inverse/chat work. Missing: confidence display, rollout viz, sim-vs-WALRUS comparison.
5. **2D dashboard well-built** — Clean architecture (Zustand, typed REST, WebSocket reconnect), good dark theme. Rating: 6/10 dashboard, 0/10 scientific viz.

**Concerns:**
1. **Architecture decision needed before 3D work**: Three.js (lowest cost, stays in React) vs Godot 4.x (best Vulkan perf, highest cost) vs WebGPU (future-proof, immature). Recommends **Three.js**.
2. **Data throughput requirements unknown** — needs answers from plasma-physics and hpc-engineer on snapshot sizes and update rates.
3. **No GUI tests** — 37 source files, zero tests. Refactoring is flying blind.

**Cross-domain questions raised:**
- → plasma-physics: How large are field snapshots? What update rate?
- → hpc-engineer: Can Metal GPU share buffers with renderer? Zero-copy path?
- → architect: Three.js vs Godot vs WebGPU — which fits the architecture?

### vv-engineer → plasma-physics
**Topic:** Convergence Evidence is Insufficient (4 Challenges)
1. **~1.86 convergence order is unjustified without MMS** — tests measure deviation from IC, not convergence to true solution. Zero MMS tests exist in the codebase.
2. **Temporal vs spatial order never isolated** — all convergence tests use fixed CFL, coupling dx and dt. Cannot separate spatial from temporal error contributions.
3. **Only 3 resolutions [16, 32, 64]** — gives one convergence rate estimate, likely pre-asymptotic at coarse end.
4. **Conservation verification gaps** — mass and momentum never tested; div(B) tolerance is 0.1 (10%), vs Athena++ machine precision ~1e-15.

### vv-engineer → domain-science
**Topic:** Validation Suite is Empty (5 Challenges)
1. **Validation suite tests scoring functions, never runs simulations** — `sim_summary` dicts are hand-crafted, not from actual solver output.
2. **Zero digitized experimental waveforms** in the repository — no experimental data to compare against.
3. **Circuit-only validation ≠ plasma validation** — damped sinusoid ≠ real DPF I(t) with current dip at pinch.
4. **"8.9/10 fidelity" grades algorithms, not physics** — no experimental comparison supports this claim.
5. **No DPF-specific physics validation** — no pinch dynamics, no sheath propagation, no neutron yield vs experiment.

### vv-engineer → Coordinator
**Topic:** Cross-Cutting Finding
**Key finding:** The single most damaging gap is **C1: No experimental validation**. The codebase has excellent code verification (solving equations right) but zero validation (solving the right equations). The validation suite plumbing exists but has never been connected to actual simulations + experimental data. This is the **#1 priority for the entire project**.

### plasma-physics → Coordinator
**Topic:** Audit Complete, Awaiting Challenges
**Message:** Task #2 complete. Available for follow-up, clarifications, deep-dives, or implementation tasks.

### vv-engineer → Coordinator
**Topic:** Executive Summary — Top 5 Findings & 3 V&V Gaps

**Top 5 Findings:**
1. ~1,520 test functions across 60 files — code verification is genuinely strong.
2. **ZERO experimental validation** — validation suite plumbing exists but tests use hand-crafted dicts, never actual solver output.
3. **ZERO MMS tests** — cornerstone of ASME V&V 20 Tier 1 is absent.
4. **CI never runs Metal/MPS tests** — 100+ GPU tests require Apple Silicon, never executed in GitHub Actions.
5. **"ASME V&V 20 Tiers 1-5" claim unsupported** — Tier 1 partial (no MMS), Tiers 2-5 not implemented.

**Top 3 V&V Gaps:**
1. No experimental I(t)/V(t) validation against any DPF device (fix: digitize 3 waveforms, run coupled sims).
2. No MMS tests for MHD solver (fix: ~200 LOC manufactured solution).
3. Metal/MPS tests excluded from CI (fix: add macOS Apple Silicon runner).

### plasma-physics → Coordinator
**Topic:** Top 5 Findings & 3 Concerns from Plasma Physics Audit

**Top 5 Findings:**
1. **(Critical) Python engine uses NON-CONSERVATIVE pressure evolution** — `mhd_solver.py:460-475` evolves dp/dt instead of total energy dE/dt. Violates Rankine-Hugoniot at shocks. Metal engine is correct (8-component conservative).
2. **(Critical) No beam-target neutron yield model** — only thermonuclear Bosch-Hale. Real DPF beam-target dominates by 10-100×.
3. **(Critical) No electrode ablation / material injection** — anode vapor contamination quenches pinch, missing entirely.
4. **(Major) WENO5 FV/FD mismatch in Python engine** — uses FV coefficients on point values → ~3rd order, not 5th. Metal engine correct.
5. **(Major) Silent physics disabling** — `engine.py:895` silently skips viscosity for Metal backend. No warning.

**Top 3 Concerns:**
1. Fidelity reassessed: **6/10 (Python), 7.5/10 (Metal)** vs claimed 8.9/10.
2. No single backend is both numerically correct AND physically complete.
3. Convergence evidence is weak — no MMS, no Richardson extrapolation.

### hpc-engineer → Coordinator
**Topic:** Cross-Domain Challenges Sent & Key Findings

**Key HPC Findings:**
- Python WENO5 is architecturally broken (~1.2M Numba transitions per step at 128³).
- Memory ceiling: 256³ max for float64 on 36GB, 512³ impossible.
- Metal GPU float32 limitation undermines highest-fidelity config.
- Recommended: PLM+HLLD on Metal for production, AthenaK for reference V&V.

**Challenges sent to:**
- plasma-physics: 5 challenges on computational feasibility (Python WENO5 broken, Hall CFL, memory ceiling).
- godot-viz: Data throughput analysis (22MB/frame at 64³, 176MB/frame at 128³; 10-30fps at 64³, 2-10fps at 128³).

**Wants to hear from:** architect (NumPy state dict bottleneck), vv-engineer (float32 vs float64 reproducibility), pulsed-power (circuit coupling frequency).

### architect → Coordinator
**Topic:** Round 1 Survey Sent to All 6 Agents

**Key Cross-Cutting Architectural Issues:**
1. engine.py is a 1,321-LOC god object mixing orchestration, physics, and diagnostics.
2. State dict is untyped and fragile — any change silently breaks WALRUS.
3. Backend capabilities are not advertised — physics silently disabled for Metal/AthenaK.
4. No API versioning, no authentication, no job persistence.
5. Two competing frontends (Electron GUI exists, Godot planned) with no protocol versioning.

**Challenges sent to:** all 6 other principal agents on interface requirements.

---

## Verification Agent Findings (Round 1)

### circuit-theory-verifier → Coordinator
**Topic:** 10 Findings on Circuit Model — Independent Audit

**CRITICAL (4):**
1. **NX2 preset capacitance WRONG by factor of 31** — `presets.py:79` has 0.9 µF, `experimental.py:122` has 28 µF. Stored energy: 64.8 J vs 2,744 J. Fatal for NX2 modeling.
2. **PF-1000 external inductance discrepancy — factor of 2.2** — `presets.py:58` has 15 nH, `experimental.py:108` has 33 nH. ~15% error in peak current timing.
3. **Missing dL_plasma/dt back-EMF — NO current dip possible** — `CouplingState.dL_dt` always 0.0. The I×dLp/dt motional EMF is THE dominant mechanism for DPF current dip. Without it, cannot validate against ANY experimental waveform.
4. **Back-EMF from plasma motion is zero** — `coupling.emf` always 0.0. Combined with #3, circuit is a free-ringing RLC with dissipative load. Cannot reproduce DPF waveform morphology.

**MAJOR (4):**
5. Lumped-element only — no transmission line, MITL, PFN, or Marx transient analysis.
6. No crowbar/diode circuit element — affects late-time behavior of large DPFs (PF-1000, LLNL-DPF).
7. No frequency-dependent electrode resistance (skin effect) — factor ~440× for 58mm anode at 1 MHz.
8. Plasma resistance cap too low (10 Ω) — clips real physics during pinch when R can exceed 100 Ω.

**MODERATE (2):**
9. Circuit timestep may miss fast dR_plasma/dt during pinch.
10. Implicit midpoint integrator is correct and A-stable — "the one thing the circuit model gets unambiguously right."

**Overall circuit fidelity: 3/10** — "a generic RLC oscillator, NOT a DPF circuit model."

---

## Emerging Key Conflicts

| Conflict | Agents | Issue |
|----------|--------|-------|
| **Fidelity Rating** | plasma-physics vs project claim | 6-7.5/10 vs claimed 8.9/10 |
| **Python Engine Validity** | plasma-physics, hpc-engineer | Non-conservative pressure + FV/FD mismatch + broken WENO5 |
| **Circuit Model Credibility** | circuit-theory-verifier, vv-engineer | dLp/dt=0, wrong presets, no current dip, fidelity 3/10 |
| **No Experimental Validation** | vv-engineer, domain-science (pending) | Zero digitized waveforms, zero sim→experiment comparison |
| **Backend Completeness** | plasma-physics, architect | No backend is numerically correct AND physically complete |
| **Silent Physics Disabling** | plasma-physics, architect | Metal/AthenaK silently skip physics with no warnings |
| **CI Coverage Gap** | vv-engineer, hpc-engineer | 100+ Metal tests never run in CI |

---

## Round 2: Rebuttals & Cross-Domain Synthesis

### kinetic-verifier → Coordinator
**Topic:** 19 Physics Issues Found in Transport Implementation

**Category A — Formula Errors (HIGH):**
1. **(Critical) Spitzer resistivity missing Z-dependent correction factor** — code uses simplest form; NRL Formulary has alpha(Z) from Rosenbluth potential integrals. For high-Z impurities (Cu/W ablation, Z_eff=5-20), error is 30-50%.
2. **(Moderate-High) T-relaxation uses m_p instead of m_d** — overestimates relaxation rate by 2× for deuterium. Also T_eq formula wrong for Z>1.
3. **(Moderate) Three inconsistent Coulomb logarithm implementations** across spitzer.py, nernst.py, viscosity.py. nernst.py missing Z_eff in argument.
4. **(Moderate) kappa_perp inconsistency** — spitzer.py uses crude approx (kappa_par/(1+x²)), anisotropic_conduction.py uses correct Braginskii (4.66 coefficient). 32% discrepancy.
5. **(Moderate) tau_e hardcodes Z=1** — overestimates collision time for multi-species.

**Category B — Missing Physics (MODERATE):**
6. No electron viscosity. 7. No thermal force (friction from ∇Te). 8. No Ettingshausen effect. 9. Z_eff assumed constant (no ionization model). 10. No lower-hybrid drift instability (LHDI). 11. No thermonuclear <σv> rate.

**Category C — Implementation Bugs:**
12. **(BUG) Nernst velocity missing grid spacing in np.gradient** — `nernst.py:169-173` uses np.gradient WITHOUT dx,dy,dz → wrong units [K/cell] not [K/m].
13. **(BUG) Radiation transport uses m_p not m_d** — ne ~2× too high for deuterium.
14. Braginskii sub-cycling doesn't recompute gradients. 15. Viscous heating ignores anisotropic stress. 16. PIC has no collision operator.

**Category D — Scope Gaps:**
17. No collisional-to-collisionless transition. 18. No two-fluid effects beyond Hall term. 19. Radiation opacity model only free-free.

**Strengths noted:** Braginskii viscosity tensor (full η_0-η_3), Sharma-Hammett flux limiter, Nernst Epperlein-Haines fits, RKL2 super time-stepping, 3-channel radiation model.

### circuit-exp-verifier → Coordinator
**Topic:** 8 Findings on Experimental Pulsed Power Fidelity

**CRITICAL:**
1. **NX2 preset fundamentally wrong** — independently confirms circuit-theory-verifier. Published Lee & Saw (2009): C=28µF, R₀=2.3mΩ, anode=9.5mm, cathode=32mm. Preset stored energy=64.8J vs description "3 kJ" (off by 46×).
2. **PF-1000 L₀ and R₀ too low** — confirms circuit-theory-verifier. Missing: collector plate inductance (~10-15nH), transmission line feed-through (~5-10nH). Peak current overshoot: 4.63 MA vs measured 1.8-2.2 MA (2.1-2.6×).

**MAJOR:**
3. **Missing circuit elements**: switch impedance, multi-module jitter, skin effect, crowbar/freewheel.
4. **dL/dt coupling inconsistency** — engine computes two DIFFERENT Lp estimates (volume integral vs solver linearization), then overrides Lp but keeps dL_dt from different source. dL_dt may have wrong magnitude or sign at pinch time.
5. **Grid domains don't contain electrode geometry** — PF-1000 grid covers only 40% of anode radius. Grid r_max < anode_radius means sheath dynamics unresolvable.
6. **Cannot produce characteristic two-hump DPF waveform** — needs snowplow mass sweeping, correct pinch radius evolution, m=0 instability physics.

**Experimental fidelity: 4/10** (slightly more generous than circuit-theory-verifier's 3/10).

### architect → Coordinator
**Topic:** Round 2 Synthesis — Devastating Cross-Cutting Finding

**"NO SINGLE BACKEND WORKS":**
| Backend | Numerics | Physics | Circuit Coupling | Verdict |
|---------|----------|---------|-----------------|---------|
| Python | BROKEN | Most complete | Yes (but dLp/dt=0) | Incorrect numerics |
| Metal | CORRECT | Incomplete | None | Missing physics |
| Athena++ | CORRECT | Partial C++ | Hardcodes R=0 | Broken coupling |
| AthenaK | CORRECT | Basic MHD only | None | No DPF physics |

**Architecture Kill List:** Python WENO5, "8.9/10 fidelity" claim, "ASME V&V 20 Tiers 1-5" claim, silent physics disabling.

**Proposed path forward:** Converge on Metal as production engine. Port Python physics to Metal. Fix circuit model. Add experimental validation.

### hpc-engineer → Coordinator
**Topic:** Round 2 — Strategic Misallocation Finding

**KEY INSIGHT: 2D cylindrical (r,z) is the practical production regime, not 3D Cartesian.**
- 3D minimum (48M cells) takes 30-60 min per µs — publication only.
- 3D production (384M cells) is OOM on 36GB.
- 2D cylindrical (3M cells) achieves 5-10 steps/sec on Metal — usable for exploration.
- **BUT the 2D CylindricalMHDSolver is the LEAST developed** — only PLM+HLL, no WENO5/HLLD/RK3.
- All Phase O/P accuracy improvements went into 3D Cartesian solver.
- **This is a strategic misallocation of development effort.**

Performance data provided to all requesting agents: Metal 64³=60-200fps, Python 64³=30-50fps, WebSocket binary adequate for 64³ real-time.

### godot-viz → Coordinator
**Topic:** Round 2 Revised Recommendations — 2D First, Not 3D

**Fundamental pivot:** DPF physics is axisymmetric → primary viz is 2D r-z pseudocolor, not 3D volumetric.

**Revised top priorities:**
1. Fix binary WebSocket handler (1 line fix + ~30 LOC parser)
2. Add server-side J_z = curl(B)_z computation (~10 LOC)
3. Add 2D pseudocolor panel (ECharts heatmap or Canvas)
4. Resurrect dead TraceConfig component for log/linear toggle
5. Add simulation health indicator

**Three.js/Godot 3D is now Phase 3** (future, only if 3D demand exists).

### domain-science → Coordinator
**Topic:** Round 1 Challenges Completed — Devastating Fidelity Reassessment

**Proposed revised fidelity grading:**
- Numerical methods: 8.9/10
- Physics completeness: 6.5/10
- Validation evidence: **1/10**
- **Overall system: 4/10**

**Confirmed "8.9/10 fidelity" is unsupported for DPF physics prediction.**

**5 minimum validation tests proposed (V1-V5):**
- V1: PF-1000 I(t) waveform vs Scholz 2006
- V2: NX2 I(t) vs Lee & Saw 2008
- V3: UNU/ICTP I(t) vs Lee 2014 review
- V4: Current dip timing vs pinch radius
- V5: Neutron yield scaling (Y_n vs E₀)

**PF-1000 L₀ wrong** — independently confirms circuit-theory-verifier and circuit-exp-verifier (should be ~33.5 nH).
**Energy conservation metric critically wrong** — only tracks circuit energy, ignores plasma thermal/kinetic/magnetic.

---

## Round 2 Key Conflicts & Consensus

### CONSENSUS ITEMS (5+ agents agree)

| Item | Agents | Status |
|------|--------|--------|
| **"8.9/10 fidelity" is unsupported** | plasma-physics, vv-engineer, domain-science, architect, circuit-theory-verifier | UNANIMOUS (6 agents) |
| **NX2 preset wrong by 31×** | circuit-theory-verifier, circuit-exp-verifier, domain-science | INDEPENDENTLY CONFIRMED (3 agents) |
| **PF-1000 L₀ wrong by ~2.2×** | circuit-theory-verifier, circuit-exp-verifier, domain-science | INDEPENDENTLY CONFIRMED (3 agents) |
| **dLp/dt=0 is the #1 circuit deficiency** | circuit-theory-verifier, circuit-exp-verifier, plasma-physics, vv-engineer, domain-science | UNANIMOUS (5 agents) |
| **Zero experimental validation** | vv-engineer, domain-science, plasma-physics, architect | UNANIMOUS (4 agents) |
| **No single backend is correct AND complete** | plasma-physics, architect, hpc-engineer | UNANIMOUS (3 agents) |
| **Python engine numerically broken** | plasma-physics, hpc-engineer, architect | UNANIMOUS (3 agents) |
| **2D viz before 3D viz** | godot-viz, hpc-engineer, plasma-physics | UNANIMOUS (3 agents) |
| **Binary WebSocket must be fixed first** | godot-viz, hpc-engineer, architect | UNANIMOUS (3 agents) |

### REMAINING DISAGREEMENTS

| Topic | Position A | Position B | Agents |
|-------|-----------|-----------|--------|
| **Overall fidelity** | 4/10 (domain-science) | 6-7.5/10 (plasma-physics) | Rating depends on whether you count circuit + validation |
| **Circuit fidelity** | 3/10 (circuit-theory) | 4/10 (circuit-exp) | Minor disagreement |
| **Python engine fate** | Kill WENO5, demote to teaching (architect, hpc) | Fix and maintain (plasma-physics implied) | Needs resolution |
| **Production engine** | Metal GPU (architect, hpc) | Converge later (plasma-physics) | Needs resolution |

### CRITICAL NEW FINDINGS FROM ROUND 2

1. **Nernst velocity has a BUG** — np.gradient without dx gives wrong units (kinetic-verifier C3)
2. **Radiation ne uses m_p not m_d** — 2× error in electron density (kinetic-verifier C4)
3. **Grid domains too small for electrodes** — PF-1000 grid covers only 40% of anode (circuit-exp-verifier F5)
4. **2D cylindrical solver is least developed** but is the practical production regime (hpc-engineer)
5. **dL/dt coupling uses inconsistent Lp estimates** — engine overrides solver Lp but keeps solver dL_dt (circuit-exp-verifier F4)

---

## Round 3: Synthesis, Corrections & Final Consensus

### CRITICAL CORRECTIONS ISSUED

#### circuit-theory-verifier CORRECTION: dLp/dt is NOT always zero
**Original claim (Finding 3):** "dL_dt is structurally dead code — never set to non-zero"
**CORRECTED:** Fluid solvers DO compute dL_dt via backward difference: `mhd_solver.py:2099-2104`, `cylindrical_mhd.py:644-647`, `metal_solver.py:897-900`. The error was that the verifier checked only the base class defaults without checking subclass implementations.
**Revised Finding:** dL_dt IS computed but from an INCONSISTENT inductance estimate. Engine overwrites coupling.Lp with volume integral but keeps dL_dt from solver's crude Bθ-average Lp_est. Circuit sees Lp and dLp/dt from two different L definitions.
**back_emf downgraded:** From CRITICAL to MODERATE — R_star formulation captures dominant I×dL/dt term.
**Confirmed by:** domain-science, pulsed-power, circuit-exp-verifier (all verified dL_dt code paths independently)

#### pulsed-power SELF-CORRECTIONS (3 findings)
1. Back-EMF: Downgraded CRITICAL→MAJOR. I×dL/dt in circuit equation DOES provide partial back-EMF.
2. Anomalous resistivity direction: Was backwards. Lighter m_p gives HIGHER v_ti threshold.
3. Skin effect magnitude: Analytical shows ~2% of R₀ for PF-1000, not 10-30%. Downgraded to ENHANCEMENT.

---

### NEW CRITICAL FINDINGS (Round 3)

#### FINDING: BREMSSTRAHLUNG COEFFICIENT 10⁸× TOO LARGE (neutron-rad-verifier)
**Location:** `src/dpf/radiation/bremsstrahlung.py:25`, also `line_radiation.py:409`
**Bug:** `BREM_COEFF = 1.69e-32` is the NRL Plasma Formulary coefficient for CGS units (nₑ in cm⁻³, Tₑ in eV). Code applies it with SI units (nₑ in m⁻³, Tₑ in K) WITHOUT unit conversion.
**Correct SI coefficient:** 1.569e-40
**Impact:** At DPF pinch conditions (nₑ=10²⁵ m⁻³, Tₑ=1 keV): code gives 6.91×10²¹ W/m³ (2.17 MJ radiated from 1mm³ in 100ns — physically impossible for a 3 kJ device). Correct: 6.42×10¹³ W/m³ (20 mJ — reasonable).
**Consequence:** ALL radiation-enabled simulations have catastrophically wrong radiation losses. Implicit Newton solver masks the bug by preventing T<0, but electrons crash to floor temperature, corrupting Spitzer resistivity, thermal conductivity, and every Te-dependent quantity. THIS IS THE SINGLE WORST BUG FOUND IN THE ENTIRE AUDIT.

#### FINDING: METAL HLLD MISSING DOUBLE-STAR STATES (vv-engineer, confirmed by mhd-theory-verifier)
**Location:** `metal_riemann.py:807-1014`
Metal implements only 4-region flux selection (FL, F*L, F*R, FR). Full Miyoshi & Kusano (2005) HLLD has 6 regions with double-star states between Alfvén waves and contact (F**L, F**R). Python engine `mhd_solver.py:416-619` correctly implements all 6 regions.
**Impact:** Metal "HLLD" is actually HLLC-like — does NOT resolve Alfvén/rotational discontinuities. More diffusive on slow compound waves. Cross-backend parity tests unknowingly compare different algorithms.
**V&V Gap:** No test in the suite can detect this — Alfvén wave test uses uniform states on Python engine only.

#### FINDING: CONVERGENCE TEST METHODOLOGY FUNDAMENTALLY FLAWED (numerics-stability-verifier)
Tests use fixed n_steps with CFL-limited dt, meaning fine grids run for LESS physical time than coarse grids. Measures deviation from IC, not convergence to exact solution. The ~1.86 order is an artifact of mixed physical-evolution + numerical-error measurement, NOT the true formal order.
**Standalone WENO5 reconstruction test IS properly designed** — correctly measures ~5th order by comparing to exact function values.

#### FINDING: OPERATOR SPLITTING IS LIE (1st ORDER) THROUGHOUT (numerics-stability-verifier)
Resistive MHD, Hall, Braginskii, Nernst, ADI — all use Lie splitting (ideal MHD → explicit resistive → transport). Reduces overall accuracy to O(dt) regardless of SSP-RK3's O(dt³) for the ideal MHD step. Strang splitting would give O(dt²) but is not implemented.

#### FINDING: RESISTIVE CFL NOT ENFORCED (numerics-stability-verifier)
Neither Metal nor Python solver restricts dt for resistive stability: dt < dx²μ₀/(2η). Explicit resistive step WILL go unstable for large η. No enforcement, no warning.

---

### VERIFICATION AGENT DEEP-DIVES

#### mhd-theory-verifier — 22 Items Verified Correct, 3 Critical, 3 Moderate
**Verified Correct (22):** Conservative state vector, physical fluxes (Stone et al. 2020), Python HLLD (full 6-region Miyoshi & Kusano), induction equation (all 3 Ohm's law terms), Dedner divergence cleaning, Powell 8-wave sources, SSP-RK2/RK3 (both solvers), cylindrical geometry (Goedbloed & Poedts), electrode BC (Ampere's law), EOS, all transport coefficients, physical constants (CODATA 2018), two-temperature physics, constrained transport (Gardiner & Stone 2005), slope limiters.
**Critical (3):** C1: Non-conservative pressure (Python, already known). C2: Metal HLLD missing double-star states (NEW). C3: Python hybrid WENO5 boundary mismatch (already known).
**Moderate (3):** Unit convention mismatch between solvers. Python HLL tracks only 3/8 fluxes. Fast magnetosonic speed approximation (upper bound, safe but suboptimal).

#### numerics-stability-verifier — SSP-RK Verified, CFL Issues, Splitting Concerns
**Verified Correct:** SSP-RK2 (both solvers), SSP-RK3 (both solvers, vs Gottlieb et al. 2001), Metal CFL computation, fast magnetosonic speed (numerically stable form), WENO5-Z FD coefficients (independently derived via Lagrange interpolation), WENO-Z nonlinear weights (p=2).
**Issues:** Python CFL 2-4× over-conservative. Resistive CFL missing. Lie splitting throughout (1st order). ADI Douglas-Gunn (1st order). Convergence test methodology weak. CT simplified (reduces but doesn't eliminate div(B)). Velocity clamping breaks momentum conservation.

#### diagnostics-verifier — 45% Diagnostic Completeness
**Exists:** Interferometry (Abel transform + inversion), neutron yield (Bosch-Hale + Lee model), radiation (3-channel + FLD), ionization (Saha + CR), HDF5 output, derived quantities (|J|, β, Mach).
**Missing (ZERO):** Synthetic X-ray imaging, Thomson scattering, Faraday rotation, SXR time histories, streak camera, magnetic probes/dI/dt synthesis, detector response models, digitized experimental data files, time-resolved virtual diagnostics.
**Key finding:** `validate_current_waveform()` function EXISTS but is NEVER CALLED IN ANY TEST.

#### neutron-rad-verifier — CRITICAL Bremsstrahlung Bug + Integration Gaps
**CRITICAL:** BREM_COEFF 10⁸× too large (CGS→SI conversion missing).
**MAJOR:** Beam-target yield module exists but NOT called from engine.py (thermonuclear only). Electrode ablation module exists but NOT integrated into continuity equation.
**MINOR:** DD reactivity D(d,p)T branch reuses wrong C2-C7 coefficients (<1% error).
**Strengths:** Comprehensive radiation architecture, implicit solvers, beam-target module well-implemented.

---

### FIDELITY RATING CONVERGENCE (Round 3 Final)

| Agent | Isolated MHD (Metal) | Coupled DPF | Circuit | Validation | Overall |
|-------|---------------------|-------------|---------|------------|---------|
| plasma-physics | 7.5/10 | **4.5/10** | — | — | **4.5/10** (coupled) |
| domain-science | — | — | — | 1/10 | **3/10** (revised down) |
| circuit-theory-verifier | — | — | 3/10 | — | — |
| circuit-exp-verifier | — | — | 4/10 | — | — |
| vv-engineer | — | — | — | 0/10 | "unratable" |
| architect | 7.5/10 Metal, 4/10 Python | — | 3/10 | 0/10 | **0/10 validated, ~5/10 framework** |
| hpc-engineer | — | — | — | — | — |
| **Project self-assessment** | **8.9/10** | **8.9/10** | — | — | **8.9/10** |

**CONSENSUS FIDELITY:** The "8.9/10" must be retracted. Honest ratings:
- Numerical methods quality (Metal): **7.5/10** (pending HLLD double-star fix)
- Physics completeness: **5/10** (missing beam-target, ablation, dLp/dt coupling broken)
- Circuit model: **3-4/10**
- Experimental validation: **0/10**
- **Overall DPF simulator: 3-4.5/10** depending on weighting

---

### CONFIRMED BUGS — MASTER LIST

| # | Bug | Severity | Location | Found By | Impact |
|---|-----|----------|----------|----------|--------|
| **B1** | BREM_COEFF 10⁸× too large | **CRITICAL** | bremsstrahlung.py:25, line_radiation.py:409 | neutron-rad-verifier | ALL radiation-enabled sims wrong |
| **B2** | NX2 preset C wrong by 31× | **CRITICAL** | presets.py:79 | circuit-theory, circuit-exp, diagnostics | NX2 sims invalid |
| **B3** | PF-1000 preset L₀ wrong by 2.2× | **HIGH** | presets.py:58 | circuit-theory, circuit-exp, domain-science | PF-1000 timing wrong |
| **B4** | Grid domains don't contain electrodes | **HIGH** | presets.py (all cylindrical) | circuit-exp-verifier | DPF implosion impossible |
| **B5** | Nernst velocity missing dx in np.gradient | **HIGH** | nernst.py:169-173 | kinetic-verifier | Wrong transport units |
| **B6** | Radiation ne uses m_p not m_d | **MODERATE** | radiation/transport.py:302 | kinetic-verifier | ne 2× too high |
| **B7** | Metal HLLD missing double-star states | **HIGH** | metal_riemann.py:807-1014 | vv-engineer, mhd-theory-verifier | Alfvén waves unresolved |
| **B8** | Convergence tests measure wrong quantity | **MODERATE** | test_phase_o:901+ | numerics-stability-verifier | ~1.86 order unreliable |
| **B9** | Resistive CFL not enforced | **MODERATE** | metal_solver.py, mhd_solver.py | numerics-stability-verifier | Blowup risk for large η |
| **B10** | PF-1000 R₀ wrong by 4× | **MODERATE** | presets.py:58 | circuit-exp-verifier | Energy dissipation wrong |
| **B11** | dL_dt/Lp inconsistency | **MODERATE** | engine.py:588-589 | circuit-exp-verifier, domain-science | Current dip magnitude wrong |
| **B12** | presets.py ↔ experimental.py parameter mismatch | **MODERATE** | presets.py, experimental.py | diagnostics-verifier | Data integrity |

---

### ARCHITECT'S 8 DEFINITIVE DECISIONS

1. **Three.js** — NOT Godot, NOT WebGPU. Unanimous. 2D r-z pseudocolor is primary need.
2. **Conservative MHDState dataclass** — replaces untyped dict. Conservative variables (ρ, ρv, E, B).
3. **DeviceState wrapper** — keeps data on solver's native device, materializes to NumPy only for diagnostics/coupling.
4. **BackendCapabilities Flag enum** — fail-loud when requested physics unavailable.
5. **Priority order:** P0: Fix dLp/dt+emf. P1: BackendCapabilities. P2: Fix presets. P3: MHDState dataclass. P4: DeviceState. P5: Observer diagnostics. P6: MMS tests. P7: Viz WebSocket. P8: Three.js 2D panel.
6. **PhysicsContext dataclass** — replaces **kwargs in solver step().
7. **Observer pattern** for diagnostics — physics-critical quantities inline, all diagnostics attachable.
8. **Separate /ws/viz/ endpoint** — decouples viz frame rate from solver step rate.

---

### FINAL CONSENSUS & KEY DISAGREEMENTS

#### RESOLVED DISAGREEMENTS
| Topic | Resolution | How Resolved |
|-------|-----------|-------------|
| **dLp/dt = 0** | CORRECTED: dL_dt IS computed but inconsistently | circuit-theory-verifier self-corrected after cross-verification |
| **Python engine fate** | DEMOTE to teaching/prototyping only | plasma-physics, hpc-engineer, architect all agree |
| **Production engine** | Metal GPU as primary production path | architect, hpc-engineer, plasma-physics converged |
| **3D viz framework** | Three.js inside Electron (2D first) | godot-viz, architect, hpc-engineer unanimous |
| **Fidelity rating** | "8.9/10" RETRACTED. Overall ~3-4.5/10 | 6 agents agree; domain-science: 3/10, plasma-physics: 4.5/10 coupled |

#### UNRESOLVED (needs Phase 2 investigation)
| Topic | Positions | Agents |
|-------|-----------|--------|
| Minimum circuit fix scope | Option A: fix dL_dt consistency (~days) vs Option B: Lee model (~weeks) | circuit-theory, circuit-exp |
| 2D cylindrical solver path | Port Metal accuracy to cylindrical vs improve CylindricalMHDSolver vs use Athena++ cylindrical | hpc-engineer, plasma-physics |
| Operator splitting upgrade | Strang splitting (O(dt²)) vs current Lie (O(dt)) — how much accuracy gain? | numerics-stability-verifier |

---

### MASTER PRIORITY LIST (Synthesized from all 17 agents)

**TIER 0 — STOP THE BLEEDING (1-2 days)**
1. Fix BREM_COEFF: 1.69e-32 → 1.569e-40 (neutron-rad-verifier). Also fix line_radiation.py:409.
2. Fix NX2 preset: C=28µF, R₀=2.3mΩ, anode=9.5mm, cathode=32mm (circuit-theory, circuit-exp, diagnostics)
3. Fix PF-1000 preset: L₀=33.5nH, R₀=12mΩ (circuit-theory, circuit-exp, domain-science)
4. Fix Nernst grid spacing bug (kinetic-verifier)
5. Fix radiation ne m_p→m_d (kinetic-verifier)

**TIER 1 — ENABLE VALIDATION (1-2 weeks)**
6. Fix grid domains: ensure nr×dr > cathode_radius for all cylindrical presets (circuit-exp-verifier)
7. Fix dL_dt/Lp consistency in engine.py (circuit-exp-verifier, domain-science)
8. Add BackendCapabilities + fail-loud for missing physics (architect, plasma-physics)
9. Add MMS test for Metal engine (~200 LOC) (vv-engineer)
10. Digitize PF-1000 I(t) from Scholz 2006 + run comparison (domain-science, vv-engineer)

**TIER 2 — PRODUCTION QUALITY (2-4 weeks)**
11. Add Metal HLLD double-star states (vv-engineer, mhd-theory-verifier)
12. MHDState conservative dataclass (architect, plasma-physics)
13. Fix binary WebSocket + add 2D pseudocolor viz (godot-viz, ~80 LOC + moderate effort)
14. Wire beam-target neutron yield into engine loop (neutron-rad-verifier)
15. Add resistive CFL enforcement (numerics-stability-verifier)
16. Add macOS Apple Silicon CI runner (vv-engineer)

**TIER 3 — RESEARCH GRADE (4+ weeks)**
17. Port Metal accuracy (WENO5-Z, HLLD, SSP-RK3) to 2D cylindrical solver (hpc-engineer)
18. Strang splitting for resistive/transport operators (numerics-stability-verifier)
19. Wire electrode ablation into continuity equation (neutron-rad-verifier)
20. Synthetic diagnostic suite (X-ray, Thomson, magnetic probes) (diagnostics-verifier)

---

---

## ROUND 3 FINAL RESPONSES — Complete Agent Reports

*All 16 reporting agents have now submitted final findings. walrus-verifier (Task #17) still pending.*

### circuit-theory-verifier (Agent 8) — Final Circuit Analysis

**Key contributions:**
- **Option C recommended** (~35 lines, circuit 3/10 → 6/10): Fix grid domains, presets, dL_dt consistency, add crowbar. NOT Lee model replacement (Lee model replaces the MHD solver, not just the circuit).
- **Circuit subcycling analysis**: Circuit evolves ~3500× slower than MHD. Subcycle N_sub=10-50 steps. Exception: reduce N_sub during pinch phase when |dL_dt × dt_circuit| > 0.1 × L_total.
- **Integrator confirmed adequate**: Implicit midpoint is A-stable, handles stiff dLp/dt coupling correctly. Even during fast pinch (dLp/dt → 10⁶ H/s), the implicit scheme stays stable — just degrades to 1st order.
- **CouplingState design is correct** — problems are all implementation: emf never populated, Lp overridden without matching dL_dt, crude Lp estimator.
- **Revised rating**: 4/10 (up from 3/10 after correcting dL_dt=0 claim). Consensus with circuit-exp-verifier.
- **Skin effect downgraded**: MAJOR → MINOR (~2% of R₀, baked into experimental waveform fits).
- **Lee model validation path**: Lee model code EXISTS at `validation/lee_model_comparison.py`. Running both models with identical parameters directly measures coupling quality.

### circuit-exp-verifier (Agent 9) — Three-Way Circuit Consensus

**Unanimous findings (Agents 1+8+9):**
1. NX2 C wrong by 31× (0.9→28 µF)
2. PF-1000 L₀ wrong by 2.2× (15→33.5 nH)
3. ALL cylindrical grids too small for electrodes (root blocker)
4. dL_dt exists but inconsistent (engine overrides Lp, keeps solver's different dL_dt)
5. Cannot produce characteristic DPF waveform
6. Energy accounting is circuit-only
7. Implicit midpoint integrator is excellent (9/10)

**CORRECTION**: PF-1000 R₀ revised from "4× error" to "~2× error" (~6 mΩ, not 12 mΩ). Original conflated PF-1000 and NX2 values.

**Consensus score: 3.5/10** overall circuit model.

**Agreed fix priority**: Grid domains → Preset parameters → dL_dt consistency → Enhancements.

**NEW finding (from Agent 8)**: Cartesian B_theta proxy `√(Bx²+By²)` at `mhd_solver.py:2093` is incorrect — includes radial component. Proper B_theta requires coordinate transform.

### mhd-theory-verifier (Agent 10) — Dual-Axis Fidelity Clarification

**Critical insight**: The "8.9/10" and "6.5/10" ratings measure DIFFERENT axes:
- **Numerical Method Sophistication: 8.5-8.9/10** — WENO5-Z + HLLD + SSP-RK3 + float64 + CT. Genuinely state-of-the-art, comparable to Athena++/FLASH/PLUTO.
- **Physical Realism: 6.0-7.5/10** — Limited by ideal gas EOS, no beam-target, non-conservative Python pressure, Metal HLLD missing double-star, only Buneman anomalous resistivity, no regime-switching diagnostics.
- **Overall DPF simulator: 7.0-7.5/10** — physics gaps limit what excellent numerics can deliver.

**Bremsstrahlung 10⁸× independently confirmed** via dimensional analysis:
- WRONG coefficient: P_ff drains ALL thermal energy in ~3 ps (absurd)
- CORRECT coefficient: cooling time ~350 µs (physically reasonable)
- Cascading corruption: Te→floor → η_Spitzer→∞ → κ_Braginskii→0 → T_i dragged to floor
- Energy conservation tests PASS (implicit solver is internally consistent) but absolute balance WRONG

**Revised ratings**: With bremsstrahlung enabled: 4.0-5.0/10. With bremsstrahlung disabled: 7.0-7.5/10. Numerical methods: 8.5-8.9/10 (unchanged).

### kinetic-verifier (Agent 11) — Beam-Target Quantitative Verification

**Beam-target vs thermonuclear ratio (definitive calculation):**

| Ti [keV] | Y_therm (100ns) | Y_beam (20ns) | Ratio Y_beam/Y_therm |
|----------|-----------------|---------------|---------------------|
| 1 | 5×10³ | 8×10⁸ | **160,000×** |
| 5 | 3.3×10⁶ | 8×10⁸ | **240×** |
| 10 | 3×10⁷ | 8×10⁸ | **27×** |

plasma-physics and domain-science claims of "10-100×" were CONSERVATIVE. For typical DPFs (Ti~1-5 keV), beam-target dominates by 100-100,000×.

**neutron_yield.py header is MISLEADING**: Line 16 calls beam-target "a correction that requires particle tracking (future Phase)." Scientifically wrong — beam-target is the DOMINANT mechanism. Lee model in beam_target.py doesn't require PIC.

**CORRECTION**: Retracted B6. Thermonuclear yield DOES exist and IS called. The real issue: beam_target.py exists but is never called from engine.py.

**PIC module assessment**: Valuable but not urgent. Boris pusher correct, CIC correct. Missing: collision operator, MHD coupling, tests. Lee model gives 80% of benefit for 5% of effort. Revive PIC as future phase.

**Final issue count**: 14 actionable (3 showstoppers, 3 high, 3 moderate, 5 nice-to-have). Down from 19 after consolidation and correction.

### numerics-fv-verifier (Agent 12) — Complete FV & Shock-Capturing Audit

**20+ items verified correct** coefficient-by-coefficient against published papers:
- WENO5-Z Metal FD coefficients ✅ (Lagrange interpolation verified)
- WENO5-Z Python FV coefficients ✅ (Jiang-Shu 1996)
- WENO-Z weights with p=2 ✅ (Borges 2008)
- HLL flux formula ✅ (Harten-Lax-van Leer 1983)
- HLLD Python: full 5-region with double-star ✅ (Miyoshi & Kusano 2005 Eqs. 43-62)
- SSP-RK2/RK3 coefficients ✅ (Shu-Osher 1988)
- CT preserves ∇·B=0 ✅ (topological property)
- Physical MHD fluxes all 8 components ✅
- PLM minmod/MC limiters TVD ✅
- Fast magnetosonic stable discriminant ✅

**6 findings:**
- F1 (MODERATE): Metal HLLD is 4-region HLLC-MHD, missing double-star Alfvén states
- F2 (DOCUMENTED): Python hybrid WENO5+np.gradient boundary mismatch (known)
- F3 (LOW): Metal HLLD uses |denom| instead of signed — no practical impact (abs is no-op for physical states)
- F4 (LOW): Metal HLLD uses Bn_L in v·B_L instead of averaged Bn — O(Δx) error
- F5 (LOW): Python HLL uses only normal B for cf — safe/dissipative overestimate
- F6 (LOW): Metal CFL max-over-dims instead of sum-over-dims — safe with CFL=0.3 < 1/3

### numerics-stability-verifier (Agent 13) — New Bugs from Cross-Verification

**NEW BUG: Hall CFL uses ne_max instead of ne_min** (`mhd_solver.py:1314-1321`):
Hall speed `v_H = |B|/(µ₀ nₑ e Δx)` is largest where nₑ is smallest. Code uses ne_max (minimum Hall speed). Underestimates CFL constraint. Affects Python engine with `enable_hall=True` and non-uniform density.

**3 additional findings from cross-verification with Agent 12:**
- A: Unsplit CFL under-constraining — max-over-dims gives effective 3D CFL of 0.9 (10% stability margin)
- B: SSP-RK on primitives not conservatives — formally voids SSP guarantee but practical impact low (many production codes do this)
- C: Python WENO5 velocity clamping breaks momentum conservation — up to 19% of cells affected at boundaries

**Total: 13 findings** (10 original + 3 from cross-verification).

### vv-engineer (Agent 4) — Final V&V Synthesis

**ANSWER: Option (D)** — Fix foundational geometry/data bugs FIRST, then MMS + circuit in parallel.

**Rationale**: ALL three device grids are smaller than their electrode radii. MMS tests would verify a solver that can't simulate its target physics.

**Revised 6-tier priority**: Tier 0 (structural prerequisites) → Tier 1 (code verification: MMS, HLLD fix) → Tier 2 (circuit) → Tier 3 (validation: digitize PF-1000/NX2) → Tier 4 (V&V infra) → Tier 5 (physics completeness).

**REVISED FIDELITY (incorporating ALL agents):**
- Grid geometry: **0/10** (doesn't contain electrodes)
- Device presets: **1/10** (NX2 31×, PF-1000 2.2×)
- SSP-RK3 coefficients: 10/10
- WENO5-Z Metal coefficients: 9/10
- Experimental validation: **0/10**
- **Overall system for DPF: 2/10** — "cannot simulate target physics due to geometry"

**Reproducibility**: Accepts hpc-engineer's position — statistical reproducibility (cross-backend L1 parity < 15%) is the correct target. Bitwise reproducibility across float32/float64 is neither achievable nor necessary.

### pulsed-power (Agent 1) — Definitive dL/dt Resolution

**DEFINITIVE PROOF**: dL_dt IS computed and passed to circuit. Exact code path traced:
1. `coupling = self.fluid.coupling_interface()` (engine.py:584) — reads solver's _coupling
2. Engine overrides R_plasma (585) and Lp (589) but NOT dL_dt
3. Circuit receives dL_dt at `rlc_solver.py:133`: `dLp_dt = coupling.dL_dt`
4. Enters R_star at line 152: `R_star = R_eff + dLp_dt`

**Two real problems**: (A) Lp/dL_dt inconsistency — engine's Lp and solver's dL_dt from different estimators. (B) One-step lag — dL_dt from step N-1 reaches circuit at step N.

**NEW BUG CONFIRMED: R_plasma double-counting** — Circuit drains energy via R_plasma AND MHD adds it back as η|J|². Net: current decays ~5-15% too fast.

**Final consolidated list**: 5 CRITICAL, 8 MAJOR, 2 MINOR. Rating: **3.5/10** (consensus).

**7 priority fixes** (ordered by impact/effort): (1) NX2 C=28µF, (2) PF-1000 L₀=33.5nH, (3) R_plasma double-counting, (4) Lp/dL_dt consistency, (5) Grid domains, (6) Lee model validation test, (7) Snowplow initialization.

### hpc-engineer (Agent 3) — Strategic Production Path

**NEW PRIORITY 0: Wire ADI implicit diffusion to Metal resistive MHD**. Explicit resistive diffusion requires 100-200 sub-steps at DPF anomalous resistivity (η~10⁻⁴). ADI already exists in `metal_stencil.py` but isn't connected. 50-100× speedup for resistive DPF. Effort: 1-2 days.

**Fastest path to production 2D cylindrical: Option D (Athena++ cylindrical short-term) + Option A (Metal cylindrical medium-term)**.

Phase 1 (now): Athena++ with `--coord=cylindrical` already supports this at compile time. 500K-1M zcps, float64, HLLD+PPM.
Phase 2 (next sprint): Build Metal cylindrical for interactive use + parameter sweeps.

**Why NOT Option B** (upgrade Python cylindrical): stays CPU, non-conservative, FV WENO5 wrong, all effort wasted when Metal cylindrical built anyway.

**Why NOT Option C** (AthenaK cylindrical): AthenaK has NO native cylindrical coordinates.

**Grid domain impact**: If domains must be 10cm×15cm instead of 6.4cm×6.4cm, cell count increases ~3×. Production 2D becomes ~2-3 min/step instead of ~1 sps. Makes WALRUS surrogate more compelling.

### plasma-physics (Agent 2) — Final Deficiency List & Rating

**3 decisions made:**
1. Python engine: DEMOTE to teaching/prototyping only. Keep as physics reference (most complete coupling).
2. Production engine: Metal with cylindrical geometry (Path 1). Note: Metal "HLLD" is actually HLLC-MHD — fix alongside cylindrical port.
3. Fidelity: ACCEPT 4.0-4.5/10 for coupled DPF.

**Weighted breakdown** (Metal engine, coupled DPF):

| Component | Weight | Score | Contribution |
|-----------|--------|-------|--------------|
| MHD numerics (WENO5+HLLC+RK3) | 25% | 7.0/10 | 1.75 |
| Circuit coupling (Lp, dL/dt, EMF) | 20% | 3.0/10 | 0.60 |
| DPF-specific physics (yield, ablation) | 20% | 2.0/10 | 0.40 |
| Transport/EOS (Spitzer, ideal gas) | 15% | 3.5/10 | 0.53 |
| V&V evidence (convergence, validation) | 20% | 3.0/10 | 0.60 |
| **TOTAL** | 100% | | **3.88 → 4.0/10** |

**Final list**: 5 Critical, 9 Major, 6 Minor (20 total deficiencies).

### domain-science (Agent 7) — Minimum Viable DPF Path

**MUST-HAVE physics (3 items):**
1. Fix presets + grids (correct params ALREADY EXIST in `validation/experimental.py` — 1-hour fix)
2. Current dip must appear (fix dL/dt coupling ordering)
3. Wire Lee Model as validation reference (`validation/lee_model_comparison.py` already implements phases 1-2)

**Validation priority**: V3 (Lee Model comparison) FIRST — requires NO digitized experimental data, code already exists, can be done TODAY after fixing presets.

**Kinetic-verifier triage**: 3 showstoppers (Nernst dx, radiation ne, disconnected CR), 5 important, 11 acceptable for now.

**2D cylindrical: YES, absolutely** — every major DPF code (Lee Model, RADMHD, MH2D, MACH2) uses axisymmetric (r,z).

**Revised rating: 3.5/10** (down from initial 6.5/10). Gap between 3.5 and ~6 is mostly configuration fixes, not fundamental rewrites.

### godot-viz (Agent 5) — Final LOC Estimates

**Phase 1 revised: ~478 LOC** (390 client + 88 server):
- Binary WS fix: 28 LOC
- Server derived fields: 54 LOC (J_z, log_rho, n_e, beta, CFL_local)
- Engine exposure: 15 LOC (Z_bar_field, L_plasma, diagnostics())
- 2D pseudocolor FieldViewer + store: 210 LOC
- Colormap LUTs (10 maps + percentile auto-scale): 50 LOC
- TraceConfig resurrection: 21 LOC
- MultiTraceView mount + traces: 22 LOC
- Health indicator: 48 LOC
- DPF Phase indicator: 22 LOC
- Energy conservation trace: 8 LOC

**Phase 2: ~400 LOC** (crosshair+lineouts, time cursor, comparison overlay, conservation traces, SXR diagnostic, convergence plot, smoke tests).

**Three.js NOT needed for Phase 1** — Canvas wins for 2D heatmaps. `ctx.putImageData()` renders 256×512 in <1ms.

**NEW FINDING**: `Z_bar_field` computed at `engine.py:477` but NOT in state dict. Need 3 LOC to expose. `L_plasma` not in scalar stream — 5 LOC across 4 files.

### diagnostics-verifier (Agent 14) — 100% Cross-Verification Rate

**19/19 challenges confirmed across 3 agents** (domain-science, neutron-rad-verifier, vv-engineer). Zero disputes.

**Dual fidelity assessment** (joint with domain-science):
- Solver Numerics Quality: **8.5/10**
- DPF Physics Fidelity: **3.5/10**

**16 bugs ranked** P0-P3. Top P0: NX2 preset 31×, PF-1000 L₀ 2.2×, bremsstrahlung 10⁸×, 3 contradictory ionization models (Saha/coronal/CR).

**Fastest path to Tier 3 validation**: Fix P0 presets (10 min) → Digitize PF-1000 I(t) (2 hrs) → Run simulation → Compare using existing `validate_current_waveform()` → Abel transform ne(r) comparison. Steps 1-4 achievable in 1 day.

### neutron-rad-verifier (Agent 15) — Coronal Equilibrium & Cooling Time Analysis

**domain-science CONFIRMED correct**: Coronal equilibrium overestimates radiation for DPF pinch periphery.

At bulk keV plasma: τ_ionz ~ 10 ps << τ_pinch → coronal equilibrium VALID.
At sheath (1-50 eV): τ_ionz ~ 1 µs >> τ_pinch → NOT in coronal equilibrium → overestimates line radiation.

**Radiative cooling times (CORRECT bremsstrahlung coefficient):**

| Mechanism | Cooling time | vs τ_pinch (100 ns) | Impact |
|-----------|-------------|---------------------|--------|
| Bremsstrahlung (core, 1 keV) | 37.5 µs | 375× longer | MINOR |
| Bremsstrahlung (BUGGY code) | 0.35 ps | 300,000× shorter | CATASTROPHIC |
| Cu line radiation (sheath, 100 eV) | 0.36 ns | Comparable | SIGNIFICANT |

The 0.35 ps cooling time with the buggy coefficient is the "smoking gun" — no physical plasma radiates all thermal energy in sub-picosecond timescales.

---

## UPDATED CONFIRMED BUGS — MASTER LIST (Revised)

| # | Bug | Severity | Location | Found By | Impact |
|---|-----|----------|----------|----------|--------|
| **B1** | BREM_COEFF 10⁸× too large (CGS→SI) | **CRITICAL** | bremsstrahlung.py:25, line_radiation.py:409 | neutron-rad, mhd-theory (independently confirmed) | ALL radiation-enabled sims wrong |
| **B2** | NX2 preset C wrong by 31× | **CRITICAL** | presets.py:79 | circuit-theory, circuit-exp, diagnostics | NX2 sims invalid |
| **B3** | PF-1000 preset L₀ wrong by 2.2× | **HIGH** | presets.py:58 | circuit-theory, circuit-exp, domain-science | PF-1000 timing wrong |
| **B4** | Grid domains don't contain electrodes | **HIGH** | presets.py (all cylindrical) | circuit-exp, domain-science | DPF implosion impossible |
| **B5** | Nernst velocity missing dx in np.gradient | **HIGH** | nernst.py:169-173 | kinetic-verifier | Wrong transport units (100-1000×) |
| **B6** | Radiation ne uses m_p not m_d | **MODERATE** | radiation/transport.py:302 | kinetic-verifier | ne ~2× too high |
| **B7** | Metal HLLD missing double-star states | **HIGH** | metal_riemann.py:807-1014 | vv-engineer, mhd-theory, fv-verifier | Alfvén waves unresolved |
| **B8** | Convergence tests measure wrong quantity | **MODERATE** | test_phase_o:901+ | numerics-stability | ~1.86 order unreliable |
| **B9** | Resistive CFL not enforced | **MODERATE** | metal_solver.py, mhd_solver.py | numerics-stability | Blowup risk for large η |
| **B10** | PF-1000 R₀ wrong by ~2× | **MODERATE** | presets.py:58 | circuit-exp (CORRECTED from 4×) | Energy dissipation wrong |
| **B11** | dL_dt/Lp inconsistency | **MODERATE** | engine.py:588-589 | circuit-exp, pulsed-power | Current dip magnitude wrong |
| **B12** | presets.py ↔ experimental.py parameter mismatch | **MODERATE** | presets.py, experimental.py | diagnostics-verifier | Data integrity |
| **B13** | Hall CFL uses ne_max instead of ne_min | **MEDIUM** | mhd_solver.py:1314-1321 | numerics-stability (NEW) | Underestimates Hall CFL constraint |
| **B14** | R_plasma double-counting (circuit + MHD) | **MODERATE** | engine.py ~570 | plasma-physics, pulsed-power (NEW) | Current decays ~5-15% too fast |
| **B15** | Metal CFL max-over-dims (not sum-over-dims) | **LOW** | metal_solver.py:320-337 | fv-verifier, stability-verifier | 10% stability margin at CFL=0.3 |
| **B16** | Cartesian B_theta proxy incorrect | **LOW** | mhd_solver.py:2093 | circuit-theory (NEW) | Lp estimation wrong in Cartesian |

**Total: 16 confirmed bugs** (up from 12).

---

## ARCHITECT'S PHASE 2 MILESTONE PLAN (Final)

### Milestone Dependency Graph

```
M0   (1-2 days)  — Immediate bug fixes (presets, WebSocket, bremsstrahlung)
M0.5 (1-2 days)  — Wire ADI implicit diffusion to Metal resistive MHD [NEW]
M1   (3-5 days)  — BackendCapabilities + fail-loud + conservation guards
M2   (5-7 days)  — Metal circuit coupling + energy conservation + R_plasma fix
M3   (5-7 days)  — Metal cylindrical geometry + two-temp EOS
M4   (5-7 days)  — Conservative MHDState + DeviceState
M5   (7-10 days) — Experimental validation + MMS + frozen baselines (MOMENT OF TRUTH)
M6   (5-7 days)  — 2D visualization + observer pattern (PARALLEL with M1-M5)
M7   (3-5 days)  — Beam-target neutron yield (post-processing observer)
```

**Critical path**: M0 → M0.5 → M1 → M2 → M3 → M5 (~24-34 days)
**Parallel path**: M6 can proceed independently after M0.

### Fidelity Projections

| After Milestone | Projected Fidelity | Justification |
|-----------------|-------------------|---------------|
| Current | ~4/10 framework, 0/10 validated | All-agent consensus |
| After M0-M1 | 5.5/10 | Correct presets, no silent physics disabling |
| After M2 | 6.5/10 | Metal has circuit coupling, energy conservation |
| After M3 | 7.5/10 | Production cylindrical solver with full numerics |
| After M5 | **7.5-8.5/10 if validated** | Depends on experimental comparison |
| After M7 | 8-9/10 | Beam-target neutrons + validated waveforms |

### Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| M5 validation fails | 40% | CRITICAL | Improve L_plasma, add ablation, tune anomalous η |
| M4 MHDState breaks WALRUS | 30% | HIGH | walrus-verifier pre-validates channel mapping |
| M3 axis singularity NaN | 25% | MEDIUM | Ghost cell reflection + L'Hôpital limit |
| M2 crude L_plasma coupling | 40% | HIGH | Start with Python's proven coupling, compare |
| M1 tests rely on silent disabling | 30% | LOW | Audit all fixtures before merge |

---

## FINAL CONSENSUS TABLE (All 17 Agents)

### Fidelity Ratings by Agent

| Agent | MHD Numerics | Coupled DPF | Circuit | Validation | Overall |
|-------|-------------|-------------|---------|------------|---------|
| plasma-physics | 7.0/10 Metal | **4.0/10** | — | — | **4.0/10** |
| domain-science | 8.5/10 | **3.5/10** | — | — | **3.5/10** |
| circuit-theory | — | — | 4/10 | — | — |
| circuit-exp | — | — | 3.5/10 | — | — |
| pulsed-power | — | — | 3.5/10 | — | — |
| vv-engineer | 8-9/10 | — | — | 0/10 | **2-3.5/10** |
| mhd-theory | 8.5-8.9/10 | 7.0-7.5/10 | — | — | 7.0-7.5/10 (physics realism) |
| numerics-fv | Production-quality | — | — | — | — |
| numerics-stability | Verified correct | — | — | — | — |
| diagnostics | 8.5/10 | **3.5/10** | — | — | 3.5/10 |
| hpc-engineer | — | — | — | — | Metal+cylindrical = production path |
| architect | 7.5/10 Metal | — | 3/10 | 0/10 | **0/10 validated, ~4/10 framework** |
| godot-viz | — | — | — | — | ~478 LOC to full diagnostic viz |
| **Project self-assessment** | **8.9/10** | **8.9/10** | — | — | **8.9/10** (RETRACTED) |

### Backend-Specific Fidelity (plasma-physics final assessment)

| Backend | MHD Numerics | Physics Coupling | Transport/EOS | Circuit | V&V | **Overall DPF** |
|---------|-------------|-----------------|---------------|---------|-----|----------------|
| Python | 5.5/10 | 6.0/10 | 5.0/10 | 4.5/10 | 3.0/10 | **4.5/10** |
| Metal | 7.5/10 | 3.0/10 | 2.0/10 | 4.5/10 | 4.0/10 | **4.0/10** |
| Athena++ | 8.0/10 | 5.0/10 | 4.0/10 | 3.0/10 | 5.0/10 | **5.0/10** |
| AthenaK | 7.5/10 | 2.0/10 | 2.0/10 | 2.0/10 | 3.0/10 | **3.0/10** |

**No single backend scores above 5.0/10 for the full DPF problem.**

### Resolved Items

| Topic | Resolution | Agents |
|-------|-----------|--------|
| "8.9/10 fidelity" | **RETRACTED**. Replace with dual rating: Numerics 8.5-8.9/10, DPF Physics 3.5-4.5/10 | All 17 agents |
| dL_dt status | IS computed from step 2+, but Lp/dL_dt inconsistent | pulsed-power, circuit-theory (corrected), circuit-exp |
| Python engine fate | DEMOTE to teaching/reference. Metal = production. | architect, hpc-engineer, plasma-physics |
| Production engine | Metal GPU + cylindrical geometry source terms | architect, hpc-engineer, plasma-physics |
| 3D viz framework | Canvas 2D first (not Three.js, not Godot). Three.js deferred to Phase 3+. | godot-viz, architect, hpc-engineer |
| Production 2D path | Option D+A: Athena++ cylindrical (short-term) + Metal cylindrical (medium-term) | hpc-engineer |
| Lee model role | Validation target, NOT circuit replacement. Code exists at validation/lee_model_comparison.py | circuit-theory, domain-science |
| Reproducibility | Statistical (L1 < 15%), not bitwise | hpc-engineer, vv-engineer |
| Subcycling | Circuit should be subcycled N_sub=10-50, adaptive during pinch | circuit-theory |
| PIC module | Keep dormant. Lee model + beam_target.py gives 80% benefit for 5% effort. | kinetic-verifier |

### Unresolved (Phase 2)

| Topic | Positions | Agents |
|-------|-----------|--------|
| Minimum circuit fix scope | Option A (10 LOC, 4/10) vs Option C (35 LOC, 6/10) | circuit-theory |
| Operator splitting upgrade | Strang (O(dt²)) vs Lie (O(dt)) — how much accuracy gain? | numerics-stability |
| L_plasma estimation method | Volume-integral vs snowplow analytical | pulsed-power, circuit-exp |
| Coronal eq. correction | Wire CR ionization to line radiation or accept overestimate | neutron-rad, diagnostics |

---

## REVISED MASTER PRIORITY LIST (Synthesized from all 17 agents, Round 3 Final)

**TIER 0 — STOP THE BLEEDING (1-2 days)**
1. Fix BREM_COEFF: 1.69e-32 → 1.569e-40 + fix line_radiation.py:409 (neutron-rad, mhd-theory)
2. Fix NX2 preset: C=28µF, R₀=2.3mΩ, anode=9.5mm, cathode=32mm (circuit-theory, circuit-exp, diagnostics)
3. Fix PF-1000 preset: L₀=33.5nH, R₀≈6mΩ (circuit-theory, circuit-exp, domain-science)
4. Fix Nernst grid spacing bug — add dx to np.gradient (kinetic-verifier)
5. Fix radiation ne: m_p→m_d (kinetic-verifier)
6. Fix binary WebSocket handler (godot-viz, ~28 LOC)

**TIER 1 — ENABLE VALIDATION (1-2 weeks)**
7. Fix grid domains: nr×dr ≥ cathode_radius for all cylindrical presets (circuit-exp, domain-science)
8. Fix dL_dt/Lp consistency in engine.py + fix R_plasma double-counting (pulsed-power, circuit-exp)
9. Wire ADI implicit diffusion to Metal resistive MHD path (hpc-engineer, 50-100× speedup)
10. Add BackendCapabilities + fail-loud for missing physics (architect, plasma-physics)
11. Add 3 MMS tests (~300 LOC: density advection, MHD wave, resistive diffusion) (vv-engineer)
12. Lee Model validation comparison — run with corrected presets (domain-science, circuit-theory)

**TIER 2 — PRODUCTION QUALITY (2-4 weeks)**
13. Add Metal HLLD double-star states (~100 LOC) (vv-engineer, mhd-theory, fv-verifier)
14. Metal cylindrical geometry source terms (~100 LOC, Mignone 2007) (architect, hpc-engineer)
15. Metal circuit coupling (port R_plasma, L_plasma, dL_dt to PyTorch) (architect)
16. MHDState conservative dataclass + DeviceState wrapper (architect)
17. Wire beam-target neutron yield into engine loop (~20 LOC) (kinetic-verifier, neutron-rad)
18. 2D pseudocolor visualization + Diagnostics tab (~478 LOC) (godot-viz)
19. Digitize PF-1000 I(t) from Scholz 2006 + NX2 from Lee & Saw 2008 (domain-science, vv-engineer)
20. Add resistive CFL enforcement (numerics-stability)

**TIER 3 — RESEARCH GRADE (4+ weeks)**
21. Metal two-temperature EOS (plasma-physics, 2 days)
22. Strang splitting for resistive/transport operators (numerics-stability)
23. Wire electrode ablation into continuity equation (neutron-rad)
24. Synthetic diagnostic suite — SXR(t), Abel→X-ray, neutron TOF (diagnostics)
25. Port Braginskii transport to Metal (plasma-physics, 5-7 days)
26. Spitzer Z-correction α(Z) + unify Coulomb log (kinetic-verifier, 10 LOC)
27. Fix Hall CFL: ne_max→ne_min (numerics-stability)

---

## PHASE 1 AUDIT STATISTICS

| Metric | Count |
|--------|-------|
| Total agents | 17 (7 principal + 9 verification + 1 WALRUS pending) |
| Dialogue rounds | 3 |
| Inter-agent messages (logged) | 80+ |
| Confirmed bugs | 16 (B1-B16) |
| Total deficiencies | 60+ findings across all agents |
| Architecture decisions | 8 (architect) + 9 milestones (M0-M7) |
| Cross-verification challenges | 19 confirmed, 0 disputed (diagnostics-verifier: 100% confirmation rate) |
| Agent self-corrections | 3 (circuit-theory dL_dt, pulsed-power 3 findings, kinetic-verifier B6) |
| Priority items | 27 (organized in 4 tiers) |
| Fidelity consensus | Numerics 8.5-8.9/10, DPF Physics 3.5-4.5/10, Validated 0/10 |

---

---

## LATE ROUND 3 — Critical Discoveries & Final Corrections

### B17: Solver Lp Formula Dimensionally Wrong by 66,000× (pulsed-power, circuit-theory-verifier)

**ROOT CAUSE BUG — Resolves the entire dL_dt debate.**

All three solver Lp formulas (`mhd_solver.py:2092-2098`, `cylindrical_mhd.py:638-640`, `metal_solver.py`) compute:
```python
Lp_est = mu_0 * B_theta_avg / I * dx * nx
# Units: [H/m] × [T/A] × [m] = [H²/m²] ≠ Henry
```

Has `μ₀` (1.26e-6) where it should have `(b-a)` (~0.0225 m for PF-1000). Produces Lp ~1.6e-13 H vs correct ~10 nH. The dL_dt from backward differences of this Lp is ~1e-5 of R_eff — **negligible**.

**The engine's** volume-integral L_plasma (engine.py:572-578) IS dimensionally correct (~10 nH), and overrides coupling.Lp. But the engine does NOT compute dL_dt from its own Lp — it uses the solver's broken value. Result: circuit is effectively free-ringing despite appearing coupled.

**circuit-theory-verifier was operationally correct** — dL_dt is effectively zero in practice, even though it's technically computed. pulsed-power revised rating to 3/10.

**Fix**: 5 lines in engine.py — store previous L_plasma, compute `dL_dt = (L_plasma - prev) / dt`, override `coupling.dL_dt`.

### Bremsstrahlung Fix Dependency Chain (kinetic-verifier + mhd-theory-verifier consensus)

**Fix order MATTERS — canceling errors can produce false validation:**

```
1. Fix BREM_COEFF → 1.569e-40 in BOTH bremsstrahlung.py AND line_radiation.py
2. Fix ne: m_p → m_d in radiation/transport.py
3. Add absolute-magnitude bremsstrahlung validation test (NRL reference)
4. THEN wire beam_target.py into engine
5. Full regression
```

**Rationale**: Buggy bremsstrahlung → Te crash → faster implosion → higher I_pinch → inflated beam-target yield that could accidentally "agree" with experiment. This is the most dangerous validation failure mode: correct answer for wrong reasons.

### Agent Self-Corrections (Late Round 3)

- **pulsed-power**: Revised circuit fidelity 4.5/10 → **3/10** after discovering Lp dimensional error. Acknowledges circuit-theory-verifier was operationally correct.
- **godot-viz**: Self-corrected bremsstrahlung analysis — original "108×" was incomplete (only caught Te unit mismatch, missed ne unit mismatch). Actual error is **10⁸×**.
- **numerics-fv-verifier**: Corrected F5 direction after plasma-physics challenge — Python HLL UNDERESTIMATES wave speeds (not overestimates). Upgraded to MODERATE.
- **architect**: Revised fidelity 4/10 → **3.0/10** after incorporating Lp dimensional error and HLLD→HLLC downgrade.

### numerics-fv-verifier — Definitive Coefficient Verification

**Metal WENO5-Z: TEXTBOOK-PERFECT** — Every coefficient verified against Lagrange interpolation. Ideal weights sum to 1. Combined polynomial matches unique 5-point interpolant. Smoothness indicators match Jiang-Shu 1996 Eq. 2.15. WENO-Z weights match Borges 2008 Eq. 25.

**Metal HLLD: CONFIRMED INCOMPLETE** — Missing SL*, SR* (M&K Eq. 51), U**_L/U**_R (Eqs. 59-62), F**_L/F**_R. Docstring at line 815 is misleading. Correctly classified as HLLC-MHD.

**Additional finding F7**: Metal SSP-RK operates on PRIMITIVE variables, not conservatives. Formally voids SSP guarantee. Low practical severity (many production codes do this).

**Revised fidelity after HLLD fix**: 8.5-9.0/10 (achievable with ~35-40 lines of code).

### Triple Confirmation: Metal HLLD is HLLC-MHD

| Agent | Method | Conclusion |
|-------|--------|------------|
| vv-engineer | Code review of metal_riemann.py:807-1014 | 4-region flux, missing double-star |
| mhd-theory-verifier | Equation-level verification vs M&K 2005 | Missing Eqs. 51, 59-62 |
| numerics-fv-verifier | Coefficient-by-coefficient audit | HLLC-MHD, not HLLD |
| numerics-stability-verifier | Cross-verification with Agent 12 | Confirmed, docstring misleading |

### Bremsstrahlung Exact Fix Locations (diagnostics-verifier, from neutron-rad-verifier)

| File | Line | Current | Correct |
|------|------|---------|---------|
| `src/dpf/radiation/bremsstrahlung.py` | 25 | `BREM_COEFF = 1.69e-32` | `BREM_COEFF = 1.569e-40` |
| `src/dpf/radiation/line_radiation.py` | 409 | `1.69e-32` (hardcoded copy) | `1.569e-40` |
| `src/dpf/radiation/transport.py` | 302 | `ne = Z * rho / m_p` | `ne = Z * rho / m_D` (3.344e-27 kg) |

**Test impact**: `test_brem_coefficient_matches_nrl` validates WRONG coefficient. Scaling tests (ne², √Te, Z²) remain valid.

### neutron-rad-verifier — Radiative Cooling Time Verification

With **CORRECT** coefficient (1.569e-40): τ_rad = 37.5 µs >> τ_pinch (100 ns) → bremsstrahlung is minor during pinch.
With **BUGGY** coefficient (1.69e-32): τ_rad = 0.35 ps << τ_pinch → plasma impossibly over-cools.

Cu line radiation at sheath (100 eV): τ_rad = 0.36 ns ≈ τ_pinch → SIGNIFICANT. Coronal equilibrium overestimates by 2-5× at sheath.

---

## FINAL CONFIRMED BUGS — MASTER LIST (Revised v3)

| # | Bug | Severity | Location | Found By | Error Magnitude |
|---|-----|----------|----------|----------|-----------------|
| **B1** | BREM_COEFF CGS→SI conversion missing | **CRITICAL** | bremsstrahlung.py:25, line_radiation.py:409 | neutron-rad, mhd-theory | **10⁸×** |
| **B2** | NX2 preset C wrong | **CRITICAL** | presets.py:79 | circuit-theory, circuit-exp, diagnostics | **31×** |
| **B17** | Solver Lp formula dimensionally wrong | **CRITICAL** | cylindrical_mhd.py:638, mhd_solver.py:2092, metal_solver.py | pulsed-power, circuit-theory | **66,000×** |
| **B3** | PF-1000 preset L₀ wrong | **HIGH** | presets.py:58 | circuit-theory, circuit-exp, domain-science | 2.2× |
| **B4** | Grid domains don't contain electrodes | **HIGH** | presets.py (all cylindrical) | circuit-exp, domain-science | Domain < geometry |
| **B5** | Nernst velocity missing dx | **HIGH** | nernst.py:169-173 | kinetic-verifier | 100-1000× |
| **B7** | Metal HLLD missing double-star states | **HIGH** | metal_riemann.py:807-1014 | vv-engineer, mhd-theory, fv-verifier, stability-verifier (4×) | Alfvén diffusion |
| **B6** | Radiation ne uses m_p not m_d | **MODERATE** | radiation/transport.py:302 | kinetic-verifier | ~2× |
| **B14** | R_plasma double-counting | **MODERATE** | engine.py ~570 | plasma-physics, pulsed-power | 5-15% |
| **B8** | Convergence tests measure wrong quantity | **MODERATE** | test_phase_o:901+ | numerics-stability | Unreliable order |
| **B9** | Resistive CFL not enforced | **MODERATE** | metal_solver.py, mhd_solver.py | numerics-stability | Blowup risk |
| **B10** | PF-1000 R₀ wrong by ~2× | **MODERATE** | presets.py:58 | circuit-exp (corrected) | ~2× |
| **B11** | dL_dt/Lp inconsistency | **MODERATE** | engine.py:588-589 | circuit-exp, pulsed-power | Mismatched estimators |
| **B12** | presets.py ↔ experimental.py mismatch | **MODERATE** | presets.py, experimental.py | diagnostics-verifier | Data integrity |
| **B13** | Hall CFL uses ne_max not ne_min | **MEDIUM** | mhd_solver.py:1314-1321 | numerics-stability | CFL underestimate |
| **B15** | Metal CFL max-over-dims | **LOW** | metal_solver.py:320-337 | fv-verifier, stability-verifier | 10% margin |
| **B16** | Cartesian B_theta proxy incorrect | **LOW** | mhd_solver.py:2093 | circuit-theory | Lp wrong in Cartesian |
| **B18** | HybridEngine validation compares vs t=0 | **MODERATE** | hybrid_engine.py:173-177 | walrus-verifier | Divergence check broken |
| **B19** | InverseDesigner constraints silently ignored | **MODERATE** | realtime_server.py:293-294 | walrus-verifier | /api/ai/inverse constraints no-op |
| **B20** | Dead code signature mismatch | **LOW** | surrogate.py:728-731 | walrus-verifier | Would crash if called |

**Total: 20 confirmed bugs. Top 3 by error magnitude: 10⁸× (brem), 66,000× (Lp), 31× (NX2 C).**

---

## ARCHITECT'S DEFINITIVE PHASE 2 PLAN (Final Revision)

### Overall Assessment: 3.0/10 Framework, 0/10 Validated

| Axis | Grade | Key Evidence |
|------|-------|-------------|
| Code verification | 4/10 | WENO5-Z correct, but Metal HLLD incomplete, Python non-conservative |
| Physics completeness | 2/10 | Bremsstrahlung 10⁸×, no beam-target, no ablation, no tabular EOS |
| Circuit coupling | 1/10 | Lp 66,000× wrong → dL_dt ≈ 0 → circuit free-ringing |
| Experimental validation | 0/10 | Zero comparisons to published data |
| **Overall** | **3.0/10** | min(4, 2, 1, 0) = 0 validated, 3.0 framework |

### Critical Bug Fix Priority (by error magnitude)

| # | Bug | Fix | Error |
|---|-----|-----|-------|
| 1 | Lp dimensional error | CouplingComputer class (5 lines engine.py) | 66,000× |
| 2 | BREM_COEFF CGS→SI | 1.69e-32 → 1.569e-40 (2 files) | 10⁸× |
| 3 | NX2 preset C | 0.9µF → 28µF | 31× |
| 4 | PF-1000 preset L₀ | 15nH → 33.5nH | 2.2× |
| 5 | Metal HLLD → true HLLD | Add double-star states (~35-40 LOC) | Alfvén diffusion |
| 6 | Binary WebSocket fix | websocket.ts:117 (~28 LOC) | Viz broken |
| 7 | Radiation ne m_p→m_d | 1 line | ~2× |
| 8 | R_plasma double-counting | Remove from circuit R_eff | 5-15% |
| 9 | Nernst missing dx | Add dx to np.gradient | 100-1000× |
| 10 | coupling.emf always 0 | Dead field | Back-EMF missing |

### Milestone Plan (M0-M7, 14 weeks critical path)

- **M0** (2 days): Fix bugs #1-4, #7, #9-10, wire ADI implicit diffusion
- **M1** (8 days): MHDState, DeviceState, BackendCapabilities, PhysicsContext
- **M2** (5 days): CouplingComputer, correct dL_dt, R_plasma fix, current dip verification
- **M3** (18 days): Physics porting to Metal (beam-target, Spitzer, two-temp EOS, cylindrical, true HLLD)
- **M4** (12 days): Visualization pipeline (2D → binary WS → Three.js)
- **M5** (15 days): Experimental validation — THE MOMENT OF TRUTH
- **M6** (5 days): V&V documentation (theory manual, ASME V&V 20 report)
- **M7** (5 days): Production hardening (CI guards, frozen baselines, NaN removal)

### 8 Architecture Decisions (LOCKED)

1. Three.js (actually Canvas 2D first)
2. Conservative MHDState frozen dataclass
3. DeviceState wrapper
4. BackendCapabilities Flag enum
5. PhysicsContext dataclass
6. Observer pattern for diagnostics
7. Separate /ws/viz/ WebSocket
8. CouplingComputer class (two Lp estimators with cross-check)

---

## PHASE 1 FINAL STATISTICS

| Metric | Count |
|--------|-------|
| Total agents | 17 (7 principal + 9 verification + 1 WALRUS pending) |
| Dialogue rounds | 3 + late Round 3 addenda |
| Inter-agent messages logged | 100+ |
| Confirmed bugs | **17** (B1-B17) |
| Total deficiencies | 65+ findings across all agents |
| Architecture decisions | 8 + 8 milestones (M0-M7) |
| Cross-verification challenges | 19/19 confirmed (100% rate) |
| Agent self-corrections | **5** (circuit-theory dL_dt, pulsed-power ×2, godot-viz brem, fv-verifier F5) |
| Priority items | 27+ (organized in 4 tiers) |
| Top 3 bugs by magnitude | 10⁸× (brem), 66,000× (Lp), 31× (NX2 C) |
| Fidelity consensus | Numerics 8.5-8.9/10, DPF Framework 3.0/10, Validated 0/10 |
| Critical path to validation | ~14 weeks (M0→M1→M2→M3→M5) |

---

---

## FINAL CONFIRMATIONS (Post-Round 3)

### Executive Framing (vv-engineer consensus)

> **"World-class MHD engine wrapped in broken DPF physics configuration."**

- MHD solver numerics: **8.0/10** (domain-science, numerics-fv-verifier, vv-engineer agree)
- DPF physics configuration: **3.0/10** (bottlenecked by configuration errors, not numerical methods)
- Overall system: **3.0/10**

### domain-science — Lp Dimensional Error Independently Verified

domain-science confirmed pulsed-power's Lp finding: extra `μ₀` factor makes Lp ~10⁵ too small. Circuit never sees meaningful current dip. Code is effectively free-ringing.

### plasma-physics — Lp Dimensional Error Triple-Confirmed, Circuit TRIPLY BROKEN

plasma-physics independently verified: code gives ~2.5 pH vs correct ~78 nH (~31,000× too small). Confirmed the formula error exists in ALL THREE solvers (cylindrical_mhd.py:640, mhd_solver.py:2095, metal_solver.py:892).

**Circuit-plasma coupling is TRIPLY BROKEN:**
1. L_p ≈ 0 (dimensionally wrong formula, B17)
2. EMF = 0 (coupling.emf never populated)
3. R_plasma double-counted (circuit + MHD both drain, B14)

The circuit is completely free-ringing. No plasma feedback exists in any backend.

plasma-physics revised fidelity to **3.0/10** (concurring with architect, domain-science, vv-engineer). Final deficiency count: **7 Critical, 9 Major, 6 Minor (22 total)**.

### Fidelity Consensus — FINAL (4 agents at 3.0/10)

| Agent | Final Rating | Notes |
|-------|-------------|-------|
| architect | 3.0/10 | "0/10 validated, 3.0 framework" |
| plasma-physics | 3.0/10 | "Triply broken circuit coupling" |
| domain-science | 3.0/10 | "Configuration fixes, not rewrites" |
| vv-engineer | 3.0/10 | "World-class MHD engine, broken DPF config" |
| mhd-theory-verifier | 3.0/10 coupled | "8.5-8.9 numerics, 3.0 coupled DPF" |

### mhd-theory-verifier — Final Report, Lp Error Confirmed as C7

Revised coupled DPF rating to **3.0/10**. Confirms B17 (Lp dimensional error) in all 3 solvers. Final findings: C1 (non-conservative pressure), C2 (Metal HLLD incomplete), C3 (WENO5 boundary), C4 (bremsstrahlung 10⁸×), C7 (Lp dimensional). Recommended fix chain: C4→S3→S1→validation + bremsstrahlung ON/OFF sensitivity test.

### architect — M0 Scope Finalized with All 3 Solver Files

CouplingComputer fix must touch all 3 solver files (15-30 LOC total). Accepts "triply broken" characterization. No further Phase 2 plan changes.

### walrus-verifier (Agent 16) — FINAL REPORT (Last Agent to Report)

**WALRUS integration is GENUINE — not a stub.** walrus-verifier loaded the real 1.29B-parameter IsotropicModel, ran actual inference (95s/step CPU, 16³ grid), and verified the complete pipeline (RevIN → forward → denormalize → residual). All 12 WALRUS verification tests pass (507s).

**However, 3 bugs found + critical physics issues:**

**B18 (MODERATE): HybridEngine validation is BROKEN** (`hybrid_engine.py:173-177`):
Creates a FRESH SimulationEngine every validation check → compares surrogate prediction at step N against initial conditions (t=0), NOT physics state at time N. Validation divergence check is meaningless.

**B19 (MODERATE): InverseDesigner constraints silently ignored** (`realtime_server.py:293-294`):
`constraints` parameter (metric bounds) is used to update parameter ranges instead, then NOT passed to `find_config()`. The `/api/ai/inverse` endpoint's constraints argument does nothing.

**B20 (LOW): Dead code signature mismatch** (`surrogate.py:728-731`):
`_states_to_tensor()` passes string `field_name` to `dpf_scalar_to_well()` which expects `traj_idx: int`. Would crash at runtime but is dead code (never called by active inference path).

**Critical physics issue: WALRUS produces negative density/pressure/temperature** on DPF-like input:
- `rho min = -8.76`, `pressure min = -253.06`, `Te min = -4.10`
- Root cause: pretrained on 16+ diverse PDE datasets, NEVER fine-tuned on DPF data
- No post-prediction physics constraints (density/pressure/temperature floors)

**WALRUS subsystem fidelity: 4/10 physics accuracy, 8/10 code quality/architecture.**

**Challenges issued to**: architect (HybridEngine broken), plasma-physics (negative density), domain-science (no DPF training data), hpc-engineer (CPU 95s perf), vv-engineer (tests check structure not physics), kinetic-verifier (can't capture kinetic effects), godot-viz (output format correct).

### B17 Confirmation Count: 4 Independent Agents

| Agent | Method | Magnitude |
|-------|--------|-----------|
| pulsed-power | Dimensional analysis | ~66,000× |
| domain-science | Independent dimensional analysis | ~10⁵× |
| plasma-physics | Quantitative (2.5 pH vs 78 nH) | ~31,000× |
| mhd-theory-verifier | Cross-verified via plasma-physics | ~10⁴-10⁵× |

### Bug Cascade by Error Magnitude (domain-science, definitive)

| Bug | Location | Error Factor | Effect |
|-----|----------|-------------|--------|
| BREM_COEFF CGS→SI | bremsstrahlung.py:25 | **10⁸** | All radiation wrong, Te→floor |
| BREM_COEFF duplicate | line_radiation.py:409 | **10⁸** | Line radiation brem wrong |
| Lp dimensional error | mhd_solver.py:2095 | **10⁵** | No current dip, free-ringing |
| Beam-target unwired | engine.py | **10⁴-10⁵** | Neutron yield 10⁴× too low |
| NX2 capacitance | presets.py | **31×** | Wrong device entirely |
| Nernst gradient | nernst.py:170-172 | **100-1000×** | B-field transport corrupted |
| PF-1000 inductance | presets.py | **2.2×** | Peak current 35% too high |
| ne mass (m_p→m_d) | radiation module | **2×** | Opacity 2× wrong |

### Validation Sequence (vv-engineer + domain-science consensus)

```
V(-1): Fix BREM_COEFF CGS→SI (gating fix, must be first)
V0:    Fix grid domains (structural prerequisite)
V0.5:  Fix presets (NX2 C, PF-1000 L₀)
V1:    PF-1000 I(t) comparison (Scholz 2006)
V2:    NX2 I(t) comparison (Lee & Saw 2008)
V3:    Lee model benchmark (existing code, no experimental data needed)
V4:    Neutron yield comparison (published scaling laws)
V5:    Energy conservation (< 1% total drift)
```

### B7 Acceptance Criteria (vv-engineer + mhd-theory-verifier)

After adding Metal HLLD double-star states:
- **Quantitative**: L1(B_y) reduced ≥50% on Brio-Wu vs current HLLC-MHD
- **Qualitative**: 6 distinct wave structures in density at ≥400 cells (vs 5 currently)
- **Documentation**: Docstring at metal_riemann.py:820-826 corrected (classified as ASME V&V 20 code verification failure)

---

*End of Phase 1 Inter-Agent Dialogue. 17 agents, 3+ rounds, 100+ messages, 17 confirmed bugs, 65+ findings, 8 architecture decisions, 8 milestones planned, 5 agent self-corrections tracked. The "8.9/10 fidelity" claim is RETRACTED. Honest assessment: **"World-class MHD engine wrapped in broken DPF physics configuration" — 3.0/10 framework, 0/10 validated.** Top 3 bugs span 13 orders of magnitude (10⁸×, 10⁵×, 31×). Path to credible DPF simulator: fix 8 configuration bugs (most are 1-line fixes), then validate against published PF-1000/NX2 data. Critical path: ~14 weeks.*
