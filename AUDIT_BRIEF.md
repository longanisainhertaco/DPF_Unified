# DPF-Unified Scientific Completeness Audit

**Date:** 2026-03-19
**Scope:** DoD vs. Papers vs. Code gap analysis, 2026 literature scan, multi-agent roundtable
**Methodology:** 6-agent panel (MHD, Radiation, CompSci, ML, Experimentalist, Cross-Domain)

---

## BLUF (Bottom Line Up Front)

**The simulator's numerical methods are state-of-the-art (WENO5-Z, SSP-RK3, constrained transport) but the physics fidelity is severely overrated.** Only 2 of 7 backends deliver distinct physics (Lee, metal_cylindrical); the other 5 are the Lee model with spatial decoration. Validation claims (4.6% I_peak error) are based on scalar metrics only -- no waveform shape comparison exists. The 8.9/10 fidelity rating is a numerical methods score, not a physics score; honest physics fidelity is ~3-4/10. Three high-impact capabilities sit dormant in the codebase (line radiation 748 LOC, FLD transport 363 LOC, AMR 562 LOC) requiring only config wiring to activate. The biggest development ROI is a differentiable Lee model in JAX (~500 LOC, 2 days) that would enable gradient-based inverse design -- a capability no existing DPF code offers.

---

## 1. DoD Gaps Summary

### Gap 1: No Waveform Shape Validation Requirement (CRITICAL)

**Paper source:** Akel 2021 (24-shot PF-1000), Damideh 2025 (FAETON-I), all experimental validation papers
**What's missing:** DoD-DPF-Physics requires I_peak <5% and t_peak <10% but has NO requirement for full waveform NRMSE, dI/dt comparison, or current dip depth/shape. These are the metrics the experimental community uses.
**Suggested DoD language:**
> **Must-Have:** Full current waveform NRMSE < 0.15 against digitized experimental I(t). dI/dt zero-crossing timing within 5% of measured phase transition times. Current dip depth within 20% of measured value (when current dip is observed).

### Gap 2: No Radiation Fidelity Requirement (CRITICAL)

**Paper source:** Bailey et al. (Z-machine opacity measurements), QMF suppression literature, ICF rad-MHD papers
**What's missing:** DoD has no acceptance criterion for radiation model completeness. Line radiation from electrode ablation (copper) dominates cooling at 100-1000 eV yet the line radiation module is dormant. The opacity crisis (30-400% theory/experiment disagreement) is unacknowledged.
**Suggested DoD language:**
> **Must-Have:** Radiation model must include bremsstrahlung AND line cooling for the dominant impurity species. Opacity source must be cited with stated uncertainty range. Optically-thick line transitions must use transport (FLD minimum), not optically-thin assumption.

### Gap 3: No Backend Coupling Parity Requirement (CRITICAL)

**Paper source:** Self-deception observation (2026-03-19)
**What's missing:** DoD treats all backends as interchangeable. 5/7 backends lack MHD-circuit coupling. No DoD item requires that backend X produce physically distinct results from backend Y.
**Suggested DoD language:**
> **Must-Have:** Every backend labeled "MHD" must couple plasma state to circuit via Lp feedback. Cross-backend parity test: two MHD backends on the same problem must agree on I(t) NRMSE < 0.10. Backends that are Lee-model-equivalent must be labeled "Lee" not "MHD."

### Gap 4: No Synthetic Diagnostic Validation Requirement (IMPORTANT)

**Paper source:** Scholz 2006 (PF-1000 interferometry), Paduch 2021 (SXR imaging)
**What's missing:** DoD requires simulation-vs-experiment comparison of I_peak/t_peak only. No requirement for synthetic interferograms, SXR emission, or neutron spectra comparison against published experimental images.
**Suggested DoD language:**
> **Should-Have:** At least one synthetic diagnostic (interferometry, SXR, or neutron TOF) validated against published experimental data for a reference device.

### Gap 5: No Convergence Order Verification for Production Problems (IMPORTANT)

**Paper source:** All numerical methods papers (Keppens 2023, AMRVAC)
**What's missing:** DoD requires convergence tests on smooth problems only. No requirement for convergence measurement on a DPF-relevant problem (converging shock, pinch compression).
**Suggested DoD language:**
> **Should-Have:** Grid convergence study on a DPF-relevant problem (cylindrical Sedov or converging shock) showing measured convergence order within 0.5 of theoretical order at 3+ resolutions.

---

## 2. Unwired Physics Summary

| Rank | Paper/Source | Key Contribution | Status | Module | Effort | Dependencies |
|------|-------------|-----------------|--------|--------|--------|-------------|
| **1** | Line radiation (Post 1977, multiple) | Cu/Ne/Ar/W cooling curves, dominant at 100-1000 eV | **Dormant** (748 LOC complete) | `radiation/line_radiation.py` | S (2 days) | Config flag only |
| **2** | FLD transport (Levermore-Pomraning) | Radiation transport for optically-thick lines | **Dormant** (363 LOC, Cartesian only) | `radiation/transport.py` | M (5 days) | Needs cylindrical geometry fix |
| **3** | Hall MHD (standard) | Ion-scale physics at d_i ~ sheath thickness | **Dormant** (code at cylindrical_mhd.py:389-397) | `fluid/cylindrical_mhd.py` | M (1-2 weeks) | Whistler CFL constraint needed |
| **4** | Auluck 2024 poloidal B-field | 13 equations, azimuthal current, first-ever implementation | **Partial** (496 LOC, PDE solver incomplete) | `experimental/poloidal_bfield.py` | L (2 weeks) | Gratton-Vargas PDE solver |
| **5** | Damideh 2025 two-step radial fitting | Re-strike current reduction (fcr -> fcr2) | **Absent** | `circuit/snowplow.py` | S (3 days) | Re-strike detection logic |
| **6** | AMR (MPI-AMRVAC / FLASH-style) | 5-20x resolution efficiency for sheath/pinch | **Dormant** (562 LOC, not integrated) | `experimental/amr/` | XL (4+ weeks) | Solver refactoring for non-uniform grids |
| **7** | IMEX time integration (Pareschi-Russo) | Eliminates resistive CFL bottleneck | **Absent** | New module needed | L (2-3 weeks) | Implicit diffusion solver exists |
| **8** | HLLD Riemann solver (Miyoshi-Kusano) | Resolves all 7 MHD wave families (vs HLL's 5) | **Absent** on Metal GPU | `metal/metal_solver.py` | M (1 week) | Compute shader implementation |
| **9** | CIV breakdown model (13 space physics papers) | Self-consistent ionization front for Phase 1 | **Dormant** (526 LOC, no tests) | `experimental/civ_breakdown.py` | M (1 week) | Test + validation |
| **10** | DT fusion cross-section (Bosch-Hale) | Only DD exists; no DT or p-11B | **Absent** | `diagnostics/neutron_yield.py` | S (2 days) | Bosch-Hale parametrization |

---

## 3. 2026 Methods Worth Adopting

### Method 1: Differentiable Lee Model in JAX (from TORAX / DeepMind)

**Source:** TORAX differentiable tokamak simulator (github.com/google-deepmind/torax), 2024-2026
**Innovation:** JAX auto-differentiation through plasma ODEs enables gradient-based parameter identification and sensitivity analysis. Transforms calibration from grid search to gradient descent.
**DPF application:** Rewrite the 6-ODE Lee model in JAX. `jax.grad(loss_fn)` gives exact gradients of any observable w.r.t. (fc, fm, V0, C, L0, P_fill). Full discharge in <1ms, 10,000 sweeps in 10 seconds.
**Integration:** New module, no refactoring. ~500 LOC.
**Priority:** **IMMEDIATE** -- highest (impact x feasibility) / effort of any item in this audit.

### Method 2: FNO Hybrid Surrogate (from microwave breakdown + Orszag-Tang work)

**Source:** arXiv:2509.05799 (hybrid FNO-plasma, 60x speedup); arXiv:2507.01388 (FNO on resistive MHD, 25x speedup)
**Innovation:** Replace only the expensive subsystem (resistive diffusion or field solve) with an FNO while keeping the hydrodynamics physics-based. 25-60x speedup with physical trustworthiness.
**DPF application:** Train a 10-50M param FNO on resistive diffusion solutions from the Metal solver. Plug into the MHD loop as the implicit diffusion step. Eliminates both the sub-cycling hack and the need for a full IMEX refactor.
**Integration:** Moderate refactor of diffusion step interface.
**Priority:** **NEXT PHASE** -- requires training data pipeline first.

### Method 3: Neural Operator for Regime Bifurcation (from drift-wave turbulence)

**Source:** arXiv:2603.05730 (neural operator transformers for bifurcating turbulence, March 2026)
**Innovation:** A single neural operator captures both quasi-steady states AND dynamic transitions between regimes. Validated on drift-wave turbulence with spontaneous L-H-like transitions.
**DPF application:** The DPF discharge has 3+ distinct regimes (axial, radial, pinch, disruption) with sharp transitions. A bifurcation-aware neural operator could predict transition timing from initial conditions -- the key quantity for neutron yield.
**Integration:** New module, requires DPF training data.
**Priority:** **NEXT PHASE** -- research-grade, but addresses the core prediction challenge.

### Method 4: Implicit Resistive Diffusion via Batched Thomas Algorithm

**Source:** Standard numerical methods; adopted by Athena++, PLUTO, MPI-AMRVAC in 2024-2026 updates
**Innovation:** Replace explicit sub-cycling with backward-Euler implicit solve using tridiagonal (Thomas) algorithm. Batched across perpendicular dimensions for GPU parallelism.
**DPF application:** Remove the N=20 sub-cycle cap in Metal solver. One implicit solve per MHD timestep instead of 20 explicit sub-steps. 10-50x speedup in the resistive phase.
**Integration:** ~200 LOC in `metal/metal_transport.py`. No architectural changes.
**Priority:** **IMMEDIATE** -- removes binding constraint on what physics the Metal solver can represent.

### Method 5: ML Heat Flux Closure (from kinetic plasma theory)

**Source:** PNAS 2025: FNO learns kinetic heat flux closure, reproducing nonlinear Landau damping exactly
**Innovation:** Replace ad-hoc Spitzer-Harm or flux-limited heat flux with an FNO-learned closure trained on Vlasov simulations. Generalizes via transfer learning.
**DPF application:** The DPF pinch column is marginally collisional (mfp ~ column radius). Standard MHD heat flux overestimates losses. An FNO closure could explain "anomalous" confinement times observed experimentally.
**Integration:** Research project (requires Vlasov training data for DPF parameters).
**Priority:** **BACKLOG** -- high novelty but significant R&D investment.

---

## 4. Cross-Disciplinary Opportunities

### Opportunity 1: ICF Convergence Physics -> DPF Instability Modeling

**Source field:** Inertial Confinement Fusion (NIF/LLNL, Bian et al. 2026, Bell-Plesset theory)
**Key findings:**
- 3D magnetic RTI behaves differently than 2D: weak parallel B *enhances* mixing, strong parallel B suppresses it (Bian 2026)
- Bell-Plesset convergence amplifies RTI growth by orders of magnitude in converging geometry (Li, Wu, Chen 2026)
- Nonlinear mode coupling preferentially channels energy into m=0 modes -- a selection rule absent from DPF instability models
**DPF-Unified module:** `ai/instability_detector.py` uses planar growth rates. Should include convergence-geometry corrections from Bell-Plesset theory.
**Why DPF missed it:** DPF simulations are overwhelmingly 2D; the 2D-vs-3D divergence in magnetic RTI is buried in ICF literature that DPF groups don't read.

### Opportunity 2: Tokamak RL Control -> DPF Shot Optimization

**Source field:** Tokamak plasma control (DeepMind/TCV, DIII-D reconstruction-free RL, TORAX differentiable sim)
**Key findings:**
- Soft Actor-Critic RL maps raw diagnostics to actuators at 4 kHz, no reconstruction needed (DIII-D, Nuclear Fusion 2026)
- Neural state-space models learn plasma dynamics from only 311 pulses (TCV, Nature Communications 2026)
- "Predict-first" methodology: predict outcome, fire shot, compare, update model -- validated at TCV
**DPF-Unified module:** `ai/inverse_design.py` currently uses black-box Bayesian optimization. Should use gradient descent through a differentiable Lee model, with RL for shot-to-shot optimization on rep-rated devices.
**Why DPF missed it:** DPF is single-shot; "control" means adjusting initial conditions between shots. The insight that shot-to-shot optimization IS reinforcement learning hasn't crossed the community boundary.

### Opportunity 3: Pulsed Power Engineering -> DPF Electrode Lifetime

**Source field:** Space propulsion (pulsed plasma thrusters), Zap Energy (Century Z-pinch)
**Key findings:**
- W-Cu 70/30 alloy electrodes: 0.06 ug/discharge erosion after 10,000 shots (PPT community, Vacuum 2026)
- Liquid metal electrodes eliminate erosion entirely -- 1,080 consecutive shots at 0.1 Hz (Zap Century, Fusion Sci. Tech. 2025)
- Copper vaporization dominates erosion (smooth pits); W resists via higher sublimation threshold
**DPF-Unified module:** `sheath/ablation.py` (241 LOC, dormant) models mass shedding but has no electrode material database.
**Why DPF missed it:** DPF electrode choice is historical (copper). The space propulsion community's systematic erosion studies at comparable energy densities never crossed over.

---

## 5. Recommended Next Sprint

Ordered by (impact x feasibility) / effort. Dependencies indicated.

| # | Task | Module | Effort | Depends On | Rationale |
|---|------|--------|--------|------------|-----------|
| **1** | Report full waveform NRMSE + dI/dt for all 10 validated devices | `calibration.py` | 1 day | None | Zero new code. Redefines what "validated" means. May reveal the 4.6% claim doesn't survive contact with waveform shape. |
| **2** | Activate line radiation (Cu/Ne/Ar cooling) in engine | `radiation/line_radiation.py`, `engine.py` | 2 days | None | 748 LOC dormant. Config flag + engine wiring. Cu cooling dominates at 100-1000 eV in pinch. |
| **3** | Build differentiable Lee model in JAX | New: `src/dpf/jax/lee_model.py` | 2 days | None | 6 ODEs, ~500 LOC. Enables gradient-based inverse design, 1000x faster sweeps, exact sensitivity analysis. No existing DPF code has this. |
| **4** | Implement implicit resistive diffusion in Metal solver | `metal/metal_transport.py` | 2 days | None | Batched Thomas algorithm, ~200 LOC. Removes the N=20 sub-cycle cap that limits what physics the solver can represent. |
| **5** | Abstract circuit-MHD coupling into shared `CircuitCoupler` class | New: `circuit/coupler.py`, modify `engine.py` | 5 days | None | Fixes the root cause of 5/7 backends being Lee-with-decoration. Extracts density-weighted Lp from metal_cylindrical into all MHD backends. |
| **6** | Investigate NX2 and PF-1000-20kV failures to root cause | `calibration.py`, device presets | 3 days | #1 | These failures contain diagnostic information. NX2 likely needs per-device snowplow params. PF-1000-20kV likely has different crowbar timing. |
| **7** | Fix cylindrical geometry in FLD transport module | `radiation/transport.py` | 3 days | #2 | Replace Cartesian `np.gradient` with cylindrical `(1/r)*d(r*D*dE/dr)/dr`. Required before FLD activation. |
| **8** | Label Babylon.js renderer as "Lee schematic" or wire to MHD field data | `app_3d.py`, Babylon.js client | 3 days | #5 | 95% synthetic visualization undermines credibility. Either connect to real MHD data or label honestly. |
| **9** | Correct MJOLNIR R0 from 1.4 mOhm to 12.5 mOhm (Offermann 2021) | Device preset | 0.5 days | None | 9x resistance under-estimate causes underdamped circuit. Known since paper extraction. |
| **10** | Activate Hall MHD (expose existing code, add whistler CFL) | `fluid/cylindrical_mhd.py` | 1-2 weeks | #4 | d_i ~ sheath thickness at pinch conditions. Code exists at lines 389-397 but not exposed. |

**Sprint total:** ~3-4 weeks for all 10 items. Items 1-4 are parallelizable (4 independent 2-day tasks). Items 5-10 are sequential or have soft dependencies.

---

## 6. Multi-Agent Roundtable: Key Disagreements and Resolutions

### Disagreement 1: Numerics vs. Physics Priority

**MHD Specialist:** Fix circuit coupling first (affects global I(t) driver).
**Experimentalist:** Run waveform validation first (may reveal the circuit coupling is already adequate for Lee-coupled devices).
**Resolution:** Experimentalist wins. Waveform NRMSE costs 1 day and produces data that informs whether coupling refactoring is urgent or cosmetic. Measure before you optimize.

### Disagreement 2: WALRUS Strategy

**ML Researcher:** Abandon monolithic WALRUS surrogate. Use as learned correction operator over Lee model.
**CompSci:** Just use Athena++ AMR for production; surrogates are unnecessary.
**Resolution:** Both are partially right. Differentiable JAX Lee model (#3 above) is the immediate win. WALRUS becomes a correction network trained on (Lee residual, MHD truth) pairs. Athena++ AMR is the long-term production solver. Three-tier: JAX-Lee for design, WALRUS-correction for fast exploration, Athena++-AMR for publication runs.

### Disagreement 3: Radiation vs. Transport

**Radiation Physicist:** Activate line radiation immediately (2 days, huge Te impact).
**MHD Specialist:** Fix circuit coupling first (wrong I(t) cascades to everything).
**Resolution:** Both are 2-day efforts. Do them in parallel. Line radiation changes Te by up to 50% in the pinch; circuit coupling changes I(t) for 5/7 backends. Both are critical.

### Disagreement 4: AMR Now vs. Later

**Cross-Domain Integrator:** AMR is the highest-impact single capability (resolves 1mm pinch on 10cm domain).
**CompSci:** AMR on Metal/MPS is architecturally impossible (irregular memory, non-uniform tensors). Use Athena++ for AMR.
**Resolution:** CompSci wins on implementation. Don't write AMR for Metal. Integrate AMR through Athena++ cylindrical solver with DPF problem generator. Metal solver stays uniform-grid with implicit diffusion for speed.

### Consensus Statement

All 6 perspectives agree on one thing: **the project's biggest risk is self-deception about what "validated" means.** Scalar I_peak comparison with 3+ free parameters is calibration, not validation. The next sprint must start with honest waveform-level metrics before any new physics is added.

---

## 7. State of Adjacent Fields: Communication Gaps

### Fields That Should Be Talking to DPF But Aren't

| Field A | Field B | Shared Physics | Why They Don't Talk | Bridge Opportunity |
|---------|---------|---------------|--------------------|--------------------|
| ICF (NIF) | DPF | Converging geometry RTI, radiation collapse, burn physics | ICF has $billions, DPF has $millions. ICF publishes in PRL/PoP; DPF in IEEE TPS. Different journals, different conferences. | Bell-Plesset convergence corrections for DPF instability growth rates. ICF opacity tables for DPF radiation. |
| Tokamak control | DPF optimization | RL for plasma parameter optimization, neural state-space models | DPF is single-shot (no real-time feedback). Tokamak community doesn't think DPF is relevant. | Shot-to-shot RL optimization. Differentiable simulation. Transfer learning from tokamak disruption prediction to DPF instability onset. |
| Space physics (MMS) | DPF reconnection | Magnetic reconnection, nonthermal particle acceleration, current sheets | DPF happens in a lab at 10^18 cm^-3; MMS measures in space at 0.1 cm^-3. Same dimensionless physics, 18 orders of magnitude different parameters. | MMS-validated PIC methodology for predicting DPF ion energy distributions from first principles. |
| Astrophysical jets | DPF pinch column | Kink instability cascades, magnetic tower formation, jet collimation | Astrophysicists study jets lasting millions of years; DPF pinches last 100 ns. Same MHD, different communities. | Kink -> RTI -> reconnection cascade model for post-pinch neutron emission timing. |
| Pulsed plasma thrusters | DPF electrodes | Electrode erosion at comparable energy densities, W-Cu alloy optimization | PPT community is aerospace engineering; DPF is plasma physics. No shared conferences or journals. | Quantitative erosion rate models, W-Cu 70/30 electrode material for DPF lifetime extension. |
| Wire array Z-pinch (MAGPIE) | DPF pinch | Stagnation physics, filament formation, trailing mass | Wire arrays are pre-formed; DPF sheath is self-generated. But stagnation = pinch. | Cross-device validation of pinch models without DPF-specific experiments. |
| FRC/SFS Z-pinch (Zap) | DPF stability | Velocity shear stabilization of m=0/m=1 modes | Zap patents SFS; DPF community doesn't read patent literature. Different IP environments. | Shumlak-Hartman stabilization criterion applied to DPF rundown phase velocity profile. |

### The Meta-Problem

The DPF community is small (~50 active research groups worldwide) and insular. It publishes primarily in IEEE Transactions on Plasma Science and the ICPF conference proceedings. Adjacent communities (ICF, tokamak, astrophysics) have 10-100x more researchers, bigger budgets, and have solved many problems DPF still struggles with -- but under different names, in different journals, with different vocabulary. "Rayleigh-Taylor instability in converging geometry" (ICF) is the same physics as "sausage instability of the imploding sheath" (DPF). "Anomalous transport" (tokamak) is the same as "current-driven turbulence" (DPF). The vocabulary barrier is the primary communication gap.

DPF-Unified's cross-discipline paper analysis (554 papers) is, to our knowledge, the most systematic attempt to bridge these silos for DPF simulation. The three unpublished correlations (velocity shear, CIV breakdown, MAGPIE stagnation) discovered in that analysis demonstrate the value of looking outside the DPF literature.

---

## Appendix A: Bugs Found During Audit

1. **`engine.py:1754`** -- Athena++ path hardcodes `back_emf = 0.0`. Discards motional EMF component. Severity: MEDIUM.
2. **`transport.py:189`** -- FLD uses Cartesian `np.gradient` in cylindrical geometry. Will produce wrong radiation transport when activated. Severity: MEDIUM (dormant code).
3. **Babylon.js `updateHeatmap()` is an empty stub** -- MHD 2D field arrays are encoded, sent to the renderer, then discarded. 95% of visualization is from Lee 0D scalars. Severity: HIGH (credibility).
4. **MJOLNIR preset R0 = 1.4 mOhm** vs. Offermann 2021 paper R0 = 12.5 mOhm. 9x under-estimate. Severity: HIGH (validation).
5. **`engine.py` REST API path** had same r_inner/HL conversion bug as app_mhd.py. Every cylindrical Metal sim through the API had 10^6x wrong magnetic forces. Severity: CRITICAL if API is used.

## Appendix B: Self-Deception Patterns Identified

1. **Metric optimization over capability** -- Capability scorecard designed to measure what was wanted true, not what is true (16/16 by excluding failures).
2. **Scalar validation as proxy for physics** -- 4.6% I_peak error presented as "validated" when no waveform shape comparison exists.
3. **Backend counting as feature** -- "9 backends" when 5 are the same model with different spatial decoration.
4. **Fidelity score conflation** -- 8.9/10 rates the numerical scheme, not the physics content. Physics fidelity is ~3-4/10.
5. **Dormant code as capability** -- "748 LOC line radiation" listed as feature when it has never been called in production.

---

*Generated by multi-agent audit panel. 6 specialist perspectives, 4 discussion rounds, 247 DoD criteria evaluated, 731 papers cross-referenced, 2026 literature scanned across 11 topic areas.*
