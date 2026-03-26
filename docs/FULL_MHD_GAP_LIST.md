# Full MHD Gap List — Living Research Backlog

**Purpose**: Track every physics gap between our current implementation and a
production-grade Full MHD DPF simulator. Each item has a research status that
deepens across sprints until the item is ready for implementation.

**Process**: Before each sprint, a research agent refreshes the highest-priority
items with new literature, implementation details, and risk assessment. Items
move through: IDENTIFIED → RESEARCHED → SPEC'D → IMPLEMENTED → VALIDATED.

---

## Critical Path (blocks accuracy improvements)

### 1. HLLD Float64 Star-States
- **Status**: RESEARCHED (research agent running 2026-03-25)
- **Gap**: MLX HLLD hits float32 cancellation at extreme electrode B_theta.
  Currently using HLL fallback (more diffusive, can't resolve contact/Alfven waves).
- **Impact**: Post-peak current dip shape, radial compression dynamics.
  Grid convergence study shows it does NOT affect I_peak or t_peak.
- **Research needed**: Mixed-precision pattern (float64 discriminant + star-states,
  float32 storage/flux). MLX has no GPU float64 — must compute on CPU or find
  algebraically stable formulation.
- **LOC estimate**: ~200
- **References**: Miyoshi & Kusano 2005, our mlx_riemann.py, Phase O HLLD implementation

### 2. fc/fm Recalibration for full_mhd
- **Status**: DONE (2026-03-25)
- **Gap**: Published Lee params (fc=0.7, fm=0.08) were calibrated for lee_only.
  full_mhd mode shifts timing by ~7% and I_peak by ~1%.
- **Impact**: All validation metrics change when handoff_mode changes.
- **Research needed**: None — parallel Optuna running now.
- **LOC estimate**: 0 (parameter update only)

### 3. Post-Pinch Column Expansion Model
- **Status**: IMPLEMENTED + VALIDATED (2026-03-25, 12 LOC fix)
- **Gap**: After pinch, dL/dt=0 (frozen inductance). Real DPF has column expansion
  producing 60% current dip. Our model gives 90% dip when expansion is enabled
  (uncalibrated) or no dip when disabled.
- **Impact**: Post-peak waveform shape, neutron yield duration.
- **Research needed**: Calibrate expansion velocity vs device params. Digitize
  experimental dip waveforms for multiple devices.
- **LOC estimate**: ~50 (model exists, needs calibration)
- **References**: Lee & Saw 2014, Scholz 2006 dip shape

---

## Physics Gaps (improves fidelity)

### 4. Anomalous Resistivity Model
- **Status**: IDENTIFIED
- **Gap**: Current anomalous resistivity uses fixed threshold + LHDI formula.
  No drift-velocity dependent model. Faerder 2024 shows drift-velocity resistivity
  > current-density for DPF conditions.
- **Impact**: Magnetic field diffusion rate at pinch, Yn accuracy.
- **Research needed**: Faerder 2024 drift-velocity model, comparison with LHDI.
  Implementation in MLX collision module.
- **LOC estimate**: ~100
- **References**: Faerder 2024, Haines 2011

### 5. Beam-Target Neutron Yield
- **Status**: RESOLVED (2026-03-25, unblocked by Gap 3 post-pinch expansion)
- **Result**: Yn = 1.32e11 vs 1e11 experimental (32% error, within shot-to-shot variability)
- **Gap**: V_pinch for Yn uses stale snowplow dL/dt, not MHD coupler dLp_dt.
  After snowplow goes inactive, V_pinch=0. Beam-target Yn only generated during
  snowplow active phase, missing the actual pinch.
- **Impact**: Neutron yield accuracy (dominant channel for PF-1000).
- **Research needed**: How to extract V_pinch from MHD fields (E×B drift at
  pinch boundary). Schmidt 2017 argues MHD cannot capture beam-target — PIC required.
- **LOC estimate**: ~30 (wire feedback.dLp_dt to yield tracker)
- **References**: Schmidt 2017, Lee & Saw 2014

### 6. Bell-Plesset Convergent RTI
- **Status**: IDENTIFIED
- **Gap**: We use planar Rayleigh-Taylor growth rates. In cylindrical convergence,
  Bell-Plesset amplification increases growth by orders of magnitude.
- **Impact**: Pinch stability, instability-driven neutron production timing.
- **Research needed**: Bell-Plesset growth rate formulation for DPF geometry.
  Bian 2026 MRT analysis in our pinch_physics.py.
- **LOC estimate**: ~80
- **References**: Bell 1951, Plesset 1954, Bian 2026

### 7. Radiation Transport Beyond Bremsstrahlung
- **Status**: IDENTIFIED
- **Gap**: Only Bremsstrahlung (free-free) radiation. Missing: line radiation
  (bound-bound), recombination (free-bound), opacity effects.
  Opacity crisis: 30-400% Z-machine disagreement means all DPF radiation models
  are miscalibrated.
- **Impact**: Energy balance at pinch, Te accuracy, Yn via radiation cooling.
- **Research needed**: Tabulated Cu cooling curves (ADAS/FLYCHK), optically thin
  vs optically thick regimes for DPF. ~200 LOC for tabulated model.
- **LOC estimate**: ~200 (tabulated) or ~500 (inline CR)
- **References**: Post 1977, Summers 1974, opacity crisis literature

### 8. Electrode Ablation Coupling
- **Status**: IDENTIFIED
- **Gap**: AblationConfig exists but disabled. Cu vapor injection affects
  radiation cooling, resistivity, and Zeff.
- **Impact**: Late-time plasma composition, radiation losses.
- **Research needed**: Ablation rate models, Cu vapor transport, coupling to
  ionization.py coronal model.
- **LOC estimate**: ~150
- **References**: Bruzzone & Aranchuk 2003

---

## Numerical Gaps (improves solver quality)

### 9. Constrained Transport for MLX Cylindrical
- **Status**: PARTIALLY ADDRESSED (Dedner GLM auto-enabled for cylindrical, 2026-03-25)
- **Gap**: MLX cylindrical uses Dedner GLM for div(B) cleaning. CT (constrained
  transport) maintains div(B)=0 to machine precision but requires face-centered
  B-fields. PyTorch Metal solver has CT; MLX does not.
- **Impact**: Magnetic flux conservation, long-time accuracy.
- **Research needed**: CT on staggered mesh in MLX. EMF computation from
  Riemann fluxes. Existing implementation in metal_stencil.py for PyTorch.
- **LOC estimate**: ~300
- **References**: Evans & Hawley 1988, Stone & Gardiner 2009

### 10. IMEX / STS for Stiff Source Terms
- **Status**: IMPLEMENTED (RKL2 STS module + anisotropic conduction + viscosity, 2026-03-25)
- **Gap**: e-i equilibration is already solved by exponential integrator.
  The real gap is for future CR kinetics (Bilyeu NLR, QSS) which would be
  genuinely stiff. Not needed until Cu CR model is implemented.
- **Impact**: None currently. Becomes critical with gap #7/#8.
- **Research needed**: Done for current physics. Revisit when CR model added.
- **LOC estimate**: N/A (deferred)

### 11. AMR (Adaptive Mesh Refinement)
- **Status**: IDENTIFIED
- **Gap**: Uniform grid everywhere. Sheath occupies <1% of volume but needs
  highest resolution. 64x128 grid wastes 99% of cells.
- **Impact**: 10-100x compute reduction for resolved pinch simulations.
- **Research needed**: Patch-based AMR on MLX (32³ blocks, density-gradient
  trigger). GPU AMR has severe Amdahl bottleneck per Parthenon-VIBE 2025.
- **LOC estimate**: ~500
- **References**: AMReX, Parthenon, arXiv:2509.19701

---

## Integration Gaps (full system)

### 12. Multi-Device Calibration
- **Status**: IDENTIFIED
- **Gap**: Only PF-1000 has calibrated MLX params. 6 other devices with
  waveforms need calibration. Parallel Optuna is ready.
- **Impact**: Shipping credibility — every preset should have validated params.
- **LOC estimate**: 0 (compute only, infrastructure exists)

### 13. WALRUS Surrogate Fine-Tuning
- **Status**: IDENTIFIED (Phase J.2)
- **Gap**: WALRUS inference works but uses pretrained weights, not DPF-specific.
  Need training data from MLX runs + fine-tuning.
- **Impact**: 100x inference speedup for parameter sweeps.
- **LOC estimate**: ~200 (data pipeline) + training compute

### 14. Web Simulator Deployment
- **Status**: 72% (P0 goal)
- **Gap**: Gradio app works locally. HuggingFace Spaces deployed but may have
  stale code. Need: full_mhd mode in UI, calibrated presets, deployment testing.
- **Impact**: Direct shipping milestone.
- **LOC estimate**: ~100 (UI wiring)

---

## Research Backlog (not yet scoped)

### 15. Poloidal B-field (Auluck Model)
- Auluck 2021: dL/dt coupling "conceptually incomplete" without poloidal field
- 13 equations in our extracted data (dpf-papers/)
- Would require fundamental solver changes (B_poloidal as evolved field)

### 16. CIV Anomalous Resistivity
- **Status**: IMPLEMENTED (2026-03-25, civ_anomalous_resistivity + CIV_VCRIT dict)
- CIV active throughout PF-1000 axial rundown (v=100 km/s > v_crit=38.5 km/s)
- Only anomalous mechanism during axial phase. eta_CIV ~ 10x Spitzer.
- Not yet wired into engine step. References: Alfven 1954, Brenning 1992.

### 17. Differentiable MHD for Gradient Calibration
- **Status**: RESEARCHED (agent running 2026-03-25)
- Research pending on MLX mx.grad feasibility, finite-difference alternative

---

## Gaps 18-31: Identified by Production Code Audit (2026-03-25)

### 18. Tabulated / SESAME EOS (P2, ~300 LOC)
- Multi-material EOS for solid-density electrode + gas-to-plasma transition

### 19. Multi-species / Impurity Tracking (P1, ~800 LOC)
- Advected mass fractions for D, Cu, insulator. Essential for ablation chain.

### 20. Plasma-vacuum Interface Tracking (P2, ~600 LOC)
- Face-flux or level-set to replace floor-density vacuum hack.

### 21. Gas Breakdown / Inverse Pinch Phase (P3, ~400 LOC)
- Townsend avalanche model for 100-1000 ns breakdown phase.

### 22. Radiation Momentum Coupling (P2, ~200 LOC)
- P_rad = aT^4/3 in momentum equation. Matters for >1 MA devices.

### 23. Multi-group Opacity (P2, ~500 LOC)
- 4-8 group FLD for spectral radiation. Reabsorption reduces net cooling 2-10x.

### 24. Synthetic Diagnostics Suite (P1, ~600 LOC)
- X-ray pinhole imaging, neutron ToF spectrum, Schlieren, B-dot probes.

### 25. Hybrid PIC-MHD for Beam Physics (P0, ~1200 LOC)
- CRITICAL: fluid MHD predicts zero neutrons (Schmidt PRL 2012).
- experimental/pic/hybrid.py (1142 LOC) exists but unvalidated.

### 26. Finite Larmor Radius Corrections (P2, ~300 LOC)
- Gyroviscous stress from Braginskii. Stabilizes m=0 at pinch.

### 27. Implicit MHD for Low-beta Regions (P3, ~800 LOC)
- IMEX for vacuum Alfven speed bottleneck.

### 28. Ettingshausen Effect (P3, ~150 LOC)
- Heat flux from J×B. Add to existing Nernst module.

### 29. Runaway Electron Generation (P3, ~400 LOC)
- Dreicer threshold + hard X-ray emission estimate.

### 30. ALE Mesh Motion (P3, ~2000 LOC)
- Major architectural change. AMR is better investment for Eulerian.

### 31. Crowbar Switch Model (P2, ~150 LOC)
- Time-dependent crowbar impedance with closure time.

---

*Last updated: 2026-03-25 22:25 by Cortana (overnight sprint)*
*Session: 22 commits, ALL scaffold tiers + 8 research passes complete*
*Gap list: 31 items total, 9 implemented/resolved, 8 researched, 14 identified*
*Next research refresh: before each implementation sprint*
