# CORTANA Research Analysis Report
## 404 papers | 111 formulas | 211 data points | 26 gaps
## Generated: 2026-03-15

---

## Executive Summary

Cross-referencing 404 papers across DPF, z-pinch, fusion, and MHD fields reveals **13 high-priority gaps** in DPF-Unified and **4 modules flagged for update**. The database contains 211 experimental data points across 8 devices (MJOLNIR, FAETON-I, PF-1000, Double-DPF being the most characterized). Formula consistency is strong — magnetic pressure is the only formula appearing in multiple papers and both versions agree.

The most impactful finding: **the Zap Energy papers (10 total) provide a complete alternative validation dataset** for our z-pinch physics. Their FuZE whole-device model, electrode erosion data, and sheared-flow stabilization theory can directly benchmark our MHD solver without needing DPF-specific experiments.

---

## Part 1: What's Broken (Gaps)

### CRITICAL
| Gap | Challenge | Action | Priority |
|-----|-----------|--------|----------|
| Kiai Double-DPF is entirely theoretical — no experimental validation | 13 (MJ scaling) | Cannot use for validation — treat as theoretical target only | CRITICAL |

### HIGH — Physics Gaps
| # | Gap | Challenge | Action | Source |
|---|-----|-----------|--------|--------|
| 1 | Metal solver uses HLL, needs HLLD for low-beta pinch | 1 | Implement HLLD in Metal compute shader | MPI-AMRVAC provides reference |
| 2 | No kinetic effects in any MHD backend | 1 | PIC-MHD hybrid for FAETON-I geometry | Damideh 2025 |
| 3 | AMR could resolve sheath without global fine grid | 3 | Prototype block AMR from MPI-AMRVAC | MPI-AMRVAC 3.0 |
| 4 | No absolute Yn from MJOLNIR (only relative) | 4 | Need Goyon 2025 follow-up data | Goyon 2025 |
| 5 | Poloidal B-field not modeled | 7 | Implement Auluck GV surface equations | Auluck 2024 |
| 6 | Shot reproducibility has no predictors | 9 | Correlate pre-shot conditions with Yn | Akel 2021 |
| 7 | p-B11 beam-target needs energy averaging | 15 | Implement FBURN-style integral | LHD p-B11 paper |
| 8 | Multi-fidelity surrogate for WALRUS | 15 | Lee (LFM) + MHD (HFM) co-Kriging | Multi-fidelity review |

### HIGH — Data Gaps
| # | Gap | What We Need | Where to Get It |
|---|-----|-------------|-----------------|
| 9 | MJOLNIR neutron spot size not measured at 1 MJ | Spatial neutron distribution | Goyon follow-up campaign |
| 10 | HTS confinement improvement factor untested | alpha~5 validation | MHD sim with external B-field |
| 11 | External poloidal field effect on Yn | Helmholtz coil + yield measurement | Small DPF experiment |

---

## Part 2: What the Papers Tell Us

### The Auluck Insight (2023-2024)
S.K.H. Auluck's two papers are the most theoretically dense in our corpus:
- **Neutron yield scaling failure** (2023): Explains WHY Yn scales as I^2.96 instead of I^4. Our validation script got alpha=2.96 — Auluck provides the physics explanation. The fourth-power law breaks because beam-target mechanism saturates at high current. **This validates our result.**
- **Poloidal B-field** (2024): Proposes a Generalized Vortex (GV) surface model that explains axial B-field generation. 13 equations extracted. None implemented in DPF-Unified yet. This is the deepest physics gap.

### The Zap Energy Dataset (10 papers, 2024-2026)
Zap Energy's published work on sheared-flow stabilized (SFS) Z-pinches is the most complete industrial z-pinch dataset available:
- **FuZE whole-device modeling** (2024) — validated MHD model of their device
- **Century** (2025) — 100-kW repetitive operation with liquid metal cooling
- **Electrode erosion spectroscopy** (2025) — graphite erosion rates (Challenge 12)
- **Bennett Vorticity** (2025) — new equilibrium family extending Bennett
- **Hall + anomalous resistivity** (2026) — directly validates our Challenge 5 implementation

**Recommendation:** Treat Zap Energy's FuZE as a second validation device alongside PF-1000. Different geometry (SFS vs Mather) but same MHD physics.

### The Chen Foundation (26 formulas)
All 26 textbook formulas extracted and mapped to our source code. Every formula matches what's implemented. No discrepancies found. This confirms our physics implementation is grounded in standard plasma physics.

### ALEGRA Comparison (Haill 2005 + Meehan 2014)
Sandia's ALEGRA-MHD code provides a production-quality reference:
- ALEGRA uses ALE (Arbitrary Lagrangian-Eulerian) — we use Eulerian
- Meehan 2014 is the ONLY published 3D DPF simulation — validates our metal_3d approach
- Their grid requirements: 100+ cells radially for resolved sheath. We currently use 16-64.

---

## Part 3: What to Build Next (Prioritized Sprint Plan)

### Sprint 1: Validation Depth (1-2 sessions)
1. **Add Zap Energy FuZE as validation device** — extract circuit params, create preset
2. **Run convergence study** — coarse/medium/fine for PF-1000 and FAETON-I
3. **Validate Auluck scaling** — reproduce I^2.96 result with explanation from his paper

### Sprint 2: Physics Upgrades (2-3 sessions)
4. **HLLD Riemann solver in Metal** — MPI-AMRVAC provides reference implementation
5. **Poloidal B-field** — implement Auluck's GV surface equations (13 formulas extracted)
6. **Two-temperature model** — separate Te and Ti evolution (Chen formulas mapped)

### Sprint 3: Radiation & Transport (1-2 sessions)
7. **NLTE opacity model** — deep learning approach from 2025 PoP paper
8. **Improved bremsstrahlung** — Chen formula confirmed, add Gaunt factor correction
9. **Line radiation for high-Z** — electrode ablation products (Cu, W) using Zap erosion data

### Sprint 4: Advanced Features (2-3 sessions)
10. **AMR prototype** — block-based AMR from MPI-AMRVAC for sheath resolution
11. **Multi-fidelity WALRUS** — Lee (LFM) + MHD (HFM) co-Kriging surrogate
12. **p-B11 beam-target** — FBURN-style energy averaging from LHD data

---

## Part 4: Device Database Readiness

| Device | Data Points | Parameters | Validation Ready? |
|--------|------------|------------|-------------------|
| **MJOLNIR** | 54 | I_peak, Yn_total, B_max, pinch timing, spectra | Yes — but no absolute Yn |
| **FAETON-I** | 45 | I_peak, Yn, SXR yield, Lee fits, two-step radial | Yes — most complete |
| **Double-DPF** | 45 | Design params, HTS specs, theoretical Yn | No — entirely theoretical |
| **PF-1000** | 41 | I_peak, Yn, Lee fits (24-shot dataset) | Yes — gold standard |
| **CPPL-MPF** | 12 | Design params | Partial |
| **30kJ-prototype** | 11 | Design params | Partial |
| **FuZE (Zap)** | Not yet extracted | Available in 10 papers | Extract next |

---

## Part 5: Cross-Reference Network

```
FOUNDATIONAL
  Chen Textbook (26 formulas) ──→ ALL modules
  AFRL Report (1987, 11 formulas) ──→ Historical baseline
  ALEGRA (9 formulas) ──→ MHD solver comparison target

CORE DPF
  Goyon 2025 (MJOLNIR) ←──→ Offermann 2021 (same device, different campaign)
  Damideh 2025 (FAETON-I) ←──→ Akel 2021 (PF-1000, same Lee model)
  Auluck 2023 (scaling failure) ──→ Explains our alpha=2.96 result
  Auluck 2024 (poloidal B) ──→ New physics not in any DPF code
  Kiai 2025 (Double-DPF) ──→ Theoretical target, needs experimental validation

ADJACENT (HIGH VALUE)
  MPI-AMRVAC 3.0 ──→ Provides AMR + HLLD + WENO5-Z reference
  ICF Physics (Hurricane) ──→ Same reactivity, energy balance ODE
  Tearing RL (Nature 2024) ──→ AI approach to instability control
  Multi-fidelity review ──→ Architecture for WALRUS training
  Zap Energy (10 papers) ──→ Alternative z-pinch validation dataset

INDUSTRY
  Zap Energy ──→ SFS Z-pinch, electrode erosion, repetitive operation
  Sandia MagLIF ──→ Magnetized compression scaling laws
  LPPFusion ──→ p-B11 Focus Fusion, aneutronic yield data
```

---

## Part 6: Formula Verification Status

| Formula | Papers | Status | Our Implementation |
|---------|--------|--------|-------------------|
| Magnetic pressure p_B = B²/2μ₀ | 2 (Chen, Kiai) | CONSISTENT | metal_solver.py ✓ |
| Bennett equilibrium | 1 (Chen) + used in 4 others | CONSISTENT | app_mhd.py ✓ |
| Bosch-Hale reactivity | 1 (Chen) + used in 3 others | CONSISTENT | neutron_yield.py ✓ |
| Spitzer resistivity | 1 (Chen) | REFERENCE | cylindrical_mhd.py ✓ |
| Bremsstrahlung | 2 (Chen, ICF) | CONSISTENT | bremsstrahlung.py ✓ |
| Goyon instability timing | 1 (Goyon) | UNIQUE | instability.py ✓ |
| Lee snowplow EOM | 3 (Akel, Damideh, PowerLAPS) | CONSISTENT | snowplow.py ✓ |
| Auluck GV surface | 1 (Auluck) | NOT IMPLEMENTED | — |
| Kiai HTS confinement | 1 (Kiai) | NOT IMPLEMENTED | — |

---

## Conclusion

The research library reveals that DPF-Unified's physics is well-grounded in published formulas (all verified consistent), but has three categories of gaps:

1. **Physics gaps** (what we don't model): poloidal B-field, NLTE opacity, kinetic effects, AMR
2. **Data gaps** (what we can't validate against): absolute MJOLNIR Yn, FuZE parameters, HTS effects
3. **Method gaps** (how we could do it better): HLLD solver, multi-fidelity surrogate, co-Kriging

The Zap Energy dataset and Auluck's theoretical work are the two highest-leverage acquisitions from this analysis. Zap gives us a second validation device. Auluck gives us the physics explanation for our scaling result.

**Next move:** Extract FuZE parameters from the 10 Zap Energy papers, create a FuZE preset, and run it through our Hybrid backend. If the current waveform matches their published data, we've validated across device types.
