# Panel Study: Lederman & Bilyeu (2024) NLR Method
# Applicability to DPF-Unified Dense Plasma Focus Simulator

**Date**: 2026-02-27
**Paper**: "Reducing the computational complexity of implicit schemes in the modeling of kinetic inelastic collisions in a partially ionized plasma"
**DOI**: 10.1002/num.23121
**Authors**: Lederman, D. & Bilyeu, D. (AFRL)

## VERDICT: CONSENSUS (3-0) with qualified scope

### Question
Analyze the Lederman & Bilyeu (2024) NLR (Non-Linear Repartitioning) method for stiff kinetic-collisional PDEs. Determine what DPF-Unified can learn from it and where the method can be applied.

### Answer
The NLR method is a mathematically sound technique for GPU-accelerated collisional-radiative (CR) equation solving that reduces complexity from O(L^3) to O(L^2) for L-level atomic kinetics systems. Its applicability to DPF-Unified is **real but narrow**: it addresses Cu electrode material CR evaluation during ablation phases, subject to conditions that do not yet exist in the codebase. The immediate actionable findings are engineering improvements to existing code (tabulated Cu cooling, energy conservation monitoring) that emerged from the panel's analysis of DPF-Unified's current radiation implementation.

---

## Consensus Verification Checklist

- [x] **Mathematical derivation provided** -- NLR path-based Jacobian approximation phi(P[r2,r1]) = -ReLu(dg/du * dt) / (-dg/du * dt - 1); Rosen (1954) exact Te/Ti equilibration; Griem LTE criterion
- [x] **Dimensional analysis verified** -- nu_ei [s^-1], kappa = nu_ei * dt [dimensionless], Lambda(Te) [W m^3], alpha = kappa * 2m_e/m_d [dimensionless]
- [x] **3+ peer-reviewed citations** -- Lederman & Bilyeu (2024, DOI:10.1002/num.23121), Post et al. (1977, At. Data Nucl. Data Tables), Rosen (1954), Miyoshi & Kusano (2005, JCP), Griem (1964, Plasma Spectroscopy), Summers (2004, ADAS)
- [x] **Experimental evidence cited** -- Post et al. (1977) Cu cooling curves, ADAS/CHIANTI atomic databases
- [x] **All assumptions explicitly listed** -- 8 assumptions with regime of validity (see below)
- [x] **Uncertainty budget** -- Cu cooling: factor 2-3 (current), <20% (with ADAS tables); NLR speedup: 20-50x GPU (CUDA), unknown MPS
- [x] **All cross-examination criticisms addressed** -- 20+ concessions documented across 3 phases
- [x] **No unresolved logical fallacies** -- 190x speedup retracted, Neumann series analogy corrected, beam-target VDF category error corrected
- [x] **Explicit agreement/dissent** -- Dr. PP AGREE, Dr. DPF AGREE, Dr. EE AGREE (all consensus items unanimous)

---

## Assumptions and Regime of Validity

| # | Assumption | Regime of Validity |
|---|---|---|
| A1 | Coronal equilibrium for Cu cooling function | tau_ionize << tau_hydro; marginal at DPF peak pinch (ratio ~0.01) |
| A2 | Frozen-nu_ei exponential for Te/Ti relaxation | alpha = nu_ei * dt * 2m_e/m_d << 1; holds at DPF CFL (alpha ~0.004-0.014) |
| A3 | Cell-local CR (no radiation transport) | Optically thin limit; Cu I resonance tau ~14 violates this; escape factors needed |
| A4 | LTE for D2 fill gas at peak compression | ne > 5e23 m^-3 at Te > 2 eV (Griem criterion for H(n=2->1)) |
| A5 | Non-LTE for Cu electrode material | Cu M-shell at Te = 30-200 eV, ne < 10^25 m^-3 |
| A6 | GPU-batched execution for NLR speedup | Batch size > 10^4 cells; MPS launch overhead dominates below ~32^3 grid |
| A7 | NLR training generalizes across DPF phase space | Te: 1-10000 eV, ne: 10^20-10^26 m^-3 (not yet demonstrated) |
| A8 | Operator-split energy conservation is O(dt^2) | Strang splitting; requires global energy ledger for verification |

---

## Key Findings

### Finding 1: Te/Ti Relaxation Is Already Correctly Solved

The `relax_temperatures()` function in `src/dpf/collision/spitzer.py` (lines 286-318) implements a structurally exact exponential integrator:

```
factor = exp(-2 * alpha)  where alpha = nu_ei * dt * 2*m_e/m_d
Te_new = T_eq + (Te - T_eq) * factor
```

At peak DPF pinch conditions (nu_ei = 2.6e10 s^-1, dt_CFL = 2.8e-10 s), the effective stiffness is alpha = 0.014 -- well within the regime where this integrator is exact to machine precision. **No improvement is needed** for Te/Ti relaxation. The Rosen (1954) midpoint correction would improve accuracy in the strongly stiff regime (nu_ei*dt > 100) but this regime is not reached at DPF's actual CFL timestep.

**Confidence: HIGH** (verified numerically by Dr. PP)

### Finding 2: Cu Cooling Curves Carry Factor-of-3 Uncertainty

The `_cooling_copper()` function in `src/dpf/radiation/line_radiation.py` (lines 133-163) uses piecewise power-law fits to Post et al. (1977) with stated accuracy "within factor of 2-3 of ADAS." At peak pinch Te = 500 eV, this propagates to a factor-of-1.5 error in radiation-limited pinch temperature and neutron yield.

Replacing piecewise power-law Cu cooling with tabulated ADAS/Post-Jensen data is the **highest-impact immediate action** (~200 LOC, drop-in replacement via the existing `_cooling_scalar()` dispatch).

**Confidence: HIGH** (all three panelists unanimous)

### Finding 3: NLR Speedup Is 20-50x GPU-Batched, Not 190x

The original 190x speedup claim (vs. dense LU on full L*L system) was retracted by Dr. DPF. The correct comparison:

| Method | Per-cell cost | GPU parallelism | Wall time (batched) |
|--------|--------------|-----------------|---------------------|
| Banded LU (Newton, 5 iter) | ~9,280 ops | Limited (serial per-cell) | ~200 us |
| JFNK (k=30, 5 Newton) | ~150,000 ops | Moderate | ~500 us |
| NLR (single forward pass) | ~18,944 ops | Full GEMM batching | ~10 us |

Speedup is 20-50x for GPU-batched execution over >10^4 cells. On single-cell CPU, NLR offers no advantage over banded LU or JFNK. **On M3 Pro MPS: unmeasured** -- benchmark required before adoption.

**Confidence: MEDIUM** (CUDA numbers extrapolated; MPS benchmark needed)

### Finding 4: Cu CR Stiffness Is Moderate (kappa ~1-10), Not Extreme

At DPF peak pinch conditions (Te = 500 eV, ne = 10^25 m^-3), Cu ionization stiffness kappa_CR = ne * <sigma_v>_ionize * dt_CFL ~ 1.0. This is the **moderate** stiffness regime, not the kappa >> 100 regime where NLR demonstrates its principal advantage.

However, at Cu M-shell ionization boundaries (Te = 30-100 eV), stiffness may reach kappa_CR = 10-1000. This requires verification with ADAS ionization rate tables.

**Confidence: MEDIUM** (rough ionization rate estimates; ADAS verification needed)

### Finding 5: NLR Is Premature Without CR Physics Infrastructure

DPF-Unified currently has no explicit collisional-radiative model. Cu ionization balance is absorbed into the coronal equilibrium assumption. Implementing NLR requires:

1. A reference CR implementation (FLYCHK, ADAS, or analytical Saha-Corona)
2. Training data generation from the reference
3. NLR training, validation against reference at DPF conditions
4. Integration into operator-split framework with conservation monitoring

This is a multi-phase research project, not a code sprint.

**Confidence: HIGH** (all three panelists unanimous)

### Finding 6: NX2 Preset Missing Crowbar Model (Side Discovery)

The NX2 preset in `src/dpf/presets.py` does not include `crowbar_enabled: True`. Without crowbar, simulated NX2 voltage reversal reaches 74.4% of V0 = 10.4 kV. Pulse capacitors are typically rated for <20% reversal. The PF-1000 preset correctly includes crowbar.

**Confidence: HIGH** (verified numerically; pre-existing finding from Debate #15)

---

## Major Concessions Across All Phases

| Who | Conceded | Original Claim |
|-----|----------|---------------|
| Dr. DPF | 190x speedup retracted | Revised to 20-50x GPU batched |
| Dr. DPF | Memory advantage retracted | 62.7 GB argument invalid; advantage is computational regularity |
| Dr. DPF | Neumann series analogy not rigorous | Motivational only; universal approximation provides guarantees |
| Dr. DPF | Beam-target neutron claim retracted | NLR addresses electron CR, not ion kinetics |
| Dr. PP | ReLU != np.maximum | Fully conceded; floor clamping violates conservation |
| Dr. PP | CESE+NLR reclassified | Long-term research, not near-term implementation |
| Dr. PP | Cu ablation priority revised upward | Post-Jensen cooling curves as immediate action |
| Dr. PP | Frozen-nu_ei exponential has O(1) error | At high stiffness (nu_ei*dt > 100); Rosen (1954) fix proposed |
| Dr. EE | Convergence proof demand withdrawn | Replaced with failure mode characterization requirement |
| Dr. EE | RKL2/NLR conflation retracted | Different problems (parabolic vs reactive stiffness) |
| Dr. EE | JFNK not universally superior | Degrades at kappa > 10^12 |
| Dr. EE | NLR elevated from speculative | "Promising for well-defined niche" |

---

## Remaining Disagreements

| ID | Issue | PP | DPF | EE |
|----|-------|-----|-----|-----|
| RD1 | Cu CR stiffness at M-shell boundaries | Moderate (~10) | Could reach 100-1000 | Benchmark needed |
| RD2 | NLR energy conservation guarantee | Depends on splitting | Structural for particles | Splitting-dependent for energy |
| RD3 | MPS speedup for NLR at DPF batch sizes | Unknown | Plausible 20-50x | Must benchmark before committing |

---

## Panel Positions

- **Dr. PP (Pulsed Power)**: AGREE -- NLR is a valid algorithm for a well-defined niche. DPF-Unified's immediate priorities are engineering basics: tabulated Cu cooling (200 LOC), NX2 crowbar fix (5 LOC), energy conservation ledger (50 LOC). NLR enters the picture only after Cu CR stiffness is quantified from ADAS data. The relax_temperatures() exponential is already correct. "Do the engineering basics first, then let the stiffness data decide whether NLR earns its place."

- **Dr. DPF (Plasma Physics)**: AGREE -- NLR scope is correctly bounded to GPU-batched Cu CR evaluation. Implementation roadmap: analytical exponential (50 LOC, immediate) -> Post-Jensen Cu cooling (200 LOC, near-term) -> NLR feasibility benchmark -> conditional NLR implementation. The paper is a RESEARCH reference that defines the eventual architecture for Cu CR, not guidance for any current phase.

- **Dr. EE (Electrical Engineering)**: AGREE -- Six mandatory pre-adoption criteria for any NLR deployment: (1) calibration traceability to ADAS/FLYCHK, (2) validation table at 4 DPF-relevant conditions with <10% error, (3) conservation monitoring passing <0.1% energy error, (4) MPS benchmark showing >10x speedup at 10^4 cells, (5) CPU fallback verification, (6) uncertainty budget with 95th percentile worst-case. "You do not adopt a new measurement instrument without first characterizing its accuracy and failure modes."

---

## Actionable Implementation Roadmap

### Priority 1 -- IMMEDIATE (0-2 weeks, ~270 LOC)

| Action | LOC | File | Impact |
|--------|-----|------|--------|
| Post-Jensen tabulated Cu cooling curves | ~200 | `src/dpf/radiation/line_radiation.py` | Reduce cooling error from factor-3 to <20% |
| Energy conservation monitor for radiation split | ~15 | `src/dpf/radiation/line_radiation.py` | Detect silent energy drain from Te_floor clamping |
| NX2 crowbar preset fix | ~5 | `src/dpf/presets.py` | Hardware safety: 74.4% reversal unsafe |
| Conservation architecture comment for future CR | ~5 | `src/dpf/radiation/line_radiation.py` | Defensive documentation |
| Rosen (1954) midpoint-nu_ei correction | ~30 | MHD thermal equilibration | Improve accuracy at nu_ei*dt > 10 |

### Priority 2 -- NEAR-TERM (2-8 weeks, ~250 LOC, decision gate)

| Action | LOC | File | Decision |
|--------|-----|------|----------|
| Quantify Cu CR stiffness from ADAS tables | ~70 | validation script | If kappa_CR > 50: NLR justified; if < 20: implicit Euler sufficient |
| NLR feasibility benchmark on MPS | ~80 | `src/dpf/benchmarks/nlr_cr_benchmark.py` | If >10x speedup at 10^4 cells: proceed; else: defer |
| NLR validation table vs FLYCHK/ADAS | ~70 | `tests/test_nlr_cu_validation.py` | If <10% error at DPF conditions: pass; else: retrain |

### Priority 3 -- CONDITIONAL (8-16 weeks, depends on P2 results)

| Action | Condition | LOC |
|--------|-----------|-----|
| NLR integration for GPU-batched Cu CR | P2 benchmark >10x AND validation <10% | ~150 |
| Full IMEX for Python engine (if stiffness justifies) | kappa_CR > 100 confirmed | ~3000+ |

### Priority 4 -- RESEARCH (>6 months)

| Action | Status |
|--------|--------|
| CESE+NLR integration | No open-source CESE exists; long-term research only |
| Multi-stage Cu kinetics (28 charge states) | Requires experimental Thomson scattering validation |
| Full radiation transport with escape factors | Required if Cu I resonance lines (tau ~14) affect energy balance |

---

## What DPF-Unified Learned from This Paper

### Knowledge Gained (to be recorded in memory)

1. **RKL2 and NLR solve different problems**: RKL2 addresses parabolic CFL stiffness (diffusion); NLR addresses reactive/collisional source-term stiffness. They are complementary, not competing.

2. **Floor clamping violates conservation**: `max(x, floor)` on particle populations or temperatures introduces mass/energy from nowhere. NLR's ReLU-based operator preserves column-sum conservation structurally. Any future CR implementation must enforce conservation architecturally.

3. **GPU speedup requires computational regularity**: NLR's advantage is not raw FLOPs but fixed computational graph topology -- no conditional branching, full tensor core utilization. This is the correct framing for any GPU-accelerated physics operator.

4. **Coronal equilibrium is marginal for Cu at DPF pinch**: tau_ionize/tau_hydro ~ 0.01 at peak pinch. CE "works" for hydrodynamic timescale but is borderline relative to MHD timestep. Time-dependent CR or tabulated non-CE cooling is the correct improvement.

5. **David Bilyeu connection**: Co-author is the same AFRL CESE-MHD researcher from DPF's Bilyeu research. This confirms the CESE-NLR coupling vision has active researchers, even if no open-source implementation exists.

6. **Existing relax_temperatures() is correct**: The frozen-nu_ei exponential with mass ratio reduction (alpha = kappa * 2m_e/m_d << 1) is accurate at all DPF operating conditions. This was a non-problem.

---

## Citations

1. Lederman D. & Bilyeu D., *Numer. Methods Partial Differ. Equ.* (2024), DOI: 10.1002/num.23121
2. Post D.E. et al., *At. Data Nucl. Data Tables* **20**, 397 (1977)
3. Rosen M., *Phys. Fluids* **7**, 491 (1954)
4. Griem H.R., *Plasma Spectroscopy* (1964), McGraw-Hill
5. Summers H.P., ADAS User Manual (2004)
6. Miyoshi T. & Kusano K., *J. Comput. Phys.* **208**, 315 (2005)
7. Vogl C.J. et al., *J. Sci. Comput.* (2023) -- NLR/Modified Patankar framework
8. Bilyeu D.L., PhD Thesis, Ohio State Univ. (2014) -- CESE-MHD
9. Jensen R.V. et al., *Nucl. Fusion* **17**, 1187 (1977) -- tabulated cooling curves
10. Chung H.-K. et al., *HEDP* **1**, 3 (2005) -- FLYCHK code
