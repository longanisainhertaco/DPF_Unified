# PhD Debate #8 — Phase Z Assessment

**Date**: 2026-02-25
**Scope**: Back-EMF, Bennett equilibrium, magnetized Noh, B-field init, calibration benchmarks, neutron yield, WALRUS benchmarks
**Previous Score**: 6.5/10 (Debate #7)

## VERDICT: CONSENSUS (3-0) — 5.8/10

The discovery of a critical double circuit-step bug (D1) that was present in all prior assessments retroactively degrades the effective score. The bug explains the anomalous calibration results (fm=0.95 vs published fm∈[0.05,0.15]).

### Panel Positions
- **Dr. PP (Pulsed Power)**: AGREE 5.8/10 — "Double circuit step corrupts RLC integrator state. R_plasma=0 on first call means no resistive damping. This is the most impactful single bug remaining."
- **Dr. DPF (Plasma Physics)**: AGREE 5.8/10 — "Bennett equilibrium (Grade A) and magnetized Noh (Grade A-) are excellent. But the circuit-MHD coupling is fundamentally broken by the double step."
- **Dr. EE (Electrical Engineering)**: AGREE 5.8/10 — "The circuit solver's implicit midpoint method maintains energy conservation per call, but calling it twice with different R_plasma/L_p values per MHD step violates the coupling contract."

### Scoring Breakdown

| Category | Score | Change | Notes |
|----------|-------|--------|-------|
| MHD Numerics | 7.5/10 | — | Unchanged. WENO-Z, HLLD, SSP-RK3 remain solid. |
| Transport | 7.5/10 | — | Braginskii, Spitzer, anomalous resistivity. |
| Circuit | 4.0/10 | ↓2.0 | Double step bug. Was 6.0, now 4.0 due to D1. |
| DPF-Specific | 4.0/10 | ↑0.7 | Snowplow, calibration framework, cross-validation exist but calibration results invalid due to D1. |
| Validation | 3.5/10 | ↑1.2 | Bennett (A), Noh (A-), but I(t) waveform validation corrupted by D1. |
| AI/ML | 3.0/10 | — | WALRUS pipeline complete but normalization stats empty (C3). |
| Software | 7.0/10 | — | Good test coverage, CI gates. |

**Composite**: 0.20×7.5 + 0.15×7.5 + 0.15×4.0 + 0.20×4.0 + 0.15×3.5 + 0.05×3.0 + 0.10×7.0 = **5.8/10**

### CRITICAL BUG D1: Double Circuit Step (NEW)

**Location**: `src/dpf/engine.py` lines 649 and 817 (pre-fix)

The Python/Metal engine path calls `circuit.step(coupling, back_emf, dt)` **twice per MHD step**:

1. **First call (line 649)**: `CouplingState(Lp=L_p, dL_dt=dL_dt, R_plasma=0.0)` with field-based back_emf
   - R_plasma=0.0 → no resistive damping
   - Advances circuit time by `dt`

2. **Second call (line 817)**: `CouplingState(Lp=sp_L, dL_dt=sp_dL_dt, R_plasma=R_plasma)` with back_emf=0
   - Full physics: R_plasma from Spitzer+anomalous, L_plasma from snowplow
   - Advances circuit time by another `dt`

**Impact**: Circuit advances 2×dt per MHD step. The implicit midpoint integrator's state (current, voltage, time) is corrupted. This explains:
- **fm=0.95 calibration anomaly**: Optimizer compensates for doubled inductance rate by maximizing mass fraction
- **I(t) waveform validation failures**: Timing and amplitude systematically wrong

**Note**: The Athena++ path (`_step_athena`) correctly has only ONE circuit.step() call.

### Bug D1 Fix (Applied in Phase AA)

Removed the first `circuit.step()` call. The single remaining call (previously line 817) has the full physics:
- R_plasma from volume-integral of η|J|²/I²
- L_plasma from snowplow model (when active) or volume-integral ∫B²/μ₀ dV / I²
- back_emf from `_compute_back_emf(dt)` — motional EMF from resolved MHD fields (∫v×B·dl), which is the spatially-resolved contribution separate from the lumped I·dL/dt in R_star

Electrode BCs now use `self._coupling.current` (previous step's current) — standard explicit coupling.

### MODERATE BUG D2: Bosch-Hale Branch 2 Coefficients

**Location**: `src/dpf/diagnostics/neutron_yield.py` lines 90-93 (pre-fix)

Branch 2 D(d,p)T reused Branch 1 D(d,n)He3 C2-C7 coefficients, only changing C1. The correct coefficients from Bosch & Hale (1992) Table IV are:

| Coefficient | Branch 1 D(d,n)He3 | Branch 2 D(d,p)T | Was in code |
|-------------|--------------------|--------------------|-------------|
| C1 | 5.43360e-12 | 5.65718e-12 | 5.65718e-12 ✓ |
| C2 | 5.85778e-3 | 3.41267e-3 | 5.85778e-3 ✗ |
| C3 | 7.68222e-3 | 1.99167e-3 | 7.68222e-3 ✗ |
| C4 | 0.0 | 0.0 | 0.0 ✓ |
| C5 | -2.96400e-6 | 1.05060e-5 | -2.96400e-6 ✗ |
| C6 | 0.0 | 0.0 | 0.0 ✓ |
| C7 | 0.0 | 0.0 | 0.0 ✓ |

**Impact**: 2-8% error in DD reactivity at DPF temperatures (1-5 keV), growing to 10-24% above 10 keV. The θ parameter differs between branches due to different C2, C3, C5 values.

### Bug D2 Fix (Applied in Phase AA)

Branch 2 now computes its own θ₂ and ξ₂ using the correct D(d,p)T coefficients.

### Phase Z Achievements (Verified)

1. **Bennett Equilibrium** (Grade A): Force balance <1e-12, 26 tests, analytically exact
2. **Magnetized Noh** (Grade A-): 23 R-H jump conditions <1e-10, self-similar solution, 69 tests
3. **B-field Initialization**: `int()` → `round()` fix for ir_shock (Z.3)
4. **Calibration Benchmarks**: 3-term objective (1 DOF), crossval, Lee & Saw benchmark — but results invalid due to D1
5. **Neutron Yield**: Beam-target + thermonuclear, 19 tests
6. **WALRUS Benchmarks**: 24 tests, sliding-window physics validation

### Reclassified Bugs

- **C1 (back_emf=0.0)**: Reclassified from CRITICAL to DESIGN NOTE. Dr. DPF proved mathematically that for thin-sheath limit, ∫(-v×B)·dl = I·dL/dt, which is already included in R_star = R_eff + dLp_dt in rlc_solver.py. Setting back_emf=0 with dL/dt in the circuit equation is physically correct.

### Roadmap to 7.0/10

1. ✅ **Remove double circuit step** (D1) — 1-line fix, +0.5-0.8 projected
2. ⬜ **Re-run calibration** to verify fm returns to [0.05, 0.15] range
3. ✅ **Fix Bosch-Hale Branch 2** (D2) — correct coefficients applied
4. ⬜ **Add regression test**: Assert circuit.step() called exactly once per engine step

### References

- Bosch, H.-S. & Hale, G.M., "Improved formulas for fusion cross-sections and thermal reactivities," Nuclear Fusion 32:611 (1992). DOI: 10.1088/0029-5515/32/4/I07
- Lee, S. & Saw, S.H., "Plasma focus ion beam fluence and flux," Physics of Plasmas 19:112703 (2012)
- Lee, S. & Saw, S.H., "Pinch current limitation," Applied Physics Letters 92:021503 (2008)
- Miyoshi, T. & Kusano, K., "A multi-state HLL approximate Riemann solver for ideal MHD," Journal of Computational Physics 208:315 (2005)
