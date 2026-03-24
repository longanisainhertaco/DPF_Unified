# Metal Engine v2: Corrected Architecture Specification

**Project**: DPF-Unified Custom MHD Solver on Apple Metal
**Version**: 2.0-REVIEW (post-4-agent verification)
**Date**: 2026-03-24
**Status**: DRAFT — requires validation before implementation
**Process**: Six Sigma DMAIC + Swarm Methodology

---

## 0. Review Findings Incorporated

This spec incorporates corrections from 4 independent review agents (MHD physicist ×2, engine architect, validation engineer) that audited two prior architecture proposals. Every claim below has been verified or flagged.

| Prior Claim | Verdict | Correction Applied |
|-------------|---------|-------------------|
| Switching criterion `e_int/(E-KE-ME)` | FAIL — self-referential | Use `p_from_S / E` (entropy-derived, no subtraction) |
| Back-EMF = d/dt(Lp·I) | FAIL — double-counts | Use V_back = -I·dLp/dt; fix existing double-count bug |
| Lp from B² volume integral | FAIL — electrode artifacts | Use density-weighted Lp from CircuitCoupler |
| AOSOA layout | FAIL — CUDA concept | Use SoA (current layout, correct for Metal) |
| Characteristic WENO5-Z | FAIL — rejected Phase O | Drop; dual-energy solves the priority problem |
| "Maeder & Bouquet 2022" | FAIL — paper unverifiable | Removed |
| DISPATCH attribution | FAIL — wrong author | Corrected to Popovas (2025) |
| MLX `F.pad` equivalent | Gap identified | Write custom Metal kernel via `mx.fast.metal_kernel()` |
| Shock detector compression > 1.1 | NEEDS-REVISION | Use `div(v) < 0 AND Δp/p > threshold` |
| 3-week timeline | FAIL | Corrected to phased 10-12 weeks |
| Performance at 16×32 | FAIL — GPU overhead dominates | Target 128×512+ for GPU advantage; PyTorch patch validates physics first |

---

## 1. Problem Statement

### 1.1 Root Cause (Proven — 14-layer RCA + 5 research agents)

In DPF simulations at MA-level currents, the cathode boundary has ME/p > 10⁶. The standard pressure recovery:

```
p = (γ-1)(E - ½ρv² - ½B²)
```

loses all significant digits in float32 (7-digit mantissa). This cascades:
corrupted p → wrong Te → wrong Spitzer η → wrong R_plasma → shifted I(t) waveform.

### 1.2 Proven Solution

**Dual-energy formalism** with entropy tracer Sρ = ρ·p/ρ^γ. Avoids the subtraction entirely in both flux computation and pressure recovery. Standard in Enzo (Bryan+ 2014), FLASH (Lee 2013). Entropy-based variant from DISPATCH HLLS (Popovas 2025, A&A).

### 1.3 What This Spec Does NOT Cover

- Characteristic WENO5-Z decomposition (rejected — 500 LOC, marginal gain)
- AOSOA memory layout (wrong for Metal architecture)
- Well-balanced schemes (no verified citation, not critical for DPF)
- IMEX-RK2 (existing Thomas solver handles stiff terms; revisit if splitting error proven limiting)

---

## 2. Development Strategy: Two Phases

### Phase A: PyTorch Dual-Energy Patch (Week 1-2)
**Purpose**: Validate physics on existing 262-test suite before clean-room build.

- Add entropy tracer Sρ as 10th state component (infrastructure 80% exists: `IEE`, `e_electron` advection)
- Implement switching criterion using entropy-derived pressure
- Fix back-EMF double-counting bug (`rlc_solver.py:52` + `coupler.py:194`)
- Increase ghost cells 2→3
- Add 3 CRITICAL V&V tests
- **~400 LOC, 1 week, zero framework risk**
- **Exit criterion**: PF-1000 I_peak < 10%, no negative pressure at electrodes, all 262 tests pass

### Phase B: Clean-Room MLX Solver (Week 3-12)
**Purpose**: Purpose-built solver for production DPF grids (128×512+) where GPU wins.

- Built from scratch in MLX with custom Metal kernels
- Physics validated by Phase A reference implementation
- Every module cross-verified against Phase A PyTorch output
- **~4,200 LOC, 8-10 weeks**
- **Exit criterion**: Cross-backend parity with Phase A on all standard tests + PF-1000

---

## 3. Mathematical Framework

### 3.1 State Vector (10 components, SoA layout)

```
Q = [ρ, ρvr, ρvz, ρvθ, E, Sρ, Br, Bz, Bθ, ρee]
```

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | ρ | Mass density |
| 1-3 | ρvr, ρvz, ρvθ | Momentum density |
| 4 | E | Total energy (kinetic + thermal + magnetic) |
| 5 | Sρ = ρS | Entropy tracer: S = p/ρ^γ, so Sρ = p·ρ^(1-γ) |
| 6-8 | Br, Bz, Bθ | Magnetic field (Br, Bz face-centered for CT; Bθ cell-centered) |
| 9 | ρee | Electron internal energy (two-temperature model) |

### 3.2 Entropy Tracer Evolution

In smooth flow, Sρ is passively advected (DS/Dt = 0 for adiabatic flow):

```
∂(Sρ)/∂t + ∇·(Sρ v) = (γ-1) · S · Q_vol / p
```

where Q_vol = η·J² + Q_cond + Q_rad [W/m³] is the total irreversible volumetric heating rate.

**Implementation**: Sρ is advected as a passive scalar through the Riemann solver (upwind, using contact speed S_M for intermediate state selection). The source term is applied operator-split.

### 3.3 Pressure Recovery with Entropy-Based Switching

```python
def recover_pressure(Q, gamma=5.0/3.0, eta1=1e-3, eta2=0.1):
    rho, E, Srho = Q[0], Q[4], Q[5]
    KE = 0.5 * (Q[1]**2 + Q[2]**2 + Q[3]**2) / rho
    ME = 0.5 * (Q[6]**2 + Q[7]**2 + Q[8]**2)

    # Pressure from entropy (NEVER involves E - KE - ME subtraction)
    S = Srho / rho
    p_S = S * rho**gamma

    # Pressure from total energy (accurate when thermal energy dominates)
    p_E = (gamma - 1) * (E - KE - ME)

    # Switching criterion: use entropy-derived pressure as NUMERATOR
    # This avoids the circular problem of using the corrupted subtraction
    # in the criterion itself.
    eta = p_S / max(E, 1e-30)  # NOT (E - KE - ME) / E

    # Smooth blend (cubic Hermite) to prevent switching artifacts
    w = clamp((eta - eta1) / (eta2 - eta1), 0, 1)
    w = w * w * (3 - 2 * w)

    # w=0 (magnetically dominated): use p_S entirely
    # w=1 (thermally dominated): use p_E (conservative, accurate here)
    p = w * p_E + (1 - w) * p_S
    return max(p, p_floor)
```

**Key difference from prior proposals**: The switching criterion `eta = p_S / E` uses the entropy-derived pressure in the numerator, avoiding the corrupted `E - KE - ME` subtraction entirely. This was the CRITICAL flaw identified in both prior documents.

### 3.4 Shock Entropy Resynchronization

At shocks, entropy increases per the 2nd law. The advected Sρ does not capture this jump. After each RK stage, resync in cells where:

```
shock_detected = (div_v < -0.33 * c_s / dx) AND (|Δp| / p > 0.33)
```

In shocked cells where thermal energy is a significant fraction of total (β > 0.01):

```
Sρ_new = p_from_E / ρ^(γ-1)    where p_from_E = (γ-1)(E - KE - ME)
```

This subtraction is SAFE at shocks because ME is NOT dominant there (shocks convert KE/ME → thermal). The resync is performed per-substep without breaking SSP properties (Gottlieb, Shu & Tadmor 2001).

---

## 4. Algorithmic Pipeline

| Step | Method | Notes |
|------|--------|-------|
| Reconstruction | WENO5-Z (Borges 2008) | Component-wise. 5th-order smooth, PLM fallback at boundaries |
| Riemann solver | HLLD (Miyoshi & Kusano 2005) | Add Sρ to intermediate states (passive scalar). Keep NaN-safe LF fallback |
| Time integration | SSP-RK3 (Shu-Osher 1988) | 3 stages. Entropy sync after each stage |
| div(B) = 0 | 2D CT (Gardiner & Stone 2005) | Only Br, Bz need CT. Bθ is cell-centered (no div contribution in axisymmetry) |
| Resistive diffusion | Implicit Thomas solver | Operator-split. Already implemented and tested. Unconditionally stable |
| Conduction | Implicit Thomas solver | Braginskii κ∥. Same operator-split approach |
| Two-temp exchange | Implicit pointwise | νei exchange. Already implemented |

---

## 5. Circuit Coupling

### 5.1 Plasma Inductance: Density-Weighted Lp (CircuitCoupler)

**NOT B²-energy integral.** The density-weighted method from `coupler.py` is proven more robust:

```
z_sheath = argmax(∫ ρ·r dr)        # axial density peak
r_eff = ∫(r·ρ·dV) / ∫(ρ·dV)       # density-weighted radius
Lp = (μ₀/2π) · z_sheath · ln(b/r_eff)
```

Monotonicity enforced: Lp ≥ Lp_prev. BDF2 for dLp/dt. Back-EMF clamped ±50 kV.

### 5.2 Back-EMF (CORRECTED)

```
V_back = -I · dLp/dt
```

**NOT** d/dt(Lp·I). The Lp·dI/dt term is already on the LHS of the circuit equation.

**Existing bug to fix**: `rlc_solver.py:52` adds `dLp_dt` to `R_star`, AND `coupler.py:194` computes `back_emf = I * dLp_dt`. This double-counts. Fix: remove `dLp_dt` from `R_star` and let `back_emf` carry it alone, OR set `back_emf = 0` in coupler and let `R_star` handle it.

### 5.3 MLX Zero-Copy Coupling

Volume integrals for Lp, R_plasma computed directly on MLX GPU arrays. Scalar extraction via `mx.eval()` + `.item()` has ~20-50μs latency per extraction — acceptable for 2 scalars per timestep. Circuit solver runs in float64 on CPU via standard Python.

---

## 6. Framework: MLX + Custom Metal Kernels

### 6.1 Why MLX (for production grids)

| Property | Benefit for DPF |
|----------|----------------|
| True zero-copy unified memory | No CPU↔GPU transfer for circuit coupling |
| `mx.compile()` graph fusion | Reduces dispatch overhead for elementwise chains |
| `mx.fast.metal_kernel()` | Custom stencil kernels in MSL from Python |
| NumPy-compatible API | Mechanical port of reconstruction/Riemann logic |
| Lazy evaluation | Deferred execution optimizes GPU utilization |

### 6.2 Custom Metal Kernels (3 hotspots)

Written in MSL via `mx.fast.metal_kernel()`, JIT-compiled and cached:

1. **Ghost cell padding** (~50 LOC MSL): replicate + electrode BC fill. Replaces missing `F.pad`
2. **HLLD Riemann solver inner loop** (~200 LOC MSL): fused wave speed + flux computation
3. **Cylindrical geometric source terms** (~80 LOC MSL): r-weighted source with L'Hôpital at axis

All other operations use standard MLX ops with `mx.compile()`.

### 6.3 Memory Layout

**Structure-of-Arrays (SoA)**: `Q[var, r, z]` shape `(10, nr, nz)`. Each variable contiguous in memory. This is:
- Optimal for Metal GPU SIMD groups (threads access consecutive spatial indices)
- Compatible with MLX's row-major arrays
- The same layout used by Athena++, FLASH, and the current Metal solver

### 6.4 Precision Strategy

- All GPU computation: float32
- Entropy tracer provides accurate pressure without float64
- Circuit coupling scalars (Lp, R_plasma): extracted to CPU, promoted to float64
- Global energy integrals for diagnostics: compensated summation (Kahan) in float32

---

## 7. Coordinate System: Axisymmetric Cylindrical (r, z)

### 7.1 r-Weighted Finite Volumes

Cell volumes: V_ij = π(r²_{i+½} - r²_{i-½})·Δz
Radial face areas: A_r = 2πr·Δz
Axial face areas: A_z = π(r²_{i+½} - r²_{i-½})

### 7.2 Axis Singularity (r = 0)

- Inner cell boundary at r_min = Δr/2 (never at r = 0)
- L'Hôpital for geometric source: p/r → dp/dr at first active cell
- Reflecting BC: vr(r=0) = 0, Br(r=0) = 0, sign flip for θ-components

### 7.3 Electrode Boundary Conditions

- **Cathode (outer)**: Ghost cells with Bθ = μ₀I/(2πr), zero-gradient ρ/p, vr = 0, Br = 0
- **Axis (inner)**: Ghost cells with Bθ = 0 (axis symmetry), reflecting
- **Axial ends**: Reflecting (z=0 insulator) and outflow (z=L open)

---

## 8. Validation & Verification Plan

### 8.1 Phase A Gates (PyTorch patch — Week 1-2)

| Test | Criterion | Severity |
|------|-----------|----------|
| Float32 cancellation acceptance | Create cell ME/p=10⁶, verify p_from_S accurate to 0.1% | CRITICAL |
| Dual-energy conservation | 100+ steps at β<<1, total energy drift < 10⁻⁵ | CRITICAL |
| Electrode BC pressure positivity | PF-1000 at 2MA, no negative p after 50 steps | CRITICAL |
| Sod shock tube | L1(ρ) < 0.02 at N=256, dual-energy doesn't degrade | HIGH |
| Brio-Wu shock tube | All 7 MHD waves resolved, no switching artifacts | HIGH |
| Existing 262 Metal tests | All pass, zero tolerance relaxation | HIGH |
| PF-1000 I_peak | < 10% error vs 1.87 MA | HIGH |

### 8.2 Phase B Gates (MLX solver — Week 3-12)

| Stage | Tests | Gate |
|-------|-------|------|
| 1. Unit (W3-4) | Each module vs NumPy reference; Metal kernels vs MLX-op equivalents | All pass |
| 2. 1D shocks (W5) | Sod, Brio-Wu, high-β→low-β transition tube | L1(ρ) < 0.05 cross-backend |
| 3. 2D MHD (W6) | Orszag-Tang, field-loop advection, cylindrical explosion | L2 convergence measured |
| 4. Linear waves (W7) | Fast/slow/Alfvén wave convergence at 64, 128, 256 | PLM ~2nd, WENO5 ~4th order |
| 5. Z-pinch (W8-9) | Cylindrical plasma column, prescribed I(t), no circuit | Stable Bennett equilibrium |
| 6. Cross-backend (W9) | Phase A PyTorch vs Phase B MLX on all standard problems | L1 parity < 5% |
| 7. PF-1000 (W10-11) | Full circuit-coupled discharge | I_peak < 10%, NRMSE < 0.25, V_pinch > 50 kV |
| 8. Multi-device (W12) | UNU-ICTP, NX2, POSEIDON-60kV | No device regresses > 2× |

### 8.3 Diagnostics Chain Test

Verify the full Lp → dLp/dt → back-EMF → V_pinch pipeline:
1. Lp computed from density field (CircuitCoupler)
2. Lp monotonically increasing during radial phase
3. back_emf = -I·dLp/dt produces physical voltage
4. V_pinch at peak compression > 50 kV

---

## 9. Migration Path

| Current Component | LOC | Action | Phase |
|-------------------|-----|--------|-------|
| `_riemann_solvers.py` | 397 | Port + add Sρ to HLLD states | B, W3-4 |
| `_riemann_reconstruction.py` | 391 | Port (component-wise WENO5-Z) | B, W3-4 |
| `metal_stencil.py` | 866 | Port + simplify CT for 2D | B, W5-6 |
| `metal_transport.py` | 919 | Port Thomas + Braginskii | B, W6-7 |
| `metal_riemann.py` | 504 | Rewrite: new flux with Sρ + geometric sources | B, W5-6 |
| `metal_solver.py` | 2226 | Rewrite: new time integration + switching | B, W7-9 |
| `device.py` | 336 | Replace with MLX device/stream (~150 LOC) | B, W3 |
| `mlx_surrogate.py` | 492 | Keep as-is (already MLX) | — |
| Custom Metal kernels | — | New: ghost pad, HLLD, geometric sources (~330 LOC MSL) | B, W4-6 |

### Estimated Totals

| Metric | Phase A | Phase B | Total |
|--------|---------|---------|-------|
| New/modified LOC | ~400 | ~4,200 | ~4,600 |
| Calendar time | 1-2 weeks | 8-10 weeks | 10-12 weeks |
| Risk | Low | Medium-High | — |

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Entropy switching artifacts at electrode | Medium | High | Smooth blending with cubic Hermite; tune η₁, η₂ on Brio-Wu before DPF; Phase A validates physics first |
| MLX custom kernel debugging | Medium | Medium | `verbose=True` for generated MSL; Xcode Metal GPU debugger; unit test each kernel vs NumPy |
| MLX API breaking changes | Low | Low | Pin MLX version; core array API stable since 0.1 |
| Grid too small for GPU advantage | HIGH | Medium | Phase A runs on PyTorch (any grid); Phase B targets 128×512+; Athena++ remains primary for small grids |
| Phase A dual-energy breaks existing tests | Low | High | Infrastructure 80% exists; incremental addition of switching; all 262 tests gated |
| Shock entropy sync breaks SSP | Low | Low | Per Gottlieb+ 2001, pointwise ops preserve SSP; verified in Phase A |
| Back-EMF double-count fix changes I(t) | Medium | Medium | Compare I(t) before/after fix on PF-1000; should IMPROVE accuracy |
| Total energy not conserved with dual-energy | Medium | High | Phase A acceptance test: energy drift < 10⁻⁵ over 100 steps at β<<1 |

---

## 11. Six Sigma Quality Framework

### 11.1 DMAIC Integration

- **Define**: Eliminate float32 pressure corruption at DPF electrodes
- **Measure**: I_peak error, NRMSE, pressure positivity rate, energy conservation
- **Analyze**: Root cause proven (14-layer RCA). Solution validated (4 research agents, 3 literature sources)
- **Improve**: Dual-energy formalism with entropy tracer
- **Control**: 3 CRITICAL tests gate every commit; Phase A→B cross-backend parity

### 11.2 Quality Gates Per Work Unit

Every WU produces:
1. Code that passes `ruff check`
2. Module import test: `python3 -c "import <module>"`
3. Targeted test run (specific test file, not full suite)
4. MATH blind verification for any physics formula
5. Cross-backend parity check where applicable

### 11.3 Sigma Targets

| Phase | Target | DPMO |
|-------|--------|------|
| Phase A (PyTorch patch) | 4.0σ | < 6,210 |
| Phase B stages 1-4 (unit + 1D + 2D) | 3.5σ | < 22,750 |
| Phase B stages 5-8 (DPF validation) | 3.0σ | < 66,807 |

---

## 12. References

### Dual-Energy Formalism
- Bryan, G. L. et al. (2014). ENZO: An Adaptive Mesh Refinement Code. ApJS 211:19. [η₁=10⁻³, η₂=10⁻¹ switching]
- Lee, D. (2013). Unsplit staggered mesh scheme for 3D MHD. JCP 243:269. [FLASH dual-energy MHD]
- Fryxell, B. et al. (2000). FLASH. ApJS 131:273.
- **Popovas, A.** (2025). DISPATCH methods: entropy-based Riemann solver for ideal MHD. A&A. DOI: 10.1051/0004-6361/202554028

### MHD Numerics
- Miyoshi, T. & Kusano, K. (2005). HLLD solver. JCP 208:315.
- Borges, R. et al. (2008). WENO-Z scheme. JCP 227:3191.
- Gardiner, T. A. & Stone, J. M. (2005). CT method. JCP 205:509.
- Gottlieb, S., Shu, C.-W. & Tadmor, E. (2001). SSP methods. SIAM Review 43:89.

### DPF Physics
- Lee, S. & Saw, S. H. (2014). Plasma focus ion beam fluence and flux. PoP 21:062702.
- Scholz, M. et al. (2006). PF-1000 characterization. NIM-A 551:554.

### Framework
- Apple Machine Learning Research (2024-2025). MLX. github.com/ml-explore/mlx
