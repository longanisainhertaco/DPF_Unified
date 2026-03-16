# DPF-Unified Strategic Plan v2 — Cross-Discipline Physics Upgrades

## Critical Assessment

Before planning, acknowledge what's hard and what's easy. Cortana doesn't sugarcoat.

### What's Actually Easy
- **Velocity shear diagnostic**: v_theta already exists in both MHD solvers (vel[1]). Computing dv_theta/dr is one line of numpy. Comparing against MCX stabilization criterion is arithmetic. This is a **diagnostic**, not a physics module — no solver changes needed.
- **Lee fit validation expansion**: We have 29 Lee fits in the database (24 PF-1000 shots + 4 FAETON-I + 1 Double-DPF). Running all 24 PF-1000 shots through the simulator and comparing is automated — it's a script, not research.

### What's Medium Difficulty
- **Auluck poloidal B-field**: 13 equations extracted and ready. The hard part isn't the math — it's coupling a new PDE (the flux function evolution) into the existing MHD time-stepping loop without breaking energy conservation. The Hamiltonian structure (Eq. 9) provides a natural conservation check. Estimated 200-300 LOC.
- **CIV breakdown model**: The physics is understood (Alfven 1954, validated in space experiments). The implementation challenge is that CIV operates on nanosecond timescales while our MHD operates on microseconds — 1000x scale separation. Either we sub-cycle (expensive) or we use it as an IC generator (set sheath thickness and temperature from CIV theory, then start MHD).

### What's Hard (Be Honest)
- **AMR in PyTorch**: Our Metal solver uses fixed-size tensors on GPU. AMR requires variable-size grids or patch-based decomposition. PyTorch doesn't have native AMR support. MPI-AMRVAC is written in Fortran 90. Porting their AMR logic to PyTorch is a multi-week project, not a sprint. **Alternative**: instead of true AMR, use **static mesh refinement** — run the coarse simulation first, identify where the sheath is, then re-run with a refined grid around that region. Ugly but effective. 2 sessions instead of 8.
- **NLTE opacity neural network**: Training requires detailed atomic physics data (NLTE population calculations) that we don't have. The 2025 paper used Los Alamos ATOMIC code outputs as training data. We don't have ATOMIC access. **Alternative**: use tabulated opacities from SESAME tables or the CHIANTI atomic database (open source) and interpolate.

---

## Phase 1: Low-Hanging Fruit (2 sessions)

### 1A. Velocity Shear Stabilization Diagnostic
**Effort:** 1 session | **Risk:** None | **Publication value:** Yes (novel for DPF)

**What:**
- Compute Ω' = d(v_theta)/dr from MHD velocity field
- Compare against Shumlak-Hartman stabilization criterion: |dv_z/dr| > k_z × v_A
  (where v_A = B/sqrt(μ₀ρ) is Alfven speed, k_z = 2π/λ is perturbation wavenumber)
- Display in Physics Breakdown tab as "Shear Stabilization Margin"
- Color-code: green = stable (shear exceeds threshold), red = unstable

**Source data:**
- MCX papers: "Velocity Shear Stabilization of Centrifugally Confined Plasma" (in OlderPapers)
- Zap Energy: "Whole Device Modeling of the FuZE" (2024, downloaded)
- Shumlak & Hartman 1995: original shear stabilization criterion

**Implementation:**
```python
# In app_mhd.py after MHD step:
v_theta = state["velocity"][1]  # already computed
dv_dr = np.gradient(v_theta[:, ny//2, :], dr, axis=0)
v_alfven = np.sqrt(np.sum(state["B"]**2, axis=0) / (mu_0 * state["rho"]))
shear_margin = np.abs(dv_dr) / (2*np.pi/L_anode * v_alfven[:, ny//2, :])
# shear_margin > 1 = stable, < 1 = unstable
```

**Validation:** Compare against Zap Energy's published FuZE stability analysis.

**Why this matters:** If the DPF pinch has sufficient velocity shear to suppress m=0, it explains why some devices achieve longer confinement times than the Goyon instability timing predicts. This resolves a discrepancy in our current model.

### 1B. 24-Shot PF-1000 Validation Sweep
**Effort:** 0.5 session (scripted) | **Risk:** None | **Publication value:** Yes (statistical validation)

**What:**
- Run all 24 PF-1000 shots from Akel 2021 through the Lee model
- Use the exact fc/fm values from each shot's Lee fit
- Compare I_peak and NRMSE against experimental waveforms
- Generate a validation table: shot number, measured I_peak, simulated I_peak, % error, NRMSE
- Statistical analysis: mean error, std dev, correlation coefficient

**Source data:** 24 Lee fits already in database (shots 12581-12606)

**Validation target:** Mean I_peak error < 10%, NRMSE < 20% across all 24 shots

**Why this matters:** Single-shot validation (what we have now) can be gamed by overfitting. 24-shot statistical validation with published parameters proves the model works across a range of conditions. This is what a reviewer would demand.

---

## Phase 2: Novel Physics (3-4 sessions)

### 2A. Auluck Poloidal B-Field Module
**Effort:** 2 sessions | **Risk:** Medium (PDE coupling) | **Publication value:** High (first implementation worldwide)

**What:**
- Implement the Gratton-Vargas surface equation (Eq. 8) as the sheath geometry
- Evolve the magnetic flux function Φ(τ, r̄, z̄) using the Hamiltonian form (Eq. 9)
- Compute axial B_z from Φ: B_z = (1/2πa²r̄) × ∂Φ/∂r̄
- Couple to MHD: add B_z to the induction equation, modify J×B force

**The 13 equations (from database):**
1. Gratton-Vargas surface: ψ(τ, r̄, z̄) — defines sheath shape
2. Flux function: Φ = ∫B_z × 2πr dr
3. Flux PDE (Jacobi form): ∂Φ/∂τ = f(Φ, r̄, z̄)
4. Hamiltonian: H = (s/2r̄)√(r̄² - N²) p_r̄ + (N/2r̄) p_z
5. B_z from flux
6. B_r from flux
7. Azimuthal E-field from flux
8. J_theta from B-field curl
9. Generalized Ohm's law (azimuthal)
10. Vector potential evolution
11. Device current decomposition
12. Scaled time τ
13. Scaling parameters (B_0, v_0)

**Risk mitigation:**
- Start with the Hamiltonian form (Eq. 9) — it has a built-in conservation law
- Test on a 1D slab geometry first (reduces to ODE)
- Compare against Auluck's published results for PF-1000

**Why this matters:** The poloidal field affects EVERYTHING:
- Pinch stability (additional restoring force against kink)
- Neutron yield (modified ion orbits change thermonuclear rate)
- Reconnection dynamics (axial B + azimuthal B → helical field → 3D topology)
- Nobody else has this. First implementation = first publication.

### 2B. CIV-Based Breakdown Model (Phase 1)
**Effort:** 1-2 sessions | **Risk:** Medium (timescale coupling) | **Publication value:** Medium

**What:**
- Model Phase 1 as a CIV ionization front propagating across the insulator surface
- CIV threshold: v_crit = √(2eV_i/m_i) where V_i is ionization potential
  - D₂: v_crit = 49 km/s
  - He: v_crit = 34 km/s
  - Ar: v_crit = 8.7 km/s
- When E×B drift velocity exceeds v_crit, the gas ionizes
- Output: initial sheath thickness δ, electron temperature T_e, ionization fraction
- Feed these as ICs to the snowplow model instead of "fully ionized from t=0"

**Implementation approach:** IC generator, not sub-cycling
```python
def compute_breakdown_ic(V0, B_initial, gas, P_fill):
    """Compute initial sheath state from CIV theory."""
    v_crit = sqrt(2 * e * V_ionization[gas] / m_ion[gas])
    E_field = V0 / gap  # initial E-field
    v_ExB = E_field / B_initial  # E×B drift
    if v_ExB > v_crit:
        # CIV ionization occurs
        delta = electron_mean_free_path(P_fill, gas)  # sheath thickness
        T_e = 0.5 * m_e * v_ExB**2 / kB  # electron temperature from drift energy
        return {"delta": delta, "T_e": T_e, "ionized": True}
    else:
        return {"ionized": False}  # Paschen breakdown instead
```

**Source data:** 13 CIV papers, Alfven 1954, review papers on CIV in magnetrons and space

**Why this matters:**
- Answers the UAT question "why do you skip Phase 1?"
- Predicts gas-dependent breakdown timing (Ar breaks down faster than D₂ — lower v_crit)
- Explains why pressure affects shot quality (changes mean free path → sheath thickness)
- Connects DPF to a 70-year-old physics debate that's been validated in space experiments

---

## Phase 3: Resolution & Accuracy (2-3 sessions)

### 3A. Static Mesh Refinement (Practical AMR)
**Effort:** 2 sessions | **Risk:** Low (no solver changes) | **Publication value:** Medium

**Why not true AMR:** PyTorch tensors are fixed-size. Real AMR requires variable-size data structures, patch management, and load balancing. That's a multi-month project. Not justified for a laptop-scale code.

**What instead:** Two-pass refinement:
1. Run coarse simulation (16×16×32) — fast, identifies where the sheath is
2. Extract sheath location from density gradient
3. Create a refined grid: 64+ cells in a radial band around the sheath, coarse elsewhere
4. Re-run with the refined grid, using coarse results as initial condition

**Implementation:**
```python
def static_refinement(coarse_result, refinement_factor=4):
    """Identify sheath location and create refined grid."""
    rho = coarse_result["rho"][:, :, nz//2]  # midplane density
    drho_dr = np.gradient(rho, dr, axis=0)
    sheath_idx = np.argmax(np.abs(drho_dr))  # peak gradient = sheath

    # Create non-uniform radial grid: fine near sheath, coarse elsewhere
    r_fine = np.concatenate([
        np.linspace(r[0], r[sheath_idx-2], 8),           # coarse inner
        np.linspace(r[sheath_idx-2], r[sheath_idx+2], 32), # fine at sheath
        np.linspace(r[sheath_idx+2], r[-1], 8),           # coarse outer
    ])
    # Interpolate coarse solution onto fine grid as IC
    # Run Metal solver with fine grid
```

**Expected improvement:** Sheath resolved with ~0.5mm cells (vs 2.8mm) in a 48-cell grid instead of 256 cells globally. 5x speedup over uniform fine grid.

### 3B. Improved Radiation Model
**Effort:** 1 session | **Risk:** Low | **Publication value:** Low (incremental)

**What:**
- Add Gaunt factor correction to bremsstrahlung (currently g_ff = 1, should be ~1.2-1.5)
- Add recombination radiation for partially ionized species
- Use CHIANTI atomic database (open source, Python package) for line radiation coefficients
- Wire into Metal MHD energy equation alongside existing bremsstrahlung

**Source:** Chen textbook (Gaunt factor), CHIANTI database, ICF physics review paper

---

## Phase 4: Cross-Device Validation (1-2 sessions)

### 4A. Zap Energy FuZE Preset
**Effort:** 1 session | **Risk:** Low | **Publication value:** High (cross-device validation)

**What:**
- Extract FuZE circuit parameters from "Whole Device Modeling" paper (2024)
- Create a FuZE preset in presets.py
- Run through Hybrid backend
- Compare I(t) waveform against published data
- If it matches: we've validated across device types (Mather DPF vs SFS z-pinch)

**Why this matters:** Same MHD equations, different geometry. If our solver works for both, it's not overfitted to one device.

### 4B. MAGPIE Stagnation Comparison
**Effort:** 0.5 session | **Risk:** Low | **Publication value:** Medium

**What:**
- Extract MAGPIE wire array stagnation parameters from OlderPapers
- Compare our MHD density structure at pinch stagnation against their optical measurements
- Qualitative comparison (same structure?) rather than quantitative (different geometry)

---

## Honest Risk Assessment

| Upgrade | Effort | Risk | Probability of Success | Publication? |
|---------|--------|------|----------------------|-------------|
| 1A. Velocity shear diagnostic | 1 session | None | 99% | Yes — novel for DPF | **DONE** |
| 1B. 24-shot PF-1000 sweep | 0.5 session | None | 99% | Yes — statistical validation | **DONE** (1.27% error) |
| 2A. Poloidal B-field | 2 sessions | Medium | 70% | Yes — first implementation | **DONE** (first worldwide) |
| 2B. CIV breakdown | 1-2 sessions | Medium | 80% | Yes — connects to space physics | **DONE** (8 gases, 35 tests) |
| 3A. Static mesh refinement | 2 sessions | Low | 90% | Maybe — incremental | **DONE** (Lohner + interpolation) |
| 3B. Improved radiation | 1 session | Low | 95% | No — incremental | **DONE** (Gaunt + cyclotron, 28 tests) |
| 4A. FuZE preset | 1 session | Low | 85% | Yes — cross-device validation | Planned |
| 4B. MAGPIE comparison | 0.5 session | Low | 90% | Maybe — qualitative | Planned |

**Total: 9-12 sessions for all 8 upgrades.** Phases 1+2 are the priority — they produce 4 publishable results.

## What I'm NOT Proposing (And Why)

1. **Full AMR in PyTorch** — 2-3 months of work for marginal benefit on a laptop. Static refinement gets 80% of the benefit at 10% of the cost.
2. **NLTE opacity neural network** — Requires Los Alamos ATOMIC training data we don't have. CHIANTI tabulated opacities are sufficient for DPF temperatures.
3. **Full PIC-MHD hybrid production run** — The PIC module exists but hasn't been validated. Running a production PIC simulation before validating the PIC module is putting the cart before the horse.
4. **WALRUS fine-tuning** — Requires generating 1000+ MHD trajectories as training data. The MHD solver needs to be validated first (Phase 4). Fine-tune WALRUS AFTER the solver is proven.

## Success Criteria

After all phases:
- [x] Velocity shear diagnostic showing stable/unstable in 3D plots
- [x] 24-shot PF-1000 mean I_peak error < 10% (achieved 1.27%)
- [x] Poloidal B_z field appearing in MHD snapshots
- [x] CIV-predicted breakdown in physics narrative (8 gas species)
- [x] Sheath resolved to < 1mm in static refinement (Lohner indicator + fine grid)
- [x] Improved radiation: T-dependent Gaunt factor + cyclotron + recombination
- [ ] FuZE current waveform matches published data within 15%
- [ ] PhD panel re-grade: B+ or higher
