# Gap Analysis: DPF-Unified vs Production MHD Codes

**Date**: 2026-03-26 | **Fidelity**: 7.5/10 current | **Target**: 9.0/10

---

## Executive Summary

DPF-Unified is competitive with production codes on core MHD numerics (reconstruction, Riemann solvers, time integration, div(B) control) and has unique strengths in circuit-plasma coupling and DPF-specific physics. The primary gaps are: (1) no multi-material/ALE capability, (2) no implicit MHD for stiff regimes, (3) AMR is prototype-only without refluxing, (4) radiation transport is optically thin only, and (5) EOS is ideal gas only. Of these, only items 3-4 are actionable for DPF fidelity improvement. Items 1-2 matter for ICF/HED but not for DPF at our operating conditions.

---

## 1. ALEGRA (Sandia National Laboratories)

Multi-physics ALE code for magnetically-driven experiments (Z-machine, MagLIF).

### Physics We Are MISSING Entirely

| Feature | ALEGRA | Impact on DPF | Priority |
|---------|--------|---------------|----------|
| Multi-material ALE (arbitrary Lagrangian-Eulerian) | Full multi-material interface tracking with remap | LOW -- DPF has one working gas (D2), not multi-material liner implosions. Electrode ablation is a secondary effect. | Not actionable |
| Tabular EOS (SESAME/LEOS) | Material properties from national EOS databases | LOW -- DPF plasmas are well-described by ideal gas EOS at our conditions (fully ionized D2, T < 10 keV). Tabular EOS matters for solid liners and high-Z materials. | Not actionable |
| Material strength models | Elastic-plastic constitutive models for solids | NONE -- DPF has no solid dynamics. Electrodes are rigid boundaries. | Irrelevant |
| IMC radiation transport | Implicit Monte Carlo for optically thick radiation | MEDIUM -- DPF pinch is optically thin for bremsstrahlung but optically thick for some line radiation at high Z. Our optically thin model is adequate for D2 but inadequate for Ne/Ar fill gases or Cu impurities at high density. | Phase S+ |
| Magneto-Rayleigh-Taylor growth rates | Built-in MRT instability tracking with interface reconstruction | LOW -- MRT matters for liner implosions. DPF sausage (m=0) instability is handled by our MHD solver naturally. Dedicated tracking would improve diagnostics but not physics. | Nice-to-have |

### Physics We HAVE at Lower Fidelity

| Feature | ALEGRA | DPF-Unified | Gap |
|---------|--------|-------------|-----|
| Resistivity models | Spitzer + Lee-More + anomalous (user-defined) | Spitzer + CIV/drift-velocity anomalous | We lack Lee-More (matters below 10 eV where Spitzer diverges). ~20% error at sheath front. |
| Radiation | Multi-group diffusion + IMC | Optically thin bremsstrahlung + line cooling (6-segment Cu) | Adequate for D2, inadequate for high-Z fill or impurity-dominated pinch. |
| Magnetic diffusion | Full tensor diffusivity (Ohmic + Hall + ambipolar) | Ohmic (implicit) + Hall (explicit, operator-split) | We lack ambipolar diffusion. Irrelevant for fully ionized DPF plasmas. |

### Physics We HAVE at Comparable Fidelity

- Ohmic heating (eta * J^2)
- Braginskii anisotropic thermal conduction (implicit Thomas solver)
- Cylindrical MHD with geometric source terms
- Circuit-plasma coupling (ALEGRA has this for Z-machine too)

---

## 2. HYDRA (Lawrence Livermore National Laboratory)

Rad-hydro + MHD for ICF capsule implosions and HED experiments.

### Physics We Are MISSING Entirely

| Feature | HYDRA | Impact on DPF | Priority |
|---------|-------|---------------|----------|
| Multi-group radiation diffusion | Frequency-dependent radiation transport in diffusion limit | LOW -- DPF is not radiation-dominated. Bremsstrahlung is < 1% of total energy budget. Multi-group matters for ICF hohlraums, not DPF. | Irrelevant |
| Laser-plasma interaction (LPI) | Ray tracing + inverse bremsstrahlung + SRS/SBS | NONE -- DPF has no laser. | Irrelevant |
| Tabular EOS (LEOS) | National lab EOS tables | LOW (same as ALEGRA) | Not actionable |
| Electron thermal transport (flux-limited) | Flux-limited Spitzer-Harm with nonlocal corrections | MEDIUM -- Our Braginskii conduction is equivalent to Spitzer-Harm. We lack flux limiting, which matters when mean-free-path approaches gradient scale length (at pinch). | Moderate |
| Thermonuclear burn | In-flight DD/DT fusion yield with alpha heating | LOW -- Our PIC beam-target model is more appropriate for DPF (beam-target dominates over thermonuclear by 10-100x). | Already better |

### Physics We HAVE at Comparable Fidelity

- Braginskii transport coefficients
- Two-temperature (Te, Ti) plasma model
- Magnetic field evolution with resistive diffusion

### Key Insight

HYDRA is designed for radiation-dominated ICF implosions. DPF is magnetically-dominated. Most of HYDRA's sophistication (multi-group radiation, LPI, hohlraum physics) is irrelevant to DPF. The one transferable idea is flux-limited thermal conduction.

---

## 3. PERSEUS (Princeton/NRL)

Extended MHD code specifically designed for Z-pinch and DPF experiments. **This is our closest competitor.**

### Physics We Are MISSING Entirely

| Feature | PERSEUS | Impact on DPF | Priority |
|---------|---------|---------------|----------|
| Displacement current (dE/dt in Ampere's law) | Full Maxwell vs MHD approximation | LOW -- Displacement current matters when v_flow ~ c or omega ~ omega_pe. DPF sheath velocity is ~10^5 m/s << c. MHD approximation is valid. | Not actionable |
| Vacuum region handling (Boris correction) | Special treatment of low-density vacuum regions with modified Ohm's law | HIGH -- DPF has near-vacuum behind the sheath. PERSEUS uses Boris correction to handle the J x B force in vacuum without the E = -v x B + eta*J becoming singular. We use density floors and vacuum CFL masking but no Boris correction. | Phase S+ |
| Generalized Ohm's law with electron inertia | Full electron momentum equation, not just E = -v x B + eta*J + J x B/(ne) | LOW -- Electron inertia matters at skin depth scale (~0.1 mm). Our grid resolution (dr ~ 1 mm) doesn't resolve this. Would need AMR at pinch. | Future (with AMR) |
| Multi-fluid (separate ion + electron momentum) | Separate ion and electron fluids with friction coupling | MEDIUM -- Multi-fluid captures ion-electron decoupling during fast transients at pinch. Single-fluid with two-temperature is adequate for most of the discharge but misses some kinetic effects at pinch. Our PIC module partially covers this. | Phase S+ |

### Physics We HAVE at Lower Fidelity

| Feature | PERSEUS | DPF-Unified | Gap |
|---------|---------|-------------|-----|
| Hall MHD | Fully implicit, stable at all timesteps | Explicit operator-split, requires sub-cycling | Our Hall is correct (HL units verified) but explicit Hall CFL requires 10-100 sub-cycles at pinch. PERSEUS uses implicit Hall which is unconditionally stable. |
| Vacuum handling | Boris correction + separate vacuum region solver | Density floor + vacuum CFL masking | We handle vacuum passively (floor + mask). PERSEUS actively solves the vacuum region with a modified Ohm's law. This matters for current redistribution behind the sheath. |
| Resistivity at low T | Lee-More model for partially ionized plasma | Spitzer only (diverges below ~10 eV) | PERSEUS handles the full temperature range from cold gas to hot plasma. We assume fully ionized. |

### Physics We HAVE at Comparable Fidelity

- Axisymmetric cylindrical MHD (r,z)
- Circuit-plasma coupling (PERSEUS has the same Lee-model-style coupling)
- Anisotropic thermal conduction
- Bremsstrahlung radiation
- Hall MHD (our implementation is physically correct, just less robust numerically)

### Physics Where We EXCEED PERSEUS

- **AI/ML surrogate model (WALRUS)** -- no production z-pinch code has this
- **Multi-backend architecture** -- Python/Athena++/AthenaK/MLX
- **Hybrid PIC** -- PERSEUS is pure MHD; we have PIC for beam-target neutron yield
- **Multi-device calibration** -- automated Optuna parameter optimization across devices

### Key Insight

PERSEUS is the gold standard for DPF/z-pinch MHD. The actionable gaps are: (1) vacuum handling via Boris correction (~200 LOC, high impact), (2) implicit Hall MHD (~500 LOC, medium impact for pinch stability), and (3) Lee-More resistivity for partially ionized gas (~100 LOC, medium impact for early rundown).

---

## 4. FLASH (University of Chicago)

AMR MHD with unsplit Godunov scheme. Community code used at 100+ institutions.

### Physics We Are MISSING Entirely

| Feature | FLASH | Impact on DPF | Priority |
|---------|-------|---------------|----------|
| Unsplit MHD (CTU + CT) | Corner Transport Upwind -- dimensionally unsplit, 2nd-order accurate in 2D/3D without operator splitting | MEDIUM -- Our operator-split approach introduces O(dt) splitting errors at strong shocks. Unsplit avoids this. However, in axisymmetric (r,z) with operator splitting, the error is small because we use Strang splitting (2nd order). | Moderate |
| Production AMR with flux correction | Paramesh/SAMR with full conservative refluxing at coarse-fine boundaries | HIGH -- Our AMR is prototype (2-level, no refluxing). FLASH has battle-tested AMR with arbitrary levels and flux correction that guarantees conservation. | Phase S+ |
| Nuclear burning networks | Multi-species thermonuclear reaction networks | NONE -- Irrelevant for DPF. | Irrelevant |
| Gravity (multipole/tree) | Self-gravity solver | NONE -- DPF plasma is not self-gravitating. | Irrelevant |
| Laser energy deposition | Ray-based laser driver | NONE -- No laser in DPF. | Irrelevant |

### AMR Comparison

| Feature | FLASH | DPF-Unified | Status |
|---------|-------|-------------|--------|
| Levels | Arbitrary (typically 4-8) | 2 max | Prototype |
| Refinement criterion | User-defined (Lohner error estimator, density gradient) | Density gradient (basic) | Partial |
| Refluxing (conservation) | Full flux correction at CF boundaries | None (0.3-3% mass drift per step) | MISSING |
| Block structure | Paramesh octree | Flat block list | Simpler |
| Load balancing | Morton curve SFC | None (single GPU) | N/A (single device) |
| Performance overhead | 15-30% vs uniform at same effective resolution | Unknown (not benchmarked) | Untested |

### Numerical Methods Comparison

| Method | FLASH | DPF-Unified | Assessment |
|--------|-------|-------------|------------|
| Reconstruction | PPM (3rd order) or WENO | WENO5-Z (5th order) or PLM (2nd) | We win on order |
| Riemann solver | HLLD, HLL, Roe | HLLD, HLL, HLLS | Comparable |
| Time integration | Unsplit CTU (2nd order) | SSP-RK3 + operator split | Different approach; comparable accuracy |
| div(B) | Staggered mesh CT + dedner | CT + Dedner GLM | Comparable |
| Dual energy | FLASH uses e_int/e_kin switching | Entropy tracer with Popovas switching | We have a more modern approach |

### Key Insight

FLASH's main advantage over us is production AMR. Their unsplit MHD is elegant but our Strang-split SSP-RK3 achieves comparable accuracy for axisymmetric problems. The actionable gap is AMR refluxing.

---

## 5. Athena++ (Princeton)

Already our backend via pybind11. What stock features do we not use?

### Athena++ Features We Don't Use

| Feature | Athena++ | Why Not Used | Impact |
|---------|----------|-------------|--------|
| PPM + characteristic reconstruction | 3rd-order PPM with characteristic decomposition into MHD eigenvectors | Not wired through pybind11 wrapper | MEDIUM -- would improve shock resolution. Available if we configure `xorder=3` with `nghost=3` rebuild. |
| Orbital advection | FARGO-style shearing box for accretion disks | Irrelevant to DPF | NONE |
| General relativity (GR) | Kerr-Schild metric MHD | Irrelevant | NONE |
| Static/adaptive mesh refinement | Block-based SMR/AMR | Not wired through wrapper | HIGH -- Athena++ has production AMR that we could use instead of building our own |
| Shearing box BCs | For MRI studies | Irrelevant | NONE |
| Particle module | Lagrangian tracer particles + cosmic ray transport | Could supplement PIC | LOW |
| Non-ideal MHD (field_diffusion) | Ohmic + Hall + ambipolar as built-in diffusion operators | Only Ohmic wired; Hall/ambipolar not exposed | MEDIUM |
| Viscosity (hydro_diffusion) | Isotropic and anisotropic Braginskii viscosity | Not wired | LOW -- viscosity is secondary for DPF |
| Super-time-stepping | RKL2 STS for parabolic terms | Not wired through wrapper | MEDIUM -- would speed up resistive/conductive steps |

### What We Should Wire

1. **Athena++ AMR** -- rather than building AMR from scratch in MLX, wire Athena++'s proven AMR. Requires pybind11 extension but the AMR code already exists and works.
2. **Athena++ PPM + characteristic** -- configure `xorder=3`, rebuild with `nghost=3`. Minimal code change for improved shock resolution.
3. **Athena++ Hall diffusion** -- already implemented in `src/field/field_diffusion/`. Just needs pybind11 exposure.

---

## 6. GORGON (Imperial College London)

3D resistive MHD code specifically for z-pinches and DPF, used for MAGPIE experiments.

### Physics We Are MISSING Entirely

| Feature | GORGON | Impact on DPF | Priority |
|---------|--------|---------------|----------|
| Full 3D (not axisymmetric) | Cartesian 3D with r,theta,z option | HIGH for instability studies -- m=0 and m=1 instabilities are inherently 3D. Our axisymmetric solver cannot capture m=1 kink modes at all. | Phase T+ |
| Van der Waals / real gas EOS | Non-ideal gas corrections at high density | LOW -- DPF plasma is fully ionized at pinch. Only matters during initial gas breakdown. | Not actionable |
| Radiation magnetohydrodynamics (RMHD) | Coupled rad-MHD with optical depth effects | MEDIUM -- same gap as ALEGRA. Our optically thin model misses reabsorption at high density. | Phase S+ |
| Wire array physics | Multi-wire ablation and implosion dynamics | NONE -- DPF has no wire arrays. | Irrelevant |
| Thomas-Fermi ionization model | Pressure ionization at extreme density | LOW -- DPF densities (10^24-10^26 m^-3) don't reach pressure ionization. | Not actionable |

### Physics We HAVE at Lower Fidelity

| Feature | GORGON | DPF-Unified | Gap |
|---------|--------|-------------|-----|
| 3D MHD | Full 3D Cartesian | 2D axisymmetric (r,z) + Cartesian 3D (MLX, no DPF physics) | Our MLX solver has Cartesian 3D capability but it's not wired for DPF physics (no electrode BCs, no circuit coupling in 3D). |
| Resistivity | Spitzer + Lee-More + anomalous | Spitzer + CIV/drift-velocity | Same Lee-More gap as PERSEUS. |
| Radiation transport | P1/diffusion + escape factors | Optically thin + optical escape factor | We have escape factors (added Phase R). Comparable for DPF. |

### Physics We HAVE at Comparable Fidelity

- Braginskii anisotropic conduction
- Ohmic heating
- Circuit-plasma coupling
- Z-pinch geometry

### Key Insight

GORGON's main advantage is full 3D, which is essential for studying kink instabilities and asymmetric effects. Our Cartesian 3D MLX solver exists but lacks DPF-specific physics wiring. Extending it to 3D DPF would be a major project (~2000 LOC) but is the path to studying instability physics.

---

## 7. Numerical Methods Gap Assessment

### Methods We Have

| Method | Implementation | Quality |
|--------|---------------|---------|
| WENO5-Z reconstruction | MLX + Python | Production (Borges 2008) |
| PLM reconstruction | MLX + Metal + Python | Production |
| PPM reconstruction | MLX (Phase R) | Production |
| HLLD Riemann solver | MLX (float64) + Metal (float32) | Production (Miyoshi & Kusano 2005) |
| HLL Riemann solver | MLX + Metal + Python | Production |
| HLLS entropy Riemann solver | MLX | Production (Popovas 2025) |
| SSP-RK3 time integration | MLX + Python | Production (Shu-Osher 1988) |
| SSP-RK2 time integration | MLX + Metal + Python | Production |
| Constrained transport (CT) | MLX (cylindrical) | Production |
| Dedner GLM div(B) cleaning | MLX (cylindrical + Cartesian) | Production |
| Powell 8-wave div(B) | MLX | Production |
| Dual-energy (entropy tracer) | MLX | Production (Popovas switching) |
| Implicit resistive diffusion | Thomas solver (float64) | Production |
| Implicit thermal conduction | Thomas solver (float64) | Production |
| RKL2 super-time-stepping | MLX | Production |
| Operator (Strang) splitting | Engine | Production (2nd order) |

### Methods We Are MISSING

| Method | What It Does | Who Has It | Impact on DPF | Priority |
|--------|-------------|-----------|---------------|----------|
| Unsplit CTU (Corner Transport Upwind) | Dimensionally unsplit MHD -- avoids operator-split errors at oblique shocks | FLASH, Athena++ | LOW -- Strang splitting is 2nd order and adequate for axisymmetric. CTU matters more in 3D. | Future (with 3D) |
| Implicit MHD (fully implicit) | Unconditionally stable for stiff MHD -- eliminates fast-wave CFL constraint | ALEGRA (implicit), HYDRA | MEDIUM -- would eliminate whistler CFL constraint for Hall MHD. Currently mitigated by sub-cycling. | Moderate |
| Characteristic decomposition for reconstruction | Projects variables into MHD eigenvectors before reconstruction, reducing oscillations | Athena++ (PPM+char), FLASH | LOW -- WENO5-Z in conservative variables works well. Characteristic adds ~500 LOC for marginal gain. | Low |
| AMR with flux correction (refluxing) | Conservative interpolation at coarse-fine boundaries | FLASH, Athena++, GORGON | HIGH -- our AMR prototype has 0.3-3% mass drift without refluxing. | Phase S+ |
| Implicit radiation diffusion | For optically thick radiation transport | ALEGRA (IMC), HYDRA, GORGON (P1) | LOW-MEDIUM -- DPF pinch can be optically thick for line radiation. Our escape factor approximation is adequate for bremsstrahlung. | Phase S+ |
| Moving mesh / ALE | Mesh follows material interfaces, reducing diffusion | ALEGRA | NONE for DPF -- Eulerian is fine. | Irrelevant |

### Methods We Have That Production Codes Often Lack

| Method | Notes |
|--------|-------|
| HLLS (entropy-stable Riemann solver) | Popovas 2025 -- validated for float32 MHD. Most codes use HLLD/HLL only. |
| Dual-energy with entropy tracer | Modern approach (vs FLASH's e_int/E switching). Avoids circular dependency. |
| MLX Metal GPU acceleration | First-ever MLX PDE solver. No other MHD code runs on Apple Silicon GPU natively. |
| AI/ML surrogate (WALRUS 1.3B) | No production MHD code has a neural network surrogate model. |
| Multi-backend architecture | Python/Athena++/AthenaK/MLX -- unique flexibility. |
| Automated calibration (Optuna TPE) | Multi-device parameter optimization -- not available in any production code. |

---

## 8. Prioritized Action Items

Ranked by impact on DPF simulation fidelity, filtered for actionability.

### Tier 1: High Impact, Moderate Effort (would raise fidelity to 8.5-9.0)

| # | Item | Source Code | Effort | Impact |
|---|------|------------|--------|--------|
| 1 | **Wire Athena++ AMR** through pybind11 instead of building MLX AMR from scratch | FLASH, Athena++ gap | 1-2 weeks | Proven AMR with refluxing, eliminates R01 risk entirely |
| 2 | **Boris correction for vacuum regions** | PERSEUS | 200 LOC, 2-3 days | Fixes current redistribution behind sheath; eliminates need for aggressive density floors |
| 3 | **Flux-limited thermal conduction** | HYDRA, PERSEUS | 50 LOC | Prevents unphysical heat flux when mfp > gradient scale at pinch |
| 4 | **Lee-More resistivity model** for partially ionized gas | PERSEUS, GORGON, ALEGRA | 100 LOC | Correct resistivity below 10 eV (sheath front, early rundown) |

### Tier 2: Medium Impact, Moderate Effort (would raise fidelity to 9.0-9.5)

| # | Item | Source Code | Effort | Impact |
|---|------|------------|--------|--------|
| 5 | **Implicit Hall MHD** (backward Euler or Crank-Nicolson) | PERSEUS | 500 LOC, 1 week | Eliminates whistler CFL constraint; stable at all timesteps |
| 6 | **Optically thick line radiation** (escape probability or P1 diffusion) | GORGON, ALEGRA | 300 LOC, 1 week | Correct radiation for high-Z fills (Ne, Ar) and Cu impurities |
| 7 | **Athena++ PPM + characteristic** reconstruction | Athena++ stock | Rebuild + config, 1 day | Better shock resolution through pybind11 backend |
| 8 | **3D DPF physics in MLX Cartesian solver** (electrode BCs, circuit coupling) | GORGON | 2000 LOC, 3-4 weeks | m=1 kink instability, asymmetric effects |

### Tier 3: Low Impact or Long-term (fidelity 9.5+)

| # | Item | Source Code | Effort | Impact |
|---|------|------------|--------|--------|
| 9 | Unsplit CTU MHD | FLASH | 1000+ LOC | Only matters for 3D; Strang adequate for 2D |
| 10 | Multi-fluid (separate ion/electron momentum) | PERSEUS | 2000+ LOC | Captures decoupling at pinch; PIC partially covers this |
| 11 | Tabular EOS (SESAME) | ALEGRA, HYDRA | 500 LOC + data | Only for solid/liquid phases (electrode ablation) |
| 12 | Fully implicit MHD | ALEGRA | 5000+ LOC | Eliminates all CFL constraints; massive rewrite |

---

## 9. Honest Self-Assessment

### Where DPF-Unified is Strong (relative to production codes)

1. **DPF-specific circuit-plasma coupling** -- production codes (ALEGRA, HYDRA) have generic circuit models. Ours has Lee-model-calibrated coupling with density-weighted inductance, back-EMF, and crowbar closure.
2. **Multi-backend flexibility** -- no production code offers Python + C++ + GPU backends swappable at runtime.
3. **AI/ML integration** -- WALRUS surrogate + Optuna calibration is unique in the z-pinch community.
4. **Numerical methods** -- WENO5-Z + HLLD + SSP-RK3 + dual-energy entropy tracer is state-of-the-art. Most production codes use PPM + HLLD + RK2 or CTU.
5. **Hybrid PIC** -- beam-target neutron yield calculation is a capability PERSEUS/GORGON lack in their public versions.
6. **Test coverage** -- 4,900 tests is exceptional. Production codes typically have 100-500.

### Where DPF-Unified is Weak

1. **No production AMR** -- this is the single biggest gap. Every production code has it.
2. **2D only for DPF physics** -- Cartesian 3D exists but isn't wired for DPF. GORGON does full 3D z-pinch.
3. **Vacuum handling** -- density floors and CFL masking are hacks. Boris correction is the correct approach.
4. **No tabular EOS** -- fine for our current scope but limits extension to new materials.
5. **Single-node only** -- no MPI parallelism. Production codes run on thousands of cores.
6. **Operator-split transport** -- implicit Thomas solver is serial (column-by-column). Production codes use parallel multigrid.

### What Matters Most for DPF Fidelity 9.0

The path from 7.5 to 9.0 does NOT require multi-material, tabular EOS, MPI, or unsplit MHD. It requires:

1. Boris correction for vacuum (PERSEUS gap) -- **high impact, moderate effort**
2. AMR with refluxing (FLASH/Athena++ gap) -- **high impact, high effort** (or wire Athena++ AMR)
3. Lee-More resistivity (PERSEUS/GORGON gap) -- **medium impact, low effort**
4. Flux-limited conduction (HYDRA gap) -- **medium impact, low effort**
5. Implicit Hall (PERSEUS gap) -- **medium impact, moderate effort**

These five items, combined with existing Hall MHD + PIC + Thomson diagnostics, would bring DPF-Unified to parity with PERSEUS on DPF-relevant physics while retaining our unique advantages in AI/ML, multi-backend architecture, and test coverage.

---

## 10. Summary Table

| Capability | ALEGRA | HYDRA | PERSEUS | FLASH | Athena++ | GORGON | DPF-Unified |
|------------|--------|-------|---------|-------|----------|--------|-------------|
| Reconstruction | 2nd | 2nd | 2nd | PPM (3rd) | PPM+char (3rd) | 2nd | WENO5-Z (5th) |
| Riemann solver | HLLD | HLLD | HLLD | HLLD/Roe | HLLD/Roe | HLL | HLLD/HLL/HLLS |
| Time integration | Implicit | Implicit | Semi-implicit | CTU (2nd) | VL2/RK3 | RK2 | SSP-RK3 (3rd) |
| div(B) control | CT | CT | CT | CT+Dedner | CT | CT | CT+Dedner+Powell |
| AMR | Yes | Yes | No | Yes (Paramesh) | Yes (SMR/AMR) | No | Prototype (no reflux) |
| 3D | Yes | Yes | No (2D) | Yes | Yes | Yes | Partial (Cartesian only) |
| Multi-material | Yes | Yes | No | No | No | No | No |
| Tabular EOS | Yes | Yes | No | Yes | Yes (general) | Yes | No (ideal gas) |
| Circuit coupling | Yes | No | Yes | No | Custom pgen | No | Yes (Lee + MHD) |
| Hall MHD | Yes | No | Yes (implicit) | No | Yes | No | Yes (explicit) |
| Radiation transport | IMC | Multi-group | No | No | No | P1/diffusion | Optically thin |
| Braginskii transport | Yes | Yes | Yes | No | Yes | Yes | Yes |
| PIC/kinetic | No | No | No | No | Particles | No | Hybrid PIC |
| AI/ML surrogate | No | No | No | No | No | No | WALRUS 1.3B |
| Auto-calibration | No | No | No | No | No | No | Optuna TPE |
| Test suite size | ~500 | ~300 | ~100 | ~1000 | ~500 | ~200 | ~4,900 |
| MPI parallel | Yes | Yes | Yes | Yes | Yes | Yes | No |
