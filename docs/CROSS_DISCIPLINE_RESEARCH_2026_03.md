# Cross-Disciplinary Research Findings — March 2026

**Purpose**: Insights from adjacent fields, MLX practitioners, and simulation communities
beyond our usual DPF MHD focus. Each finding assessed for actionability.

---

## 1. MLX Practitioner Lessons Learned

### 1.1 Float64 Status: Definitively Closed
MLX Issue #1905 was **closed Dec 2025**. Awni Hannun (MLX team): "Emulating FP64 on GPU
would be quite slow and there's a good chance it will wipe out any speed improvements."
Metal hardware has no float64 silicon. PyTorch MPS has the same limitation.

**Recommended workaround** (from MLX team):
- Hybrid CPU/GPU: offload precision-critical ops (star-states, discriminants) to CPU float64
- File issues for slow CPU ops — team will optimize specific operations
- This aligns with our existing HLLD float64 star-states plan (Gap #1)

**Actionability**: HIGH. Our mixed-precision HLLD plan (float64 CPU discriminant, float32 GPU
storage/flux) is the exact pattern MLX team recommends. ~200 LOC. Priority: next sprint.

### 1.2 Performance Guide: "Writing Fast MLX" (Awni Hannun)
Key gotchas relevant to our solver:

| Gotcha | Impact on DPF | Fix |
|--------|---------------|-----|
| `.item()` in loops triggers sync eval | Our CFL computation calls `.item()` | Batch CFL: `mx.eval()` once per step |
| Scalar type promotion: `array * mx.array(2.0)` upcasts to float32 | Our float16 experiments would break | Use Python scalars: `array * 2.0` |
| `mx.compile()` recompiles on shape change | Grid resize triggers recompile | Use fixed grid sizes or `shapeless=True` |
| Closure over `mx.array` captures full compute graph | Our compiled RHS might leak | Pass arrays as explicit inputs |
| Delete temporaries before `mx.eval()` | Memory pressure during long runs | Add `del` for intermediate flux arrays |

**Actionability**: MEDIUM. 3-4 specific optimizations in our MLX solver. ~2 hours work.
Could reduce memory pressure and improve CFL computation latency.

### 1.3 ZMLX: Triton-Style Kernel Toolkit for MLX
New project (2025) providing Python-first Metal kernel authoring with `elementwise()` API.
70+ kernel catalog, automatic caching, autograd support. +12% decode speedup on LFM2-8B.

**Relevance**: Our 3 custom Metal kernels (ghost pad, HLLD, geo source) could potentially
be rewritten using ZMLX's `elementwise()` for simpler maintenance. However, ZMLX targets
ML workloads (MoE fusion), not stencil operations. Our kernels need neighbor access patterns
that `elementwise()` doesn't support.

**Actionability**: LOW. Monitor for stencil/reduce kernel support. Not useful today.

### 1.4 No Other MLX PDE Solvers Exist (Confirmed)
GitHub search for "mlx" + {"simulation", "physics", "PDE", "stencil", "CFD"} returns zero
non-ML scientific computing projects. Our MLX MHD solver remains the **first and only**
MLX PDE solver. The "Speed up scientific computing with MLX" blog post (vincent.codes.finance)
only covers Monte Carlo finance simulations (embarrassingly parallel, no stencils).

**Actionability**: NONE (validation of our uniqueness claim for the JOSS paper).

### 1.5 WWDC 2025 MLX Sessions
Two sessions: "Get started with MLX" and "Explore LLMs on Apple silicon with MLX."
Both focus on LLM inference. No scientific computing content. M5 chip benchmarks show
continued Metal GPU improvements but no float64 hardware.

---

## 2. Adjacent Simulation Fields

### 2.1 PERSEUS Extended-MHD Vacuum Handling (HIGH PRIORITY)
**Source**: Gourdain et al., arXiv:2506.06625 (June 2025) + Rososhek et al., arXiv:2603.00330 (March 2026)

PERSEUS (Cornell) solves z-pinch problems nearly identical to ours with a fundamentally
different vacuum approach:

| Feature | Our approach | PERSEUS |
|---------|-------------|---------|
| Vacuum model | Density floor ~1e-4 * rho_max | Floor ~1e-10 g/cc (6 orders lower) |
| Alfven speed limit | None (vacuum cells masked from CFL) | Displacement current bounds v_A by c_light |
| Resistivity in vacuum | Spitzer + anomalous threshold | Full 3x3 conductivity tensor via Hall physics |
| Convergence | Sensitive to floor value | Converges for sufficiently low floors |
| Hall physics | Implemented but not wired to vacuum | Core of vacuum handling |

**Key insight**: Including displacement current (even numerically reduced c by factor 30)
bounds Alfven speed in vacuum without our hack of masking vacuum cells from CFL. This is
more physically correct AND more robust.

**Their anomalous resistivity model**: LHDI-driven, eta* proportional to (v_de/v_i)^2/(1+(v_de/v_i)^2)
with saturation bound B/(n_e*e). Matches our Gap #4 (drift-velocity resistivity).

**What we'd need**: Add displacement current term to Faraday's law (~50 LOC in MLX),
reduce c_light numerically (1 parameter), remove vacuum CFL masking hack.

**Actionability**: HIGH. Solves vacuum handling more robustly than our current approach.
~100 LOC. Would eliminate the vacuum-cell CFL masking that's currently a known fragility.

### 2.2 FLASH Validation for MagLIF (MEDIUM PRIORITY)
**Source**: arXiv:2504.10760 (April 2025)

FLASH achieves 10-30% agreement with Z-facility MagLIF experiments. Key methods:

- **Implicit magnetic diffusion** via backward-Euler + HYPRE (avoids resistive CFL constraint)
- **Nernst advection** with upwinding to prevent unphysical negative B_z
- **Circuit coupling** via lumped-element model + dynamic load impedance
- **PPM + Godunov + CTU** for hyperbolic terms
- **Block-structured AMR** with Lohner error estimator

**Applicable to DPF**: Their implicit magnetic diffusion approach would solve our
sub-cycling problem (Gap #10 comment about resistive CFL << MHD CFL). Currently we
sub-cycle resistive diffusion explicitly with N=ceil(dt_mhd/dt_res) capped at 20.
An implicit solve would remove this entirely.

**Nernst implementation detail**: They upwind the Nernst advection term. We have Nernst
in our physics but not upwinded — potential stability issue for convergent geometry.

**Actionability**: MEDIUM. Implicit diffusion is ~200 LOC but needs a linear solver
(conjugate gradient). Nernst upwinding is ~20 LOC fix.

### 2.3 Hall Term in Z-Pinch MRTI (HIGH RELEVANCE)
**Source**: Rososhek et al. (Cornell), arXiv:2603.00330 (March 2026)

PERSEUS simulations of neon gas-puff z-pinch on COBRA show:
- Hall term reproduces MRTI wavelength ~2.5mm (experiment: ~3.5mm)
- Plasma sheath width matches interferometry only with anomalous resistivity
- Hall+anomalous resistivity produces directionality matching experiment
- Without Hall: symmetric MRTI. With Hall: asymmetric, matching cathode-side pinching

**Key for DPF**: Our Hall MHD module exists but is Python-only (not in MLX). PERSEUS
shows Hall effects are NOT negligible for z-pinch dynamics — they affect MRTI morphology
and sheath structure. This is more important than we thought.

**Actionability**: MEDIUM-HIGH. Port Hall MHD to MLX (~150 LOC). Validates our Gap list
item but raises its priority. Affects pinch stability predictions.

### 2.4 Astrophysical Jet / Athena++ Tricks
Already covered in our Phase R. Key unused trick: **orbital advection** for quasi-steady
azimuthal flow (not applicable to DPF implosion). **AMR refinement criteria** based on
current density gradient (useful for sheath tracking) — we should use J-based refinement
when AMR arrives, not just density gradient.

**Actionability**: LOW now. Note for AMR implementation (Gap #11).

---

## 3. Unexplored Numerical Methods

### 3.1 DISPATCH HLLS Entropy-Based Solver (HIGH PRIORITY)
**Source**: Popovas et al., A&A (2025) — published version of arXiv:2211.02438

We already reference this paper but haven't fully digested its implications. The published
2025 version has significant detail:

**Core idea**: Replace total energy conservation with entropy evolution. Pressure derived
from entropy EOS inversion instead of P = (gamma-1)*(E - 0.5*rho*v^2 - 0.5*B^2).

**Why this matters for float32**:
- Entropy is logarithmic — stays near unity, well-conditioned
- Avoids the catastrophic subtraction that causes our HLLD float32 failures
- Our `dp_dt` chain rule cancellation site (metal_riemann.py:271-273) would be bypassed

**Implementation requirements**:
1. Add entropy S as evolved variable (sidecar array like Dedner psi) — ~30 LOC
2. Entropy flux in HLLD: F_S = S * F_rho / rho (passive scalar advection) — ~10 LOC
3. Entropy production Q_S from kinetic+magnetic energy flux divergences — ~80 LOC
4. EOS inversion: P = rho^gamma * exp((gamma-1)*S/R_specific) — ~5 LOC
5. Positivity enforcement: max(0, Q_S/T) — ~5 LOC

**Total**: ~130 LOC. Would potentially solve HLLD float32 stability WITHOUT needing
CPU float64 star-states. This is an alternative path to Gap #1.

**Risk**: Entropy is NOT conserved across shocks (by definition). The paper handles this
by computing entropy production Q_S from energy flux divergences, but this requires
careful implementation to avoid thermodynamic inconsistency. Strong shocks in DPF
(Mach 10+ at pinch) are the stress test.

**Actionability**: HIGH. Alternative to float64 HLLD. ~130 LOC. Should prototype and
compare against float64 approach. If it works, it's cheaper and runs entirely on GPU.

### 3.2 ADER Time Integration
**Source**: Dumbser et al., ExaHyPE framework

ADER replaces multi-stage RK time integration with a single space-time predictor step.
Advantages: single Riemann solve per timestep (vs 3 for SSP-RK3), higher order in time
without extra stages. Used in ExaHyPE for relativistic MHD.

**Assessment**: Significantly more complex than RK (requires space-time DG predictor).
Our SSP-RK3 works well. ADER would save 2 Riemann solves per step but add a local
space-time DG solve per cell. Net benefit unclear for our grid sizes (64-256 cells).

**Actionability**: LOW. Interesting for future AMR but not worth the complexity now.

### 3.3 Lattice Boltzmann MHD
Recent work (2024-2026) focuses on GPU-accelerated LB for fluids, with grid refinement
achieving 1596x840x840 on a single A100. However, LB-MHD remains niche — most work is
incompressible MHD natural convection (nanofluids, heat transfer). No compressible
MHD z-pinch work found.

**Actionability**: NONE. Wrong regime for DPF (compressible, high Mach, strong shocks).

### 3.4 Well-Balanced Schemes for Cylindrical MHD
**Source**: Datta (UW WARPX group), April 2024 + Klingenberg et al.

Well-balanced schemes maintain equilibrium solutions exactly, preventing spurious
oscillations from geometric source terms. For cylindrical MHD, the 1/r geometric
source terms (hoop stress, magnetic pressure) create a non-trivial equilibrium that
standard methods don't preserve.

**Key paper**: Klingenberg's semi-implicit well-balanced MHD scheme handles gravitational
source terms. Analogous to our 1/r geometric sources. Their approach: reconstruct
primitive variables in equilibrium-subtracted form.

**Assessment**: Our r-weighted finite volume already handles geometric sources well
(observation: "r-weighted FV > operator-split for cylindrical MHD"). A well-balanced
extension would subtract the Bennett equilibrium profile before reconstruction, improving
accuracy near the pinch where the equilibrium gradient is steep.

**Actionability**: MEDIUM. ~100 LOC to implement equilibrium-subtracted reconstruction.
Most beneficial for pinch-phase accuracy. Requires computing Bennett equilibrium profile
each step.

### 3.5 Divergence-Free Reconstruction (Balsara)
Face-centered B with WENO reconstruction guarantees local+global div(B)=0 by construction.
Different from CT (which evolves face-centered B via EMF) and Dedner (which cleans div(B)
errors). Balsara's approach reconstructs B at faces from cell-centered values using a
constrained L2 projection.

**Assessment**: We have CT (PyTorch Metal) and Dedner (MLX). Balsara reconstruction is
a third option that works with cell-centered storage (like our MLX solver) but gives
machine-precision div(B)=0 without a staggered mesh. However, it requires a multidimensional
Riemann solver at cell corners, which is significantly more complex.

**Actionability**: LOW. Dedner is sufficient for our accuracy targets. CT is better.
Balsara reconstruction adds complexity without clear benefit over CT.

---

## 4. DPF-Specific Literature

### 4.1 FAETON-I: 100 kV DPF (NEW DATA)
**Source**: Damideh et al., Scientific Reports (July 2025)

FAETON-I is the highest-voltage DPF (100 kV, 1 MA peak), producing 2.5e10 D-D neutrons/shot
consistently, peak 8e10 at 12 Torr. Key physics insight: **re-strikes divert current but
don't affect the target pinch plasma** — the dynamics-induced voltage peak of ~150 kV
produces the deuteron beam first.

**Relevance**: Our Lee model doesn't model re-strikes. For high-voltage devices like
FAETON-I, re-strike physics becomes important. Also: their TWO-STEP radial fitting
methodology (documented in our dpf-papers/damideh-2025-faeton-i.md) shows fm_radial
needs separate calibration from fm_axial.

**Actionability**: MEDIUM. Add FAETON-I as a validation target (100 kV regime tests our
solver at different operating conditions). Re-strike modeling is future work.

### 4.2 LPPFusion FF-2B Progress
- Record single-shot yield: 0.26 J (June 2025)
- Ion energy records: >200 keV (2016 data, still standing)
- Key challenge: boride stripe formation — sheath "leaves behind" boron
- Plans: pB11 experiments in early 2026

**Physics insight**: The "boride stripe" problem (sheath velocity > critical ionization
velocity for boron) is exactly our CIV mechanism in reverse — CIV matters for multi-species
DPF operation.

**Actionability**: LOW for simulator (we don't model pB11). Validates CIV importance.

### 4.3 MJOLNIR (LLNL) — Simulation-Guided Design
Dr. Andrea Schmidt (LLNL) used Chicago PIC code for MJOLNIR design. 1 MJ, 2.2 MA pinch,
4e11 neutrons/pulse. No new public simulation data found since the 2019 presentation.

**Actionability**: LOW. We already have MJOLNIR in our presets (Offermann 2021 data).

### 4.4 Hall + Anomalous Resistivity in Z-Pinches (March 2026)
Already covered in Section 2.3. The Rososhek et al. paper is the most significant new
DPF-adjacent paper. PERSEUS shows Hall effects matter for MRTI structure.

### 4.5 No New DPF Simulation Codes Found
Searched extensively. The DPF simulation landscape remains:
- Lee model (1D circuit): still dominant in the community
- USIM (2D/3D MHD): Tech-X, commercial
- Chicago (PIC): LLNL, not public
- Our dpf-unified (1D Lee + 2D MHD): unique open-source full-MHD DPF code
- PERSEUS: z-pinch but not DPF-specific

No new DPF simulation codes published in 2025-2026.

---

## 5. Machine Learning for Plasma Physics

### 5.1 Differentiable MHD (HIGH PRIORITY — Strategic)
**Source**: arXiv:2603.11231 (March 2026) — comprehensive review

The differentiable programming paradigm is exploding in plasma physics:
- **TORAX** (Google DeepMind): Differentiable tokamak transport in JAX. Open-source, v1.0.
- **JAX-in-Cell**: Differentiable PIC in JAX. End-to-end gradient-based optimization.
- **GANDALF**: Differentiable spectral MHD in JAX. Runs on Apple Silicon natively.
- **ADEPT**: Differentiable fluid code for learning kinetic closures.

**Key capabilities enabled by differentiability**:
1. Gradient-based calibration (140x faster than grid search for Thomson scattering)
2. Learning closures for sub-grid physics (e.g., kinetic effects in fluid models)
3. Inverse design optimization with O(1000) parameters
4. End-to-end training of surrogate models

**Strategic insight**: MLX supports `mx.grad()` (automatic differentiation). Our solver
is NOT currently differentiable because we use custom Metal kernels and explicit loops.
Making it differentiable would enable:
- Gradient-based fc/fm calibration (instead of Optuna black-box: 69 evals, hours)
- Learning anomalous resistivity models from experimental data
- Differentiable rendering of synthetic diagnostics

**Actionability**: HIGH but LARGE scope. Making the MLX solver differentiable requires
rewriting custom Metal kernels as composable MLX ops (~500 LOC refactor). Gap #17 on
our list. Should be a Phase S or T goal.

### 5.2 GANDALF on Apple Silicon (JAX Lessons)
**Source**: arXiv:2511.21891 (Nov 2025)

GANDALF demonstrates research-grade plasma simulations on Apple Silicon:
- M1 Pro / M2 Max via JAX Metal backend
- 128^3 grids practical on laptops
- 580 MB memory for 128^3 with 16 Hermite moments
- "JAX implementations competitive with compiled languages"

**Lessons for our MLX solver**:
- Real-to-complex FFT optimization reduces memory 50% (relevant if we add spectral methods)
- Pure functional design enables automatic parallelization
- @jax.jit produces hardware-specific kernels without manual optimization
- Single-command pip install is achievable for serious physics codes

**Actionability**: MEDIUM. Validates our choice of Apple Silicon for physics simulation.
GANDALF's JAX Metal backend approach vs our MLX approach: both work. JAX has broader
community, MLX has Apple-native advantages (unified memory, zero-copy). No action needed
but confirms our strategy.

### 5.3 Neural Operator Surrogates for Plasma
**Source**: arXiv:2502.17386 (Feb 2025), Plasma-FNO (GitHub)

FNOs applied to JOREK MHD and STORM turbulence codes for tokamak edge. Two-step transfer
learning addresses new-variable flexibility. DeepONet for Vlasov-Poisson (Landau damping).

**No z-pinch/DPF neural operator work exists.** This is a gap in the ML-plasma literature.

**Actionability**: MEDIUM-LOW. Our WALRUS surrogate path is more mature. FNO could be
an alternative to WALRUS but would need DPF training data (which we're generating anyway).
No immediate action; note for JOSS paper as future work direction.

### 5.4 Learned Adaptive Time-Stepping (NEW)
**Source**: ShockCast (arXiv:2506.07969, June 2025), MultiPDENet (ICML 2025)

ShockCast: ML model predicts optimal dt for high-speed flows with shocks. Two-phase:
(1) predict timestep, (2) advance state using predicted dt. Key innovation: physically-
motivated timestep prediction using Mixture-of-Experts conditioning.

MultiPDENet: Neural network corrects PDE prediction at coarse timescale, achieving 5x
speedup over classical methods while maintaining accuracy.

**Relevance to DPF**: Our CFL-based timestep is conservative (often 10-100x smaller than
needed for accuracy). A learned timestep predictor could identify when larger steps are
safe, especially during the slow axial phase where CFL is dominated by Alfven speed in
vacuum cells (which we already mask as a hack).

**Actionability**: LOW-MEDIUM. Interesting research direction but not priority. Our
PERSEUS-style displacement current approach (Section 2.1) is a physics-based solution
to the same timestep problem. Physics solution > ML solution when available.

---

## Priority Ranking

### Tier 1: Act This Sprint (~300 LOC total, high impact)
1. **DISPATCH entropy-based solver** (3.1): ~130 LOC. Potential float32 HLLD fix without CPU.
2. **PERSEUS vacuum handling** (2.1): ~100 LOC. Displacement current + remove CFL hack.
3. **MLX performance fixes** (1.2): ~50 LOC. Memory, CFL, compilation improvements.

### Tier 2: Next Sprint (~350 LOC total)
4. **Hall MHD in MLX** (2.3): ~150 LOC. PERSEUS shows it matters for z-pinch MRTI.
5. **Well-balanced cylindrical** (3.4): ~100 LOC. Bennett equilibrium subtraction.
6. **FLASH-style implicit diffusion** (2.2): ~200 LOC but needs linear solver.

### Tier 3: Strategic / Future
7. **Differentiable MHD** (5.1): ~500 LOC refactor. Enables gradient-based calibration.
8. **FAETON-I validation** (4.1): Compute only. New operating regime.
9. **Multi-device calibration** (existing Gap #12): Compute only.

### Not Worth Pursuing
- ADER time integration: Too complex for marginal benefit
- Lattice Boltzmann MHD: Wrong regime
- Balsara divergence-free reconstruction: Dedner/CT sufficient
- PINNs for z-pinch: Too immature, no prior work exists
- ZMLX for stencils: Wrong abstraction level

---

## Sources

### MLX
- [Writing Fast MLX (Awni Hannun)](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)
- [MLX float64 Issue #1905](https://github.com/ml-explore/mlx/issues/1905)
- [MLX for Scientific Computing (vincent.codes)](https://vincent.codes.finance/posts/apple-mlx/)
- [MLX Benchmark Suite](https://github.com/TristanBilot/mlx-benchmark)
- [ZMLX Kernel Toolkit](https://github.com/Hmbown/ZMLX)
- [WWDC 2025: Get Started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)

### Adjacent Fields
- [PERSEUS Extended-MHD Vacuum Model (2025)](https://arxiv.org/html/2506.06625v1)
- [Hall + Anomalous Resistivity in Z-Pinch (2026)](https://arxiv.org/html/2603.00330v1)
- [FLASH MagLIF Validation (2025)](https://arxiv.org/html/2504.10760v1)
- [DISPATCH HLLS Entropy Solver (2025)](https://www.aanda.org/articles/aa/full_html/2025/06/aa54028-25/aa54028-25.html)

### Numerical Methods
- [Chandrashekar & Klingenberg Entropy Stable MHD](https://epubs.siam.org/doi/10.1137/15M1013626)
- [Well-Balanced Semi-Implicit MHD](https://link.springer.com/article/10.1007/s10915-023-02422-z)
- [WARPX Cylindrical MHD Terms](https://faculty.washington.edu/shumlak/WARPX/html/mhd_cylindrical.pdf)

### DPF-Specific
- [FAETON-I Results (2025)](https://www.nature.com/articles/s41598-025-07939-x)
- [LPPFusion Record Yield](https://www.lppfusion.com/another-record-yield-understanding-our-progress/)
- [LPPFusion 2025 Review](https://www.lppfusion.com/2025-fusion-in-review/)

### ML for Plasma
- [Differentiable Programming for Plasma Physics (2026)](https://arxiv.org/html/2603.11231)
- [JAX-in-Cell (2025)](https://arxiv.org/abs/2512.12160)
- [GANDALF Spectral MHD Solver (2025)](https://arxiv.org/abs/2511.21891)
- [TORAX Differentiable Tokamak (2024)](https://arxiv.org/abs/2406.06718)
- [Neural Operator Plasma Surrogates (2025)](https://arxiv.org/html/2502.17386v2)
- [ShockCast Adaptive Time-Stepping (2025)](https://arxiv.org/abs/2506.07969)
- [MultiPDENet (ICML 2025)](https://arxiv.org/abs/2501.15987)
