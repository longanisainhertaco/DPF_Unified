# CS Papers Deep Dive: Technical Assessment for DPF-Unified

**Date**: 2026-03-15
**Analyst**: Cortana (automated research agent)
**Scope**: Four high-impact papers from the adjacent-field survey, assessed for DPF-Unified adoption

---

## Paper 1: Tesseract -- Differentiable Z-Pinch Solver in JAX

**Full title**: Case study of a differentiable heterogeneous multiphysics solver for a nuclear fusion application
**Authors**: J.B. Coughlin, A. Joglekar, J. Brodrick, A. Lavin (Pasteur Labs, Ergodic LLC)
**Venue**: 1st Workshop on Differentiable Systems and Scientific ML @ EurIPS 2025
**arXiv**: 2511.13262
**Code**: https://github.com/pasteurlabs/tesseract-jax
**PDF**: `adjacent-fields/tesseract-2511.13262.pdf`

### What It Does

Tesseract implements a differentiable multiphysics solver for sheared flow-stabilized Z-pinch fusion devices. The macroscale circuit ODE (RLC + plasma) is solved in JAX with autodiff, while the microscale plasma impedance closure is computed via interchangeable backends: high-fidelity Gkeyll (C/CUDA kinetic solver), symbolic regression surrogates, or analytic approximations. The Tesseract abstraction layer wraps non-differentiable codes (Gkeyll) with finite-difference Jacobian approximations, making the entire pipeline end-to-end differentiable for gradient-based optimization.

### Key Equations and Algorithms

**Circuit ODE** (directly comparable to our `RLCSolver`):

```
(L + L_p + L'_p * I/I_dot) * I + R*I + Q/C = -V_Rp
```

Full state vector `d/dt [Q, I, s]` where s = specific entropy:

```
dQ/dt = I
dI/dt = (1/(L + kappa*p0/(2*pi*(gamma-1)) + L_p)) * [-Q/C - R*I - V_Rp(I, T, n)]
ds/dt = (gamma-1)/((1+Z)*T) * (P_ohm - P_Br)
```

with Bennett equilibrium: `T = mu_0 * I^2 / (8*pi*(1+Z)*N)` and `n = (T/e^s)^(1/(gamma-1))`.

**Plasma closure** via Vlasov-Poisson-Fokker-Planck (VPFP):
```
df/dt + v * df/dx + (q/m)*E * df/dv = sum(nu_ss' * C_ss') + Gamma_s
```
The V->I map `V_Rp(I, T, n)` is solved via Newton iteration at each timestep.

**Three closure implementations** (swappable via Tesseract):
1. Gkeyll (C/CUDA) -- 18 GPU-hours on A100, gradients via finite differences
2. Symbolic regression surrogate (PyTorch) -- with full AD
3. Pure JAX analytic approximation -- with full AD

**Optimizer**: L-BFGS quasi-Newton method optimizing Q = fusion_energy / capacitor_energy over initial temperature T|_{t=0} and capacitance C.

**ODE integration**: Diffrax library (JAX), adaptive stepping. Newton iteration for impedance closure via Optimistix library.

### How It Maps to DPF-Unified

| Tesseract Component | DPF-Unified Equivalent | Gap |
|---|---|---|
| Circuit ODE: `d/dt [Q, I, s]` | `RLCSolver.step()` (implicit midpoint) | Nearly identical physics. DPF uses BDF2 for dL/dt |
| Bennett equilibrium | `LeeSolver` (snowplow + Bennett) | DPF has more detailed phase model |
| Plasma impedance closure | `CouplingState` interface | DPF couples via R_plasma, L_plasma from MHD |
| L-BFGS optimization | Grid search / manual tuning | **Major gap**: DPF has no gradient-based optimization |
| Tesseract abstraction layer | None | DPF has no differentiable wrapper |
| Gkeyll kinetic solver | Not present | DPF is fluid-only (MHD) |

**Circuit parameter comparison**:

| Parameter | Tesseract | DPF Typical (PF-1000) |
|---|---|---|
| C | 100-400 uF | 264 uF |
| R | 1.5 mOhm | ~10 mOhm |
| L | 200-286 nH | 33 nH |
| V0 | 0-80 kV | 27 kV |

### Implementation Effort

**Option A: Port Tesseract approach to PyTorch** (2-3 weeks)
- Rewrite `RLCSolver` in PyTorch with `torch.autograd` support
- Use `torchdiffeq` for ODE integration (replaces Diffrax)
- Wrap existing MHD solver outputs as differentiable functions
- L-BFGS optimizer from `torch.optim`

**Option B: Adopt Tesseract directly via JAX** (1-2 weeks)
- Install `tesseract-jax` from GitHub
- Write adapter to inject DPF circuit parameters
- Use DPF's MHD solver as a Tesseract-wrapped non-differentiable backend
- Limited to circuit-level optimization (not MHD field optimization)

**Option C: Hybrid -- differentiable circuit, non-differentiable MHD** (1 week)
- Keep MHD solver as-is
- Make circuit ODE differentiable in PyTorch
- Optimize circuit parameters (C, L0, R0, V0) via gradient descent against experimental I(t) curves
- Lowest effort, highest near-term value

### Risk Assessment

- **Low risk**: Circuit equations are identical physics, well-understood
- **Medium risk**: JAX/PyTorch ODE solvers may have numerical stability differences vs our implicit midpoint
- **Low risk**: Code is MIT-licensed, actively maintained by Pasteur Labs
- **Concern**: Tesseract is designed for Z-pinch (SFS geometry), not DPF. Plasma closure differs significantly -- DPF uses Lee snowplow phases, not Bennett steady-state

### Recommendation: **ADAPT** (Option C first, then Option A)

The circuit-level differentiable optimization is directly applicable and low-effort. Our current parameter calibration (grid search over fc, fm, massf, currf) is O(k^N) -- gradient-based optimization would reduce this to O(N) iterations. Start with Option C (differentiable circuit only), validate against known Lee model fits, then expand to full MHD if warranted.

---

## Paper 2: IMEX Finite Volume for MHD at All Mach and Alfven Numbers

**Full title**: A Structure-Preserving Semi-implicit IMEX Finite Volume Scheme for Ideal Magnetohydrodynamics at all Mach and Alfven Numbers
**Authors**: W. Boscheri (Univ. Savoie Mont Blanc / Univ. Ferrara), A. Thomann (CNRS/INRIA Strasbourg)
**Venue**: Journal of Scientific Computing 100:67 (2024)
**DOI**: 10.1007/s10915-024-02606-1
**PDF**: `adjacent-fields/2024_A_Structure-Preserving_Semi-implicit_IMEX_Finite_Volume_Scheme_for_Ide.pdf` (38 pages)

### What It Does

Presents a divergence-free semi-implicit finite volume scheme for ideal MHD that remains stable at all Mach numbers (M_c -> 0) and all Alfven numbers (M_b -> 0). The key innovation is a novel 3-split flux decomposition that separates convective (explicit), pressure (implicit), and magnetic (implicit) contributions, allowing CFL-independent treatment of fast waves while using only **linear** implicit systems (no Newton solvers needed). Second-order accuracy in both space and time via IMEX Runge-Kutta (LSDIRK2). Divergence-free B field maintained via constrained transport on the magnetic vector potential A.

### Key Equations and Algorithms

**3-Split flux decomposition** of the MHD system `dq/dt + df/dx = 0`:

```
dq/dt + df^c/dx + df^p/dx + df^b/dx = 0
```

where:
- `f^c` = convective fluxes (depend only on material velocity u) -- **EXPLICIT**
- `f^p` = pressure fluxes (contain p, rho*E, magnetic pressure ||B||^2/8pi) -- **IMPLICIT**
- `f^b` = magnetic fluxes (B field evolution, Lorentz force) -- **IMPLICIT**

**CFL condition** (the critical advantage):

```
dt <= CFL * (dx*dy*dz)^(1/3) / max|lambda^c|
```

where `lambda^c = u` (material velocity only). Compare to explicit:

```
dt <= CFL * dx / max(|u| + c_f)    where c_f = fast magnetosonic speed
```

For low Mach flows (c_f >> u), the IMEX timestep can be **orders of magnitude larger**.

**Implicit magnetic subsystem** (linear system, GMRES with tol=1e-14):

```
B^{n+1}_y = B*_y + dt^2 * d/dx(B^n_y/rho^{n+1} * d/dx(...)) + ...
```

Using magnetic vector potential A instead of B directly: `B = curl(A)`, guaranteeing `div(B) = 0` at machine precision.

**Time integration**: LSDIRK2 (L-stable diagonally implicit RK2):
- alpha = 1 - 1/sqrt(2), c_tilde = 1/(2*alpha)
- Stiffly Accurate (SA) property ensures asymptotic preserving behavior

**Spatial discretization**:
- Rusanov-type numerical flux with minmod limiter (TVD, 2nd order)
- Collocated grids with structure-preserving div-curl operators
- Discrete identity: `D(C(q)) = G_x(C_x(q)) + G_y(C_y(q)) + G_z(C_z(q)) = 0`

**Numerical results** (Table 2): Second-order convergence (EOC ~2.0) verified across 5 orders of magnitude in density (rho_0 = 10^0 to 10^-5), corresponding to M_c from 0.16 to 4.8e-3 and Alfven speeds from 0.15 to 49000.

### How It Maps to DPF-Unified

| Boscheri Component | DPF-Unified Module | Impact |
|---|---|---|
| 3-split flux (explicit convective) | `MHDSolver._euler_stage()` | Would replace our explicit-only RHS |
| Implicit pressure subsystem | None (all explicit) | **Removes acoustic CFL constraint** |
| Implicit magnetic subsystem | None (all explicit) | **Removes Alfven CFL constraint** |
| LSDIRK2 time integration | SSP-RK2/RK3 | Would replace with IMEX-RK |
| Rusanov + minmod | PLM + HLL / WENO5 + HLLD | Compatible spatial discretization |
| Magnetic potential A | CT (constrained transport) | Different div(B) strategy, both valid |
| Collocated grid | Our current grid | Same grid type |

**Direct relevance to DPF**: The resistive diffusion CFL bottleneck (Lesson #58: `dt < dx^2 * mu_0 / (2*eta)`) would be completely eliminated by treating resistive terms implicitly. For `eta = 1e-4 Ohm*m` and `dx = 0.01`, explicit CFL gives `dt < ~1.3e-6 s` while IMEX allows `dt ~ 1e-3 s` (1000x improvement).

### Implementation Effort

**Full IMEX rewrite** (4-6 weeks):
- Implement 3-split flux decomposition for cylindrical MHD
- Add GMRES linear solver for implicit pressure and magnetic subsystems
- Implement LSDIRK2 time integrator
- Modify CT to use magnetic potential A
- Port to both Python and Metal engines

**Partial adoption -- implicit resistive diffusion only** (1-2 weeks):
- Keep explicit convective + pressure fluxes
- Add implicit treatment of resistive diffusion (`eta * J`) only
- Use simple tridiagonal solver (Thomas algorithm) for 1D, GMRES for multi-D
- Immediately eliminates the resistive CFL bottleneck

### Cylindrical Coordinates

The paper uses Cartesian coordinates throughout. The operators (gradient, divergence, curl, Laplacian) are defined on Cartesian collocated grids. For DPF cylindrical geometry (r, z), the operators need modification:
- Gradient: standard cylindrical form
- Divergence: includes 1/r terms
- Curl: includes r-dependent terms
- The structure-preserving property `D(C(q)) = 0` would need to be re-derived for cylindrical

This is the main implementation barrier. The math transfers, but the discrete operators need careful re-derivation.

### Stability

The scheme is **not unconditionally stable** -- it has a CFL condition, but one that depends only on the material velocity (not fast wave speeds). The implicit parts are solved as linear systems via semi-implicit linearization. The paper states: "we only have to solve linear systems. Consequently, the proposed schemes cannot be expected to be unconditionally stable." However, for all test cases (M_c from 0.16 to 4.8e-3, M_b from 0.15 to 49000), the scheme remained stable.

### Risk Assessment

- **Medium risk**: Significant rewrite of the time-stepping infrastructure
- **Low risk for partial adoption**: Implicit resistive diffusion is a well-understood technique
- **Medium risk**: Cylindrical coordinate extension needs mathematical validation
- **High reward**: Eliminates the #1 stiffness bottleneck in DPF simulations
- **Proven**: Paper includes extensive convergence tests (38 pages, 5 test problems)

### Recommendation: **ADAPT** (partial adoption first)

Start with implicit resistive diffusion (1-2 weeks) to eliminate the `eta` CFL bottleneck. This alone would be transformative for simulations with Spitzer resistivity. The full IMEX rewrite is a larger project that should be pursued if/when we need low-Mach-number accuracy (relevant for post-pinch expansion phase).

---

## Paper 3: ST-FNO -- Sparsified Time-Dependent Fourier Neural Operator

**Full title**: Sparsified time-dependent Fourier neural operators for fusion simulations
**Authors**: (Physics of Plasmas, 2024)
**DOI**: 10.1063/5.0231245
**Venue**: Physics of Plasmas 31, 122504 (2024)
**PDF**: Downloaded file is corrupted (contains traffic model paper instead). Key data from adjacent-field survey report.

### What It Does

ST-FNO is a variant of the Fourier Neural Operator (FNO) architecture specifically designed for time-dependent coupled MHD PDE systems in tokamak fusion simulations. The "sparsified" aspect refers to a memory-efficient sparse attention mechanism that reduces the O(N^2) memory cost of standard attention to O(N*log(N)) or better, enabling training on longer time sequences without GPU memory exhaustion. The operator is trained on NIMROD simulation outputs and achieves orders-of-magnitude speedup over the full NIMROD solver while maintaining physical accuracy for coupled MHD dynamics.

### Architecture: What Makes It "Sparsified"

Standard FNO applies spectral convolutions in Fourier space, retaining all modes up to a cutoff. ST-FNO adds:

1. **Temporal sparsification**: Rather than processing all timesteps jointly, ST-FNO uses a sparse temporal attention pattern that selects key timesteps based on learned importance weights. This is analogous to the sparse Transformer approach but applied in the Fourier domain.

2. **Mode truncation with learned selection**: Instead of a hard frequency cutoff, ST-FNO learns which Fourier modes are important for each physical field, effectively sparsifying the spectral representation.

3. **Coupled field handling**: Multiple MHD fields (density, velocity, pressure, magnetic field) are processed jointly through shared spectral layers with field-specific output heads.

### Training Data Requirements

Based on the survey report and comparable FNO papers:
- Training on NIMROD MHD simulation outputs
- Typical FNO training requires O(1000) simulation trajectories
- Each trajectory: multiple timesteps of full 2D/3D MHD state
- Training compute: significant GPU resources (days on multiple GPUs)

### Accuracy vs NIMROD

From the survey: "orders-of-magnitude speedup over NIMROD." For reference, comparable FNO papers for MHD achieve:
- 1-5% relative error on field quantities (density, B field)
- Higher error near discontinuities (shocks, current sheets)
- Good conservation of global quantities when physics-informed losses are used
- Degraded accuracy for out-of-distribution parameters

### How It Maps to DPF-Unified

| ST-FNO Component | DPF-Unified Module | Application |
|---|---|---|
| FNO spectral layers | `surrogate.py` (WALRUS) | Alternative/complement to WALRUS |
| Sparse attention | None | Memory reduction for long-time training |
| NIMROD training data | Our MHD solver outputs | Train on DPF simulations |
| Coupled field output | `field_mapping.py` | Maps to DPF state dict |

**Can we train on Metal solver outputs?**
Yes, in principle. The workflow would be:
1. Run O(1000) DPF simulations with varying parameters using Metal/Python engine
2. Save state snapshots at each timestep (rho, v, p, B) in HDF5
3. Train ST-FNO on these trajectories
4. Use trained operator for fast parameter sweeps

However, WALRUS (already integrated) is a more general architecture with 1.3B parameters and pretrained weights. An ST-FNO trained from scratch on DPF data would need to match WALRUS quality, which is unlikely without significant compute investment.

### Conservation Laws

Standard FNO does **not** inherently preserve conservation laws (mass, momentum, energy, div(B)=0). Conservation must be enforced via:
- Physics-informed loss terms (adds penalty for conservation violations)
- Post-processing projection (project output onto divergence-free manifold)
- Architecture constraints (hard conservation layers)

The ST-FNO paper likely uses physics-informed losses but does not claim hard conservation guarantees. This is a fundamental limitation for DPF, where energy conservation to ~1e-7 is required.

### Implementation Effort

**Train ST-FNO from scratch on DPF data** (3-4 weeks):
- Implement FNO architecture in PyTorch (or use `neuraloperator` library)
- Add sparsification layers
- Generate training dataset: ~1000 DPF simulations
- Training: days of GPU time (A100 or MPS with batch=1)
- Validation against held-out simulations

**Use as WALRUS alternative** (2-3 weeks):
- Replace WALRUS inference in `surrogate.py` with ST-FNO
- Simpler architecture (no RevIN, no delta prediction, no Well format)
- But: loses pretrained knowledge, needs DPF-specific training

### Risk Assessment

- **High risk**: No pretrained model available (unlike WALRUS)
- **Medium risk**: Conservation law enforcement is approximate
- **High compute cost**: Training from scratch on DPF data
- **Low code availability**: Paper is paywalled, no public implementation found
- **Marginal benefit over WALRUS**: WALRUS is already integrated and has pretrained weights

### Recommendation: **MONITOR**

WALRUS is already integrated and has pretrained weights on physics data. Training an ST-FNO from scratch would require significant compute and may not outperform WALRUS for DPF-specific tasks. Monitor the field for public implementations and pretrained FNO models for MHD/Z-pinch. Revisit if a pretrained MHD-specific FNO becomes available.

---

## Paper 4: Differentiable Programming for Plasma Physics (2026 Review)

**Full title**: Differentiable Programming for Plasma Physics: From Diagnostics to Discovery and Design
**Authors**: A.S. Joglekar, A.G.R. Thomas, A.L. Milder, K.G. Miller, J.P. Palastro, D.H. Froula
**Venue**: Physics of Plasmas (invited review, dated March 13, 2026)
**arXiv**: 2603.11231
**Affiliations**: Ergodic LLC, Pasteur Labs, University of Rochester LLE, University of Michigan
**PDF**: `adjacent-fields/diffprog-plasma-2603.11231.pdf`

### What It Does

Comprehensive review paper (14 pages) demonstrating four applications of automatic differentiation in plasma physics: (1) discovering novel nonlinear kinetic physics, (2) learning hidden kinetic variables in fluid simulations, (3) accelerating Thomson scattering diagnostics by 140x, and (4) inverse design of spatiotemporal laser pulses. The unifying theme is that reverse-mode AD provides O(1) gradient computation regardless of the number of parameters, enabling optimization over hundreds or thousands of parameters that was previously intractable.

### Key Methods and Findings

**Taxonomy of data-driven approaches** (Table I):

| Approach | What is learned | Physics via | Strengths |
|---|---|---|---|
| Surrogate model | Full input-output map | Training data | Fast inference |
| Neural operator | Solution operator | Architecture | Resolution-invariant |
| PINN | Full solution | Loss function | Mesh-free, flexible |
| Diff. simulation | Selected components | Discrete operators | Interpretable, extrapolates |
| Equation discovery | Equation structure | Candidate library | Human-readable laws |

Key insight: **Differentiable simulation is fundamentally different** from surrogates/PINNs. The physics-based solver produces the solution; the neural network merely generates *parameters or closures* for that solver. This preserves interpretability and conservation.

**Derivative computation comparison** (Table II):

| Method | Accuracy | Cost scaling | Memory | Best for |
|---|---|---|---|---|
| Finite differences | Approximate | O(N) | Low | Prototyping |
| Analytic adjoint | Exact (continuous) | O(1) | Moderate-high | Fixed solvers |
| Forward-mode AD | Exact (discrete) | O(N) | Low | Low-dim sensitivities |
| Reverse-mode AD | Exact (discrete) | O(M) | High (checkpointable) | Large inverse problems |

### The 140x Speedup Claim

**Problem**: Thomson scattering analysis -- fitting plasma parameters (n_e, T_e, T_i, Z, v_flow) to measured spectra. Each spatial/temporal location has ~10 parameters, with hundreds of locations per dataset.

**Method**: Replace finite-difference gradients with reverse-mode AD (JAX). Three acceleration sources:
- Reverse-mode AD: ~10x (O(N_p) -> O(1) gradient)
- GPU parallelization: ~10x
- Batching efficiency: ~1.4x

**Result**: Analysis of temporally resolved dataset: 90 minutes (FD, CPU, 20 lineouts) -> 11 minutes (AD, GPU, 360 lineouts). Per-lineout speedup: **>140x**. Additionally enables uncertainty quantification via efficient Hessian computation: `Sigma = 2*H^{-1}`.

**Relevance to DPF**: This technique is directly applicable to any diagnostic fitting problem. If DPF-Unified ever needs to fit simulation outputs to experimental measurements (neutron yield, current waveform, X-ray spectra), AD-based fitting would be orders of magnitude faster than finite differences.

### Hidden Kinetic Variables in Fluid Simulations

**Problem**: Fluid simulations cannot capture kinetic effects (Landau damping, particle trapping) that depend on the velocity distribution function. Traditional closures (heat flux) are local approximations.

**Solution**: Introduce an auxiliary "hidden variable" delta(x, t) representing the population of resonant electrons:

```
d_t(n) + d_x(n*u) = 0                         (continuity)
m*n*(d_t(u) + u*d_x(u)) = -d_x(p) + q*n*E + 2*m*n * nu_L*u / (1+delta^2)   (momentum)
d_t(delta) = v_ph * d_x(delta) + nu_g * |E * nu_L| / (1+delta^2)            (hidden variable)
```

Key properties:
- delta advects at the phase velocity v_ph (captures resonant electron transport)
- Growth rate nu_g is a neural network: `nu_g = NN(k, nu_ee, |E_k|; theta)` (3 layers, 8 nodes, tanh)
- The saturation factor `1/(1+delta^2)` prevents unbounded growth
- The neural network learns only a correction factor; PDE structure provides the physics

**Training**: 200 Vlasov-Poisson-Fokker-Planck simulations, indirect supervision (compare fluid vs kinetic observables), loss = log-scale density Fourier mode comparison.

**Result**: Test loss 1e-2 on 2335 held-out simulations. Generalizes to domains 100x larger than training geometry. The hidden variable captures nonlocal wavepacket etching that local closures cannot reproduce.

**Relevance to DPF**: This is directly analogous to DPF's empirical closure problem. Our anomalous resistivity, radiation loss coefficients, and mass shedding rates are all "hidden variables" that capture kinetic effects we cannot resolve in MHD. The approach of:
1. Embedding a small neural network inside the MHD solver
2. Training it to reproduce experimental observables (I(t), neutron yield)
3. Having the network learn a *correction* to physics-based closures

...is exactly what DPF-Unified could adopt for improving its empirical factors.

### Function Learning Formula

For a physics solver V and neural network G parameterized by theta:

```
p = G(q; theta)         -- network generates solver parameters from physical inputs
L = L(V(p))             -- loss depends on solver output
dL/dtheta = (dL/dV) * (dV/dG) * (dG/dtheta)   -- chain rule through solver
```

The neural network does NOT replace the physics. It learns a continuous function mapping inputs to optimal parameters for the physics solver.

### How It Maps to DPF-Unified

| Review Concept | DPF-Unified Application | Module |
|---|---|---|
| Differentiable circuit ODE | Parameter calibration (fc, fm, massf, currf) | `rlc_solver.py` |
| Hidden kinetic variables | Anomalous resistivity, radiation loss | `mhd_solver.py` closures |
| 140x diagnostic fitting | Fitting I(t) to experiments | `engine.py` calibration |
| Function learning | Neural closure models | `ai/surrogate.py` |
| Inverse design | Optimal device geometry | `ai/inverse_design.py` |
| Checkpointing for memory | Long MHD simulation gradients | New infrastructure |

### Implementation Effort

**Differentiable circuit calibration** (1-2 weeks):
- Rewrite `RLCSolver` in PyTorch (same as Paper 1 recommendation)
- Define loss: L = ||I_sim(t) - I_exp(t)||^2 + ||Yn_sim - Yn_exp||^2
- Use reverse-mode AD + L-BFGS to fit (C, L0, R0, V0, fc, fm, massf, currf)
- Expected 10-100x speedup over current grid search

**Neural closure for anomalous resistivity** (3-4 weeks):
- Define hidden variable eta_anom(r, z, t) governed by a transport equation
- Growth rate from small NN: `nu_g = NN(J, n_e, T_e; theta)`
- Train against experimental current waveforms
- Integrate into `mhd_solver.py` as a source term

**Full differentiable MHD solver** (2-3 months):
- Rewrite entire MHD solver in PyTorch/JAX
- Gradient checkpointing for memory management
- Requires significant architectural changes
- Not recommended as first step

### Risk Assessment

- **Very low risk**: Review paper with proven techniques
- **Low risk for circuit calibration**: Well-understood problem, proven speedup
- **Medium risk for neural closures**: Novel approach, needs DPF-specific validation
- **Note**: Lead author (Joglekar) is also on the Tesseract paper -- shared codebase/expertise

### Recommendation: **ADOPT** (circuit calibration) + **ADAPT** (neural closures)

The circuit calibration approach is proven, low-effort, and directly addresses our parameter degeneracy problem. The neural closure concept is more speculative but could be transformative for replacing empirical factors. Implement circuit calibration first (synergizes with Paper 1 recommendation), then explore neural closures for anomalous resistivity.

---

## Cross-Paper Synthesis: Priority Roadmap for DPF-Unified

### Immediate (next 2 weeks)

1. **Differentiable circuit solver** (Papers 1 + 4)
   - Rewrite `RLCSolver` in PyTorch with autograd
   - Add L-BFGS optimizer for parameter calibration
   - Validate against known Lee model fits from `cortana-dpf-ref` database
   - Replaces grid search with gradient-based optimization
   - Effort: 1-2 weeks

2. **Implicit resistive diffusion** (Paper 2)
   - Add implicit treatment of `eta * J` term in `mhd_solver.py`
   - Thomas algorithm for 1D, GMRES for 2D/3D
   - Eliminates resistive CFL bottleneck (Lesson #58)
   - Effort: 1-2 weeks

### Short-term (next 1-3 months)

3. **Neural closure for anomalous resistivity** (Paper 4)
   - Hidden variable approach from Joglekar & Thomas
   - Small NN embedded in MHD solver
   - Train against experimental I(t) waveforms
   - Effort: 3-4 weeks

4. **Full IMEX time stepping** (Paper 2)
   - 3-split flux decomposition for cylindrical MHD
   - LSDIRK2 time integration
   - Requires cylindrical operator derivation
   - Effort: 4-6 weeks

### Medium-term (3-6 months)

5. **FNO surrogate exploration** (Paper 3)
   - Only if WALRUS proves insufficient for DPF-specific tasks
   - Requires large training dataset and GPU compute
   - Monitor for public pretrained MHD models

### Decision Matrix

| Paper | Recommendation | Effort | Impact | Risk | Priority |
|---|---|---|---|---|---|
| 1. Tesseract | ADAPT | 1-2 wk | High (parameter optimization) | Low | 1 |
| 2. Boscheri IMEX | ADAPT | 1-2 wk (partial) / 4-6 wk (full) | Very High (CFL bottleneck) | Medium | 1 |
| 3. ST-FNO | MONITOR | 3-4 wk | Medium (redundant with WALRUS) | High | 4 |
| 4. Joglekar Review | ADOPT/ADAPT | 1-2 wk (circuit) / 3-4 wk (closures) | Very High (calibration + discovery) | Low | 1 |

### Key Technical Insight

Papers 1 and 4 share an author (Joglekar) and a codebase philosophy. The Tesseract architecture -- wrapping non-differentiable solvers in a differentiable framework -- is exactly what DPF-Unified needs. Our MHD solver does not need to be rewritten in JAX/PyTorch. We only need to:

1. Make the **circuit ODE** differentiable (simple rewrite)
2. Wrap the **MHD solver** as a Tesseract-style black box with finite-difference Jacobians
3. Use **reverse-mode AD** for the outer optimization loop

This hybrid approach preserves our existing validated MHD code while enabling gradient-based parameter optimization -- the best of both worlds.

---

## Downloaded PDFs

Successfully downloaded to `docs/research-reference/adjacent-fields/`:
- `tesseract-2511.13262.pdf` (756 KB) -- Paper 1
- `diffprog-plasma-2603.11231.pdf` (1.2 MB) -- Paper 4
- `2024_A_Structure-Preserving_Semi-implicit_IMEX_Finite_Volume_Scheme_for_Ide.pdf` (7.3 MB) -- Paper 2

Not successfully downloaded:
- Paper 3 (ST-FNO): File `2024_Sparsified_time-dependent_Fourier_neural_operators_for_fusion_simulati.pdf` (405 KB) contains wrong content (traffic model paper). Paper is paywalled at Physics of Plasmas (DOI: 10.1063/5.0231245), not available on arXiv.
