# CS Deep Dive Part 2: GNNs, Equation Discovery, and Uncertainty Quantification

**Date**: 2026-03-15
**Papers found**: 15 new (database: 710 -> 725)
**Sources**: arXiv, OpenAlex, WebSearch, journal sites
**Coverage**: 2021-2026, focused on three transformative CS approaches for DPF simulation

---

## Executive Summary

Three computational science approaches could fundamentally transform DPF-Unified:

1. **Graph Neural Networks** -- Learn mesh-based physics simulation with 10-100x speedup. Conservation-preserving variants exist. Direct application: replace AMR with learned adaptive resolution.
2. **SINDy/Symbolic Regression** -- Automatically discover governing equations from data. Could find new DPF scaling laws from our 602 experimental data points that the I^4 assumption misses.
3. **Uncertainty Quantification** -- Bayesian inference + polynomial chaos expansion to systematically quantify how parameter uncertainty (fc +/-10%, fm +/-5%) propagates to outputs (Yn, Ipeak). Addresses PhD panel gap.

---

## Topic 1: Graph Neural Networks for Mesh-Based Simulation

### Background

Graph Neural Networks represent simulation meshes as graphs (nodes = cells/particles, edges = connectivity) and learn physics via message-passing. The foundational work is MeshGraphNets (Pfaff et al., ICLR 2021), which introduced the encoder-processor-decoder architecture for mesh-based simulation.

### Best Papers Found

| # | Paper | Year | DOI | Key Contribution |
|---|-------|------|-----|------------------|
| 1 | Learning Mesh-Based Simulation with Graph Networks | 2021 | 10.48550/arXiv.2010.03409 | Foundational MeshGraphNets: 10-100x speedup |
| 2 | X-MeshGraphNet: Scalable Multi-Scale GNNs | 2024 | 10.48550/arXiv.2411.17164 | Graph partitioning + halo for scalability |
| 3 | PI-MGNs: Physics-Informed MeshGraphNets | 2024 | 10.48550/arXiv.2402.10681 | PDE-constrained training, no labeled data needed |
| 4 | Learning 1D Plasma Dynamics with GNNs | 2024 | 10.1088/2632-2153/ad4ba6 | GNN for plasma kinetics: thermalization, Landau damping |
| 5 | Dynami-CAL GraphNet | 2025 | 10.48550/arXiv.2501.07373 | Conservation of linear + angular momentum in GNNs |
| 6 | MeshGraphNet-Transformer | 2026 | 10.48550/arXiv.2601.23177 | Transformer global processor for long-range interactions |
| 7 | Conservation-informed Graph Learning | 2024 | 10.48550/arXiv.2412.20962 | Mass/momentum/energy conservation in graph learning |

### Key Algorithms Explained

**MeshGraphNets (Encoder-Processor-Decoder)**:
1. **Encoder**: Maps mesh nodes/edges to latent vectors (128-dim typical). Node features = field values (rho, v, B, T). Edge features = relative position, mesh connectivity.
2. **Processor**: M rounds of message passing (M=5-15). Each round: aggregate neighbor messages, update node/edge latent vectors. Information propagates M hops per step.
3. **Decoder**: Maps latent vectors back to field updates (delta_rho, delta_v, etc.). Output is acceleration/rate-of-change, integrated with semi-implicit Euler.

**Why message-passing works for PDEs**: A single message-passing step is analogous to a finite-difference stencil operation. M steps = M-wide stencil. The GNN learns the effective stencil weights from data.

**Conservation-preserving GNNs (Dynami-CAL GraphNet)**:
- Enforce pairwise conservation via antisymmetric message functions: `m_ij = -m_ji`
- Edge-local reference frames that are equivariant to rotations, invariant to translations
- Result: net force on system = 0, net torque = 0, by construction
- Stable error accumulation over 1000+ rollout steps vs exponential divergence for unconstrained GNNs

**MeshGraphNet-Transformer (2026)**:
- Replaces deep message-passing stack with physics-attention Transformer
- All nodes updated simultaneously -- captures long-range interactions in one step
- Handles industrial-scale meshes (100K+ nodes) where standard MGN message-passing under-reaches

### How It Maps to DPF-Unified

**Direct application: Learned AMR alternative**
- DPF has an experimental AMR module (`src/dpf/experimental/amr/`, 756 LOC) that needs solver refactoring for non-uniform grids
- A GNN naturally handles variable resolution: dense nodes where physics is interesting (pinch column), sparse elsewhere
- Train on high-resolution DPF simulations, deploy as fast surrogate with adaptive resolution

**Training data generation**:
- Use existing Python/Athena++/AthenaK engines to generate training trajectories
- Need ~1000-10000 trajectories across parameter space (device geometry, fill pressure, voltage)
- Each trajectory: ~100 timesteps of (rho, v, B, T) on mesh
- Estimate: 50-200 GPU-hours for data generation, 10-50 GPU-hours for training

**Conservation concern**:
- Standard MeshGraphNets do NOT conserve mass/momentum/energy by construction
- Dynami-CAL GraphNet and conservation-informed methods fix this but add ~20% training cost
- For DPF, conservation is critical -- use conservation-enforcing variant

**Accuracy vs classical solvers**:
- MeshGraphNets: 10-100x speedup, ~1-5% L2 error vs ground truth for aerodynamics/cloth
- For plasma: Carvalho et al. (2024) achieve "considerably small" MAE for 1D plasma kinetics
- NOT a replacement for high-fidelity simulation -- a fast surrogate for parameter sweeps, real-time prediction, inverse design
- Think of it as a smarter version of what WALRUS does, but mesh-aware

### Python Packages

| Package | Install | Notes |
|---------|---------|-------|
| PyTorch Geometric (PyG) | `pip install torch-geometric` | De facto GNN library. MessagePassing base class. |
| NVIDIA Modulus | `pip install nvidia-modulus` | X-MeshGraphNet, physics-informed training |
| DGL (Deep Graph Library) | `pip install dgl` | Alternative to PyG, Facebook-backed |
| torch-cluster, torch-scatter | `pip install torch-cluster torch-scatter` | Required by PyG for efficient message passing |
| meshio | `pip install meshio` | Mesh I/O: VTK, Gmsh, XDMF -> graph conversion |

### Implementation Effort

- **Phase 1** (2-3 sessions): Data pipeline -- convert DPF simulation output to graph format. mesh -> PyG Data objects with node features (rho, v, B, T), edge features (dr, dz), edge connectivity from mesh topology.
- **Phase 2** (3-4 sessions): Train baseline MeshGraphNet on 2D cylindrical DPF simulations. Encoder-processor-decoder with 128-dim latent, 10 message-passing steps.
- **Phase 3** (2-3 sessions): Add conservation constraints (Dynami-CAL style antisymmetric messages). Validate mass/energy conservation.
- **Phase 4** (2 sessions): Integrate as surrogate engine alongside WALRUS in `src/dpf/ai/`.
- **Total: 9-12 sessions**

### Recommendation

**Priority: MEDIUM-HIGH.** The GNN approach is more natural for mesh-based physics than WALRUS (which requires fixed-grid Well format). However, WALRUS is already integrated and has a 4.8GB pretrained checkpoint. Recommend implementing GNN as a second surrogate option in Phase K, focusing on the conservation-preserving variant. The real win is if GNN can replace the dormant AMR module -- learned adaptive resolution without solver refactoring.

---

## Topic 2: SINDy / PDE-FIND -- Automated Equation Discovery

### Background

Sparse Identification of Nonlinear Dynamics (SINDy) discovers governing equations from data by performing sparsity-promoting regression on a library of candidate nonlinear functions. Given time-series data X(t), it finds the sparsest f such that dX/dt = f(X). For PDEs, PDE-FIND extends this to discover spatial operators.

### Best Papers Found

| # | Paper | Year | DOI | Key Contribution |
|---|-------|------|-----|------------------|
| 1 | Data-driven discovery of turbulent convection equations | 2025 | 10.48550/arXiv.2505.10109 | SINDy vs SPIDER on Rayleigh-Benard DNS |
| 2 | PDE-LEARN: Deep learning PDE discovery from noisy data | 2024 | 10.48550/arXiv.2212.04971 | Rational NN + sparse vector for PDE identification |
| 3 | Mesh-free SINDy | 2025 | 10.48550/arXiv.2505.16058 | NN + autodiff, arbitrary sensor placement |
| 4 | Symbolic Regression from Simulations to Scaling Laws | 2025 | 10.48550/arXiv.2511.08784 | PySR discovers compact scaling laws from simulations |
| 5 | GS-SINDy: Earth-Mover distance group similarity | 2025 | Chaos journal | Robust parametric SINDy identification |
| 6 | PySINDy: A Python package for SINDy | 2020 | 10.48550/arXiv.2004.08424 | Reference implementation (v2.1+) |

### Key Algorithms Explained

**SINDy (Brunton et al., 2016)**:
1. Collect measurement data: X(t) = [x1(t), x2(t), ..., xn(t)] at times t1..tN
2. Compute time derivatives: dX/dt (finite differences or smoothed differentiation)
3. Build candidate library: Theta(X) = [1, X, X^2, X*Y, sin(X), ...] -- all plausible nonlinear terms
4. Sparse regression: dX/dt = Theta(X) * Xi, where Xi is sparse (most coefficients = 0)
5. Result: human-readable equations with only the dominant terms

**Example**: Given oscillator data, SINDy recovers `dx/dt = y`, `dy/dt = -x - 0.5*y` from raw time series.

**PDE-FIND**: Same idea but for spatial PDEs. Library includes spatial derivatives: u_x, u_xx, u*u_x, etc.

**PySR (Symbolic Regression)**:
- Genetic programming approach: evolve mathematical expressions
- Fitness = accuracy + parsimony (shorter expressions preferred)
- Discovers entirely new functional forms (not limited to predefined library)
- PySR benchmark: "most suitable for inferring equations" across 9 dynamical systems

### How It Maps to DPF-Unified

**Application 1: Discover new neutron yield scaling laws**

Current assumption: Yn ~ I_pinch^4 (Lee model). But Auluck and others suggest this is incomplete. We have 602 experimental data points across 15 devices.

SINDy/PySR approach:
```
Input features: I_pinch, V0, C, L, P_fill, a_anode, z_anode, gas_type, ...
Target: Yn (neutron yield)
```

Could discover: `Yn = A * I^alpha * P^beta * (a/z)^gamma * f(gas)` where alpha != 4, or entirely non-power-law forms.

**Application 2: Discover anomalous resistivity models**

Our anomalous resistivity is empirical (Bohm-like or threshold-based). SINDy could discover the actual relationship from high-fidelity kinetic simulation data:
```
Input: j, n_e, T_e, B, drift_velocity
Target: eta_anomalous
```

**Application 3: Discover radiation loss models**

Current radiation model uses fitted power-law cooling functions. PDE-FIND could discover more accurate models from detailed radiation transport calculations.

**Data requirements**:
- SINDy needs clean time-series data. Our 602 experimental points are cross-sectional (different devices, different conditions), not time-series. This limits standard SINDy.
- PySR (symbolic regression) works directly on cross-sectional data -- better fit for our use case.
- For time-series discovery (anomalous resistivity model), need simulation output: ~100+ trajectories with 1000+ time points each.

**Accuracy of discovered equations**:
- Wareing et al. (2025): SINDy recovers Rayleigh-Benard equations accurately at low Ra, degrades at high Ra
- PySR: discovered neutron star merger scaling laws that "outperform existing fitting formulae"
- Typical: correct functional form with coefficients accurate to 1-10%

### Python Packages

| Package | Install | Notes |
|---------|---------|-------|
| PySINDy | `pip install pysindy` | Official SINDy implementation. v2.1+. Weak-form, ensemble, constrained. |
| PySR | `pip install pysr` | Symbolic regression via genetic programming. Julia backend. |
| DEAP | `pip install deap` | Genetic programming framework (alternative to PySR) |
| gplearn | `pip install gplearn` | Scikit-learn compatible symbolic regression |
| Dedalus | `pip install dedalus` | Spectral PDE solver (for generating training data) |

### Implementation Effort

- **Phase 1** (1-2 sessions): PySR on existing 602 experimental data points. Feature engineering: I_pinch, V0, C, L0, P_fill, a_anode, z_anode -> Yn. Compare discovered law vs I^4.
- **Phase 2** (2-3 sessions): PySINDy on DPF simulation time-series output. Library: polynomial + trig + custom plasma terms. Target: discover effective closure models.
- **Phase 3** (1-2 sessions): Validate discovered equations against held-out experimental data. Cross-validate across devices.
- **Phase 4** (1 session): Integrate discovered scaling law into `src/dpf/presets.py` as alternative yield predictor.
- **Total: 5-8 sessions**

### Recommendation

**Priority: HIGH.** This is the lowest-hanging fruit of the three topics. Phase 1 (PySR on 602 data points) can be done in a single session and could immediately reveal whether the I^4 scaling is correct or if a better law exists. The required data already exists in `cortana-dpf-ref` database (391 experimental data points + additional data). PySR is the right tool (not SINDy) because our data is cross-sectional, not time-series.

---

## Topic 3: Uncertainty Quantification for Simulation

### Background

Uncertainty Quantification (UQ) systematically propagates input parameter uncertainties through a computational model to quantify output uncertainty. For DPF, the key question is: "If fc = 0.7 +/- 10%, fm = 0.15 +/- 20%, how uncertain is the predicted neutron yield?"

### Best Papers Found

| # | Paper | Year | DOI | Key Contribution |
|---|-------|------|-----|------------------|
| 1 | Physics-constrained PCE for scientific ML and UQ | 2024 | 10.48550/arXiv.2402.15115 | PDE-constrained polynomial chaos surrogates |
| 2 | Epistemic/Aleatoric UQ in Plasma Simulations | 2023 | 10.48550/arXiv.2306.07913 | GP regression surrogate for tokamak turbulence UQ |
| 3 | Thinking Bayesian for Plasma Physicists | 2024 | 10.1063/5.0205668 | Tutorial: Bayesian methods for plasma parameter inference |
| 4 | UQ for Multiscale Fusion Plasma (VECMA) | 2020 | 10.1007/978-3-030-50436-6_53 | EasyVVUQ + Sobol sensitivity for fusion workflows |
| 5 | Hybrid PCE-GPR Bayesian UQ | 2025 | ScienceDirect | Closed-form PCE coefficients via GPR kernels |

### Key Methods Explained

**Monte Carlo UQ (simplest)**:
1. Define parameter distributions: fc ~ N(0.7, 0.07), fm ~ N(0.15, 0.03)
2. Sample N parameter sets (N = 1000-10000)
3. Run simulation for each sample
4. Compute output statistics: mean(Yn), std(Yn), 95% CI
- **Cost**: N full simulations. With DPF-Unified (~1s per Lee model run), 10000 runs = ~3 hours. Feasible.

**Polynomial Chaos Expansion (PCE)**:
1. Expand output as polynomial of input uncertainties: Yn(fc, fm) = sum(a_ij * P_i(fc) * P_j(fm))
2. Compute coefficients a_ij via quadrature (non-intrusive) or Galerkin projection (intrusive)
3. Once fitted, extract mean, variance, Sobol indices analytically from coefficients
4. **Cost**: (p+1)^d evaluations for full tensor product, where p=polynomial order, d=number of uncertain params. For p=3, d=2: 16 simulations. Orders of magnitude cheaper than Monte Carlo.
- **Accuracy**: Exponential convergence for smooth response surfaces

**Sobol Sensitivity Analysis**:
1. Decompose output variance: Var(Yn) = V_fc + V_fm + V_fc_fm
2. First-order index: S_fc = V_fc / Var(Yn) -- fraction of variance due to fc alone
3. Total-order index: ST_fc = 1 - V_~fc / Var(Yn) -- includes interaction effects
4. **Interpretation**: If S_fc >> S_fm, focus calibration effort on fc
- **Cost**: (d+2)*N evaluations (Saltelli scheme), where d=params, N=samples. For d=2, N=1000: 4000 simulations.

**Bayesian Parameter Inference**:
1. Prior: p(fc, fm) -- what we believe before seeing data
2. Likelihood: p(data | fc, fm) -- how likely is the experimental data given these parameters
3. Posterior: p(fc, fm | data) ~ p(data | fc, fm) * p(fc, fm) -- updated belief
4. Use MCMC (Markov Chain Monte Carlo) to sample the posterior
5. **Result**: Joint probability distribution over (fc, fm) -- includes correlations and multi-modality
- **Cost**: 10000-100000 likelihood evaluations. Each evaluation runs the DPF model once.

**Gaussian Process Regression Surrogate** (for expensive models):
1. Run model at ~50-100 carefully chosen parameter combinations (Latin Hypercube Sampling)
2. Fit GP to the input-output mapping
3. GP provides mean prediction + uncertainty estimate at any new point
4. Use GP as cheap surrogate for Monte Carlo/PCE/Bayesian inference
- **Key advantage**: quantifies surrogate model uncertainty alongside parameter uncertainty

### How It Maps to DPF-Unified

**Immediate application: Lee model parameter UQ**

The Lee model has two fitted parameters (fc, fm) that control mass sweep-up and axial speed. Currently calibrated via grid search. UQ would answer:
- What is the 95% confidence interval on Yn given fc, fm uncertainties?
- Which parameter matters more (Sobol indices)?
- Are fc and fm correlated in the posterior (they almost certainly are)?
- Is the Lee model identifiable from I(t) data alone (parameter degeneracy)?

**Implementation plan**:
1. Define priors on fc (0.6-0.9) and fm (0.05-0.3) from literature bounds
2. Run PCE with 25 quadrature points (5x5 grid in fc-fm space)
3. Extract Sobol indices: S_fc, S_fm, S_fc_fm for each output (Ipeak, t_pinch, Yn)
4. Run Bayesian inference with PF-1000 experimental data (24 shots from Akel 2021)

**Broader application: Multi-device calibration UQ**

For each of the 15 devices in our database, propagate circuit parameter uncertainties (C +/- 5%, L0 +/- 10%, R0 +/- 20%) through the model. Identify which devices have well-constrained parameters and which have large UQ bounds (indicating model inadequacy).

**PhD panel gap**: This directly addresses the "no uncertainty quantification" critique. A PCE + Sobol analysis can be included as a section in any thesis/paper.

### Python Packages

| Package | Install | Notes |
|---------|---------|-------|
| SALib | `pip install SALib` | Sobol, Morris, FAST sensitivity analysis. Mature, well-documented. |
| UQpy | `pip install UQpy` | Full UQ toolkit: sampling, surrogates, reliability, sensitivity. |
| PyMC | `pip install pymc` | Bayesian inference with MCMC/NUTS. Best-in-class for posterior sampling. |
| Chaospy | `pip install chaospy` | Polynomial chaos expansion. Non-intrusive PCE with arbitrary distributions. |
| OpenTURNS | `pip install openturns` | Industrial UQ: PCE, Kriging, Sobol, reliability. French CEA/EDF origin. |
| GPy | `pip install GPy` | Gaussian Process regression. Sheffield. |
| EasyVVUQ | `pip install easyvvuq` | VECMA project. Workflow-based UQ for HPC simulations. |
| scipy.stats.sobol_indices | (built-in) | SciPy 1.9+ has native Sobol sensitivity analysis. |

### Implementation Effort

- **Phase 1** (1 session): SALib Sobol analysis on Lee model. 2 params (fc, fm), 3 outputs (Ipeak, t_pinch, Yn). Saltelli scheme with N=1024. ~4000 model evaluations = ~1 hour.
- **Phase 2** (1-2 sessions): Chaospy PCE surrogate. Order-3 polynomial, 2D parameter space. 16 training evaluations. Extract mean, variance, Sobol indices analytically.
- **Phase 3** (2-3 sessions): PyMC Bayesian inference with PF-1000 24-shot data. NUTS sampler, 4 chains, 5000 samples each. Posterior over (fc, fm) with uncertainty.
- **Phase 4** (1 session): Extend to multi-parameter UQ: fc, fm, R0, L0, C, P_fill (6D). Requires GP surrogate due to curse of dimensionality.
- **Phase 5** (1 session): Generate publication-quality UQ figures: posterior contours, Sobol bar charts, uncertainty bands on I(t) waveforms.
- **Total: 6-8 sessions**

### Recommendation

**Priority: HIGHEST.** This directly addresses the PhD panel's identified gap. Phase 1 (SALib Sobol) can be done in a single session with zero new dependencies (SALib is pure Python). The Lee model runs in ~1 second, so even Monte Carlo with 10000 samples is computationally trivial. This is the fastest path to a publishable result and the most defensible contribution to any thesis chapter on DPF simulation.

---

## Cross-Topic Synergies

These three approaches combine powerfully:

1. **SINDy + UQ**: Discover new scaling law with PySR, then run UQ on the discovered equation to quantify confidence in the new law vs I^4.

2. **GNN + UQ**: Use GNN surrogate for fast UQ propagation. A GNN that runs 100x faster than the full solver enables Monte Carlo with 1M samples instead of 10K.

3. **SINDy + GNN**: Discover closure models with SINDy, embed them as physics constraints in GNN training (PI-MGN approach).

4. **All three**: GNN provides fast forward model -> PyMC uses GNN as likelihood evaluator -> posterior over parameters -> SINDy discovers simplified analytical model from posterior samples.

---

## Summary Table

| Topic | Priority | Sessions | Dependencies | First Result |
|-------|----------|----------|-------------|--------------|
| UQ (Sobol/PCE/Bayesian) | HIGHEST | 6-8 | SALib, Chaospy, PyMC | 1 session (Sobol) |
| SINDy/Symbolic Regression | HIGH | 5-8 | PySR, PySINDy | 1 session (PySR on data) |
| GNN for Mesh Simulation | MEDIUM-HIGH | 9-12 | PyG, torch-geometric | 3 sessions (data pipeline + training) |

**Recommended execution order**: UQ first (PhD panel gap, fastest to publishable result), then SINDy (novel contribution, existing data), then GNN (most complex, builds on WALRUS infrastructure).

---

## References

### Topic 1: Graph Neural Networks
1. Pfaff et al. (2021). "Learning Mesh-Based Simulation with Graph Networks." ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
2. Nabian et al. (2024). "X-MeshGraphNet: Scalable Multi-Scale GNNs for Physics Simulation." [arXiv:2411.17164](https://arxiv.org/abs/2411.17164)
3. Iparraguirre et al. (2026). "MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation." [arXiv:2601.23177](https://arxiv.org/abs/2601.23177)
4. Wuerth et al. (2024). "PI-MGNs: Physics-Informed MeshGraphNets." [arXiv:2402.10681](https://arxiv.org/abs/2402.10681)
5. Carvalho et al. (2024). "Learning plasma dynamics with GNNs." [DOI:10.1088/2632-2153/ad4ba6](https://doi.org/10.1088/2632-2153/ad4ba6)
6. (2025). "Dynami-CAL GraphNet: Conservation GNN." Nature Communications. [arXiv:2501.07373](https://arxiv.org/abs/2501.07373)
7. (2024). "Conservation-informed Graph Learning." [arXiv:2412.20962](https://arxiv.org/abs/2412.20962)

### Topic 2: SINDy / Equation Discovery
8. Wareing et al. (2025). "Data-driven discovery of turbulent convection equations." GAFD. [arXiv:2505.10109](https://arxiv.org/abs/2505.10109)
9. (2024). "PDE-LEARN: Deep Learning PDE Discovery." Neural Networks. [arXiv:2212.04971](https://arxiv.org/abs/2212.04971)
10. (2025). "Mesh-free SINDy." [arXiv:2505.16058](https://arxiv.org/abs/2505.16058)
11. (2025). "Symbolic Regression: Scaling Laws in Neutron Star Mergers." [arXiv:2511.08784](https://arxiv.org/abs/2511.08784)
12. (2025). "GS-SINDy: Earth-Mover Distance Group Similarity." Chaos.

### Topic 3: Uncertainty Quantification
13. Sharma et al. (2024). "Physics-constrained PCE for UQ." [arXiv:2402.15115](https://arxiv.org/abs/2402.15115)
14. Yudin et al. (2023). "Epistemic/Aleatoric UQ in Plasma Simulations." [arXiv:2306.07913](https://arxiv.org/abs/2306.07913)
15. Kruger et al. (2024). "Thinking Bayesian for Plasma Physicists." Phys. Plasmas. [DOI:10.1063/5.0205668](https://doi.org/10.1063/5.0205668)

### Software References
- PySINDy: [github.com/dynamicslab/pysindy](https://github.com/dynamicslab/pysindy)
- PySR: [github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR)
- SALib: [salib.readthedocs.io](https://salib.readthedocs.io)
- Chaospy: [chaospy.readthedocs.io](https://chaospy.readthedocs.io)
- PyMC: [pymc.io](https://www.pymc.io)
- PyG: [pyg.org](https://pyg.org)
- NVIDIA Modulus: [developer.nvidia.com/modulus](https://developer.nvidia.com/modulus)
- EasyVVUQ: [easyvvuq.readthedocs.io](https://easyvvuq.readthedocs.io)
- UQpy: [sites.google.com/site/jhusurg/UQpy](https://sites.google.com/site/jhusurg/UQpy)
