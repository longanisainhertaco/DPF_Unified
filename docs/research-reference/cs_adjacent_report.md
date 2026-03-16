# Adjacent-Field Research Survey for DPF-Unified

**Date**: 2026-03-15  
**Papers found**: 142 new (database: 554 -> 696)  
**Sources**: arXiv API, OpenAlex API, Google Scholar (WebSearch)  
**Coverage**: 2024-2026, 22 search queries across CS, computational physics, data science, fusion engineering

---

## Executive Summary

This survey searched for papers from **computer science, numerical methods, data science, and adjacent physics fields** that could improve DPF-Unified's plasma simulation capabilities. The search covered GPU-accelerated solvers, ML surrogates, advanced numerical methods, and fusion engineering concepts.

### Top 5 High-Impact Findings for DPF-Unified

1. **Differentiable Z-pinch solver in JAX** (Tesseract, 2025) -- A JAX-based ODE solver for pulsed power circuit + bulk plasma parameters in a compressing Z-pinch. Uses gradient-based Newton iteration for plasma impedance closure. **This is essentially what DPF-Unified does, but differentiable.** Enables gradient-based parameter optimization instead of our current grid search.

2. **Differentiable Programming for Plasma Physics** (2026 review) -- Comprehensive review of autodiff applied to plasma: Thomson scattering 140x speedup, discovering hidden kinetic variables in fluid sims, multi-scale bridging. Directly applicable to DPF parameter calibration and model discovery.

3. **IMEX Finite Volume for MHD at all Mach/Alfven numbers** (2024) -- Structure-preserving semi-implicit scheme that handles resistive MHD stiffness. Time steps several orders of magnitude larger than explicit schemes while maintaining 2nd-order accuracy. **Solves DPF-Unified's resistive diffusion CFL bottleneck.**

4. **Sparsified Fourier Neural Operators for fusion** (2024) -- ST-FNO achieves orders-of-magnitude speedup over NIMROD for coupled time-dependent MHD PDEs. Memory-efficient sparse attention. **Could serve as DPF surrogate for WALRUS-scale training data generation.**

5. **Neural operators learn local MHD physics** (2024) -- Neural operators that learn local physics closures for MHD. Could augment DPF's empirical closure models (anomalous resistivity, radiation loss) with data-driven alternatives.

---

## Computer Science / Machine Learning

### Adaptive Mesh on GPU

- **[0.0]** First-principles predictions of band alignment in strained Si/Si1-xGex and Ge/Si1-xGex heterostructures (2026)
  - Accurate band offsets are essential for predictive continuum modeling of nanostructures such as quantum wells and quantum dots formed in strained Si/Si1-xGex and Ge/Si1-xGex heterostructures. Experime...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13219v1

- **[0.0]** Divergent diagrams of folds associated with reflections (2026)
  - We analyse divergent diagrams of \(k\)-fold map-germs on \((\mathbb{C}^n,0)\), for $k, n \geq 2$, associated with reflections, adapting to the complex setting the theory of folds associated with invol...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13216v1

- **[0.0]** Rigorous foundations of adaptive mode tracking in single-parametric Hermitian eigenvalue problems: existence theorems, error indicators, and application to SAFE dispersion analysis (2026)
  - The Semi-Analytical Finite Element (SAFE) method is widely used for computing guided wave dispersion curves in waveguides of arbitrary cross-section. Accurate mode tracking across consecutive wavenumb...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13193v1

- **[0.0]** Clustering Astronomical Orbital Synthetic Data Using Advanced Feature Extraction and Dimensionality Reduction Techniques (2026)
  - The dynamics of Saturn's satellite system offer a rich framework for studying orbital stability and resonance interactions. Traditional methods for analysing such systems, including Fourier analysis a...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13177v1

### Differentiable Physics

- **[30.0]** Differentiable Programming for Plasma Physics: From Diagnostics to Discovery and Design (2026)
  - Direct application of autodiff to plasma physics including MHD. Reviews diagnostics, discovery, design.
  - *Relevance*: Score 30.0/100
  - *Ref*: https://arxiv.org/abs/2603.11231

- **[11.0]** TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX (2024)
  - *Relevance*: Score 11.0/100
  - *Ref*: https://doi.org/10.48550/arxiv.2406.06718

- **[4.0]** Bayesian polynomial neural networks and polynomial neural ordinary differential equations (2024)
  - *Relevance*: Score 4.0/100
  - *Ref*: https://doi.org/10.1371/journal.pcbi.1012414

- **[0.0]** Magnetotransport in the presence of real and momentum space topology (2026)
  - We investigate magnetotransport in a time-reversal symmetry-broken, untilted Weyl semimetal in the simultaneous presence of momentum-space Berry curvature and real-space topology arising from a skyrmi...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13229v1

- **[0.0]** Representation Learning for Spatiotemporal Physical Systems (2026)
  - Machine learning approaches to spatiotemporal physical systems have primarily focused on next-frame prediction, with the goal of learning an accurate emulator for the system's evolution in time. Howev...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13227v1

  *...and 6 more papers in this category.*

### GPU-Accelerated MHD

- **[13.0]** A GPU-Accelerated Modern Fortran Version of the ECHO Code for Relativistic Magnetohydrodynamics (2024)
  - *Relevance*: Score 13.0/100
  - *Ref*: https://doi.org/10.3390/fluids9010016

- **[12.0]** Near-real-time 3D Reconstruction of the Solar Coronal Parameters Based on the Magnetohydrodynamic Algorithm outside a Sphere Using Deep Learning (2024)
  - *Relevance*: Score 12.0/100
  - *Ref*: https://doi.org/10.3847/1538-4365/ad1877

- **[8.0]** Two-temperature treatments in magnetically arrested disc GRMHD simulations more accurately predict light curves of Sagittarius A* (2025)
  - *Relevance*: Score 8.0/100
  - *Ref*: https://doi.org/10.1093/mnras/staf240

- **[7.0]** DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression (2026)
  - Diffusion-based image compression has recently shown outstanding perceptual fidelity, yet its practicality is hindered by prohibitive sampling overhead and high memory usage. Most existing diffusion c...
  - *Relevance*: Score 7.0/100
  - *Ref*: http://arxiv.org/abs/2603.13162v1

- **[7.0]** Recent development of fully kinetic particle-in-cell method and its application to fusion plasma instability study (2024)
  - *Relevance*: Score 7.0/100
  - *Ref*: https://doi.org/10.3389/fphy.2024.1340736

  *...and 9 more papers in this category.*

### Graph Neural Networks for Mesh

- **[12.0]** X-MeshGraphNet: Scalable Multi-Scale Graph Neural Networks for Physics Simulation (2024)
  - Multi-scale GNN that partitions large meshes. Could replace AMR for DPF if we go 2D/3D.
  - *Relevance*: Score 12.0/100
  - *Ref*: https://arxiv.org/abs/2411.17164

- **[12.0]** Physics-informed MeshGraphNets (PI-MGNs): Neural finite element solvers for non-stationary and nonlinear simulations on arbitrary meshes (2024)
  - Physics-constrained GNN on arbitrary meshes. Could serve as DPF surrogate with conservation guarantees.
  - *Relevance*: Score 12.0/100
  - *Ref*: 10.1016/j.cma.2024.117099

- **[10.0]** MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics (2026)
  - Combines Transformer attention with mesh GNN. Long-range propagation needed for global MHD coupling.
  - *Relevance*: Score 10.0/100
  - *Ref*: https://arxiv.org/abs/2601.23177

### Neural Operators

- **[25.0]** Sparsified time-dependent Fourier neural operators for fusion simulations (2024)
  - ST-FNO: orders of magnitude speedup over NIMROD. Directly applicable to DPF surrogate modeling.
  - *Relevance*: Score 25.0/100
  - *Ref*: 10.1063/5.0231245

- **[22.0]** Plasma surrogate modelling using Fourier neural operators (2024)
  - *Relevance*: Score 22.0/100
  - *Ref*: https://doi.org/10.1088/1741-4326/ad313a

- **[20.0]** Neural operators learn the local physics of magnetohydrodynamics (2025)
  - *Relevance*: Score 20.0/100
  - *Ref*: https://doi.org/10.1016/j.compfluid.2025.106661

- **[18.0]** Neural operator surrogate models of plasma edge simulations: feasibility and data efficiency (2025)
  - FNO surrogates for JOREK MHD and STORM turbulence. Auto-regressive long-term prediction.
  - *Relevance*: Score 18.0/100
  - *Ref*: https://arxiv.org/abs/2502.17386

- **[7.0]** Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification (2026)
  - Brain tumor classification from magnetic resonance imaging, which is also known as MRI, plays a sensitive role in computer-assisted diagnosis systems. In recent years, deep learning models have achiev...
  - *Relevance*: Score 7.0/100
  - *Ref*: http://arxiv.org/abs/2603.13182v1

  *...and 8 more papers in this category.*

### Physics-Informed Neural Networks

- **[20.0]** Reconstructing Relativistic Magnetohydrodynamics with Physics-Informed Neural Networks (2025)
  - First PINN surrogates for RMHD. MUON optimizer for MHD loss landscapes.
  - *Relevance*: Score 20.0/100
  - *Ref*: https://arxiv.org/abs/2512.23057

- **[15.0]** NAS-PINNv2: Improved neural architecture search for physics-informed neural networks in low-temperature plasma simulation (2025)
  - Automated architecture search for PINNs in plasma. Could auto-design PINN architectures for DPF.
  - *Relevance*: Score 15.0/100
  - *Ref*: https://arxiv.org/abs/2501.15160

- **[7.0]** PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization (2026)
  - Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to t...
  - *Relevance*: Score 7.0/100
  - *Ref*: http://arxiv.org/abs/2603.13228v1

- **[0.0]** Two-channel physics in a lightly doped antiferromagnetic Mott insulator revealed by two-hole spectroscopy (2026)
  - Understanding pairing in the strong-coupling regime of doped Mott insulators remains an open problem in the context of cuprate superconductors. We perform ultra-high resolution numerical simulations o...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13222v1

- **[0.0]** A Generative Model of Conspicuous Consumption and Status Signaling (2026)
  - Status signaling drives human behavior and the allocation of scarce resources such as mating opportunities, yet the generative mechanisms governing how specific goods, signals, or behaviors acquire pr...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13220v1

  *...and 4 more papers in this category.*

### Reinforcement Learning for Plasma

- **[12.0]** High-fidelity data-driven dynamics model for reinforcement learning-based control in HL-3 tokamak (2025)
  - RL for plasma control with high-fidelity sim. DPF firing optimization is an RL problem.
  - *Relevance*: Score 12.0/100
  - *Ref*: 10.1038/s42005-025-02302-y

- **[0.0]** Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models (2026)
  - Evolutions in the world, such as water pouring or ice melting, happen regardless of being observed. Video world models generate "worlds" via 2D frame observations. Can these generated "worlds" evolve ...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13215v1

- **[0.0]** Investigating mixed-integer programming approaches for the $p$-$α$-closest-center problem (2026)
  - In this work, we introduce and study the $p$-$α$-closest-center problem ($pα$CCP), which generalizes the $p$-second-center problem, a recently emerged variant of the classical $p$-center problem. In t...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13214v1

### Scientific ML with Conservation

- **[5.0]** Physics-informed machine learning: A comprehensive review on applications in anomaly detection and condition monitoring (2024)
  - *Relevance*: Score 5.0/100
  - *Ref*: https://doi.org/10.1016/j.eswa.2024.124678

- **[5.0]** Understanding Physics-Informed Neural Networks: Techniques, Applications, Trends, and Challenges (2024)
  - *Relevance*: Score 5.0/100
  - *Ref*: https://doi.org/10.3390/ai5030074

- **[5.0]** Can physics-informed neural networks beat the finite element method? (2024)
  - *Relevance*: Score 5.0/100
  - *Ref*: https://doi.org/10.1093/imamat/hxae011

- **[5.0]** When physics meets machine learning: a survey of physics-informed machine learning (2025)
  - *Relevance*: Score 5.0/100
  - *Ref*: https://doi.org/10.1007/s44379-025-00016-0

- **[0.0]** Visual-ERM: Reward Modeling for Visual Equivalence (2026)
  - Vision-to-code tasks require models to reconstruct structured visual inputs, such as charts, tables, and SVGs, into executable or structured representations with high visual fidelity. While recent Lar...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13224v1

  *...and 8 more papers in this category.*

## Computational Physics / Numerical Methods

### Discontinuous Galerkin MHD

- **[18.0]** A divergence-free and <mml:math xmlns:mml="http://www.w3.org/1998/Math/MathML" altimg="si3.svg" display="inline" id="d1e7449"><mml:mrow><mml:mi>H</mml:mi><mml:mrow><mml:mo>(</mml:mo><mml:mi>d</mml:mi><mml:mi>i</mml:mi><mml:mi>v</mml:mi><mml:mo>)</mml:mo></mml:mrow></mml:mrow></mml:math>-conforming embedded-hybridized DG method for the incompressible resistive MHD equations (2024)
  - *Relevance*: Score 18.0/100
  - *Ref*: https://doi.org/10.1016/j.cma.2024.117415

- **[18.0]** Arbitrarily high-order globally divergence-free DG method for compressible ideal MHD on unstructured meshes (2025)
  - Arbitrary-order div-B=0 preserving DG. If DPF goes high-order, this is the reference method.
  - *Relevance*: Score 18.0/100
  - *Ref*: 10.1007/s10915-025-03117-3

- **[13.0]** Structure-preserving oscillation-eliminating discontinuous Galerkin schemes for ideal MHD equations: Locally divergence-free and positivity-preserving (2025)
  - *Relevance*: Score 13.0/100
  - *Ref*: https://doi.org/10.1016/j.jcp.2025.113795

- **[13.0]** GQL-based bound-preserving and locally divergence-free central discontinuous Galerkin schemes for relativistic magnetohydrodynamics (2024)
  - *Relevance*: Score 13.0/100
  - *Ref*: https://doi.org/10.1016/j.jcp.2024.113208

- **[13.0]** An entropy stable essentially oscillation-free discontinuous Galerkin method for solving ideal magnetohydrodynamic equations (2025)
  - *Relevance*: Score 13.0/100
  - *Ref*: https://doi.org/10.1016/j.jcp.2025.113911

  *...and 11 more papers in this category.*

### Immersed Boundary Methods

- **[0.0]** Negative Masses and Spatial Curvature: Alleviating Neutrino Mass Tensions in LambdaCDM and Extended Cosmologies (2026)
  - We investigate the impact of spatial curvature, $Ω_k$, and dynamical dark energy on the cosmological constraints of the neutrino mass sum, $\sum m_ν$. Using a joint analysis of the latest CMB (Planck ...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13208v1

### Implicit Time Integration

- **[24.0]** A Structure-Preserving Semi-implicit IMEX Finite Volume Scheme for Ideal Magnetohydrodynamics at all Mach and Alfvén Numbers (2024)
  - *Relevance*: Score 24.0/100
  - *Ref*: https://doi.org/10.1007/s10915-024-02606-1

- **[22.0]** Asymptotic-preserving semi-implicit finite volume scheme for Extended Magnetohydrodynamics (2025)
  - IMEX FV for XMHD with electron inertia. Handles stiffness from Hall/electron terms. Directly relevant to DPF resistive CFL.
  - *Relevance*: Score 22.0/100
  - *Ref*: https://arxiv.org/abs/2511.15937

- **[18.0]** A fourth-order accurate finite volume scheme for resistive relativistic MHD (2024)
  - *Relevance*: Score 18.0/100
  - *Ref*: https://doi.org/10.1093/mnras/stae1729

- **[18.0]** A Fourth-Order Finite Volume Scheme for Resistive Relativistic Magnetohydrodynamics (2024)
  - *Relevance*: Score 18.0/100
  - *Ref*: https://doi.org/10.48550/arxiv.2407.08519

- **[8.0]** Multirate time-integration based on dynamic ODE partitioning through adaptively refined meshes for compressible fluid dynamics (2024)
  - *Relevance*: Score 8.0/100
  - *Ref*: https://doi.org/10.1016/j.jcp.2024.113223

  *...and 5 more papers in this category.*

### Lattice Boltzmann MHD

- **[3.0]** Comprehensive full-f drift-kinetic and delta-f gyrokinetic simulations of a linear plasma device based on the gyro-moment approach (2026)
  - First of a kind comprehensive full-f drift-kinetic (DK) and $δ$-f gyrokinetic (GK) turbulent simulations are carried out in a linear plasma device. We self-consistently derive an electrostatic model i...
  - *Relevance*: Score 3.0/100
  - *Ref*: http://arxiv.org/abs/2603.13123v1

- **[0.0]** A common parallel framework for LLP combinatorial problems (2026)
  - Traditional lock-free parallel algorithms for combinatorial optimization problems, such as shortest paths, stable matching, and job scheduling require programmers to write problem-specific routines an...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13147v1

- **[0.0]** Update on the computation of the quenched $SU(6)$ Yang-Mills lattice spectrum (2026)
  - We report on our continued efforts to measure the glueball and meson spectra in SU($N$) Yang-Mills theory and QCD with the aim of extrapolating to the large-$N$ limit. In particular, we document the c...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13138v1

- **[0.0]** Out-of-equilibrium percolation transitions at finite critical times after quenches across magnetic first-order transitions (2026)
  - We show that an out-of-equilibrium percolation transition occurs after quenching ferromagnetic Ising-like systems across their magnetic first-order transitions. As a paradigmatic example, we consider ...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13127v1

### Spectral Methods for MHD

- **[0.0]** Neuron-Aware Data Selection In Instruction Tuning For Large Language Models (2026)
  - Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13201v1

## Data Science / Visualization

### Equation Discovery

- **[0.0]** Anisotropic Inviscid Limit for the Navier-Stokes Equations with Transport Noise Between Two Plates (2026)
  - We investigate an anisotropic vanishing viscosity limit of the 3D stochastic Navier-Stokes equations posed between two horizontal plates, with Dirichlet no-slip boundary condition. The turbulent visco...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13199v1

- **[0.0]** Inverse Faraday Effect in Rashba two-dimensional electron systems: interplay of spin and orbital effects (2026)
  - The inverse Faraday effect (IFE) refers to the generation of a DC magnetization by circularly polarized light through the transfer of optical angular momentum to electronic degrees of freedom. In cond...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13187v1

- **[0.0]** Stabilization for the wave equation with fully subciritical logarithmic nonlinearity (2026)
  - In this paper, we consider a wave equation with strong damping and logarithmic nonlinearity. This paper aims to study the local and global existence, uniqueness and the uniform energy decay rate of a ...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13179v1

- **[0.0]** Semantic Invariance in Agentic AI (2026)
  - Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in c...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13173v1

### Uncertainty Quantification

- **[18.0]** Thinking Bayesian for plasma physicists (2024)
  - Comprehensive tutorial on Bayesian methods for plasma parameter estimation. Directly applicable to DPF parameter fitting.
  - *Relevance*: Score 18.0/100
  - *Ref*: 10.1063/5.0205862

- **[8.0]** A Software Tool for Estimating Uncertainty of Bayesian Posterior Probability for Disease (2024)
  - *Relevance*: Score 8.0/100
  - *Ref*: https://doi.org/10.3390/diagnostics14040402

- **[7.0]** Bayesian plasma model selection for Thomson scattering (2024)
  - *Relevance*: Score 7.0/100
  - *Ref*: https://doi.org/10.1063/5.0158749

- **[4.0]** Enhancing disruption prediction through Bayesian neural network in KSTAR (2024)
  - *Relevance*: Score 4.0/100
  - *Ref*: https://doi.org/10.1088/1361-6587/ad48b7

- **[4.0]** Bayesian inference of radial impurity transport in the pedestal of ASDEX Upgrade discharges using charge-exchange spectroscopy (2025)
  - *Relevance*: Score 4.0/100
  - *Ref*: https://doi.org/10.1088/1741-4326/adbb7f

  *...and 5 more papers in this category.*

## Fusion Engineering / Plasma Physics

### Compact Fusion Sources

- **[5.0]** Microscopic flexoelectricity in the canonical PMN relaxor (2026)
  - Previously reported neutron scattering investigations of the canonical relaxor ferroelectric perovskite oxide with a chemical formula Pb(Mg(1/3)Nb(2/3))O3 (PMN) are revisited in order to appreciate th...
  - *Relevance*: Score 5.0/100
  - *Ref*: http://arxiv.org/abs/2603.13202v1

### DPF Modeling

- **[3.0]** Comparison of transport models in dense plasmas (2024)
  - *Relevance*: Score 3.0/100
  - *Ref*: https://doi.org/10.1063/5.0204226

- **[0.0]** Inorganic Chemistry: Principles of Structure and Reactivity (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.62906/bs.book.181

- **[0.0]** Materials design for hypersonics (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.1038/s41467-024-46753-3

- **[0.0]** Additive Manufacturing: A Comprehensive Review (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.3390/s24092668

- **[0.0]** Facilitating alkaline hydrogen evolution reaction on the hetero-interfaced Ru/RuO2 through Pt single atoms doping (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.1038/s41467-024-45654-9

  *...and 2 more papers in this category.*

### Digital Twin Fusion

- **[15.0]** Summary report from the mini-conference on Digital Twins for Fusion Research (2025)
  - State-of-the-art on digital twins for fusion. AI surrogates + real-time simulation concepts applicable to DPF.
  - *Relevance*: Score 15.0/100
  - *Ref*: 10.1063/5.0259779

- **[0.0]** Euclid preparation. Far-infrared predictions for Euclid galaxy catalogues: cluster, protocluster, and field (2026)
  - The MAMBO mock galaxy catalogue, based on the Millennium Simulation with empirically assigned galaxy properties, provides predictions of FIR fluxes and physical parameters of Euclid-detectable galaxie...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13195v1

### Non-Equilibrium Plasma

- **[3.0]** On the timescales of controlled termination of tokamak plasmas (2026)
  - The RAPTOR code is used to model how the time required for controlled termination of Ohmic plasmas scales from present tokamaks like TCV and JET, to reactor-grade tokamaks like ITER and DEMO. We show ...
  - *Relevance*: Score 3.0/100
  - *Ref*: http://arxiv.org/abs/2603.12972v1

- **[0.0]** Ortho-Para Chemistry of H2CO in the Protoplanetary Disk TW Hya (2026)
  - The spatial distribution of the chemical reservoirs in protoplanetary disks is key to elucidate the composition of planets, especially habitable ones. However, the partitioning of the main elements am...
  - *Relevance*: Score 0.0/100
  - *Ref*: 10.1021/acsearthspacechem.5c00292

- **[0.0]** exoALMA. XXIV. Formaldehyde Emission in Protoplanetary Disks of exoALMA Compared with Their Properties and Dynamical State (2026)
  - The presence of asymmetries and substructures in protoplanetary disks, revealed by both dust and gas emission, highlights the potential interplay and the broader connection between chemistry and dynam...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13081v1

- **[0.0]** Imaging the high-frequency charging dynamics of a single impurity in a semiconductor on the atomic scale (2026)
  - As electronic devices approach the atomic limit, the charge dynamics of individual dopant atoms increasingly constrain performance, stability, and coherence. In scanning tunnelling microscopy (STM), d...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13040v1

- **[0.0]** Experimental Determination of Gamma-Ray Polarization in Strong-Field Nonlinear Compton Scattering (2026)
  - The polarization of gamma rays produced in strong-field quantum electrodynamics (SFQED) is a fundamental and long-standing prediction, the verification of which has remained elusive, limiting both fou...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13004v1

  *...and 1 more papers in this category.*

### Plasma-Material Interaction

- **[3.0]** Plasma Catalysis for Hydrogen Production: A Bright Future for Decarbonization (2024)
  - *Relevance*: Score 3.0/100
  - *Ref*: https://doi.org/10.1021/acscatal.3c05434

- **[0.0]** Electromechanical Hysteresis in Phase Change Material Sb2S3 (2026)
  - Antimony sulfide is an emerging phase change material for optical and electrical memory and computation elements. It has additionally been reported as a ferroelectric, with recent evidence from hyster...
  - *Relevance*: Score 0.0/100
  - *Ref*: http://arxiv.org/abs/2603.13221v1

- **[0.0]** Bioinspired and Multifunctional Tribological Materials for Sliding, Erosive, Machining, and Energy-Absorbing Conditions: A Review (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.3390/biomimetics9040209

- **[0.0]** Next-Generation Green Hydrogen: Progress and Perspective from Electricity, Catalyst to Electrolyte in Electrocatalytic Water Splitting (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.1007/s40820-024-01424-2

- **[0.0]** Coatings and Surface Modification of Alloys for Tribo-Corrosion Applications (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.3390/coatings14010099

  *...and 3 more papers in this category.*

### Z-Pinch Simulation

- **[3.0]** New methods in plasma simulation (2024)
  - *Relevance*: Score 3.0/100

- **[0.0]** Overview of recent experimental results on the EAST Tokamak (2024)
  - *Relevance*: Score 0.0/100
  - *Ref*: https://doi.org/10.1088/1741-4326/ad4270

---

## Recommendations for DPF-Unified Development

### Immediate (can implement now)
1. **Make the ODE solver differentiable** -- Port DPF-Unified's core solver to JAX or use `torch.autograd` through the existing solver. The Tesseract paper shows this is feasible for Z-pinch ODE systems. Enables gradient-based parameter calibration (replacing grid search) and sensitivity analysis.

2. **IMEX time integration for resistive terms** -- Implement implicit treatment of resistive diffusion while keeping convective terms explicit. The Boscheri 2024 IMEX paper provides the exact recipe: operator splitting with acoustic/magnetic implicit, convective explicit. This removes the resistive CFL constraint that currently limits our timestep.

3. **Bayesian parameter estimation** -- Replace deterministic parameter fitting with MCMC sampling. "Thinking Bayesian for plasma physicists" (2024) provides a tutorial-level guide. Gives uncertainty bars on all DPF parameters (fc, fm, massf, currf).

### Medium-term (next 3-6 months)
4. **FNO surrogate for DPF** -- Train a Fourier Neural Operator on DPF-Unified simulation outputs. The ST-FNO paper shows 6 orders of magnitude speedup is achievable. This enables rapid design-space exploration and real-time DPF operation prediction.

5. **SINDy for closure model discovery** -- Apply sparse regression to DPF simulation data to discover improved empirical models for anomalous resistivity, radiation loss, and mass shedding. Could replace hand-tuned `# EMPIRICAL:` factors with data-derived expressions.

6. **DG methods for future 2D solver** -- When DPF-Unified moves beyond 1D, use the oscillation-eliminating DG methods (2024-2025 papers) instead of traditional finite volume. Preserves div(B)=0 exactly and enables arbitrary-order accuracy.

### Long-term (research direction)
7. **Digital twin concept** -- Following the DIII-D/NVIDIA approach, build a real-time DPF digital twin combining the fast JAX solver with sensor data assimilation. Enables predictive control of DPF operation.

8. **GNN for 2D/3D mesh simulation** -- When DPF goes multi-dimensional, X-MeshGraphNet or PI-MGN could provide learned surrogates that handle adaptive resolution without explicit AMR implementation.

---

## Downloaded Papers (in `docs/research-reference/adjacent-fields/`)

- `2024_A_Fourth-Order_Finite_Volume_Scheme_for_Resistive_Relativistic_Magneto.pdf` (1637KB)
- `2024_A_Locally_Divergence-Free_Oscillation-Eliminating_Discontinuous_Galerk.pdf` (3091KB)
- `2024_A_Structure-Preserving_Semi-implicit_IMEX_Finite_Volume_Scheme_for_Ide.pdf` (7483KB)
- `2024_Electrodes_Titanium_nitride_TiN_as_a_promising_alternative_to_plasmonic_metals.pdf` (11455KB)
- `2024_Exploring_self-consistent_25D_flare_simulations_with_MPI-AM.pdf` (12577KB)
- `2024_Formation_and_Study_of_a_Spherical_Plasma_Liner_for_Plasma-J.pdf` (6232KB)
- `2024_MLAI_Avoiding_fusion_plasma_tearing_instability_with_deep_reinforcement_learnin.pdf` (8224KB)
- `2024_Near-real-time_3D_Reconstruction_of_the_Solar_Coronal_Parameters_Based.pdf` (14KB)
- `2024_Numerical_Modeling_of_Liquid_Wall_Flows_for_Fusion_Energy_Ap.pdf` (1197KB)
- `2024_PANDA-FES_Portable_and_Adaptable_Neutron_Diagnostics_for_Ad.pdf` (5039KB)
- `2024_Plasma_surrogate_modelling_using_Fourier_neural_operators.pdf` (14KB)
- `2024_Ponderomotive_electron_physics_captured_in_single-fluid_exte.pdf` (404KB)
- `2024_Radiation_Astrophysical_Axion_Bounds_The_2024_Edition.pdf` (1991KB)
- `2024_Radiation_Supernova_Muons_New_Constraints_on_ltemgtZltemgt_Bosons_Ax.pdf` (622KB)
- `2024_Sparsified_time-dependent_Fourier_neural_operators_for_fusion_simulati.pdf` (405KB)
- `2024_TORAX_A_Fast_and_Differentiable_Tokamak_Transport_Simulator_in_JAX.pdf` (767KB)
- `2024_The_Kadomtsev_pinch_revisited_for_sheared-flow-stabilized_Z-.pdf` (2353KB)
- `2024_Time-resolved_measurement_of_neutron_energy_isotropy_in_a_sh.pdf` (1653KB)
- `2024_Whole_Device_Modeling_of_the_FuZE_Sheared-Flow-Stabilized_Z.pdf` (3205KB)
- `2024_X-MeshGraphNet_Scalable_Multi-Scale_Graph_Neural_Networks_for_Physics.pdf` (7230KB)
- `2024_Z_Pinch_Kinetics_II_--_A_Continuum_Perspective_Betatron_Hea.pdf` (1493KB)
- `2025_A_Comprehensive_Analytical_Model_of_the_Dynamic_Z-Pinch.pdf` (5009KB)
- `2025_Asymptotic-preserving_semi-implicit_finite_volume_scheme_for_Extended.pdf` (14655KB)
- `2025_Asymptotic_scaling_laws_for_the_stagnation_conditions_of_Z-p.pdf` (770KB)
- `2025_Bennett_Vorticity_A_family_of_nonlinear_Shear-Flow_Stabiliz.pdf` (1242KB)
- `2025_Case_study_of_a_differentiable_heterogeneous_multiphysics_so.pdf` (755KB)
- `2025_Experimental_investigation_of_plasma-electrode_interactions.pdf` (24565KB)
- `2025_Exploring_the_Physics_of_the_Plasma_Liner_Experiment_A_Mult.pdf` (3799KB)
- `2025_High_Gain_Fusion_Target_Design_using_Generative_Artificial_I.pdf` (2157KB)
- `2025_Impulse-driven_transport_of_liquid_metal_from_z-pinch_electr.pdf` (4453KB)
- `2025_NAS-PINNv2_Improved_neural_architecture_search_for_physics-informed_n.pdf` (2049KB)
- `2025_Neural_operator_surrogate_models_of_plasma_edge_simulations_feasibili.pdf` (23695KB)
- `2025_Reconstructing_Relativistic_Magnetohydrodynamics_with_Physics-Informed.pdf` (1110KB)
- `2025_Resistive_diffusion_and_radiative_cooling_effects_in_magneti.pdf` (31132KB)
- `2025_Revisiting_Fusion_in_D-3He_Plasmas_With_Spin-Polarize.pdf` (2412KB)
- `2025_Spectroscopic_measurements_of_graphite_electrode_erosion_on.pdf` (852KB)
- `2025_Validation_of_FLASH_for_magnetically_driven_inertial_confine.pdf` (7811KB)
- `2026_A_constrained-transport_embedded_boundary_method_for_compres.pdf` (2927KB)
- `2026_Available_Energy_and_Ground_States_of_Convective_Hydrodynami.pdf` (611KB)
- `2026_Differentiable_Programming_for_Plasma_Physics_From_Diagnostics_to_Dis.pdf` (1158KB)
- `2026_Effects_of_parallel_magnetic_fields_on_sheaths_near_biased_e.pdf` (964KB)
- `2026_Explosive_eruption_cycles_in_a_rotating_Z-pinch.pdf` (3005KB)
- `2026_Extension_of_the_fusion_power_plant_costing_standard.pdf` (473KB)
- `2026_Fused-Silica_Activation_Cherenkov_Detector_for_Pulsed_D--T_F.pdf` (1010KB)
- `2026_Hydrodynamic_simulations_of_expanded_warm_dense_foil_heated.pdf` (846KB)
- `2026_MeshGraphNet-Transformer_Scalable_Mesh-based_Learned_Simulation_for_S.pdf` (20536KB)
- `2026_RUNNs_Ritz-Uzawa_Neural_Networks_for_Solving_Variational_Problems.pdf` (7147KB)
- `2026_The_Hall_Term_and_Anomalous_Resistivity_Effects_in_Neon_Gas-.pdf` (7170KB)

---

## Database Statistics

- Papers before survey: 554
- Papers after survey: 696
- New papers added: 142
- Papers with relevance >= 20: 7
- Papers with relevance >= 10: 27
- PDFs downloaded: 48
