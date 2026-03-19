# DPF, MHD, and Z-Pinch Literature Survey: 2025-2026

Research scan conducted 2026-03-18. Only verified, actually-found papers and results included. No fabricated citations.

---

## 1. Dense Plasma Focus Papers (2025-2026)

### 1.1 Goyon et al. -- Neutron Generation Dynamics in MJOLNIR (March 2025)

- **Title**: Neutron generation dynamics inside a MA-class dense plasma focus Z-pinch
- **Authors**: C. Goyon et al. (LLNL)
- **Journal**: Physics of Plasmas, Vol. 32, Issue 3, 033105 (March 1, 2025)
- **URL**: https://pubs.aip.org/aip/pop/article/32/3/033105/3339917
- **OSTI**: https://www.osti.gov/pages/biblio/2531031
- **Summary**: Presents MHD-kinetic description of neutron generation in the MJOLNIR DPF. Uses 2D PIC simulations benchmarked against visible, X-ray, and neutron emissions. Demonstrates that BOTH thermonuclear and beam-target mechanisms contribute to neutron yield in MJ-class DPFs. Neutron yields up to 1.2x10^12 at 3.7 MA with 3.8 mm FWHM spot size. Critically, shows neutron yield scales with current BEYOND the previously observed saturation/drop at MA level.
- **Key data**: At stagnation, spectrum is narrow around 2.45 MeV (thermonuclear). As pinch evolves, spectrum broadens to 5 MeV (beam-target). Optimal Kr doping ~0.1% by volume at MA level.
- **Relevance to DPF-Unified**: **HIGH** -- Provides validation data for our MJOLNIR preset. Confirms that both neutron mechanisms must be modeled. PIC simulation benchmarks could validate our MHD solver dynamics. The current-scaling result is directly relevant to our neutron yield module.

### 1.2 MJOLNIR Rebuild and High Current Experiments (2024, published late 2024)

- **Title**: MegaJOuLe Neutron Imaging Radiography (MJOLNIR) Dense Plasma Focus Rebuild and High Current Experiments
- **Authors**: LLNL team
- **Journal**: IEEE Transactions on Plasma Science (2024)
- **URL**: https://ieeexplore.ieee.org/document/10705961/
- **Summary**: Documents MJOLNIR rebuild with new insulator design. 900 high-voltage/plasma shots, 570+ on single hardware setup over 11 months. Neutron yields >1x10^12 on highest current shots. 30-fold yield increase since 2018 first light. Reports neutron spot size as function of voltage, pressure, anode implosion radius, and gas dopant levels.
- **Key data**: Circuit parameters (R0=12.5 mOhm, C=204 uF, L=67.4 nH already in our database). New insulator design details. Spot size vs. operating parameters.
- **Relevance to DPF-Unified**: **HIGH** -- Updated experimental data for MJOLNIR validation. Already partially incorporated in our preset.

### 1.3 Double 3 MJ DPF for ICF Drive (2025)

- **Title**: Double 3 MJ dense plasma focus for thermonuclear drive inertial confinement fusion
- **Authors**: Kiai, S.M.S., Adlparvar, S., Sadeghi, H. et al.
- **Journal**: Scientific Reports, Vol. 15, Article 14081 (2025)
- **URL**: https://www.nature.com/articles/s41598-025-96736-7
- **Summary**: Proposes novel double-DPF system using two 3 MJ coaxial DPF devices on opposite sides of a cryogenic DT pellet. Metallic discs with central holes allow current sheets to converge at pellet. Integrated HTS magnetic field lenses improve confinement. Claims 3x fusion power output vs. single-DPF.
- **Relevance to DPF-Unified**: **MEDIUM** -- Novel DPF geometry concept. Could inform future multi-device simulation modes. The 3 MJ energy scale is within our simulation range.

### 1.4 Schmidt -- Simulation-Guided Design of MJ DPF (Feb 2026 Seminar)

- **Title**: Simulation-Guided Design of a MegaJoule Dense Plasma Focus
- **Speaker**: Dr. Andrea Schmidt (LLNL)
- **Event**: IEEE Southeastern Michigan Section / MIPSE Seminar, University of Michigan (February 2026)
- **URL**: https://eecs.engin.umich.edu/event/simulation-guided-design-of-a-megajoule-dense-plasma-focus/
- **Summary**: Seminar on LLNL's approach to using PIC simulations (Chicago code) to guide MJOLNIR design and upgrades. MJOLNIR is described as "one of the first DPFs whose design and continual upgrades are heavily influenced by model predictions."
- **Relevance to DPF-Unified**: **HIGH** -- Directly relevant methodology. Their simulation-guided approach validates our own simulation-driven design philosophy. Potential collaboration/benchmarking opportunity.

### 1.5 LPPFusion -- Hydrogen-Boron (p-B11) Progress

- **Status**: Active experimental campaign (2025-2026)
- **URL**: https://www.lppfusion.com/
- **Summary**: LPPFusion continues p-B11 focus fusion work. Achieved highest confined ion energies (>200 keV) and lowest impurities of any fusion plasma. Fusion-energy to input-energy ratio comparable to Chinese laser-boron results (~0.28 J for 80 kJ input). nTT product of 3.4x10^20 keV-s/m^3.
- **Relevance to DPF-Unified**: **LOW** -- Different fuel (p-B11 vs D-D/D-T), but DPF device physics overlaps.

---

## 2. MHD Simulation Advances (2025-2026)

### 2.1 GRaM-X: GPU-Accelerated Resistive MHD (Oct 2025)

- **Title**: A resistive MHD module in the GPU-accelerated GRMHD code GRaM-X
- **Authors**: Sara Azizi et al.
- **Journal**: arXiv:2510.18968 (October 21, 2025)
- **URL**: https://arxiv.org/abs/2510.18968
- **Summary**: Implements resistive MHD in a GPU-accelerated GRMHD code. Uses second-order IMEX Runge-Kutta (IMEX-SSP2/SSP3 variants) for stiff resistive source terms. HLLE Riemann solver with TVD and WENO reconstruction. Validated on 1D shocktubes, current sheets, Alfven waves, 2D cylindrical explosions, 3D TOV stars.
- **Key technique**: IMEX-RK solves the implicit part point-wise at each grid cell (no global linear system), making it parallelizable and GPU-friendly. This is the same approach our Metal solver uses for sub-cycling resistive diffusion.
- **Relevance to DPF-Unified**: **HIGH** -- Validates our IMEX approach to resistive MHD on GPU. Their cylindrical explosion test is directly comparable to our cylindrical geometry. The SSP-IMEX schemes they catalog could improve our time integration.

### 2.2 Minimally Implicit Runge-Kutta for Resistive MHD (Jan 2025)

- **Title**: Numerical evolution of the resistive relativistic magnetohydrodynamic equations: a minimally implicit Runge-Kutta scheme
- **Authors**: (multiple)
- **Journal**: arXiv:2502.00990 (January 19, 2025)
- **URL**: https://arxiv.org/abs/2502.00990
- **Summary**: Proposes Minimally-Implicit Runge-Kutta (MIRK) methods as alternative to IMEX for resistive MHD. MIRK reduces the number of primitive-variable recoveries vs. IMEX, with computational cost similar to explicit methods. Handles stiff resistive terms while avoiding convergence problems in primitive recovery.
- **Key advantage**: Fewer implicit solves per timestep than full IMEX, easier to retrofit into existing explicit codes.
- **Relevance to DPF-Unified**: **MEDIUM** -- Could simplify our resistive sub-cycling. Worth evaluating whether MIRK would reduce our sub-cycle cap (currently N=20) while maintaining stability.

### 2.3 Structure-Preserving Semi-Implicit IMEX for MHD (2024, published J. Sci. Comput. 2024)

- **Title**: A Structure-Preserving Semi-implicit IMEX Finite Volume Scheme for Ideal Magnetohydrodynamics at all Mach and Alfven Numbers
- **Authors**: Boscheri et al.
- **Journal**: Journal of Scientific Computing, 2024
- **URL**: https://arxiv.org/abs/2403.04517
- **Summary**: Cell-centered finite volume IMEX scheme that preserves div(B)=0 on 3D Cartesian meshes. Stable at all Mach and Alfven numbers with CFL based on local transport speed (not fast wave). Operator splitting: convective terms explicit, pressure and magnetic field implicit. Two decoupled linear systems.
- **Relevance to DPF-Unified**: **MEDIUM** -- The div(B) preservation technique and all-Mach stability are relevant. However, this is Cartesian-only; adaptation to cylindrical coordinates would require work.

### 2.4 No Metal/Apple Silicon MHD Solvers Found

- Searched specifically for MHD codes targeting Apple Metal or Apple Silicon GPUs.
- **Result**: No published MHD solver implementations on Metal as of March 2026.
- **Implication**: DPF-Unified's Metal MHD solver appears to be unique in the literature. This is a differentiator worth highlighting in any publication.

---

## 3. Z-Pinch and Pulsed Power (2025-2026)

### 3.1 Pacific Fusion -- Z Machine Pre-Magnetization Experiments (Feb 2026)

- **Title**: Pacific Fusion Reports Results From Experiments Conducted at Sandia's Z Pulsed Power Facility
- **Date**: February 5, 2026
- **URL**: https://www.businesswire.com/news/home/20260205335703/en/
- **Also**: https://www.ans.org/news/2026-02-06/article-7739/
- **Summary**: Pacific Fusion conducted 4 experiments on Sandia's Z machine (22 MA, 120 ns pulse). Tested pre-magnetization of simplified targets (plastic wrapped in aluminum, 50-200 um Al thickness). Magnetic sensors confirmed field diffusion into targets as designed. Eliminates need for single-use magnetic coils in magneto-inertial fusion targets.
- **Key data**: 22 MA peak current, 120 ns rise time, Al thickness 50/200 um, target size ~pencil eraser.
- **Relevance to DPF-Unified**: **LOW** -- Different geometry (liner implosion vs. coaxial DPF), but circuit-coupled MHD methodology overlaps. The field-diffusion-through-conductor physics is related to our resistive diffusion module.

### 3.2 Zap Energy -- FuZE-3 Gigapascal Pressures (Late 2025)

- **Title**: Zap Energy exceeds gigapascal fusion plasma pressures on new fusion device, FuZE-3
- **Date**: November 2025
- **URL**: https://www.zapenergy.com/news/zap-energy-exceeds-gigapascal-fusion-plasma-pressures-on-new-fusion-device-fuze-3
- **Summary**: FuZE-3 (with novel third electrode) achieved 830 MPa electron pressure (1.6 GPa total). Highest pressure in any sheared-flow-stabilized Z-pinch. Electron densities 3-5x10^24 m^-3, Te > 1 keV. Measured via optical Thomson scattering. Preliminary results presented at APS-DPP 2025.
- **Relevance to DPF-Unified**: **LOW** -- Sheared-flow z-pinch is a different configuration than DPF, but the Z-pinch MHD physics overlaps.

### 3.3 Zap Energy -- Century: 100 kW Repetitive Z-Pinch (2025-2026)

- **Title**: Century: Zap Energy's 100-kW-Scale Repetitive Sheared-Flow-Stabilized Z-Pinch System with Liquid Metal Cooling
- **Journal**: Fusion Science and Technology (2025)
- **URL**: https://www.tandfonline.com/doi/full/10.1080/15361055.2025.2532331
- **Summary**: World's first 100 kW-scale repetitive Z-pinch. 1000+ consecutive plasma shots at 0.1 Hz certified by DOE (Feb 2025). Phase record: 0.2 Hz (shot every 5 seconds), 100+ shots. Total input power 57 kW, 39 kW to plasma chamber. Liquid metal walls for heat capture.
- **Relevance to DPF-Unified**: **LOW** -- Engineering/repetition-rate focus, not directly applicable to single-shot DPF simulation.

### 3.4 General Fusion -- LM26 Magnetized Target Fusion (2025-2026)

- **Title**: Peer-reviewed publication confirms General Fusion achieved plasma energy confinement time required for its LM26 large-scale fusion machine
- **Journal**: Nuclear Fusion (March 2025)
- **URL**: https://generalfusion.com/post/peer-reviewed-publication-confirms-plasma-energy-confinement-time-for-lm26/
- **Summary**: PI3 plasma injector achieved >10 ms energy confinement time. ~20,000 plasmas formed at 50% commercial scale. 190x density compression, 13x magnetic field amplification during compression. 600 million neutrons/second at peak. Now integrated into LM26 demonstration machine targeting scientific breakeven by 2026.
- **Relevance to DPF-Unified**: **LOW** -- Different concept (MTF), but the circuit-coupled compression physics and magnetic field amplification during implosion are analogous to DPF radial phase.

---

## 4. Upcoming Conferences and Special Issues (2026)

- **PAMIR 2026**: Covers tokamaks, short-lived plasmas (plasma focus, z-pinch), laser plasma
- **SPIG 2026**: 33rd Summer School on Physics of Ionized Gases
- **Solar MHD 2026**: A Coruna, Spain (August 10-14, 2026, during solar eclipse)
- **IEEE TPS Special Issues (2026)**:
  - Pulsed Power Science and Technology (October 2026, submission deadline Feb 2026)
  - ICAPST-25 Selected Papers (July 2026)
  - Electrical Discharges in Vacuum (September 2026)
- **ICOPS 2026**: International Conference on Plasma Science

---

## 5. Key Takeaways for DPF-Unified

### Validation Opportunities
1. **Goyon 2025 MJOLNIR paper** provides the richest new validation dataset: neutron spectrum evolution, PIC benchmarks, current-scaling data beyond the saturation drop.
2. **Double-DPF concept** (Kiai et al. 2025) could motivate multi-device simulation capability.

### Numerical Methods to Evaluate
1. **MIRK methods** (arXiv:2502.00990) -- potential simplification of our resistive sub-cycling with fewer implicit solves.
2. **IMEX-SSP schemes** from GRaM-X -- additional time-integration options for our Metal solver.
3. **Structure-preserving IMEX** (Boscheri et al.) -- div(B) preservation techniques applicable if we move to 3D.

### Competitive Landscape
1. **No other Metal/Apple Silicon MHD solver exists** in the literature. DPF-Unified is unique.
2. **LLNL uses PIC (Chicago code)** for MJOLNIR simulation, not MHD. Our MHD approach is complementary.
3. **GRaM-X** is the closest GPU MHD solver but targets NVIDIA/CUDA for astrophysics, not DPF.

### Papers to Extract for dpf-papers/
1. Goyon et al. 2025 -- MJOLNIR neutron dynamics (Physics of Plasmas 32, 033105)
2. Kiai et al. 2025 -- Double 3 MJ DPF (Scientific Reports 15, 14081)
3. Azizi et al. 2025 -- GRaM-X resistive MHD (arXiv:2510.18968) [for numerical methods]
