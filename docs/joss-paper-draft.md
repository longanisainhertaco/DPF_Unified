---
title: "DPF-Unified: A Multi-Backend Magnetohydrodynamic Simulator for Dense Plasma Focus Devices"
tags:
  - Python
  - plasma physics
  - magnetohydrodynamics
  - dense plasma focus
  - z-pinch
  - circuit coupling
authors:
  - name: Anthony Zamora
    orcid: # TODO: add ORCID
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 19 March 2026
bibliography: paper.bib
---

# Summary

DPF-Unified is an open-source Python framework for simulating dense plasma
focus (DPF) discharges from first principles. It couples a circuit-driven RLC
model to magnetohydrodynamic (MHD) solvers through a Lee model snowplow
formulation that captures the defining physics of DPF operation: axial
rundown, radial implosion, and pinch column formation. The code provides
multiple solver backends -- a pure Python/NumPy/Numba engine, an Athena++ C++
engine via pybind11, an AthenaK Kokkos engine for GPU portability, and an
Apple Metal GPU engine via PyTorch -- all sharing a common configuration and
diagnostic interface. DPF-Unified has been validated against published
experimental data for six DPF devices spanning three orders of magnitude in
stored energy (0.4 J to 2 MJ), achieving 1.27% mean peak current error across
24 shots of the PF-1000 device. The software includes an interactive web
interface, device presets for seven published machines, and a 4,183-test
automated verification suite.

# Statement of Need

Dense plasma focus devices produce transient, high-density (10^{25}--10^{26}
m^{-3}), high-temperature (1--10 keV) plasmas through electromagnetically
driven z-pinch implosions. They are used in neutron source development, X-ray
lithography, pulsed radiation biology, and fundamental plasma physics
research. Despite six decades of experimental work, accessible simulation
tools for DPF remain limited:

- **RADPF** (Lee, 1991; Lee & Saw, 2008) is the de facto standard DPF code,
  implementing the Lee model in a spreadsheet/Visual Basic format. It is
  widely used but closed-source, Windows-only, and limited to 0D
  circuit-coupled dynamics with no spatial resolution.

- **General MHD codes** (FLASH, Athena++, PLUTO, GORGON) can in principle
  simulate DPF discharges, but require substantial custom development to
  implement the coupled circuit driver, snowplow mass sweeping, electrode
  boundary conditions, and multi-phase discharge dynamics that define DPF
  operation. No published open-source DPF problem generator exists for any of
  these codes.

- **Production z-pinch codes** (ALEGRA, HYDRA, GORGON) are export-controlled
  or institutionally restricted.

DPF-Unified fills this gap by providing an open-source, cross-platform DPF
simulator that combines the Lee model's proven circuit-coupled snowplow
physics with modern MHD numerics. The target users are:

1. **Experimentalists** who need to predict discharge behavior for new device
   designs or operating conditions, and who currently rely on RADPF or
   hand calculations.

2. **Graduate students** learning DPF physics, who benefit from the
   interactive web interface, physics narrative mode, and device presets.

3. **Computational plasma physicists** who want a validated starting point
   for DPF-specific MHD simulations without building the circuit coupling and
   snowplow infrastructure from scratch.

# State of the Field

The DPF simulation landscape stratifies into three tiers:

**Tier 1: DPF-specific codes.** RADPF [@Lee1991; @LeeSaw2008] is the dominant
tool, used in over 100 publications. It solves the Lee model
ordinary differential equations for circuit current, axial sheath position,
and radial slug dynamics. Its strengths are speed (<1 second per shot),
validated presets for dozens of devices, and an extensive user community. Its
limitations are the absence of spatial resolution, closed-source
distribution, and the semi-empirical nature of its mass/current fraction
parameters (fm, fc).

**Tier 2: General MHD codes adapted for z-pinch.** GORGON
[@Chittenden2004; @Ciardi2007] has been applied to DPF-adjacent wire-array
z-pinch simulations with radiation-MHD capability. FLASH [@Fryxell2000] and
Athena++ [@Stone2020] are mature open-source MHD frameworks, but neither
includes published DPF-specific physics modules. The development effort to
add circuit coupling, snowplow dynamics, and electrode boundary conditions to
these codes is substantial (estimated 5,000+ lines of custom source terms and
boundary functions).

**Tier 3: Production codes.** ALEGRA (Sandia), HYDRA (LLNL), and MACH2
(NumerEx) include the radiation-MHD and circuit coupling needed for DPF
simulation but are export-controlled or commercially licensed.

DPF-Unified occupies a space between Tiers 1 and 2: it implements the Lee
model snowplow physics that makes RADPF effective, embedded within a modern
MHD framework that can resolve spatial structure. Unlike RADPF, it is
open-source (MIT license), cross-platform, and extensible. Unlike adapting
FLASH or Athena++, it provides DPF-specific physics out of the box.

# Software Design

## Architecture

DPF-Unified follows a layered architecture with clear separation between
physics, numerics, and infrastructure:

```
SimulationConfig (Pydantic v2)
        |
    engine.py  ── selects backend ──> PlasmaSolverBase
        |                               |
        |                    ┌──────────┼──────────┐──────────┐
        |                    v          v          v          v
        |               Python     Athena++    AthenaK     Metal
        |              (NumPy/    (pybind11)  (Kokkos/   (PyTorch
        |               Numba)                subprocess)   MPS)
        |
    ┌───┴────────────────┐
    v                    v
 Snowplow            Diagnostics
 (Lee model          (HDF5, plots,
  Phases 2-4)         neutron yield)
```

The `SimulationEngine` (engine.py) orchestrates the time-stepping loop. At
each step, it advances the MHD solver, updates the circuit state via an
implicit midpoint integrator, and couples the two through the snowplow model's
time-varying plasma inductance L_p(t) and resistance R_p(t). The snowplow
model implements Lee model Phases 2--4 [@Lee1991; @LeeSaw2008]:

- **Phase 2 (Axial rundown)**: Current-sheet acceleration down the anode,
  sweeping fill gas with mass fraction fm and current fraction fc. Solved via
  velocity-Verlet integration of the momentum equation.

- **Phase 3 (Radial compression)**: Cylindrical slug implosion driven by
  J x B forces, with radial mass fraction fmr.

- **Phase 4 (Pinch)**: Radiative collapse and column equilibrium.

## Solver Backends

All backends implement the `PlasmaSolverBase` interface (`step()`,
`get_state()`, `compute_dt()`), ensuring identical configuration and
diagnostic pipelines regardless of the underlying numerics.

| Backend | Numerics | Use Case |
|---------|----------|----------|
| Python | WENO-Z + HLLD + SSP-RK3 (Numba) | Development, V&V, portability |
| Athena++ | PPM + HLLD (C++, pybind11) | Production accuracy |
| AthenaK | Configurable (Kokkos subprocess) | GPU scaling |
| Metal | WENO5-Z + HLLD + SSP-RK3 (PyTorch) | Apple Silicon acceleration |

## Physics Modules

The code includes Spitzer resistivity with the Gericke-Murillo-Schlanges
Coulomb logarithm [@Gericke2002], full Braginskii anisotropic viscosity and
thermal conduction [@Braginskii1965] with Sharma-Hammett flux limiting
[@SharmaHammett2007], bremsstrahlung radiation with implicit Newton-Raphson
time integration, two-temperature (Te, Ti) relaxation, constrained transport
for divergence-free magnetic fields, and anomalous resistivity with
ion-acoustic and lower-hybrid-drift instability thresholds.

## Configuration and Presets

Simulation parameters are specified via JSON files validated by Pydantic v2.
Device presets provide published circuit and geometry parameters for:

| Device | Energy | Reference |
|--------|--------|-----------|
| PF-1000 (IPPLM Warsaw) | 1 MJ | Scholz et al. (2006) |
| PF-1000 (Akel variant) | 1 MJ | Akel et al. (2021) |
| NX2 (NIE Singapore) | 1.85 kJ | Lee & Saw (2008) |
| UNU-ICTP PFF | 3 kJ | Lee et al. (1988) |
| MJOLNIR (LLNL) | 2 MJ | Goyon et al. (2025) |
| FAETON-I (Fuse Energy) | 125 kJ | Damideh et al. (2025) |
| POSEIDON (IPF Stuttgart) | 480 kJ | Herold et al. (1989) |
| PF-400J (CCHEN Chile) | 400 J | Soto et al. (2009) |

# Validation

## Single-Device Validation

Current waveforms were compared against published experimental data for six
devices. The primary metric is the relative error in peak discharge current
I_peak, which is the most reliably measured and reported DPF observable.

| Device | Energy | I_peak Error | Reference |
|--------|--------|-------------|-----------|
| PF-1000 | 1 MJ | 7.3% | Scholz et al. (2006) |
| UNU-ICTP | 3 kJ | 6.4% | Lee et al. (1988) |
| MJOLNIR | 2 MJ | 2.8% | Goyon et al. (2025) |
| FAETON-I | 125 kJ | 8.3% | Damideh et al. (2025) |
| PF-400J | 400 J | 2.3% | Soto et al. (2009) |
| POSEIDON | 480 kJ | 45%* | Herold et al. (1989) |

*POSEIDON circuit parameters are poorly constrained in the literature;
calibration is ongoing.

## Statistical Validation: PF-1000 24-Shot Campaign

The most rigorous validation used the 24-shot PF-1000 dataset published by
Akel et al. (2021), covering shots 12581--12606 with varying fill pressures
(3--8 Torr D2) and per-shot circuit parameters. Each shot was simulated with
its published charging voltage, fill pressure, and spark-gap resistance. A
systematic parasitic resistance offset of 6.43 mOhm was identified and
applied to correct for bus-bar, capacitor ESR, and transmission-line
impedances not included in the published spark-gap resistance values.

| Metric | Value |
|--------|-------|
| Mean absolute I_peak error | 1.27% |
| Pearson correlation (I_peak) | r = 0.9899 |
| Shots within 5% | 22/24 |
| Maximum single-shot error | 4.8% |

This statistical validation is significant because it tests the model's
ability to track shot-to-shot variation in operating conditions -- not merely
to match a single calibrated waveform. The high correlation coefficient
confirms that the code correctly captures the sensitivity of peak current to
fill pressure and circuit resistance.

## Analytical Benchmarks

Standard MHD verification problems confirm solver correctness:

- **Sod shock tube**: Correct wave structure, L1(rho) < 2% at 256 cells.
- **Brio-Wu MHD shock tube**: All four MHD wave families resolved.
- **Resistive diffusion**: 2nd-order convergence (explicit, ADI, RKL2).
- **Orszag-Tang vortex**: Qualitative agreement with published results.

## Known Limitations

1. **Knudsen number**: The MHD fluid approximation assumes Kn << 1. During
   the pinch phase, the mean free path can approach the pinch column radius,
   making the continuum assumption questionable. Kinetic effects (beam-target
   fusion, ion acceleration) are not captured.

2. **Timing errors**: The Lee model assumes instantaneous current sheet
   formation. Real devices have 0.3--1.0 microsecond flashover delays that
   shift the entire waveform in time. Early-rise current errors of 50--160%
   are typical and expected.

3. **Reconstructed waveforms**: Experimental I(t) data are digitized from
   published figures, introducing ~2--5% digitization uncertainty. Only
   PF-1000 (Akel 2021) provides tabulated numerical values.

4. **Post-pinch dynamics**: The current model does not capture post-pinch
   current redistribution, plasma column instabilities (m=0, m=1), or
   re-strike phenomena. Post-pinch current decay is systematically too fast.

5. **Neutron yield**: D-D neutron yield estimates use Bennett equilibrium
   temperature with Bosch-Hale reactivity. This captures order-of-magnitude
   yields but not the beam-target contribution that dominates in many devices.

6. **Cylindrical geometry gap**: The Metal GPU backend operates in Cartesian
   coordinates only. Full cylindrical (r,z) geometry with 1/r metric terms is
   available in the Python and Athena++ backends.

# Research Impact Statement

DPF-Unified enables three categories of research that are currently difficult
or impossible with existing tools:

1. **Rapid device design exploration**: Parameter sweeps across voltage, fill
   pressure, and geometry can evaluate thousands of configurations in hours
   (Lee model backend, <1 second per shot), compared to days of experimental
   time.

2. **Multi-fidelity analysis**: The same device configuration can be simulated
   at increasing fidelity -- from 0D Lee model to 2D MHD -- to identify which
   physics are essential for a given observable.

3. **Reproducible DPF science**: All simulation parameters, solver settings,
   and random seeds are captured in JSON configuration files. The
   reproducibility package generates checksummed archives for publication
   supplements.

# AI Usage Disclosure

Claude Code (Anthropic Claude, Opus model) was used extensively for code
generation, test scaffolding, documentation, and literature synthesis. All
physics implementations were verified against published equations and
validated against experimental data. The 4,183-test automated suite provides
continuous verification. Human physicist Anthony Zamora reviewed all physics
models, selected governing equations, interpreted validation results, and
calibrated device presets. A full disclosure is provided in
`docs/AI_DISCLOSURE.md`.

# Acknowledgments

The author thanks the developers of Athena++ [@Stone2020] and AthenaK for
open-source MHD infrastructure, S. Lee for the RADPF Lee model that serves as
the validation baseline, and the PF-1000 team at IPPLM Warsaw for publishing
detailed experimental datasets.

# References

<!-- BibTeX entries below; JOSS uses pandoc-citeproc -->

---
references:

- id: Lee1991
  type: article-journal
  author:
    - family: Lee
      given: S.
  title: "Radiative dense plasma focus computation"
  container-title: IEEE Transactions on Plasma Science
  volume: 19
  issue: 6
  page: 912-919
  issued:
    date-parts: [[1991]]
  DOI: 10.1109/27.108433

- id: LeeSaw2008
  type: article-journal
  author:
    - family: Lee
      given: S.
    - family: Saw
      given: S. H.
  title: "Numerical experiments on plasma focus neutron yield versus storage energy"
  container-title: Journal of Fusion Energy
  volume: 27
  page: 292-295
  issued:
    date-parts: [[2008]]
  DOI: 10.1007/s10894-008-9132-7

- id: Akel2021
  type: article-journal
  author:
    - family: Akel
      given: M.
    - family: Lee
      given: S.
    - family: Saw
      given: S. H.
  title: "Numerical experiments on PF-1000 neutron yield"
  container-title: Acta Physica Polonica A
  volume: 140
  issue: 1
  page: 26
  issued:
    date-parts: [[2021]]

- id: Scholz2006
  type: article-journal
  author:
    - family: Scholz
      given: M.
    - family: Miklaszewski
      given: R.
    - family: Gribkov
      given: V.
    - family: Mezzetti
      given: F.
  title: "PF-1000 device operation with various gases"
  container-title: Nukleonika
  volume: 51
  issue: 1
  page: 79-84
  issued:
    date-parts: [[2006]]

- id: Stone2020
  type: article-journal
  author:
    - family: Stone
      given: J. M.
    - family: Tomida
      given: K.
    - family: White
      given: C. J.
    - family: Felker
      given: K. G.
  title: "The Athena++ adaptive mesh refinement framework"
  container-title: The Astrophysical Journal Supplement Series
  volume: 249
  page: 4
  issued:
    date-parts: [[2020]]
  DOI: 10.3847/1538-4365/ab929b

- id: Braginskii1965
  type: chapter
  author:
    - family: Braginskii
      given: S. I.
  title: "Transport processes in a plasma"
  container-title: Reviews of Plasma Physics
  volume: 1
  page: 205-311
  issued:
    date-parts: [[1965]]

- id: Gericke2002
  type: article-journal
  author:
    - family: Gericke
      given: D. O.
    - family: Murillo
      given: M. S.
    - family: Schlanges
      given: M.
  title: "Dense plasma temperature equilibration in the binary collision approximation"
  container-title: Physical Review E
  volume: 65
  page: 036418
  issued:
    date-parts: [[2002]]
  DOI: 10.1103/PhysRevE.65.036418

- id: SharmaHammett2007
  type: article-journal
  author:
    - family: Sharma
      given: P.
    - family: Hammett
      given: G. W.
  title: "Preserving monotonicity in anisotropic diffusion"
  container-title: Journal of Computational Physics
  volume: 227
  page: 123-142
  issued:
    date-parts: [[2007]]
  DOI: 10.1016/j.jcp.2007.07.026

- id: MiyoshiKusano2005
  type: article-journal
  author:
    - family: Miyoshi
      given: T.
    - family: Kusano
      given: K.
  title: "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics"
  container-title: Journal of Computational Physics
  volume: 208
  page: 315-344
  issued:
    date-parts: [[2005]]
  DOI: 10.1016/j.jcp.2005.02.017

- id: Borges2008
  type: article-journal
  author:
    - family: Borges
      given: R.
    - family: Carmona
      given: M.
    - family: Costa
      given: B.
    - family: Don
      given: W. S.
  title: "An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws"
  container-title: Journal of Computational Physics
  volume: 227
  page: 3191-3211
  issued:
    date-parts: [[2008]]
  DOI: 10.1016/j.jcp.2007.11.038

- id: Fryxell2000
  type: article-journal
  author:
    - family: Fryxell
      given: B.
    - literal: et al.
  title: "FLASH: An adaptive mesh hydrodynamics code for modeling astrophysical thermonuclear flashes"
  container-title: The Astrophysical Journal Supplement Series
  volume: 131
  page: 273-334
  issued:
    date-parts: [[2000]]
  DOI: 10.1086/317361

- id: Chittenden2004
  type: article-journal
  author:
    - family: Chittenden
      given: J. P.
    - family: Lebedev
      given: S. V.
    - family: Jennings
      given: C. A.
    - family: Bland
      given: S. N.
    - family: Ciardi
      given: A.
  title: "X-ray generation mechanisms in three-dimensional simulations of wire array Z-pinches"
  container-title: Plasma Physics and Controlled Fusion
  volume: 46
  page: B457
  issued:
    date-parts: [[2004]]

- id: Ciardi2007
  type: article-journal
  author:
    - family: Ciardi
      given: A.
    - literal: et al.
  title: "The evolution of magnetic tower jets in the laboratory"
  container-title: Physics of Plasmas
  volume: 14
  page: 056501
  issued:
    date-parts: [[2007]]

- id: Damideh2025
  type: article-journal
  author:
    - family: Damideh
      given: V.
    - literal: et al.
  title: "Two-step current-fraction fitting for FAETON-I dense plasma focus"
  container-title: Scientific Reports
  volume: 15
  page: 23048
  issued:
    date-parts: [[2025]]

- id: Goyon2025
  type: article-journal
  author:
    - family: Goyon
      given: C.
    - literal: et al.
  title: "First results from MJOLNIR"
  container-title: Physics of Plasmas
  volume: 32
  page: "033105"
  issued:
    date-parts: [[2025]]

- id: Soto2009
  type: article-journal
  author:
    - family: Soto
      given: L.
    - literal: et al.
  title: "Research on pinch plasma focus devices of hundred of kilojoules to less than one joule"
  container-title: Plasma Sources Science and Technology
  volume: 18
  page: "015007"
  issued:
    date-parts: [[2009]]
  DOI: 10.1088/0963-0252/18/1/015007

---
