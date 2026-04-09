# DPF-Unified: Scope, Claims, and Regime of Validity

## What This Code Is

DPF-Unified is a **Lee-MHD hybrid dense plasma focus simulator**. It couples a
validated lumped-circuit snowplow model (Lee & Saw 2014) to a spatially-resolved
conservative MHD solver on a cylindrical (r,z) grid.

- **Circuit-level physics**: Lee 5-phase snowplow model validated against 6 devices
  with 24 PF-1000 shots at 1.27% mean absolute I_peak error using published RADPF
  parameters and zero calibration (Akel et al. 2021).
- **MHD-level physics**: Conservative resistive MHD (HLLS/HLLD + WENO5-Z + SSP-RK3)
  verified against standard benchmarks (Sod, Brio-Wu). Not yet validated against
  spatially-resolved experimental measurements.

The distinction matters. Circuit validation proves the integrated plasma dynamics
(sheath mass, inductance, timing) are correct. MHD verification proves the numerical
methods solve the equations they claim to solve. MHD **validation** against spatial
data (density profiles, temperature maps, B-field structure) has not been performed.

## What This Code Is Not

- **Not a kinetic code.** MHD assumes Kn << 1. In the DPF pinch column, Kn > 1
  (diagnosed by the code's own regime classifier). Pinch microphysics (beam-target
  neutron generation, ion acceleration, m=0 disruption timing) use Lee phenomenological
  parameters (fmr, fcr), not first-principles MHD.

- **Not a radiation-hydrodynamics code.** Bremsstrahlung and line radiation are
  energy sinks (cooling terms). There is no photon transport, no radiation pressure,
  and no frequency-dependent opacity. At >100 eV, radiation-dominated regimes are
  not self-consistently modeled.

- **Not a 3D instability code.** The production solver is 2D axisymmetric (r,z).
  Azimuthal (m=1 kink) instabilities cannot develop. A 3D Cartesian mode exists
  in the PyTorch Metal backend for electrode BC testing, but has not been validated
  for DPF discharge simulations. 3D MLX support requires ~350 LOC of kernel work.

- **Not a multi-physics platform.** No electrode ablation, no impurity transport,
  no charge-state-resolved atomic physics, no MPI parallelism. These are documented
  future work, not hidden gaps.

## Regime of Validity

### Where MHD Applies (Bulk Plasma)

| Parameter | Typical Value (rundown) | Typical Value (pinch) | MHD Requirement |
|-----------|------------------------|----------------------|-----------------|
| Knudsen (Kn) | ~1e-4 | **>1** | Kn << 1 |
| Lundquist (S) | ~1e3 | ~1e2 | S >> 1 |
| Magnetic Reynolds (Rm) | ~1e3 | ~1e2 | Rm >> 1 |
| Beta (thermal/magnetic) | ~0.1-1 | ~0.01-0.1 | Any |
| Hall parameter (omega_ci*tau_i) | ~100 | ~1-10 | >>1 for single-fluid |

**MHD is valid for**: sheath formation, axial rundown, radial compression, shock
convergence, circuit loading (L_p feedback). These phases determine I_peak and the
current waveform shape.

**MHD breaks down at**: peak pinch compression (Kn > 1), where mean free paths
exceed the column radius. The code handles this transition by:
1. Computing per-cell Kn via `diagnostics/regime_classifier.py`
2. Using Lee phenomenological parameters (fmr, fcr) for the pinch phase
3. Offering a PIC hybrid module (`experimental/pic/hybrid.py`) for kinetic effects

### Equation of State Limitations

The production code uses ideal gas with gamma = 5/3 (monoatomic). This is adequate
for fully ionized deuterium plasma at T > 50 eV but neglects:

- **Ionization energy sinks**: Energy absorbed during ionization of neutral fill gas
  and partially ionized species (significant at T < 20 eV during early compression)
- **Real-gas effects**: Non-constant gamma from internal degrees of freedom
- **Non-equilibrium ionization**: Saha equilibrium assumption may not hold during
  rapid compression (sub-microsecond timescales)

A tabulated EOS module exists (`src/dpf/fluid/tabulated_eos.py`, 449 LOC) with
bilinear interpolation in log-space, but is not yet wired into the MHD solver.

### Impurity Physics Limitations

DPF devices with metal anodes (copper, tungsten) produce high-Z impurities through
electrode ablation. These impurities:
- Dominate radiation losses at T > 100 eV (Pottier et al. 1979)
- Modify the effective charge state Z_eff
- Affect pinch dynamics through enhanced radiation cooling

The code has no ablation model or charge-state tracking. Multi-species advection
exists (`metal/mlx_species.py`) but tracks bulk transport only, with no ionization
or recombination. For pure deuterium fill gas with clean electrodes, this is
acceptable. For devices with significant electrode erosion, radiation losses may
be underestimated.

### 2D Axisymmetric Limitations

The 2D (r,z) assumption enforces azimuthal symmetry:
- **m=0 sausage instabilities**: Can develop (radial perturbations allowed)
- **m=1 kink instabilities**: Cannot develop (requires azimuthal variation)
- **m>=2 fluting instabilities**: Cannot develop

Kink instabilities are a primary mechanism for pinch disruption in real DPF devices
(Stepniewski 2006, Deng et al. 2006). The 2D assumption is standard in the DPF
simulation community (Lee model, MACH2, Auluck 2014) but limits the code's ability
to self-consistently predict disruption timing and post-pinch dynamics.

## Error Budget (PF-1000 at 27 kV)

| Error Source | Contribution | Evidence |
|--------------|-------------|----------|
| Snowplow mass model (fm sensitivity) | ~40-60% of total | Sobol: fm has ST < 0.08 on I_peak, but dominates t_peak |
| Measurement uncertainty | ~30-50% of total | Gribkov flat-top: t_peak ambiguous by ~10% (5.2-6.6 us) |
| Numerical scheme diffusion | ~7-20% of total | HLL vs HLLD shows ~2% I_peak shift |
| Grid resolution | ~5-15% of total | 32x64 vs 64x128 shows <1% I_peak change |

**Total I_peak error**: 2.8% (sim 1.818 MA vs exp ~1.87 MA)
**Total t_peak error**: 10-14% (structural; see docs/TIMING_ERROR_RCA.md)

The I_peak error is dominated by the circuit LC period and plasma loading, which
the Lee model captures well. The t_peak error is structural — the sheath propagates
too slowly during axial rundown, and Lee parameters (fc, fm) cannot simultaneously
optimize both I_peak and t_peak. This is a known limitation of the snowplow
approximation, not a numerical bug.

## Validation Summary

### Validated (Experimental Comparison)

| Claim | Evidence | Metric |
|-------|----------|--------|
| PF-1000 I_peak at 27 kV | Scholz 2006 (26 points), Gribkov 2007 (94 points) | 2.8% error |
| PF-1000 24-shot campaign | Akel et al. 2021 (shots 12581-12606) | 1.27% MAE, r=0.9957 |
| 6-device cross-validation | Lee & Saw 2014, Damideh 2025, Offermann 2021 | 6/7 PASS (NX2 excluded) |
| Current waveform shape | NRMSE vs digitized experimental traces | <0.20 for passing devices |

### Verified (Numerical Correctness)

| Claim | Evidence | Metric |
|-------|----------|--------|
| Sod shock tube | Analytic solution comparison | L1(rho) < 0.02 at N=256 |
| Brio-Wu MHD shock | No NaN in float32 with dual-energy | Completes to t_final |
| Deterministic reproducibility | 50 identical runs | std = 2.2e-16 |
| Mass conservation | Full discharge simulation | < 5% relative |
| Energy conservation | Full discharge simulation | < 10% relative |

### Not Yet Validated (Known Gaps)

| Claim | Required Evidence | Status |
|-------|-------------------|--------|
| Density profiles match experiment | Interferometry data (Malir et al. 2024) | Data not yet integrated |
| Temperature profiles match experiment | Thomson scattering data | No experimental data on disk |
| Pinch radius evolution | X-ray framing camera data | No experimental data on disk |
| Neutron yield from MHD state | Experimental yield comparison | Lee model only; MHD yield not tested |
| Grid convergence of spatial quantities | Multi-resolution density/B-field study | I_peak converges; spatial fields not studied |

## Backend Physics Parity

Not all physics modules are available on all backends. See `docs/BACKEND_PARITY.md`
for the complete matrix. Key points:

- **MLX** (production GPU): Most complete — 9 active, 8 available behind flags
- **Python** (teaching/fallback): 7 active, non-conservative pressure equation
- **Athena++** (reference C++): 3 active, limited by compile-time physics selection
- **AthenaK** (Kokkos C++): 2 active, Cartesian only (no cylindrical coordinates)

When a user enables physics not supported by their selected backend, the code
**silently skips** the unsupported module. This is a known usability gap — explicit
warnings should be added (tracked as a future improvement).

## Citation

If citing this code, use the appropriate framing:

- **For circuit-level results** (I_peak, waveform, neutron yield via Lee model):
  "Validated against published experimental data with zero calibration"
- **For MHD spatial results** (density, temperature, B-field profiles):
  "Verified against standard test problems; experimental validation in progress"
- **For numerical methods**:
  "HLLS (Popovas 2025) + WENO5-Z (Borges 2008) + SSP-RK3 (Shu-Osher 1988)
  in conservative form with dual-energy entropy switching for float32 robustness"
