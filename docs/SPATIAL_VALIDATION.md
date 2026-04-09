# Spatial MHD Validation: Density Profiles

## Reference: Malir et al. (2024)

**Paper**: "Comparison of density profiles measured via laser interferometry
with MHD simulations during shock wave reflection on mega-ampere dense
plasma focus" — Phys. Plasmas 31, 042513 (2024). CC BY 4.0.

**PDF**: `references/papers/core-dpf/malir-2024-interferometry-dpf.pdf`

This is the **first published direct comparison** of MHD-simulated electron
density profiles against interferometric measurements on a DPF device.

## What They Measured

- Device: PF-1000 at IPPLM Warsaw
- Shots: #13317 (0.9 Torr D2, I_peak~1.5 MA) and #13328 (0.75 Torr D2, I_peak~1.3 MA)
- Diagnostic: 15-frame Mach-Zehnder laser interferometer (unique worldwide)
- 29 interferometric images across 2 shots in a 220 ns window around pinch
- Abel inversion for radial ne(r) profiles at z ~ 1 cm above anode surface

## Key Experimental Results

| Quantity | Experimental Value | PERSEUS Simulation |
|----------|-------------------|-------------------|
| Shock width | ~10 mm | ~25 mm (2.7x wider) |
| Peak ne | ~2.6 x 10^18 cm^-3 | ~4.5 x 10^18 cm^-3 (1.7x higher) |
| Minimum radius | ~10 mm | ~10 mm (agreement) |
| Ne per unit length | ~1.0 x 10^19 cm^-1 | ~1.25 x 10^19 cm^-1 (within 20% error) |

## What Their Simulation Used (PERSEUS)

- 1D radial only (no axial motion)
- XMHD: Hall + Biermann battery + Spitzer resistivity
- 1600 cells, dr = 75 um, domain = 120 mm
- gamma = 1.5 (compromise for partial ionization)
- Linearly decreasing current: I(t) = 0.8 * (1.35 - 0.35t) MA
- Reduced speed of light: 10^7 m/s

## Our Validation Approach

DPF-Unified can improve on the PERSEUS comparison because:

1. **2D (r,z) vs 1D (r only)**: Our axisymmetric solver captures axial mass
   redistribution, which the paper identifies as a primary limitation of their
   1D setup (Section VII: "no initial axial movement... should cause mass
   redistribution along the axis").

2. **Self-consistent circuit coupling**: Their current is prescribed I(t). Our
   Lee-MHD hybrid computes I(t) from the circuit ODE coupled to plasma L_p.

3. **Same device parameters**: PF-1000 is our primary validation target with
   published RADPF parameters and multiple experimental waveforms.

4. **Abel transform infrastructure**: `src/dpf/diagnostics/interferometry.py`
   implements the same forward Abel transform used in the paper's Eq. (1).

## Digitized Data

Experimental profiles digitized in `src/dpf/validation/malir_2024_data.py`:
- 8 radial profiles from shot #13328 at times -171 to +39 ns
- Integrated N_e vs time from Figure 11
- Key comparison metrics (shock width, peak ne, min radius)
- PERSEUS simulation parameters for reproducibility

## Comparison Metrics

Following Malir et al., we will compare:

1. **N_e (electrons per unit length)**: Spatially integrated, least sensitive to
   profile shape errors. Target: within 20% error bars (their Fig 11).
2. **Shock width**: FWHM of the density peak. Target: within factor 2.
3. **Peak ne**: Maximum electron density during compression. Target: within factor 2.
4. **Minimum radius**: Radius of peak density at stagnation. Target: within 5 mm.

## Status

- [x] Paper acquired and on disk
- [x] Data digitized from Figures 6-7 and 11
- [x] Spatial comparison module built (`src/dpf/validation/spatial_comparison.py`)
- [x] Abel transform infrastructure exists
- [ ] Run DPF-Unified MHD solver with matching conditions
- [ ] Compare ne(r) profiles at matching timestamps
- [ ] Document results with quantified NRMSE
