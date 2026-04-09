# Impurity Physics Limitations

## Current State

DPF-Unified has partial impurity infrastructure but does not model the full
impurity lifecycle in production simulations:

### What Exists

| Module | Location | Status |
|--------|----------|--------|
| Electrode ablation source terms | `src/dpf/atomic/ablation.py` | Implemented, not enabled by default |
| Copper/tungsten material parameters | `src/dpf/atomic/ablation.py` | Empirical yields from Bruzzone 2003, Lee 1996 |
| Multi-species advection | `src/dpf/metal/mlx_species.py` | Bulk advection only |
| Saha ionization (H/D) | `src/dpf/atomic/ionization.py` | Full Saha + CR model |
| Ionization potentials (Cu, W) | `src/dpf/atomic/ionization.py` | 29-state Cu, 10-state W |
| Line radiation (multi-Z) | `src/dpf/metal/mlx_line_radiation.py` | Piecewise power-law cooling curves |

### What Is Missing

1. **Charge-state-resolved transport**: Impurity ions at different charge states
   have different radiation rates, resistivities, and transport properties.
   The CR model computes Z_bar per cell but does not feed back into MHD transport.

2. **Impurity-enhanced radiation**: At T > 100 eV, copper impurities (even at ~1%
   concentration) can dominate radiation losses over deuterium bremsstrahlung.
   The code does not include impurity line radiation in the energy equation during
   MHD evolution.

3. **Ablation-MHD coupling**: The ablation source term (`ablation_source_array`)
   injects mass into boundary cells but is not active by default in any backend.
   No momentum or energy source from ablated material is included.

4. **Impurity mixing**: No turbulent or convective mixing model exists. Ablated
   material stays in boundary cells unless advected by bulk flow.

## Why It Matters

Pottier et al. (J. Appl. Phys. 50:6, 1979) showed that copper impurities
dominate radiation power output in DPF devices with copper anodes at temperatures
above ~50 eV. The radiation cooling from even 0.1% copper can exceed deuterium
bremsstrahlung by an order of magnitude.

### Bounding Estimate

At typical DPF pinch conditions (ne = 1e25 m^-3, Te = 500 eV, pinch volume ~1 mm^3):

| Source | Power Density | Total Power |
|--------|--------------|-------------|
| D bremsstrahlung | ~1e13 W/m^3 | ~10 MW |
| Cu line radiation (1% Cu) | ~1e14 W/m^3 | ~100 MW |
| Cu line radiation (0.1% Cu) | ~1e13 W/m^3 | ~10 MW |

Even 0.1% copper makes impurity radiation comparable to bremsstrahlung.
At 1% copper, impurity radiation dominates by 10x.

### Impact on Simulation Accuracy

For the Lee model (circuit-level): **Minimal impact.** The Lee model absorbs
impurity effects into the phenomenological fit parameters (fc, fm, fmr, fcr),
which are calibrated from experimental waveforms that already include impurity
radiation. The 2.8% I_peak accuracy is not affected.

For the MHD solver (spatial profiles): **Significant impact.** If the MHD solver
is used to predict pinch temperature, density profiles, or radiation output,
neglecting impurity radiation will overestimate pinch temperature and
underestimate radiation power. This primarily affects the post-compression phase
(t > t_peak) where the pinch column reaches maximum temperature.

## Recommended Approach for Future Work

1. **Enable ablation source** in MLX solver (flag exists, needs wiring: ~50 LOC)
2. **Feed Z_bar from CR model into transport**: resistivity, radiation, conduction
3. **Add Cu/W line radiation** to `mlx_line_radiation.py` using existing cooling
   curve infrastructure (Post et al. 1977 data already in code)
4. **Validate against X-ray spectroscopy** data from published DPF experiments

Estimated effort: ~300 LOC for basic impurity radiation, ~800 LOC for full
charge-state-resolved transport with ablation coupling.
