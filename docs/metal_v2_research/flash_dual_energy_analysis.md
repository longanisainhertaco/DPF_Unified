# FLASH Dual-Energy Formalism for MHD: Research Analysis

**Date**: 2026-03-24
**Purpose**: Inform dual-energy design for DPF-Unified Metal v2 solver
**Context**: Cylindrical MHD, float32 GPU, ME/p > 10^6 at cathode boundary

---

## 1. FLASH's Internal Energy Equation (e_int PDE)

### 1.1 The Equation

FLASH evolves an auxiliary internal energy equation alongside the conservative total energy equation. From the FLASH User Guide (Section 15.1, Eq. 15.7) and confirmed in the `hy_uhd_unsplitUpdate.F90` source code:

```
d(rho*e_int)/dt + div[(rho*e_int + P) * v] - v . grad(P) = Q_ohm
```

This is equivalent to the non-conservative form:

```
D(e_int)/Dt = -(P/rho) * div(v) + Q_ohm/rho
```

where `D/Dt` is the material derivative. The key term is `-(P/rho) * div(v)` -- the pdV work. This form is NOT conservative and does NOT satisfy Rankine-Hugoniot jump conditions at shocks. That is by design: the internal energy equation is only trusted in smooth regions where the total energy subtraction fails.

### 1.2 Source Terms Included

From the `hy_uhd_unsplitUpdate.F90` source (lines ~895-940):

**Ohmic heating (eta * J^2)**: YES. When `hy_useMagneticResistivity` is enabled:
```fortran
call hy_uhd_addOhmicHeating(blockID, blkLimits, i, j, k, Qohm, res_eta(i,j,k))
Qohm = Qohm * U(DENS_VAR, i, j, k)
```
The Ohmic heating is passed to `updateInternalEnergy` as a source term `dt*Qohm`.

**Braginskii conduction**: NO -- handled as a separate operator-split diffusion step via the `Diffuse` unit, not included in the internal energy advection equation directly.

**Radiation losses**: NO -- handled via the `RadTrans` unit (multigroup diffusion) or `Heatexchange` unit, again operator-split.

### 1.3 Flux Discretization

From `updateInternalEnergy` in `hy_uhd_unsplitUpdate.F90` (lines ~1280+):

```fortran
eint = eint + dt/dx*(FL(1)-FR(1) + pres*(FL(2)-FR(2)))
if (NDIM > 1) then
    eint = eint + dt/dy*(GL(1)-GR(1) + pres*(GL(2)-GR(2)))
    if (NDIM > 2) then
        eint = eint + dt/dz*(HL(1)-HR(1) + pres*(HL(2)-HR(2)))
    endif
endif
eint = eint + dt*Qohm
```

The `FL(1)/FR(1)` are the advective fluxes of `rho*e_int`, and `FL(2)/FR(2)` are volume fluxes used for the `P*div(v)` work term. This is a finite-volume discretization of the non-conservative internal energy equation.

---

## 2. FLASH's Switching Criterion

### 2.1 The eintSwitch Parameter

FLASH uses a single parameter `eintSwitch` (stored as `hy_eswitch` in the Fortran code). The switching logic from `hy_uhd_unsplitUpdate.F90` (lines ~1000-1020):

```fortran
ekin = 0.5 * dot_product(U(VELX_VAR:VELZ_VAR,i,j,k), U(VELX_VAR:VELZ_VAR,i,j,k))
eint = U(ENER_VAR,i,j,k) - ekin

if (.not. hy_useAuxEintEqn .or. eint > hy_eswitch * ekin) then
    newEint = max(hy_smallE, eint)
else
    newEint = max(hy_smallE, IntEner)
endif

U(ENER_VAR,i,j,k) = newEint + ekin
```

**Criterion**: `eint_from_total_energy > eintSwitch * kinetic_energy`

- If TRUE: trust the total energy subtraction (smooth region, sufficient precision)
- If FALSE: use the separately evolved internal energy (high-Mach / low-beta region)

**Default value**: `1.0e-4` (from FLASH User Guide; PPM solver testing showed this threshold maintains accuracy without affecting dynamics).

**Note on MHD**: In the MHD case, magnetic energy is added to the total energy before the subtraction:
```fortran
#if defined(FLASH_USM_MHD)
U0(HY_ENER) = U0(HY_ENER) + 0.5*dot_product(U(MAGX_VAR:MAGZ_VAR,i,j,k), ...)
#endif
```

So the subtraction is: `e_int = E_total - 0.5*rho*v^2 - 0.5*B^2` and the switching compares `e_int` against `eintSwitch * 0.5*rho*v^2`. The magnetic energy is NOT in the denominator of the switching criterion -- only kinetic energy is.

### 2.2 How This Differs From Enzo

FLASH and Enzo use fundamentally different switching criteria. See the comparison table in Section 6.

---

## 3. Interaction with Constrained Transport (CT)

### 3.1 Variable Centering

- **e_int is cell-centered**, NOT face-centered. It is a volume-averaged scalar quantity stored at cell centers, just like density and total energy.
- **B-fields for CT are face-centered** (staggered mesh). The EMFs (electric fields) used to update face-centered B are edge-centered.
- **e_int and CT are decoupled**: The internal energy update uses cell-centered quantities and reconstructed interface values. CT evolves B-fields independently via Faraday's law. They communicate only through the EOS (pressure recovery) and Ohmic heating.

### 3.2 Update Ordering

In the USM (Unsplit Staggered Mesh) scheme (Lee 2013):

1. Reconstruct primitives at cell interfaces (PLM or PPM)
2. Solve Riemann problems for fluxes (Roe, HLL, HLLC, HLLD, or Marquina)
3. Compute edge-centered EMFs from interface fluxes
4. Update cell-centered conserved variables (rho, rho*v, E) with flux divergence
5. Update face-centered B with CT (using EMFs from step 3)
6. Update auxiliary internal energy (separate flux computation)
7. Apply switching criterion to select eint source
8. Call EOS (`Eos_wrapped` with `hy_eosModeAfter`) to synchronize thermodynamic state

The internal energy update (step 6) is computed from the SAME reconstructed interface states used in step 2, ensuring consistency. But its fluxes are computed independently -- they do not go through the Riemann solver.

### 3.3 Energy Correction After CT

After obtaining new cell-centered B-fields from the CT update, the total energy may need correction to preserve positivity (from FLASH MHD docs):

> "After obtaining the new cell-centered magnetic fields, the total plasma energy may need to be corrected in order to preserve the positivity of the thermal temperature and pressure. This energy correction is very useful especially in problems involving very low beta plasma flows."

This correction ensures `E_total >= 0.5*rho*v^2 + 0.5*B^2 + e_floor` after the magnetic field update.

---

## 4. Cylindrical Coordinates

### 4.1 FLASH USM in Cylindrical Geometry

FLASH's USM solver has been extended to 2D and 2.5D cylindrical (R-Z) geometries for both uniform grids and AMR, following Mignone et al. (2007) and Skinner & Ostriker (2010).

**Geometric source terms**: In cylindrical coordinates, the conservation laws acquire geometric source terms from the divergence operator. For the momentum equation in the R-direction:

```
S_r = (P_total + rho*v_theta^2) / r
```

where `P_total = P_gas + B^2/(2*mu_0)` (includes magnetic pressure).

For the internal energy equation, the geometric source term enters through the divergence of velocity:

```
div(v) = (1/r) * d(r*v_r)/dr + dv_z/dz + (1/r) * dv_theta/d_theta
```

In 2.5D (axisymmetric, no theta-derivative), this becomes:

```
div(v) = dv_r/dr + v_r/r + dv_z/dz
```

The `v_r/r` term is the geometric source for internal energy. FLASH handles this through a modified prolongation algorithm based on Balsara (2004) and Li & Li (2004) to account for geometrical factors, particularly important for AMR refinement/derefinement.

### 4.2 Dual Energy in Cylindrical

There is no special handling of the dual energy switching criterion for cylindrical coordinates. The same `eintSwitch` threshold applies. The geometric source terms are included in BOTH the total energy and internal energy evolution, so the switching logic remains valid.

**Implication for DPF-Unified**: Our cylindrical MHD solver must include the `v_r/r` geometric source in the internal energy equation when using dual energy. This is straightforward in operator-split form.

---

## 5. Float32 / Single Precision

### 5.1 FLASH Precision

FLASH promotes all floating-point variables to double precision (float64) at compile time. From the FLASH 2.5 User Guide: "FLASH promotes all floating point variables to double precision at compile time for maximum portability."

**There are no published float32 tests of FLASH's dual energy for MHD.** FLASH has always been a float64 code.

### 5.2 Implications for DPF-Unified

This is a significant gap. The dual energy formalism was designed and tuned for float64. In float32:

- The `eintSwitch = 1e-4` threshold may need adjustment. Float32 has ~7 decimal digits of precision, so `e_int/e_kinetic < 1e-4` means only ~3 significant digits remain in the subtraction. The threshold should probably be higher (e.g., `1e-2` to `1e-3`).
- The internal energy flux computation involves `P * (FL_vol - FR_vol)` which is itself a subtraction -- vulnerable to cancellation in float32.
- The entropy tracer approach (as in our METAL_V2_SPEC) avoids the subtraction entirely and is inherently more robust in float32.

---

## 6. Negative Pressure Handling

### 6.1 Priority System

When the total-energy-derived pressure is negative but the internal-energy-derived pressure is positive, FLASH uses the following priority:

1. **First**: Compute `eint_from_etot = E_total - 0.5*rho*v^2 - 0.5*B^2`
2. **Check**: If `eint_from_etot > eintSwitch * 0.5*rho*v^2`, use it
3. **Fallback**: If check fails, use `eint_from_aux` (separately evolved)
4. **Floor**: Apply `max(hy_smallE, eint)` regardless of source
5. **Sync**: Recompute `E_total = eint + 0.5*rho*v^2 + 0.5*B^2` to maintain consistency

The `hy_smallE` floor is the absolute minimum specific internal energy, set from the pressure floor: `smallE = P_floor / ((gamma-1) * rho)`.

### 6.2 Known Failure Mode

From the FLASH users mailing list (April 2019), users reported "Zero or negative advected specific internal energy detected" in 3T MHD simulations with laser-generated plasmas. Root causes identified:

- Extremely low density regions (rho ~ 1e-9) where magnetic pressure dominates
- Self-generated B-fields creating severe pressure/density gradients in low-density regions
- Magnetic energy exceeding total energy after CT update (round-off)

Workarounds: reduce CFL, increase initial density/temperature, adjust `smlrho` density floor.

**For DPF-Unified**: Our cathode boundary has ME/p > 10^6 and density gradients of ~10^4. This is exactly the failure regime FLASH users encounter. The entropy tracer avoids this entirely because it never subtracts large numbers.

---

## 7. Known Failure Modes and Limitations

### 7.1 Non-Conservative at Shocks

The internal energy equation uses `P*div(v)` which is non-conservative. At shocks, this produces incorrect entropy jumps. The switching criterion is supposed to use the total energy (which IS conservative) at shocks, but:

- If the shock is in a magnetically dominated region, the switching may incorrectly choose the internal energy equation
- The transition zone between "trust total energy" and "trust internal energy" can produce glitches

### 7.2 Operator Splitting with Source Terms

When operator-split physics (conduction, radiation, Ohmic heating) modify the internal energy between hydro steps, the separately evolved `e_int` and the total-energy-derived `e_int` can drift apart. The synchronization step (selecting one and recomputing the other) can discard real physics.

### 7.3 AMR Refinement Boundaries

At AMR refinement boundaries, the internal energy fluxes from the coarse grid may not be consistent with the fine grid. FLASH's flux correction mechanism handles this for conservative variables but the internal energy is non-conservative, creating potential inconsistencies.

### 7.4 Three-Temperature (3T) Complications

In 3T mode, the internal energy is split into electron, ion, and radiation components. Each must be separately tracked and the splitting ratio depends on the thermodynamic state. Negative component energies can occur even when the total internal energy is positive.

### 7.5 Float32 Amplification

All of the above failure modes are amplified in float32. The subtraction `E_total - KE - ME` loses more digits, the pdV work term has more cancellation error, and the switching criterion has a narrower margin. This is the primary motivation for the entropy tracer approach in DPF-Unified.

---

## 8. Enzo vs. FLASH Comparison

### 8.1 Enzo's Dual Energy: Two Formulations

Enzo (and its successor Enzo-E) supports two dual-energy formulations:

**Bryan95 formulation** (classic Enzo, Bryan et al. 1995):
- Two parameters: `eta_1 = 0.001` (default), `eta_2 = 0.1` (default)
- `eta_1` controls when to synchronize eint from etot during the hydro update
- `eta_2` controls when to use eint vs etot for pressure recovery
- Internal energy evolved via non-conservative `P*div(v)` equation
- Designed for cosmological simulations (hypersonic flows, v >> c_s)

**Modern formulation** (Enzo-E VL+CT integrator, Abruzzo et al.):
- Single parameter: `eta`
- Switching criterion (from `EnzoPhysicsFluidProps.cpp`):

```cpp
enzo_float cs2_1 = max(0, gamma*(gamma-1)*eint_from_etot);

if ((cs2_1 > max(eta*v2, eta*b2/rho)) && (eint_from_etot > 0.5*eint_tracked)) {
    use eint_from_etot;
} else {
    use eint_tracked;  // separately evolved
}
```

Key features of the modern formulation:
- Compares sound speed squared against BOTH kinetic (eta*v^2) and magnetic (eta*B^2/rho) energy
- Includes a sanity check: eint_from_etot must be > 50% of tracked eint
- Internal energy treated as **passively advected scalar** in the Riemann solver (confirmed in `EnzoRiemannHLLD.hpp`: "eint fluxes are computed by assuming that specific internal energy is a passive scalar")
- When eta=0, always synchronize from total energy (half_factor=0 disables the 50% check)

### 8.2 Comparison Table

| Feature | Enzo (Bryan95) | Enzo-E (Modern) | FLASH (USM) |
|---------|---------------|-----------------|-------------|
| **Parameters** | eta_1=0.001, eta_2=0.1 | single eta | eintSwitch (default 1e-4) |
| **Switching criterion** | e_int/e_total ratios | cs^2 > eta*max(v^2, B^2/rho) AND eint > 0.5*eint_tracked | eint > eintSwitch * ekin |
| **Magnetic energy in criterion** | Not directly | YES (eta*B^2/rho) | NO (only ekin in denominator) |
| **e_int advection** | Non-conservative P*div(v) | Passive scalar in Riemann solver | Non-conservative P*div(v) with separate fluxes |
| **e_int centering** | Cell-centered | Cell-centered | Cell-centered |
| **Ohmic heating in e_int** | N/A (ideal MHD) | N/A (ideal MHD) | YES (Qohm source term) |
| **CT interaction** | Separate (Dedner div-clean) | Separate (VL+CT) | Separate (USM-CT) |
| **Shock synchronization** | Via eta_1/eta_2 thresholds | Via cs^2 comparison | Via eintSwitch threshold |
| **Cylindrical support** | No (Cartesian only) | No (Cartesian only) | YES (2D/2.5D R-Z) |
| **Precision** | float64 only | float64 (enzo_float configurable) | float64 only |
| **AMR support** | Yes (with flux correction) | Yes (with flux correction) | Yes (with flux correction) |
| **3T support** | No | No | YES (electron/ion/radiation) |
| **Entropy alternative** | Yes (DualEnergyFormalism=1 can use entropy) | No | No |
| **Floor** | Separate pressure floor | pressure_floor -> eint_floor per cell | hy_smallE floor |
| **Post-sync EOS call** | Yes | Yes (apply_floor_to_energy_and_sync) | Yes (Eos_wrapped with hy_eosModeAfter) |

---

## 9. Recommendation for DPF-Unified

### 9.1 Why Neither FLASH nor Enzo's Approach is Optimal for Us

1. **Float32 constraint**: Both codes assume float64. Their switching criteria and flux computations involve subtractions that lose precision in float32.

2. **Cylindrical MHD**: Only FLASH supports cylindrical coordinates, but its switching criterion ignores magnetic energy -- a major deficiency for our cathode boundary where ME >> KE >> e_int.

3. **Non-conservative e_int**: Both FLASH and Enzo evolve e_int via non-conservative equations. At DPF shocks (Mach 5-20 in the sheath), the non-conservative form introduces errors that contaminate the "fallback" energy.

4. **Passive scalar approach** (Enzo-E modern): This is the best of the three for our use case. Treating e_int (or entropy S) as a passively advected scalar in the Riemann solver means it gets the correct upwind advection without needing a separate flux computation. This is what our METAL_V2_SPEC already prescribes with the entropy tracer.

### 9.2 Recommended Hybrid: Entropy Tracer (Our Existing Design)

Our METAL_V2_SPEC's entropy tracer approach `S_rho = rho * p / rho^gamma` is superior to all three approaches above for our specific regime because:

1. **No subtraction**: Pressure recovered as `p = S_rho * rho^(gamma-1)` -- no cancellation error
2. **Float32 safe**: Full 7-digit precision preserved regardless of ME/KE ratio
3. **Passive advection**: Identical to Enzo-E's modern approach for the transport
4. **Conservative shock handling**: We still evolve total energy E conservatively; entropy is only used for pressure recovery in magnetically dominated regions

The switching criterion from our spec -- comparing `p_from_entropy / E_total` -- avoids the self-referential problem of FLASH's approach (which computes eint from E then compares it to ekin, both derived from E).

### 9.3 What to Adopt from FLASH/Enzo

1. **From Enzo-E**: The modern switching criterion that includes magnetic energy: `cs^2 > eta * max(v^2, B^2/rho)`. Adapt for entropy: switch to entropy-derived pressure when `p_from_etot` is unreliable.

2. **From FLASH**: The Ohmic heating source term in the auxiliary energy equation. When using resistive MHD, `eta*J^2` must be added to the entropy tracer evolution (it's a non-adiabatic source).

3. **From FLASH**: The cylindrical geometric source term `v_r/r` in the auxiliary energy equation.

4. **From both**: The post-synchronization EOS call to ensure thermodynamic consistency after selecting the energy source.

### 9.4 Specific Parameter Recommendations

| Parameter | FLASH | Enzo-E | DPF-Unified (recommended) |
|-----------|-------|--------|---------------------------|
| Switching threshold | 1e-4 | user-set eta | 1e-2 (float32-safe) |
| Magnetic energy in criterion | No | Yes | Yes (essential for DPF) |
| Floor | hy_smallE | pressure_floor/((g-1)*rho) | 1e-10 * p_ref per cell |
| Auxiliary variable | e_int (non-conservative) | e_int (passive scalar) | S_rho (entropy, passive scalar) |
| Shock detection | Via switching criterion | Via cs^2 comparison | div(v) < 0 AND dp/p > threshold |

---

## 10. Sources

### Papers
- Lee, D. (2013). "A solution accurate, efficient and stable unsplit staggered mesh scheme for three dimensional magnetohydrodynamics." [JCP 243:269](https://www.sciencedirect.com/science/article/abs/pii/S0021999113001836)
- Lee, D. & Deane, A. E. (2009). "An unsplit staggered mesh scheme for multidimensional magnetohydrodynamics." [JCP 228:952](https://www.sciencedirect.com/science/article/abs/pii/S0021999108004506)
- Fryxell, B. et al. (2000). "FLASH: An Adaptive Mesh Hydrodynamics Code." [ApJS 131:273](https://ui.adsabs.harvard.edu/abs/2000ApJS..131..273F/abstract)
- Bryan, G. L. et al. (1995). "A piecewise parabolic method for cosmological hydrodynamics." [Comp. Phys. Comm. 89:149](https://www.sciencedirect.com/science/article/abs/pii/0010465594001914)
- Bryan, G. L. et al. (2014). "ENZO: An Adaptive Mesh Refinement Code for Astrophysics." [ApJS 211:19](https://ui.adsabs.harvard.edu/abs/2014ApJS..211...19B/abstract)

### Source Code
- [FLASH MHD Jet (hy_uhd_unsplitUpdate.F90)](https://github.com/yihaochen/FLASH_MHD_Jet/blob/master/hy_uhd_unsplitUpdate.F90) -- FLASH USM update with dual energy
- [FLASH MHD Jet (hy_uhd_unsplit.F90)](https://github.com/yihaochen/FLASH_MHD_Jet/blob/master/hy_uhd_unsplit.F90) -- FLASH USM main driver
- [Enzo-E DualEnergyConfig](https://github.com/enzo-project/enzo-e/blob/main/src/Enzo/fluid-props/EnzoDualEnergyConfig.hpp) -- Enzo-E dual energy config class
- [Enzo-E FluidProps](https://github.com/enzo-project/enzo-e/blob/main/src/Enzo/fluid-props/EnzoPhysicsFluidProps.cpp) -- Switching logic implementation
- [Enzo-E HLLD](https://github.com/enzo-project/enzo-e/blob/main/src/Enzo/hydro-mhd/riemann/EnzoRiemannHLLD.hpp) -- HLLD with passive eint flux

### Documentation
- [FLASH User Guide: MHD](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node107.html)
- [FLASH User Guide: Hydrodynamics](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node104.html)
- [FLASH User Guide: 3T Capabilities](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node103.html)
- [Enzo Parameter List](https://enzo.readthedocs.io/en/latest/parameters/index.html)
- [Enzo-E Methods](https://enzo-e.readthedocs.io/en/latest/user/problem_method.html)
- [FLASH Users: Negative Internal Energy in 3T MHD](https://flash.rochester.edu/pipermail/flash-users/2019-April/002848.html)
