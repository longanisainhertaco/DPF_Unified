# Enzo Dual-Energy Formalism: Implementation Analysis

Research date: 2026-03-24
Sources: Bryan et al. (1995, CoPhC 89, 149), Bryan et al. (2014, ApJS 211, 19),
enzo-project/enzo-dev source code (main branch), enzo-project/enzo-e source code

## 1. Switching Criterion (eta1, eta2)

Enzo uses TWO thresholds with DIFFERENT roles. The logic is NOT a simple
if/else -- it's a two-stage selection with a synchronization step.

### Default values
- `DualEnergyFormalismEta1 = 0.001`
- `DualEnergyFormalismEta2 = 0.1`

### What the thresholds compare

**eta1** compares `e_int / E_total` (specific internal energy vs specific
total energy, in the same cell):

```fortran
! From pgas2d_dual.F line ~95:
if (ge1/eslice(i,j) .gt. eta1) then
    ge2 = ge1          ! use E_total-derived internal energy
else
    ge2 = geslice(i,j) ! use separately-evolved internal energy
endif
```

Where `ge1 = E_total - 0.5*(u^2 + v^2 + w^2)` is the subtraction-derived
internal energy, and `geslice` is the separately-evolved internal energy field.

**eta2** compares `e_int * rho / max_nearby(rho * E_total)` (internal energy
density vs maximum nearby total energy density):

```fortran
! From pgas2d_dual.F line ~85:
demax = max(rho(i)*E(i), rho(i-1)*E(i-1), rho(i+1)*E(i+1))
if (ge1*rho(i)/demax .gt. eta2) geslice(i,j) = ge1
```

This is a SPATIAL criterion -- if the subtraction-derived internal energy
is large enough relative to the maximum nearby total energy, it
OVERWRITES the separately-evolved internal energy with the subtraction
result. This keeps the two energy fields synchronized in regions where the
total energy equation is reliable.

### Logic flow in pgas2d_dual.F (PPM solver)

1. Compute `ge1 = E_total - KE` (subtraction from total energy)
2. Find `demax = max(rho*E in i-1, i, i+1)` (neighbor stencil)
3. If `ge1*rho/demax > eta2`: overwrite `ge_internal = ge1`
   (total energy is trustworthy here -- synchronize)
4. If `ge1/E_total > eta1`: use `ge1` for pressure
   Else: use `ge_internal` for pressure
5. Apply pressure floor
6. Resynchronize total energy: `E_total = E_total - ge1 + ge2`

### Logic in hydro_rk MHD solver (Grid_UpdateMHDPrim.C)

The RK solver uses a DIFFERENT, SIMPLER criterion. Only eta1, via cs^2:

```cpp
// From Grid_UpdateMHDPrim.C line ~273:
float eint1 = etot - 0.5*v2 - 0.5*B2/rho;  // subtraction includes B
EOS(p, rho, eint1, h, cs, dpdrho, dpde, EOSType, 2);

if (cs*cs > DualEnergyFormalismEta1*v2 &&
    cs*cs > DualEnergyFormalismEta1*B2/rho &&
    eint1 > 0.5*eint) {
    eint = eint1;  // use subtraction result
}
// Then resync: E_total = eint + 0.5*v2 + 0.5*B2/rho
```

The condition `cs^2 > eta1 * v^2` is equivalent to checking
`gamma*(gamma-1)*e_int > eta1 * v^2`, which is approximately
`e_int/E_kinetic > eta1/[gamma*(gamma-1)]`.

The ADDITIONAL condition `eint1 > 0.5*eint` prevents the subtraction
result from halving the separately-evolved value -- a sanity check against
catastrophic cancellation artifacts.

### Enzo-E "modern" formulation

Enzo-E (the newer exascale version) introduced a simplified single-parameter
formulation in `EnzoPhysicsFluidProps.cpp`:

```cpp
// From EnzoPhysicsFluidProps.cpp line ~274:
enzo_float cs2_1 = max(0., ggm1 * eint_1);  // gamma*(gamma-1)*eint

if (cs2_1 > max(eta*v2, eta*B2/rho) && eint_1 > 0.5*cur_eint) {
    cur_eint = eint_1;  // use subtraction result
}
// Then resync: etot = cur_eint + non_thermal_e
```

This uses a single `eta` parameter (no separate eta1/eta2) and drops the
spatial neighbor check entirely. The implementation comment confirms:
"In enzo-dev's hydro_rk, eta was set equal to eta1 (it didn't ever use eta2)".


## 2. Synchronization Between Energy Equations

Yes, there is explicit synchronization, and it goes BOTH directions:

### Total -> Internal (when total energy is reliable)
In `pgas2d_dual.F`: if `ge1*rho/demax > eta2`, the internal energy field
is OVERWRITTEN with the subtraction result from total energy. This prevents
drift between the two fields in subsonic regions.

### Internal -> Total (when internal energy is reliable)
In `pgas2d_dual.F` line ~101:
```fortran
eslice(i,j) = eslice(i,j) - ge1 + ge2
```
This replaces the internal-energy portion of the total energy with the
selected `ge2` value. This is the KEY synchronization step -- it prevents
the total energy from accumulating drift relative to the internal energy.

### In the RK/MHD solver (Grid_UpdateMHDPrim.C)
```cpp
BaryonField[GENum][igrid] = eint;
BaryonField[TENum][igrid] = eint + 0.5*v2 + 0.5*B2/D_new;
```
When the separately-evolved internal energy is selected, the total energy
is RECONSTRUCTED from it. This is a complete resynchronization -- the total
energy field is replaced, not incrementally adjusted.

### Frequency
Synchronization happens EVERY timestep, AFTER the Riemann solve and
conservative update. It is not a periodic correction -- it is integral to
the pressure computation step.


## 3. Ohmic Heating (eta*J^2) in MHD Dual Energy

**Enzo's resistivity implementation (Grid_AddResistivity.C) does NOT add
ohmic heating to EITHER energy equation.** The resistive term only updates
the magnetic field via diffusion:

```cpp
B[i] += dt/dx * eta/dx * laplacian(B)
```

There is no `E_total += eta*J^2` or `e_int += eta*J^2` term. This is a
known limitation -- energy is not conserved under resistive MHD in Enzo.

**In Grid_MHDSourceTerms.C**, the dual energy source terms handle:
- `-p * div(v)` for the internal energy (the pdV work term)
- Cosmic ray heating (added to BOTH e_int and E_total when CRHeating=true)

But resistive/ohmic heating is absent from both energy equations.

**Implication for DPF:** If implementing ohmic heating, it MUST be added to
BOTH energy equations to maintain consistency. Otherwise the switching
criterion will see different energies and produce artifacts at the transition.


## 4. Internal Energy Advection Method

### PPM solver (ppm_de / euler.F / flux_twoshock.F)
The internal energy is advected through the **Riemann solver flux machinery**
with a SEPARATE flux and source term. From `flux_twoshock.F` line ~387:

```fortran
! Flux term: advection of gas energy through interface
gef(i,j) = (dt/dx) * dub(i) * geb(i)

! Source term: -p * div(v) (the pdV work)
ges(i,j) = (dt/dx) * p_center * (u_left - u_right)
```

Where:
- `dub(i) = rho_interface * u_interface` is the mass flux
- `geb(i) = p_interface / ((gamma-1) * rho_interface)` is the interface
  specific internal energy (derived from the Riemann-solved pressure)
- The source term uses cell-centered pressure and interface velocities

These are combined in `euler.F`:
```fortran
geslice(i,j) = (geslice(i,j)*rho_old + (gef(i,j) - gef(i+1,j)) + ges(i,j))
               / rho_new
```

This is NOT a simple passive scalar -- the flux uses the Riemann-solved
interface pressure to compute the interface internal energy, and there is an
explicit `-p*div(v)` source term.

### PPM interface reconstruction (inteuler.F)
The gas energy IS reconstructed like an advected quantity (same `intvar`
call as passive scalars), and the left/right states are selected based on
the upwind direction:

```fortran
if (uslice(i-1,j) .le. 0) then
    gels(i,j) = gela(i)    ! left state from averaging
else
    gels(i,j) = gel0(i)    ! left state from characteristics
endif
```

### Enzo-E VLCT/HLLD solver
Internal energy is explicitly treated as a **passive scalar** in the Riemann
solver. From `EnzoRiemannHLLD.hpp` line 9:

> "Currently, eint fluxes are computed by assuming that specific internal
> energy is a passive scalar."

The implementation in `EnzoRiemannUtils.hpp`:
```cpp
// Upwind flux: F(eint) = eint_upwind * F(rho)
enzo_float upwind = (density_flux > 0) * left + (density_flux <= 0) * right;
return upwind * density_flux;
```

The `-p*div(v)` source term is added separately in `Grid_MHDSourceTerms.C`:
```cpp
if (DualEnergyFormalism) {
    dU[iEint][n] -= p * divVdt;
}
```

### Summary of advection methods

| Solver | Flux method | Source term |
|--------|------------|-------------|
| PPM (enzo-dev) | Riemann-solved interface e_int | -p*div(v) via interface velocities |
| hydro_rk (enzo-dev) | Conservative RK update of rho*e_int | -p*div(v) operator-split source |
| VLCT/HLLD (enzo-e) | Passive scalar (upwind * F_rho) | -p*div(v) operator-split source |


## 5. Shock Detector for Synchronization

**Enzo does NOT use an explicit shock detector (like div(v) < 0) for the
dual-energy synchronization.**

Instead, the switching criterion IS the shock detector, implicitly:

- Where `e_int/E_total > eta1` (subsonic/mildly supersonic flow), the
  subtraction from total energy is reliable, so the code uses it and
  synchronizes the separately-evolved field.
- Where `e_int/E_total < eta1` (highly supersonic flow), the subtraction
  is corrupted by cancellation, so the code trusts the separately-evolved
  internal energy.

The `eta2` spatial criterion in the PPM solver adds neighborhood awareness:
by comparing against `max(rho*E)` in the 3-cell stencil, it avoids
synchronizing internal energy in cells adjacent to strong shocks where the
total energy might be unreliable.

There IS a related shock detector used for **cell flagging** (AMR
refinement), where `div(v) < 0` triggers refinement. The documentation
mentions "an extra filter which removes weak shocks or noise in the dual
energy fields from triggering the shock detection." But this is for AMR, not
for the dual-energy switching itself.


## 6. Contact Discontinuities and -p*div(v)

**Enzo does nothing special at contact discontinuities.**

The `-p*div(v)` source term in the internal energy equation is computed from
interface velocities:

```fortran
ges(i,j) = (dt/dx) * p_center * (u_left_interface - u_right_interface)
```

At a contact discontinuity, `u` and `p` are continuous, so
`u_left - u_right` should be zero (or very small). The numerical error is
proportional to the reconstruction error of the velocity field at the
interfaces, which is typically small for a well-resolved contact.

However, there is no explicit detection or special treatment. The PPM
flattening algorithm (in `calcdiss.F`) reduces reconstruction order near
shocks but does NOT specifically target contacts.

The dual-energy formalism provides indirect protection: if `-p*div(v)`
errors corrupt the internal energy at a contact, the synchronization step
(total -> internal when `ge1*rho/demax > eta2`) will overwrite the corrupted
value with the subtraction result from total energy, which is more reliable
at contacts (since kinetic energy doesn't change across a contact).


## 7. Known Failure Modes

### 7.1 Conservation violation at synchronization
When the separately-evolved internal energy is selected for pressure AND
used to resynchronize total energy (`E = E - ge1 + ge2`), the total energy
field is modified outside the conservative update. This means **total energy
is NOT strictly conserved** when the dual-energy formalism is active.

The violation is small in practice (limited to cells where `e_int/E < eta1`),
but it IS present. For DPF simulations with ME/p > 10^6, a large fraction
of the domain would trigger switching, amplifying conservation errors.

### 7.2 Circular dependency in the switching criterion
The switching criterion `ge1/E_total > eta1` requires computing
`ge1 = E_total - KE`, which IS the catastrophic cancellation that the
dual-energy formalism is supposed to avoid. If the subtraction result is
garbage (negative, or orders of magnitude wrong), the comparison against
eta1 may make the wrong choice.

The `eint1 > 0.5*eint` safety check (in the RK solver) partially mitigates
this: even if `ge1` is wrong, we won't use it if it's less than half the
separately-evolved value. But if `ge1` is LARGE and wrong (e.g., from
floating-point overflow in the subtraction), the check could fail.

**This is the exact problem an entropy-based criterion would solve.**
Computing `p_from_entropy / E_total` never involves the corrupted subtraction.

### 7.3 Drift between energy fields in magnetically-dominated regions
In cells where `cs^2 < eta1*v^2` (or `cs^2 < eta1*B^2/rho` for MHD), the
code uses the separately-evolved internal energy and does NOT synchronize
with total energy. Over many timesteps, the two fields can diverge
significantly. When the flow changes and the cell transitions back to
using total energy, there may be a discontinuous jump in pressure.

### 7.4 Missing ohmic heating
As noted in Section 3, resistive heating is not added to either energy
equation. In resistive MHD, this means energy is not conserved at all,
and the dual-energy switching criterion sees inconsistent energies.

### 7.5 MHDCTDualEnergyMethod entropy variant
Enzo has a parameter `MHDCTDualEnergyMethod` that, when set, switches
from evolving internal energy to evolving entropy. This addresses some
of the `-p*div(v)` issues at contacts but introduces its own problems:
entropy is not conserved across shocks, requiring explicit entropy
resetting at detected shocks.

### 7.6 Float32 precision
Enzo uses float32 for field data (`R_PREC = float`). With
`eta1 = 0.001`, the switching happens when `e_int/E_total < 0.001`,
which means ~3 digits of the ~7 available in float32 are lost to
cancellation. In float64 this leaves 12 good digits; in float32 it
leaves only 4, which can still produce ~0.01% pressure errors in the
"reliable" regime.

For DPF with ME/p > 10^6, we need `e_int/E_total < 10^-6`, which
means ALL significant digits in float32 are lost. The dual-energy
formalism becomes essential, not optional.

### 7.7 AMR boundary synchronization
From Enzo-E source (EnzoMethodMHDVlct.cpp line ~481):
> "the interface velocities on the edge of the blocks will be different
> if using SMR/AMR. This means that the internal energy source terms
> won't be fully self-consistent along the edges."

The `-p*div(v)` source term uses interface velocities that may differ
between AMR levels, leading to inconsistent internal energy updates at
refinement boundaries.


## 8. Implications for DPF Metal Solver

### Why Enzo's approach is a good baseline
1. Proven in production for 30+ years
2. The two-threshold system (eta1 for local, eta2 for spatial) is
   well-thought-out
3. The `-p*div(v)` source term approach is standard and correct
4. The passive-scalar advection in Enzo-E is simple and GPU-friendly

### Why we need improvements
1. **Float32 precision**: At ME/p > 10^6, the switching criterion itself
   involves the corrupted subtraction. Entropy-based switching avoids this.
2. **Ohmic heating**: Must be added to BOTH energy equations. Enzo skips this.
3. **Conservation**: The synchronization step breaks conservation. For DPF
   energy balance matters (neutron yield depends on pinch energy). Consider
   tracking the conservation error explicitly.
4. **Magnetically dominated regime**: Enzo's criterion checks both
   `cs^2 > eta1*v^2` AND `cs^2 > eta1*B^2/rho`. For DPF near electrodes,
   the `B^2/rho` term dominates and will force dual-energy usage almost
   everywhere.

### Recommended implementation
1. **Evolve**: internal energy via passive scalar advection (upwind flux
   using HLL density flux, like Enzo-E)
2. **Source term**: `-p*div(v)` computed from interface velocities,
   operator-split, applied to internal energy only
3. **Switching criterion**: entropy-based.
   `p_entropy = rho^gamma * exp(s/cv)` computed from the evolved entropy
   scalar. Switch when `p_entropy / (rho * E_total) < eta`. This never
   touches the corrupted subtraction.
4. **Synchronization**: When using total-energy-derived pressure, also
   update the internal energy field to prevent drift. When using
   internal-energy-derived pressure, reconstruct total energy. Same as Enzo.
5. **Ohmic heating**: Add `eta*J^2/rho` to BOTH energy equations.
6. **Conservation tracking**: Log `sum(E_total)` before and after
   synchronization to monitor conservation error.


## Source Files Referenced

### enzo-dev (github.com/enzo-project/enzo-dev)
- `src/enzo/pgas2d_dual.F` -- pressure recovery with dual energy (PPM)
- `src/enzo/euler.F` -- Eulerian update with gas energy flux+source
- `src/enzo/flux_twoshock.F` -- Riemann flux computation including ge flux
- `src/enzo/inteuler.F` -- interface reconstruction including ge states
- `src/enzo/Grid_SolvePPM_DE.C` -- PPM driver passing dual energy params
- `src/enzo/Grid_SolveHydroEquations.C` -- hydro driver
- `src/enzo/Grid_ComputePressure.C` -- pressure with MHD dual energy
- `src/enzo/hydro_rk/Grid_UpdatePrim.C` -- RK hydro dual energy sync
- `src/enzo/hydro_rk/Grid_UpdateMHDPrim.C` -- RK MHD dual energy sync
- `src/enzo/hydro_rk/Grid_MHDSourceTerms.C` -- -p*div(v) source term
- `src/enzo/hydro_rk/Grid_AddResistivity.C` -- resistive MHD (no ohmic heating)
- `src/enzo/Grid_MHDCTEnergyToggle.C` -- CT energy format conversion

### enzo-e (github.com/enzo-project/enzo-e)
- `src/Enzo/fluid-props/EnzoDualEnergyConfig.hpp` -- dual energy config (modern vs bryan95)
- `src/Enzo/fluid-props/EnzoPhysicsFluidProps.cpp` -- sync implementation
- `src/Enzo/hydro-mhd/EnzoMethodMHDVlct.cpp` -- VL+CT MHD solver
- `src/Enzo/hydro-mhd/riemann/EnzoRiemannHLLD.hpp` -- HLLD with passive eint
- `src/Enzo/hydro-mhd/riemann/EnzoRiemannUtils.hpp` -- passive scalar flux
