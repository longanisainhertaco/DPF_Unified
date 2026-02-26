# Radiation Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 3 files (~1,100 LOC)

---

## CRITICAL

### ✅ FIXED — CRIT-1: Bremsstrahlung Z-Scaling Bug for Z > 1
- **File:Line**: `bremsstrahlung.py:51`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Status**: **FIXED 2026-02-25** — `Z * Z` → `Z` at `bremsstrahlung_power` (line 51),
  `apply_bremsstrahlung_losses` alpha (line 121), and the inline formula in
  `line_radiation._total_rad_power_scalar`. Module docstring and comments updated.
- **Description**: The bremsstrahlung power formula used `Z * Z * ne^2` but for quasi-neutral
  plasmas (n_i = n_e / Z), the correct scaling is `Z * ne^2`.
- **Correct behavior**: Rybicki & Lightman (1979) Eq. 5.14a gives:
  ```
  P_ff = 1.426e-40 * g_ff * n_e * n_i * Z^2 * sqrt(Te)  [W/m^3]
  ```
  With quasi-neutrality (n_i = n_e / Z):
  ```
  P_ff = 1.426e-40 * g_ff * Z * n_e^2 * sqrt(Te)  [W/m^3]
  ```
- **Impact**: For Z=1 (hydrogen/deuterium — default): NO ERROR. For Z=2: 2x overestimate was present.
  For Z=10 (neon impurity): 10x overestimate was present. Now corrected.

---

## MEDIUM

### MOD-1: Hydrogen Cooling Function ~3x Higher Than Post et al. (1977)
- **File:Line**: `line_radiation.py:65`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as CONCERN (not definitively verifiable without original reference)
- **Description**: The hydrogen cooling function peaks at ~1e-31 W*m^3 near 5 eV. Published Post et al. (1977) peaks at ~3e-32 W*m^3 at ~3 eV.
- **Evidence**:
  ```python
  # Line 65:
  return 1.0e-31 * (Te_eV / 5.0) ** 1.5 * np.exp(-13.6 / Te_eV)
  ```
  At 5 eV: 1e-31 * 1.0 * exp(-2.72) = 1e-31 * 0.066 = 6.6e-33. Actually evaluates to less than 1e-31 at the peak. The concern may be less severe than initially reported.
- **Impact**: Potentially excessive cooling in partially-ionized deuterium regions. Affects thermal balance in low-temperature plasma zones.

### MOD-2: Line Radiation Cooling Function Fits Unverified
- **File:Line**: `line_radiation.py` (piecewise fits throughout)
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as concern
- **Description**: Cooling function fits for H, Ne, Ar, Cu, W are hand-tuned piecewise power laws. No systematic comparison against CHIANTI or ADAS databases.
- **Impact**: Order-of-magnitude representations are adequate for DPF simulation, but quantitative radiation loss predictions may be off by factors of 2-5.

### MOD-3: Hydrogenic Recombination Approximation
- **File:Line**: `line_radiation.py:313`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `chi_J = 13.6 * Z * Z * eV` assumes hydrogenic scaling of ionization potential. Only accurate for hydrogen-like ions (single electron).
- **Impact**: Overestimates recombination radiation for multi-electron ions. Moderate for DPF where Z is typically low.

### ✅ FIXED — MOD-4: Radiation Transport Mutates Input State In-Place
- **File:Line**: `transport.py:316-317`
- **Found by**: py-ai-diag
- **Cross-review verdict**: CONFIRMED
- **Status**: **FIXED 2026-02-25** — `apply_radiation_transport()` now creates `new_state = dict(state)`
  at entry and writes `Te` and `E_rad` to `new_state` only. The caller's state dict is no longer mutated.
- **Description**: `apply_radiation_transport()` was modifying `state["E_rad"]` and `state["Te"]` directly.
- **Impact**: Could cause operator-splitting ordering bugs if the same state is used in other operators.

---

## INFORMATIONAL

### INFO-1: Kramers Opacity Coefficient Verified
- **Found by**: phys-diag
- **Assessment**: Code uses 3.7e-2, derived value is 3.68e-2. Difference is ~0.5% (within rounding). Z-scaling in opacity is CORRECT (Z * ne^2, unlike the bremsstrahlung power bug).

### INFO-2: FLD Levermore-Pomraning Limiter Verified
- **Found by**: phys-diag
- **Assessment**: The flux-limited diffusion implementation with `lambda(R) = (coth(R) - 1/R) / R` is correct. Limits: R->0 gives 1/3 (diffusion), R->inf gives 1/R (free streaming). Sub-cycling criterion is standard.

---

## REJECTED FINDINGS

None.
