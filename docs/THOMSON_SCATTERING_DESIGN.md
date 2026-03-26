# Thomson Scattering Synthetic Diagnostic — Design Document

## 1. Physics

Thomson scattering measures ne, Te, and v_bulk by analyzing laser light scattered
from plasma electrons. The differential cross-section per electron per solid angle
per wavelength interval is:

    dP/d(lambda) = ne * sigma_T * P_laser / A_beam
                   * S(k, omega)

where sigma_T = 6.652e-29 m^2 is the total Thomson cross-section and S(k, omega)
is the spectral density function (the "form factor").

### Scattering Geometry

The scattering wavevector magnitude:

    k = (4 * pi / lambda_0) * sin(theta / 2)

where lambda_0 is the probe laser wavelength and theta is the scattering angle.

### Collective vs Non-Collective Regimes

The Salpeter parameter alpha determines which regime applies:

    alpha = 1 / (k * lambda_D)

    lambda_D = sqrt(epsilon_0 * k_B * Te / (ne * e^2))

- alpha << 1: non-collective (incoherent). Spectrum reflects single-electron
  thermal motion. Standard for DPF conditions with visible lasers.
- alpha >> 1: collective (coherent). Ion acoustic and electron plasma wave
  features appear.

For PF-1000 pinch (ne ~ 1e25 m^-3, Te ~ 300 eV, lambda_0 = 1064 nm, theta = 90 deg):

    lambda_D ~ 1.3e-7 m, k ~ 8.4e6 m^-1 => alpha ~ 0.9

This places DPF plasmas near the transition. Both regimes must be supported.

### Non-Collective Spectral Shape (alpha < 0.5)

The spectrum is a Gaussian reflecting the electron velocity distribution:

    S_e(lambda) = (c / (lambda_0 * v_th)) * (1 / sqrt(2*pi))
                  * exp(-0.5 * ((lambda - lambda_0 - delta_D) / delta_th)^2)

Thermal width (1/e half-width in wavelength):

    delta_th = lambda_0 * (2 * k_B * Te / (m_e * c^2))^0.5 * sin(theta/2)

For Te = 300 eV, lambda_0 = 1064 nm, theta = 90 deg: delta_th ~ 15 nm.

Doppler shift from bulk flow v_bulk:

    delta_D = (lambda_0 / c) * (v_bulk . k_hat) * 2 * sin(theta/2)

### Collective Spectral Shape (alpha > 1)

The full Salpeter form factor splits into electron and ion features:

    S(k, omega) = S_e(k, omega) + Z * S_i(k, omega)

The ion feature (narrow, near lambda_0) yields Ti; the electron feature (broad
wings) yields Te. Implementation uses the Salpeter approximation:

    S_e = (2*pi/k) * (1/v_th_e) * |1 - chi_e/epsilon|^2 * f_e(omega/k)
    S_i = (2*pi/k) * (Z/v_th_i) * |chi_e/epsilon|^2 * f_i(omega/k)

where epsilon = 1 + chi_e + chi_i is the plasma dielectric function and chi_e,i
are the electron/ion susceptibilities (Fried-Conte Z-function).

For alpha ~ 0.5-2 (DPF transition regime), the full form factor is required.
Simplification: use the Gaussian approximation with a spectral broadening
correction factor (1 + alpha^2)^(-1/2) as in Sheffield (2011) Ch. 5.

### Scattered Power

Total scattered power per steradian per unit wavelength per unit length:

    dP/(d_Omega * d_lambda * dL) = ne * r_e^2 * P_laser / A_beam * S(k, omega)

where r_e = 2.818e-15 m is the classical electron radius.

## 2. Implementation Design

### Module: `src/dpf/diagnostics/thomson_scattering.py`

```python
def thomson_spectrum(
    ne: np.ndarray,          # electron density [m^-3], shape (N,)
    Te_eV: np.ndarray,       # electron temperature [eV], shape (N,)
    v_bulk: np.ndarray,      # bulk velocity [m/s], shape (N,) — projected onto k_hat
    wavelength_grid: np.ndarray,  # scattered wavelengths [m], shape (M,)
    scattering_angle: float = np.pi / 2,  # theta [rad]
    laser_wavelength: float = 1064e-9,    # probe wavelength [m]
) -> np.ndarray:
    """Spectral power density at each spatial point.

    Returns: shape (N, M) — power spectral density [W/m^3/sr/m] at each
    spatial point and wavelength.
    """
```

Core computation:
1. Compute delta_th from Te_eV (vectorized over N spatial points)
2. Compute delta_D from v_bulk
3. Compute alpha = 1/(k * lambda_D) to select regime
4. For alpha < 0.5: pure Gaussian
5. For alpha > 2: Salpeter two-feature form
6. For 0.5 <= alpha <= 2: Gaussian with Sheffield correction factor
7. Scale by ne * r_e^2

```python
def thomson_line_integrated(
    ne_2d: np.ndarray,       # shape (nr, nz)
    Te_2d: np.ndarray,       # shape (nr, nz) [eV]
    vz_2d: np.ndarray,       # axial velocity [m/s], shape (nr, nz)
    r_cell: np.ndarray,      # radial grid [m], shape (nr,)
    chord_positions_z: np.ndarray,  # axial positions of chords [m], shape (Nc,)
    wavelength_grid: np.ndarray,    # shape (M,)
    scattering_angle: float = np.pi / 2,
    laser_wavelength: float = 1064e-9,
) -> np.ndarray:
    """Line-integrated Thomson spectrum along radial chords.

    Uses Abel transform from interferometry.py for the line integration
    through the axisymmetric plasma.

    Returns: shape (Nc, M) — spectral power [W/m^2/sr/m] per chord.
    """
```

Core computation:
1. For each chord at z = chord_positions_z[c], extract radial profiles
   ne(r), Te(r), v(r) by interpolation to the nearest z-index
2. At each radial point, call thomson_spectrum to get S(r, lambda)
3. Abel-transform the spectral emissivity: for each wavelength bin,
   treat S(r, lambda_m) as a radial profile and apply abel_transform()
4. Result: line-integrated spectrum at each chord position

```python
def fit_te_ne_v(
    wavelength_grid: np.ndarray,  # [m], shape (M,)
    spectrum: np.ndarray,         # measured/synthetic spectrum, shape (M,)
    scattering_angle: float = np.pi / 2,
    laser_wavelength: float = 1064e-9,
) -> dict[str, float]:
    """Extract Te, ne, v_flow from a Thomson spectrum.

    Returns: {"Te_eV": float, "ne_m3": float, "v_flow_ms": float}
    """
```

Fitting procedure:
1. Total power (integral of spectrum) -> ne
2. Spectral width (Gaussian fit sigma) -> Te = m_e * c^2 * (sigma/lambda_0)^2 / (2 * k_B * sin^2(theta/2))
3. Peak shift from lambda_0 -> v_flow = c * delta_lambda / (2 * lambda_0 * sin(theta/2))

### Abel Transform Reuse

The `abel_transform(profile, r)` from `interferometry.py` integrates a radial
profile f(r) along chords at impact parameter y. Thomson scattering reuses this
by treating the spectral emissivity at each wavelength bin as a separate radial
profile:

```python
from dpf.diagnostics.interferometry import abel_transform

for m in range(n_wavelengths):
    line_integrated[:, m] = abel_transform(emissivity[:, m], r_cell)
```

This avoids reimplementing the singularity-handled quadrature.

## 3. Input/Output Specification

### Inputs

From MHD state dict:
- `rho` (nr, nz): mass density [kg/m^3]
- `pressure` (nr, nz): total pressure [Pa]
- `velocity` (3, nr, nz): velocity components [m/s]
- `Te` (nr, nz): electron temperature [K] (convert to eV via k_B)

Derived:
- ne = rho / m_i (fully ionized deuterium, Z=1)
- Te_eV = Te * k_B / eV  (or from pressure: Te_eV = p / (2 * ne * eV) for Z=1)
- v_proj = velocity projected onto scattering k-vector

Probe geometry:
- `laser_wavelength`: float [m] (default 1064 nm — Nd:YAG)
- `scattering_angle`: float [rad] (default pi/2)
- `chord_positions_z`: array [m] — axial locations of measurement chords

### Outputs

- `thomson_spectrum` returns: shape (N, M) array [W/m^3/sr/m]
- `thomson_line_integrated` returns: shape (Nc, M) array [W/m^2/sr/m]
- `fit_te_ne_v` returns: dict with Te_eV, ne_m3, v_flow_ms

### Units Convention

All SI. Wavelengths in meters (not nm). Temperatures in eV for physics functions,
matching the existing xray_imaging.py convention.

## 4. Validation Plan

### Analytical Tests

1. **Uniform plasma, known Te**: Generate spectrum for uniform ne=1e24 m^-3,
   Te=100 eV, v=0. Verify Gaussian width matches delta_th formula to <1%.
   Verify total power matches ne * sigma_T * P_laser.

2. **Doppler shift**: Uniform plasma with v_bulk = 1e5 m/s. Verify peak
   shift matches delta_D formula.

3. **Abel transform roundtrip**: Parabolic ne(r) profile. Thomson spectrum
   at each r -> Abel transform -> verify line-integrated spectrum is
   consistent with direct chord integration (trapezoidal, independent code).

4. **alpha regime transition**: Sweep ne from 1e22 to 1e26 at fixed Te=200 eV.
   Verify alpha transitions through 0.5, 1, 2. Spectral shape should
   smoothly transition (no discontinuities at regime boundaries).

5. **fit_te_ne_v roundtrip**: Generate synthetic spectrum with known (ne, Te, v),
   feed to fit function, recover inputs to <5% error.

### Literature Comparison

- Sheffield (2011) Fig 5.2: spectral shape for alpha = 0.1, 1, 3. Reproduce
  qualitatively.
- DPF Thomson measurements are rare. Closest: Decker et al. (1996) on
  NX-1 device — reported Te ~ 200-500 eV from Thomson scattering of pinch
  plasma. Use as order-of-magnitude sanity check if PF-1000 simulation
  produces Te in same range.

## 5. LOC Estimate

| Component | LOC |
|-----------|-----|
| `thomson_spectrum()` | ~50 |
| `thomson_line_integrated()` | ~40 |
| `fit_te_ne_v()` | ~30 |
| Constants, imports, docstrings | ~30 |
| Salpeter correction (alpha > 0.5) | ~50 |
| **Total core module** | **~200** |
| Test file (5 tests) | ~80 |

## 6. Dependencies

- `numpy` (existing)
- `dpf.constants`: e, epsilon_0, m_e, k_B, c, pi, eV
- `dpf.diagnostics.interferometry`: abel_transform
- `scipy.optimize.curve_fit` (for fit_te_ne_v only)

No new external dependencies.

## 7. Key Equations Summary

| Quantity | Formula |
|----------|---------|
| Thomson cross-section | sigma_T = 8*pi*r_e^2/3 = 6.652e-29 m^2 |
| Classical electron radius | r_e = e^2/(4*pi*epsilon_0*m_e*c^2) = 2.818e-15 m |
| Debye length | lambda_D = sqrt(epsilon_0*k_B*Te / (ne*e^2)) |
| Salpeter parameter | alpha = 1/(k*lambda_D), k = (4*pi/lambda_0)*sin(theta/2) |
| Thermal spectral width | delta_th = lambda_0 * sqrt(2*k_B*Te/(m_e*c^2)) * sin(theta/2) |
| Doppler shift | delta_D = (lambda_0/c) * v_proj * 2*sin(theta/2) |
| Te from width | Te = m_e*c^2*(sigma_fit/lambda_0)^2 / (2*k_B*sin^2(theta/2)) |
| ne from total power | ne = P_scattered / (sigma_T * P_laser * dL / A_beam) |

## 8. References

1. Sheffield, Froula, Glenzer, Luhmann — "Plasma Scattering of Electromagnetic
   Radiation", 2nd ed., Academic Press (2011). Chapters 3-5.
2. Hutchinson — "Principles of Plasma Diagnostics", 2nd ed., Cambridge (2002).
   Chapter 7.
3. Salpeter — "Electron Density Fluctuations in a Plasma", Phys. Rev. 120:1528
   (1960). Original form factor derivation.
4. Decker, Kies, Malzig et al. — "Thomson scattering diagnostics of a plasma
   focus", Plasma Sources Sci. Technol. 5:112 (1996). DPF-specific Thomson.

## 9. Integration Notes

- Register in `__init__.py` alongside interferometry and xray_imaging
- Follow the same function signature pattern: 2D arrays in (nr, nz),
  radial grid as r_cell, derived quantities computed internally
- The module is diagnostic-only (reads state, never mutates it)
- Compatible with all backends (Python, Metal, MLX, Athena++) since it
  operates on the unified state dict
