# Diagnostics Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 7 files (~1,100 LOC)

---

## CRITICAL

### CRIT-1: neutron_yield.py Branch 2 Reuses Branch 1's theta/xi Coefficients — ✅ FIXED
- **File:Line**: `neutron_yield.py:90-110`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: The D(d,p)T branch (Branch 2) reuses the theta and xi computed with Branch 1's C2-C7 coefficients. Bosch & Hale (1992) Table VII provides different C2-C7 for each branch.
- **Evidence**:
  ```python
  # Lines 90-93: Branch 2 reuses theta/xi from Branch 1
  C1_2 = 5.65718e-12
  sv_2 = C1_2 * theta * np.sqrt(xi / (mu_c2 * T**3)) * np.exp(-3.0 * xi)
  ```
- **Correct behavior**: Branch 2 should have:
  - C2 = 3.41267e-3, C3 = 1.99167e-3, C4 = 0.0, C5 = 1.05060e-5, C6 = 0.0, C7 = 0.0
  - Compute theta2 and xi2 separately using these coefficients.
- **Proposed fix**:
  ```python
  # Branch 2: D(d,p)T — use CORRECT Bosch-Hale coefficients
  C1_2 = 5.65718e-12
  C2_2, C3_2 = 3.41267e-3, 1.99167e-3
  C4_2, C5_2 = 0.0, 1.05060e-5
  C6_2, C7_2 = 0.0, 0.0
  denom2 = 1.0 + T * (C3_2 + T * (C5_2 + T * C7_2))
  numer2 = T * (C2_2 + T * (C4_2 + T * C6_2))
  theta2 = T / (1.0 - numer2 / denom2)
  xi2 = (B_G**2 / (4.0 * theta2)) ** (1.0 / 3.0)
  sv_2 = C1_2 * theta2 * np.sqrt(xi2 / (mu_c2 * T**3)) * np.exp(-3.0 * xi2)
  ```
- **Justification**: Bosch & Hale (1992) Table VII: different polynomial fits for D(d,n)He-3 vs D(d,p)T.
- **Impact**: ~5-10% error in total DD reactivity at DPF temperatures (1-10 keV). Bounded because both branches have similar magnitude. Still violates principle of faithful reference implementation.

### CRIT-2: instability.py m=0 Growth Rate Formula Never Allows Stabilization — ✅ FIXED
- **File:Line**: `instability.py:77-83`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: The dispersion relation `arg = 1.0 - beta_p / (2.0 + gamma * beta_p)` can never produce a negative argument, meaning the code predicts that ALL Z-pinches are unstable regardless of pressure.
- **Evidence**:
  ```python
  # Line 86-87:
  denom = 2.0 + gamma * beta_p
  arg = 1.0 - beta_p / max(denom, 1e-30)
  ```
  Mathematical proof: beta_p / (2 + gamma*beta_p) < 1/gamma for all beta_p > 0 (since d/d(beta_p)[beta_p/(2+gamma*beta_p)] = 2/(2+gamma*beta_p)^2 > 0 and the limit is 1/gamma). For gamma=5/3: arg > 1 - 3/5 = 2/5 > 0 always.
- **Correct behavior**: The Kadomtsev (1966) result for m=0 with adiabatic compressibility should allow stability at high beta_p. The simplest correct form is:
  ```
  gamma^2 = k^2 * v_A^2 * (1 - gamma_gas * beta_p / 2)
  ```
  which gives stability when beta_p > 2/gamma_gas = 1.2 for gamma=5/3.
- **Proposed fix**:
  ```python
  # Correct Kadomtsev (1966) formula:
  arg = 1.0 - gamma * beta_p / 2.0
  ```
  And update stability criterion: `beta_p_crit = 2.0 / gamma` (not `2.0 / (gamma - 1.0)`).
- **Justification**: Kadomtsev (1966) Rev. Plasma Phys. 2:153; Haines (2011) PPCF 53:093001.
- **Impact**: m=0 growth rate calculations never predict stability. Diagnostic-only (doesn't feed back into MHD), but produces misleading instability analysis. The stability margin calculation is also wrong.

---

## HIGH

### HIGH-1: derived.py geometry Parameter Accepted But Ignored — ✅ FIXED
- **File:Line**: `derived.py:17-62`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `current_density_magnitude()` accepts `geometry='cylindrical'` but always computes Cartesian curl(B)/mu_0.
- **Evidence**: No conditional on `geometry` in the function body. Lines 37-39 always use `np.gradient` with Cartesian form.
- **Correct behavior**: For cylindrical geometry (r, theta, z):
  ```
  curl(B)_r = (1/r)*dBz/dtheta - dB_theta/dz
  curl(B)_z = (1/r)*d(r*B_theta)/dr - (1/r)*dBr/dtheta
  ```
- **Proposed fix**: Implement cylindrical curl or raise `NotImplementedError` for `geometry="cylindrical"`.
- **Impact**: Wrong J magnitude in cylindrical simulations. Used for diagnostics only, not fed back to solver.

---

## MEDIUM

### MOD-1: Beam-Target Neutron Anisotropy Model Is Ad-Hoc
- **File:Line**: `beam_target.py:311`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as known approximation
- **Description**: `A_bt = 1.0 + 0.3 * sqrt(E_beam / 100 keV)` has no published reference. The coefficient 0.3 and sqrt(E) scaling are phenomenological.
- **Impact**: Low — anisotropy is diagnostic-only, output range (1.3-2.5) is physically reasonable per Bernard et al. (1977) measurements.

### MOD-2: interferometry.py O(N^2) Abel Transform
- **File:Line**: `interferometry.py:56-87`
- **Found by**: phys-diag (performance), py-ai-diag (confirmed)
- **Cross-review verdict**: CONFIRMED
- **Impact**: Slow for grids >256 radial cells. Correctness is fine. Consider `scipy.ndimage` or `abel` package.

---

## INFORMATIONAL

### INFO-1: Beam-Target Missing Beam Slowing-Down
- **Found by**: phys-diag
- **Assessment**: Single-pass beam traversal model is justified for DPF conditions (n_target ~ 10^25 m^-3, L_target ~ 10 mm → mean free path >> L_target). Adding Bethe-Bloch stopping power would improve accuracy for dense targets but is not a bug.

### INFO-2: Bosch-Hale Coefficients Verified Correct
- **Found by**: phys-diag
- **Assessment**: All beam-target DD cross section coefficients (A1-A5, B1-B4, B_G, mu_c^2) match Bosch & Hale (1992) Table IV exactly. Lab-to-CM conversion E_cm = E_lab/2 is correct for equal-mass DD.

---

## REJECTED FINDINGS

None — all diagnostics findings from both reviewers were confirmed or accepted as known approximations.
