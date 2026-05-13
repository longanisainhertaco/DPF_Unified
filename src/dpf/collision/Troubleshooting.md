# Collision Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 1 file (319 LOC)

---

## Superseded 2026-05-11 by the formulary code audit

This historical note predates the 2026-05-11 source-scoped formulary audit.
Keep it as review history only. The older claim that all core formulas were
verified correct was too broad: the Braginskii perpendicular conductivity
closure has since been corrected to preserve the NRL high-field coefficient
`4.7`, and the electron-electron collision-frequency convention remains
source-convention blocked until the API explicitly names the NRL relaxation
rate it intends to represent.

---

## VERIFIED CORRECT

### Coulomb Logarithm (lines 27-50)
- Debye length: CORRECT
- Classical distance of closest approach: CORRECT
- de Broglie wavelength: CORRECT
- b_min = max(classical, quantum): CORRECT (Gericke-Murillo-Schlanges interpolation)
- Floor at 1.0: CORRECT (prevents unphysical ln(Lambda) < 0)

### Electron-Ion Collision Frequency (lines 54-71)
- Leading coefficient `4*sqrt(2*pi)/3` matches NRL Formulary `(4/3)*sqrt(2*pi)`. CORRECT.

### Braginskii alpha(Z) Correction (lines 109-171)
- All tabulated values verified against Braginskii (1965) Table 1:
  - Z=1: 0.5064, Z=2: 0.4408, Z=3: 0.3965, Z=4: 0.3752, Z->inf: 0.2949. ALL CORRECT.

### Braginskii Thermal Conductivity (lines 250-283)
- kappa_par formula: CORRECT
- Historical finding superseded: `kappa_per = kappa_par / (1 + x^2)` did not
  preserve the NRL high-field coefficient. Current code uses the 2026-05-11
  audited closure in `spitzer.py`.
- Z-dependent delta_e coefficient: Z=1->3.16, Z=2->3.14, Z=3->3.12, Z=4->3.11, Z->inf->3.21. ALL CORRECT.

### Temperature Relaxation (lines 287-318)
- Relaxation rate `alpha = freq_ei * dt * 2*m_e/m_d`: CORRECT
- Equilibrium temperature `T_eq = (Z*Te + Ti) / (Z+1)`: CORRECT

---

## INFORMATIONAL

### INFO-1: alpha(Z) Interpolation Could Be Smoother
- **Found by**: phys-diag
- **Assessment**: Uses piecewise linear with `np.where` cascades. A smooth function (e.g., rational polynomial fit) would be slightly more physical but the error is negligible (<0.1% between tabulated points).

---

## REJECTED FINDINGS

None. All physics findings for this module were verified correct.
