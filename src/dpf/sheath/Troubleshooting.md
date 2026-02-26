# Sheath Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 1 file (270 LOC)

---

## No critical issues found.

---

## LOW

### ✅ FIXED — LOW-1: Floating Potential Hardcodes m_p Instead of Configurable m_i
- **File:Line**: `bohm.py:149`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED (minor)
- **Fixed**: `floating_potential(Te, mi=m_p)` now accepts `mi` parameter; hardcoded `m_p` in the log argument replaced with `mi`.
- **Impact**: For hydrogen: no change. For deuterium: V_f now correct (ln(2*m_p) instead of ln(m_p)).

---

## VERIFIED CORRECT

### Bohm Velocity (line 42)
- `v_B = sqrt(kB * Te / mi)`: CORRECT per Lieberman & Lichtenberg (2005)

### Child-Langmuir Law (line 64)
- `J_CL = (4/9) * epsilon_0 * sqrt(2e/mi) * V^{3/2} / d^2`: CORRECT

### Debye Length (line 84)
- `lambda_D = sqrt(epsilon_0 * kB * Te / (ne * e^2))`: CORRECT

### 1D Poisson Solver (lines 152+)
- Tridiagonal solve with Dirichlet BCs: CORRECT implementation

---

## REJECTED FINDINGS

None.
