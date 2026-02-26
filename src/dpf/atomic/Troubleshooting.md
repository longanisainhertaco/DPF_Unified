# Atomic Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 2 files (~726 LOC)

---

## MEDIUM

### MOD-1: Saha Equation Statistical Weight Convention
- **File:Line**: `ionization.py:99-100`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as acceptable convention (NOT a bug)
- **Description**: The code uses `g_ratio = 2.0` in the Saha parameter. The physics reviewer initially raised this as potentially wrong, noting that the exact partition function ratio for hydrogen is subtle.
- **Evidence**:
  ```python
  # Lines 99-100:
  g_ratio = 2.0
  S = thermal_factor * g_ratio * exp_factor / ne
  ```
- **Assessment**: The "2" here represents the electron continuum degeneracy (spin-up/spin-down states of the free electron). This is the standard convention in Chen, Griem, and most plasma physics textbooks. The alternative convention (explicit g_ion/g_neutral = 1/2 with leading factor 2) gives the same Saha parameter. The physics reviewer ultimately concluded "ACCEPTABLE — this is a matter of convention."
- **Impact**: None for hydrogen. The overall Saha parameter is within convention bounds.

### MOD-2: Lotz Ionization Rate — Non-Standard Approximation ✅ FIXED
- **File:Line**: `ionization.py:201`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as known approximation
- **Description**: The code uses `sqrt(u) * exp(-u) * (1 + 0.5/u)` as an approximation to the Lotz (1967) formula involving the exponential integral E_1(u). This is reasonable for u ~ 1-10 (DPF conditions) but deviates at low u (high temperature).
- **Impact**: Moderate for high-T plasmas. Adequate for DPF operating range.
- **Fix applied**: Expanded inline comment to document accuracy bounds (~20% for u in [0.5, 20], i.e. Te ~ 0.5-2x I_Z) and the high-T regime where the full Ei-based formula would be needed.

### MOD-3: Ablation Model Lacks Temperature Dependence and Latent Heat ✅ FIXED
- **File:Line**: `ablation.py:144`
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED
- **Description**: `S_rho = ablation_efficiency * eta * J^2` uses constant efficiency. No temperature threshold (melting/sublimation), no latent heat, no plasma shielding effects.
- **Evidence**: `ablation_efficiency` is a fixed scalar (Cu: 5e-5 kg/J, W: 2e-5 kg/J).
- **Impact**: Overestimates ablation at low surface temperatures, underestimates plasma shielding effects. Adequate for order-of-magnitude ablation modeling.
- **Fix applied**: Added `COPPER_MELT_TEMP = 1356.0 K` and `TUNGSTEN_MELT_TEMP = 3695.0 K` constants. Added optional `T_threshold_K: float = 0.0` parameter to `ablation_source()` — when set to a material melting point, ablation is suppressed below that temperature. Default is 0.0 (backward compatible, threshold disabled). Latent heat and plasma shielding remain future work.

---

## REJECTED FINDINGS

### REJECTED: Ionization Potential Values
- **Reason**: phys-diag noted Cu and W ionization potentials are "from NIST (credible)." Verified: `_IP_CU` and `_IP_W` arrays use standard NIST ASD values. No issues.
