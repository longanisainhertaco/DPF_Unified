# Turbulence Module Troubleshooting

Cross-review by: xreview-ai (Python + Physics expert synthesis)
Date: 2026-02-25
Files reviewed: 1 file (486 LOC)

---

## MEDIUM

### ✅ FIXED — MOD-1: Ion-Acoustic Threshold Uses v_ti Instead of c_s (Sound Speed)
- **File:Line**: `anomalous.py:140-142` (original); fixed throughout
- **Found by**: phys-diag
- **Cross-review verdict**: CONFIRMED as known approximation (previously debated by PhD panel)
- **Description**: The ion-acoustic instability threshold used `v_d > v_ti = sqrt(kB*Ti/mi)` (ion thermal speed). The standard threshold is `v_d > c_s = sqrt(kB*Te/mi)` (ion sound speed, which involves ELECTRON temperature).
- **Fix applied** (2026-02-25):
  - Added `ion_sound_speed(Te, mi)` njit helper (`c_s = sqrt(k_B*Te/m_i)`)
  - `ion_acoustic_threshold(J, ne, Te, mi)`: parameter renamed `Ti→Te`, threshold changed to `v_d > c_s`
  - `anomalous_resistivity(J, ne, Te, ...)`: parameter renamed `Ti→Te`, uses `c_s` in loop
  - `anomalous_resistivity_scalar(...)`: `ion_acoustic` case now uses `Te_val` for threshold (`c_s`)
  - `anomalous_resistivity_field(...)`: `ion_acoustic` case now passes `Te` (not `Ti`) to threshold; raises `ValueError` if `Te is None`
  - `buneman_threshold(...)`: deprecated alias updated; accepts optional `Te`, falls back to `Ti` if not provided
  - `ion_sound_speed` exported from `__init__.py`
- **Impact**: Anomalous resistivity now triggers at the physically correct drift velocity for Te >> Ti DPF conditions. For Te ~ Ti (collisional equilibrium), behavior is nearly unchanged.

---

## VERIFIED CORRECT

### LHDI Threshold (lines 170-205)
- `v_d > (m_e/m_i)^{1/4} * v_ti`: CORRECT per Davidson & Gladd (1975)
- Factor `(m_e/m_i)^{1/4} ~ 0.129` for deuterium: CORRECT

### Buneman Classic Threshold (lines 208-233)
- `v_d > v_te`: CORRECT per Buneman (1959)

### Anomalous Resistivity Magnitude (lines 241-258)
- `eta_anom = alpha * m_e * omega_pe / (ne * e^2)`: CORRECT
- Equivalent to Sagdeev formula `eta = alpha / (omega_pe * epsilon_0)`: VERIFIED
- For alpha=0.05, omega_pe ~ 10^12: eta ~ 5.6e-6 Ohm*m (~1000x Spitzer at 1 keV): CORRECT order of magnitude

### Deprecation Handling
- `buneman_threshold()` correctly issues DeprecationWarning and redirects to `ion_acoustic_threshold`. Good software practice.

---

## REJECTED FINDINGS

None. The ion-acoustic threshold finding is a known approximation, not a strict rejection.
