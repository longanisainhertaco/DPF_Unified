# RADPF Reference Data Regeneration Playbook

**Owner:** Anthony Zamora (PI). Cortana cannot generate RADPF reference data —
this is anchored to Anthony's running of the canonical RADPF v5.16 spreadsheet
or its Python port with paper-cited inputs. See
`feedback/papers-are-truth.md` and `feedback/published-parameters-are-inputs-not-knobs.md`.

**Date opened:** 2026-04-30
**Trigger:** `tests/test_mhd_acceptance.py::test_angle1_ipeak` xfail'd pending regen.

---

## 1. Why regen is needed

The current reference JSON `tests/reference_data/radpf_pf1000_27kv.json`
was generated 2026-04-09 (commit `91f4e8b`) with **stale** Lee fit parameters:

| Param | JSON (stale) | Production (Malek 2025) | Source |
|-------|--------------|-------------------------|--------|
| `fc`  | 0.70         | 0.70                    | unchanged |
| `fm`  | 0.13         | 0.13                    | unchanged |
| `fmr` | 0.35         | 0.35                    | unchanged |
| `fcr` | **0.70**     | **0.65**                | `[KR: plasma-physics-and-technology-1211-9-2025.md §3 lines 177-180]` |

Production code switched to `fcr=0.65` in commit `b08c615` (the Malek 2025
re-anchor). The 0.05 drift produces:

- Simulator I_peak: **2.013 MA** (production, papers-are-truth)
- JSON I_peak:     **1.818 MA** (stale RADPF baseline)
- Drift:           **+10.7%** in I_peak, ~+4.0% in t_peak

This **cannot** be fixed by retuning the simulator. Per `papers-are-truth.md`,
the published Lee fits are inputs, not knobs. The JSON must be regenerated to
match the same KR-canonical inputs the simulator uses today.

---

## 2. Stale JSON inventory

`find tests/reference_data -name "*.json"` returned **1 file**:

| File | Last commit | Drifted params | Action |
|------|-------------|----------------|--------|
| `tests/reference_data/radpf_pf1000_27kv.json` | `91f4e8b` (2026-04-09) | `fcr: 0.70 → 0.65` | **Regenerate** |

No other reference JSONs contain Lee fits or bank parameters. This is the
sole regeneration target.

---

## 3. Regeneration procedure (Anthony's perspective)

### 3.1 Inputs to use (verbatim, KR-canonical)

**Lee fit parameters** (from Malek 2025 PPT 12(1):9 §3 lines 177-180):

```
fc  = 0.70
fm  = 0.13
fmr = 0.35
fcr = 0.65   <-- changed from 0.70
```

**PF-1000 device parameters** (from Akel 2021, Scholz 2006):

```
L0_nH         = 33.5    (bank inductance, RADPF input — NOTE: simulator
                         uses L0=25 nH per Akel 2021 Table 1; the 33.5 nH
                         in current JSON came from Lee & Saw 2014 fit and
                         should be reviewed against Akel 2021. If the
                         RADPF spreadsheet expects the lumped value
                         including transmission line, retain 33.5 nH;
                         otherwise switch to 25 nH.)
C0_uF         = 1332.0  (1.332 mF per Akel 2021)
V0_kV         = 27.0
R0_mOhm       = 6.12    (RADPF input — historically the lumped value;
                         simulator uses 2.3 mOhm bare-bank with plasma R
                         via sheath model. Use the value the RADPF
                         spreadsheet expects, NOT the simulator value.)
b_cm          = 16.0    (cathode radius)
a_cm          = 11.5    (anode radius)
z0_cm        = 60.0    (anode length — RADPF historical input;
                         Akel 2021 says 48 cm. Confirm which length the
                         RADPF spreadsheet was originally calibrated to.)
gas           = D2
pressure_Torr = 3.5
```

> **Anthony decision points** (params marked above):
> Resolve `L0_nH`, `R0_mOhm`, `z0_cm` against the RADPF spreadsheet's
> historical input convention. If RADPF expects lumped values, keep the
> stale JSON values. If RADPF expects bare-bank values, switch to
> `L0=25 nH, R0=2.3 mOhm, z0=48 cm`. Whichever path is taken must be
> documented in the new JSON's `parameters.notes` field.

### 3.2 Run RADPF v5.16

Execute the canonical RADPF v5.16 calculation (spreadsheet or Python port)
with the inputs above. Capture:

- `I_peak_A`, `t_peak_s`, `Lp_max_nH` (scalars)
- Full `I(t)` waveform on the same time grid as the current JSON
  (uniform spacing, ~7000 points spanning 0 → ~60 µs)

### 3.3 Match the JSON schema

The replacement file must preserve the existing schema:

```json
{
  "device": "PF-1000",
  "source": "RADPF v5.16 with Malek 2025 KR-canonical Lee fits",
  "reference": "Malek et al., Plasma Physics and Technology 12(1):9 (2025)",
  "parameters": { L0_nH, C0_uF, V0_kV, R0_mOhm, b_cm, a_cm,
                  z0_cm, fc, fm, fmr, fcr, gas, pressure_Torr,
                  notes },
  "scalars":    { I_peak_A, I_peak_MA, t_peak_s, t_peak_us, Lp_max_nH },
  "acceptance_criteria": {
                  I_peak_tolerance: 0.10,
                  t_peak_tolerance: 0.15,
                  waveform_L2_tolerance: 0.20,
                  dI_dt_rise_tolerance: 0.25 },
  "time_series": { t_us: [...], I_kA: [...] }
}
```

Do not add or remove top-level keys. The 5-angle test reads each by name.

### 3.4 Save

Overwrite `tests/reference_data/radpf_pf1000_27kv.json`. Commit with:

```
test: regenerate RADPF PF-1000 27kV reference with Malek 2025 fcr=0.65

Stale JSON used fcr=0.70 (commit 91f4e8b, 2026-04-09). Production switched
to Malek 2025 fcr=0.65 in b08c615. This commit re-runs RADPF v5.16 against
the KR-canonical fit set so test_angle1_ipeak gates on truth, not drift.

[KR: plasma-physics-and-technology-1211-9-2025.md §3 lines 177-180]
```

---

## 4. Post-regen verification

```bash
# 1. Remove the xfail marker
$EDITOR tests/test_mhd_acceptance.py
# Delete the @pytest.mark.xfail line above test_angle1_ipeak

# 2. Run the 5-angle gate
python3 -m pytest tests/test_mhd_acceptance.py -v --no-header

# Expected: all 5 angles PASS, including test_angle1_ipeak
# I_peak should be ~2.01 MA in both simulator AND JSON,
# differing by <10% (within I_peak_tolerance).
```

If `test_angle1_ipeak` still fails after regen, the simulator and RADPF
are diverging on physics other than Lee fits — escalate to
`dpf-mhd-physicist`.

---

## 5. References

- `feedback/papers-are-truth.md` — top rule
- `feedback/published-parameters-are-inputs-not-knobs.md`
- `CRITICAL_BLOCKER.md` — current PF-1000 accuracy state (+7.6% I_peak budget)
- `[KR: plasma-physics-and-technology-1211-9-2025.md §3]` — Malek 2025 fits
- `src/dpf/validation/experimental_devices.py:32-58` — production PF-1000 record
