# D1 PF-1000 Parameter Unification — Design Proposal

**Status:** Design only. No implementation. No commit.
**Date:** 2026-04-30
**Author:** dpf-engine-architect

---

## 1. Storage Sites Inventory

| Site | File:line | Format | Primary consumer |
|------|-----------|--------|------------------|
| **A** `PF1000_DATA` | `validation/experimental_devices.py:31` | `ExperimentalDevice` dataclass | calibration suite, Lee model comparison |
| **B** `pf1000` preset | `presets.py:65` | raw dict | SimConfig factory (CLI + engine) |
| **C** `run_rlc_snowplow_pf1000()` | `validation/engine_validation.py:83` | inline locals | acceptance harness |
| **D** `PF1000` | `validation/suite.py:79` | `DeviceData` dataclass | validation scoring |
| **E** `PF1000_MALIR` | `validation/malir_2024_data.py:52` | dict | Malek 2025 variant (27 kV/3.5 Torr) |

Sites A, B, C are the three primary paths named in D1. Sites D and E are secondary
consumers that also hold (partially) independent numbers.

---

## 2. Discrepancy Table (27 kV / 3.5 Torr canonical operating point)

| Parameter | **A** `exp_devices.py` | **B** `presets.py:pf1000` | **C** `engine_validation.py` | **D** `suite.py` | Truth / Comment |
|-----------|------------------------|---------------------------|------------------------------|-----------------|-----------------|
| `L0` (nH) | 25 | **33.5** | 25 | **33.5** | Akel 2021 Table 1: 25 nH. 33.5 nH is RADPF default, not the device value. **CRITICAL** — affects quarter-period and I_peak. |
| `R0` (mΩ) | 2.3 | 2.3 | 2.3 | 2.3 | Consistent. Scholz 2006 Table 1. |
| `C` (mF) | 1.332 | 1.332 | 1.332 | 1.332 | Consistent. |
| `anode_length` (m) | 0.48 | **0.60** | 0.48 | — | Akel 2021 p.1 / Table 1: 480 mm. presets.py `pf1000` uses 600 mm with wrong citation ("Scholz 2006 Table 1: 600 mm" — Scholz lists 480 mm). **CRITICAL** — changes snowplow inductance profile. |
| `anode_radius` (m) | 0.115 | 0.115 | 0.115 | 0.115 | Consistent. |
| `fm` | 0.13 | 0.13 | 0.13 (default) | — | Consistent on canonical path. Malek 2025. |
| `fc` | 0.70 | 0.70 | 0.70 (default) | — | Consistent on canonical path. |
| `fmr` | 0.35 | 0.35 | 0.35 (default) | — | Consistent. |
| `fcr` | 0.65 | — (missing) | — (missing) | — | **MISSING** from presets.py and engine_validation.py. ExperimentalDevice has it; no other site does. |
| `crowbar_resistance` | 1.5e-3 | 1.5e-3 | — (not passed) | — | engine_validation.py uses `crowbar_mode="voltage_zero"` without specifying arc resistance; falls to solver default (unknown). |
| `peak_current_time` (µs) | 5.8 | — | — | **5.5** | suite.py differs from experimental_devices.py by 0.3 µs. |

### Summary of critical divergences

1. **`L0` = 25 vs 33.5 nH** — `presets.py:pf1000` and `suite.py:PF1000` both use 33.5 nH.
   The `pf1000_akel` and `pf1000_20kv` presets have 25 nH and 33.5 nH respectively,
   meaning even within presets.py the same device has two values. The quarter-period
   `T/4 = pi/2 * sqrt(L0*C)` shifts by ~15% (10.27 µs vs 9.11 µs), which directly
   drives I_peak timing error.

2. **`anode_length` = 0.48 vs 0.60 m** — `presets.py:pf1000` uses 0.60 m with an
   incorrect attribution ("Scholz 2006 Table 1: 600 mm"). Akel 2021 Table 1 and p.1
   both state 480 mm. The snowplow inductance `L_plasma ~ ln(b/a) * z` scales linearly
   with `z_max`, so a 25% longer anode produces a 25% larger plasma inductance at rundown
   end, depressing I at pinch. This is a physically significant deviation, not a rounding
   difference.

3. **Missing `fcr`** in presets.py and engine_validation.py — The radial-phase current
   fraction is defined only in `PF1000_DATA`. Any path that constructs SnowplowModel
   from presets or engine_validation inlines defaults silently.

---

## 3. Single-Source-of-Truth Proposal

**Promote `PF1000_DATA` (`experimental_devices.py:31`) as the canonical authority.**

Rationale: it already carries the strongest schema — waveform arrays, uncertainty
budget, provenance strings, and `lee_reference` citation. The other sites are subsets
of it.

### Target schema — `pf1000_device.yaml` (or keep as `ExperimentalDevice` dataclass)

Recommendation: **keep as `ExperimentalDevice` dataclass** (not YAML). The dataclass
already exists, is typed, is importable, and survives refactors better than a file
loaded at runtime. Introduce a YAML only if the device registry needs to be editable
without a code deploy (no current requirement).

Required corrections before promoting to SSoT:

```python
# experimental_devices.py:31 — changes needed
PF1000_DATA = ExperimentalDevice(
    ...
    inductance=25e-9,        # KEEP (already correct)
    anode_length=0.48,       # KEEP (already correct)
    lee_fcr=0.65,            # KEEP (already present)
    # Add: crowbar_resistance is already present at line 54
)
```

`PF1000_DATA` needs NO changes. It is already the most correct copy.

### Consumer rewrites

**`presets.py:pf1000`**
- Replace the `circuit.L0 = 33.5e-9` line with `circuit.L0 = PF1000_DATA.inductance`.
- Replace `snowplow.anode_length = 0.6` with `PF1000_DATA.anode_length`.
- Replace `snowplow.current_fraction = 0.7` with `PF1000_DATA.lee_fc`.
- Replace `snowplow.mass_fraction = 0.13` with `PF1000_DATA.lee_fm`.
- Replace `snowplow.radial_mass_fraction = 0.35` with `PF1000_DATA.lee_fmr`.
- Add `radial_current_fraction = PF1000_DATA.lee_fcr` (currently absent).
- The preset dict becomes a thin `SimConfig` builder around `PF1000_DATA`; no numbers live in it.

**`engine_validation.py:run_rlc_snowplow_pf1000()`**
- Change signature to accept `device: ExperimentalDevice = PF1000_DATA` (already
  documented in D1 recommendation).
- Delete the 8 inline local assignments (`C`, `V0`, `L0`, `R0`, `a`, `b`, `z_max`,
  `p_torr`) and replace with `device.*`.
- Default kwargs `fc`, `fm`, `f_mr` become `device.lee_fc`, `device.lee_fm`,
  `device.lee_fmr`. Callers that need override can still pass them.

**`suite.py:PF1000`**
- `L0=33.5e-9` → `PF1000_DATA.inductance` (25 nH).
- `peak_current_time_s=5.5e-6` → `PF1000_DATA.current_rise_time` (5.8 µs).
- Long-term: consolidate `DeviceData` into `ExperimentalDevice` (separate debt item).

---

## 4. Migration Plan

| Step | File | Change | Risk |
|------|------|--------|------|
| 1 | `validation/experimental_devices.py` | Verify `PF1000_DATA` is already correct — read Akel 2021 and Scholz 2006 PDFs, confirm all values. Add `# SSoT:D1` comment at line 31. | Low |
| 2 | `validation/engine_validation.py` | Refactor `run_rlc_snowplow_pf1000()` to accept `device` arg; derive locals from `device.*`. | Low — purely mechanical. |
| 3 | `presets.py:pf1000` | Replace inline numbers with references to `PF1000_DATA`. Keep `pf1000_akel` as a separate operating-point variant (different `V0`, `rho0`). | Medium — must not break CLI tests. |
| 4 | `validation/suite.py:PF1000` | Fix `L0` and `peak_current_time_s`. | Low — test tolerance is 15%. |
| 5 | `tests/` | Add equality assertion test: load `PF1000_DATA`, build preset, call `run_rlc_snowplow_pf1000(device=PF1000_DATA)`, assert circuit params match. | Low — new test. |
| 6 | `tests/baselines/pf1000_peak_current.json` | Replace with `PF1000_DATA.peak_current` value (D6 cross-dependency). | Medium — requires D6 audit. |

---

## 5. Effort Estimate

| PR | Scope | Estimate |
|----|-------|----------|
| PR-D1-A | Steps 1–2 (engine_validation refactor) | 2–3 hours |
| PR-D1-B | Step 3 (presets.py rewrite) | 3–4 hours |
| PR-D1-C | Steps 4–5 (suite.py fix + SSoT equality test) | 2 hours |

**Total: ~7–9 hours across 3 PRs.** Run full test suite between each PR. Do NOT batch.

---

## 6. Blockers

1. **PDF-on-disk verification required before Step 1.** Before marking `PF1000_DATA` as
   SSoT, physically read `references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf`
   Table 1 and `references/papers/core-dpf/scholz-2006-pf1000-mega-joule.pdf` Table 1 to
   confirm `L0=25 nH` and `anode_length=480 mm`. Per CLAUDE.md: no "MATCH" without PDF open.

2. **D6 baseline audit must run concurrently with PR-D1-C.** `pf1000_peak_current.json`
   is a 2-line self-baseline of unknown provenance. PR-D1-C cannot close until D6
   determines whether that file is replaced or deleted.

3. **`presets.py:pf1000` anode_length citation is wrong.** The comment
   `"Scholz (2006) Table 1: 600 mm"` is a factual error (Scholz 2006 Table 1 states
   480 mm, not 600 mm). This needs paper-on-disk correction as part of PR-D1-B,
   not a comment fix. If 600 mm is intentional for a different reason, that reason must
   be documented with a paper citation before PR-D1-B can merge.

4. **`PF1000_GRIBKOV_DATA` uses `inductance=33.5e-9`** (`experimental_devices.py:232`).
   This is the same IPPLM bank but a different data source (Malek 2025 cites 33.5 nH;
   Akel 2021 cites 25 nH for the same device). The discrepancy between these two papers
   must be resolved against the PDFs before the SSoT can claim a single `L0` for the
   device. This may result in `PF1000_DATA` and `PF1000_GRIBKOV_DATA` legitimately
   having different `L0` values (different measurement campaigns), which is acceptable
   if documented.
