# Preset ↔ DEVICES Drift Audit — 2026-05-04

Read-only audit. Do not fix in this file — separate PRs handle each device.

Sources:
- `src/dpf/presets.py` (preset definitions)
- `src/dpf/validation/experimental_devices.py` (DEVICES registry)

Devices compared: those appearing in **both** files (by matching physical device).
Preset-only devices (tutorial, llnl_dpf, custom, cartesian_demo, phase_p_fidelity, poseidon_60kv,
aecs_pf2, pf400j) are skipped — no DEVICES counterpart with matching operating point.

---

## 1. PF-1000 (preset: `pf1000` ↔ DEVICES: `PF-1000`)

| Parameter | preset `pf1000` | DEVICES `PF-1000` | Drift? | Severity | KR-anchored side |
|-----------|----------------|-------------------|--------|----------|-----------------|
| C | 1.332e-3 F | 1.332e-3 F | None | — | Both (Scholz 2006) |
| V0 | 27e3 V | 27e3 V | None | — | Both |
| L0 | 33.5e-9 H | **25e-9 H** | YES — 34% delta | HIGH | DEVICES (Akel 2021 Table 1); preset uses Gribkov/RADPF L0=33.5 nH |
| R0 | 2.3e-3 Ohm | 2.3e-3 Ohm | None | — | Both (Scholz 2006) |
| anode_radius | 0.115 m | 0.115 m | None | — | Both |
| cathode_radius | 0.16 m | 0.16 m | None | — | Both |
| anode_length | 0.60 m | **0.48 m** | YES — 25% delta | MEDIUM | DEVICES (Akel 2021: z0=48 cm); preset uses Scholz 2006 600 mm |
| fill_pressure | 3.5 Torr (466 Pa) | 3.5 Torr | None | — | Both (Scholz 2006) |
| fm (mass_fraction) | 0.13 | 0.13 | None | — | Both (Malek 2025) |
| fc (current_fraction) | 0.7 | 0.7 | None | — | Both |
| fmr | 0.35 | 0.35 | None | — | Both |
| fcr | 0.45 (radial_current_fraction_2) | 0.65 | YES | HIGH | DEVICES (Malek 2025 [KR]); preset has two-step model with 0.45 post-re-strike |

**Notes:** The L0 split reflects a genuine literature ambiguity (Akel 2021 Table 1: L0=25 nH vs
Gribkov/RADPF: L0=33.5 nH for the same physical bank). The preset intentionally uses 33.5 nH with
comment "RADPF default". The anode_length split reflects Akel 2021 (48 cm) vs Scholz 2006 (60 cm) —
different papers, same machine. The fcr difference arises because the preset uses a two-step radial
model (fcr_1=0.45, fcr_2=0.5) while DEVICES stores a single Lee-fit fcr=0.65.

---

## 2. PF-1000 (preset: `pf1000_akel` ↔ DEVICES: `PF-1000-16kV`)

| Parameter | preset `pf1000_akel` | DEVICES `PF-1000-16kV` | Drift? | Severity | KR-anchored side |
|-----------|---------------------|------------------------|--------|----------|-----------------|
| C | 1.332e-3 F | 1.332e-3 F | None | — | Both |
| V0 | 16e3 V | 16e3 V | None | — | Both |
| L0 | 25e-9 H | 25e-9 H | None | — | Both (Akel 2021) |
| R0 | 2.3e-3 Ohm | 2.3e-3 Ohm | None | — | Both |
| anode_radius | 0.1155 m | 0.115 m | YES — 0.04% | LOW | DEVICES rounds; preset has 115.5 mm per Akel Table 1 |
| anode_length | 0.48 m | 0.48 m | None | — | Both (Akel 2021) |
| fill_pressure | (not set) | 1.05 Torr | YES — missing | MEDIUM | DEVICES (Akel 2021); preset omits fill_pressure_Pa key |
| fm | 0.20 | lee_fm=0.20 | None | — | Both |
| fc | 0.7 | lee_fc=0.70 | None | — | Both |
| fmr | 0.35 | lee_fmr=0.12 | YES — 3x delta | HIGH | DEVICES (Akel 2021: fmr=0.12); preset uses 0.35 from Malek 2025 27kV fit |
| fcr | (none) | lee_fcr=0.47 | YES — missing | HIGH | DEVICES (Akel 2021); preset has no fcr key |

---

## 3. NX2 (preset: `nx2` ↔ DEVICES: `NX2`)

| Parameter | preset `nx2` | DEVICES `NX2` | Drift? | Severity | KR-anchored side |
|-----------|-------------|---------------|--------|----------|-----------------|
| C | 28e-6 F | 28e-6 F | None | — | Both (Lee & Saw 2008) |
| V0 | 11.5e3 V | 11.5e3 V | None | — | Both |
| L0 | 20e-9 H | 20e-9 H | None | — | Both (RADPF Module 1) |
| R0 | 2.3e-3 Ohm | 2.3e-3 Ohm | None | — | Both |
| anode_radius | 0.019 m | 0.019 m | None | — | Both |
| cathode_radius | 0.041 m | 0.041 m | None | — | Both |
| anode_length | 0.05 m | 0.05 m | None | — | Both |
| fill_pressure | 400 Pa (3 Torr) | 3.0 Torr | None | — | Both |
| fm | **1.0** | lee_fm=**0.10** | YES — 10x delta | HIGH | DEVICES (Lee & Saw 2008 [KR]); preset has fm=1.0 labeled EMPIRICAL |
| fc | 0.7 | lee_fc=0.7 | None | — | Both |
| fmr | 0.14 | lee_fmr=0.12 | YES — 17% delta | HIGH | DEVICES (Akel 2021/RADPF); preset uses Arwinder thesis value 0.14 |
| fcr | 0.69 | lee_fcr=0.7 | YES — 1.4% | LOW | DEVICES (rounded); effectively same |

**Notes:** fm=1.0 in the preset is explicitly marked `# EMPIRICAL: calibrated to minimize loading vs
RADPF 400 kA target`. The published Lee-fit value per DEVICES is fm=0.10 (Lee & Saw 2008). This is
the single largest physics-affecting drift in the entire audit.

---

## 4. UNU-ICTP (preset: `unu_ictp` ↔ DEVICES: `UNU-ICTP`)

| Parameter | preset `unu_ictp` | DEVICES `UNU-ICTP` | Drift? | Severity | KR-anchored side |
|-----------|------------------|-------------------|--------|----------|-----------------|
| C | 30e-6 F | 30e-6 F | None | — | Both (Lee & Saw 2014 KR p.152) |
| V0 | **14e3 V** | **15e3 V** | YES — 7% delta | HIGH | DEVICES (KR p.152: V0=15 kV); preset uses 14 kV (prior code value) |
| L0 | 110e-9 H | 110e-9 H | None | — | Both |
| R0 | 12e-3 Ohm | 12e-3 Ohm | None | — | Both (UNVERIFIED in both) |
| anode_radius | 0.0095 m | 0.0095 m | None | — | Both |
| cathode_radius | 0.032 m | 0.032 m | None | — | Both |
| anode_length | 0.16 m | 0.16 m | None | — | Both |
| fill_pressure | 400 Pa (3 Torr) | **4.0 Torr** | YES — 33% delta | HIGH | DEVICES (KR p.152: P0=4 Torr); preset has 3 Torr |
| fm | 0.08 | lee_fm=0.08 | None | — | Both |
| fc | 0.7 | lee_fc=0.7 | None | — | Both |
| fmr | 0.16 | lee_fmr=0.16 | None | — | Both |
| fcr | (none) | lee_fcr=0.7 | YES — missing | HIGH | DEVICES (IPFS/Lee & Saw 2014) |

**Notes:** V0=14 kV vs 15 kV is the Wave-6 RCA drift. DEVICES adopted 15 kV as the canonical KR
value; the preset was not updated in the same commit. fill_pressure 3 Torr vs 4 Torr is a second
uncorrected KR drift in the preset.

---

## 5. MJOLNIR (preset: `mjolnir` ↔ DEVICES: `MJOLNIR`)

| Parameter | preset `mjolnir` | DEVICES `MJOLNIR` | Drift? | Severity | KR-anchored side |
|-----------|-----------------|-------------------|--------|----------|-----------------|
| C | **408e-6 F** | **204e-6 F** | YES — 2x delta | HIGH | DEVICES (Schmidt 2021 §III.A [KR]: lumped C=204 uF); preset doubles |
| V0 | **60e3 V** | **100e3 V** | YES — 1.67x delta | HIGH | DEVICES (Schmidt 2021: 100 kV erected); preset uses 60 kV Goyon 2025 |
| L0 | 67.4e-9 H | 67.4e-9 H | None | — | Both (Schmidt 2021) |
| R0 | **6.25e-3 Ohm** | **12.5e-3 Ohm** | YES — 2x delta | HIGH | DEVICES (Schmidt 2021 §III.A: R0=12.5 mOhm lumped); preset halves |
| anode_radius | **0.1143 m** | **0.076 m** | YES — 50% delta | HIGH | DEVICES (Schmidt 2021: 15.2 cm dia → a=0.076 m); preset uses Goyon 2025 228.6 mm OD |
| cathode_radius | **0.157 m** | **0.119 m** | YES — 32% delta | HIGH | DEVICES (Schmidt 2021: A-K gap 4.3 cm → cathode_r=0.119 m); preset uses 157 mm estimate |
| anode_length | 0.20 m | 0.20 m | None | — | Both (Schmidt 2021 midpoint) |
| fm | **1.0** | lee_fm=0.50 | YES — 2x delta | HIGH | DEVICES (Gemini synthesis, unverified); both values unanchored |
| fc | 0.70 | lee_fc=0.70 | None | — | Both |
| fmr | 0.1 | lee_fmr=0.10 | None | — | Both |
| fcr | (none) | lee_fcr=0.14 | YES — missing | HIGH | DEVICES |

**Notes:** The MJOLNIR split is the most severe. The preset (added from Goyon 2025, 2-MJ config)
uses a different operating point from DEVICES (Schmidt 2021, 1-MJ config). This is a real
device-configuration split: C=408 uF is the 2-MJ bank, C=204 uF is the 1-MJ bank.
R0 and anode_radius drifts follow from the same source split.

---

## 6. FAETON (preset: `faeton` ↔ DEVICES: `FAETON-I`)

| Parameter | preset `faeton` | DEVICES `FAETON-I` | Drift? | Severity | KR-anchored side |
|-----------|----------------|-------------------|--------|----------|-----------------|
| C | 25e-6 F | 25e-6 F | None | — | Both (Damideh 2025) |
| V0 | 100e3 V | 100e3 V | None | — | Both |
| L0 | 220e-9 H | 220e-9 H | None | — | Both (Damideh 2025) |
| R0 | 7.6e-3 Ohm | 7.6e-3 Ohm | None | — | Both |
| anode_radius | 0.05 m | 0.05 m | None | — | Both |
| cathode_radius | 0.106 m | 0.106 m | None | — | Both |
| anode_length | 0.17 m | 0.17 m | None | — | Both |
| fill_pressure | 1600 Pa (12 Torr) | 12.0 Torr | None | — | Both |
| fm | **0.70** | lee_fm=**0.70** | None | — | Both (Damideh 2025) |
| fc | 0.7 | lee_fc=0.70 | None | — | Both |
| fmr | 0.1 | lee_fmr=0.10 | None | — | Both |
| fcr (pre) | 0.8 | lee_fcr=**0.14** | YES — 5.7x delta | HIGH | DEVICES uses post-fit fcr=0.14; preset uses pre-re-strike fcr=0.8 |
| fcr (post) | 0.5 | — | N/A | — | Two-step model in preset, single-value in DEVICES |

**Notes:** FAETON is the cleanest match on circuit/geometry. The fcr delta is structural: the preset
implements two-step radial (Damideh 2025 re-strike model) with fcr_pre=0.8 and fcr_post=0.5, while
DEVICES stores only the Lee-fit scalar fcr=0.14. The Damideh 2025 paper (Lee co-author) is the
source for both, but the two-step vs single-step representation is inherently incompatible.

---

## 7. POSEIDON (preset: `poseidon` ↔ DEVICES: `POSEIDON` in _REFERENCE_ONLY)

Not compared — POSEIDON 40 kV is in `_REFERENCE_ONLY` (Herold 1989 not on disk). The `poseidon_60kv`
preset maps to `POSEIDON-60kV` in DEVICES and has no major circuit drifts (L0 17.7 vs 18 nH is
within KR rounding). Skipped from this table.

---

## Summary: Top 5 Fix Priorities

| Rank | Device | Parameter | Preset | DEVICES | Action |
|------|--------|-----------|--------|---------|--------|
| 1 | **NX2** | fm | 1.0 (EMPIRICAL) | 0.10 (Lee & Saw 2008) | Replace fm=1.0 with published 0.10 in preset; remove EMPIRICAL label |
| 2 | **MJOLNIR** | C / V0 / R0 / anode_radius | 2-MJ Goyon values | 1-MJ Schmidt 2021 values | Decide canonical config; split into `mjolnir_1mj` and `mjolnir_2mj` presets |
| 3 | **UNU-ICTP** | V0 | 14 kV | 15 kV (KR p.152) | Update preset V0 to 15 kV to match DEVICES |
| 4 | **UNU-ICTP** | fill_pressure | 3 Torr (400 Pa) | 4 Torr (KR p.152) | Update preset fill_pressure_Pa to 533 Pa (4 Torr) |
| 5 | **PF-1000-16kV** | fmr | 0.35 (27kV Malek fit) | 0.12 (Akel 2021) | Update pf1000_akel fmr to 0.12 and add fcr=0.47 |

---

## Devices with No Actionable Drift

- **PF-1000 (27kV)**: L0 and anode_length drifts are deliberate (different papers cited). The fcr
  discrepancy is structural (two-step model). No fix needed — document intent.
- **FAETON-I**: Circuit/geometry aligned. fcr two-step vs scalar is structural.
- **POSEIDON-60kV**: Sub-rounding delta only (L0 17.7 vs 18 nH). No fix needed.

---

Generated: 2026-05-04 | Source read: presets.py (734 lines), experimental_devices.py (617 lines)
