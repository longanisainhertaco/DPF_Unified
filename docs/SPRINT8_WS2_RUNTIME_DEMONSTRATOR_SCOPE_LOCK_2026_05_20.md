# Sprint 8 WS2 — Runtime Demonstrator Scope Lock (2026-05-20)

## Status: GOVERNANCE — NOT SCIENTIFIC AUTHORITY

This document records the control-plane scope decision for the Sprint 8
runtime demonstrator.  It is governance documentation, not a KnowledgeReference
scientific claim and not a V&V certificate.  No entry here carries
`accepted_runtime_claim=true` or `is_scientific_authority=true`.

Typed encoding: `src/dpf/first_principles/runtime_demonstrator_scope.py`

---

## Decision

**Option B — PF-1000 full-energy campaign, 27–40 kV (Gribkov/Scholz 2007 era)**

Selected scope label (canonical string used by all downstream artifacts):

    pf1000_full_energy_27_to_40_kv

This label must appear identically in:
- the runtime deck preset
- source ledger scope fields for all in-scope sources
- the same-scope packet `declared_scope` argument
- the comparator decision scope field

---

## Basis

The decision was reached in the Sprint 4 scope-decision memo
(`docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md`, Option B
recommendation) and reaffirmed by every audit since.  Sprint 7 WS-B encoded
the geometry for this scope in `PF1000GeometryPacket.scholz_2001_24rod_large_electrode()`.

### Channel coverage comparison (from decision memo)

| Channel | Akel 16 kV | Full-energy 27–40 kV |
|---|---|---|
| I(t) | SUPPORTED | SUPPORTED |
| V(t) | ABSENT | SUPPORTED |
| Te | ABSENT | TEXT-ONLY |
| Ti | ABSENT | TEXT-ONLY |
| ne | ABSENT | SUPPORTED |
| X-ray | ABSENT | SUPPORTED |
| Yn | SUPPORTED | SUPPORTED |
| Neutron spectrum | ABSENT | SUPPORTED |
| Neutron anisotropy | ABSENT | SUPPORTED |

**Channels supported: Akel 16 kV = 2 of 9. Full-energy = 7 of 9.**

Te and Ti are absent corpus-wide for the DPF pinch phase from direct
spectroscopic measurement.  This is a structural gap in the field, not a
gap specific to this scope choice.

---

## Scope change note

This is a scope change from the prior Akel-16-kV runtime defaults.  The
V&V certificate must document this change explicitly and state that 16 kV
results are extrapolated from the 27–40 kV validated regime:

- I_peak ~2 MA (full-energy) vs ~1 MA (Akel 16 kV)
- Stored energy ~810 kJ (full-energy) vs ~170 kJ (Akel 16 kV)

---

## Source classification

### In-scope sources

Sources whose scope_tag or declared_scope is consistent with
`pf1000_full_energy_27_to_40_kv`.  These may be used for runtime candidate
wiring, target extraction, and comparator construction subject to KR
availability and all other Sprint 8 guardrails.

| source_id | KR path | Notes |
|---|---|---|
| scholz_gribkov_2007_partii | KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md | Primary I(t), V(t), Yn, anisotropy |
| gribkov_2007_pf1000_part2_existing_kr_equivalent | same KR | Sprint 6 extraction packet |
| scholz_2001_recent_progress_pf1000_hardware | KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md | 24-rod hardware dimensions |
| scholz_2000_pf1000_device | KnowledgeReference/pf-1000-device-a2d6bc15.md | Facility bank and chamber |
| scholz_gribkov_2007_part1 | KnowledgeReference/scholz-2006-pf1000-mega-joule.md | SXR/HXR/Yn waveforms |
| malir_2024_interferometry_dpf | KnowledgeReference/malir-2024-interferometry-dpf.md | 16-frame ne interferometry |
| klir_2011_tof_detector_pf1000 | KnowledgeReference/fusion-neutron-detector-for-tof...md | TOF detector calibration |
| krasa_2008_anisotropy_pf1000 | KnowledgeReference/anisotropy-of-the-emission-of-dd-fusion-neutrons...md | TLD/Bonner anisotropy |
| auluck_2021_plasma_focus_update | KnowledgeReference/... | Full-energy review context |
| stepniewski_2004_pf1000_mhd_modelling | sprint5 extraction | PF-1000 simulation context |

### Context-only sources

Sources that belong to the PF-1000 or DPF corpus but are cross-configuration,
reduced-model comparators, or methodology reviews.  Usable for requirements
and schema definition only; cannot close same-scope comparator channels.

| source_id | Reason |
|---|---|
| shakya_2015_pf1000_pf400_lee_model | Reduced Lee-model comparison; baseline/comparator evidence only |
| gribkov_malaquias_2006_dmp_applications | IAEA CRP applications review; no runtime blocker closure |
| scholz_1999_foam_liner_current_sheath | Modified PF-1000 with foam liner; not standard whole-shot geometry |
| herold_1989_poseidon_pf360_comparative | POSEIDON/PF-360 cross-machine; not PF-1000 same-scope |
| loarer_2007_tokamak_gas_balance_fuel_retention | Tokamak PWI methodology; not a DPF authority source |
| bruzzone_bernal_2001_lhi_duplicate_verification | LHI anomalous-resistivity context; quantitative closure still pending |

### Wrong-scope sources

Sources that are explicitly wrong-scope for `pf1000_full_energy_27_to_40_kv`.
Cross-scope use to close a selected-scope comparator channel requires a
reviewed transfer rule (Sprint 8 guardrail 7).

| source_id | Reason |
|---|---|
| talebitaher_2012_nx2_detector_anisotropy | NX2 (NTU Singapore, 3 kJ) — different device |
| bernard_1977_dpf_high_intensity_neutron_source | Historical Mather DPF review; wrong geometry class |
| ucsd_beg_current_sheath_initiation | UCSD 10 kJ Mather device; wrong device class |
| bennett_2017_kinetic_dpf_breakdown | scope_tag pf1000_generic; startup BVP not full-energy validated |
| akel_2021_pf1000_neutron_yield_16kv | PF-1000 Akel 16 kV — wrong voltage/energy scope |

---

## Guardrails enforced by this packet

1. `is_scientific_authority: false` — this document does not provide KR-backed
   physics claims.
2. `accepted_runtime_claim: false` — no runtime channel is accepted by this
   governance record.
3. Mixed-scope source sets (in-scope + wrong-scope, without a reviewed transfer
   rule) fail the `check_scope_consistency()` gate in the typed module.
4. Te and Ti remain absent as direct measurements; model-only estimates are
   TEXT-ONLY and cannot close same-scope comparator channels for those fields.

---

## Tests

`tests/test_runtime_demonstrator_scope.py` — enforces:
- Mixed-scope source packets fail `check_scope_consistency()`.
- `is_scientific_authority=False` and `accepted_runtime_claim=False`.
- `governance_class="control_plane"`.
- Canonical label matches `SELECTED_SCOPE_LABEL` constant.
- Known wrong-scope sources (NX2, Bernard, UCSD/Beg, Akel 16 kV) are
  classified as wrong-scope.
- Known in-scope sources (Scholz/Gribkov 2007, Scholz 2001, Malir 2024) are
  classified as in-scope.

---

## Exit criteria

- [x] One canonical scope label string defined: `pf1000_full_energy_27_to_40_kv`
- [x] Runtime deck preset may reference `SELECTED_SCOPE_LABEL` from
  `dpf.first_principles.runtime_demonstrator_scope`
- [x] Source ledger scope fields for in-scope sources should use this label
- [x] Same-scope packet `declared_scope` argument should be `pf1000_full_energy_27_to_40_kv`
- [x] Comparator decision scope field should match
- [x] Tests pass: mixed-scope fails, governance flags false, label consistent
