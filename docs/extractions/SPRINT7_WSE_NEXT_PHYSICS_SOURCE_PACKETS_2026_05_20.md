# Sprint 7 WS-E — Next Physics Source Packets Index (2026-05-20)

Sprint: Sprint 7 WS-E.
Controlling doc: `docs/SPRINT7_FIRST_PRINCIPLES_RUNTIME_CONTRACT_INSTRUCTIONS_2026_05_20.md` §WS-E.
Author: WS-E research agent (read-only; no runtime code modified).

## Runtime Wiring Posture (non-negotiable)

**NO runtime coefficients are wired in Sprint 7.** This index contains
source-availability and verification records only. A coefficient may not
move to `implemented-candidate` until all of the following gates pass at
a single reviewed commit: target extraction complete, code consumption
implemented, unit tests passing, numerical tests passing, same-scope impact
classification documented, and independent Codex + external-team convergence
on exact values.

---

## Packet 1 — Braginskii 1965 Table 2 (Re-Audit)

### Source identity

| field | value |
| --- | --- |
| Citation | Braginskii, S. I. (1965). "Transport processes in a plasma," in M. A. Leontovich (ed.), *Reviews of Plasma Physics, Vol. 1*, Consultants Bureau, New York, pp. 205-311. |
| PDF on disk | `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf` |
| PDF SHA-256 | `9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404` |
| PDF size | 5,089,370 bytes (4.85 MiB) |
| PDF total pages | 56 (2-up scanned; each PDF page holds two journal pages) |
| Render image | `docs/extractions/braginskii_1965_render_evidence/pdf_p026_journal_p250_p251.png` |
| Render image SHA-256 | `c914283871fbf6f18a68b1ee91f08f051c7cb5d9e9feb9acceaa79c0606cb726` |
| Render manifest | `docs/extractions/braginskii_1965_render_evidence/render_manifest.json` |

### Page/line range

- Table 2 (Z-dependent transport coefficients): journal p. 251, right half of
  PDF p. 26 (1-indexed) 2-up spread.
- Equations 4.30–4.45 (closure relations): journal pp. 249–253, PDF pp. 25–28.

### Re-audit — SHA-256 verification

Re-audit performed by computing `shasum -a 256` on both files this session:

- PDF SHA-256 computed: `9687440676b43b02758373ef83d5a471730e9ca4bc207cb98f248fcafcb9d404`
  — **matches Sprint 6 extraction doc exactly. CONFIRMED.**
- Render image SHA-256 computed: `c914283871fbf6f18a68b1ee91f08f051c7cb5d9e9feb9acceaa79c0606cb726`
  — **matches render manifest exactly. CONFIRMED.**

### Re-audit — Z=1 column values

The Sprint 6 extraction (`BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md`
§4) records the following bolded (two-pass spot-confirmed) Z=1 values:

| coefficient | Z=1 (Sprint 6) | re-audit status |
| --- | --- | --- |
| α₀ | 0.5129 | confirmed — matches manifest and Sprint 5 spot-check |
| β₀ | 0.7110 | confirmed |
| γ₀ | 3.1616 | confirmed |
| δ₀ | 3.7703 | confirmed |
| δ₁ | 14.79 | confirmed |
| α₁' | 6.416 | confirmed |
| α₀' | 1.837 | confirmed |
| α₁'' | 1.704 | confirmed |
| α₀'' | 0.7796 | confirmed |
| β₁' | 5.101 | confirmed |
| β₀' | 2.681 | confirmed |
| β₁'' | 3/2 | confirmed |
| β̃₀'' | 3.053 | confirmed |
| γ₁' | 4.664 | confirmed |
| γ₀' | 11.92 | confirmed |
| γ₁'' | 5/2 | confirmed |
| γ₀'' | 21.67 | confirmed |

Cells flagged `(review-required)` in Sprint 6 (α₀' Z=3/4/∞, α₀'' Z=3
anomalous, γ₁' Z=∞) carry forward as unresolved; they are NOT confirmed
by this re-audit and remain blocked from code consumption. See Sprint 6
extraction §4 for the full flag list.

### Re-audit verdict

**Re-audit confirms Sprint 6 extraction.** Both SHA-256 hashes match. All
17 Z=1 bolded values are consistent across the render manifest, Sprint 5
spot-check records, and Sprint 6 extraction table. No discrepancies found.
The 5 `(review-required)` cells carry forward as open items requiring
second-reader resolution before code consumption.

### Units

All Table 2 entries are dimensionless numerical coefficients (pure numbers).
Units enter through the full transport expressions Eqs. 4.34–4.45 (closure
relations bind the table to physical quantities via `x = ω_e τ_e`,
`Δ = x⁴ + δ₁ x² + δ₀`, etc.).

### Symbol map

See Sprint 6 extraction §4 and §5 for the full symbol map. Key Z=1 entries:

- `α₀`, `β₀`, `γ₀`: parallel transport coefficients (friction, thermal
  force, electron heat flux) via `δ₀` denominator.
- `δ₀`, `δ₁`: Z-dependent denominator terms.
- `α₁'`, `α₀'`: perpendicular-component coefficients for friction force.
- `α₁''`, `α₀''`: Hall-component coefficients.
- `β₁'`, `β₀'`, `β₁''`, `β̃₀''`: thermal-force coefficients.
- `γ₁'`, `γ₀'`, `γ₁''`, `γ₀''`: electron heat-flux coefficients.

### Scope tag

`generic_formulary` — Z-dependent transport coefficients for fully-ionized
plasmas. Z=1 column applies to the DPF deuterium plasma case.

### Intended code consumer

`src/dpf/first_principles/closure_packet.py::electrical_thermal_transport`

### Required tests (before any coefficient may be wired)

1. Unit test: load the Z=1 row from the target-extracted KR record and assert
   each value matches the Sprint 6 extraction to the stated precision.
2. Numerical test: exercise the Z=1 resistivity and electron heat-conductivity
   formulas (Eqs. 4.34–4.38) through `closure_packet.py` and verify output
   units are consistent (Ω·m, W·m⁻¹·K⁻¹).
3. Same-scope comparator test: run a reduced-model comparator (Lee/snowplow,
   comparator-baseline only) alongside the Braginskii closure and confirm the
   closure output does not diverge non-physically.
4. Second-reader confirmation: the 5 `(review-required)` cells must be
   confirmed by an independent reader before any code path consuming those
   cells can be promoted past `target_extracted`.
5. PlasmaPy cross-check (second-witness): run
   `scripts/compare_plasmapy_braginskii_table2.py` (to be written) at a
   pinned PlasmaPy version against all 17 Z=1 cells; output pass/discrepancy
   per cell. This is a cross-check lane only, not a source-equivalent
   substitute.

### Remaining blockers

- `CLOSURE-BLK-BRAG-001`: status `target_extracted_source_supported_pending_
  runtime_consumption_and_review`. Blocks on: code consumption in
  `closure_packet.py`, numerical-fidelity test, same-scope comparator,
  certificate gate.
- 5 `(review-required)` cells: α₀' (Z=3, 4, ∞), α₀'' (Z=3 anomalous),
  γ₁' (Z=∞) — require second-reader confirmation before any code path
  consuming those cells may proceed.

### Acceptance flags

```python
accepted_runtime_claim = False
can_support_first_principles_acceptance = False
runtime_coefficients_wired_sprint_7 = False
```

---

## Packet 2 — Bennett 2017 Startup Channels (Line/Page Verification)

### Source identity

| field | value |
| --- | --- |
| Actual citation | Bennett, N. et al. (2017). "Kinetic simulations of gas breakdown in the dense plasma focus," *Phys. Plasmas* 24, 062705. DOI: 10.1063/1.4985313 |
| On-disk PDF | `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` |
| Filename mislabel | Filename says "schmidt-2017"; actual first author is N. Bennett |
| PDF SHA-256 | `c5e6f5f1e2ca150a41c18c83f82a2fbaf35a2deb75d4a50b60cb7a45b0f0b92a` |
| PDF size | ~2.4 MiB (6 pages: 062705-1 through 062705-6) |
| Journal | *Physics of Plasmas* 24, 062705 (2017). Published 9 June 2017. |

SHA-256 computed this session: `shasum -a 256 archive_reference_OLD/references/
papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf`.

### Page/line range

All target values are on journal pages 062705-2 through 062705-5 (PDF pages
3–6, where page 1 is the AIP abstract/landing page and page 2 is journal
page 062705-1).

### Line/page verification of Sprint 5 verbatim quotes

Each target from `src/dpf/first_principles/sprint5_target_extractions.py::
BENNETT_2017_STARTUP_EXTRACTION` verified against the PDF pages read this
session:

#### T1 — Seed plasma density (CH03)

- Sprint 5 claim: `value=1e7 cm⁻³`, `source_page="p.2 (062705-2)"`.
- Sprint 5 verbatim: "The DPF volume is also initialized with a 10^7-cm^-3
  density plasma of deuterium ions and electrons to provide seed electrons
  for the avalanche ionization process."
- PDF page 062705-2 (read this session): text reads "The DPF volume is also
  initialized with a 10⁷-cm⁻³ density plasma of deuterium ions and electrons
  to provide seed electrons for the avalanche ionization process."
- **CONFIRMED.** Verbatim match. Page confirmed.

#### T2 — Fill pressure / neutral density (corroborative, not CH01 closer)

- Sprint 5 claim: `value=5.5 Torr`, neutral density 3.52×10¹⁷ cm⁻³,
  `source_page="p.2 (062705-2)"`, flagged `corroborative_only=True`.
- PDF page 062705-2: "The volume in Fig. 3 is filled with neutral deuterium
  atoms with an initial density of 3.52 × 10¹⁷ cm⁻³ (5.5 Torr of molecular
  deuterium, D₂)."
- **CONFIRMED.** Verbatim match. Corroborative-only status confirmed.

#### T3 — Breakdown delay ~20 ns (CH04)

- Sprint 5 claim: `value=20.0 ns`, `source_page="p.4 (062705-4)"`.
- Sprint 5 verbatim: "The measured breakdown time (the time between the rise
  of voltage and the rapid rise in current) is approximately 20 ns."
- PDF page 062705-4 (read this session): "The measured breakdown time (the
  time between the rise of voltage and the rapid rise in current) is
  approximately 20 ns."
- **CONFIRMED.** Verbatim match. Page confirmed.

#### T4 — Explosive emission threshold bulk 250 kV/cm (CH07)

- Sprint 5 claim: `value=250.0 kV/cm`, `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "We use an electric field stress threshold of 250 kV/cm
  except for the cathode knife-edge, where the threshold is reduced to
  10 kV/cm to approximate the field enhancement of its 3D structures in our
  2D model."
- PDF page 062705-3: "We use an electric field stress threshold of 250 kV/cm
  except for the cathode knife-edge, where the threshold is reduced to
  10 kV/cm to approximate the field enhancement of its 3D structures in our
  2D model."
- **CONFIRMED.** Verbatim match. Both thresholds (250 kV/cm and 10 kV/cm)
  confirmed on the same sentence on p.3.

#### T5 — Explosive emission threshold knife-edge 10 kV/cm (CH07)

- Sprint 5 claim: `value=10.0 kV/cm`, `source_page="p.3 (062705-3)"`.
- **CONFIRMED** (same sentence as T4 above). Verbatim match.

#### T6 — Electron temperature at breakdown ~3.5–4 eV (CH07)

- Sprint 5 claim: `value_eV=4.0`, `range_eV=(3.5, 4.0)`,
  `source_page="p.5 (062705-5)"`.
- Sprint 5 verbatim: "the mean local temperatures (T_e) in the electron
  distributions from simulation remain near 4 eV, well into breakdown, as
  shown in Fig. 7."
- PDF page 062705-5: "The mean local temperatures (T_e) in the electron
  distributions from simulation remain near 4 eV, well into breakdown, as
  shown in Fig. 7." Fig. 8 on the same page plots "3.5-eV Maxwellian" and
  "4-eV Maxwellian" curves, confirming the (3.5, 4.0) range.
- **CONFIRMED.** Verbatim match. Range (3.5–4 eV) confirmed by Fig. 8
  caption on p.5.

#### T7 — 71% sheath current at 1 µs (CH08)

- Sprint 5 claim: `value_percent=71.0`, `at_time_us=1.0`,
  `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "by 1 us [Fig. 4(c)], it is carrying 71% of the
  current."
- PDF page 062705-3: "and by 1 µs [Fig. 4(c)], it is carrying 71% of the
  current."
- **CONFIRMED.** Verbatim match. The Sprint 5 audit-row-7 correction
  (71% at 1 µs, not 500 ns) is confirmed: p.3 unambiguously attributes
  71% current fraction to t = 1 µs (Fig. 4(c)), not to t = 500 ns
  (Fig. 4(b), which only shows channel formation).

#### T8 — Bulk ionization 10¹³ cm⁻³ at 100 ns (CH08)

- Sprint 5 claim: `value=1e13 cm⁻³`, `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "By 100 ns, as the plasma sheath is forming, a bulk
  ionization of order 10¹³ cm⁻³ has already occurred in the volume."
- PDF page 062705-3: "By 100 ns, as the plasma sheath is forming, a bulk
  ionization of order 10¹³ cm⁻³ has already occurred in the volume."
- **CONFIRMED.**

#### T9 — Plasma channel 10¹⁵ cm⁻³ at 500 ns (CH08)

- Sprint 5 claim: `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "By 500 ns [Fig. 4(b)], a plasma channel has formed
  across the coaxial electrode gap."
- PDF page 062705-3: "By 500 ns [Fig. 4(b)], a plasma channel has formed
  across the coaxial electrode gap and, by 1 µs [Fig. 4(c)], it is
  carrying 71% of the current."
- **CONFIRMED.**

#### T10 — Plasma 10¹⁵ cm⁻³ along insulator at 400 ns (CH08)

- Sprint 5 claim: `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "By 400 ns into the pulse a plasma of 10¹⁵ cm⁻³
  density has formed along the insulator with the aid of the cathode
  knife-edge."
- PDF page 062705-3: "By 400 ns into the pulse a plasma of 10¹⁵ cm⁻³
  density has formed along the insulator with the aid of the cathode
  knife-edge."
- **CONFIRMED.**

#### T11 — Photoionization 1.2% by 125 ns (CH04/CH06)

- Sprint 5 claim: `source_page="p.3 (062705-3)"`.
- Sprint 5 verbatim: "Preliminary simulations run with the addition of
  photoionization showed a 1.2% increase in electron density by 125 ns...
  Photons are, therefore, neglected here."
- PDF page 062705-3: "Preliminary simulations run with the addition of
  photoionization showed a 1.2% increase in electron density by 125 ns.
  This results from a relatively low photon population (below 10¹⁰ cm⁻³
  from excited deuterium) and a photoionization cross section that is an
  order of magnitude smaller than the electron impact ionization cross
  section. Photons are, therefore, neglected here."
- **CONFIRMED.** (Sprint 5 verbatim uses ellipsis correctly; full PDF text
  is confirmed.)

#### T12 — Pressure regime low: volumetric uniform breakdown (CH04)

- Sprint 5 claim: `source_page="p.5 (062705-5)"`.
- Sprint 5 verbatim: "At low pressures, the electron ionization path length
  exceeds 20 cm so electrons traveling axially are more likely to ionize
  the gas leading to bulk breakdown in the DPF volume."
- PDF page 062705-5: "At low pressures, the electron ionization path length
  exceeds 20 cm so electrons traveling axially are more likely to ionize
  the gas leading to bulk breakdown in the DPF volume [Fig. 2(a)]."
- **CONFIRMED.** (Sprint 5 verbatim omits "[Fig. 2(a)]" reference tag;
  substantive content confirmed verbatim.)

#### T13 — Pressure regime medium: surface ionization (CH04)

- Sprint 5 claim: `source_page="p.5 (062705-5)"`.
- Sprint 5 verbatim: "In an intermediate pressure range, the ionization path
  length may exceed the coaxial gap but approach the length of the insulator,
  which is longer than the gap in typical DPF designs."
- PDF page 062705-5: "In an intermediate pressure range, the ionization path
  length may exceed the coaxial gap but approach the length of the insulator,
  which is longer than the gap in typical DPF designs."
- **CONFIRMED.**

#### T14 — Pressure regime high: radial filamentation above 15 Torr (CH04)

- Sprint 5 claim: `source_page="p.5 (062705-5)"`.
- Sprint 5 verbatim: "At pressures above 15 Torr, electron impact ionization
  occurs within a few cms, so the gas may breakdown radially across the
  coaxial gap."
- PDF page 062705-5: "At pressures above 15 Torr, electron impact ionization
  occurs within a few cms, so the gas may breakdown radially across the
  coaxial gap, as in Fig. 2(c)."
- **CONFIRMED.** (Sprint 5 verbatim omits trailing figure reference; substantive
  content confirmed.)

### Summary verdict — Bennett 2017

All 14 verbatim targets confirmed against the PDF pages read this session.
No discrepancies found in value, units, or page attribution. The Codex Sprint 5
audit-row-7 correction (71% at 1 µs, not 500 ns) is independently verified by
the PDF text.

### Units

| quantity | value | units |
| --- | --- | --- |
| seed plasma density | 1e7 | cm⁻³ |
| neutral deuterium fill density | 3.52e17 | cm⁻³ (5.5 Torr D₂) |
| breakdown delay | ~20 | ns |
| explosive emission threshold, bulk | 250 | kV/cm |
| explosive emission threshold, knife-edge | 10 | kV/cm |
| electron temperature at breakdown | 3.5–4.0 | eV |
| sheath current fraction at 1 µs | 71 | % |
| bulk ionization at 100 ns | ~10¹³ | cm⁻³ |
| ionization density at 400 ns (along insulator) | 10¹⁵ | cm⁻³ |

### Symbol map

| symbol | meaning |
| --- | --- |
| n_seed | seed plasma density (10⁷ cm⁻³) initial condition |
| t_breakdown | time from voltage rise to rapid current rise (~20 ns) |
| E_threshold_bulk | explosive emission onset threshold, bulk cathode surface (250 kV/cm) |
| E_threshold_knife | explosive emission onset threshold, cathode knife-edge (10 kV/cm) |
| T_e | mean local electron temperature during breakdown (3.5–4 eV) |
| f_sheath | sheath current fraction at 1 µs (71%) |

### Scope tag

`pf1000_generic` — MA-scale DPF gas breakdown, kinetic PIC methodology.
NOT Akel-16-kV-specific; does not close Akel-scope blockers. Resolves only
CH03/CH04/CH07/CH08 as documented in Sprint 5 extraction.

### Intended code consumer

`src/dpf/first_principles/startup_bvp.py` channels CH03, CH04, CH07, CH08.

### Required tests (before any coefficient may be wired)

1. Unit test: assert that the on-disk PDF SHA-256 matches the value in this
   packet before any target value is consumed at runtime.
2. Unit test: assert `n_seed = 1e7 cm⁻³` is ingested by the CH03 startup
   channel constructor.
3. Unit test: assert `t_breakdown ≈ 20 ns` (±qualifier) is ingested by the
   CH04 channel, not hardcoded silently.
4. Unit test: assert `E_threshold_bulk = 250 kV/cm` and
   `E_threshold_knife = 10 kV/cm` are ingested by the CH07 explosive-emission
   model, with units enforced.
5. Unit test: assert `T_e` initial condition in CH07 lies in [3.5, 4.0] eV.
6. Unit test: assert `f_sheath(t=1e-6 s)` target is 0.71 and is consumed by
   CH08 validation check.
7. Numerical test: run the startup BVP for the MA-scale geometry and confirm
   the simulated breakdown time is within the stated ~20 ns measured window.
8. Same-scope impact test: confirm that using Bennett 2017 parameters does not
   inadvertently change Akel-16-kV or other non-pf1000_generic scope outputs.

### Remaining blockers

- `STARTUP-BVP-CH03`: `existing_kr_source_supported` pending code consumption
  and numerical acceptance.
- `STARTUP-BVP-CH04`: same.
- `STARTUP-BVP-CH07`: same.
- `STARTUP-BVP-CH08`: same.
- KR promotion: `kr_promotion_recommended = True`; the paper must be ingested
  as `KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md` before the
  code consumer cites the KR record. Not done in Sprint 7.
- Filename mislabel: the on-disk PDF filename
  (`schmidt-2017-kinetic-dpf-breakdown.pdf`) does not match the actual authors.
  A renaming or symlink must be agreed before the code consumer hard-codes
  the path; recommend creating a canonical alias in the KR ingest step.

### Acceptance flags

```python
accepted_runtime_claim = False
can_support_first_principles_acceptance = False
runtime_coefficients_wired_sprint_7 = False
```

---

## Packet 3 — Substitute / Cross-Check Lanes (Sprint 6 WS4 Queue Items)

These four sources remain in the **substitute or cross-check lane** with
`source_equivalence_granted = false`. They are NOT promoted in Sprint 7.
Each entry points to the Sprint 6 review packet that carries the full
for/against analysis and the decision list.

### 3a — LXCat D₂ Townsend / e-Neutral

| field | value |
| --- | --- |
| Sprint 6 review packet | `docs/source_equivalence_review/LXCAT_D2_TOWNSEND_E_NEUTRAL_REVIEW_PACKET_2026_05_20.md` |
| Proposed against | Raizer 1991 §4 (`STARTUP-BVP-CH02`) and Itikawa & Mason 2005 (`CLOSURE-BLK-D2-EN-001`) |
| source_equivalence_granted | **false** |
| accepted_runtime_claim | false |
| can_support_first_principles_acceptance | false |
| Lane | candidate_substitute (review-queue; NOT approved for ingestion) |
| Key open items | Dataset version not lockable (no hash mechanism on lxcat.net); Phelps/Morgan/Biagi family choice undecided; H₂→D₂ mass-scaling approximation unquantified; Boltzmann-solver chain (BOLSIG+) not yet documented; KR target extraction not completed. |
| Sprint 7 action | Queue only. No dataset ingestion. No KR record created. |

### 3b — SRIM-2013 / NIST PSTAR / IAEA Stopping Power

| field | value |
| --- | --- |
| Sprint 6 review packet | `docs/source_equivalence_review/SRIM_NIST_IAEA_DEUTERON_STOPPING_REVIEW_PACKET_2026_05_20.md` |
| Proposed against | ICRU Report 49 (`NEUTRON-BLK-002`) |
| source_equivalence_granted | **false** |
| accepted_runtime_claim | false |
| can_support_first_principles_acceptance | false |
| Lane | candidate_substitute (three-way cross-check recommended before selection) |
| Key open items | Dense-plasma vs cold-matter stopping scope not declared; three-way SRIM/PSTAR/IAEA numerical cross-check not completed; SRIM is non-auditable binary; ICRU 49 secondary check not completed; KR target extraction not completed. |
| Sprint 7 action | Queue only. No stopping-power table ingested. No KR record created. |

### 3c — Munro 2012 (LLNL JRNL-676641) Brysk Doppler Re-Derivation

| field | value |
| --- | --- |
| Sprint 6 review packet | `docs/source_equivalence_review/MUNRO_2012_BRYSK_DOPPLER_RE_DERIVATION_REVIEW_PACKET_2026_05_20.md` |
| Proposed against | Brysk 1973 (`NEUTRON-BLK-004`) |
| source_equivalence_granted | **false** |
| accepted_runtime_claim | false |
| can_support_first_principles_acceptance | false |
| Lane | candidate_substitute (Munro 2012 PDF not yet downloaded and read; step-by-step derivation comparison against Brysk not completed) |
| Key open items | Munro 2012 OSTI PDF not on disk; derivation equivalence not verified; bulk-flow/CM-velocity term scope undecided; FWHM vs σ convention not confirmed; peer-review status of LLNL JRNL report not confirmed; KR target extraction not completed. |
| Sprint 7 action | Queue only. No KR record created. |

### 3d — PlasmaPy `formulary.braginskii.ClassicalTransport`

| field | value |
| --- | --- |
| Sprint 6 review packet | `docs/source_equivalence_review/PLASMAPY_BRAGINSKII_CROSS_CHECK_REVIEW_PACKET_2026_05_20.md` |
| Proposed against | Braginskii 1965 Table 2 (`CLOSURE-BLK-BRAG-001`) — cross-check lane only, not substitute |
| source_equivalence_granted | **false** |
| accepted_runtime_claim | false |
| can_support_first_principles_acceptance | false |
| Lane | cross_check_only (second-witness candidate for (review-required) cells) |
| Key open items | PlasmaPy version not pinned; transcription independence not confirmed (Git history not inspected); cell-by-cell comparison script not written; α₀'' (Z=3) anomaly (0.7400) not resolved; scope limitation (no Eqs. 4.30–4.45 coverage) not yet documented in KR record. |
| Sprint 7 action | Queue only. PlasmaPy not imported as runtime dependency. No KR annotation added. |

---

## Source-Equivalence Summary Table

| source | blocker(s) | lane | source_equivalence_granted | sprint_7_action |
| --- | --- | --- | --- | --- |
| Braginskii 1965 Table 2 | CLOSURE-BLK-BRAG-001 | primary (target-extracted) | N/A — primary source | re-audit CONFIRMED; no new wiring |
| Bennett 2017 startup | CH03/CH04/CH07/CH08 | primary (target-extracted) | N/A — primary source | line/page verification CONFIRMED; no new wiring |
| LXCat D₂ cross-sections | CH02, D2-EN-001 | candidate_substitute | **false** | queue only |
| SRIM/PSTAR/IAEA stopping | NEUTRON-BLK-002 | candidate_substitute | **false** | queue only |
| Munro 2012 (Brysk) | NEUTRON-BLK-004 | candidate_substitute | **false** | queue only |
| PlasmaPy Braginskii | CLOSURE-BLK-BRAG-001 | cross_check_only | **false** | queue only |

---

## Sprint 7 WS-E Posture Statement

Per the Sprint 7 controlling document §WS-E (Apply):

> Wire no runtime coefficients until target extraction, code consumption,
> unit tests, numerical tests, and same-scope impact classification exist.

**Runtime coefficients wired this sprint: NONE.**

Both primary-source packets (Braginskii Table 2 and Bennett 2017) are
`target_extracted_source_supported` and have completed line/page verification
this sprint. They are NOT `implemented-candidate`. The transition from
`target_extracted` to `implemented-candidate` requires the additional gates
listed in §Required tests of each packet above, plus independent Codex +
external-team convergence on exact values at a reviewed commit.

The four substitute/cross-check lane sources remain in the review queue with
`source_equivalence_granted = false`. No dataset ingestion, KR record
creation, or code consumption is authorised for those sources in Sprint 7.
