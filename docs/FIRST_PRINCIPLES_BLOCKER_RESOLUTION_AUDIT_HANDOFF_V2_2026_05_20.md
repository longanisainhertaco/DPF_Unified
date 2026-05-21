# First-Principles DPF Blocker Resolution — Audit Handoff **V2** (2026-05-20)

This V2 packet incorporates the ten blocking corrections in
`docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md` and adopts
the required V2 structure (HEAD reconciliation; per-blocker table; separate
source-acquisition table; full KR paths; explicit scope tags;
`accepted_runtime_claim` field; reclassification appendix).

V1 (`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_2026_05_20.md`) is
**superseded** by this V2. Conditional-accept status from the Codex audit lifts
only when V2 is consumed as the controlling errata.

No first-principles runtime acceptance, whole-shot readiness, neutron authority,
startup-BVP acceptance, transport-closure acceptance, or validation certificate
is promoted by V2. Every `accepted_runtime_claim` is `false`.

## 1. HEAD Reconciliation

| Reference | Commit | Role |
| --- | --- | --- |
| V1 research-completion HEAD (stated in V1 header) | `022b774` | Corpus-rescan KR promotion baseline (Bernard 1977 + Plasma Focus Update). Superseded. |
| Sprint 4 implementation HEAD (referenced in V1 exec summary) | `da97ed2` | Sprint 4 wrap-up commit. Superseded. |
| V1 handoff commit | `8f6a0ae` | V1 doc. Superseded by this V2. |
| Post-V1 housekeeping commit | `7999265` | Vetting + source-truth regen + gitignore. Periodic audit 10/10 PASS at this HEAD per `/private/tmp/dpf-unified-audit-logs/20260520T051600Z/summary.md`. |
| Codex audit HEAD reviewed | `7999265` | The audit was performed against this HEAD. |
| V2 handoff commit | `8e6b5e9` | Documentation-only V2 handoff; superseded for automation by the normalized ledger added after Codex V2 audit. |

The Codex audit at `7999265` accepts the qualitative direction of V1 and the
periodic audit 10/10 PASS, then requires the V2 corrections.

## 2. Audit Corrections Incorporated

The ten corrections from `docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md:183-237`
are applied as follows.

| # | Audit correction | V2 fix |
| ---: | --- | --- |
| 1 | Reconcile HEAD narrative | §1 lists every HEAD reference and which supersedes which. |
| 2 | Fix category counts | §3 separates `blocker_count = 31 named items` from `source-acquisition_row_count = 31 visible table rows`; normalized ledgers are now the automation source. |
| 3 | Reclassify Talebitaher | `NEUTRON-BLK-001` keeps primary `corrected_status = existing_kr_target_extraction_pending`; Talebitaher is recorded only as already-target-extracted child context, not counted as a blocker-row status. |
| 4 | Reclassify Bernard 1977 | `Thermonuclear 1/4 prefactor` and `NEUTRON-BLK-001` (historical context) rows in §4 use `existing_kr_target_extraction_pending`. |
| 5 | Reclassify Gribkov Part II | `NEUTRON-BLK-001` (PF-1000 fast-ion) row uses `existing_kr_target_extraction_pending` citing `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`. |
| 6 | Fix current-sheath line ranges | `STARTUP-BVP-CH12` row cites lines `597-601` (`massf`), `631-660` (pressure regimes / `Liz/Li`), `642-660` (Te ≈ 4 eV). Lines `615-670` from V1 are replaced. |
| 7 | Correct Bennett timing | `STARTUP-BVP-CH08` row states 71 % current fraction observed at **1 µs** (not 500 ns); 100 ns / 500 ns / 1 µs contour targets retained. |
| 8 | Downgrade Braginskii table/equation certainty | `CLOSURE-BLK-BRAG-001` row uses `corrected_status = pdf_present_needs_rendered_page_or_ocr_verification`; the Z=1…∞ table at p.251 and Eqs. 4.30-4.45 are not yet machine-verifiable from `pdftotext` and must be rendered or OCR-checked before target extraction. |
| 9 | Replace shorthand citations | Every KR citation in §4 and §5 uses the full `KnowledgeReference/<file>.md:<lines>` form on first mention. |
| 10 | Narrow "promotion resolves blocker" language | No row in §4 has `corrected_status = resolved`; promotions and target extractions create source/typed-evidence levels only. `accepted_runtime_claim = false` everywhere. |

## 3. Executive Summary — corrected counts

- **Blockers (named-ID items)**: 31. Counted by domain: 6 geometry + 13 startup-BVP channels + 5 transport/closures + 6 neutron/mechanism rows including the thermonuclear prefactor + 1 same-scope-comparator decision. Each appears once in the normalized ledger.
- **Source-acquisition rows**: 31 visible rows. Counted in §5 and normalized in `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`. Each row maps to one or more blockers or an explicit context-only intake source. Multiple blockers may share a source (e.g., Bennett 2017 closes source-availability for three startup channels).
- **Normative implementation ledger**: `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`. The Markdown tables below are human-readable summaries; automation and Sprint 5 implementation should consume the CSV ledger.
- **Primary status distribution across the 31 normalized blocker rows**:

  | corrected_status | count |
  | --- | ---: |
  | `existing_kr_source_supported` | 4 |
  | `existing_kr_target_extraction_pending` | 4 |
  | `kr_promotion_recommended` | 4 |
  | `pdf_present_needs_rendered_page_or_ocr_verification` | 1 |
  | `external_acquisition_required` | 13 |
  | `dependency_blocked` | 1 |
  | `absent_from_literature` | 4 |

  Sum = 31. `accepted_runtime_claim = false` on every row.

- **Scope-decision recommendation unchanged**: Option B (PF-1000 full-energy
  27-40 kV) from `docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md`. No
  comparator-matrix cell flips after the Plasma Focus Update + Bernard 1977
  ingest were considered.

## 4. Per-Blocker Table (one row per blocker)

Columns in the normalized ledger: `blocker_id` · `current_repo_status` · `corrected_status` · `source_or_acquisition` · `exact_path_or_full_citation` · `line_or_page_range` · `scope_tag` · `runtime_claim_allowed` · `remaining_action`.

`runtime_claim_allowed` is read as `accepted_runtime_claim`: it is `false`
everywhere unless code, tests, certificate gate, and same-scope review all
exist. The audit's narrowing-of-language correction (§2 row 10) is enforced.

`scope_tag` ∈ {`pf1000_full_energy`, `pf1000_akel_16kv`, `pf1000_generic`,
`nx2_wrong_scope`, `historical_mather_wrong_scope`, `generic_formulary`,
`external_candidate`, `absent`}.

The Markdown tables in §4 are compact human-review summaries and are not the
machine contract. Sprint 5 automation must use
`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`, where every
row has the same schema and explicit `runtime_claim_allowed=false`,
`accepted_runtime_claim=false`, and
`can_support_first_principles_acceptance=false`.

### 4.1 Geometry blockers (6)

| blocker_id | current_repo_status | corrected_status | source_or_acquisition | exact_path_or_full_citation | line_or_page_range | scope_tag | runtime_claim_allowed | remaining_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `PF1000-BLK-009` anode hollow-bore radius | blocked (sim-parameter) | `existing_kr_target_extraction_pending` (Stepniewski review) + `external_acquisition_required` (hardware-scope confirmation) | KR: Stepniewski 2004; KR: Krauz 2012 (bore existence + r≥12 mm probe access); external: Miklaszewski 2001 OR Schmidt 2002 | `KnowledgeReference/doi-10-1016-j-vacuum-2004-05-019-f931cb0b.md`; `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`; Miklaszewski et al. 2001 *Nukleonika* 46:S61-S64 OR Schmidt et al. 2002 *Physica Scripta* 66:168-172 | Stepniewski: `:310-314`; Krauz 2012: `:373` | `pf1000_generic` | false | Complete hardware-scope review of Stepniewski 0.015 m value OR acquire external hardware paper |
| Cathode-cage radius (200 mm) | conflict (Krauz 200 vs Akel 160) | `existing_kr_source_supported` for 200 mm hardware; 160 mm = Lee-fit category | 3 independent KR hardware sources | `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`; `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`; `KnowledgeReference/chunks/update-on-the-scientific-status-of-the-plasma-focus-1385adeb/pages-0026-0050.md` | `:344-349`; `:392-400`; `:822-849` | `pf1000_full_energy` (multi-config — runtime must select Krauz 200 mm explicitly) | false | Runtime constructor must annotate selected configuration explicitly |
| `PF1000-BLK-015` insulator outer radius | blocked source-available revision not mapped | `existing_kr_source_supported` | Scholz 2001 PF-1000 2001 24-rod hardware source | `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md` | `:96-98` | `pf1000_2001_24_rod` | false | Add revision-specific runtime mapping; do not mix into Akel/Krauz scopes |
| `PF1000-BLK-016` insulator wall thickness | blocked (no KR) | `absent_from_literature` | IPPLM facility request | (none on disk) | n/a | `absent` | false | Facility outreach |
| `PF1000-BLK-017` backplate radial extent | blocked (no KR) | `absent_from_literature` | IPPLM facility request | (none on disk) | n/a | `absent` | false | Facility outreach |
| `PF1000-BLK-018` backplate axial thickness | blocked (no KR) | `absent_from_literature` | IPPLM facility request | (none on disk) | n/a | `absent` | false | Facility outreach |

### 4.2 Startup-BVP channels (13)

All rows have `current_repo_status = candidate` or `blocked` (0 of 13
source-supported) and `accepted_runtime_claim = false`.

| blocker_id | corrected_status | source_or_acquisition | exact_path_or_full_citation | line_or_page_range | scope_tag | remaining_action |
| --- | --- | --- | --- | --- | --- | --- |
| `STARTUP-BVP-CH01` gas_and_fill_conditions | `external_acquisition_required` | start-of-shot density / species / Te / Ti for PF-1000 D₂ fill — no single KR source | (no on-disk source closes this) | n/a | `pf1000_generic` | Compose from external (NIST LXCat fill physics + IPPLM operating-point spec) |
| `STARTUP-BVP-CH02` breakdown / Paschen | `external_acquisition_required` | Raizer 1991 §4 OR NIST LXCat Phelps/Morgan | Raizer, Yu. P. (1991) *Gas Discharge Physics* 2nd ed. Springer; LXCat (lxcat.net) | Raizer §4; LXCat: H₂/D₂ datasets | `external_candidate` | Acquire Raizer 1991 OR ingest LXCat dataset |
| `STARTUP-BVP-CH03` preionization | `kr_promotion_recommended` (Bennett 2017 §II: seed density 10⁷ cm⁻³ at 5.5 Torr) + `external_acquisition_required` (cosmic-ray rate, optional) | Bennett 2017 (on-disk PDF) | `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` — **filename mislabel; this is Bennett et al. 2017 Phys. Plasmas 24:062705 DOI 10.1063/1.4985313** | §II (Bennett seed density datum) | `pf1000_generic` (kinetic PIC of DPF gas breakdown — applies broadly) | Rename PDF, promote, target-extract Bennett §II |
| `STARTUP-BVP-CH04` flashover | `kr_promotion_recommended` (Bennett 2017 §I-III: 3-regime taxonomy, λ_ioniz ~ L_i criterion, breakdown delay ~20 ns, Coulomb collision Eq. 1) | Bennett 2017 | (same as CH03) | §I-III | `pf1000_generic` | Promote + target-extract Bennett §I-III; **language: closes source-availability for CH04 — does NOT accept startup BVP** |
| `STARTUP-BVP-CH05` secondary_emission γ (Cu / pyrex / alumina / SS) | `external_acquisition_required` | Hagstrum 1956 + Vaughan formula + CRC Handbook | Hagstrum, H. D. (1956) *Phys. Rev.* 104:317 (Cu); ceramics: Vaughan empirical formula; CRC Handbook electron-emission section | per source | `external_candidate` | Acquire Hagstrum 1956 |
| `STARTUP-BVP-CH06` photoemission | `external_acquisition_required` | Yordanov 2003 (Bennett 2017 Ref. 21) | Yordanov et al. (2003) *Vacuum* 76:365 | full paper | `external_candidate` | Acquire Yordanov 2003 |
| `STARTUP-BVP-CH07` surface_plasma E-field set | partial `kr_promotion_recommended` (Bennett 2017 §III breakdown-phase E-field + 250/10 kV/cm thresholds) + `existing_kr_target_extraction_pending` (UCSD/Beg Te ≈ 4 eV) | Bennett 2017 §III; UCSD/Beg current-sheath-initiation KR | (Bennett path as CH03); `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md` | Bennett §III; b2e95b88: `:642-660` (Te ≈ 4 eV) | `pf1000_generic` (Bennett) + `wrong_scope_method_context` (UCSD 4.6 kJ) | Promote Bennett; target-extract b2e95b88 Te ≈ 4 eV with scope caveat |
| `STARTUP-BVP-CH08` initial E/J distributions | `kr_promotion_recommended` (Bennett 2017 §III) — **71 % sheath current fraction observed at 1 µs (not 500 ns); 100 ns / 500 ns / 1 µs contour targets retained** | Bennett 2017 §III, Fig. 4, Fig. 7 | (same as CH03) | §III | `pf1000_generic` | Promote + target-extract; **corrected timing per audit row 7** |
| `STARTUP-BVP-CH09` species_and_charge_state | `external_acquisition_required` (start-of-shot fields; KR has end-of-rundown only) | composed from CH01 + CH03 + atomic-physics references | (no single source) | n/a | `pf1000_generic` | Compose from CH01 + CH03 + Janev/ADAS |
| `STARTUP-BVP-CH10` ionization / recombination (D₂) | partial `existing_kr_source_supported` (Lotz atomic + NRL hydrogenic) + `external_acquisition_required` (D₂ molecular) | KR Lotz 1967 + NRL Formulary; external Voronov 1997 / Janev-Smith 1993 / Janev 1987 / Open ADAS | `KnowledgeReference/an-empirical-formula-for-the-electron-impact-ionization-cross-section-b5fde85c.md`; `KnowledgeReference/2019nrlplasma-formulary-037290d4.md`; Voronov, G. S. (1997) *At. Data Nucl. Data Tables* 65:1 DOI 10.1006/adnd.1997.0732; Janev-Smith 1993 IAEA-TECDOC-697; Janev et al. 1987 *Elementary Processes in Hydrogen-Helium Plasmas* Springer; Open ADAS (open.adas.ac.uk) `adf11` | Lotz: whole paper; NRL: `:4589-4630` | `generic_formulary` (atomic) + `external_candidate` (D₂ molecular) | Acquire Voronov 1997 + Open ADAS adf11 |
| `STARTUP-BVP-CH11` electron/ion temperature initial | `external_acquisition_required` (no validated start-of-shot Te/Ti for DPF; runtime gates must enforce this) | (no source closes) | (none) | n/a | `absent` | Treat as inherently fail-closed — no acquisition makes it source-supported for the Akel 16 kV scope |
| `STARTUP-BVP-CH12` sheath / surface liftoff | `existing_kr_target_extraction_pending` (`massf = 0.4·p₀^(-1/2)` and pressure regimes) — **wrong-scope method context** | UCSD/Beg current-sheath-initiation KR | `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md` | **`:597-601` (`massf` formula; corrected from V1's `:615-670`)**; `:631-660` (Paschen regimes + `Liz/Li = 2.4`); `:642-660` (Te ≈ 4 eV) | `wrong_scope_method_context` (UCSD 4.6 kJ Mather; not PF-1000 acceptance) | Target-extract with explicit scope caveat |
| `STARTUP-BVP-CH13` handoff_interval into 3D solver | `external_acquisition_required` (no DPF-specific numerical handoff definition in KR) | (no source closes) | (none) | n/a | `absent` | Spec by V&V team; not closed by literature |

### 4.3 Transport / closures (5)

| blocker_id | corrected_status | source_or_acquisition | exact_path_or_full_citation | line_or_page_range | scope_tag | remaining_action |
| --- | --- | --- | --- | --- | --- | --- |
| `CLOSURE-BLK-BRAG-001` Braginskii Z-dep coefficients | **`pdf_present_needs_rendered_page_or_ocr_verification`** (audit row 8) | Braginskii 1965 PDF (on disk) | `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf` | Table 2 reportedly p.251; Eqs. 4.30-4.45 reportedly pp.249-253 — **not yet machine-verifiable via `pdftotext`** | `generic_formulary` | Render / OCR the table+equation pages; attach rendered-page evidence before promotion |
| `CLOSURE-BLK-D2-EN-001` D₂ e-neutral cross-section | `external_acquisition_required` | Itikawa & Mason 2005 OR LXCat | Itikawa, Y. & Mason, N. (2005) *J. Phys. Chem. Ref. Data* 34:1 DOI 10.1063/1.1799251; LXCat Biagi/BSR | full | `external_candidate` | Acquire Itikawa-Mason OR ingest LXCat |
| `CLOSURE-BLK-ION-001` D₂ ionization / recombination | partial `existing_kr_source_supported` (atomic; CH10) + `external_acquisition_required` (molecular) | (same as CH10) | (same as CH10) | (same as CH10) | `generic_formulary` + `external_candidate` | Acquire Voronov 1997 OR Janev-Smith 1993 OR Open ADAS |
| `CLOSURE-BLK-ANOM-001` anomalous resistivity DPF-regime | partial `existing_kr_source_supported` (qualitative DPF-LHDI) + `external_acquisition_required` (quantitative η*) | KR LLNL kinetic sim; external Davidson-Gladd 1975, Bruzzone 2001 (×2) | `KnowledgeReference/comparisons-of-dense-plasma-focus-kinetic-simulations-with-experimental-measurements.md`; Davidson, R. C. & Gladd, N. T. (1975) *Phys. Fluids* 18:1327 DOI 10.1063/1.861021; Bruzzone, H. & Bernal, L. (2001) *Nukleonika* 46:59-61; Bruzzone, H. (2001) *Nukleonika* 46:S3-S7 | LLNL: `:46-49`, `:156-171`, `:185-190` (per audit) | `pf1000_generic` (kinetic-sim qualitative) + `external_candidate` (quantitative) | Acquire Davidson-Gladd 1975 + Bruzzone 2001 (both) |
| `CLOSURE-BLK-REST-001` DPF restrike | `absent_from_literature` (no governing equation anywhere) — closest available: Barbaglia 2010 multiple-pinch (proxy) | Barbaglia et al. (2010) *Plasma Phys. Control. Fusion* 52:032001 | full | `external_candidate` (proxy) | Acquire Barbaglia 2010 as nearest-available; document as proxy, not direct restrike equation |

### 4.4 Neutron mechanism (6)

| blocker_id | corrected_status | source_or_acquisition | exact_path_or_full_citation | line_or_page_range | scope_tag | remaining_action |
| --- | --- | --- | --- | --- | --- | --- |
| `NEUTRON-BLK-001` fast-ion distribution | **`existing_kr_target_extraction_pending`** (Gribkov/Scholz Part II; Bernard 1977 historical; reclassified per audit rows 4 + 5) + `already_target_extracted_in_kr_targets` (Talebitaher NX2 — audit row 3) + `external_acquisition_required` (PF-1000 quantitative — Gribkov 2007 J. Phys. D 40:3592 confirmed already in KR; Schmidt 2012 PRL still external) | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` (Gribkov-Scholz Part II); `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md` (Bernard 1977 historical); Schmidt et al. (2012) *Phys. Rev. Lett.* 109:205003 (external) | Scholz/Gribkov Part II: `:318-323` (direct fast-deuteron section); `:445-460` (5-counter anisotropy `Y0/Y90 ≈ 1.8`); `:1138-1168` (neutron spectra + 100 keV beam-target). Bernard 1977: `:1185-1193` (3-angle TOF) | `pf1000_full_energy` (Scholz/Gribkov Part II); `historical_mather_wrong_scope` (Bernard 1977); `nx2_wrong_scope` (Talebitaher) | Target-extract Scholz/Gribkov Part II at the cited lines, with `pf1000_full_energy` scope tag |
| `NEUTRON-BLK-002` deuteron stopping power | `external_acquisition_required` | ICRU Report 49 (1993); Andersen & Ziegler (1977); SRIM-2013 | ICRU Report 49 (1993) "Stopping Powers and Ranges for Protons and Alpha Particles"; Andersen, H. H. & Ziegler, J. F. (1977) *Stopping and Range of Ions in Matter*, Pergamon | full tables | `external_candidate` | Acquire ICRU 49 (primary) |
| `NEUTRON-BLK-003` beam-target yield | dependency-blocked on BLK-001 + BLK-002 | σ(E) is `existing_kr_source_supported` (Bosch-Hale); f(E,θ) and dE/dx are required from BLK-001 + BLK-002 | `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md` (σ(E)); BLK-001 + BLK-002 references | full | `pf1000_full_energy` (after BLK-001/002) | Resolves automatically once BLK-001 and BLK-002 are extracted |
| `NEUTRON-BLK-004` Brysk Doppler | `external_acquisition_required` | Brysk 1973 | Brysk, H. (1973) *Plasma Phys.* 15:1282 | derivation + FWHM = 82.5 √(T_i[keV]) keV | `external_candidate` | Acquire Brysk 1973 |
| `NEUTRON-BLK-005` same-scope anisotropy (Akel 16 kV) | `external_acquisition_required` (Akel-authored 16 kV anisotropy paper search) | (search Nukleonika / J. Fusion Energy / Phys. Plasmas) | external | n/a | `pf1000_akel_16kv` (only the 16 kV scope) | Targeted literature search for Akel 16 kV anisotropy publication |
| Thermonuclear `1/4` volumetric prefactor | **`existing_kr_target_extraction_pending`** (Bernard 1977 may carry it; audit row 4) + `external_acquisition_required` if not in Bernard 1977 | `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md` (Bernard 1977); Glasstone & Lovberg (1960) *Controlled Thermonuclear Reactions* Ch. 1 | Bernard 1977: target-extract for n_d²·⟨σv⟩/4 derivation; Glasstone external if Bernard insufficient | `historical_mather_wrong_scope` (Bernard 1977 device, but the identical-particle 1/4 derivation is device-independent) | Target-extract Bernard 1977 for the prefactor derivation; if absent, acquire Glasstone & Lovberg |

### 4.5 Same-scope comparator decision (1)

| blocker_id | corrected_status | source_or_acquisition | exact_path_or_full_citation | line_or_page_range | scope_tag | runtime_claim_allowed | remaining_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Same-scope comparator scope decision | `existing_kr_source_supported` (matrix) | `docs/FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md` | full memo | matrix lives in the memo | `pf1000_akel_16kv` (current) → recommendation `pf1000_full_energy` (Option B) | false (no certificate gate) | Audit team to approve / replace scope decision before Sprint 5 |

## 5. Source-acquisition Table (one row per source)

Columns: `priority` · `source` · `resolves_blockers` · `already_in_kr` · `on_disk_path` · `external_required` · `notes`.

| priority | source | resolves_blockers | already_in_kr | on_disk_path | external_required | notes |
| --- | --- | --- | --- | --- | --- | --- |
| **P1** | Bennett et al. 2017 *Phys. Plasmas* 24:062705 DOI 10.1063/1.4985313 | CH03, CH04, CH08, partial CH07 | no | `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf` (mislabel — rename to `bennett-2017-kinetic-dpf-breakdown.pdf`) | no | Closes source-availability and target-extraction blockers for 3 startup channels; **does not by itself accept any startup BVP** |
| **P1** | Braginskii 1965 (Reviews of Plasma Physics Vol. 1) | `CLOSURE-BLK-BRAG-001` | no | `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf` | no | **`pdf_present_needs_rendered_page_or_ocr_verification`** per audit row 8 — render or OCR p.251 + pp.249-253 before extraction |
| **P1** | Raizer 1991 *Gas Discharge Physics* 2nd ed. Springer §4 | CH02 (D₂ Townsend α + Paschen A/B) | no | (none) | yes | Or substitute NIST LXCat Phelps/Morgan |
| **P1** | NIST LXCat database (lxcat.net) | CH02, `CLOSURE-BLK-D2-EN-001` | no | (online) | yes | Phelps / Morgan / Biagi datasets |
| **P1** | Brysk, H. (1973) *Plasma Phys.* 15:1282 | `NEUTRON-BLK-004` | no | (none) | yes | Foundational Doppler-broadening derivation |
| **P1** | ICRU Report 49 (1993) | `NEUTRON-BLK-002` | no | (none) | yes | Stopping powers for p and α; deuteron interpolation |
| **P1** | Davidson, R. C. & Gladd, N. T. (1975) *Phys. Fluids* 18:1327 DOI 10.1063/1.861021 | `CLOSURE-BLK-ANOM-001` (quantitative) | no | (none) | yes | LHDI anomalous transport derivation |
| **P1** | Bruzzone, H. & Bernal, L. (2001) *Nukleonika* 46:59-61 | `CLOSURE-BLK-ANOM-001` (DPF-scope) | **yes** | `KnowledgeReference/the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-instabilities-in-plasma-magnet-73668d0e.md` | no | User-supplied duplicate confirmed 2026-05-20; target extraction/review still required before quantitative closure |
| **P1** | Bruzzone, H. (2001) *Nukleonika* 46:S3-S7 | `CLOSURE-BLK-ANOM-001` companion | no | (none) | yes | Companion source still external/not located; do not accept quantitative anomalous-resistivity closure from one paper alone |
| **P2** | Scholz / Gribkov 2007 Part II (already in KR) | `NEUTRON-BLK-001` (PF-1000 fast-ion); `NEUTRON-BLK-005` adjacent (anisotropy at full energy, not 16 kV) | **yes** | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` (also `archive_reference_OLD/references/papers/core-dpf/scholz-2007-pf1000-part2-jphysd.pdf`) | no | **Reclassified from external (V1) to existing KR target-extraction (V2) per audit row 5.** J. Phys. D 40:3592 DOI 10.1088/0022-3727/40/12/008. Targets: `:318-323`, `:445-460`, `:1138-1168` |
| **P2** | Voronov, G. S. (1997) *At. Data Nucl. Data Tables* 65:1 DOI 10.1006/adnd.1997.0732 | CH10, `CLOSURE-BLK-ION-001` | no | (none) | yes | Hydrogen ionization rate fit |
| **P2** | Janev, R. K. & Smith, J. J. (1993) IAEA-TECDOC-697 | CH10, `CLOSURE-BLK-ION-001` | no | (none) | yes | Cross sections + rates for H |
| **P2** | Itikawa, Y. & Mason, N. (2005) *J. Phys. Chem. Ref. Data* 34:1 DOI 10.1063/1.1799251 | `CLOSURE-BLK-D2-EN-001` | no | (none) | yes | H₂ cross-sections (D₂ via isotope substitution) |
| **P2** | Schmidt et al. (2012) *Phys. Rev. Lett.* 109:205003 | `NEUTRON-BLK-001` (kinetic PIC benchmark) | no | (none) | yes | Fully kinetic PIC reference |
| **P2** | Hagstrum, H. D. (1956) *Phys. Rev.* 104:317 | CH05 (Cu γ) | no | (none) | yes | + Vaughan formula + CRC Handbook for ceramics |
| **P2** | Miklaszewski et al. (2001) *Nukleonika* 46:S61-S64 OR Schmidt et al. (2002) *Physica Scripta* 66:168-172 | `PF1000-BLK-009` (hardware-scope hollow bore) | no | (none) | yes | PF-1000 commissioning / overview papers |
| **P1** | Scholz et al. 2001 *Nukleonika* 46:35-39 | `PF1000-BLK-004`, `PF1000-BLK-015` | **yes** | `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md` | no | User-supplied duplicate confirmed 2026-05-20; existing KR source reused and target-extracted for 24-rod rod length and insulator outer radius; runtime revision mapping still required |
| **P1** | Scholz et al. 2000 *Nukleonika* 45:155-158 | `PF1000-BLK-004` | **yes** | `KnowledgeReference/pf-1000-device-a2d6bc15.md` | no | User-supplied duplicate confirmed 2026-05-20; corroborates early 24-rod PF-1000 hardware and bank/chamber context; runtime revision mapping still required |
| **P2** | Herold et al. 1989 *Nuclear Fusion* 29:1255-1269 | context only | **yes** | `KnowledgeReference/comparative-analysis-of-large-plasma-focus-experiments-performed-at-ipf-stuttgart-and-ipj-51a54695.md` | no | Cross-machine POSEIDON/PF-360 context only; no PF-1000 same-scope acceptance |
| **P2** | Scholz et al. 1999 *Physics Letters A* 262:453-456 | context only | **yes** | `KnowledgeReference/foam-liner-driven-by-a-plasma-focus-current-sheath-8324d619.md` | no | Modified PF-1000 foam-liner load context only; not standard whole-shot geometry |
| **P3** | Loarer et al. 2007 *Nuclear Fusion* 47:1112-1120 | context only | **yes** | `KnowledgeReference/gas-balance-and-fuel-retention-in-fusion-devices-09d09d6a.md` | no | Tokamak plasma-wall fuel-retention context only; not a DPF authority source |
| **P2** | Shakya et al. 2015 *Journal of Nepal Physical Society* 3:55-62 | context only | **yes** | `KnowledgeReference/comparison-of-plasma-dynamics-in-plasma-focus-devices-pf1000-and-pf400-9094f12f.md` | no | Reduced Lee-model PF1000/PF400 comparison context only; baseline/comparator only |
| **P2** | Gribkov & Malaquias 2006 *Nukleonika* 51:5-13 | context only | **yes** | `KnowledgeReference/dense-magnetized-plasma-and-its-applications-review-of-the-3-year-activity-of-the-iaea-co-cca325c9.md` | no | Dense magnetized plasma applications review context only; no direct PF-1000 runtime blocker closure |
| **P2** | Bernard 1977 (Limeil/Jülich Mather DPF review) | `NEUTRON-BLK-001` historical context; Thermonuclear 1/4 prefactor (potential) | **yes** | `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md` (also `downloaded_books_papers/Research Papers/2026-05-20-corpus-rescan/bernard1977.pdf`) | no | **Reclassified from external (V1) to existing KR target-extraction (V2) per audit row 4.** First direct Thomson Ti=700 eV (filament phase, ~500 kA Mather; historical wrong-scope but unique in corpus). Target lines: `:455-461`, `:976-1033`, `:1185-1193`, `:1546-1547` |
| **P2** | Talebitaher 2012 PhD thesis (NX2 CAI) | `NEUTRON-BLK-001` (NX2 cone model) | **yes** | `downloaded_books_papers/Research Papers/2026-05-16-user-validated-theses/PhD2012AlirezaTalebitaher.pdf`; already target-extracted in `src/dpf/first_principles/source_targets.py` as `target_extracted_nx2_detector_anisotropy_context` | no | **Reclassified from KR-promotion-recommended (V1) to already-target-extracted (V2) per audit row 3.** No further promotion needed; only `nx2_wrong_scope` use permitted |
| **P3** | Barbaglia et al. (2010) *Plasma Phys. Control. Fusion* 52:032001 | `CLOSURE-BLK-REST-001` (proxy) | no | (none) | yes | Multiple-pinch ≠ restrike strictly; closest available |
| **P3** | Janev et al. (1987) *Elementary Processes in Hydrogen-Helium Plasmas* Springer | CH10 | no | (none) | yes | Standard DPF-regime reference |
| **P3** | Yordanov et al. (2003) *Vacuum* 76:365 | CH06 (photoemission) | no | (none) | yes | Bennett 2017 Ref. 21 |
| **P3** | Glasstone & Lovberg (1960) *Controlled Thermonuclear Reactions* Ch. 1 | Thermonuclear 1/4 prefactor (if Bernard 1977 target-extraction doesn't yield it) | no | (none) | yes | Identical-particle pair-counting derivation |
| **P3** | Sagdeev & Galeev (1969) *Nonlinear Plasma Theory* Benjamin | `CLOSURE-BLK-ANOM-001`, `CLOSURE-BLK-REST-001` foundational | no | (none) | yes | Optional foundational reference |
| **P4** | IPPLM facility engineering drawings | `PF1000-BLK-016`, `-017`, `-018` | no | (none — not in any literature) | yes (facility) | Acquisition via M. Paduch / R. Miklaszewski; or IAEA CRP 11940/11941; or ICDMP workshop reports |

(Source-acquisition rows counted: 31 visible table rows. Context-only
user-supplied duplicate sources are included so intake coverage is explicit;
Miklaszewski/Schmidt remains a grouped hardware-source alternative.)

## 6. Reclassification Appendix — V1 → V2

| Item | V1 classification | V2 classification | Reason |
| --- | --- | --- | --- |
| Talebitaher 2012 PhD subset | `KR_PROMOTION_RECOMMENDED` | `already_target_extracted_in_kr_targets` | Already in `src/dpf/first_principles/source_targets.py` as `target_extracted_nx2_detector_anisotropy_context` (audit row 3). |
| Bernard 1977 | `EXTERNAL_ACQUISITION_REQUIRED` | `existing_kr_target_extraction_pending` | Already promoted into KR via `docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.md:18-19` (audit row 4). KR file: `KnowledgeReference/the-dense-plasma-focus-a-high-intensity-neutron-source-f0a3910d.md`. |
| Gribkov 2007 J. Phys. D 40:3592 | `EXTERNAL_ACQUISITION_REQUIRED` | `existing_kr_target_extraction_pending` | Already in KR as `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` (audit row 5). `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md:470-471` records this. |
| UCSD/Beg `massf` formula | cited at `:615-670` (V1) | cited at `:597-601` (V2) plus `:631-660` + `:642-660` for related material | Line range was wrong (audit row 6). Verified at `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md:597-601`. |
| Bennett 2017 71 % current fraction timing | "by 500 ns" (V1) | "at 1 µs" (V2) | Timing was wrong (audit row 7). The 100 ns / 500 ns / 1 µs contour timestamps remain accurate. |
| Braginskii 1965 Table 2 + Eqs. 4.30-4.45 | `KR_PROMOTION_RECOMMENDED` (V1) | `pdf_present_needs_rendered_page_or_ocr_verification` (V2) | `pdftotext` failed to expose Table 2 or Eqs. 4.30-4.45 (audit row 8). Page rendering or OCR required before target extraction. |
| Generic language "promotion resolves blocker" | (used in V1 prose) | replaced by "closes source-availability and target-extraction blockers" or "creates KR text source" | Promotion creates a KR text source; target extraction creates typed source evidence; runtime acceptance requires code + tests + certificate gate (audit row 10). |
| Shorthand KR citations (`experimental-study-...-705bcc83.md`) | partial shorthand (V1) | full `KnowledgeReference/...` paths on first mention (V2) | Audit row 9. |
| Category counts (V1: 21 blockers vs 23 in category table vs ~20 acquisition rows) | inconsistent (V1) | 31 normalized blocker rows and 31 visible source-acquisition rows, with separate CSV ledgers as the automation source | Audit row 2. |
| HEAD references (V1: `022b774` / `da97ed2` mixed) | inconsistent (V1) | `022b774` → `da97ed2` → `8f6a0ae` → `7999265` → V2 HEAD; supersession explicit (V2) | Audit row 1. |

## 7. Audit-team Verification Checklist

For each row in §4 the audit team should:

1. **Read the cited KR file at the cited lines.** Every citation in §4 uses the full `KnowledgeReference/...` path.
2. **Check `accepted_runtime_claim` is `false`.** No row asserts runtime acceptance; the field is `false` everywhere.
3. **For `existing_kr_source_supported`**: confirm KR file content matches the claim AND the scope tag is correct.
4. **For `existing_kr_target_extraction_pending`**: confirm the KR file is currently `text_parity_extracted_review_needed` and the cited lines contain the named targets.
5. **For `kr_promotion_recommended`**: confirm the PDF is on disk at the cited path; spot-check page contents match.
6. **For `kr_promotion_recommended_pending_ocr_verification`** (Braginskii 1965 only): confirm `pdftotext` does NOT expose the cited table/equations; render or OCR the relevant pages and attach rendered-page evidence before any extraction.
7. **For `external_acquisition_required`**: confirm the cited DOI/journal/page is real and the paper is genuinely absent from disk.
8. **For `absent_from_literature`**: confirm via independent KR + disk + DB search that no publication contains the value at the required scope.

## 8. Acceptance Boundary (V2 → Sprint 5)

Sprint 5 may proceed if and only if:

- The audit team has approved V2 corrections (or replaced specific rows).
- The 12 true external P1+P2 acquisition rows have been triaged into MUST / SHOULD / DEFER for Sprint 5.
- The 1 KR promotion (Bennett 2017) and 1 KR-promotion-pending-OCR-verification (Braginskii 1965) are scheduled.
- The 5 existing-KR target extractions (Scholz/Gribkov Part II, Bernard 1977, Plasma Focus Update Te filter-ratio, UCSD/Beg `:597-601`+`:631-660`, Stepniewski hardware-scope review) are scheduled.
- The 4 `absent_from_literature` items (PF-1000 insulator wall thickness, PF-1000 backplate radial extent, PF-1000 backplate axial thickness, and the DPF restrike-equation gap) have an explicit decision: source acquisition/facility outreach OR accept as structural gaps.
- The comparator-scope decision (Option B PF-1000 full-energy) is locked or replaced.

`can_support_first_principles_acceptance` remains `false` everywhere. No
validation, engineering-firm-ready, or whole-shot claim is implied by V2.

---

**Prepared by:** lead synthesis incorporating the 10 Codex audit corrections, 2026-05-20.
**Verifiable basis:** every claim in V2 is backed by either a full `KnowledgeReference/<file>.md:<lines>` citation, an on-disk PDF path, or a full external citation (author, year, source, vol:page, DOI).
**Supersedes:** `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_2026_05_20.md` (V1).
**Audit source:** `docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md`.
