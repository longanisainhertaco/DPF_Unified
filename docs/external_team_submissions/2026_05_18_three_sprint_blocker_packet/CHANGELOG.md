# Changelog — Submission 1 + Sprint 1.1 + Sprint 2

Base commit (audited state): `76480b0`
Branch: `codex/corpus`

## Commits

### Sprint 1 — Control Gate Hardening (`76480b0`..`fe038f7`)

| Commit | Subject |
| --- | --- |
| `3dc4c11` | docs: add Codex WP-N1/WP-N4 first-principles audit |
| `4424785` | fix(first-principles): require non-empty source_packet_hashes provenance (A-1) |
| `80654b9` | fix(scripts): deterministic output and read-only check mode for first-principles gates (A-3) |
| `b626ad9` | style(first-principles): clear broad ruff lint slice (A-5) |
| `55e3f94` | fix(first-principles): rehydrate cumulative ledger on segmented resume (A-6) |
| `1abe15a` | chore(results): quarantine stale pre-HEAD audit artifacts (A-2) |
| `bf22c33` | ci(first-principles): read-only audit gates, recursive archive scan, dirty-tree assertion (A-3/A-4) |
| `fe038f7` | docs(srs): mark DPF-PHYS-020/023 partial, regenerate RTM, close Sprint 0 with debt (A-7) |

The submission packet commit for Sprint 1 (`fa9088e` — docs: add Sprint 1
control-gate submission packet) is a documentation-only wrapper: changed paths
are all under `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`.

### Sprint 1 Wrappers + Sprint 1.1 + Sprint 2 — Hygiene and Proposals (`fe038f7`..`c52bed3`)

| Commit | Subject |
| --- | --- |
| `f07d4ba` | docs: add Sprint 1 control-gate submission packet *(documentation-only wrapper)* |
| `fa9088e` | docs: pin --date in submission audit-command transcript *(documentation-only)* |
| `06cfd64` | docs: add Codex Sprint 1 control-gate audit |
| `9df8d3b` | style: clear repository-wide ruff debt so the CI lint job passes (RC-2) |
| `49e80ee` | docs(kr): ingest verified Auluck 2021 power-balance equations; refresh source-truth index |
| `bd840f4` | docs: add Sprint 2 WP-N1B and WP-N4B proposal set |
| `0585eec` | docs: add tracked copy of verified Auluck equations to Sprint 2 packet |
| `0bef78c` | fix(first-principles): add artifact linter check C8 and package schema tests (RC-3/5/6/7) |
| `c52bed3` | docs: finalize Sprint 1.1 — current-HEAD transcript block and RC statuses (RC-4) *(documentation-only wrapper)* |

### Sprint 2 Audit and Periodic Runner (`c52bed3`..`6af4454`)

| Commit | Subject |
| --- | --- |
| `6af4454` | docs: add sprint 2 audit and periodic runner |

### Sprint 2 Implementation — WP-N1B / WP-N4B (`6af4454`..HEAD)

| Commit | Subject |
| --- | --- |
| `4b080eb` | feat(first-principles): implement WP-N1B six-term Auluck power-port ledger (fail-closed) |
| `4c8dac1` | feat(first-principles): implement WP-N4B cross-restart ledger merge and artifact combiner |
| `93620ba` | docs: synchronize Sprint 2 packet hygiene and add consistency tests (F2/F3/F4) |
| `49ead59` | docs: review sprint 3 source gap taxonomy |

### S3.1 Packet Hygiene and S3R Runtime Foundations

S3.1/S3R scoped changes to the packet documents, runtime-foundation ledgers,
SRS/RTM exports, and candidate runtime modules. These changes do not promote any
accepted physics or validation claim.

- `sprint_3/PENDING.md` deleted; superseded by `sprint_3/SPRINT_3_STATUS_LEDGER.md`
  (`research_packet_delivered=true`, `runtime_foundation_delivered=true`,
  `accepted_physics_delivered=false`, `validation_delivered=false`).
- `sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md`: stale "Sprint 2.2 open",
  "WP-N2 not delivered", "WP-N5 closure registry not delivered" language replaced
  with explicit research, runtime-foundation, accepted-physics, and validation
  delivery distinctions.
- `sprint_3/WP_N4_PERFORMANCE_AND_RUN_PLAN.md`: "unknown" grid size replaced
  with `blocked_by_missing_local_source`.
- `sprint_3/WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md`: all `[KR: same file ...]`
  shorthand citations expanded to exact local paths with line ranges.
- `README.md`, `THREE_SPRINT_FINAL_SUMMARY.md`: `sprint_3/PENDING.md` nav
  references updated to `sprint_3/SPRINT_3_STATUS_LEDGER.md`; Sprint 3 status
  updated to reflect research packets delivered.
- Runtime-foundation packet rows and traceability ledgers updated for
  `source_geometry.py`, `startup_bvp.py`, `closure_packet.py`,
  `neutron_authority.py`, `segmented_whole_shot.py`,
  `segmented_whole_shot_combine.py`, and `certificate_gate.py`. All rows retain
  `can_support_first_principles_acceptance=false`.
- `tests/test_external_team_submission_package.py`: new S3.1 consistency tests
  added (stale-state rejection, shorthand-citation rejection).

### Sprint 3 Implementation And Traceability (`0b8fa97`..HEAD)

Sprint 3 packages S3.1–S3.9 are committed: `100d87d` (S3.1 packet hygiene),
`0e91f08` (S3.2/S3.3 PF-1000 geometry packet + Sigma_p plumbing), `06744fd`
(S3.4 typed startup BVP packet), `7dd1994` (S3.5 closure registry + PlasmaPy
regime gate), `d1dc17c` (S3.6 mechanism-separated neutron authority packet),
`6660eb9` (S3.7/S3.8 numerical acceptance gates + certificate scaffold),
`f7bb9f8` (module-source-vetting regeneration), `e9b3c20` (S3.9 SRS/RTM and
packet-ledger traceability). The Sprint 3 final submission is
`sprint_3/SPRINT_3_FINAL_SUBMISSION.md`.

`269d7d1` is the Sprint 3 final-submission wrapper commit: it adds
`SPRINT_3_FINAL_SUBMISSION.md` and updates this `CHANGELOG.md`. The current
package-consistency test `test_changelog_covers_all_commits_since_base` exempts
only HEAD because the changelog is committed in the same commit that changes it
(it checks `76480b0..HEAD~1`).

`770984b` is the Codex Sprint 3 completion-audit wrapper commit: it adds
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_COMPLETION_2026_05_19.md`,
`docs/FIRST_PRINCIPLES_SPRINT3R_REMEDIATION_HANDOFF_2026_05_19.md`, and updates
the active findings docs with the Sprint 3R gate decision.

### Sprint 3R Remediation (closes Sprint 3 completion-audit findings A1-A12)

`81c7481` — S3R.1 packet hygiene (closes A10, A11, A12-docs): 4-boolean
delivery state (research_packet_delivered / runtime_foundation_delivered /
accepted_physics_delivered / validation_delivered), BLOCKER_MATRIX rows no
longer reference deleted `sprint_3/PENDING.md`, S3.1/S3.9 rows added to
`CLAIMS_LEDGER.csv` and `TEST_MAP.csv`, WP-N5 shorthand `[KR: ...]` citations
expanded to full `KnowledgeReference/<file>.md:<lines>` form,
`first_principles/closures.py` → `closure_packet.py` and
`first_principles/certificate.py` → `certificate_gate.py` corrected in
`DPF_REQUIREMENTS_BASELINE`, `SRS_TRACEABILITY_MATRIX.{csv,json}`, packet
CHANGELOG/CLAIMS_LEDGER/BLOCKER_MATRIX/SPRINT_3_STATUS_LEDGER,
`tests/test_external_team_submission_package.py` strengthened to reject the
stale `runtime_implementation_delivered` field, live `PENDING.md` references,
broader shorthand citation forms, and the wrong module paths.
`CodexFindings.md` and `CortexFindings.md` gain a `## Sprint 3R Status
(2026-05-19)` section listing A1-A12.

`c9c7160` — S3R.2 bind startup acceptance to typed packet (closes A1):
`build_startup_bvp_packet()` now derives `can_support` from the embedded typed
`StartupPacket`'s `can_support_first_principles_acceptance` as the leading
AND-term; caller-declared `accepted_channels` can no longer override the typed
packet's fail-closed posture. Since WP-N2 guarantees 0 computed channels of 13,
`status='accepted_startup_bvp_packet'` is structurally unreachable. Negative
tests added for accepted-mode spoof payload, reviewed-without-hashes payload,
seeded-layer mode, and the CLI blocker propagation.

`2a3e891` — S3R.3 neutron-authority status splits + NumPy 2 trapezoid fallback
(closes A2, A3). A2: scalar yield is now permanently `candidate_comparator_only`
— `_accepted_channels_from_targets()` skips `accepted.add()` for
`neutron_scalar_yield` and emits decision
`candidate_comparator_only_scalar_not_mechanism_authority`; `_channel_statuses()`
returns `candidate_comparator_only` for `same_scope_scalar_yield` regardless of
the accepted set. A3: `beam_target.py:91` uses the lazy `hasattr` fallback
(`integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz`) so the
diagnostic does not raise `AttributeError` on NumPy 2 where `np.trapz` is
absent. Negative tests cover scalar-only authority denial, accepted target
metadata without mechanism histories, missing stopping/detector keeping
total-yield blocked, and `_trapezoid_integral` working under NumPy 2.

`224a7ea` — S3R.4+S3R.5 geometry mask statuses + Sigma_p packet schema +
dict-form power-port consumption (closes A4, A5, A6, A7). PF-1000 material masks
are no longer heuristic projections without status: `PF1000MaskManifest` gains
`mask_class_status` ∈ {`source_supported`, `candidate_projection_not_source_mask`,
`blocked`} per class; per-mask SHA-256 is preserved only with the status that
generated the mask. The insulator and cathode-cage masks become
`candidate_projection_not_source_mask` until insulator outer-radius / cage
radius become source-backed. The under-resolution gate is extended to
`insulator_exposed_length_m` when source-supported. `SigmaPSurfacePacket` gains
five digest fields (`sigma_p_face_set_sha256`, `moving_classification_sha256`,
`omega_partition_sha256`, `material_mask_sha256_by_class`,
`moving_classification_status`) plus the source geometry hash preserved on
blocked returns. `power_port.py` reconstructs dict-form Sigma_p packets via
`_sigma_p_packet_from_dict()` or emits the named
`_SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED` blocker — no silent discard.
`_sigma_p_surface_term()` adds sign-convention and moving-classification
negative controls. Terms II/IV/V/VI remain blocked until the Sprint 4 surface
integrator exists.

`f390cf3` — S3R.6 close closure-matrix completeness (closes A8): the top-level
`effects` dict now contains `electron_inertia` and `stopping_collisions`
(closure_packet.py:1276 / :1287) mirroring the static registry records at
:822 and :869. The derived `closure_matrix_status_by_effect`,
`closure_effect_status`, and `missing_or_unaccepted_effects` maps now expose
all 12 required effects symmetrically. PlasmaPy remains cross-check only —
toggling `community_formula_audit` does not change any effect's status,
implemented flag, or acceptance flag. Negative tests:
`test_required_effects_symmetric_with_top_level_effects`,
`test_electron_inertia_and_stopping_collisions_blocked`,
`test_plasmapy_cannot_promote_or_reject_local_closure`.

`dfa9169` — S3R.7 extended cumulative fields through merged ledgers +
certificate fail-closed (closes A9, A12 code-side). All four extended S3.7
fields (`cumulative_field_energy_delta_J`, `cumulative_pml_removed_energy_J`,
`cumulative_power_port_work_J`, `cumulative_ionization_step_count`) now
survive `segmented_whole_shot_combine.merge_cumulative_ledgers` —
aggregation rule (all from terminal manifest sidecar, accumulated additively
per segment) documented inline. Pre-S3R.7 manifests with absent extended
fields no longer raise; from_state_dict falls back to the dataclass 0.0/0
default. Negative tests: three-segment merge preserves all 4 fields;
missing-field zero-baseline default; adversarial certificate fixture cannot
promote validation. A12 code-side: implementation was already named
`certificate_gate.py` consistently in code; no in-code path corrections
needed (the doc/RTM corrections were done in S3R.1).

`b6f7698` — pre-commit linter auto-fixes + S3R.7 CHANGELOG sync. Citation form
normalization on `WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md` and
`WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md`; ruff hoisted imports in
`tests/test_first_principles_neutron_authority.py`. No semantic change.

`46b705f` — S3R.8 prep: 12 A1-A12 closure rows appended to `BLOCKER_MATRIX.csv`
(status `closed`, Sprint 3R owner, evidence cites the S3R.X commit and tests);
7 S3R.1..S3R.7 rows appended to `CLAIMS_LEDGER.csv` (one per remediation
package, `can_support_first_principles_acceptance=false` on every row);
12 A1-A12 rows appended to `TEST_MAP.csv` with exact pytest commands;
ruff I001 import-sort fix on `tests/test_first_principles_power_port.py`.

`46a0b56` — regenerate `FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`
after Sprint 3R implementation added new functions/constants
(`mask_class_status`, `_sigma_p_packet_from_dict`,
`_SERIALIZED_SIGMA_P_PACKET_NOT_SUPPORTED`, `electron_inertia`/
`stopping_collisions` effects, extended cumulative fields).
`strict_passed=true`, `total_modules=290`,
`missing_source_reference_count=0`.

The Sprint 3R final-submission wrapper commit adds
`sprint_3/SPRINT_3R_FINAL_SUBMISSION.md` and updates this `CHANGELOG.md`; it is
HEAD-exempt from `test_changelog_covers_all_commits_since_base`.

`8d41cad` — Sprint 3R final-submission wrapper (amended for shorthand-citation
regex compliance in the A11 table cell). Adds
`sprint_3/SPRINT_3R_FINAL_SUBMISSION.md` and the corresponding CHANGELOG
sections.

### Sprint 4 prep — target extractions (closes
`source_available_not_target_extracted` material into typed targets)

Pre-Sprint-4 prep that converts already-local `KnowledgeReference/` material
into typed target records without promoting validation or whole-shot
first-principles acceptance. Documented in
`docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md`. Adds the
`sprint4_source_available_target_extractions()` packet
(`src/dpf/first_principles/source_targets.py`) and
`pf1000_krasa_vessel_scatter_anisotropy_targets()`
(`src/dpf/validation/kr_targets.py`); promotes `chamber_wall_material` and
`chamber_wall_thickness_m` to `source_supported` in `PF1000GeometryPacket` via
Krasa 2008. The chamber-wall mask remains
`candidate_projection_not_source_mask` until the cathode-cage radial split
becomes source-supported; a focused test enforces this. Stepniewski 2004's
0.015 m hollow-bore value is extracted but intentionally not promoted into
`PF1000GeometryPacket` because it is a simulation-parameter context, not
reviewed hardware-scope geometry. Tests: 165 passed.

### Sprint 4 Priority closures (converts target extractions into source-supported-or-blocked closures)

`66bed52` — Sprint 4 prep target extractions (above section): per
`docs/FIRST_PRINCIPLES_TARGET_EXTRACTIONS_2026_05_20.md`, source-available KR
material converted to typed target records; chamber-wall promoted to
source_supported via Krasa 2008; Stepniewski 0.015 m extracted but not
promoted into `PF1000GeometryPacket`.

`a6fc1d6` — S4-Priority 1 (Geometry Authority closures): (1a) Stepniewski
0.015 m hollow-bore **blocked** — KR text is a simulation-parameter section,
not hardware metrology, so it cannot promote
(`PF1000-BLK-009-anode-bore-radius-target_extracted_modeling_context_requires_review`).
(1b) Cathode-cage radius conflict **unresolvable** — Krauz 200 mm is geometric
metrology; Akel 160 mm is a Lee-model fit parameter (category mismatch);
field stays `status=conflict`. (1c/1d) Insulator outer-radius / wall-thickness
and backplate radial-extent / axial-thickness **blocked** with named missing
data (`PF1000-BLK-015`, `-016`, `-017`, `-018`). 8 new tests cover all four
verdicts across krauz_2012, akel_shot_12581, scholz_gribkov_revision
constructors.

`3d862a6` — S4-Priority 2 (Startup BVP surface tests): the source module
already exposed all 13 channels with the right blocker IDs after S3R; this
commit adds 13 Sprint 4 surface tests that pin each channel verdict by exact
blocker ID and missing-parameter ID set. 0/13 channels source_supported;
UCSD/Beg KR is labeled `candidate_method_context_not_acceptance` only (UCSD
4.6 kJ DPF wrong scope and no closed BVP initial/boundary data); D2 Townsend
α(E/p), Paschen A/B, and Cu/pyrex/alumina SEE γ are confirmed absent from KR.

`57360f9` — S4-Priority 3 (Transport and closures): five stable Sprint 4
blocker IDs exported as module constants on `closure_packet.py`.
`CLOSURE-BLK-BRAG-001` — Braginskii 1965 PDF on disk but no KR extract;
direct coefficients unavailable as authority. `CLOSURE-BLK-D2-EN-001` — no
KR file for D2 electron-neutral cross-sections. `CLOSURE-BLK-ION-001` — NRL
formulary registered as `nrl_crosscheck_only` (Saha + S(Z) + α_r + α_3)
cannot promote the closure; no non-NRL DPF-regime source. `CLOSURE-BLK-
ANOM-001` — neon Hall/LHDI paper attached as
`candidate_zpinch_formula_not_dpf_authority` with full symbol map and
`dpf_applicability=not_established_no_kr_source`. `CLOSURE-BLK-REST-001` —
restrike appears only as experimental context in KR; no governing equation.
Cross-cutting test confirms no blocker flips
`can_support_first_principles_acceptance`.

`7362251` — S4-Priority 4 (Neutron mechanism authority): three scope-
mismatched method-context labels exported as
`SPRINT4_METHOD_CONTEXT_LABELS` (Talebitaher 2012 NX2 wrong device, Krasa
2008 PF-1000 full-energy wrong scope, Klir 2011 ToF wrong scope), each with
`can_promote_authority=False`. Bosch-Hale 1992 D-D cross-section + reactivity
attached via `bosch_hale_dd_reactivity_ref`
(`KR:bosch-hale-1992-fusion-reactivity.md:59-93,106-109`): upgrades the
thermonuclear channel from `missing_or_blocked` to `inferred_candidate`
(cross-section source-supported) but stays non-accepting because the 1/4
volumetric prefactor is uncited (WP-N6 §4). `NEUTRON-BLK-001` ion
distribution, `NEUTRON-BLK-002` deuteron stopping, `NEUTRON-BLK-003`
beam-target yield (depends on 001+002), `NEUTRON-BLK-004` Brysk Doppler,
`NEUTRON-BLK-005` same-scope anisotropy — all blocked. New negative test:
all three scope-mismatched method contexts combined still cannot promote
`accepted_neutron_authority`.

`7bd6dcb` — S4-Priority 5 (Same-scope comparator decision memo): delivers
the scope decision input as a comparator MATRIX, not prose. `docs/
FIRST_PRINCIPLES_SCOPE_DECISION_MEMO_2026_05_20.md` (9 devices × 9 channels =
81 cells; SUPPORTED 26 / TEXT-ONLY 14 / ABSENT 41). Strongest alternate:
PF-1000 full-energy 27-40 kV (Gribkov/Scholz/Krauz/Malir era), 7/9 channels
SUPPORTED; Te and Ti remain TEXT-ONLY across the entire DPF corpus.
Recommendation: Option B (scope change to PF-1000 full-energy) over Option A
(acquire Akel 16 kV channels, 3-6 month IPPLM campaign that would still
leave Te/Ti as model-only). Supporting doc:
`docs/FIRST_PRINCIPLES_PDF_CORPUS_RESCAN_2026_05_20.md`. `CodexFindings.md`
and `CortexFindings.md` updated.

`da97ed2` — Sprint 4 wrap-up: housekeeping fixes. Regenerated
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.json` for the new
Sprint 4 constants (closure blocker IDs, neutron method-context labels);
strict_passed=true, 290 modules. Applied ruff I001 import-sort to
`tests/test_first_principles_closures.py`. Added `tmp/pdfs/` to `.gitignore`.

### Sprint 4 corpus-rescan KR promotion (2026-05-20)

Promoted two P0 candidate PDFs into KR as `text_parity_extracted_review_needed`:
`bernard1977.pdf` (Bernard 1977 "The Dense Plasma Focus - A High Intensity
Neutron Source", 28 pages) and `plasma-04-00033.pdf` ("Update on the
Scientific Status of the Plasma Focus", 220 pages). New scoped utility
`scripts/promote_corpus_rescan_2026_05_20.py` records the promotion;
`docs/CORPUS_RESCAN_KR_PROMOTION_2026_05_20.{json,md}` capture the ledger.
Source-truth indices regenerated for the 2026-05-20 baseline date.

`022b774` — Sprint 4 corpus-rescan KR promotion (above section): brings
Bernard 1977 and Plasma Focus Update 2021 into KR; refreshes source-truth
indices and source-truth-exhaustion exports for the 2026-05-20 baseline.

### Sprint 4 blocker-resolution audit handoff (2026-05-20)

Seven parallel research sub-agents fanned out across KR + on-disk corpus +
legacy DB to convert every Sprint 4 named blocker (21 distinct items across
geometry, startup, transport/closures, neutron, and comparator) into one of
four definite resolution categories:
- `RESOLVED_VIA_KR`: cathode-cage 200 mm (3 hardware-scope KR sources),
  qualitative DPF anomalous resistivity (LLNL kinetic simulations KR).
- `KR_PROMOTION_RECOMMENDED`: Bennett 2017 on disk
  (filename `schmidt-2017-kinetic-dpf-breakdown.pdf` is mislabeled — actually
  Bennett et al. 2017 Phys. Plasmas 24, 062705) resolves 3 startup channels;
  Braginskii 1965 PDF on disk (Table 2 p.251) resolves CLOSURE-BLK-BRAG-001;
  Talebitaher PhD pp.100-119 resolves NEUTRON-BLK-001 at NX2 scope.
- `EXISTING_KR_TARGET_EXTRACTION_RECOMMENDED`: UCSD/Beg b2e95b88:615-670
  (massf relation, Paschen-regime boundaries); Plasma Focus Update
  pages-0026-0050:512-517 and pages-0126-0150 (Te filter-ratio, beam-target).
- `EXTERNAL_ACQUISITION_REQUIRED`: 11 named citations with DOI/journal/page
  (Raizer 1991; Brysk 1973; ICRU 49; Davidson-Gladd 1975; Bruzzone 2001;
  Voronov 1997; Janev-Smith 1993; Itikawa-Mason 2005; Gribkov 2007;
  Schmidt 2012; Hagstrum 1956; Miklaszewski 2001; etc.).
- `ABSENT_FROM_LITERATURE`: PF-1000 insulator wall thickness, insulator outer
  radius, backplate radial extent, backplate axial thickness (all likely
  IPPLM-internal engineering drawings).
Notable finding: Bernard 1977 contains the first direct Thomson-scattering
Ti = 700 eV measurement of a DPF (filament phase, ~500 kA Limeil/Jülich
Mather-type) — historically significant; does not map to PF-1000 but is the
only direct spectroscopic Ti measurement in the searched corpus. Comparator
matrix recommendation Option B (PF-1000 full-energy 27-40 kV) is reaffirmed;
no cell flips after the new KR ingests. Handoff document:
`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_2026_05_20.md`.

`8f6a0ae` — Audit handoff V1 document (above section) committed.

`7999265` — Housekeeping: regenerate
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}` and
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}` after
the Sprint 4 corpus-rescan additions; add `.gitignore` pattern for the
2026-05-20 rescan intake directory. Periodic audit 10/10 PASS at this HEAD
per `/private/tmp/dpf-unified-audit-logs/20260520T051600Z/summary.md`.

### Sprint 4 audit-handoff V2 (closes the Codex audit corrections)

`docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md`
(Codex audit of V1 at HEAD `7999265`): verdict *conditional accept as a
research triage artifact*; **ten blocking corrections required** before
Sprint 5 consumes the handoff. The audit confirms Bennett 2017 startup
relevance, the PF-1000 cathode-cage 200 mm hardware evidence, qualitative
DPF lower-hybrid anomalous resistivity from the LLNL kinetic-sim KR, and
the unchanged comparator-scope problem.

`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_V2_2026_05_20.md`
incorporates all ten corrections and adopts the required V2 structure:
HEAD reconciliation, per-blocker table (31 rows), separate source-acquisition
table (19 sources), full `KnowledgeReference/...` paths, explicit scope tags,
`accepted_runtime_claim = false` field on every row, reclassification
appendix V1→V2. **V2 supersedes V1.** Key reclassifications: Talebitaher
PhD → `already_target_extracted_in_kr_targets`; Bernard 1977 → `existing_kr_
target_extraction_pending` (not external); Gribkov 2007 J. Phys. D 40:3592
→ `existing_kr_target_extraction_pending` (KR file:
`KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`, not external);
UCSD/Beg `massf` line range corrected from `:615-670` to `:597-601` +
`:631-660` + `:642-660`; Bennett 71 % current-fraction timing corrected
from 500 ns to 1 µs; Braginskii 1965 downgraded to `pdf_present_needs_
rendered_page_or_ocr_verification` because `pdftotext` did not expose
Table 2 / Eqs. 4.30-4.45.

## Changed paths

### Sprint 1

`3dc4c11` — added: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`.

`4424785` — modified: `src/dpf/first_principles/manifest.py`,
`scripts/audit_first_principles_artifacts.py`,
`tests/test_first_principles_manifest.py`,
`tests/test_first_principles_artifact_linter.py`.

`80654b9` — modified: `scripts/verify_first_principles_source_truth_exhaustion.py`,
`scripts/verify_first_principles_module_source_vetting.py`,
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`; added:
`tests/test_first_principles_verification_check_mode.py`.

`b626ad9` — modified: `src/dpf/fields/__init__.py`,
`src/dpf/fields/maxwell_3d.py`, `src/dpf/fields/particle_boundaries.py`,
`src/dpf/first_principles/gv_waveforms.py`,
`tests/test_first_principles_mhd.py`.

`55e3f94` — modified: `src/dpf/first_principles/segmented_whole_shot.py`,
`tests/test_first_principles_segmented_whole_shot.py`.

`1abe15a` — renamed (quarantine): `results/audit_first_principles_3d_smoke.json`,
`results/audit_experimental_whole_shot_smoke.json`,
`results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json` →
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/`; added:
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/QUARANTINE_NOTICE.md`.

`bf22c33` — modified: `.github/workflows/ci.yml`.

`fe038f7` — modified: `docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.csv`, `docs/SRS_TRACEABILITY_MATRIX.json`,
`docs/FIRST_PRINCIPLES_EXPERIMENTAL_SIMULATOR_SPRINT_PLAN_2026_05_18.md`,
`tests/test_srs_traceability_export.py`.

`f07d4ba` — added all files under
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/`
*(documentation-only wrapper commit for Sprint 1 packet)*.

`fa9088e` — modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/AUDIT_COMMANDS.md`
(pinned `--date 2026_05_18` on the two verification-gate commands so the
read-only check resolves to the committed baseline when re-run on any future
date) *(documentation-only)*.

`06cfd64` — added:
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`.

### Sprint 1.1 — Hygiene

`9df8d3b` — style: repository-wide ruff cleanup (RC-2); 52 files changed across
`src/dpf/` and `tests/`; also regenerated
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

`49e80ee` — docs(kr): ingested verified Auluck 2021 equations; modified:
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.{json,md}`.

`bd840f4` — docs: Sprint 2 proposals; modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv`,
`CLAIMS_LEDGER.csv`, `README.md`, `SOURCE_PACKET_INDEX.csv`;
removed `sprint_2/PENDING.md` (placeholder superseded by proposal files);
added six `sprint_2/WP_N1B_*` and `sprint_2/WP_N4B_*` proposal docs.

`0585eec` — docs: tracked copy of verified Auluck extract; modified:
`SOURCE_PACKET_INDEX.csv`; added:
`sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md`.

`0bef78c` — fix(first-principles): artifact linter C8 + package schema tests
(RC-3/5/6/7); modified: `scripts/audit_first_principles_artifacts.py`,
`tests/test_first_principles_artifact_linter.py`; added:
`tests/test_external_team_submission_package.py`.

`c52bed3` — docs: finalized Sprint 1.1 (RC-4); modified:
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/AUDIT_COMMANDS.md`,
`BLOCKER_MATRIX.csv` *(documentation-only wrapper commit; RC-2..RC-7 rows closed)*.

### Sprint 2 Audit and Periodic Runner

`6af4454` — docs: added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_PACKET_2026_05_19.md`,
`scripts/run_codex_periodic_audit.py`.

### Sprint 2 Implementation — WP-N1B / WP-N4B

`4b080eb` — feat: WP-N1B six-term Auluck power-port ledger; modified
`src/dpf/first_principles/power_port.py`,
`tests/test_first_principles_power_port.py`, and
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`
(module-source-vetting regenerated after the first-principles module edit).

`4c8dac1` — feat: WP-N4B cross-restart ledger merge and artifact combiner;
added `src/dpf/first_principles/segmented_whole_shot_combine.py` and
`tests/test_first_principles_segmented_whole_shot_combine.py`.

`93620ba` — docs: Sprint 2 packet hygiene (F2/F3/F4); modified
`THREE_SPRINT_FINAL_SUMMARY.md`, `CHANGELOG.md`, `PATCH_SCOPE.md`,
`UNKNOWN_AND_INFERENCE_LOG.md` under the packet, and
`tests/test_external_team_submission_package.py`.

`092871b` — docs: changelog catch-up for WP-N1B/WP-N4B and HEAD-exempt
changelog test; modified `CHANGELOG.md` and
`tests/test_external_team_submission_package.py`.

`ada801c` — docs: add Sprint 2 implementation audit and next directions
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_IMPLEMENTATION_2026_05_19.md`);
documentation-only.

### Sprint 2.1 Implementation Pass — F1/F3/F4/F5

`65c477f` — feat: compute Auluck power-port terms I and III, drop electrode_work
wording (F1/F3/F5); modified `src/dpf/first_principles/power_port.py`,
`src/dpf/fields/hybrid_simulator.py`, `src/dpf/fields/hybrid_stepper.py`,
`tests/test_first_principles_power_port.py`, and
`docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

`21d4e07` — fix: harden the WP-N4B combiner with cross-restart ledger
invariants (F4); modified
`src/dpf/first_principles/segmented_whole_shot_combine.py`,
`tests/test_first_principles_segmented_whole_shot_combine.py`, and
`sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`.

`07fe76a` — docs: synchronize SRS/RTM and packet traceability after
WP-N1B/WP-N4B implementation (F2); modified
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`, and packet traceability files.

`9e1407d` — docs: add Codex Sprint 2 follow-up implementation audit and next
directions; added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`.

`e984176` — docs: update Sprint 2 audit changelog after the follow-up audit;
modified this `CHANGELOG.md` so the packet consistency gate covered the Sprint
2 traceability and audit commits.

`4e3987b` — docs: add parallel work directions to the Sprint 2 follow-up audit;
modified
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md` with
allowed non-conflicting Sprint 3 research/spec lanes and audit acceptance
rules.

`6d015c4` — docs: update changelog for parallel directions; modified this
`CHANGELOG.md` so the packet consistency gate covered the parallel-directions
documentation commit.

`df71d99` — docs: add Sprint 3 WP-N2/N3/N4/N5/N6/N7 research and spec packets;
added source-backed packet drafts under
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/`.

`173e10d` — fix(first-principles): close Sprint 2.2 combiner and traceability
gaps; modified `src/dpf/first_principles/segmented_whole_shot_combine.py`,
`tests/test_first_principles_segmented_whole_shot_combine.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`, `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`,
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`,
and `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT2_FOLLOWUP_2026_05_19.md`.

`d44601f` — docs: audit Sprint 3 research packets; added
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_PACKET_2026_05_19.md` and updated
this `CHANGELOG.md` with the Sprint 2.2 closure commit.

`0b8fa97` — docs: add sprint 3 completion handoff; added
`docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`.

### Sprint 3 Implementation — S3.1–S3.8

| Commit | Subject |
| --- | --- |
| `100d87d` | docs: S3.1 packet hygiene — status ledger, citation normalization, consistency tests |
| `0e91f08` | feat(first-principles): S3.2/S3.3 source-tagged PF-1000 geometry packet and Sigma_p plumbing |
| `06744fd` | feat(first-principles): S3.4 typed startup BVP packet (fail-closed) |
| `7dd1994` | feat(first-principles): S3.5 closure registry and PlasmaPy regime gate |
| `d1dc17c` | feat(first-principles): S3.6 mechanism-separated neutron authority packet |
| `6660eb9` | feat(first-principles): S3.7/S3.8 numerical acceptance gates and certificate scaffold |
| `f7bb9f8` | chore(first-principles): regenerate module-source-vetting after Sprint 3 implementation |

### S3.1 Implementation — packet hygiene (`100d87d`)

`100d87d` — docs: S3.1 packet hygiene (status ledger, citation normalization, consistency
tests); modified
`docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/SPRINT_3_STATUS_LEDGER.md`
and related sprint_3 packet docs; added or updated S3.1 consistency tests in
`tests/test_external_team_submission_package.py`.

### S3.2/S3.3 Implementation — geometry packet and Sigma_p plumbing (`0e91f08`)

`0e91f08` — feat: S3.2/S3.3 source-tagged PF-1000 geometry packet and Sigma_p plumbing;
added/modified `src/dpf/fields/source_geometry.py` (`PF1000GeometryPacket`,
`PF1000GeometryField`, `PF1000GeometryConflict`, `PF1000MaskManifest`,
`SigmaPSurfacePacket`; geometry: 5 source-supported, 4 conflict, 10 blocked fields;
Sigma_p packet fail-closed; Auluck terms II/IV/V/VI stay fail-closed;
I/III computed); modified `tests/test_source_geometry_packet.py`,
`tests/test_first_principles_geometry.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.4 Implementation — startup BVP packet (`06744fd`)

`06744fd` — feat: S3.4 typed startup BVP packet (fail-closed); added
`src/dpf/first_principles/startup_bvp.py` (13-channel `StartupBVPPacket`,
0 source-supported, all candidate/blocked; startup authority blocked);
modified `tests/test_first_principles_startup_bvp.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.5 Implementation — closure registry and PlasmaPy regime gate (`7dd1994`)

`7dd1994` — feat: S3.5 closure registry and PlasmaPy regime gate; added/modified
`src/dpf/first_principles/closure_packet.py` (12 closures + 2 sub-closures;
1 active_source_backed_candidate, rest blocked/not-simulated;
`PhysicsClosurePacket` embedding); modified `tests/test_first_principles_closures.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.6 Implementation — neutron authority packet (`d1dc17c`)

`d1dc17c` — feat: S3.6 mechanism-separated neutron authority packet; added/modified
`src/dpf/first_principles/neutron_authority.py` (`NeutronAuthorityPacket`,
10 channels; 5 candidate, 4-5 blocked; scalar yield comparator-only);
modified `tests/test_first_principles_neutron_authority.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.7/S3.8 Implementation — numerical acceptance gates and certificate scaffold (`6660eb9`)

`6660eb9` — feat: S3.7/S3.8 numerical acceptance gates and certificate scaffold;
modified `src/dpf/first_principles/segmented_whole_shot.py` (extended cumulative
ledgers), `src/dpf/first_principles/certificate_gate.py` (certificate scaffold with
all channels missing — no accepted certificate); added/modified
`tests/test_first_principles_long_run_integrity.py`,
`tests/test_first_principles_certificate_negative_controls.py`,
`docs/DPF_REQUIREMENTS_BASELINE.md`,
`docs/SRS_TRACEABILITY_MATRIX.{csv,json}`.

### S3.9 Traceability — module-source-vetting regeneration (`f7bb9f8`)

`f7bb9f8` — chore: regenerate module-source-vetting after Sprint 3 implementation;
modified `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`.

### Sprint 4 audit-handoff V2 and normalized blocker ledgers (`8e6b5e9`, `85a7f05`)

`8e6b5e9` — docs(first-principles): Sprint 4 audit-handoff V2 -- 10 Codex
corrections applied; V1 superseded. Added
`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_V2_2026_05_20.md` and
kept all first-principles runtime, validation, neutron-authority, startup-BVP,
and transport-closure claims fail-closed.

`85a7f05` — docs(first-principles): normalize V2 blocker handoff ledger. Added
`docs/CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md`,
`docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`,
`docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`, and
`tests/test_first_principles_v2_handoff_ledgers.py`; updated `CodexFindings.md`
and `CortexFindings.md`. Codex audit verdict: **V2 accepted as controlling
errata for source triage**; findings A1-A5 closed by this normalization patch.

`4cb260c` — docs(first-principles): add physics acceptance promotion protocol.
Adds `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_PROMOTION_PROTOCOL_2026_05_20.md`
and `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv` (the
machine-readable gate ledger that requires triple verification: other-team
pass + Codex pass + executable gate pass at the same commit before any row
can become `accepted_physics_module`). Adds
`tests/test_first_principles_physics_acceptance_protocol.py` enforcing the
triple-verification invariant. No physics is accepted by adding these
artifacts — they define the only future promotion path.

`b287a66` — style(tests): fix ruff I001 + B905 in V2 handoff and
acceptance-protocol ledger tests. Two new test files (`test_first_principles_
v2_handoff_ledgers.py`, `test_first_principles_physics_acceptance_protocol.py`)
introduced by 85a7f05 and 4cb260c tripped the periodic-audit `ruff_src_tests`
gate with 4 errors; ruff --fix resolved the I001 import-sort and explicit
`strict=True` was added to the two `zip(header, row)` sites (each is
preceded by an `assert not bad_rows` length guard, so `strict=True` is the
correct semantic).

### Sprint 5 WS2 target extractions + x-ray inconsistency fixes

Adds the seven Sprint 5 Workstream-2 target-extraction packets specified by
`docs/CODEX_FIRST_PRINCIPLES_V2_HANDOFF_AUDIT_AND_NEXT_PLAN_2026_05_20.md`:
Bennett 2017 (kinetic DPF breakdown, mislabeled PDF — actual authors are
Bennett et al.); Braginskii 1965 (Table 2 at journal p.251 / PDF p.26
**RENDER-VERIFIED** via the Read-tool PDF page renderer with 13 spot-checked
Z=1/2/3/4/inf coefficients; closes the Codex V1 row 8 downgrade); Scholz/
Gribkov 2007 Part II (PF-1000 full-energy fast-deuteron, Y0/Y90 = 1.8 at
shot 3121 / 35 kV / 0.810 MJ); Bernard 1977 (filament-phase Ti = 700 eV
historical; thermonuclear 1/4 prefactor **VERIFIED NOT FOUND**); UCSD/Beg
current-sheath (audit row 6 line-range corrections confirmed: massf at
`:597-601`, Paschen regimes at `:631-640`, Te ≈ 4 eV at `:642-644`,
Liz/Li = 2.4 at `:654-660`); Stepniewski 2004 (formal review; stays blocked
as simulation-parameter context); Plasma Focus Update 2021 (third hardware
source confirming cathode-cage 200 mm; Te = 7.5 keV filter-ratio text-only
with strong method caveats; **audit correction**: 320/500 keV deuteron is
FF-1 / Lerner, NOT PF-1000). All 7 packets carry
`accepted_runtime_claim = false` and `can_support_first_principles_
acceptance = false`. 17 new tests pass. X-ray inconsistency fixes:
`closure_packet.py:402-413` corrected "Table 1" → "Table 2" with render-
verified evidence cite; `neutron_authority.py:16` stale WP-N6 mechanism-map
counts replaced by an authoritative pointer to the V2 blocker-resolution
ledger and the Sprint 4 P4 Bosch-Hale / scope-mismatch label additions;
`startup_bvp.py:15-21` stale "11 channels" replaced by "13 channels
(CH01-CH13)" with the Sprint 5 numeric targets cited. New memo
`docs/SPRINT5_FREE_ACQUISITIONS_2026_05_20.md` lists six free-download or
free-substitute paths for V2 P1/P2 external-acquisition blockers
(Bruzzone ×2 + Miklaszewski 2001 ICHTJ open-access; LXCat D₂; SRIM-2013
deuteron stopping; OSTI/Munro Brysk re-derivation; PlasmaPy Braginskii
cross-check) — all URLs HTTP-200-verified during the x-ray pass.

`82694e7` — Sprint 5 WS2 commit (above section): 7 target-extraction
packets + 17 enforcing tests + 3 inline-docstring x-ray fixes + free-
acquisition memo. Module-source-vetting docs regenerated by the
immediately-following commit `8fba1bf` to account for the new
`src/dpf/first_principles/sprint5_target_extractions.py` module
(291 modules, `strict_passed=true`).

`8fba1bf` — chore: regenerate module-source-vetting after Sprint 5 WS2
addition.

`558de6f` — docs(s5): CHANGELOG sync entries for 82694e7 + 8fba1bf so
`test_changelog_covers_all_commits_since_base` stays green.

### Codex Sprint 5 WS2 audit A1-A4 corrections

`docs/CODEX_SPRINT5_WS2_AUDIT_2026_05_20.md` (Codex audit of Sprint 5 WS2
at HEAD `558de6f`) **ACCEPTED Sprint 5 WS2** as a fail-closed source-
availability and target-extraction pass, with four bookkeeping
corrections required before WS3:

- **A1** Bennett packet CH01 mapping ambiguity: the per-target
  `fill_pressure_baseline.resolves` field listed `STARTUP-BVP-CH01` while
  the top-level `resolves_blockers` did not. Corrected by setting the
  per-target `resolves = ()`, adding `corroborative_only = True` and
  `corroborative_for = ("STARTUP-BVP-CH01",)`, and adding a structural
  test `test_sprint5_audit_a1_per_target_resolves_subset_of_top_level`
  that asserts every per-target `resolves` is a subset of the packet's
  top-level `resolves_blockers` (corroborative-only targets must have
  empty `resolves`).
- **A2** Te/Ti broad wording: the Sprint 5 free-acquisition memo said
  "no DPF in any literature publishes pinch-phase spectroscopic Te/Ti."
  That overstates the field-wide claim. Replaced with the narrow Codex-
  accepted statement: "no accepted same-scope PF-1000 bulk pinch Te/Ti
  history exists for the selected certificate scope," plus explicit
  acknowledgement that Bernard 1977 contains direct historical
  filament-phase Ti (wrong-scope) and Plasma Focus Update 2021 contains
  PF-1000 local hot-spot Te method context (text-only). Doc-lint test
  `test_sprint5_audit_a2_free_acquisitions_memo_no_broad_te_ti_wording`
  rejects the broad phrasing.
- **A3** "Closes blockers" overclaim: free-download URLs do not by
  themselves close blockers. Replaced "closes" language with "may close
  source availability after acquisition, KR ingestion, target
  extraction, and review" throughout the memo. SRIM-2013, Munro 2012,
  and PlasmaPy are explicitly relabeled as **substitute / cross-check
  lanes** pending source-equivalence review. Doc-lint test
  `test_sprint5_audit_a3_free_acquisitions_memo_softens_closes_language`
  enforces the narrow qualifier.
- **A4** Stale 74-vs-72 commit count: prior memory entry said "74
  unpushed"; `git rev-list --count origin/codex/corpus..HEAD` reports
  72 at HEAD `558de6f`. Memory entry corrected; no doc artifact carried
  the 74 claim.

No physics is accepted by these corrections; `accepted_runtime_claim`
remains false on every blocker-ledger row;
`can_support_first_principles_acceptance` remains false everywhere.
The structural blockers (3D-runner-vs-acceptance-gate code gap;
`same_scope.py` forcing Te/Ti as blocking) remain open exactly as the
Codex audit's "Structural Blockers Remain" section instructs.

### Sprint 6 — Convert source leads into fail-closed KR evidence packets (2026-05-20)

`97ebd94` — fix(s5-ws2): Codex Sprint 5 WS2 audit A1-A4 corrections —
Bennett CH01 corroborative-only, Te/Ti narrow wording, "closes" → "may
close source availability after acquisition + KR ingestion + target
extraction + review", commit-count drift fix.

`5719a69` — chore(audit): add the Codex-Claude dual-audit runner
(`scripts/run_codex_claude_dual_audit.py` + 414-line implementation)
and the supporting `docs/CODEX_CLAUDE_DUAL_AUDIT_AUTOMATION_2026_05_20.md`
automation memo. No physics change; tooling only.

The Sprint 6 main commit (next entry, hash assigned at commit time)
delivers, per the Sprint 6 goal:

- **WS1** Acquired three free Nukleonika PDFs (Bruzzone & Bernal 2001
  v46n2p059, Bruzzone 2001 v46s1p003, Szydłowski/Miklaszewski 2001
  v46s1p061) into
  `downloaded_books_papers/Research Papers/2026-05-20-sprint6-acquisitions/`
  with verified SHA-256 hashes; ledgered in
  `docs/SPRINT6_KR_PROMOTION_2026_05_20.{json,md}`. Intake directory
  added to `.gitignore`.
- **WS2** KR text-parity ingestion via the new
  `scripts/promote_sprint6_acquisitions_2026_05_20.py`. All three KR
  records carry
  `status = text_parity_extracted_review_needed`,
  `validation_status = source_available_not_target_extracted`, and
  `accepted_runtime_claim = false` in their `kr_ingestion` metadata.
  PyMuPDF was installed in `.venv312` as a development-time dependency
  for KR ingestion + render verification (not used by any runtime
  physics path).
- **WS3** Braginskii Table 2 target-extracted KR packet:
  human-readable extraction at
  `docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md`
  + render evidence at
  `docs/extractions/braginskii_1965_render_evidence/` (4 PNG pages at
  200 dpi via PyMuPDF + per-page SHA-256 manifest) + Python module
  `src/dpf/first_principles/sprint6_braginskii_table2_target_extraction.py`
  + 14 enforcing tests in
  `tests/test_sprint6_braginskii_table2.py`. Status transition recorded:
  `CLOSURE-BLK-BRAG-001` from
  `pdf_present_needs_rendered_page_or_ocr_verification` (Codex V1 row 8)
  to
  `target_extracted_source_supported_pending_runtime_consumption_and_review`.
  PDF p.26 = journal pp.250-251 spread (2-up scanned); Table 2 on the
  right half. Z=1 and Z=∞ columns two-pass verified; Z=2/3/4 cells
  render-visible but five cells explicitly flagged
  `review_required` for re-verification at consumption time. Runtime
  acceptance NOT granted.
- **WS4** Four source-equivalence review packets under
  `docs/source_equivalence_review/`: LXCat (D₂ Townsend / e-neutral),
  SRIM/NIST/IAEA (deuteron stopping), Munro 2012 (Brysk Doppler
  re-derivation), PlasmaPy `formulary.braginskii` (cross-check only).
  **No source-equivalence granted.** Each packet documents the
  substitution argument FOR/AGAINST and lists pre-conditions for a
  future review-session acceptance vote.
- **WS5** Two structural-code-sprint design memos (NOT implementations):
  `docs/SPRINT6_WS5_PACKAGE_NATIVE_3D_ACCEPTANCE_CONTRACT_DESIGN_2026_05_20.md`
  proposes a named 3D acceptance contract function dispatching from the
  existing readiness gate;
  `docs/SPRINT6_WS5_CLAIM_LIMITED_TE_TI_OBSERVABLE_EXCLUSION_DESIGN_2026_05_20.md`
  proposes an `ObservableExclusion` per-channel state (Te + Ti only)
  with certificate-text + reviewer signoff requirements — explicitly
  NOT a generic `caveat_accepted` lane, per the Codex Sprint 5 WS2
  audit's structural-blockers guidance.
- **DoD** Dual Codex-Claude audit packet at
  `docs/SPRINT6_DUAL_CODEX_CLAUDE_AUDIT_PACKET_2026_05_20.md` records
  both lanes (Codex directives carried forward + Claude implementation
  evidence) and the convergence table (both lanes agree across 6
  questions).

`accepted_runtime_claim` and `can_support_first_principles_acceptance`
remain `false` everywhere. No runtime physics is accepted by Sprint 6.
The user's parallel automation track for user-supplied papers
(`USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.{json,md}` and
`sprint6_user_target_extractions.py`) is a separate intake stream and
is handled by its own commit chain.

`7a34d0b` — Sprint 6 main commit (above section). A follow-up commit
regenerates `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`,
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`, and
`docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.{json,md}` so the new Sprint 6
KR files (Bruzzone ×2, Miklaszewski 2001, Braginskii Table 2 target-
extracted) are indexed and the periodic-audit `module_source_vetting` +
`source_truth_exhaustion` gates pass with `strict_passed=true` and
`open_issue_count=0` respectively (293 modules total, up from 291).

### Sprint 7 — first-principles runtime contract (2026-05-20)

`b176c47` — Sprint 6 vetting/source-truth regeneration (above section).

Sprint 7 commit `35bb1a9` delivers, per
`docs/SPRINT7_FIRST_PRINCIPLES_RUNTIME_CONTRACT_INSTRUCTIONS_2026_05_20.md`,
five workstreams plus the Sprint 6 user-supplied-papers intake baseline
that Sprint 7 WS-A consumes (the two tracks are interleaved in the same
runtime files and committed together):

- **Sprint 6 user-supplied intake (WS-A input):** 9 user-supplied PDFs
  recorded in `docs/USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.{json,md}` (0
  promoted / 9 skipped-existing / 0 failed) via
  `scripts/promote_user_supplied_papers_2026_05_20.py`;
  `src/dpf/first_principles/sprint6_user_target_extractions.py` +
  `tests/test_sprint6_user_supplied_extractions.py`. Render evidence for
  Scholz 2000/1999, Herold 1989, Shakya 2015 under `docs/extractions/`.
- **WS-A Source-Ledger Closure:** all 9 intake records have a matching
  source-acquisition-ledger row (31 rows, 0 dup `source_id`);
  `PF1000-BLK-015` blocker status is `existing_kr_source_supported`;
  Bruzzone/Bernal partial pair split. `tests/test_first_principles_v2_
  handoff_ledgers.py` strengthened (34 tests).
- **WS-B Revision-Scoped Geometry:** new
  `PF1000GeometryPacket.scholz_2001_24rod_large_electrode` constructor —
  10 source-supported fields cited to
  `recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98` and
  `pf-1000-device-a2d6bc15.md`; bore length / insulator wall thickness /
  backplate dims stay blocked; Akel + Krauz constructors unchanged and
  proven not to inherit Scholz 2001 dimensions (7 non-inheritance tests).
- **WS-C Package-Native 3-D Acceptance Contract:** `hybrid_pic_3d_readiness`
  is a first-class telemetry packet from `dpf.validation.hybrid_pic_3d`
  (NOT the cylindrical `first_principles_mhd.py` gate), surfacing in
  runtime telemetry, manifest candidate evidence, CLI `telemetry_packets`,
  and `validation_packet`. 4 negative tests prove acceptance cannot be
  produced by candidate-only records, missing top-level keys, wrong
  backend labels, or missing same-scope 3-D validation.
- **WS-D Same-Scope Te/Ti Rejection:** `same_scope.py` accepts
  `electron_temperature_history` / `ion_temperature_or_distribution_history`
  only from direct same-scope diagnostics with review + uncertainty;
  manual `accepted_same_scope_channels` injection rejected; **no generic
  `caveat_accepted` lane** (Codex Sprint 5 WS2 audit constraint honored).
  4 dedicated negative controls.
- **WS-E Next Physics Sources:**
  `docs/extractions/SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md`
  re-audits Braginskii Table 2 (PDF + render SHA-256 re-confirmed) and
  Bennett 2017 (14 verbatim targets re-verified against the PDF pages);
  LXCat / SRIM / Munro / PlasmaPy stay `source_equivalence_granted=false`
  review-queue lanes. NO runtime coefficient wired.
- **Deliverable audit memo:**
  `docs/SPRINT7_RUNTIME_CONTRACT_AUDIT_MEMO_2026_05_20.md`.

`accepted_runtime_claim` and `can_support_first_principles_acceptance`
remain `false` everywhere. Sprint 7 is not a validation sprint and produces
no engineering-firm certificate. 233 focused tests pass; ruff clean.

### Sprint 8 — super-sprint source-to-runtime, Phase A P0 (2026-05-20)

`35bb1a9` — Sprint 7 first-principles runtime contract (above section).

Sprint 8 Phase A commit `bd5be3a` delivers the three P0 workstreams of
`docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md`:

- **WS0 Ledger / KR / Traceability Repair:** Bennett 2017 corrected to
  `on_disk_line_page_verified_kr_promotion_required` (4 STARTUP-BVP rows);
  `CLOSURE-BLK-BRAG-001` corrected to
  `target_extracted_source_supported_pending_equation_extraction_and_review`
  (Eqs. 4.30-4.45 and the five flagged Table-2 cells stay blocked);
  `SAME-SCOPE-COMPARATOR-DECISION` reclassified to
  `scope_governance_decision_pending` (control-plane governance, not KR
  authority); SRS RTM CSV/JSON regenerated (DPF-PHYS-014/022 drift fixed);
  source-truth index refreshed (`exhausted=true`, 0 open issues); dated
  findings-tail entries added to `CodexFindings.md` / `CortexFindings.md`;
  ledger commit pins moved to a per-row scheme (no stale global `8e6b5e9`).
- **WS1 Shared Acceptance Channel Contract:** new
  `src/dpf/first_principles/channel_state.py` defines exactly seven canonical
  channel states (`accepted`, `blocked_missing_source`, `blocked_wrong_scope`,
  `blocked_missing_review`, `blocked_missing_uncertainty`,
  `excluded_not_validated`, `not_claimed`), shared by `same_scope.py`,
  `numerical_fidelity.py`, and `certificate_gate.py`. Manual same-scope
  channel lists are demoted to *requested* channels — never accepted evidence
  without a reviewed, scope-matched target with uncertainty. The legacy
  cylindrical `first_principles_mhd.py` gate now refuses to judge a
  package-native 3-D run (it defers to the `hybrid_pic_3d` gate).
- **WS2 Runtime-Demonstrator Scope Lock:** Option B (PF-1000 full-energy
  27-40 kV) encoded as a control-plane scope packet in
  `src/dpf/first_principles/runtime_demonstrator_scope.py`
  (`is_scientific_authority=false`, `governance_class=control_plane`) with a
  single canonical scope label `pf1000_full_energy_27_to_40_kv` and an
  in-scope / context-only / wrong-scope source classification. Governance
  memo: `docs/SPRINT8_WS2_RUNTIME_DEMONSTRATOR_SCOPE_LOCK_2026_05_20.md`.

`accepted_runtime_claim` and `can_support_first_principles_acceptance` remain
`false` everywhere. Sprint 8 Phase A is not a validation sprint. 305 focused
tests pass; ruff clean.

### Sprint 8 — super-sprint source-to-runtime, Phase B+C P1/P2 (2026-05-20)

`bd5be3a` — Sprint 8 Phase A P0 foundation (above section).

Sprint 8 Phase B+C commit `b270cb5` delivers the four P1 and two P2
workstreams of
`docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md`:

- **WS3 PF-1000 Geometry Source-To-Runtime:** new engineering-candidate deck
  `pf1000_scholz_2001_24rod_full_energy_deck` in
  `src/dpf/first_principles/deck.py`, scope-tagged
  `pf1000_full_energy_27_to_40_kv`, consuming
  `PF1000GeometryPacket.scholz_2001_24rod_large_electrode()`. Five fields stay
  explicitly blocked (anode hollow-bore length, insulator wall thickness,
  backplate radial extent, backplate axial thickness, same-scope reviewed
  mask); the deck declares no hollow anode.
- **WS4 Bennett Startup BVP Consumption:** Bennett 2017 promoted to canonical
  KR markdown (gitignored local); new typed packet
  `sprint8_bennett_startup_target_extraction.py` target-extracts CH03/04/07/08
  with enforced SI unit conversions. The four channels become source-backed
  runtime candidate channels; their same-scope status for the demonstrator is
  `blocked_wrong_scope` (Bennett is wrong-scope per WS2). CH01/02/05/06/09-13
  stay blocked or wrong-scope. Source-ledger `bennett_2017_startup`
  `already_in_kr` false -> true.
- **WS5 Braginskii Z=1 Transport Candidate:** Braginskii 1965 Eqs. 4.30-4.45
  render-verified (PyMuPDF, journal pp.249-253); new
  `sprint8_braginskii_z1_transport.py` carries the Z=1 resistivity and
  electron/ion parallel thermal conductivity; candidate closure wired through
  `closure_packet.py`; PlasmaPy cross-check agrees within 0.36 %. The five
  review-required Table-2 cells stay unavailable. `CLOSURE-BLK-BRAG-001` ->
  `equations_4_30_to_4_45_render_verified_z1_transport_wired_as_candidate_acceptance_blocked`.
- **WS6 Power-Port and Sigma-p Operator Ledger:** `power_port.py` gains an
  explicit six-term Auluck eq.(6) presence roster; active-load placeholders
  demoted to `excluded_not_validated` engineering-only telemetry; new
  `sign_convention` / `time_centering` / `domain` / `residual` fields; Sigma /
  quasi-TEM line-voltage recorded as not-source-verified. Terms II/IV/V/VI
  fail closed pending Sigma-p geometry.
- **WS7 3-D Runtime Ratchet:** `experimental-segmented-whole-shot` gains
  `--dt-policy` / `--vacuum-cfl` / `--auto-step-budget` / `--max-auto-steps`
  parity with `experimental-whole-shot`; new `combine-whole-run` CLI route;
  the WS3 24-rod deck exposed as preset `pf1000_scholz_2001_24rod_full_energy`;
  engineering-candidate run plan
  `docs/SPRINT8_WS7_ENGINEERING_CANDIDATE_3D_RUN_PLAN_2026_05_20.md`.
- **WS8 External Source Queue:** nine source packets in
  `docs/SPRINT8_WS8_EXTERNAL_SOURCE_QUEUE_2026_05_20.{md,json}`; nothing
  acquired to disk, nothing KR-ingested, nothing wired.
- **WS0 Phase B integration:** ledger deltas applied, three-tier per-row
  commit pins (`8e6b5e9` / `35bb1a9` / `bd5be3a`), RTM regenerated (no drift),
  source-truth index refreshed (`exhausted=true`), module-source vetting
  `strict_passed=true` (297 modules), stale `CLOSURE_BLK_BRAG_001` constant
  corrected.
- **Deliverable audit memo:**
  `docs/SPRINT8_SUPER_SPRINT_AUDIT_MEMO_2026_05_20.md`.

`accepted_runtime_claim` and `can_support_first_principles_acceptance` remain
`false` everywhere. Bennett startup and Braginskii Z=1 transport advanced from
blocked to source-backed runtime candidate — engineering evidence only, no
acceptance. 724 focused tests pass; ruff `src/ tests/` clean.

### Sprint 8 — Phase B+C audit follow-up (2026-05-20)

`b270cb5` — Sprint 8 Phase B+C P1/P2 (above section).

The Codex periodic audit auto-detects the latest dated baseline. Phase B
regenerated the `2026_05_18`-stamped source-truth artifacts, but the audited
latest stamp is `2026_05_20`. This follow-up commit regenerates
`FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_20.{json,md}`
(`exhausted=true`, `open_issue_count=0`) and corrects the
`FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.{json,md}` `Generated` date stamp to
`2026_05_20` (index records byte-identical). After this commit the audit's
`source_truth_exhaustion` and `module_source_vetting` `--check` gates pass at
baseline `2026_05_20`. The only remaining audit failure is `git_status_clean`,
caused solely by 145 pre-existing PDF symlink type-changes unrelated to any
Sprint 8 work.

## Notes on CHANGELOG conventions

The final commit of any pass that updates this `CHANGELOG.md` is structurally
HEAD-exempt from the `test_changelog_covers_all_commits_since_base` check
(`76480b0..HEAD~1`), because a commit cannot list its own hash before it exists.
Every non-HEAD commit in the range must be listed here.

## Deleted / quarantined artifacts

Three audit smoke artifacts were moved (not deleted) to
`results/archive_stale_pre_ssr_codex_a2_2026_05_18/` and marked quarantined. They
embed `artifact_generation_commit 466a0a5` and `dirty_worktree true`. See
`sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`.

## Generated artifacts

No new runtime result artifacts were generated. There are no active
first-principles result artifacts at HEAD `fe038f7`; the audit step "active
artifact generation commit equals HEAD" is satisfied because the active set is
empty.

The four dated source-truth docs
(`FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.{json,md}`,
`FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}`) were regenerated
once into deterministic form (A-3); the SRS RTM exports
(`SRS_TRACEABILITY_MATRIX.{csv,json}`) were regenerated from the baseline (A-7).
