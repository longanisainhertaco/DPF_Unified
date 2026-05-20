# Codex Audit And Next Plan - V2 Blocker-Resolution Handoff - 2026-05-20

Audited input:

- `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_HANDOFF_V2_2026_05_20.md`
- HEAD: `8e6b5e9`
- Prior audit source:
  `docs/CODEX_FIRST_PRINCIPLES_BLOCKER_RESOLUTION_AUDIT_2026_05_20.md`
- Source authority for physics claims: repo-local `KnowledgeReference/` only.

Verification run:

- `.venv312/bin/python -m pytest tests/test_external_team_submission_package.py -q`
- Result: `29 passed`
- `git diff --check HEAD~1 HEAD`: pass

Resolution update after this audit:

- The five bookkeeping findings A1-A5 are closed by the V2 normalization patch.
- `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv` is now the
  normative 31-row blocker ledger with uniform fields and fail-closed runtime
  flags.
- `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv` is now the
  normative 23-row source-acquisition ledger.
- `tests/test_first_principles_v2_handoff_ledgers.py` enforces row counts,
  status distribution, 12 true P1/P2 external acquisition rows, and false
  acceptance flags.

## Verdict

**V2 is accepted as the controlling errata for source triage, with bookkeeping
corrections below now fixed by the normalized ledgers.**

The V2 handoff fixed the high-risk scientific-state errors from V1:
Talebitaher, Bernard 1977, and Gribkov Part II are no longer treated as new
external acquisitions; the UCSD/Beg `massf` line range is corrected; Bennett
2017 timing is corrected; Braginskii 1965 is downgraded to rendered/OCR review
before extraction; and the handoff preserves `accepted_runtime_claim = false`.

The remaining issues are process/accounting defects, not scientific
acceptance defects. They should be fixed before the packet is converted into a
machine-readable implementation ledger, but they do **not** require another
research round before the next sprint starts.

No runtime, validation, engineering-firm-ready, whole-shot, neutron-authority,
startup-BVP, or transport-closure claim is accepted by this audit.

## Original V2 Issues Fixed By Ledger Patch

The following were the defects found during the audit. They are retained here
as audit history; the resolution update above identifies the controlling fixed
artifacts.

### A1 - Domain Count Wording Still Miscounts

V2 line 51 says the 31 blockers are:

`6 geometry + 13 startup-BVP + 5 transport/closures + 6 neutron + 1 thermonuclear-prefactor + 1 comparator`.

That arithmetic is 32, not 31. The actual table structure is 31 because the
thermonuclear `1/4` prefactor is one of the six neutron rows.

Required correction:

- Replace the wording with:
  `6 geometry + 13 startup-BVP + 5 transport/closures + 6 neutron/mechanism rows including the thermonuclear prefactor + 1 same-scope comparator decision = 31`.

### A2 - Source-Acquisition Row Count Is Still Wrong

V2 lines 52 and 173 say the source-acquisition table has 19 rows/sources. The
visible table at lines 149-171 contains 23 rows:

1. Bennett 2017
2. Braginskii 1965
3. Raizer 1991
4. NIST LXCat
5. Brysk 1973
6. ICRU Report 49
7. Davidson & Gladd 1975
8. Bruzzone & Bernal 2001 / Bruzzone 2001
9. Scholz / Gribkov 2007 Part II
10. Voronov 1997
11. Janev & Smith 1993
12. Itikawa & Mason 2005
13. Schmidt et al. 2012 PRL
14. Hagstrum 1956
15. Miklaszewski 2001 / Schmidt 2002
16. Bernard 1977
17. Talebitaher 2012
18. Barbaglia 2010
19. Janev et al. 1987
20. Yordanov 2003
21. Glasstone & Lovberg 1960
22. Sagdeev & Galeev 1969
23. IPPLM facility engineering drawings

Required correction:

- Either state `23 source-acquisition rows`, or split the table into
  `19 external/source families` plus the 23 visible rows. Do not call the table
  19 rows while it visibly has 23 rows.

### A3 - The Per-Blocker Table Is Not Uniform

V2 line 74 declares a nine-column per-blocker schema:

`blocker_id`, `current_repo_status`, `corrected_status`,
`source_or_acquisition`, `exact_path_or_full_citation`, `line_or_page_range`,
`scope_tag`, `runtime_claim_allowed`, `remaining_action`.

Only the geometry table uses all nine columns. Startup, transport, and neutron
tables omit `current_repo_status` and `runtime_claim_allowed`; the comparator
table omits `current_repo_status`.

This is acceptable for human review because V2 lines 76-78 and 97-98 state the
runtime claim is false. It is **not** acceptable for a machine-readable Sprint 5
ledger.

Required correction:

- Before implementation, generate a normalized ledger with one row per blocker
  and all nine fields present.
- If Markdown remains the source packet, add a companion CSV or JSON that
  preserves this schema exactly.

### A4 - Status Distribution Is Not Strictly Per-Blocker

V2 lines 55-65 present a status distribution across 31 blockers, but the entry
`already_target_extracted_in_kr_targets = 2` names Klir 2011 and Talebitaher
2012. Talebitaher is embedded inside the composite `NEUTRON-BLK-001` row, and
Klir 2011 does not appear as a standalone row in section 4.

Required correction:

- Either make Klir a named blocker row, or move Klir out of the blocker-status
  distribution and into the source-acquisition / existing-target summary.
- Composite rows such as `NEUTRON-BLK-001` should have a primary blocker
  status plus child source states, not be counted as multiple blocker statuses.

### A5 - One Status Name Still Drifts

V2 line 41 says Bernard rows use `existing_kr_review_pending`; the actual rows
use `existing_kr_target_extraction_pending`.

Required correction:

- Use one status string consistently. Prefer
  `existing_kr_target_extraction_pending` because that is what the current
  source lifecycle requires.

## What V2 Gets Right

- It preserves the fail-closed boundary: `accepted_runtime_claim = false` and
  `can_support_first_principles_acceptance = false`.
- It correctly treats Bennett 2017 as a high-priority on-disk source to promote
  and target-extract, not as an accepted startup solver.
- It correctly moves Braginskii 1965 behind rendered-page/OCR verification.
- It correctly reclassifies Talebitaher, Bernard 1977, and Gribkov Part II as
  already-local/current-KR work rather than new external acquisition.
- It correctly identifies the next real source-work queue: Bennett startup,
  Braginskii transport, Gribkov/Bernard neutron evidence, UCSD/Beg startup
  method context, Stepniewski hardware review, and Plasma Focus Update target
  extraction.

## Next Plan

### Sprint 5 Objective

Convert the corrected source triage into source-backed typed packets and
runtime fail-closed gates. The sprint is successful only when the codebase can
name exactly which first-principles modules are now source-supported,
candidate-only, external-blocked, or absent, without promoting any validation
claim.

### Sprint 5 Workstream 1 - Normalize The Handoff Into A Machine Ledger

Deliverables:

- `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`
- optional JSON mirror for tests and code use
- normalized rows with the exact nine V2 columns plus:
  `parent_blocker_id`, `child_source_id`, `accepted_runtime_claim`,
  `can_support_first_principles_acceptance`, `last_verified_commit`

Acceptance checks:

- 31 named blocker rows, no arithmetic mismatch.
- 23 source rows or a documented source-family collapse with no row-count
  ambiguity.
- No row has `accepted_runtime_claim = true`.
- No row uses shorthand KR paths.

### Sprint 5 Workstream 2 - Execute Existing-Local Target Extractions

Required extraction packets:

1. Bennett 2017 startup packet:
   CH03, CH04, CH07, CH08 with seed density, pressure regimes,
   explosive-emission thresholds, density contours, electron temperature,
   current-fraction timing, breakdown delay, and photoionization bound.
2. Braginskii 1965 rendered/OCR packet:
   page-render evidence first, then exact coefficient/equation extraction if
   the render verifies the claimed table/equation locations.
3. Gribkov / Scholz Part II fast-ion packet:
   PF-1000 full-energy only; direct fast-deuteron section, neutron anisotropy,
   neutron spectra / 100 keV beam-target context.
4. Bernard 1977 historical packet:
   Ti = 700 eV method context, 0/45/90 neutron spectra, U/I/L waveform context,
   and explicit `historical_mather_wrong_scope` tag.
5. UCSD/Beg current-sheath packet:
   `massf` lines `597-601`, pressure-regime lines `631-640`, Te/Liz context
   `642-660`, all marked wrong-scope method context.
6. Stepniewski 2004 review packet:
   keep `0.015 m` hollow radius blocked as simulation context unless
   hardware-scope evidence is acquired.
7. Plasma Focus Update target packet:
   PF-1000 geometry summary and Te filter-ratio context, with method caveats.

Acceptance checks:

- Each packet has exact KR paths and line ranges.
- Each packet has a scope tag.
- Each packet states whether it can affect runtime, tests, or only research
  planning.
- No packet changes first-principles acceptance flags to true.

### Sprint 5 Workstream 3 - External Acquisition Queue

MUST for first-principles whole-shot experimentation:

- D2 Townsend/Paschen or LXCat data.
- D2 electron-neutral momentum transfer cross sections.
- D2 molecular ionization/recombination sources or Open ADAS equivalents.
- Braginskii verified coefficient pages if OCR/render confirms extraction.
- ICRU/deuteron stopping tables.
- Brysk 1973 Doppler broadening.
- Davidson-Gladd and Bruzzone anomalous-resistivity sources.
- Surface secondary-emission evidence for Cu/alumina/pyrex/stainless.

Facility-request path:

- PF-1000 insulator outer radius.
- PF-1000 insulator wall thickness.
- PF-1000 backplate radial extent.
- PF-1000 backplate axial thickness.

### Sprint 5 Workstream 4 - Runtime Fail-Closed Integration

After source packets land, wire them into runtime as evidence gates, not as
accepted physics:

- startup BVP channel registry reads Bennett/UCSD/source packets and reports
  which channels are still external-blocked;
- transport closure packet can distinguish NRL cross-check, Braginskii direct
  coefficients, D2 electron-neutral gaps, and anomalous-resistivity gaps;
- neutron authority packet separates thermonuclear, beam-target, stopping,
  Doppler, detector, and anisotropy dependencies;
- geometry packet separates source-supported hardware geometry from
  simulation-context and absent facility-drawing fields;
- 12 us segmented whole-shot runner remains experimental and blocked unless
  limiter, conservation, restart, and source-packet checks pass.

Acceptance checks:

- `can_support_first_principles_acceptance` remains false unless every required
  gate has source, code, tests, numerical evidence, UQ, and same-scope
  comparator support.
- artifact linter rejects any accidental acceptance promotion.
- focused tests cover all newly source-backed and still-blocked states.

### Sprint 5 Workstream 5 - Scope Decision

The team must lock one of these before claiming comparator progress:

- **Recommended:** PF-1000 full-energy 27-40 kV as the first comparator scope.
  This has stronger I(t), V(t), neutron yield, anisotropy, and diagnostic
  support in the current corpus.
- **Alternative:** stay with Akel 16 kV, but then V(t), Te/Ti, X-ray, neutron
  spectrum, and anisotropy remain explicit same-scope gaps.

No engineering-firm-facing validation packet should be described as complete
until this scope decision is locked.

## Explanation For Project Direction

The team has moved us from "we have a pile of papers" to "we know which source
packets can close which blockers." That is real progress. We are still not at
the point where the simulator can truthfully claim a full first-principles DPF
shot, because the missing items are now the hard ones: startup gas data,
transport coefficients, stopping/beam-target physics, exact PF-1000 hardware
dimensions, and same-scope comparator channels.

The next sprint should not be another planning sprint. It should be a
source-to-runtime sprint: extract the local sources that are already identified,
acquire the few external sources that are genuinely required, and make the code
consume those packets through fail-closed registries. Once that is done, the
experimental runner can advance with honest labels: "source-supported",
"candidate", "external-blocked", or "absent", instead of relying on prose.
