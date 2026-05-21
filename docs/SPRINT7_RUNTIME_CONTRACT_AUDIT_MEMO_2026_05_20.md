# Sprint 7 First-Principles Runtime Contract — Audit Memo (2026-05-20)

Controlling instructions:
`docs/SPRINT7_FIRST_PRINCIPLES_RUNTIME_CONTRACT_INSTRUCTIONS_2026_05_20.md`.

This memo is the Sprint 7 deliverable: *"a short audit memo listing what was
closed, what remains blocked, and what exact source/code/test evidence
supports each transition."*

## 0. Posture

Sprint 7 is **not a validation sprint**. No engineering-firm certificate is
produced. Every runtime acceptance flag remains `false`:
`accepted_runtime_claim=false` and
`can_support_first_principles_acceptance=false` everywhere. Reduced
Lee/snowplow models stay comparator baselines only. The four PF-1000
revision scopes (full-energy, 2000/2001 24-rod, Krauz 12-rod, Akel 16 kV)
remain separated — no transfer rule was supplied or assumed.

## 1. What was closed (with evidence)

### WS-A — Source-Ledger Closure
- All 9 Sprint 6 user-supplied source records in
  `docs/USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json` (0 promoted / 9
  skipped-existing / 0 failed) each have exactly one row in
  `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`
  (31 rows total, 0 duplicate `source_id`).
- `PF1000-BLK-015` (insulator outer radius) blocker-ledger status is
  `existing_kr_source_supported` — the Scholz 2001 24-rod revision (KR
  `recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:96-98`,
  insulator diameter 229 mm) made the insulator outer radius source-available
  for that revision. It remains blocked for runtime revision mapping and
  acceptance.
- The Bruzzone/Bernal partial pair is split into
  `bruzzone_bernal_2001_lhi_interface` (KR-available) and
  `bruzzone_2001_lhi_companion` (still external).
- **Evidence:** `tests/test_first_principles_v2_handoff_ledgers.py`
  (34 tests pass — row counts, intake coverage, fail-closed checks);
  `CodexFindings.md` + `CortexFindings.md` Sprint 7 WS-A ratchet entries.

### WS-B — PF-1000 Revision-Scoped Geometry
- New constructor `PF1000GeometryPacket.scholz_2001_24rod_large_electrode`
  in `src/dpf/fields/source_geometry.py`. Source-supported fields, each
  cited to `recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98`
  or `pf-1000-device-a2d6bc15.md` chamber context: rod count 24, rod length
  0.600 m, rod diameter 0.032 m, cage radius 0.200 m, anode radius 0.122 m,
  insulator exposed length 0.113 m, insulator outer radius 0.1145 m, chamber
  inner radius 0.700 m, chamber length 2.500 m, anode length 0.600 m.
- Still **blocked** in that constructor: anode hollow-bore length
  (`PF1000-BLK` bore), insulator wall thickness (`PF1000-BLK-016`),
  backplate radial extent (`PF1000-BLK-017`), backplate axial thickness
  (`PF1000-BLK-018`).
- Akel and Krauz constructors are **unchanged** and verified to NOT inherit
  Scholz 2001 dimensions (rod diameter stays 0.080 m, anode radius stays
  0.1155 m, rod count stays `conflict`, rod length stays blocked).
- **Evidence:** `tests/test_source_geometry_packet.py` (79 tests pass;
  7 dedicated WS-B non-inheritance / scope-tag tests added). Codex must
  re-confirm every `source_supported` field's KR citation.

### WS-C — Package-Native 3-D Acceptance Contract
- `hybrid_pic_3d_readiness` is a first-class telemetry packet computed in
  `src/dpf/first_principles/runner.py:1167` via
  `dpf.validation.hybrid_pic_3d.hybrid_pic_3d_readiness_status` (NOT the
  cylindrical `first_principles_mhd.py` gate). It surfaces in: runtime
  `telemetry` (`runner.py:1229`), manifest candidate evidence
  (`hybrid_pic_3d_readiness_packet`), CLI `telemetry_packets`
  (`cli/main.py`), and `validation_packet`.
- Verified at runtime: `dpf first-principles-3d` emits the packet with
  `status="blocked"`, `can_support_first_principles_acceptance=false`, and
  14 missing capabilities listed.
- **Evidence:** 4 negative tests in `tests/test_first_principles_runner.py`
  prove acceptance CANNOT be produced by (a) a candidate-only record,
  (b) a missing top-level contract key, (c) a wrong backend/dimensionality
  label, (d) a missing same-scope 3-D validation packet. 55 runner tests
  pass.

### WS-D — Same-Scope Te/Ti Rejection
- `same_scope.py` accepts `electron_temperature_history` and
  `ion_temperature_or_distribution_history` ONLY from a direct same-scope
  diagnostic / ion-distribution measurement WITH review and uncertainty
  status (`_DIRECT_TEMPERATURE_EVIDENCE_TYPES` gate + review/uncertainty
  predicate). Manual `accepted_same_scope_channels` injection of those
  channels is rejected unless the same direct-evidence bar is met.
- **No generic `caveat_accepted` lane** — confirmed; `caveat_accepted` is
  not in the direct-evidence allow-list and falls through the rejection
  branch. This respects the Codex Sprint 5 WS2 audit constraint.
- **Evidence:** 4 dedicated negative controls in
  `tests/test_first_principles_certificate_negative_controls.py` —
  `caveat_accepted` Te/Ti rejected, Lee/model-derived Te/Ti rejected,
  text-only scalar temperature rejected, manual accepted-channel injection
  rejected. 32 tests pass. The certificate gate stays fail-closed:
  `build_same_scope_source_packet` always returns
  `status="blocked_same_scope_source_packet_not_available"`, which is in
  `certificate_gate.py` `BLOCKING_UPSTREAM_STATUSES`.

### WS-E — Next Physics Sources
- `docs/extractions/SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md`
  — one packet per source.
- **Braginskii Table 2 re-audit:** PDF SHA-256
  `9687440676b43b02…` and render-image SHA-256 `c914283871fbf6f1…` both
  re-confirmed against the Sprint 6 manifest. Z=1 column (17 values)
  confirmed; 5 `(review-required)` cells carried forward. Re-audit
  **confirms** the Sprint 6 extraction — no discrepancies.
- **Bennett 2017 line/page verification:** all 14 Sprint 5 extraction
  targets verified verbatim against the actual PDF pages (seed density
  1e7 cm⁻³, ~20 ns breakdown delay, 250/10 kV/cm thresholds, Te ~3.5-4 eV,
  71 % sheath current at 1 µs — Codex audit-row-7 correction independently
  confirmed). PDF SHA-256 `c5e6f5f1e2ca150a…`. Sprint 5 extraction
  **confirmed** — no discrepancies.
- LXCat, SRIM/NIST/IAEA, Munro 2012, PlasmaPy remain
  `source_equivalence_granted=false` substitute / cross-check lanes
  (Sprint 6 review packets unchanged).

## 2. What remains blocked

| Blocker | State | Why |
| --- | --- | --- |
| PF-1000 anode hollow-bore length | blocked | not numerically stated in any KR revision extract |
| PF1000-BLK-016 insulator wall thickness | blocked | absent from literature; IPPLM facility request |
| PF1000-BLK-017 backplate radial extent | blocked | absent from literature; IPPLM facility request |
| PF1000-BLK-018 backplate axial thickness | blocked | absent from literature; IPPLM facility request |
| 3-D hybrid-PIC-fluid acceptance | blocked | 14 capabilities in `hybrid_pic_3d_readiness` all unmet; no same-scope 3-D validation packet |
| Same-scope Te/Ti for the certificate scope | blocked | no direct same-scope PF-1000 bulk-pinch Te/Ti diagnostic; model/text/caveat evidence rejected by design |
| `CLOSURE-BLK-BRAG-001` runtime closure | source-supported, not implemented | Braginskii Table 2 target-extracted + render-verified, but no runtime coefficient is wired (Sprint 7 wires NONE) |
| Bennett 2017 startup channels CH03/04/07/08 | source-verified, not implemented | line/page verified, but not promoted into a runtime startup closure this sprint |
| LXCat / SRIM / Munro / PlasmaPy | review-queue | source-equivalence not granted; Sprint 7+ review session required |

## 3. Transitions and their exact evidence

| Transition | Evidence required (all present) |
| --- | --- |
| `PF1000-BLK-015` → `existing_kr_source_supported` | KR `recent-progress-…-d3e51f6c.md:96-98` insulator diameter 229 mm for the 2001 24-rod revision; ledger row; v2-handoff test |
| Braginskii Table 2 → `target_extracted_source_supported` | PDF + render SHA-256 re-confirmed; Z=1 column verified; WS-E re-audit packet |
| Bennett 2017 extraction → `line_page_verified` | 14 verbatim quotes re-read against PDF pages; PDF SHA-256 recorded |
| 24-rod geometry fields → `source_supported` | each cited to a KR line range; Codex must re-confirm |

**No transition** to `implemented-candidate` or `accepted` was made. Per the
WS-E audit clause, a source packet moves from `source-supported` to
`implemented-candidate` only after Codex and the external team
independently read the source and converge on exact values — that has not
happened in this sprint.

## 4. Sprint 8 entry preconditions

1. Codex independent re-audit of every WS-B `source_supported` geometry
   citation.
2. Codex + external-team convergence on the WS-E source packets before any
   coefficient is wired.
3. Same-scope 3-D validation packet acquisition before the
   `hybrid_pic_3d_readiness` packet can move off `blocked`.
4. IPPLM facility outreach for the 4 absent-from-literature geometry
   dimensions.

No runtime physics is accepted by Sprint 7. The deliverable is reviewed
code/data plumbing, source-packet consumption scaffolding, negative tests,
and this memo.
