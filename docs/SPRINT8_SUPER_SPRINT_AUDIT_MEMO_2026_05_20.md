# Super-Sprint 8 Source-to-Runtime — Audit Memo (2026-05-20)

Controlling instructions:
`docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md`.

This memo is the Super-Sprint 8 deliverable: the *"one summary memo"* and
*"audit packet [that] lists every remaining blocked channel."*

## 0. Posture

Super-Sprint 8 is **not a validation sprint**. No engineering-firm certificate
is produced. Every runtime acceptance flag remains `false`:
`accepted_runtime_claim=false` and
`can_support_first_principles_acceptance=false` everywhere. Reduced
Lee/snowplow models stay comparator baselines only. Candidate telemetry is
engineering evidence only. The four PF-1000 revision scopes remain separated;
no transfer rule was supplied or assumed.

The sprint ran in two commits on branch `codex/corpus` (unpushed):

- Phase A P0 — `bd5be3a` — WS0 ledger/KR/traceability repair, WS1 shared
  acceptance-channel contract, WS2 runtime-demonstrator scope lock.
- Phase B+C P1/P2 — this commit — WS3 geometry deck, WS4 Bennett startup,
  WS5 Braginskii Z=1 transport, WS6 power-port Sigma-p ledger, WS7 3-D
  runtime ratchet, WS8 external source queue.

## 1. Definition-of-Done check

| DoD clause | State |
| --- | --- |
| All P0 workstreams done (WS0, WS1, WS2) | done |
| All P1 workstreams done (WS3, WS4, WS5, WS6) | done |
| P2 workstreams done (WS7, WS8) | done |
| Tests pass | 724 passed in the Phase B+C sweep; Phase A 305; WS0 integration 98; ruff `src/ tests/` clean |
| Ledgers synchronized | blocker + source ledgers, gate ledger, RTM, source-truth index, module-source vetting all regenerated and consistent |
| No committed RTM/export drift | RTM CSV/JSON byte-match a fresh export |
| No source packet claiming target extraction while `already_in_kr=false` | enforced by `test_wse_source_packet_target_extraction_claims_match_kr_ledger_state` |
| Runnable engineering-candidate package-native 3-D path for the selected scope | `pf1000_scholz_2001_24rod_full_energy` preset on `first-principles-3d` / `experimental-segmented-whole-shot`, explicit duration + blocker telemetry |
| All acceptance flags false unless a separate acceptance review passes | held: every flag `false` |

## 2. What was closed / advanced (with evidence)

### WS0 — Ledger, KR, and Traceability Repair (P0)
- Bennett 2017: Phase A corrected to
  `on_disk_line_page_verified_kr_promotion_required`; Phase B advanced the
  four startup rows (STARTUP-BVP-CH03/04/07/08) to
  `source_backed_runtime_candidate` after WS4 KR-ingested the source.
- Braginskii `CLOSURE-BLK-BRAG-001`: Phase A →
  `target_extracted_source_supported_pending_equation_extraction_and_review`;
  Phase B → `equations_4_30_to_4_45_render_verified_z1_transport_wired_as_candidate_acceptance_blocked`.
- `SAME-SCOPE-COMPARATOR-DECISION` reclassified to
  `scope_governance_decision_pending` — control-plane governance, not KR
  scientific authority.
- Ledger commit pins are a three-tier per-row scheme (`8e6b5e9` Sprint 4,
  `35bb1a9` Sprint 7, `bd5be3a` Phase B) — no stale global commit.
- RTM CSV/JSON regenerated (no drift); source-truth index refreshed
  (`exhausted=true`, 0 open issues); module-source vetting `strict_passed=true`
  (297 modules, 0 unvetted physics).
- **Evidence:** `tests/test_first_principles_v2_handoff_ledgers.py` (rewritten
  status counts, per-row pin tiers, drift check); 98-test Phase B integration
  run green.

### WS1 — Shared Acceptance Channel Contract (P0)
- New `src/dpf/first_principles/channel_state.py` defines exactly seven
  canonical channel states: `accepted`, `blocked_missing_source`,
  `blocked_wrong_scope`, `blocked_missing_review`,
  `blocked_missing_uncertainty`, `excluded_not_validated`, `not_claimed`.
  Shared by `same_scope.py`, `numerical_fidelity.py`, `certificate_gate.py`.
- Manual same-scope channel lists are demoted to **requested** channels —
  never accepted evidence without a reviewed, scope-matched target with
  uncertainty (audit finding S7-A8).
- The accepted/missing contradiction (S7-A7) is removed: a channel has exactly
  one canonical state; `accepted` and missing sets are disjoint.
- The legacy cylindrical `first_principles_mhd.py` gate now refuses to judge a
  package-native 3-D run and defers to the `hybrid_pic_3d` gate.
- **Evidence:** `tests/test_first_principles_channel_state_contract.py`
  (17 tests); 170-test Phase A run.

### WS2 — Scope Lock For Runtime Demonstrator (P0)
- Decision: **Option B — PF-1000 full-energy 27-40 kV** (consistent with the
  Sprint 4 scope-decision memo recommendation; Sprint 7 WS-B already laid the
  24-rod geometry). Encoded as a control-plane scope packet in
  `src/dpf/first_principles/runtime_demonstrator_scope.py`
  (`governance_class=control_plane`, `is_scientific_authority=false`,
  `accepted_runtime_claim=false`).
- Canonical scope label `pf1000_full_energy_27_to_40_kv`; sources classified
  in-scope / context-only / wrong-scope. Bennett 2017, Akel 16 kV, Bernard
  1977, UCSD/Beg, NX2/Talebitaher are **wrong-scope** for this demonstrator.
- **Evidence:** `tests/test_runtime_demonstrator_scope.py` (36 tests);
  governance memo `docs/SPRINT8_WS2_RUNTIME_DEMONSTRATOR_SCOPE_LOCK_2026_05_20.md`.

### WS3 — PF-1000 Geometry Source-To-Runtime (P1)
- New engineering-candidate deck `pf1000_scholz_2001_24rod_full_energy_deck`
  in `src/dpf/first_principles/deck.py`, scope-tagged
  `pf1000_full_energy_27_to_40_kv`, consuming
  `PF1000GeometryPacket.scholz_2001_24rod_large_electrode()`.
- Five fields kept explicitly blocked: anode hollow-bore length
  (`PF1000-BLK-010`), insulator wall thickness (`PF1000-BLK-016`), backplate
  radial extent (`PF1000-BLK-017`), backplate axial thickness
  (`PF1000-BLK-018`), same-scope reviewed geometry mask. The deck does **not**
  declare a hollow anode (`anode_inner_radius_m=None`).
- **Evidence:** `tests/test_source_geometry_packet.py` (+10 WS3 tests, 76
  pass); Akel/Krauz non-inheritance proven.

### WS4 — Bennett Startup BVP Consumption (P1)
- Bennett 2017 promoted to canonical KR markdown
  `KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md` (KR is
  gitignored local-only; PDF SHA-256
  `c5e6f5f1e2ca150a…` re-verified).
- New typed packet `src/dpf/first_principles/sprint8_bennett_startup_target_extraction.py`
  target-extracts CH03/04/07/08 with enforced SI unit conversions: seed
  density 1e7 cm⁻³ → 1e13 m⁻³; breakdown delay ~20 ns; explosive-emission
  thresholds 250 / 10 kV/cm; Te 3.5-4.0 eV (startup model context only);
  71 % sheath current at 1 µs; ionization landmarks at 100/400/500 ns.
- CH03/04/07/08 are **source-backed runtime candidate channels**. Because
  Bennett is wrong-scope for the selected demonstrator (WS2), their same-scope
  status for `pf1000_full_energy_27_to_40_kv` is `blocked_wrong_scope` — they
  cannot close same-scope startup absent a reviewed transfer rule.
- CH01/02/05/06/09/10/11/12/13 stay blocked or wrong-scope.
- **Evidence:** `tests/test_sprint8_ws4_bennett_startup.py` (60 tests).

### WS5 — Braginskii Z=1 Transport Candidate (P1)
- Braginskii 1965 Eqs. 4.30-4.45 render-verified this session (PyMuPDF, journal
  pp.249-253; 2-up spread layout confirmed). PDF SHA-256
  `9687440676b43b02…`.
- New `src/dpf/first_principles/sprint8_braginskii_z1_transport.py` carries the
  Z=1 candidate coefficients: parallel resistivity η∥ [Ohm·m] from Table-2
  α₀=0.5129; electron parallel thermal conductivity κ∥ᵉ [W/(m·K)] from γ₀=3.1616;
  ion parallel thermal conductivity from Eq. 4.40 coefficient 3.906.
- Candidate closure wired through `closure_packet.py`
  (`build_braginskii_z1_transport_closure()`); the PlasmaPy cross-check
  (`scripts/plasmapy_braginskii_z1_crosscheck.py`) agrees within 0.36 %.
- The five review-required Table-2 cells stay `unavailable_review_required`;
  none is a Z=1 cell.
- **Evidence:** `tests/test_sprint8_braginskii_z1_transport.py`;
  `CLOSURE-BLK-BRAG-001` remains a non-accepted blocker.

### WS6 — Power-Port and Sigma-p Operator Ledger (P1)
- `power_port.py` gains an explicit six-term Auluck eq.(6) presence roster
  (terms I-VI), so a five-term or closure-padded balance is visibly
  incomplete. Terms II/IV/V/VI fail closed pending Sigma-p face geometry,
  face velocity, and resistivity.
- Active-load placeholders (`active_power_W`, `diagnostic_field_inductance_H`)
  demoted to `excluded_not_validated` engineering-only telemetry — they cannot
  satisfy accepted power coupling.
- New `sign_convention`, `time_centering`, `domain`, `residual` fields; the
  residual carries `accepted_residual_tolerance: not_attached` (Auluck supplies
  no balance tolerance).
- Sigma / quasi-TEM line-voltage treated as not-verified — no KR source packet
  exists.
- **Evidence:** `tests/test_first_principles_power_port.py` (+13 WS6 tests).

### WS7 — 3-D Runtime Ratchet (P2)
- CLI parity: `experimental-segmented-whole-shot` gains `--dt-policy`,
  `--vacuum-cfl`, `--auto-step-budget`, `--max-auto-steps` — the same surface
  as `experimental-whole-shot`. New `combine-whole-run` CLI route wraps the
  segment combiner. The WS3 24-rod deck is exposed as preset
  `pf1000_scholz_2001_24rod_full_energy`.
- Engineering-candidate run plan
  `docs/SPRINT8_WS7_ENGINEERING_CANDIDATE_3D_RUN_PLAN_2026_05_20.md` for the
  selected scope, explicitly labeled not-validation.
- Segmented-vs-uninterrupted equivalence remains covered by the pre-existing
  dedicated `test_segmented_whole_shot_matches_uninterrupted_run`
  (`verify_restart_equivalence=True`, bit-identical assertions).
- **Evidence:** `tests/test_ws7_3d_runtime_ratchet.py` (12 tests).

### WS8 — External Source Queue (P2)
- Nine source packets in `docs/SPRINT8_WS8_EXTERNAL_SOURCE_QUEUE_2026_05_20.{md,json}`:
  D2 Townsend/Paschen, D2 e-neutral momentum transfer, D2 ionization/
  recombination, surface secondary emission, photoemission, deuteron stopping,
  Brysk Doppler broadening, lower-hybrid anomalous resistivity, PF-1000
  facility drawings.
- Nothing freely acquired to disk this session (paywalled / database-query /
  facility-request). Nothing wired. Nothing KR-ingested.

## 3. Every remaining blocked channel

| Channel / blocker | State | Why |
| --- | --- | --- |
| PF-1000 anode hollow-bore length (`PF1000-BLK-010`) | blocked | not numerically stated in any KR revision extract |
| Insulator wall thickness (`PF1000-BLK-016`) | blocked | absent from literature; IPPLM facility request |
| Backplate radial extent (`PF1000-BLK-017`) | blocked | absent from literature; IPPLM facility request |
| Backplate axial thickness (`PF1000-BLK-018`) | blocked | absent from literature; IPPLM facility request |
| Same-scope reviewed geometry mask | blocked | no reviewed transfer rule; mask stays `candidate_geometry_mask` |
| 3-D hybrid-PIC-fluid acceptance | blocked | 14 capabilities in `hybrid_pic_3d_readiness` unmet; no same-scope 3-D validation packet |
| Same-scope Te/Ti (demonstrator scope) | blocked | no direct same-scope PF-1000 bulk-pinch Te/Ti diagnostic; model/text/caveat evidence rejected by design |
| Bennett startup CH03/04/07/08 | source-backed candidate, acceptance blocked | numerical-fidelity + same-scope startup gates not run; `blocked_wrong_scope` for the demonstrator (Bennett wrong-scope per WS2) |
| Bennett startup CH01/02/05/06/09-13 | blocked / wrong-scope | no D2 Townsend table, no SEE/photoemission closure, no closed breakdown-BVP state |
| `CLOSURE-BLK-BRAG-001` Z=1 transport | candidate wired, acceptance blocked | numerical-fidelity / same-scope comparator / certificate gates pending; 5 review-required Table-2 cells need a second reader |
| Power-port terms II, IV, V, VI | fail closed | Sigma-p face set / face velocity / resistivity not exposed by runtime |
| Power-port residual tolerance | blocked | Auluck supplies no balance tolerance; `accepted_residual_tolerance: not_attached` |
| Sigma / quasi-TEM line-voltage driver | blocked | not source-verified in local DPF sources; no KR source packet |
| `SAME-SCOPE-COMPARATOR-DECISION` | scope_governance_decision_pending | control-plane governance row; not scientific evidence |
| LXCat / SRIM / NIST / Munro / PlasmaPy | review-queue | source-equivalence not granted |
| WS8 nine external sources | queued | not acquired / not KR-ingested / not wired |
| Whole-shot first-principles runtime acceptance | blocked | all gates above unmet |

## 4. Pre-existing item noted (out of Super-Sprint 8 scope)

`tests/test_startup_breakdown_audit.py::test_reviewed_imported_pic_startup_payload_can_close_packet`
fails at HEAD and predates Super-Sprint 8. `startup_bvp.py` has zero imports
from any sprint-modified module and was untouched by all nine workstreams; the
test fails identically against `35bb1a9` content. The file is **not** in the
periodic-audit pytest scope (the `tests/test_first_principles_*.py` glob does
not match `test_startup_*`). It is recorded here as pre-existing debt for a
future scoped fix; it does not affect any Super-Sprint 8 acceptance flag.

## 5. Acceptance posture

No transition to `accepted` was made in any channel. Bennett startup and
Braginskii Z=1 transport moved from blocked to **source-backed runtime
candidate** — engineering evidence only. Acceptance for every channel remains
gated on numerical-fidelity, same-scope comparator, UQ, and certificate gates
passing at one commit, none of which were run as acceptance gates this sprint.
The deliverable is reviewed code/data plumbing, source-packet consumption,
fail-closed tests, and this memo.
