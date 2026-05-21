# Super-Sprint 8 Source-to-Runtime Instructions (2026-05-20)

Purpose: convert Sprint 7's fail-closed runtime contracts into source-grounded
runtime candidates. This super-sprint is allowed to implement runtime physics
only when the source packet is KR-authoritative or explicitly classified as a
non-authority engineering cross-check. No workstream may set
`accepted_runtime_claim=true` or `can_support_first_principles_acceptance=true`.

Controlling inputs:

- `docs/SPRINT7_CODEX_MULTIAGENT_AUDIT_2026_05_20.md`
- `docs/SPRINT7_RUNTIME_CONTRACT_AUDIT_MEMO_2026_05_20.md`
- `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_PROMOTION_PROTOCOL_2026_05_20.md`
- `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`
- `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`
- `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv`

## Non-Negotiable Guardrails

1. `KnowledgeReference/` is the only scientific authority for physics claims.
2. On-disk PDFs outside KR may support acquisition and line/page verification
   only. They may not be runtime authority until KR ingestion and target
   extraction are complete.
3. Reduced Lee/snowplow outputs are baseline/comparator evidence only.
4. Candidate telemetry is engineering evidence only.
5. Same-scope source, code consumption, numerical fidelity, comparator/UQ, and
   certificate gates must all pass at the same commit before any accepted claim
   can exist.
6. Manual channel injection cannot create accepted same-scope evidence.
7. Cross-scope material cannot close a selected-scope comparator unless a
   reviewed transfer rule exists.
8. Runtime success is not validation.

## Definition Of Done For The Super-Sprint

The super-sprint is complete only when all P0 and P1 workstreams below are
done, tests pass, ledgers are synchronized, and an audit packet lists every
remaining blocked channel.

Completion must leave:

- all acceptance flags false unless a separate explicit acceptance review is
  run and passes;
- no stale ledger rows contradicted by source packets;
- no committed RTM/export drift;
- no source packet claiming target extraction while the source ledger says
  `already_in_kr=false`;
- a runnable engineering-candidate package-native 3-D path for the selected
  scope, with explicit duration and blocker telemetry.

## Workstream 0 - Ledger, KR, And Traceability Repair

Priority: P0. Must finish before runtime physics wiring.

Research:

- Reconcile Sprint 7 audit claims against the blocker ledger, source ledger,
  physics acceptance gate ledger, RTM CSV, RTM JSON, source truth index, and
  findings docs.
- Treat `docs/SPRINT7_CODEX_MULTIAGENT_AUDIT_2026_05_20.md` findings S7-A1
  through S7-A8 as required inputs.

Solve:

- Correct Bennett 2017 to
  `on_disk_line_page_verified_kr_promotion_required` until KR ingestion exists.
- Correct Braginskii to
  `target_extracted_source_supported_pending_equation_extraction_and_review`
  for Table 2, while keeping Eqs. 4.30-4.45 and five flagged cells blocked.
- Reclassify `SAME-SCOPE-COMPARATOR-DECISION` as control-plane governance, not
  KR scientific authority.
- Regenerate RTM CSV/JSON and source-truth index artifacts.
- Add dated findings-tail entries for Sprint 7 WS-B through WS-D and this
  super-sprint handoff.

Apply:

- Update:
  - `docs/FIRST_PRINCIPLES_BLOCKER_RESOLUTION_LEDGER_2026_05_20.csv`
  - `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`
  - `docs/FIRST_PRINCIPLES_PHYSICS_ACCEPTANCE_GATE_LEDGER_2026_05_20.csv`
  - `docs/SRS_TRACEABILITY_MATRIX.csv`
  - `docs/SRS_TRACEABILITY_MATRIX.json`
  - `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.md`
  - `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json`
  - `CodexFindings.md`
  - `CortexFindings.md`

Tests:

- Add a test that fails if any WS-E source packet claims target extraction
  while its source-ledger row has `already_in_kr=false`.
- Add a test that compares committed RTM CSV/JSON against a fresh export.
- Add blocker-ID-specific ledger tests for Bennett, Braginskii, and
  comparator-scope governance.
- Keep all fail-closed tests.

Exit criteria:

- Ledger tests no longer require stale commit `8e6b5e9` for current Sprint 7
  rows.
- Braginskii/Bennett ledger states match the audit.
- RTM drift check passes.

## Workstream 1 - Shared Acceptance Channel Contract

Priority: P0.

Research:

- Review `same_scope.py`, `numerical_fidelity.py`, `certificate_gate.py`,
  `runner.py`, `manifest.py`, `first_principles_mhd.py`, and CLI payloads.

Solve:

- Replace mixed accepted/missing channel accounting with explicit per-channel
  states:
  - `accepted`
  - `blocked_missing_source`
  - `blocked_wrong_scope`
  - `blocked_missing_review`
  - `blocked_missing_uncertainty`
  - `excluded_not_validated`
  - `not_claimed`
- Manual same-scope channel lists must become requested channels, not accepted
  evidence, unless backed by a reviewed target with uncertainty.
- The package-native 3-D readiness packet must remain the 3-D gate. The legacy
  cylindrical MHD gate must not silently accept or reject package-native runs
  with unrelated key expectations.

Apply:

- Update shared helpers or packet builders so the channel-state schema is used
  consistently.
- Keep top-level acceptance false.

Tests:

- Negative tests for manual non-Te/Ti acceptance injection.
- Positive synthetic fixture tests may exist only with `synthetic_fixture=true`
  and must not be usable by production decks.
- Contract tests for runner, CLI, manifest, and certificate packet agreement.

Exit criteria:

- A claimed channel cannot be both accepted and missing.
- Excluded channels never count as comparator evidence.
- Candidate evidence never unlocks acceptance.

## Workstream 2 - Scope Lock For Runtime Demonstrator

Priority: P0.

Decision required:

- Choose exactly one engineering-candidate runtime demonstrator scope for the
  next source-to-runtime work:
  - Option A: PF-1000/Akel 16 kV, with known same-scope diagnostic gaps.
  - Option B: PF-1000 full-energy campaign, with broader diagnostics but a
    scope change from the current Akel-shaped runtime defaults.

Solve:

- Encode the choice as a control-plane scope packet.
- Do not treat this as scientific evidence.
- Define which source packets belong to the selected scope and which are
  context-only or wrong-scope.

Tests:

- Mixed-scope source packets must fail.
- Scope memo rows must not be counted as KR source-supported scientific rows.

Exit criteria:

- Runtime deck preset, source ledgers, same-scope packet, and comparator
  decision all use the same selected scope label.

## Workstream 3 - PF-1000 Geometry Source-To-Runtime

Priority: P1.

Research:

- Use the Scholz 2000/2001 and Gribkov/Scholz KR packets already extracted.
- Keep Akel/Krauz constructors unchanged unless source mapping is explicit.

Solve:

- Add or refine a revision-scoped 24-rod PF-1000 deck path that consumes
  `PF1000GeometryPacket.scholz_2001_24rod_large_electrode()`.
- Keep blocked fields explicit:
  - hollow anode bore length;
  - insulator wall thickness;
  - backplate radial extent;
  - backplate axial thickness;
  - same-scope reviewed mask.
- Do not make the active deck pretend to have a hollow anode when
  `anode_inner_radius_m` is not source-supported for that scope.

Tests:

- Deck values match the target extraction packet.
- Akel/Krauz non-inheritance tests continue to pass.
- Under-resolved rod/insulator masks fail closed.
- Any missing hollow/backplate/wall field blocks mask acceptance.

Exit criteria:

- A source-scoped 24-rod runtime deck exists as an engineering candidate.
- Geometry acceptance remains blocked with exact missing fields listed.

## Workstream 4 - Bennett Startup BVP Consumption

Priority: P1, blocked by Workstream 0 Bennett KR ingestion.

Research:

- Promote Bennett 2017 to canonical KR markdown.
- Target-extract CH03, CH04, CH07, and CH08 into a typed packet.

Solve:

- Runtime startup BVP can consume:
  - seed density `1e7 cm^-3`;
  - breakdown delay about `20 ns`;
  - explosive emission thresholds `250 kV/cm` and `10 kV/cm`;
  - electron-temperature range `3.5-4.0 eV` as startup model context only;
  - sheath current fraction `71%` at `1 us`;
  - ionization density landmarks at 100 ns, 400 ns, and 500 ns.
- Keep CH01, CH02, CH05, CH06, CH09, CH10, CH11, CH12, and CH13 blocked or
  wrong-scope as applicable.

Tests:

- Unit conversions are enforced.
- Startup packet cites the KR record, not the misnamed raw PDF.
- Bennett values cannot close Akel 16 kV same-scope startup unless the selected
  scope explicitly supports that transfer.

Exit criteria:

- CH03/04/07/08 become source-backed runtime candidate channels.
- Whole-shot startup remains blocked until all required startup channels are
  source-backed or explicitly excluded from a limited claim.

## Workstream 5 - Braginskii Z=1 Transport Candidate

Priority: P1.

Research:

- Use the target-extracted Braginskii Table 2 packet.
- Finish equation-level extraction for Eqs. 4.30-4.45 before using transport
  formulas in runtime code.
- Use PlasmaPy only as a pinned cross-check lane, not as authority.

Solve:

- Add a typed source packet or module for Z=1 Braginskii coefficients.
- Wire candidate resistivity/thermal-transport outputs through
  `closure_packet.py` only after equations and units are extracted.
- Mark five review-required cells unavailable.

Tests:

- Coefficient values match the target-extracted source.
- Units are tested for Ohm*m and W/(m*K) where applicable.
- PlasmaPy comparison script reports pass/discrepancy but cannot promote
  authority.
- Closure packet remains non-accepted.

Exit criteria:

- Z=1 Braginskii transport can run as a source-backed candidate closure.
- Acceptance remains blocked on numerical/comparator/certificate gates.

## Workstream 6 - Power-Port And Sigma-p Operator Ledger

Priority: P1.

Research:

- Re-read the Auluck power-balance packet and current `power_port.py`.
- Treat Sigma/quasi-TEM line-voltage as not verified in local DPF sources
  unless a KR source packet is produced.

Solve:

- Demote active-load placeholders to engineering-only telemetry.
- Require six-term Auluck ledger terms to be independently present or fail
  closed:
  - stored magnetic energy rate;
  - motional magnetic Sigma-p surface integral;
  - stored electric energy rate;
  - motional electric Sigma-p surface integral;
  - resistive Sigma-p surface integral;
  - anomalous/poloidal Sigma-p surface integral.
- Add explicit sign convention, time-centering, domain, and residual fields.

Tests:

- Missing Sigma-p face set blocks power-port acceptance.
- Missing face velocity or resistivity blocks terms II, IV, V, and VI.
- Diagnostic inductance or active-load fallback cannot satisfy accepted power
  coupling.

Exit criteria:

- Power-port telemetry is useful for candidate runtime review.
- Accepted power-port authority remains blocked until all terms and residual
  tolerance are source-backed.

## Workstream 7 - 3-D Runtime Ratchet

Priority: P2.

Research:

- Review `segmented_whole_shot.py`, `segmented_whole_shot_combine.py`, CLI
  routes, restart tests, and prior 12 us dossier artifacts.

Solve:

- Add operational CLI parity:
  - `experimental-segmented-whole-shot` should support the same relevant
    `--dt-policy` and auto-step budgeting controls as `experimental-whole-shot`;
  - expose a CLI route for the segment combiner or document why the combiner is
    library-only.
- Produce an engineering-candidate run plan for the selected scope with source
  hashes, limiter ledger, conservation residual, power residual, and explicit
  duration status.

Tests:

- Segmented and uninterrupted deterministic short runs agree within declared
  tolerance.
- Duration request satisfaction is explicit.
- `hybrid_pic_3d_readiness` continues to list missing capabilities until all
  evidence is accepted.

Exit criteria:

- A reproducible engineering-candidate 3-D runtime probe exists for the selected
  scope.
- It is explicitly labeled not validation.

## Workstream 8 - External Source Queue

Priority: P2.

Research and acquire, but do not wire until KR-ingested and reviewed:

- D2 Townsend/Paschen or LXCat equivalent;
- D2 electron-neutral momentum transfer;
- molecular D2 ionization/recombination rates;
- surface secondary emission for Cu/alumina/pyrex/stainless;
- photoemission source;
- deuteron stopping tables;
- Brysk Doppler broadening or accepted equivalent;
- quantitative lower-hybrid anomalous resistivity;
- PF-1000 facility drawings for wall/backplate dimensions.

Exit criteria:

- Each acquired source has a source packet with hash, scope, target values,
  units, symbol map, and explicit claim impact.

## Super-Sprint 8 Audit Requirements

The team must deliver:

- one summary memo;
- updated ledgers;
- updated RTM exports;
- updated findings docs;
- source packets for every promoted source;
- tests proving fail-closed behavior;
- a list of every remaining blocker with reason and next action.

Codex will audit by:

1. parsing all ledgers for stale status, duplicate IDs, and acceptance flags;
2. checking KR path existence for every source-supported claim;
3. diffing RTM exports against a fresh render;
4. running focused tests and source-truth/module-vetting checks;
5. reviewing runtime packet internals for accepted/missing contradictions;
6. verifying no raw PDF or external URL is treated as scientific authority;
7. verifying no candidate run, reduced model, or manual channel unlocks
   acceptance.

