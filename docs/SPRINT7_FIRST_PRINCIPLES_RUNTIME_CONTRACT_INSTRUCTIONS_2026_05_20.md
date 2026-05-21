# Sprint 7 First-Principles Runtime Contract Instructions (2026-05-20)

## Objective

Convert Sprint 6 source intake and gate-design work into fail-closed runtime
contracts that a first-principles DPF simulator can consume without promoting
unreviewed physics. This sprint is not a validation sprint and must not produce
an accepted engineering-firm certificate. Its output is reviewed code/data
plumbing, source packet consumption, negative tests, and a clearer list of the
remaining physics blockers.

## Non-Negotiable Guardrails

1. `KnowledgeReference/` is the only scientific authority for physics claims.
2. Runtime acceptance stays false until source packet, code consumption,
   numerical acceptance, same-scope comparator, UQ, and certificate gates all
   pass at the same commit.
3. Reduced Lee/snowplow models may be comparator baselines only.
4. Candidate telemetry may support engineering review only. It must not set
   `accepted_runtime_claim=true` or
   `can_support_first_principles_acceptance=true`.
5. PF-1000 full-energy, PF-1000 2000/2001 24-rod, PF-1000/Krauz 12-rod, and
   PF-1000/Akel 16 kV scopes must remain separated unless a reviewed transfer
   rule is explicitly supplied.

## Workstream A: Source-Ledger Closure

Research:
- Re-read every Sprint 6 user-supplied source record in
  `docs/USER_SUPPLIED_PAPERS_INTAKE_2026_05_20.json`.
- Confirm each non-failed record has one matching row in
  `docs/FIRST_PRINCIPLES_SOURCE_ACQUISITION_LEDGER_2026_05_20.csv`.
- Confirm the blocker ledger row for `PF1000-BLK-015` says
  `existing_kr_source_supported`, not `absent_from_literature`.

Solve:
- If a source is context-only, mark `resolves_blockers=context_only` and keep
  `external_required=false`.
- If a source is a partial pair, split the available source and the still
  external companion into separate rows.

Apply:
- Update `tests/test_first_principles_v2_handoff_ledgers.py` for row counts,
  intake coverage, and fail-closed status.
- Update `CodexFindings.md` and `CortexFindings.md` with one dated ratchet entry.

Audit:
- Codex must independently parse both CSV ledgers and verify row counts,
  duplicate source IDs, and all `accepted_runtime_claim=false` /
  `can_support_first_principles_acceptance=false` claims.

## Workstream B: PF-1000 Revision-Scoped Geometry

Research:
- Use `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98`
  for Scholz 2001 24-rod geometry.
- Use `KnowledgeReference/pf-1000-device-a2d6bc15.md:83-154` only for early
  PF-1000 facility/chamber/bank context and explicitly note any ambiguous
  diameter/radius interpretation.

Solve:
- Keep Akel/Krauz constructors unchanged.
- Add or preserve one revision-specific constructor for the 2000/2001 24-rod
  geometry.
- Source-support only direct dimensions in that constructor. Leave bore length,
  insulator wall thickness, and backplate dimensions blocked.

Apply:
- Tests must prove the 24-rod constructor emits source-supported fields for
  rod count, rod length, rod diameter, cage radius, anode radius, insulator
  exposed length, insulator outer radius, chamber radius, and chamber length.
- Tests must also prove Akel/Krauz constructors do not silently inherit Scholz
  2001 dimensions.

Audit:
- Codex must inspect source refs on every `source_supported` geometry field and
  confirm the referenced local files exist.

## Workstream C: Package-Native 3-D Acceptance Contract

Research:
- Use `src/dpf/validation/hybrid_pic_3d.py` as the capability gate for the
  full 3-D hybrid PIC-fluid finish line.
- Do not directly reuse the cylindrical
  `src/dpf/validation/first_principles_mhd.py` gate as the 3-D contract.

Solve:
- The package-native runner must expose `hybrid_pic_3d_readiness` as a
  first-class telemetry packet.
- Candidate component telemetry must be visible but rejected by the readiness
  packet unless each capability has accepted/validated evidence.

Apply:
- Update `src/dpf/first_principles/runner.py`, `src/dpf/cli/main.py`, and tests
  so the 3-D readiness packet appears in runtime telemetry, manifest candidate
  evidence, CLI `telemetry_packets`, and `validation_packet`.

Audit:
- Negative tests must fail if candidate records, missing top-level contract
  keys, wrong backend labels, or missing same-scope 3-D validation can produce
  acceptance.

## Workstream D: Same-Scope Te/Ti Rejection

Research:
- Treat Te/Ti as required observables only when the claim includes them.
- If the selected certificate scope lacks direct same-scope Te/Ti evidence,
  do not count model output, text-only references, or generic caveats as
  comparator evidence.

Solve:
- `electron_temperature_history` and
  `ion_temperature_or_distribution_history` may be accepted only from direct
  same-scope diagnostics or ion-distribution measurements with review and
  uncertainty status.
- Manual `accepted_same_scope_channels` entries for those observables must be
  rejected unless the same direct evidence requirements are met.

Apply:
- Add negative controls for `caveat_accepted`, Lee/model-derived Te/Ti,
  text-only scalar temperature, and manual accepted-channel injection.

Audit:
- The audit must verify excluded or rejected Te/Ti channels never count as
  `accepted_same_scope` and never unlock the certificate gate.

## Workstream E: Next Physics Sources

Research:
- Re-audit Braginskii Table 2 using render evidence before runtime use.
- Keep LXCat, SRIM/NIST/IAEA, Munro, and PlasmaPy as substitute or cross-check
  lanes until source-equivalence review passes.
- Promote/target-extract Bennett 2017 startup channels only after line/page
  verification of the mislabeled PDF.

Solve:
- Produce one packet per source with source path, page/line, units, symbol map,
  scope tag, hash, code consumer, tests, and remaining blockers.

Apply:
- Wire no runtime coefficients until target extraction, code consumption,
  unit tests, numerical tests, and same-scope impact classification exist.

Audit:
- Codex and the external team must independently read the source packet and
  converge on the exact values before a packet can move from source-supported
  to implemented-candidate.

## Deliverables

- Updated ledgers and findings docs.
- Runtime contract code and tests for 3-D readiness.
- Revision-scoped PF-1000 geometry constructor and tests.
- Te/Ti same-scope negative controls.
- A short audit memo listing what was closed, what remains blocked, and what
  exact source/code/test evidence supports each transition.

