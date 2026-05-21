# Codex Super-Sprint 10 Audit and Super-Sprint 11 Instructions

Date: 2026-05-21  
Branch audited: `codex/corpus`  
Audited HEAD: `fa713a83f53602d20d3eacc1a078202185f9b603`  
Prior controlling packet:
`docs/CODEX_SUPER_SPRINT9_AUDIT_AND_SUPER_SPRINT10_INSTRUCTIONS_2026_05_21.md`

## 1. Audit Boundary

Super-Sprint 10 was a correction and gate-instrumentation sprint. It was not a
physics-validation sprint and was not allowed to promote any first-principles
acceptance flag.

This audit applies the same fail-closed source rule:

- `KnowledgeReference/` and reviewed KR-derived packets are the only scientific
  authority.
- Engineering probes, dry-run ledgers, server readiness surfaces, stale result
  artifacts, and imported PIC payloads do not create physics authority.
- A gate may report `pass` only when its underlying packet explicitly supports
  first-principles acceptance. A textual `accepted` status is not enough by
  itself.

## 2. Audit Verdict

**Verdict: Super-Sprint 10 is not accepted as complete.**

Accept the following SS10 work as substantively correct:

- SS10-2 / SS10-3 geometry and segmented-manifest work: the five PF-1000
  blocked geometry fields now flow deck -> runtime telemetry -> segmented
  manifest, and hollow-anode telemetry now matches the selected deck.
- SS10-5 source-truth and module-vetting checks: source-truth exhaustion passes;
  module-vetting is strict-passed with 298 modules.
- SS10-7 current PF-1000 dry run: all eight gates are blocked, named, and
  report-only for the current runtime.
- SS10-8 periodic audit: 10/10 passes under the explicit 146-line external-churn
  exception.

Reject the full completion claim until the issues below are closed:

- A5 is only fixed inside `startup_bvp.py`; the higher-level deck/startup policy
  still lets a reviewed complete imported-PIC policy carry
  `can_support_whole_shot_acceptance=True`.
- Server readiness can still stamp full-energy labels onto an Akel runtime deck.
- Hybrid-PIC / other-scope source material still appears under `same_scope`
  named runtime structures, even though the narrow LLNL source-scope leak was
  fixed.
- Active `results/` artifacts still contain stale selected-scope / same-scope
  LLNL-like emissions.
- The dry-run pass predicate is too permissive for future accepted packets.
- The completion memo under-documents the source-truth, RTM, and periodic-audit
  evidence that was actually verified.

No acceptance-promotion defect was found in the current PF-1000 runtime.

## 3. Commands and Evidence

Commands run by Codex:

```bash
python3 ~/.codex/skills/dpf-validation/scripts/dpf_skill_preflight.py /Users/anthonyzamora/dpf-unified
python3 ~/.codex/skills/srs-traceability/scripts/srs_trace_audit.py /Users/anthonyzamora/dpf-unified
git status --short
git log --oneline -20
.venv312/bin/python -m pytest tests/test_ss10_runner_deck_segmented_coherence.py tests/test_ss10_imported_pic_context_only_policy.py tests/test_first_principles_acceptance_gate_dry_run.py tests/test_server_readiness.py tests/test_git_status_clean_exception.py -q
.venv312/bin/python -m pytest tests/test_ws9_runner_scope_source_geometry.py tests/test_startup_breakdown_audit.py tests/test_first_principles_startup_bvp.py tests/test_source_geometry_packet.py -q
.venv312/bin/python -m ruff check src tests
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_20
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_20
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
.venv312/bin/python -m dpf first-principles-acceptance-dry-run
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Observed:

- SS10/server/git focused slice: `66 passed`.
- WS9/startup/source-geometry focused slice: `152 passed`.
- Ruff: `All checks passed!`.
- Module-source vetting: `strict_passed=true`, `total_modules=298`.
- Source-truth exhaustion: `exhausted=true`, `open_issue_count=0`.
- SRS traceability export tests: `2 passed`.
- Dry-run CLI: all eight gates blocked; counts `25,17,19,31,5,24,8,19`;
  `report_only=true`; `promotes_acceptance=false`;
  `accepted_runtime_claim=false`; `fail_closed=true`.
- Periodic audit: `10/10 PASS` at
  `/private/tmp/dpf-unified-audit-logs/20260521T120055Z`.
- Worktree has 146 dirty lines: 145 PDF typechanges plus `m external/athenak`.
  Filtering out those approved exceptions leaves no additional dirty lines.

## 4. Findings

### S10-A1 - High - Imported-PIC context-only policy is not enforced at deck level

`startup_bvp.py` now correctly places `imported_pic_sheath_state` in
`CONTEXT_ONLY_STARTUP_MODES` and keeps it out of `ACCEPTED_STARTUP_MODES`.

However, `src/dpf/first_principles/deck.py` still includes
`imported_pic_sheath_state` in the broad `STARTUP_MODES` set and does not include
it in the deck-level engineering/context-only class. The deck policy only rejects
that mode when evidence is unreviewed or payload is missing. A reviewed complete
payload can still be converted into a runtime deck with
`startup_can_support_whole_shot_acceptance=True`.

Impact:

- The BVP layer remains fail-closed, so this is not an acceptance promotion
  today.
- The higher-level deck and runner surfaces still contradict the SS10 statement
  that imported PIC is context-only at the policy level.

Required SS11 correction:

- Add a deck-level `CONTEXT_ONLY_STARTUP_MODES` tuple/set that includes
  `imported_pic_sheath_state`.
- Reject or force `can_support_whole_shot_acceptance=False` for every context-only
  startup mode inside `StartupPolicy.__post_init__`.
- Ensure `FirstPrinciples3DDeck.from_deck()` can never receive
  `startup_can_support_whole_shot_acceptance=True` for imported PIC.
- Add tests that construct a reviewed complete imported-PIC
  `FirstPrinciplesInputDeck`, convert it through `FirstPrinciples3DDeck`, and
  prove the runtime startup policy is still context-only and non-promoting.

### S10-A2 - High - Server readiness can relabel an Akel deck as full-energy readiness

`src/dpf/server/readiness.py::_package_native_first_principles_readiness()` always
runs `pf1000_akel_16kv_engineering_deck(n_steps=1)`, then stamps the resulting
readiness payload with the caller-supplied `validation_scope` and `source_scope`.

Impact:

- `tests/test_server_readiness.py` passes, but the API can report full-energy
  readiness labels while the startup packet came from the Akel seed-layer deck.
- This reintroduces an Akel/full-energy mixing path at the API readiness layer.

Required SS11 correction:

- Make package-native readiness choose the runtime deck from the requested
  declared scope/source. For PF-1000 full-energy, run the full-energy deck. For
  Akel 16 kV, run the Akel deck.
- Alternatively, return both `requested_scope` and `actual_runtime_scope`, and
  fail closed when they differ. Do not stamp requested labels onto a different
  runtime deck.
- Add tests proving a full-energy readiness request contains no Akel startup
  mode, Akel source scope, or Akel validation packet blocker text.
- Preserve the existing Akel readiness path for actual Akel requests.

### S10-A3 - Medium - Hybrid-PIC material still appears under `same_scope` named structures

The narrow SS10-A1 geometry-packet defect is fixed:
`candidate_evidence.same_scope_3d_validation_packet` is no longer emitted from
`HybridPICSourceGeometry`; the architecture evidence is now under
`architecture_3d_geometry_candidate_packet`.

However, live PF-1000 full-energy output still exposes hybrid-PIC source material
under `same_scope` named structures:

- `telemetry.same_scope_source`
- `manifest.candidate_evidence.same_scope_source_packet`
- `telemetry.hybrid_pic_3d_readiness.capabilities.same_scope_3d_validation_packet`

These remain blocked/fail-closed, but the namespace still blurs architecture
source context and same-scope source evidence.

Required SS11 correction:

- Move hybrid-PIC architecture references out of `SAME_SCOPE_SOURCE_REFS`.
- If other-scope references are needed for schema context, emit them under a
  clearly named field such as `architecture_or_schema_context_sources`, not
  `same_scope`.
- In `hybrid_pic_3d_readiness`, the `same_scope_3d_validation_packet` capability
  should identify the missing packet as missing, without attaching the LLNL-like
  hybrid-PIC source as if it were the source for that capability.
- Add a runtime scan test: for the PF-1000 full-energy preset, no key containing
  `same_scope` may contain the LLNL-like source scope, the hybrid-PIC KR path, or
  the `hybrid_pic_architecture_order_of_magnitude_other_scope` role.

### S10-A4 - Medium - Active `results/` artifacts still contain stale LLNL-like leaks

The live code no longer emits the narrow A1 geometry-packet leak, but active
non-archive JSON artifacts under `results/` still contain
`same_scope_3d_validation_packet` and selected/source-scope fields with
`llnl_like_180ka_axisymmetric_hybrid_pic`. Example active files include:

- `results/experimental_machine_shot_family_all_1ns_2026_05_16.json`
- `results/experimental_reproducibility_pf1000_probe_2026_05_16.json`
- `results/experimental_numerical_family_pf1000_mesh_probe_2026_05_16.json`

Impact:

- Reviewers reading active artifacts can still see stale scope/source evidence
  that contradicts the current runtime contract.

Required SS11 correction:

- Move stale active artifacts with old LLNL-like selected/same-scope fields into
  an explicitly named archive directory, or mark them with stale artifact
  metadata that prevents active artifact consumers from treating them as current.
- Add an artifact hygiene test or audit gate that scans non-archive `results/*.json`
  and `results/**/*.json` for forbidden stale selected/same-scope scope patterns.
- Do not rewrite scientific artifact contents silently; preserve provenance by
  archiving or regenerating.

### S10-A5 - Medium - Dry-run pass predicate is too permissive for future use

The current PF-1000 dry-run ledger is correct: all eight gates are blocked and
named. The future pass predicate is too loose, though:

- A synthetic packet with `status="accepted"`, no missing list, and no explicit
  `can_support_first_principles_acceptance=True` is reported as `pass`.
- `AcceptanceGateDryRunLedger.is_fail_closed` still returns true even when most
  gates are reported `pass`.

Impact:

- This does not promote runtime acceptance today because the ledger-level flags
  remain hard-coded false.
- It is too weak for the next sprint because the dry-run could report a gate
  pass without the underlying gate explicitly authorizing first-principles
  acceptance.

Required SS11 correction:

- A dry-run gate may report `pass` only if the backing packet has an accepted
  status, an empty missing list, and
  `can_support_first_principles_acceptance is True`.
- Add adversarial tests for accepted-status packets with the acceptance flag
  missing, `None`, or false. They must remain `blocked`.
- Clarify the `is_fail_closed` semantics. If pass gates are allowed in future,
  the property should mean "report-only and no promoted ledger flags"; if not,
  it must reject pass gates. The tests and docs must use the same meaning.

### S10-A6 - Low - Completion memo under-documents verified SS10-5 and SS10-8 evidence

The underlying checks pass, but `docs/SPRINT10_COMPLETION_MEMO_2026_05_21.md`
does not quote the full source-truth command, SRS/RTM command, or the required
periodic-audit `git_status_clean` exception note.

Required SS11 correction:

- Add the exact source-truth exhaustion command/output.
- Add the exact SRS traceability test/export evidence.
- Add the periodic audit log path and quote the
  `APPROVED EXCEPTION: 146 known external-churn line(s) excused...` note.

### S10-A7 - Informational - No acceptance promotion found

No current PF-1000 runtime acceptance flag was promoted by Super-Sprint 10.

Required preservation:

- Keep `accepted_runtime_claim=false`,
  `can_support_first_principles_acceptance=false`, and
  `promotes_acceptance=false` until the source, numerical, comparator, UQ, and
  certificate gates all pass at the same commit.

## 5. Super-Sprint 11 Instructions

Super-Sprint 11 is a **policy and artifact-integrity closure sprint**. It must
not start new physics acceptance work until S10-A1 through S10-A6 are closed.

### SS11-0 - Handoff Truthfulness

Deliverables:

- Update `docs/SPRINT10_COMPLETION_MEMO_2026_05_21.md` with this audit verdict.
- Update `CodexFindings.md` and `CortexFindings.md` with a dated SS10 audit
  entry.
- Do not say SS10 is complete until S10-A1 through S10-A6 are closed.

Required check:

```bash
rg -n "Super-Sprint 10 is complete|SS10.*accepted as complete" \
  docs/SPRINT10_COMPLETION_MEMO_2026_05_21.md CodexFindings.md CortexFindings.md
```

This must return no unqualified completion claim.

### SS11-1 - Deck-Level Imported-PIC Context-Only Enforcement

Deliverables:

- Add deck-level context-only startup taxonomy.
- Force imported-PIC startup whole-shot support false in `StartupPolicy`.
- Ensure runtime deck conversion cannot carry imported-PIC
  `startup_can_support_whole_shot_acceptance=True`.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_startup_breakdown_audit.py -q
```

Add a new test that builds a complete reviewed imported-PIC
`FirstPrinciplesInputDeck`, converts it to `FirstPrinciples3DDeck`, and asserts
the runtime startup policy remains context-only and non-promoting.

### SS11-2 - Scope-Safe Server Readiness

Deliverables:

- Stop stamping requested full-energy scope labels onto an Akel runtime deck.
- Make readiness either run the matching package-native deck or report a
  mismatch as blocked.
- Preserve Akel readiness for Akel requests.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_server_readiness.py -q
```

Add tests proving:

- Full-energy readiness uses the full-energy deck.
- Akel readiness uses the Akel deck.
- A readiness payload exposes both requested and actual runtime scope/source.
- A mismatch is blocked and cannot be presented as ready.

### SS11-3 - Same-Scope Namespace Purity

Deliverables:

- Remove hybrid-PIC architecture references from `same_scope` source packets.
- Rename or move other-scope architecture/schema context into non-same-scope
  fields.
- Keep all removed material available as architecture context where appropriate.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_ss10_runner_deck_segmented_coherence.py \
  tests/test_first_principles_acceptance_gate_dry_run.py -q
```

Add a recursive runtime scan asserting that same-scope keys do not contain:

- `llnl_like_180ka_axisymmetric_hybrid_pic`
- `fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield`
- `hybrid_pic_architecture_order_of_magnitude_other_scope`

### SS11-4 - Active Results Artifact Hygiene

Deliverables:

- Archive or regenerate active result artifacts that contain old LLNL-like
  selected/same-scope emissions.
- Add an active-results audit that excludes explicitly archived stale artifacts
  and fails on stale current artifacts.
- Document the artifact policy in the audit memo.

Required check:

```bash
rg -n "same_scope_3d_validation_packet|llnl_like_180ka_axisymmetric_hybrid_pic" \
  results --glob "*.json"
```

The only remaining hits must be under directories named `archive_*` or must be
explicitly marked stale and excluded by the active-results audit.

### SS11-5 - Dry-Run Acceptance Predicate Hardening

Deliverables:

- Require explicit `can_support_first_principles_acceptance is True` before any
  dry-run gate reports `pass`.
- Add adversarial tests for missing/false/none acceptance flags.
- Clarify `is_fail_closed` naming and docs.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_acceptance_gate_dry_run.py -q
```

### SS11-6 - Complete Traceability Evidence in Memo

Deliverables:

- Add exact source-truth, RTM, module-vetting, and periodic-audit command outputs
  to the SS10 or SS11 completion memo.
- Quote the periodic audit exception note.

Required commands:

```bash
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_20
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_20
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

### SS11-7 - Final Audit Gate

Required final commands:

```bash
.venv312/bin/python -m pytest tests/test_ss10_runner_deck_segmented_coherence.py \
  tests/test_ss10_imported_pic_context_only_policy.py \
  tests/test_first_principles_acceptance_gate_dry_run.py \
  tests/test_server_readiness.py \
  tests/test_git_status_clean_exception.py \
  tests/test_ws9_runner_scope_source_geometry.py \
  tests/test_startup_breakdown_audit.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_source_geometry_packet.py -q
.venv312/bin/python -m ruff check src tests
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

## 6. Super-Sprint 11 Definition of Done

Super-Sprint 11 is complete only when:

- Imported-PIC context-only policy is enforced in both `deck.py` and
  `startup_bvp.py`.
- Server readiness cannot mix Akel runtime output with full-energy labels.
- No same-scope runtime key contains hybrid-PIC architecture source material.
- Active result artifacts no longer expose stale LLNL-like selected/same-scope
  emissions outside an explicit archive/stale policy.
- Dry-run gates cannot report `pass` without explicit acceptance support from
  the backing packet.
- The SS10/SS11 memo quotes the source-truth, RTM, module-vetting, and periodic
  audit evidence.
- All acceptance flags remain false.
- `CodexFindings.md` and `CortexFindings.md` are updated.

## 7. Work After Super-Sprint 11

After SS11 closes, the next physics sprint can begin. The highest-leverage
physics work remains:

- Same-scope PF-1000 full-energy source packet.
- Numerical-fidelity acceptance suite.
- PF-1000 geometry dimensions and same-scope reviewed mask.
- Sigma-p reviewed face set for Auluck power-port terms II/IV/V/VI.
- Startup BVP D2 breakdown/flashover/liftoff source packet.
- Comparator/UQ/certificate pipeline for PF-1000 full-energy observables.
