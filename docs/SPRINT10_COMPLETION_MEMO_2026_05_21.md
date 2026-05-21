# Super-Sprint 10 Completion Memo (2026-05-21)

---

## SUPER-SPRINT 10 AUDIT VERDICT (Codex, 2026-05-21)

**Verdict: Super-Sprint 10 is NOT accepted as complete.**

Audited HEAD: `fa713a83f53602d20d3eacc1a078202185f9b603`, branch `codex/corpus`.
Audit controlling doc: `docs/CODEX_SUPER_SPRINT10_AUDIT_AND_SUPER_SPRINT11_INSTRUCTIONS_2026_05_21.md`.
Closure sprint: **Super-Sprint 11** — see `docs/SPRINT11_COMPLETION_MEMO_2026_05_21.md`.

### What the Codex audit accepted as substantively correct

- SS10-2 / SS10-3: the five blocked PF-1000 geometry fields now flow deck -> runtime
  telemetry -> segmented manifest; hollow-anode telemetry matches the selected deck.
- SS10-5: source-truth exhaustion passes; module-vetting is strict-passed with 298 modules.
- SS10-7: all eight acceptance gates are blocked, named, and report-only for the current runtime.
- SS10-8: periodic audit passes 10/10 under the 146-line external-churn exception.

### Blocking findings (S10-A1..A6) — closed by Super-Sprint 11

- **S10-A1 (High):** Imported-PIC context-only policy not enforced at deck level — a reviewed
  complete payload could still carry `can_support_whole_shot_acceptance=True` through the
  deck/runtime conversion path.
- **S10-A2 (High):** Server readiness ran the Akel deck while stamping caller-supplied
  full-energy labels, creating an Akel/full-energy mixing path at the API readiness layer.
- **S10-A3 (Medium):** Hybrid-PIC architecture source material still appeared under
  `same_scope` named runtime structures (`telemetry.same_scope_source`,
  `manifest.candidate_evidence.same_scope_source_packet`,
  `hybrid_pic_3d_readiness.capabilities.same_scope_3d_validation_packet`).
- **S10-A4 (Medium):** Active `results/` artifacts still contained stale LLNL-like
  selected/same-scope emissions from the pre-SS10-1 code path.
- **S10-A5 (Medium):** Dry-run pass predicate too permissive — a packet with
  `status=accepted` and no explicit `can_support_first_principles_acceptance=True`
  was reported as `pass`.
- **S10-A6 (Low):** Completion memo did not quote the full source-truth, SRS/RTM, or
  periodic-audit evidence, including the approved 146-line external-churn exception note.

### S10-A7 — Informational

No acceptance-promotion defect was found in the current PF-1000 runtime.
`accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, and
`promotes_acceptance=false` were preserved throughout.

---

Controlling doc:
`docs/CODEX_SUPER_SPRINT9_AUDIT_AND_SUPER_SPRINT10_INSTRUCTIONS_2026_05_21.md`.

## 0. Posture

Super-Sprint 10 completed its workstreams but was **NOT accepted as complete by
the Codex audit (2026-05-21) — see verdict above**. Super-Sprint 11 closed the
six blocking findings S10-A1..A6.

Super-Sprint 10 was a **correction and gate-instrumentation sprint** — it closed
the Codex Super-Sprint 9 audit findings and added a report-only acceptance-gate
dry run. It is **not a validation sprint**. No acceptance flag was promoted:
`accepted_runtime_claim=false` and `can_support_first_principles_acceptance=false`
everywhere. Two commits on branch `codex/corpus` (unpushed):

- Phase 1 `a7c88cc` — SS10-1/2/3/4/6 runtime/startup/server corrections.
- Phase 2 (this commit) — SS10-7 acceptance-gate dry run, SS10-0/5/8 handoff.

## 1. Codex Super-Sprint 9 audit findings — disposition

| Finding | Severity | Workstream | Disposition |
| --- | --- | --- | --- |
| A1 misleading `same_scope_3d_validation_packet` from LLNL geometry | High | SS10-1 | closed |
| A2 WS9-6 geometry blockers not in runtime telemetry | High | SS10-2 | closed |
| A3 hollow-anode telemetry contradicts the deck | High | SS10-2 | closed |
| A4 segmented manifest lacks handoff summaries | High | SS10-3 | closed |
| A5 imported-PIC context-only not fully encoded | Medium | SS10-4 | closed |
| A6 WS9-8 artifact-regeneration claim not evidenced | Medium | SS10-5 | closed |
| A7 Sprint 9 memo/findings stale | Medium | SS10-0 | closed |
| A8 no acceptance promotion found | Info | — | preserved |

## 2. Per-workstream deliverables

### SS10-0 — Correct handoff narrative (A7)
The Sprint 9 memo gained a §Addendum recording the Codex Super-Sprint 9 audit
verdict (P0 accepted; WS9-6/7/8 incomplete; no acceptance promoted) and the
correct four-commit list; the stale "two commits" line and the inexact
`825 passed` transcript were replaced with the exact periodic-audit gate
reference. `CodexFindings.md` / `CortexFindings.md` carry dated Super-Sprint 10
entries. `rg "Two commits|825 passed|test_startup_breakdown_audit.py.*pre-existing"`
over the memo and findings docs returns no stale completion claim.

### SS10-1 — Split architecture evidence from same-scope 3-D evidence (A1)
`runner.py` no longer emits `candidate_evidence.same_scope_3d_validation_packet`
from `HybridPICSourceGeometry`. The LLNL-like hybrid-PIC 3-D geometry evidence
is emitted under the architecture-only key
`architecture_3d_geometry_candidate_packet`. For the PF-1000 full-energy preset
no `same_scope_3d_validation_packet` candidate evidence is emitted (fail-closed
— there is no genuine same-scope 3-D validation evidence yet). A runtime test
asserts no `same_scope`-named key carries `llnl_like_180ka_axisymmetric_hybrid_pic`.

### SS10-2 — PF-1000 geometry-mask runtime integrity (A2 + A3)
The deck's five blocked geometry fields (`anode_hollow_bore_length_m`,
`insulator_wall_thickness_m`, `backplate_radial_extent_m`,
`backplate_axial_thickness_m`, `same_scope_reviewed_geometry_mask`) are threaded
into runtime `boundary_policy.conductor_mask.blocked_geometry_fields`, each with
blocker ID, blocked status, and source/scope reason — and into the segmented
manifest. Hollow-anode telemetry now matches the deck:
`hollow_anode_declared_by_source=false`, `hollow_anode_inner_radius_supplied=false`,
`hollow_anode_bore_blocked=true` for the PF-1000 full-energy preset (the deck
leaves `anode_inner_radius_m=None`). Tests inspect the live runtime result.

### SS10-3 — Audit-complete segmented manifest (A4)
The `experimental-segmented-whole-shot` manifest carries four compact summary
blocks: `first_principles_scope_summary` (validation / selected-machine /
architecture scopes, acceptance flags false), `same_scope_summary` (declared
scope + channel states), `power_port_summary` (Auluck/Sigma-p term status,
II/IV/V/VI blocked on the reviewed face set), `geometry_blocker_summary` (the
five blocked fields).

### SS10-4 — Imported-PIC startup as context-only policy (A5)
`imported_pic_sheath_state` is removed from `ACCEPTED_STARTUP_MODES` and placed
in a new `CONTEXT_ONLY_STARTUP_MODES` class that can never satisfy
`mode_is_accepted`. The payload-review layer cannot mark a context-only mode
`channel_acceptance_eligible`. A regression test forces the typed startup
packet accepting path with an imported-PIC payload and proves it still cannot
promote.

### SS10-5 — Traceability and source-truth artifact truthfulness (A6)
Module-source vetting was regenerated — `strict_passed=true`, 298 modules
(the new report-only `acceptance_gate_dry_run.py` registered as
non-physics infrastructure). The RTM export (`export_srs_traceability.py`,
69 requirements) and the source-truth exhaustion `--strict --check --date
2026_05_20` were run and verified **no-ops** — no diff, exit 0. The Sprint 9
memo overclaim (A6) is corrected: only module-vetting actually changed.

### SS10-6 — Server-readiness scope drift
`tests/test_server_readiness.py` passes (10/10). Failure 1 was a stale test
expectation — the `pf1000_akel_16kv_engineering_deck` fixture uses a
`REJECTED_STARTUP_MODES` mode, so `rejected_startup_mode_for_first_principles`
is the correct fail-closed status (the test now asserts it). Failure 2 was
stale test-local Akel scope constants — replaced with canonical imports; no
Akel 16 kV source label is reintroduced for the PF-1000 full-energy demonstrator.

### SS10-7 — Report-only acceptance-gate dry run
New report-only module `src/dpf/first_principles/acceptance_gate_dry_run.py` and
CLI subcommand `dpf first-principles-acceptance-dry-run` run the eight
acceptance gates against the PF-1000 full-energy probe and emit a fail-closed
ledger (`report_only=true`, `promotes_acceptance=false`, acceptance flags
false). **All eight gates report `blocked`** with named missing inputs:

| Gate | Status | Named missing inputs (count) |
| --- | --- | --- |
| numerical_fidelity | blocked | 25 (reference solutions, convergence, restart, backend parity, …) |
| same_scope_comparator | blocked | 17 (accepted waveform/density/temperature/neutron histories, uncertainty budget, transfer rule, …) |
| uq | blocked | 19 (accepted target registry, per-observable metrics/tolerances, propagation method, …) |
| certificate | blocked | 31 (every upstream packet + negative-test matrix + hashes + reviewer metadata) |
| geometry | blocked | 5 (the five blocked PF-1000 geometry fields) |
| startup | blocked | 24 (geometry, breakdown/flashover model, sheath liftoff, initial state, …) |
| power_port | blocked | 8 (named Sigma-p domain, Poynting/J·E integral, sign/time-centering, residual tolerance, …) |
| neutron | blocked | 19 (mechanism-separated yields, cross-section source, transport/stopping, detector models, …) |

Full per-gate detail and next concrete actions:
`docs/SPRINT10_ACCEPTANCE_GATE_DRY_RUN_2026_05_21.md`. The dry-run JSON is
written to a temp path by default and is NOT committed to `results/`.

### SS10-8 — Periodic audit and worktree policy
The periodic audit is run at the Phase 2 commit. The 146-line external-churn
exception (145 PDF symlink typechanges + the `external/athenak` submodule) is
kept explicit; any new dirty line still fails `git_status_clean`.

## 3. Verification — exact commands and outputs

```
$ .venv312/bin/python -m pytest tests/test_first_principles_*.py \
    tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py \
    tests/test_ws9_runner_scope_source_geometry.py \
    tests/test_ss10_runner_deck_segmented_coherence.py \
    tests/test_ss10_imported_pic_context_only_policy.py \
    tests/test_source_geometry_packet.py tests/test_ws7_3d_runtime_ratchet.py \
    tests/test_startup_breakdown_audit.py tests/test_sprint8_ws4_bennett_startup.py \
    tests/test_server_readiness.py tests/test_git_status_clean_exception.py \
    tests/test_external_team_submission_package.py -q
800 passed   (Phase 1 integration sweep)

$ .venv312/bin/python -m pytest tests/test_first_principles_acceptance_gate_dry_run.py \
    tests/test_first_principles_runner.py tests/test_cli_first_principles_3d.py -q
65 passed   (SS10-7 dry-run + runner + CLI)

$ .venv312/bin/python -m ruff check src tests
All checks passed!

$ .venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_20
strict_passed=true, total_modules=298

$ .venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
# run at the Phase 2 commit; git_status_clean PASS with the approved
# external-churn exception note, every other gate green.
```

## 4. Definition of Done

> **Note:** SS10 completed its internal workstreams but was NOT accepted as
> complete by the Codex audit (2026-05-21) — six new audit findings (S10-A1..A6)
> were raised and closed by Super-Sprint 11. See verdict section above.

| DoD clause | State |
| --- | --- |
| Findings A1-A7 closed by code/docs/tests | done (Codex SS9 findings only — see S10-A1..A6 for SS10 audit findings closed by SS11) |
| No new acceptance flag promoted | held — all flags false |
| No LLNL-like evidence under same-scope keys | done (SS10-1) |
| Conductor-mask + segmented manifest expose the five blocked geometry fields | done (SS10-2/3) |
| Hollow-anode telemetry matches the selected deck | done (SS10-2) |
| Segmented manifest has scope/same-scope/power-port/geometry summaries | done (SS10-3) |
| Imported-PIC startup explicitly context-only at mode-policy level | done (SS10-4) |
| `test_server_readiness.py` passes | done (SS10-6, 10/10) |
| Periodic audit passes with the external-churn exception note | done (SS10-8) |
| `CodexFindings.md` / `CortexFindings.md` updated, no stale claims | done (SS10-0) |

## 5. Remaining work (post-Super-Sprint 10)

Super-Sprint 10 closed no physics blocker — it is correction and
gate-instrumentation only. The SS10-7 dry run makes the next physics sprint
measurable: every one of the eight acceptance gates is `blocked` with a named
missing-input list. Per the audit §7, the next work (not started here) is the
same-scope PF-1000 full-energy 3-D validation packet, the four absent PF-1000
geometry dimensions + IPPLM facility request, the Braginskii Table-2
second-reader cells, the Sigma-p reviewed face-set for power-port terms
II/IV/V/VI, the startup BVP D2 breakdown/flashover/liftoff source packet, the
numerical-fidelity convergence/restart/backend-parity suite, and the
comparator/UQ/certificate run for PF-1000 full-energy observables.
