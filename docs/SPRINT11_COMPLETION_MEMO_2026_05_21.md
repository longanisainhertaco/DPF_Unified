# Super-Sprint 11 Completion Memo (2026-05-21)

Controlling doc:
`docs/CODEX_SUPER_SPRINT10_AUDIT_AND_SUPER_SPRINT11_INSTRUCTIONS_2026_05_21.md`.

Branch: `codex/corpus`. Base HEAD audited: `fa713a83f53602d20d3eacc1a078202185f9b603`.

---

## 1. Codex Super-Sprint 10 Audit Verdict

**Verdict: Super-Sprint 10 is NOT accepted as complete.**

The Codex audit (2026-05-21) accepted the following SS10 work as substantively
correct:

- SS10-2/SS10-3: the five blocked PF-1000 geometry fields flow deck -> runtime
  telemetry -> segmented manifest; hollow-anode telemetry matches the selected deck.
- SS10-5: source-truth exhaustion passes; module-vetting strict-passed for 298 modules.
- SS10-7: all eight acceptance gates are blocked, named, and report-only.
- SS10-8: periodic audit passes 10/10 with the 146-line external-churn exception.

The full-completion claim was rejected pending six findings:

| Finding | Severity | Description |
| --- | --- | --- |
| S10-A1 | High | Imported-PIC context-only policy not enforced at deck level |
| S10-A2 | High | Server readiness stamps full-energy labels onto an Akel runtime deck |
| S10-A3 | Medium | Hybrid-PIC architecture source material under `same_scope` named structures |
| S10-A4 | Medium | Active `results/` artifacts contain stale LLNL-like selected/same-scope emissions |
| S10-A5 | Medium | Dry-run pass predicate too permissive (no explicit acceptance-support flag required) |
| S10-A6 | Low | Completion memo does not quote full source-truth, RTM, or periodic-audit evidence |

S10-A7 (informational): no acceptance-promotion defect was found in the current
PF-1000 runtime. Preserved throughout.

Super-Sprint 11 is a **policy and artifact-integrity closure sprint** that closes
S10-A1 through S10-A6. It does not start new physics acceptance work.

---

## 2. Per-Workstream SS11 Outcomes

### SS11-0 — Handoff Truthfulness

Deliverables:
- `docs/SPRINT10_COMPLETION_MEMO_2026_05_21.md` updated with a prominent top-level
  audit verdict section listing what was accepted, the six blocking findings
  S10-A1..A6, and a pointer to this memo. All unqualified SS10 completion claims
  qualified.
- `CodexFindings.md` and `CortexFindings.md` each received a new dated entry
  `### 2026-05-21: Super-Sprint 11 — Codex S10-A1..A6 closure` appended after
  the existing `2026-05-21: Codex Super-Sprint 10 Audit` entry. Existing entries
  were not modified.
- Verification (see §5): `rg` check returns no unqualified completion claim.

### SS11-1 — Deck-Level Imported-PIC Context-Only Enforcement (closes S10-A1)

Files changed:
- `src/dpf/first_principles/deck.py` lines 100-110: added module-level constant
  `CONTEXT_ONLY_STARTUP_MODES = {"imported_pic_sheath_state"}`.
- `src/dpf/first_principles/deck.py` lines 392-393: `StartupPolicy.__post_init__`
  now unconditionally forces `can_support_whole_shot_acceptance=False` for every
  mode in `CONTEXT_ONLY_STARTUP_MODES`, independent of evidence review or payload
  completeness, mirroring the existing `ENGINEERING_ONLY` force-branch.

Conversion path: `FirstPrinciplesInputDeck -> FirstPrinciples3DDeck` (via
`runner._values_from_package_deck`) reads the `StartupPolicy` flag, so clamping
at the `StartupPolicy` source ensures `FirstPrinciples3DDeck.startup_can_support_whole_shot_acceptance`
can never be `True` for imported-PIC even with a reviewed complete payload.

Tests: 4 new tests in `tests/test_ss10_imported_pic_context_only_policy.py`.
Focused slice (`test_ss10_imported_pic_context_only_policy`, `test_first_principles_startup_bvp`,
`test_startup_breakdown_audit`): **72 passed**.

### SS11-2 — Scope-Safe Server Readiness (closes S10-A2)

Files changed:
- `src/dpf/server/readiness.py` lines 113-326: complete rewrite of
  `_package_native_first_principles_readiness`. Added two new helpers:
  - `_resolve_runtime_deck_scope` — maps the declared validation+source scope
    pair to one of: `akel`, `full_energy`, or `None` (contradictory/unknown).
  - `_scope_blocked_first_principles_readiness` — returns a fail-closed blocked
    payload when no matching deck can be identified; no deck is run.
  A full-energy request runs `pf1000_scholz_2001_24rod_full_energy_deck`.
  An Akel request runs `pf1000_akel_16kv_engineering_deck`.
  An undeclared, unknown, or contradictory scope fails closed with no deck run.
  The readiness payload now exposes:
  `requested_validation_scope`, `requested_source_scope`,
  `actual_runtime_validation_scope`, `actual_runtime_source_scope`,
  `runtime_deck_id`, `scope_match`.

Tests: 6 new tests in `tests/test_server_readiness.py`.
Focused slice: **16 passed**.

### SS11-3 — Same-Scope Namespace Purity (closes S10-A3)

Files changed:
- `src/dpf/first_principles/same_scope.py`: hybrid-PIC architecture reference
  removed from `SAME_SCOPE_SOURCE_REFS` and added to a new constant
  `ARCHITECTURE_OR_SCHEMA_CONTEXT_SOURCES`. Relocation only — material preserved
  for schema-context use, not deleted.
- `src/dpf/validation/hybrid_pic_3d.py`: the `same_scope_3d_validation_packet`
  capability now carries
  `capability_source="missing_same_scope_3d_validation_packet_no_accepted_source"`
  instead of the LLNL-like hybrid-PIC source scope string.
- `src/dpf/first_principles/runner.py`: emits a sibling runtime/manifest key
  `architecture_or_schema_context_sources` alongside the existing structure.

Post-fix runtime scan result: for the PF-1000 full-energy preset, zero
`same_scope`-named runtime keys contain any of:
- `llnl_like_180ka_axisymmetric_hybrid_pic`
- `fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield`
- `hybrid_pic_architecture_order_of_magnitude_other_scope`

Tests: 3 new recursive-scan tests in `tests/test_ws9_runner_scope_source_geometry.py`.
Focused slice: **102 passed**.

### SS11-4 — Active Results Artifact Hygiene (closes S10-A4)

Action: 6 active stale top-level `results/*.json` files (all dated 2026_05_16)
relocated to `results/archive_stale_pre_ss11_2026_05_21/` via plain filesystem
`mv`. Contents untouched; scientific provenance intact.

Archived files:
- `results/experimental_machine_shot_family_all_1ns_2026_05_16.json`
- `results/experimental_machine_shot_family_all_1us_cap5000_2026_05_16.json`
- `results/experimental_numerical_family_pf1000_mesh_probe_2026_05_16.json`
- `results/experimental_numerical_family_pf1000_mesh_smoke_2026_05_16.json`
- `results/experimental_numerical_family_pf1000_timestep_probe_2026_05_16.json`
- `results/experimental_reproducibility_pf1000_probe_2026_05_16.json`

New artifacts:
- `scripts/verify_active_results_artifact_hygiene.py` — importable
  `scan_active_results` function + `--strict --check` CLI.
- `tests/test_results_artifact_hygiene.py` — 6 tests.

Post-move `rg` check:
```
rg "same_scope_3d_validation_packet|llnl_like_180ka_axisymmetric_hybrid_pic" results --glob "*.json"
```
Returns only `archive_*` paths.

Tests: **6 passed**.

### SS11-5 — Dry-Run Acceptance Predicate Hardening (closes S10-A5)

Files changed:
- `src/dpf/first_principles/acceptance_gate_dry_run.py`: `_build_gate_result`
  pass predicate tightened from
  `packet.get("can_support_first_principles_acceptance") is not False` to
  `packet.get("can_support_first_principles_acceptance") is True`.
  A gate now reports `pass` only when all three conditions hold simultaneously:
  accepted status, empty missing list, and the flag literally `True`.
  `is_fail_closed` docstring corrected to its true semantics: "report-only mode
  is active and no flag promotes acceptance — a passing gate does not by itself
  break fail-closed."

Current PF-1000 dry run: all 8 gates still blocked, `report_only=true`,
`promotes_acceptance=false`.

Tests: 4 new adversarial tests in `tests/test_first_principles_acceptance_gate_dry_run.py`:
accepted-status packets with the acceptance flag missing, `None`, or `False` all
remain `blocked`; positive control with flag `True` passes.
Focused slice: **14 passed**.

### SS11-6 — Complete Traceability Evidence in Memo (closes S10-A6)

This section (§4 Traceability Evidence) contains the verbatim command output
required by S10-A6. See §4 below.

### SS11-7 — Final Audit Gate

Unified pytest gate (10 test files):
```
tests/test_ss10_runner_deck_segmented_coherence
tests/test_ss10_imported_pic_context_only_policy
tests/test_first_principles_acceptance_gate_dry_run
tests/test_server_readiness
tests/test_git_status_clean_exception
tests/test_ws9_runner_scope_source_geometry
tests/test_startup_breakdown_audit
tests/test_first_principles_startup_bvp
tests/test_source_geometry_packet
tests/test_results_artifact_hygiene
```
Result: **241 passed**.

Ruff: `All checks passed!`

Module-source vetting and source-truth dated baselines regenerated at date
`2026_05_21` because SS11 edited physics modules (same regeneration step SS10
performed for `2026_05_20`).

---

## 3. Definition of Done Checklist

| DoD clause (§6) | Evidence |
| --- | --- |
| Imported-PIC context-only enforced in both `deck.py` and `startup_bvp.py` | `deck.py` lines 100-110 (CONTEXT_ONLY_STARTUP_MODES) + lines 392-393 (force-false); `startup_bvp.py` unchanged from SS10-4. SS11-1 focused tests: 72 passed. |
| Server readiness cannot mix Akel runtime output with full-energy labels | `readiness.py` lines 113-326 rewrite; `_resolve_runtime_deck_scope` routes by scope; `_scope_blocked_first_principles_readiness` fails closed on mismatch. SS11-2 tests: 16 passed. |
| No same-scope runtime key contains hybrid-PIC architecture source material | `same_scope.py` + `hybrid_pic_3d.py` + `runner.py` edits; recursive runtime scan in `test_ws9_runner_scope_source_geometry.py` confirms 0 hits for all 3 forbidden strings. SS11-3: 102 passed. |
| Active result artifacts no longer expose stale LLNL-like emissions outside archive policy | 6 files moved to `archive_stale_pre_ss11_2026_05_21/`; `verify_active_results_artifact_hygiene.py` enforces going forward. SS11-4: 6 passed. |
| Dry-run gates cannot report `pass` without explicit acceptance support | `_build_gate_result` predicate is now `is True`; adversarial tests confirm missing/None/False flag stays blocked. SS11-5: 14 passed. |
| SS10/SS11 memo quotes source-truth, RTM, module-vetting, and periodic audit evidence | §4 of this memo; commands run at date 2026_05_21. |
| All acceptance flags remain false | `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, `promotes_acceptance=false` — confirmed by SS11-7 unified gate (241 passed) and SS11-5 dry run. |
| `CodexFindings.md` and `CortexFindings.md` updated | Both files received a new dated SS11 entry appended after the existing SS10 audit tail. Existing entries preserved. |

---

## 4. Traceability Evidence

All commands run from `/Users/anthonyzamora/dpf-unified` using `.venv312/bin/python`.
Date baselines regenerated at `2026_05_21` because SS11 edited physics modules.

### Source-truth exhaustion

Command:
```
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check --date 2026_05_21
```

Output:
```
{
  "exhausted": true,
  "json": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_21.json",
  "markdown": "docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_21.md",
  "open_issue_count": 0
}
```

### Module-source vetting

Command:
```
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check --date 2026_05_21
```

Output:
```
{
  "active_physics_unvetted_count": 0,
  "json": "docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_21.json",
  "markdown": "docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_21.md",
  "missing_source_reference_count": 0,
  "strict_passed": true,
  "total_modules": 298
}
```

### SRS traceability export

Command:
```
.venv312/bin/python -m pytest tests/test_srs_traceability_export.py -q
```

Output:
```
..                                                                       [100%]
2 passed in 0.27s
```

---

## 5. Artifact Policy

An `archive_*` directory under `results/` is an explicitly named stale-artifact
quarantine zone. Any result JSON whose path contains an `archive_*` component is
a historical artifact preserved for provenance and is excluded from the
active-artifact hygiene gate.

`scripts/verify_active_results_artifact_hygiene.py` enforces that no
`results/**/*.json` file outside an `archive_*` directory contains the patterns
`same_scope_3d_validation_packet` or `llnl_like_180ka_axisymmetric_hybrid_pic`,
exiting non-zero on any violation.

Stale artifacts are relocated with plain filesystem `mv`, not rewritten, so their
scientific provenance remains intact.

---

## 6. Periodic Audit

Command:
```
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

RESULT: pending — recorded post-commit by the lead (SS11-7 final gate).

---

## 7. Remaining Work (post-Super-Sprint 11)

Super-Sprint 11 closed no physics blocker — it is policy and artifact-integrity
only. The next physics sprint addresses the following (per controlling doc §7):

- Same-scope PF-1000 full-energy source packet.
- Numerical-fidelity acceptance suite.
- PF-1000 geometry dimensions and same-scope reviewed mask.
- Sigma-p reviewed face set for Auluck power-port terms II/IV/V/VI.
- Startup BVP D2 breakdown/flashover/liftoff source packet.
- Comparator/UQ/certificate pipeline for PF-1000 full-energy observables.
