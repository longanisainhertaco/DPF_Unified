# DPF-Unified Current TODO/FIXME/XXX Audit

Generated: 2026-05-08

Task: B14 current TODO audit refresh

## Scope

This audit refreshes the historical placeholder audit against the current source
tree. It intentionally does not treat the former monolithic
`src/dpf/engine.py` path as a live blocker: that file is absent in the current
tree, and the live engine is the decomposed `src/dpf/engine/` package.

Audited as live project-owned scope:

- `src/` and `tests/`, excluding `src/dpf/engine_archive/`
- Current project docs and tooling: `docs/`, `tools/`, `assess.py`,
  `CRITICAL_BLOCKER.md`, `CortexFindings.md`, and `CodexFindings.md`
- `docs/todo_audit.md` itself is excluded from marker scans because this file
  necessarily contains marker strings and command evidence.

Excluded from live blocker status:

- `external/` vendored upstream trees
- `.claude/` and `.worktrees/` hidden worktree snapshots
- `gui/package-lock.json` hash text
- `src/dpf/engine_archive/` historical monolith archive

## Summary

| Classification | Count | Meaning |
| --- | ---: | --- |
| bug | 1 | Current code marker that points at a live backend behavior gap. |
| deferred | 4 | Planned work, roadmap test rows, or metadata still intentionally open. |
| benign | 4 | Marker strings used as detector text, templates, CI policy text, or findings context. |
| obsolete | 8 | Historical or stale references that should not be carried as live blockers. |

No current `FIXME` markers were found in active project-owned files outside
vendored and hidden worktree paths. No current active `src/` or `tests/` marker
points at `src/dpf/engine.py`.

## Current Source Findings

| ID | Classification | Location | Finding | Backlog routing |
| --- | --- | --- | --- | --- |
| B14-001 | bug | `src/dpf/engine/core.py:148` | Athena++ backend still has an inline marker that `dpf_zpinch.cpp` needs a C++ source function to inject circuit B-field. The current source comment says Python-side electrode boundary injection segfaults because pybind11 arrays are read-only views. | Engineering backlog for Athena++ circuit-field source coupling. |
| B14-002 | deferred | `src/dpf/metal/mlx_solver.py:1342` | MLX AMR gather currently copies level-0 blocks back to the global state and leaves fine-level overlay as future work. This is an AMR refinement fidelity gap, not a stale monolithic-engine blocker. | Engineering backlog for two-level AMR overlay before claiming refined-grid AMR production fidelity. |

## Documentation And Tooling Findings

| ID | Classification | Location | Finding | Action |
| --- | --- | --- | --- | --- |
| B14-003 | deferred | `CRITICAL_BLOCKER.md:124` | Self-consistent MHD `Lp` remains marked as a post-PR-B follow-up. | Keep in engineering backlog; not a TODO-audit blocker by itself. |
| B14-004 | benign | `CortexFindings.md:135,188`; `CodexFindings.md:899,940,954` | Findings docs reference the B14 audit task. The user explicitly requested that these files not be edited in this pass. | Leave untouched; this audit is the closure artifact for B14. |
| B14-005 | obsolete | `docs/DPF_UNIFIED_SRS_DRAFT.md:40,181,285,313,319,341` | The SRS draft still describes `docs/todo_audit.md` as historical or unrefreshed. This is stale after this refresh, but outside the owned edit scope. | Treat as follow-up SRS text cleanup only; do not carry as a live code blocker. |
| B14-006 | benign | `assess.py:35,37,107,112`; `tools/memory_manager.py:270,273` | These are scanners looking for marker strings in other files. | No action. |
| B14-007 | deferred | `docs/joss-paper-draft.md:12` | JOSS draft has an open ORCID metadata marker. | Publication metadata follow-up. |
| B14-008 | benign | `docs/design/frontend-dmaic-execution.md:31` | CI policy text names marker strings as blocked tokens. | No action. |
| B14-009 | obsolete | `docs/ARCHITECTURAL_DEBT.md:74` | This doc says line 706 of `mlx_solver.py` carries a current-density TODO. Current `mlx_solver.py` line 706 imports `cons_to_prim`, and the surrounding code now imports `compute_current_density_si`; the referenced marker no longer exists. | Stale architectural-debt text; do not carry as a live blocker. |
| B14-010 | benign | `docs/SPRINT_S3_QUALITY_PLAN.md:77` | `REQ-XXX` is a template placeholder in a requirements example. | No action. |
| B14-011 | deferred | `docs/VERIFICATION_AND_UAT_PLAN.md:551-567` | Verification/UAT matrix rows mark several planned tests as open. | Keep in V&V backlog; these are roadmap rows, not discovered source TODO bugs. |
| B14-012 | obsolete | `docs/RESEARCH_INDEX.md:131,137` | Research index still reports the old audit counts and memory TODO inventory. | Stale index metadata; do not carry as live source blockers. |

## Historical Audit Entries Retired

| Historical entry | Current classification | Reason |
| --- | --- | --- |
| `src/dpf/engine.py` C1 back-EMF blocker | obsolete | `src/dpf/engine.py` is absent. Current back-EMF concerns are covered by circuit-coupling code/tests and not by a current marker at that path. |
| `src/dpf/engine.py` C2 zipper-BC blocker | obsolete | `src/dpf/engine.py` is absent. No current active marker at that path exists. |
| Old `src/dpf/engine.py` priority action list | obsolete | The current engine package is decomposed under `src/dpf/engine/`; only the Athena++ source-function marker at `src/dpf/engine/core.py:148` remains live. |
| `src/dpf/engine_archive/engine_v1_monolith_20260324.py:153` duplicate marker | obsolete | This is an archive copy of the old monolith and is excluded from live blocker status. |
| Old audit severity counts | obsolete | The refreshed active scan found 2 active `src`/`tests` markers outside the archive, not the historical 3 critical / 9 medium / 5 low set. |

## Audit Commands And Results

Active code scan:

```text
$ rg -n --sort path -S "\b(TODO|FIXME|XXX)\b" src tests --glob '!src/dpf/engine_archive/**'
src/dpf/engine/core.py:148:            # TODO: dpf_zpinch.cpp needs C++ source function to inject circuit
src/dpf/metal/mlx_solver.py:1342:        # TODO: overlay fine-level data for 2-level AMR
```

Engine path check:

```text
$ test -e src/dpf/engine.py && echo 'src/dpf/engine.py exists' || echo 'src/dpf/engine.py missing'
src/dpf/engine.py missing
```

Current engine path inventory:

```text
$ find src/dpf -maxdepth 2 -path 'src/dpf/engine*' -print
src/dpf/engine_archive
src/dpf/engine_archive/__init__.py
src/dpf/engine_archive/engine_v1_monolith_20260324.py
src/dpf/engine
src/dpf/engine/circuit_coupling.py
src/dpf/engine/__init__.py
src/dpf/engine/core.py
src/dpf/engine/__pycache__
src/dpf/engine/athena_step.py
src/dpf/engine/physics_operators.py
src/dpf/engine/backend_dispatch.py
src/dpf/engine/state_management.py
```

Project docs/tooling scan used for the documentation classifications:

```text
$ rg -n --sort path -S "\b(TODO|FIXME|XXX)\b" docs tools assess.py CRITICAL_BLOCKER.md CortexFindings.md CodexFindings.md --glob '!docs/todo_audit.md' --glob '!docs/session_logs/**' --glob '!docs/EXECUTION_PLAN.md'
Result: 34 matching lines across CRITICAL_BLOCKER.md, CortexFindings.md, CodexFindings.md, assess.py, tools/memory_manager.py, docs/DPF_UNIFIED_SRS_DRAFT.md, docs/design/frontend-dmaic-execution.md, docs/SPRINT_S3_QUALITY_PLAN.md, docs/VERIFICATION_AND_UAT_PLAN.md, docs/RESEARCH_INDEX.md, docs/joss-paper-draft.md, and docs/ARCHITECTURAL_DEBT.md.
```

Excluded-scope scan used to confirm vendored, hidden worktree, archive, and
generated-lockfile markers were not promoted into the live blocker list:

```text
$ rg -n -S "\b(TODO|FIXME|XXX)\b" external .claude .worktrees gui/package-lock.json src/dpf/engine_archive --hidden --glob '!external/verif_coupling/.git/**'
Result: 680 matching lines; excluded from live blocker status by scope.
```
