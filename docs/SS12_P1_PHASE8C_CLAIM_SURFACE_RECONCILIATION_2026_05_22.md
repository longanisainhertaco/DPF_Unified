# SS12 P1 Phase 8-C Claim-Surface Reconciliation — Evaluate / Learn / Continue

Date: 2026-05-22
Task: `t_4fbfe5be`

## Scope

Phase 8-C checked user-facing claim surfaces against the current fail-closed first-principles / full-3D acceptance state.

Required first-read files checked:

- `AGENTS.md`
- `CodexFindings.md`
- `CortexFindings.md`
- `docs/MODULE_AUDIT/BACKLOG.md`
- `README.md`
- `pyproject.toml`
- `docs/SS12_P1_FIRST_PRINCIPLES_3D_COMPLETION_MASTER_PLAN_2026_05_22.md` lines 168-183

Additional claim surfaces checked because they are user-facing or user-visible:

- `app.py`
- `app_mhd.py`
- `docs/FIRST_PRINCIPLES_SIMULATOR_STATUS_BLOCKERS_2026_05_18.md`
- `tests/test_gradio_claims.py`
- `tests/test_readme_claims.py`

Search patterns used:

```text
first[- ]principles
full[- ]?3d
full 3d
3-d
validation
accepted
acceptance
predictive
publication-grade
publication quality
maximum accuracy for publications
publication use
validated against:
validated against [0-9a-z,+ -]*published devices
current sheath and compression emerge from first principles
accepted first-principles
can honestly advertise
```

## Evaluate

The active blocker sources still require fail-closed language:

- `AGENTS.md` current hard blockers preserve Akel Fig. 1 as `blocked_by_review`; S1/S2 waveform/current-dip validation blocked; Akel Figs. 2-6 digitization/review needed; Tier 2/4/5 need same-scope target packets; diagnostics, radiation, p-B11, QMF, ionization, ablation, line-cooling, detector response, neutron spectrum, and anisotropy claims remain non-predictive unless source-closed.
- `CodexFindings.md` latest SS12-P0 audit authorizes P1 work but explicitly says all eight dry-run gates remain blocked/report-only and `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, `promotes_acceptance=false`.
- `CortexFindings.md` latest SS12-P0 audit records the same no-promotion boundary and says SS12-P0 closed policy blockers only, not physics blockers.
- `docs/MODULE_AUDIT/BACKLOG.md` still carries blocked validation, diagnostics, radiation, AI/WALRUS, supplemental physics, and circuit waveform review rows.
- `docs/FIRST_PRINCIPLES_SIMULATOR_STATUS_BLOCKERS_2026_05_18.md` preserves the current blocker list: startup BVP, power port, reviewed 3-D geometry, physics closures, neutron authority, same-scope data binding, numerical fidelity, and certificate / engineering review packet.

Claim surfaces that were already conservative:

- `README.md` explicitly says the project is not yet an end-to-end predictive DPF simulator, marks predictive readiness blocked by default, separates engineering tests from scientific validation, and states the `dpf first-principles` command emits an engineering probe artifact rather than scientific validation evidence.
- `pyproject.toml` contains only package metadata and pytest marker names; no public first-principles/full-3D acceptance claim needed editing.
- `app_mhd.py` inspected hits were readiness/validation plumbing and fail-closed copy, not positive public acceptance claims.
- `docs/SS12_P1_FIRST_PRINCIPLES_3D_COMPLETION_MASTER_PLAN_2026_05_22.md` Phase 8 wording is an evaluation goal, not an acceptance claim; line 183 explicitly requires either still-false blockers or complete certificate evidence.

Claim surfaces changed:

- `app.py`
  - Replaced "Maximum accuracy for publications" with "Highest numerical-resolution preview; not publication validation".
  - Replaced "publication quality" language with "sharpest preview; publication or Reference claims still require accepted source evidence".
  - Replaced broad Lee-model "Validated against" device copy with a source-gated baseline-reduced-model authority statement.
  - Replaced "current sheath and compression emerge from first principles" / broad stability copy with source-gated, non-accepting MHD-equation wording.
  - Replaced "publication use" grid guidance with "engineering preview" wording.
- `tests/test_gradio_claims.py`
  - Added regression bans for the exact risky public-copy phrases found above.

Historical blocker strings were not deleted. No `CodexFindings.md`, `CortexFindings.md`, `BACKLOG.md`, or blocker-doc status text was rewritten; the reconciliation was limited to current UI/user-facing claim copy plus guardrail tests.

## Learn

The live README is already aligned with fail-closed Phase 8 language, but the legacy Gradio UI still carried stale marketing phrasing from earlier claim surfaces. The riskiest strings were not in the top-level README; they were in embedded help copy under `app.py` and were only partially covered by the existing Gradio claim-hygiene test. The guardrail now bans the exact variants that escaped the prior check.

The current truthful acceptance statement is:

```text
accepted_runtime_claim=false;
can_support_first_principles_acceptance=false;
promotes_acceptance=false;
first-principles/full-3D acceptance remains blocked pending complete reviewed certificate evidence.
```

## Continue

Next Phase 8 lanes should keep release/README/UI copy tied to the dry-run ledger and independent review result from Phase 8-D. If Phase 8-D does not find a complete certificate stack, do not advertise first-principles full-3D scope; use "engineering probe", "preview", "source-gated", and "fail-closed" wording.

Suggested follow-up if a later reviewer wants broader coverage: add a small claim-surface scanner test that checks user-facing Markdown/UI files for stale phrases such as `publication quality`, `Validated against:`, and `emerge from first principles` outside explicitly historical/audit contexts.
