# SRS / RTM Baseline Decision (A-7)

Audit finding: A-7. Commit: `fe038f7`.
Date: 2026-05-18.

## The audit's choice

A-7 requires: "Either initialize Doorstop from `DPF-PHYS-020..026`, or add a
dated decision that CSV/JSON remain the temporary review baseline."

## Decision

The CSV and JSON traceability exports remain the temporary review baseline.
Doorstop is **not** initialised in this sprint.

- `docs/DPF_REQUIREMENTS_BASELINE.md` — the human-authored source of truth.
- `docs/SRS_TRACEABILITY_MATRIX.csv` and `docs/SRS_TRACEABILITY_MATRIX.json` —
  generated from the baseline by `scripts/export_srs_traceability.py`, the
  review baseline an external reviewer reads.
- `SRS_TRACEABILITY_MATRIX.json` retains `doorstop_status: staged_not_imported`
  and per-requirement `doorstop_import` blocks, so a future Doorstop import is
  pre-staged but not performed.

## Rationale

1. **Scope discipline.** Sprint 1 is control-gate hardening. Adopting Doorstop
   is a tooling decision with its own workflow, directory layout, and review
   implications; bundling it into a control-gate sprint would mix concerns the
   audit explicitly tells the team to keep separate.

2. **The export path already works.** `scripts/export_srs_traceability.py`
   deterministically regenerates both exports from the baseline, and
   `tests/test_srs_traceability_export.py` gates the result (including, as of
   this sprint, assertions pinning the `DPF-PHYS-020` and `DPF-PHYS-023`
   statuses). The review baseline is reproducible and tested today.

3. **No information is lost.** The JSON `doorstop_import` blocks and
   `import_guardrail` fields carry everything a later Doorstop import needs,
   including the guardrail "import as satisfied only when status is implemented
   and evidence is present."

## Conditions to revisit

Initialise Doorstop when either holds:

- A requirement reaches `implemented` with same-scope acceptance evidence and
  needs formal Doorstop linking/verification tracking; or
- the requirement set is handed to an external engineering firm that mandates a
  Doorstop-managed baseline.

Until then, regenerate the exports with `scripts/export_srs_traceability.py`
after every edit to `docs/DPF_REQUIREMENTS_BASELINE.md`, and treat the CSV/JSON
pair as the review baseline.
