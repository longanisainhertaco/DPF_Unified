# Unknown and Inference Log — Submission 1

Every inferred parameter, reconstruction, missing value, or unresolved ambiguity
in Sprint 1. Sprint 1 is control-gate engineering, so no physics parameters were
inferred; the entries below are engineering-decision inferences.

## I-1 — CI `--date` pin

- Inference: CI pins `--date 2026_05_18` on the two read-only verification gates.
- Why: the audit's manual command list passes no `--date`, so it defaults to
  UTC today. On any CI run after 2026-05-18 the default would look for a
  `..._<later-date>.json` file that does not exist and fail with `MISSING`.
  Pinning the date makes CI compare against the committed baseline docs
  deterministically.
- Owner: control-gate / CI.
- Next action: when source-truth or module-vetting content changes, regenerate
  the dated docs (writing mode) and bump the `--date` pin in
  `.github/workflows/ci.yml` to the new baseline date.

## I-2 — `generated_at_utc` reduced to date granularity

- Inference: the per-invocation sub-second wall-clock `generated_at_utc` field in
  the two verification gates' output was reduced to the `--date` slug.
- Why: the audit observed the gates "changed only timestamps" in CI. Date
  granularity is sufficient provenance for a dated doc whose filename already
  carries the date; the wall-clock component was pure churn. No other field was
  excluded from the `--check` comparison — the churn source was removed, not
  masked.
- Owner: control-gate.
- Next action: none; resolved.

## I-3 — A-2 quarantine vs regeneration

- Inference: the three stale audit artifacts were quarantined rather than
  regenerated.
- Why: an artifact committed to git cannot embed its own containing commit hash,
  so any committed active artifact technically predates HEAD; and all three were
  generated with `dirty_worktree=true`, which voids reproducible provenance
  regardless of commit. Quarantine is explicitly offered by audit A-2 and
  eliminates the ambiguity.
- Owner: control-gate.
- Full rationale: `sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`.

## I-4 — A-6 sidecar mechanism

- Inference: cumulative-ledger continuity on resume uses a per-checkpoint sidecar
  JSON, not ledger counters embedded in the checkpoint `.npz`.
- Why: the audit offered both. The sidecar keeps the change contained to
  `segmented_whole_shot.py` and leaves the checkpoint schema and its fail-closed
  loader untouched (smallest blast radius).
- Owner: runtime engineering.
- Full rationale: `sprint_1/RESUME_LEDGER_CONTINUITY_PROPOSAL.md` section 5.

## I-5 — `partial` status mapping

- Inference: `DPF-PHYS-020` and `DPF-PHYS-023` were set to status `partial`.
- Why: the audit instructed "partial/candidate" status. The RTM status enum is
  {blocked, deferred, implemented, partial, planned, rejected} — there is no
  literal `candidate` value. `partial` is the enum value meaning
  candidate-exists-but-acceptance-blocked. The audit forbids promotion to
  `implemented`.
- Owner: control-gate / SRS.
- Next action: none; resolved.

## I-6 — C7 required-field list duplication

- Inference: the strengthened linter check C7 in
  `scripts/audit_first_principles_artifacts.py` carries a local copy of the
  required-provenance field tuple rather than importing
  `REQUIRED_PROVENANCE_FIELDS` from `dpf.first_principles.manifest`.
- Why: the linter is a standalone script; a local tuple avoids a script→package
  import coupling at audit time. The local tuple has a comment naming the
  manifest module as the source of truth.
- Owner: control-gate.
- Open question: if the two lists drift, C7 weakens silently. A follow-up could
  add a unit test asserting the linter's tuple equals
  `manifest.REQUIRED_PROVENANCE_FIELDS`. Not blocking for Sprint 1.

## Unresolved source gaps

None for Sprint 1. Sprint 1 consumes no `KnowledgeReference/` physics sources;
all source gaps belong to Sprint 2 (WP-N1B) and Sprint 3 (WP-N2/N3/N5/N6/N7).
