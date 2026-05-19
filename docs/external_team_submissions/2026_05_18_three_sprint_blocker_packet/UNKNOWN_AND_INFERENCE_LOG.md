# Unknown and Inference Log — Submission 1 + Sprint 1.1 + Sprint 2

Every inferred parameter, reconstruction, missing value, or unresolved ambiguity
across all sprints. Sprint 1 is control-gate engineering, so no physics
parameters were inferred; the Sprint 1 entries are engineering-decision
inferences. Sprint 2 entries cover source-sourcing decisions and physics
clarifications from the Auluck 2021 extract.

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

## Sprint 1 Unresolved source gaps

None for Sprint 1. Sprint 1 consumes no `KnowledgeReference/` physics sources;
all source gaps belong to Sprint 2 (WP-N1B) and Sprint 3 (WP-N2/N3/N5/N6/N7).

---

## Sprint 2 Inferences

## I-7 — Auluck OCR-garbled extract vs verified-PDF extract

- Inference: the auto-extracted `KnowledgeReference/` markdown for Auluck 2021
  was superseded by a transcript read verbatim from the on-disk primary PDF
  (pages 6–11).
- Why: the auto-extracted KR markdown renders eqs. (2)–(14) as OCR-garbled
  tokens (e.g., subscripts collapsed, sign characters mangled). The verified
  transcript captures eqs. (1)–(14), the six-term power balance, the `Sigma_p`
  moving-boundary integrands, the symbol map, and the sign convention in legible
  form. The OCR extract was not deleted — it remains in `KnowledgeReference/` as
  the indexed source — but the verified transcript is the operative reference for
  WP-N1B implementation.
- Owner: WP-N1B / power-port.
- Verified correction: eq. (1) carries a leading minus sign that the OCR extract
  dropped.
- Next action: WP-N1B implementation reads from the verified-PDF transcript.

## I-8 — WP-N1B "electrode/interface work" term retraction

- Inference: the audit's WP-N1B finding that the implementation must account for
  an "electrode/interface work" term is a category error.
- Why: Auluck eq. (6) is a six-term power balance (I. stored magnetic, II.
  motional magnetic, III. stored electric, IV. motional electric, V. resistive,
  VI. anomalous). There is no electrode-contact-work term in eq. (6); Auluck
  explicitly excludes the electrode interface from the domain bounded by
  `Sigma_p`. The retraction is verified against the PDF (pages 6–8, eqs. (5)
  and (6)).
- Owner: WP-N1B / power-port.
- Next action: WP-N1B ledger implements exactly six terms per eq. (6); any
  electrode-work term would be a source-unsupported addition.

## I-9 — Residual tolerance: no local KR source

- Inference: no `KnowledgeReference/` file provides a numeric residual-tolerance
  value for the WP-N1B power-port convergence criterion.
- Why: a systematic search of the KR corpus found no paper that states an
  explicit dimensionless residual tolerance for the Auluck power-port balance.
  The tolerance is therefore an open source gap that cannot be filled from local
  sources without a fresh literature fetch or experimental calibration.
- Owner: WP-N1B / power-port.
- Status: unresolved. Power-port acceptance is blocked in part on this gap.
- Next action: if a source is found, add it to the KR extract and record the
  citation here; otherwise document as an engineering choice with explicit
  rationale when the implementation is built.

## I-10 — 12 us compute-wall per-step measurement

- Inference: the per-step compute floor was measured at 5.23 ms on the compact
  grid; 12 us at `dt = 1e-13 s` requires 120,000,000 steps; the production
  wall-clock is therefore blocked on the WP-N3 production grid size, not the
  scheduler.
- Why: the measurement was made with the compact DPF grid (not the WP-N3
  production geometry); the step floor sets a lower bound of ~120 days
  wall-clock for the full 12 us run at that rate. Production compute requirements
  cannot be estimated until WP-N3 geometry is fixed.
- Owner: WP-N4B / 12 us orchestration.
- Next action: re-measure per-step time after WP-N3 grid is reviewed; update the
  WP-N4B orchestration proposal with the production estimate.

## I-11 — Tracked copy of Auluck extract in sprint_2/

- Inference: the verified Auluck 2021 equation extract is tracked as
  `sprint_2/AULUCK_2021_POWER_BALANCE_EQUATIONS_VERIFIED.md` in the submission
  packet, duplicating the file ingested into `KnowledgeReference/`.
- Why: `KnowledgeReference/` is gitignored; without the tracked copy, the
  WP-N1B audit evidence is not version-controlled and is unavailable to reviewers
  who do not have local KR access. The packet must be self-contained.
- Owner: packet hygiene.
- Next action: if `KnowledgeReference/` is ever added to version control, the
  tracked copy in `sprint_2/` becomes redundant; leave it as a convenience copy
  until then.
