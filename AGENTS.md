# DPF-Unified Agent Operating Contract

This file is the repo-local operating contract for coding agents working on
DPF-Unified. It is not source code, not a scientific source, and not validation
evidence. It exists to keep future work aligned with the project source of
truth, current blockers, and verification rules.

## Scope And Precedence

- Applies to the full repository unless a future nested `AGENTS.md` gives more
  specific instructions for a subtree.
- User and system instructions override this file.
- For science and physics claims, the only allowed authority is local material
  under `KnowledgeReference/`.
- Outside websites, package documentation, and research papers not present in
  `KnowledgeReference/` may support tooling, workflow, or acquisition planning
  only. They cannot support DPF physics claims or validation pass/fail status.
- If this file conflicts with `CodexFindings.md`, `CortexFindings.md`, or
  `docs/MODULE_AUDIT/BACKLOG.md`, read the current repo files and preserve the
  stricter fail-closed interpretation until the documents are reconciled.

## Required First Reads

Before changing code, tests, scientific validation docs, or status plans:

1. Read `CodexFindings.md`.
2. Read `CortexFindings.md`.
3. Read `docs/MODULE_AUDIT/BACKLOG.md`.
4. Read the relevant module audit note under `docs/MODULE_AUDIT/`, when one
   exists for the touched area.
5. Read `README.md`, `pyproject.toml`, and the specific source/tests touched by
   the task.
6. For scientific closure, read the exact local `KnowledgeReference/` source
   files or accepted local evidence packets named by the task.

Use `rg` or `rg --files` first for repo search.

## Source-Of-Truth Hierarchy

Use this hierarchy when classifying claims:

1. User/system instructions.
2. Local `KnowledgeReference/` corpus for science and physics.
3. Accepted local validation packets, source manifests, and digitization review
   records that point back to `KnowledgeReference/`.
4. `CodexFindings.md`, `CortexFindings.md`, and
   `docs/MODULE_AUDIT/BACKLOG.md` for active status and planning.
5. Code and tests for current behavior and regression protection.
6. Draft digitizations, synthetic fixtures, WALRUS/generated data, engineering
   probes, and smoke tests for non-promoting implementation evidence only.

Do not use generated data, synthetic diagnostics, WALRUS training files, or
passing smoke tests as scientific validation evidence.

## Evidence States

Use these exact status concepts in code and docs:

- `missing`: required source artifact or extracted value is absent.
- `draft` or `candidate`: an artifact exists but cannot support validation.
- `blocked_by_review`: an artifact exists but independent review has not
  accepted it.
- `accepted`: independently reviewed, same-scope, source-backed evidence that
  can support validation decisions for its declared scope.
- `engineering_probe`: useful for implementation, stability, or performance;
  not scientific validation.
- `synthetic_only`: useful for schema, UI, smoke, or numerical harness tests;
  not scientific validation.
- `not_validation_evidence`: explicit non-promoting lane for metadata,
  exports, calibration fits, guardrails, and scaffolds.

Allowed promotion path:

`missing -> draft/candidate -> blocked_by_review -> accepted`

`engineering_probe`, `synthetic_only`, and `not_validation_evidence` do not
promote into `accepted`. They may justify engineering readiness only.

## Current Hard Blockers

As of 2026-05-11, preserve these blockers unless same-scope local evidence and
tests close them:

- Akel Fig. 1 current waveform is `blocked_by_review`; internal overlay residual
  is not independent acceptance.
- S1/S2 waveform and current-dip validation remain blocked until accepted
  same-scope Akel 16 kV shot-12581 waveform evidence with uncertainty exists.
- Akel Figs. 2-6 digitization/review packets are still needed for remaining
  waveform and yield source closure.
- Tier 2, Tier 4, and Tier 5 validation need same-scope target packets before
  they can pass.
- Diagnostics formulas for Thomson, nTOF spectrum, x-ray filter/emissivity,
  regime, instability, plasmoid, shear, runaway, anisotropy, detector response,
  and uncertainty remain source-blocked unless local accepted packets exist.
- Radiation, p-B11, QMF, ionization, ablation, line-cooling, detector response,
  neutron spectrum, and anisotropy claims remain non-predictive unless
  source-closed in the local corpus.
- WALRUS/The Well/CATS/checkpoint/model-card/license material remains blocked
  until local source/provenance packets are acquired, reviewed, and hashed.

## Task Classes

Classify each task before acting:

- Scientific closure: `KnowledgeReference` acquisition, digitization, target
  extraction, same-scope validation, UQ, or acceptance criteria.
- Engineering closure: solver stability, backend parity, manifests, export
  behavior, fail-closed metadata, UI/API routing, or test-lane separation.
- Product/SRS closure: requirements, traceability, release gates, certificates,
  run manifests, security, local-first behavior, or documentation controls.
- Status hygiene: findings/backlog synchronization, stale-note supersession,
  source queue cleanup, and agent operating notes.

Scientific closure requires source-backed local evidence and should fail closed
when evidence is absent. Engineering closure may proceed with tests and probes,
but must keep scientific claims blocked.

## Completion Rules

A task is complete only when all applicable items are true:

- The changed behavior is implemented or the blocker is explicitly preserved.
- Tests or static checks cover the new behavior or fail-closed state.
- Scientific claims cite local `KnowledgeReference/` or accepted local evidence.
- No draft, synthetic, generated, or probe evidence is promoted to validation.
- `CodexFindings.md`, `CortexFindings.md`, and
  `docs/MODULE_AUDIT/BACKLOG.md` are updated when status, blockers, or task
  completion changes.
- Exact failure strings, probe values, and historical blocker evidence are
  preserved through dated addenda rather than deleted.

## Verification Command Matrix

Run the narrowest meaningful checks for the work performed, then add broader
checks when shared behavior is touched.

Documentation/status-only changes:

```bash
git diff --check -- AGENTS.md CodexFindings.md CortexFindings.md docs/MODULE_AUDIT/BACKLOG.md
```

Diagnostics guardrails:

```bash
python3 -m pytest tests/test_diagnostics_evidence_manifest.py tests/test_diagnostics_test_lanes.py tests/test_beam_tracker.py tests/test_export_scope.py -q
```

Preset/source-scope guardrails:

```bash
python3 -m pytest tests/test_preset_source_scope.py -q
```

Radiation/QMF metadata:

```bash
python3 -m pytest tests/test_qmf_suppression.py tests/test_radiation_model_metadata.py -q
```

Validation artifact and readiness propagation:

```bash
python3 -m pytest tests/test_validation_artifacts.py tests/test_export_scope.py tests/test_server_readiness.py -q
```

MHD numerical packet and backend parity evidence:

```bash
python3 -m pytest tests/test_mhd_numerical_fidelity.py tests/test_mlx_circuit_coupling.py tests/test_mlx_timestepper.py -q
```

Broad focused guardrail slice:

```bash
python3 -m pytest tests/test_diagnostics_evidence_manifest.py tests/test_diagnostics_test_lanes.py tests/test_beam_tracker.py tests/test_export_scope.py tests/test_preset_source_scope.py tests/test_qmf_suppression.py tests/test_radiation_model_metadata.py -q
```

For Python source changes, also compile the touched source and tests:

```bash
python3 -m py_compile <touched-python-files>
```

If a check cannot run, record the exact command and failure reason in the final
answer and in the relevant findings document when the failure changes project
status.

## Module Routing

- `src/dpf/validation/`: source authority, digitization, artifacts, validation
  tiers, UQ, readiness, certificates. Fail closed by default.
- `src/dpf/diagnostics/`: separate engineering smoke, synthetic-only,
  source-blocked, and source-backed lanes. Current diagnostics manifest has no
  accepted validation entries unless future source packets change that.
- `src/dpf/radiation/`: keep QMF, p-B11, line cooling, ionization, ablation,
  and neutron helpers conservative unless source-closed.
- `src/dpf/metal/`: keep MLX/Metal preview behavior separate from scientific
  validation; distinguish backend-owned operators from Python-owned operators.
- `src/dpf/fluid/` and `src/dpf/circuit/`: preserve Lee/snowplow scope labels,
  current-factor loading boundaries, and empirical post-pinch resistance labels.
- `src/dpf/engine/`: keep `production` backend labels separate from validation
  readiness; preserve first-failure telemetry and explicit fallback behavior.
- `src/dpf/io/` and `src/dpf/ai/`: treat Well/WALRUS artifacts as local
  interchange or training candidates unless source/provenance is accepted.
- `src/dpf/server/`, `gui/`, and top-level apps: display source-authority,
  readiness, backend, and Preview/Reference labels without promoting claims.
- `tests/`: keep synthetic tests, engineering smoke tests, source-blocked tests,
  and source-backed validation tests separate.

## Agent Delegation Rules

When multiple agents are authorized:

- Assign non-overlapping ownership by module or file set.
- Give each agent the current blocker state and this file as context.
- Do not let one agent promote evidence based on another agent's draft output.
- Require each agent to report changed files, tests run, and remaining blockers.
- The coordinating agent performs final checks against `CodexFindings.md`,
  `CortexFindings.md`, `docs/MODULE_AUDIT/BACKLOG.md`, and this file.

## Nested AGENTS.md Policy

Add nested `AGENTS.md` files only when a subtree needs rules that are more
specific than this root file. Good candidates are:

- `src/dpf/validation/AGENTS.md`
- `src/dpf/diagnostics/AGENTS.md`
- `src/dpf/radiation/AGENTS.md`
- `src/dpf/metal/AGENTS.md`
- `src/dpf/io/AGENTS.md`
- `gui/AGENTS.md`
- `tests/AGENTS.md`

Nested files should be short, should inherit this root contract, and should not
duplicate scientific source material.

## Maintenance Rules

Update this file when any of these change:

- Source-of-truth policy.
- Evidence state names or promotion rules.
- Required first-read files.
- Verification commands.
- Module ownership/routing.
- Active blocker categories.
- Multi-agent operating expectations.

Do not rewrite this file into a broad project specification. Keep scientific
details in `KnowledgeReference/`, implementation details in code/tests, and
status in the findings/backlog documents.

## Non-Science Reference Sources For This File

The structure of this file follows general `AGENTS.md` patterns observed in
public repositories and guidance, including root-scope instructions, concise
verification commands, nested-file routing, and repository-specific guardrails.
These references are workflow inputs only:

- `https://agents.md/`
- `https://github.com/openai/codex/blob/main/AGENTS.md`
- `https://github.com/apache/airflow/blob/main/AGENTS.md`
- `https://github.com/openai/openai-agents-python/blob/main/AGENTS.md`
- `https://github.com/cloudflare/agents/blob/main/AGENTS.md`
- `https://github.com/openclaw/openclaw/blob/main/AGENTS.md`
- `https://github.com/pydantic/pydantic-ai/blob/main/AGENTS.md`
