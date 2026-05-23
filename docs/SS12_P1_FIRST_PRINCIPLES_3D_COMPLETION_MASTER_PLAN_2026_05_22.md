# SS12 P1 First-Principles 3D Completion Master Plan

Date: 2026-05-22 UTC
Repository: `/Users/anthonyzamora/dpf-unified`
Knowledge bases: `/Users/anthonyzamora/Desktop/HeliosMatrix_KB`, `/Users/anthonyzamora/PDFs`, repository `KnowledgeReference/`
Current HEAD at plan start: `2ebe07d`
Controlling authorization packet: `docs/CODEX_SUPER_SPRINT12_P0_AUDIT_AND_P1_AUTHORIZATION_2026_05_21.md`

## Mission

Move DPF Unified toward an honest physics, first-principles, full-3D simulation tool.

This plan is intentionally fail-closed. It does **not** permit acceptance-promotion by optimism, architecture polish, or cross-scope inference. Every phase must evaluate, learn, update the evidence state, and continue only when gates pass.

## Non-negotiable guardrails

1. Local source authority only: accepted physics evidence must come from repository `KnowledgeReference/`, `/Users/anthonyzamora/PDFs`, or reviewed HeliosMatrix_KB-derived line-cited extractions.
2. HeliosMatrix_KB is retrieval and extraction support, not an authority substitute.
3. Same-scope PF-1000 full-energy evidence is the P1 control surface:
   - validation scope: `pf1000_full_energy_27_to_40_kv`
   - selected-machine source scope: `pf1000_scholz_2000_2001_24rod_large_electrode_full_energy_source`
4. `accepted_runtime_claim`, `can_support_first_principles_acceptance`, and certificate/promoted acceptance flags remain false unless the complete certificate stack passes at one commit.
5. Reduced models, imported PIC, Bennett startup, LLNL-like hybrid PIC, cross-scope diagnostics, and raw PDFs remain context only unless a reviewed transfer rule explicitly authorizes them.
6. No GPU/MPS/ollama contention: HeliosMatrix_KB jobs are sequential. Current active job PID 47875 must finish before WS4/WS5 or any new MPS/ollama run.
7. TDD for code changes: failing test first, then minimal implementation, then focused and integration verification.
8. Every phase ends with an evaluation memo: what changed, what failed, what was learned, next phase decision.

## Phase gates

### Phase 0 — Baseline and resource gate

Goal: freeze the starting state and avoid trampling the active corpus/eval work.

Tasks:
- Record git HEAD, branch, changed-file summary, active jobs, and key artifact status.
- Wait for or monitor current Helios gold eval job before starting any new GPU/MPS/ollama work.
- Snapshot `CortexFindings.md`, `CodexFindings.md`, and P1 authorization packet references.

Exit evidence:
- baseline memo with HEAD, dirty-state summary, active job list, and blocked resource notes.
- no acceptance flags changed.

### Phase 1 — P1-0 active-artifact linter policy cleanup

Goal: reconcile linter wording and implementation before physics-source work.

Decision:
- Preferred current choice: document current behavior rather than strict context-key-only enforcement.
- Current contract: architecture/closure source slugs may exist in ordinary non-`same_scope` source fields; they may not appear under `same_scope` / `same_scope_source` evidence key chains.

Tasks:
- Update linter docstring / authority policy / memo language.
- Add regression tests proving protected same-scope chains still fail closed.
- Add tests or documentation showing ordinary non-same-scope source fields are intentionally permitted under the current contract.

Exit evidence:
- focused linter-policy tests pass.
- active result hygiene script passes strict check.
- no acceptance flags changed.

### Phase 2 — PF-1000 full-energy same-scope source packet matrix

Goal: establish source truth before implementation.

Required channels:
- geometry
- bank/circuit
- gas/fill
- current waveform
- startup
- density history
- EM field history
- temperature or distribution history
- neutron scalar yield
- neutron timing
- neutron spectrum
- anisotropy
- detector response
- uncertainty budget
- review certificate

Tasks:
- Use HeliosMatrix_KB retrieval and `/Users/anthonyzamora/PDFs` to identify candidate sources.
- Convert candidates into line-cited records rooted in local source files.
- Produce a machine-readable source matrix with accepted/blocked status and reasons.
- Keep absent channels explicitly blocked.

Exit evidence:
- source packet matrix with source path, line range, value/observable, scope-match decision, uncertainty status, and blocked reason.
- source-truth exhaustion remains clean.
- no acceptance flags changed.

### Phase 3 — Numerical-fidelity acceptance suite

Goal: prove the 3D solver numerics before any physics acceptance claim.

Tasks:
- Source-backed method map.
- Analytic/manufactured cases.
- Mesh convergence families.
- Timestep convergence families.
- Backend/precision parity.
- Restart reproducibility.
- Limiter-zero proof.
- Per-observable norms and tolerances.

Exit evidence:
- numerical-fidelity test matrix and results.
- Tier-3 code verification packet updated without pretending experimental validation.

### Phase 4 — Field-derived circuit and power-port coupling

Goal: replace or clearly bound density-weighted / Lee-style coupling for first-principles claims.

Tasks:
- Define reviewed interface surface or volume domain.
- Implement Poynting or `J dot E` power integral where evidence supports it.
- Map Auluck Eq.6 terms II/IV/V/VI with sign convention, time-centering, and residual tolerance.
- Export snowplow-loaded, blended, and field-coupled interval labels.

Exit evidence:
- coupling packet and tests.
- unsupported intervals remain explicitly blocked.

### Phase 5 — Startup BVP same-scope closure

Goal: move startup from context-only toward source-backed BVP where possible.

Tasks:
- D2 breakdown/preionization evidence.
- Insulator flashover evidence.
- Sheath liftoff evidence.
- Early circuit handoff interval.
- Same-scope geometry/material surface inputs.

Exit evidence:
- startup packet says accepted only for fully line-cited same-scope inputs; otherwise remains blocked.

### Phase 6 — Spatial, thermodynamic, and neutron validation packets

Goal: close Tier-4/Tier-5 evidence without scalar-only shortcuts.

Tasks:
- Density/emission geometry and timing.
- EM field or magnetic probe history.
- Temperature/distribution history.
- Mechanism-separated neutron scalar yield, timing, spectrum, anisotropy, detector/activation response.
- Shared uncertainty budget.

Exit evidence:
- Tier-4 and Tier-5 packets accepted only if all components share one reviewed scope; otherwise explicitly blocked.

### Phase 7 — Comparator/UQ/certificate skeleton

Goal: make certification possible and fail-closed.

Tasks:
- Output-field mapping by observable.
- Measurement/model/numerical uncertainty.
- Pass/fail metrics and tolerances.
- Negative controls.
- Run/evidence hashes.
- Independent-review placeholders.

Exit evidence:
- certificate schema exists but refuses to emit accepted certificate until all gates pass.

### Phase 8 — Integration, release, and final acceptance decision

Goal: determine whether DPF Unified can honestly advertise first-principles full-3D scope.

Tasks:
- Run full focused first-principles suite.
- Run source-truth, module-vetting, artifact hygiene, acceptance dry-run, periodic audit.
- Compare against CortexFindings/CodexFindings unresolved scientific blockers.
- Update docs and user-facing claim surfaces.

Exit evidence:
- exact commit hash and changed-file list.
- dry-run ledger output.
- source-truth and module-vetting output.
- periodic audit log path.
- explicit acceptance flag statement: still false with blockers, or true only with complete certificate evidence.

## Evaluation loop after every phase

1. Evaluate: run phase-specific tests, source checks, hygiene checks, and audit commands.
2. Learn: write what failed, what evidence was missing, and what code/science assumption changed.
3. Continue: update the next phase scope. If a blocker is fundamental, preserve it as a blocker instead of coding around it.

## Immediate execution order

1. Phase 0 baseline memo.
2. Phase 1 P1-0 cleanup.
3. Helios gold-eval completion review once PID 47875 exits.
4. Phase 2 source packet matrix, assisted by HeliosMatrix_KB and `/Users/anthonyzamora/PDFs`.
5. Re-evaluate before any solver/coupling/startup implementation.
