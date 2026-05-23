# DPF Unified Full Project Kanban Plan

> **For Hermes:** Execute through the `dpf-unified` Kanban board using `dpfcodex` for implementation, `dpfclaude` for independent review, and `dpforch` for orchestration. Preserve fail-closed acceptance semantics at every sprint.

**Goal:** Move DPF Unified from fail-closed scaffolding to a source-backed, reviewable, certificate-gated DPF simulation/research platform without overstating first-principles/full-3D acceptance.

**Architecture:** The project proceeds through sprint-level Kanban lanes. Each implementation sprint has a mandatory independent review and fix/reverify task. Later sprints are dependency-gated by earlier review/fix tasks, not optimism.

**Tech Stack:** Python, pytest, ruff, local PDF/KnowledgeReference corpus, `/Users/anthonyzamora/Desktop/heliosmatrix_kb` for retrieval support only, Hermes Kanban/Cron, GitHub PR workflow, source-packet JSON/Markdown artifacts.

**KB path note:** macOS resolves `/Users/anthonyzamora/Desktop/HeliosMatrix_KB` and `/Users/anthonyzamora/Desktop/heliosmatrix_kb` to the same APFS directory/inode on this machine. Use the lowercase `heliosmatrix_kb` spelling in new sprint outputs to match the new KB name and avoid ambiguity.

---

## Global non-negotiables

1. No public or runtime first-principles/full-3D acceptance claim unless a complete same-commit certificate stack passes.
2. Local source authority only: repository `KnowledgeReference/`, `/Users/anthonyzamora/PDFs`, or reviewed `/Users/anthonyzamora/Desktop/heliosmatrix_kb`-derived local line-cited extractions.
3. Retrieval is not authority. KB hits are discovery candidates until line-cited and reviewed.
4. Same-scope PF-1000 full-energy scope remains the control surface unless a reviewed plan changes it.
5. All sprints must end with Evaluate/Learn/Continue evidence.
6. All code sprints use TDD: failing test, minimal implementation, focused verification, integrated verification.
7. No destructive corpus cleanup; avoid symlink/PDF normalization unless explicitly approved.
8. Heavy corpus/GPU/MPS/Ollama jobs must be inventoried before launching anything resource-heavy.
9. Every review task must search for acceptance-promotion leaks and provenance weakness.
10. Every fix task must rerun the sprint's verification plus acceptance scans.

## Sprint dependency map

- SS12R Release Hygiene gates the rest.
- SS13 Numerical Fidelity and SS14 Source Packet Expansion may run after SS12R.
- SS15 Power-Port Coupling depends on SS13 + SS14.
- SS16 Startup BVP, SS17 Spatial/Thermo Validation, and SS18 Neutron Diagnostics depend on SS14.
- SS19 UQ/Comparator/Certificate depends on SS13-SS18.
- SS20 Integration Acceptance Dry-Run depends on SS19.
- SS21 Product Claim/Release Decision depends on SS20.
- SS22 Research/Ops Packaging depends on SS21.

## Risk register

### R1 — False acceptance promotion
- **Probability:** High
- **Impact:** Critical
- **Triggers:** any `accepted_*`, `promotes_acceptance`, or `can_support_first_principles_acceptance` turns true without complete evidence.
- **Mitigation:** acceptance scanners, dry-run ledger, cross-packet acceptance shield, independent review every sprint.
- **Owner:** `dpfclaude` review + `dpforch` governance.

### R2 — Provenance fabrication or weak quote matching
- **Probability:** Medium
- **Impact:** Critical
- **Triggers:** quote fields are summaries, spans are too wide, source files missing, OCR lines do not match.
- **Mitigation:** strict source-truth validators, narrow line ranges, verbatim quote checks, reviewer spot audits.
- **Owner:** `dpfcodex` implementation, `dpfclaude` review.

### R3 — Cross-scope evidence creep
- **Probability:** High
- **Impact:** High
- **Triggers:** PF-1000 claims supported by other DPF machines, reduced models, or context-only papers.
- **Mitigation:** same-scope labels, transfer-candidate layer, blocked/candidate status until reviewed transfer rule exists.
- **Owner:** `dpforch` + reviewers.

### R4 — Numerical correctness debt hidden by source work
- **Probability:** Medium
- **Impact:** High
- **Triggers:** experimental packets improve while solver convergence/restart/backend parity remains weak.
- **Mitigation:** SS13 before certificate; numerical-fidelity packet required for SS19.
- **Owner:** `dpfcodex`.

### R5 — Corpus/resource contention
- **Probability:** Medium
- **Impact:** Medium/High
- **Triggers:** simultaneous Helios/Ollama/GPU/MPS jobs, large PDF processing, indexing runs.
- **Mitigation:** resource inventory before heavy jobs; sequential corpus extraction; prefer lightweight validators.
- **Owner:** `dpforch`.

### R6 — Dirty-tree/release hygiene collapse
- **Probability:** High now
- **Impact:** High
- **Triggers:** PDF symlink/type-change noise, uncommitted artifacts, staged external corpus drift.
- **Mitigation:** SS12R isolates intentional files, no blind `git add -A`, verification from committed tree.
- **Owner:** `dpfcodex`.

### R7 — CI failures unrelated to branch
- **Probability:** Medium
- **Impact:** Medium
- **Triggers:** existing repo-wide failing checks.
- **Mitigation:** compare base branch, identify branch-caused deltas, patch only scoped failures.
- **Owner:** `dpfcodex`.

### R8 — Scope explosion
- **Probability:** High
- **Impact:** High
- **Triggers:** attempting universal DPF predictive validation instead of scoped certificate.
- **Mitigation:** sprint exit criteria require bounded scope; out-of-scope findings become backlog/blockers.
- **Owner:** `dpforch`.

### R9 — Reviewer rubber-stamping
- **Probability:** Medium
- **Impact:** High
- **Triggers:** review says pass without checking source paths, quotes, flags, and tests.
- **Mitigation:** review tasks include explicit PASS/REQUEST_CHANGES contract and required audit checklist.
- **Owner:** `dpfclaude` bridge.

### R10 — Product overclaim after partial success
- **Probability:** Medium
- **Impact:** Critical
- **Triggers:** README/UI/PR language says predictive, publication-grade, validated, or full-3D accepted too early.
- **Mitigation:** SS21 claim-surface scan; keep engineering-probe/source-gated wording unless certificate passes.
- **Owner:** `dpfclaude` + `dpfcodex`.

## Sprint definitions

### SS12R — Release Hygiene and PR Packaging
**Objective:** Safely package SS12/Phase 8 work into a clean commit/PR without committing corpus noise.
**Key outputs:** clean staged file list, commit, verification log, pushed branch, PR, CI triage.
**Verification:** focused suite, source-truth, module-vetting, artifact hygiene, dry-run ledger, JSON scan, `git diff --check`.
**Risks handled:** R1, R6, R7, R10.

### SS13 — Numerical Fidelity Closure
**Objective:** Prove numerical behavior independently of experimental acceptance.
**Key outputs:** manufactured/analytic tests, mesh/timestep convergence, backend/precision parity, restart reproducibility, limiter-zero proof, numerical-fidelity packet.
**Verification:** focused numerical tests and packet validator.
**Risks handled:** R4.

### SS14 — PF-1000 Same-Scope Source Packet Expansion
**Objective:** Expand line-cited source packets for required PF-1000 full-energy channels.
**Key outputs:** channel matrix with status, source path, line range, quote, uncertainty, blocked reason; rejected/candidate transfer rows.
**Verification:** source path/line/quote validator; no accepted rows without complete criteria.
**Risks handled:** R2, R3, R5, R8.

### SS15 — Power-Port and Circuit Coupling Evidence Closure
**Objective:** Replace/bound convenience coupling with reviewed field-derived power-port evidence where possible.
**Key outputs:** Poynting or `J·E` integral design/implementation, sign/time-centering conventions, residual checks, interval labels.
**Verification:** coupling tests, residual tests, source packet checks, acceptance shield.
**Risks handled:** R1, R3, R4.

### SS16 — Startup BVP Evidence Closure
**Objective:** Close or explicitly block startup evidence: breakdown, preionization, flashover, liftoff, handoff.
**Key outputs:** startup packet, BVP scaffold/tests only where source-supported, blocked rows for missing inputs.
**Verification:** startup validators, provenance checks, acceptance scan.
**Risks handled:** R2, R3, R8.

### SS17 — Spatial/Thermodynamic Validation Packets
**Objective:** Build same-scope packets for density/emission geometry, timing, EM field history, temperature/distribution.
**Key outputs:** observable mapping, comparator stubs, uncertainty annotations, blocked/candidate status.
**Verification:** packet validators and no scalar-only acceptance shortcuts.
**Risks handled:** R2, R3, R8.

### SS18 — Neutron Diagnostic Validation Stack
**Objective:** Build mechanism-separated neutron evidence: yield, timing, spectrum, anisotropy, detector/activation response.
**Key outputs:** neutron packet, detector response mapping, uncertainty/diagnostic channel blockers.
**Verification:** neutron packet validator, diagnostic completeness check, acceptance scan.
**Risks handled:** R1, R2, R3, R8.

### SS19 — UQ, Comparator, and Certificate Pipeline
**Objective:** Turn scaffolds into an actual certificate pipeline that can refuse or accept scoped claims.
**Key outputs:** comparator mapping, uncertainty propagation, run/source hashes, negative controls, certificate emitter/refusal path.
**Verification:** certificate refuses incomplete stack; accepts only synthetic fully-complete fixture; no production acceptance unless evidence exists.
**Risks handled:** R1, R4, R9.

### SS20 — Full Integration Acceptance Dry-Run
**Objective:** Run complete gates from clean commit and produce final dry-run ledger.
**Key outputs:** exact commit hash, changed-file list, validation logs, dry-run ledger, blocker decision.
**Verification:** full focused suite, source-truth, module-vetting, artifact hygiene, periodic audit, claim scan.
**Risks handled:** R1, R6, R7, R10.

### SS21 — Product Claim Surface and Release Decision
**Objective:** Align README/UI/docs/release language with actual certificate outcome.
**Key outputs:** release decision memo, public claim scan, UI/README guardrails, PR/release notes.
**Verification:** claim hygiene tests, reviewer approval.
**Risks handled:** R10.

### SS22 — Research/Ops Packaging and Long-Run Roadmap
**Objective:** Package the project for sustained research use after the release decision.
**Key outputs:** runbook, backlog, evidence index, publication/research packet if allowed, future sprint queue.
**Verification:** docs render, links resolve, board has no orphaned active work.
**Risks handled:** R5, R8.

## Review contract for every sprint

Reviewer must return exactly one:

- `PASS:` followed by evidence checked, residual risk, and next gate.
- `REQUEST_CHANGES:` followed by blocking findings and exact files/tests to change.

Minimum review checklist:

1. Inspect changed files and task outputs.
2. Check source paths and line ranges resolve.
3. Spot-check quotes against local source text where applicable.
4. Search for acceptance-promotion flags and claim-surface leaks.
5. Confirm tests/validators listed in the sprint output actually ran.
6. Confirm unsupported channels remain blocked/candidate.
7. Confirm next sprint dependencies are still valid.

## Done definition for entire project

The project is done when one of these terminal states is reached:

### Terminal State A — Accepted scoped certificate
- Clean committed tree.
- Certificate emitted for a bounded scope and bounded observable set.
- Independent review passes.
- README/UI/release language precisely names the scope and certificate.

### Terminal State B — Honest blocked release
- Clean committed tree.
- Dry-run ledger explains why acceptance remains false.
- Product surfaces say engineering probe/source-gated preview.
- Backlog contains exact remaining evidence blockers.

Terminal State B is a success if the evidence is genuinely incomplete. The only failure mode is lying to ourselves.
