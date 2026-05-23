# SS22 Research/Ops Runbook — 2026-05-23

- Task: `t_ac939060`
- Sprint: SS22 research/ops packaging and long-run roadmap
- Release posture: HONEST-BLOCKED / SOURCE-GATED PREVIEW
- Prior decision: [SS21 product claim surface and release decision](SS21_PRODUCT_CLAIM_SURFACE_RELEASE_DECISION_2026_05_23.md)
- Full plan: [DPF unified full project Kanban plan](DPF_UNIFIED_FULL_PROJECT_KANBAN_PLAN_2026_05_22.md)

## Non-negotiable release boundary

Keep the project useful without lying about scientific acceptance:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

Retrieval is not authority. Use local source authority only: repository `KnowledgeReference/`, reviewed local source packets that point back to it, or explicitly reviewed local line-cited extractions. KB/RAG hits are discovery candidates until reviewed and line-cited.

No corpus/PDF/symlink normalization is authorized by this runbook. Do not delete, relink, rewrite, or bulk-normalize `downloaded_books_papers/`, `tmp/pdfs/`, HeliosMatrix KB material, or generated corpus state unless a separate task explicitly authorizes that action.

## Operating loop

1. Inventory the worktree and active resources.
2. Choose a narrow objective tied to one evidence blocker or one engineering packaging blocker.
3. Add or update tests/validators before implementation when code behavior changes.
4. Run the narrow focused verification first; then run the broader guardrail slice if shared claim or evidence surfaces changed.
5. Update `CodexFindings.md`, `CortexFindings.md`, and `docs/MODULE_AUDIT/BACKLOG.md` when status or blockers change.
6. End with Evaluate / Learn / Continue and a reviewer-ready handoff.

## Resource contention guard

Before heavy jobs, run a lightweight inventory before heavy jobs:

```bash
git status --short --branch
ps -axo pid,pcpu,pmem,comm,args | egrep 'Ollama|ollama|python|pytest|uv|hermes|codex|claude|mlx|mps|KB|pdf|tesseract'
```

If Helios/Ollama/GPU/MPS/PDF extraction jobs are active, do not start new indexing, OCR, PDF conversion, embedding, or model-serving tasks. Prefer read-only validators, targeted pytest, and static scans.

## Scope explosion guard

Keep all accepted-claim work scoped to a named device/shot/observable/certificate stack. Same-scope evidence, uncertainty, comparator mapping, run/source hashes, and review certificate are required before any acceptance discussion. Candidate, cross-scope, synthetic, engineering-probe, and not-validation-evidence artifacts remain useful for planning but never promote acceptance.

## Claim drift guard

Public or runtime surfaces must retain honest blocked wording:

- source-gated preview
- engineering probe
- not validation evidence
- not yet an end-to-end predictive DPF simulator

Do not use unqualified wording that says or implies production first-principles acceptance, full-3D acceptance, publication-grade validation, or generalized predictive DPF-machine capability.

## Normal SS22 handoff package

Every sustained-research handoff should point to:

- [SS22 evidence index](SS22_EVIDENCE_INDEX_2026_05_23.md)
- [SS22 future sprint queue](SS22_FUTURE_SPRINT_QUEUE_2026_05_23.md)
- [SS22 long-run roadmap](SS22_LONG_RUN_RESEARCH_ROADMAP_2026_05_23.md)
- [SS22 status memo](SS22_RESEARCH_OPS_PACKAGING_STATUS_2026_05_23.md)

## Publication packet status

publication packet status: deferred. The project can publish internal research/ops packaging and an honest source-gated-preview note, but no publication packet may claim accepted production scientific validation until the certificate stack is complete and independently reviewed.
