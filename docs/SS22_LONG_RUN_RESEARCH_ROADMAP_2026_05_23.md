# SS22 Long-Run Research Roadmap — 2026-05-23

- Task: `t_ac939060`
- Sprint: SS22 research/ops packaging and long-run roadmap
- Release posture: HONEST-BLOCKED / SOURCE-GATED PREVIEW

The long-run target remains a source-backed, reviewable, certificate-gated DPF simulation/research platform. The current project state is a source-gated preview and engineering-probe workbench, not an accepted production first-principles or full-3D simulator.

Fail-closed flags remain:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

Retrieval is not authority. Use local source authority only. No corpus/PDF/symlink normalization is part of this roadmap.

## Phase 1 — Keep the honest blocked release usable

Purpose: preserve the SS21 review-approved claim boundary while making the repository navigable for future work.

Exit requirements:

- [SS22 research/ops runbook](SS22_RESEARCH_OPS_RUNBOOK_2026_05_23.md) is current.
- [SS22 evidence index](SS22_EVIDENCE_INDEX_2026_05_23.md) points to the latest sprint artifacts.
- [SS22 future sprint queue](SS22_FUTURE_SPRINT_QUEUE_2026_05_23.md) names next work without expanding into universal DPF acceptance.
- Claim guardrails remain green.

## Phase 2 — Source and provenance closure

Purpose: convert candidate rows into reviewed same-scope source packets only when local sources support them.

Work bands:

1. Complete PF-1000 same-scope source gaps for waveform, spatial/thermodynamic channels, neutron mechanisms, detector response, and uncertainty.
2. Tighten quote/path/line validators where provenance is weak.
3. Stage cross-scope transfer evidence separately with no promotion path.
4. Require independent review before any row becomes accepted.

Scope explosion guard: each task must name one channel, one source scope, and one acceptance condition. Everything else becomes backlog.

## Phase 3 — Numerical and runtime evidence closure

Purpose: align solver evidence with the scope of any future certificate.

Work bands:

1. Maintain numerical-fidelity tests and backend parity gates for claimed methods only.
2. Preserve limiter/repair telemetry and fail closed on acceptance-blocking limiter activation.
3. Keep resource-heavy full runs behind the Resource contention guard and run them only after lightweight checks pass.
4. Attach run hashes and source hashes to certificate candidates.

## Phase 4 — Certificate and review closure

Purpose: allow the certificate pipeline to refuse incomplete stacks and accept only a fully reviewed same-scope stack if one exists.

Required inputs:

- same-scope evidence packet
- comparator mapping
- uncertainty budget
- numerical-fidelity packet
- negative controls
- source/run hashes
- review certificate

The SS19 synthetic wiring path may remain positive only as `synthetic wiring only`; it is not validation evidence.

## Phase 5 — Publication/research packet

publication packet status: deferred.

Allowed now:

- internal runbook
- evidence index
- honest blocked release memo
- future sprint queue
- research-notes packet that says the project is a source-gated preview

Not allowed now:

- production first-principles acceptance claim
- accepted full-3D claim
- publication-grade validation claim
- generalized predictive DPF-machine claim

## Long-run decision gates

| Gate | Question | Passing evidence | Current posture |
| --- | --- | --- | --- |
| Source gate | Are observables backed by reviewed same-scope local sources? | Accepted source packets with line/quote/UQ/review | Blocked / candidate |
| Numerical gate | Is the solver/backend verified for the claimed scope? | Numerical-fidelity packet and focused tests | Partially scaffolded; claim-specific closure required |
| Runtime gate | Does a run produce hashed artifacts without acceptance blockers? | Clean run/source hashes and readiness report | Engineering-probe only |
| Certificate gate | Does the complete stack pass at one commit? | Accepted certificate plus independent review | Refuses production acceptance |
| Claim gate | Do public surfaces match the certificate? | Claim scan and reviewer approval | HONEST-BLOCKED / SOURCE-GATED PREVIEW |
