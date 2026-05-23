# SS22 Future Sprint Queue — 2026-05-23

- Task: `t_ac939060`
- Sprint: SS22 research/ops packaging and long-run roadmap
- Release posture: HONEST-BLOCKED / SOURCE-GATED PREVIEW

This future sprint queue is a controlled backlog, not permission to expand into universal DPF acceptance. Retrieval is not authority; local source authority only. No corpus/PDF/symlink normalization is authorized by this queue.

Acceptance flags remain:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

## Queue rules

1. Every sprint must state its source scope, validation scope, and blocked reason.
2. Every source/provenance sprint must preserve candidate/blocked status unless independent review accepts the row.
3. Every code behavior sprint uses TDD and focused verification.
4. Every review sprint searches for acceptance-promotion leaks and claim drift.
5. Publication packet status: deferred until a reviewed certificate stack exists.

## future sprint queue

| Candidate sprint | Objective | Primary risk controlled | Verification gate | Acceptance boundary |
| --- | --- | --- | --- | --- |
| SS23 source-packet review hardening | Spot-audit SS14/SS16/SS17/SS18 line ranges, quotes, and source statuses; narrow wide spans | R2 provenance fabrication | Source path/line/quote validator plus reviewer sample log | candidate / blocked rows only unless review explicitly accepts |
| SS24 uncertainty-budget closure | Build/verify UQ rows for comparator-ready channels and block partial numeric rows | R1 false acceptance, R8 scope explosion | UQ validator rejects missing terms and partial blocked rows | no accepted runtime claim |
| SS25 comparator mapping closure | Map same-scope source observables to simulation observables and reject cross-scope shortcuts | R3 cross-scope creep | Comparator tests and source-scope scan | transfer candidates remain non-promoting |
| SS26 detector/diagnostic response closure | Source-close neutron detector/activation response and nTOF/spectrum response matrices | R2/R3 | Diagnostic packet validator and review certificate gate | no neutron diagnostic acceptance without response/UQ/review |
| SS27 numerical-fidelity refresh | Re-run/refine backend parity, restart, limiter-zero, and convergence packet for the claimed scope | R4 numerical debt | Focused numerical suite and packet hash | engineering verification only until source stack closes |
| SS28 clean release-candidate packaging | Reconcile intentional sprint artifacts, leave corpus noise untouched, and package a clean candidate branch/PR | R6 dirty-tree hygiene | `git diff --check`, focused tests, clean staged list, reviewer approval | packaging only; no claim promotion |
| SS29 certificate acceptance rehearsal | Run certificate pipeline against latest candidate stack and prove refusal or scoped acceptance | R1/R9 | Certificate tests, negative controls, review certificate | production acceptance still false unless full reviewed stack closes |
| SS30 claim-surface re-scan | Re-scan README/UI/docs/release notes after any certificate-status change | R10 claim drift | Claim tests and banned-term scan | HONEST-BLOCKED wording remains unless reviewed acceptance changes |

## Standing backlog fragments

- Same-scope evidence: close only channel-specific gaps with local line-cited source authority.
- Review certificate: require independent review for accepted rows, certificate outputs, and public claim changes.
- Resource contention guard: inventory active model/PDF/KB jobs before heavy extraction or training.
- Scope explosion guard: prefer one device/shot/observable per sprint over broad predictive-language work.
- Claim drift guard: run the SS21/SS22 claim tests after docs, README, app, release, or publication changes.
