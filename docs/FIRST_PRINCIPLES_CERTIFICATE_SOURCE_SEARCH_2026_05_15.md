# First-Principles Blocker Source Search - Certificate And Release Decision

Date: 2026-05-15

Scope: local source of truth and current repo governance artifacts. Scientific
claims remain limited to `KnowledgeReference/`; certificate governance also uses
the first-principles plan, requirements baseline, and local code tests as
repo-source evidence.

Blocker: `FP-14`, validation certificate and release decision.

Question: can a package-native DPF-Unified run write an accepted first-principles
certificate for PF-1000/Akel today?

## Verdict

No. The project has a clear fail-closed certificate contract, but the accepted
first-principles certificate cannot be written because every upstream evidence
gate from FP-5 through FP-13 still has blocked or candidate status.

The current accepted-contract answer is:

- The finish-line plan states that draft, blocked, missing-UQ, missing-review,
  cross-scope, hidden-limiter, and app-only evidence must reject a certificate.
- The local requirements baseline says the first-principles certificate requires
  same-scope waveform, phase, spatial, neutron, detector, field-coupling,
  physics-fidelity, numerical-fidelity, and UQ evidence.
- The newly ingested 2026 hybrid PIC-fluid source is searchable source truth, but
  its ingestion record explicitly does not make any paper claim, benchmark,
  geometry, or neutron yield an accepted validation target without a separate
  reviewed target packet.
- Existing repo certificate code already rejects blocked/cross-scope evidence in
  the broader artifact layer. The package-native first-principles runner now adds
  its own non-promoting `certificate_gate` packet so the 3D runner cannot be
  mistaken for a releasable first-principles claim.

Therefore `FP-14` remains blocked. An accepted certificate requires a manifest
hash, evidence packet hashes, accepted upstream packets, reviewer metadata,
metrics/UQ IDs, requirement links, release label, command provenance, and
negative-test proof.

## Source Answers

| Source | What it answers | What remains blocked |
| --- | --- | --- |
| `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:82-148` | Defines claim levels, evidence states, and that only validated physics evidence can support accepted scientific claims. | Current runner artifacts are engineering candidates and blocked packets, not validated physics evidence. |
| `docs/FIRST_PRINCIPLES_FINISH_LINE_PLAN.md:130-131,646-673` | Certificate must reject draft, cross-scope, blocked, missing-UQ, missing-review, hidden-limiter, and app-only evidence; certificate payload needs hashes, reviewers, metrics/UQ, requirements, commands, release label, and negative tests. | Packet hashes, accepted review metadata, accepted metrics/UQ, and negative-test proof are incomplete for PF-1000/Akel. |
| `docs/DPF_REQUIREMENTS_BASELINE.md:82-84` | First-principles certificate requires same-scope waveform, phase, spatial, neutron, detector, field-coupling, physics-fidelity, numerical-fidelity, and UQ evidence; certificate artifacts write only when linked gates pass. | No accepted first-principles PF-1000/Akel packet exists. |
| `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:20` | User-ingested source is searchable and citable but does not convert paper claims into accepted validation targets without separate typed target extraction and review. | The 2026 source cannot bypass target packet and review requirements. |
| `src/dpf/validation/artifacts.py:398-445` | Existing repo certificate schema rejects accepted certificates with blockers, missing reviewers, blocked/failed evidence, or cross-scope evidence. | This is a reference governance guard, not a complete first-principles PF-1000/Akel certificate. |
| `tests/test_validation_artifacts.py:531-666` | Existing tests reject blocked draft evidence, cross-scope evidence, and accepted certificates with blockers. | Package-native first-principles still needs all upstream packet evidence accepted. |

## Required Certificate Gate

An accepted first-principles certificate needs these channels:

| Channel | Current PF-1000/Akel state |
| --- | --- |
| Run manifest hash | Candidate manifests exist, but accepted certificate manifest binding is blocked. |
| Evidence packet hashes | Blocked until packet contents are accepted and frozen. |
| Validation scope and source scope | Candidate scope can be named; accepted source scope blocked by missing same-scope packets. |
| Package-native execution proof | Partial for `first-principles-3d`; broader routing still incomplete. |
| Same-scope source packet accepted | Blocked. |
| Waveform/phase packet accepted | Blocked by review. |
| Spatial/field/temperature packet accepted | Blocked. |
| Neutron-authority packet accepted | Blocked. |
| Comparator/UQ packet accepted | Blocked. |
| Numerical-fidelity packet accepted | Blocked. |
| Physics-closure packet accepted | Blocked. |
| Limiter-zero or physical-bounds packet | Blocked for full PF-1000/Akel horizon. |
| Power-port packet accepted | Blocked. |
| Startup packet accepted | Blocked. |
| Dimensionality/handoff packet accepted | Blocked. |
| Reduced-model rejection proof | Partial; accepted certificate proof incomplete. |
| Reviewer metadata and accepted review status | Blocked. |
| Comparator metrics and UQ IDs | Blocked. |
| Requirement links | Partial; not complete packet. |
| Commands and versions | Partial; must be bound to certificate. |
| Release label and release decision | Current label must remain engineering candidate. |
| Negative tests for draft/blocked/cross-scope/missing-UQ/missing-review/hidden-limiter/app-only/reduced-model fallback | Partial in broader validation artifacts; first-principles packet-specific proof incomplete. |
| Certificate artifact hash | Blocked until certificate is accepted. |

## Implementation Impact

Immediate implementation requirements:

- Emit a `certificate_gate` packet from every package-native first-principles run
  and manifest.
- Mark status `blocked_first_principles_certificate_not_available` unless every
  upstream packet is accepted and every certificate channel exists.
- Carry upstream packet statuses into the gate so blocked same-scope, waveform,
  spatial, neutron, comparator/UQ, closure, dimensionality, startup, and
  power-port packets visibly prevent release.
- Keep release label
  `engineering_candidate_not_releasable_for_first_principles_claim` until the
  full certificate path passes.
- Add first-principles-specific negative tests before any accepted certificate
  writer is allowed.

Next blocker to search after this one: `FP-15`, generalized DPF-machine path.
