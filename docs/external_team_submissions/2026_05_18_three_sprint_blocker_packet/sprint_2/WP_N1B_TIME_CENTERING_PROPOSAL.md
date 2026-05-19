# Proposal: WP-N1B Power-Port Time-Centering

Status: proposed
Sprint: 2
Blocker IDs: WP-N1B, DPF-PHYS-020, gap G3
Claim allowed: step-consistent trapezoidal time-centering of the power-port
ledger terms.
Claim forbidden: accepted high-order temporal accuracy of the power integral.

## 1. Scope

Temporal discretisation of the Auluck six-term power-port ledger
(`WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`). PF-1000/Akel scoped.

## 2. Local Source Authority

Auluck eq. (1)-(6) and the NRL Poynting theorem are **instantaneous identities** —
they hold at a single time `t` and prescribe no discretisation. A
`KnowledgeReference/` search found no source giving a time-centering or
quadrature order for a J·E or Poynting power integral. General MHD-numerics
references (LeVeque finite-volume; the IMEX structure-preserving MHD scheme)
discuss time-stepping order for hyperbolic conservation laws but **not** the
quadrature of a power integral. Source status: `blocked_no_source` for any
accuracy-order claim.

## 3. Equations And Symbol Map

The ledger time-integrates each instantaneous power term `P_k(t)` to a
cumulative energy `E_k = integral P_k dt`. The only discretisation requirement
that the source content *does* impose is internal consistency: eq. (6) is an
identity at each `t`, so all six terms must be sampled at the **same**
time-centering before quadrature, or the discrete balance is corrupted by a
centering mismatch rather than by physics.

## 4. Validity Regime

Applies to any fixed-step run of the first-principles runtime. No regime limit.

## 5. Proposed Numerical Method

- Declare a single centering for the whole ledger: **step-consistent
  trapezoidal** — every term `P_k` evaluated from the same begin/end snapshots of
  each step, combined with the trapezoidal rule over `[t_n, t_{n+1}]`.
- Emit, per term, which snapshot pair it used, so a centering mismatch is
  detectable.
- Label the ledger `time_centering: candidate_step_consistent_not_accepted`. The
  claim is *consistency* (all terms aligned), explicitly **not** an accuracy
  order — no KR source prescribes one.

## 6. Implementation Plan

In the WP-N1B implementation session, alongside the six-term ledger
(`WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md` section 6): record per-term
centering metadata; assert all six terms share centering; keep the
`not_accepted` accuracy label.

## 7. Test Plan

- Negative test: feed two terms with mismatched centering => the ledger flags a
  centering mismatch and does not claim a closed residual.
- Positive test: all six terms share `step_consistent_trapezoidal` centering =>
  the centering metadata is uniform.
- The existing N5 first-step-fallback test already checks shared centering;
  retain it.

## 8. Runtime Artifacts

Per-term centering metadata in the power-port ledger;
`time_centering: candidate_step_consistent_not_accepted`.

## 9. Acceptance And Rejection Criteria

Accept engineering progress when all six terms are provably on one centering and
mismatch is detected by test. Acceptance of a temporal **accuracy order** stays
blocked until a source prescribing one is ingested into `KnowledgeReference/`, or
a dated human review record fixes it. Reject if the ledger claims an accepted
accuracy order without such a source.

## 10. Open Questions

- Is step-consistent trapezoidal sufficient for the eventual review packet, or
  will a reviewer require a second-order-in-time scheme? Owner: WP-N1B review.
  No KR source currently settles this.

## 11. AI And External Tool Disclosure

Research by Claude Opus 4.7 (Claude Code) agents and lead verification against
`KnowledgeReference/`. No external sources used. No code implemented this sprint.
