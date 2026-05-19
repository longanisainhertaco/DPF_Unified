# Proposal: WP-N1B Power-Port Acceptance

Status: proposed
Sprint: 2
Blocker IDs: WP-N1B, DPF-PHYS-020, P-1
Claim allowed: a candidate power-port ledger with a source-verified six-term
Auluck eq. (6) decomposition path.
Claim forbidden: accepted power-port authority; validated power-port closure;
completed first-principles power balance.

## 1. Scope

Device/shot: PF-1000 / Akel only. Phase: rundown through pinch. Solver path: the
package-native first-principles 3-D hybrid runtime. Observable: the power-port
energy ledger and its residual, `src/dpf/first_principles/power_port.py`. This is
a PF-1000/Akel-scoped proposal; no general-DPF claim is made.

## 2. Local Source Authority

| Source path | Lines | Supports | Limits |
| --- | --- | --- | --- |
| `KnowledgeReference/auluck-2021-dpf-circuit-element-EQUATIONS-VERIFIED.md` | whole file | Eqs. (1)-(14): the six-term power balance, the `Sigma_p` integrands, symbol map, sign convention | no residual tolerance; no time-centering prescription |
| `KnowledgeReference/auluck-2021-dpf-circuit-element.md` | 150-260 | Domain `Omega` / `Sigma` / `Sigma_p` prose (legible) | equations OCR-garbled — superseded by the verified extract above |
| `KnowledgeReference/2019nrlplasma-formulary-037290d4.md` | 1869-1888 | Poynting theorem; stored EM energy `W = (1/2) integral (H.B + E.D)` | standard EM identity, not DPF-specific |

The verified extract was transcribed this session directly from the primary PDF
`archive_reference_OLD/references/papers/core-dpf/auluck-2021-dpf-circuit-element.pdf`
(pages 6-11). `docs/ssr_audit_2026_05_18/WP-N1_power_port_source_packet.md` is a
prior local working doc, not KR authority.

## 3. Equations And Symbol Map

Auluck eq. (6) — `I(t)V(t) = I + II + III + IV + V + VI` — and eq. (5) — the
`Sigma_p` integrands — are reproduced verbatim with the full symbol map and sign
convention in the KR verified extract (section 2). The six terms are: stored
magnetic (I), motional magnetic (II), stored electric (III), motional electric
(IV), resistive (V), anomalous/poloidal (VI).

Two source-verified corrections:
- Eq. (1) carries a leading minus: `V_12 = -(1/I) integral_Omega d^3r (J.E)`.
- The audit's "electrode/interface work" term is a category error: Auluck's
  balance has no electrode-contact-work term; the electrode interface is
  excluded from `Omega` and its flux is the input `I(t)V(t)`. See
  `WP_N1B_AULUCK_EQ_5_6_SOURCE_STATUS.md`.

## 4. Validity Regime

Auluck eq. (6) is an exact identity derived from Maxwell + Generalized Ohm's Law;
it has no density/temperature/coupling limit. Runtime validity is bounded by
where the runtime resolves `E, B, J, v, eta`. `Sigma_p` is valid only as the
**moving** part of `Sigma` (stationary boundary faces do not contribute — Auluck
p.8). Term VI is non-zero only when poloidal `B` and axial velocity are both
resolved (Auluck p.10); a purely-azimuthal-`B` run has term VI ~ 0. PlasmaPy
strong-coupling Coulomb-log warnings affect `eta` (term V) and are a WP-N5
closure-regime concern, tracked separately.

## 5. Proposed Numerical Method

Replace the current five-term ledger — whose term 4 `electrode_interface_work_J`
is a non-independent closure residual `terminal - volume - wall - stored` — with
Auluck's six-term eq. (6) decomposition, every term computed **independently**:

- Terms I, III: volume integrals over `Omega` of `d/dt(1/2 mu0^-1 B^2)` and
  `d/dt(1/2 eps0 E^2)`. Hooks `magnetic_energy_J` / `electric_energy_J` exist
  (`maxwell_3d.py` diagnostics).
- Terms II, IV: surface integrals over `Sigma_p` of `dS.v(1/2 mu0^-1 B^2)` and
  `dS.v(1/2 eps0 E^2)`.
- Term V: surface integral over `Sigma_p` of `mu0^-1 dS.(eta J x B)`.
- Term VI: surface integral over `Sigma_p` of `mu0^-1 dS.B(B.v)`.
- Residual: `residual_J = integral I.V dt - (I + II + III + IV + V + VI)` — a
  genuine, non-trivial balance check (it no longer closes by construction).

Required runtime support: a `Sigma_p` moving-boundary face set, and `v, B, J, eta`
plus outward `dS` on those faces. `Sigma_p` separation into moving vs stationary
faces depends on WP-N3 reviewed PF-1000 geometry.

This is a substantial physics implementation. Per the project execution playbook
(research -> review -> implement across sessions) it is **proposed**, not
implemented in Sprint 2. It is the first deliverable of a dedicated WP-N1B
implementation session.

## 6. Implementation Plan

- `src/dpf/first_principles/power_port.py`: replace `_WP_N1_LEDGER_KEYS` and
  `build_wp_n1_auluck_power_port_ledger()` with the six-term eq. (6)
  decomposition; each of II/IV/V/VI an independent `Sigma_p` integral.
- Runtime: expose the `Sigma_p` face set and `v/B/J/eta` on those faces
  (investigate `hybrid_stepper.py`, `maxwell_3d.py`, `source_geometry.py`).
- Correct the blocker label `G1_auluck_eq_5_6_ocr_illegible` (now false) to a
  category-accurate label.
- Verify whether `power_port.py` inherited the dropped eq. (1) minus sign.
- Keep `can_support_first_principles_acceptance = False`.

## 7. Test Plan

- Keep negative tests N1, N2, N4, N5, N6.
- **Rewrite N3**: there is no "electrode work" term; the new N3 omits one of
  terms II/IV/V/VI and asserts the residual no longer closes.
- New positive tests: each of terms I-VI computed independently and finite.
- New negative test: purely-azimuthal `B` with zero axial velocity => term VI
  ~ 0 (Auluck p.10).
- Property test: `integral I.V dt ~ I+II+III+IV+V+VI` within a declared (still
  unsourced, see section 9) residual band.
- Artifact-linter: the six-term ledger keys present; no acceptance promotion.

## 8. Runtime Artifacts

A six-term Auluck ledger replacing the five-term ledger, with `residual_J`,
`residual_fraction`, per-term `Sigma_p`/`Omega` provenance, and all fail-closed
manifest labels retained. `can_support_first_principles_acceptance` stays
`false`.

## 9. Acceptance And Rejection Criteria

Accept engineering progress when: the six terms are each computed independently
from runtime fields (no closure residual); the five retained negative tests pass
and N3 is rewritten and passes; per-term tests pass.

Power-port **acceptance remains blocked** until all of: (a) a source-backed
residual tolerance — `WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`, currently
`blocked_no_source`; (b) a source-backed or review-recorded time-centering —
`WP_N1B_TIME_CENTERING_PROPOSAL.md`; (c) WP-N3 reviewed PF-1000 geometry to
define `Sigma_p`.

Reject if: any term is computed as a closure residual; the ledger claims
acceptance; or the eq. (1) sign is left unverified.

## 10. Open Questions

- `Sigma_p` runtime definition depends on WP-N3 reviewed geometry (cross-WP).
  Owner: WP-N1B implementation session, after WP-N3.
- Residual tolerance: blocked, no KR source (G2).
- Time-centering accuracy order: blocked, no KR source (G3).
- Does `power_port.py` carry the dropped eq. (1) minus sign? Owner: WP-N1B
  implementation — verify before any sign-dependent test.
- Does the runtime resolve poloidal `B` and axial `v` well enough for term VI to
  be meaningful? Owner: WP-N1B implementation — investigate.

## 11. AI And External Tool Disclosure

Research by Claude Opus 4.7 (Claude Code) agents; the Auluck eqs. (1)-(14) were
read directly from the on-disk primary PDF by the lead and transcribed into the
KR verified extract. No external web sources or repositories were used. No code
was implemented this sprint — the implementation is proposed for a dedicated
session. The agent-reported glyph-decode of the OCR-garbled `.json` extract was
**not** trusted as physics; the PDF is the authority.
