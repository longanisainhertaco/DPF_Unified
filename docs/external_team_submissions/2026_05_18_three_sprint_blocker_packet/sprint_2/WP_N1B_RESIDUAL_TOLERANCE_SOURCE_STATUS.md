# WP-N1B — Residual Tolerance Source Status

Sprint: 2
Blocker IDs: WP-N1B, DPF-PHYS-020, gap G2
Date: 2026-05-19

## Verdict

`blocked_no_source`. No file in `KnowledgeReference/` prescribes a numerical
residual or energy-balance tolerance for a DPF power-port ledger.

## What was searched

`KnowledgeReference/` searched for energy-conservation residual tolerances,
numerical acceptance criteria, and percent-error closure bands for a power-port
or energy-balance ledger. The closest candidate,
`KnowledgeReference/beresnyak_2022_pulsed_power_ideal_mhd.md`, provides a
verification *methodology* — compare an MHD-circuit-coupled solution against an
idealised RLC-ODE solution — and states conservative systems should be simulated
"rather precisely", but gives **no numerical tolerance value** for an
energy-balance ledger. The existing WP-N1 source packet
(`docs/ssr_audit_2026_05_18/WP-N1_power_port_source_packet.md`, gap G2) reached
the identical conclusion: `accepted_residual_tolerance` stays `not_attached`.

Inventing a percentage would violate the project rule that physics numbers come
from sources, not from derivation or training data.

## Consequence for WP-N1B

The Auluck six-term ledger (`WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`) computes
and emits `residual_J` and `residual_fraction` as engineering-debug telemetry.
Power-port acceptance stays blocked: a residual, however small, cannot be an
acceptance criterion without a source-backed tolerance.

## Paths to unblock (any one)

1. A cited tolerance from a peer-reviewed source ingested into
   `KnowledgeReference/` — e.g. an energy-conservation bound from a comparable
   first-principles DPF or pulsed-power MHD verification paper.
2. An experimental error bar from a same-scope PF-1000/Akel power measurement,
   used as the comparison band.
3. An explicit, dated human review record fixing the tolerance and the rationale
   (a review-packet decision), recorded in `KnowledgeReference/` or the SRS.

Until one exists, `accepted_residual_tolerance` remains `not_attached` and the
WP-N1B power port remains non-accepting. This blocker is owned by the WP-N1B
implementation effort and is a required input before any power-port acceptance
claim.
