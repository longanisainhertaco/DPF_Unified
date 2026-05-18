# Sprint 3 — Pending

Sprint 3 is deferred. Per the Codex audit
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_WP_N1_N4_2026_05_18.md`), Sprint 3 begins
only after Sprint 2 (WP-N1B) is clean: "After WP-N1B is clean, proceed in this
order."

Sprint 3 produces reviewable proposals for every remaining physics blocker:

- WP-N2 — startup BVP, one source-backed channel packet per required channel.
- WP-N3 — reviewed PF-1000 geometry and material masks.
- WP-N5 — physics closure registry and regime gates.
- WP-N6 — neutron mechanism and detector authority.
- WP-N7 — comparator, UQ, and certificate bundle.
- numerical acceptance — convergence, limiter-zero, backend parity, restart
  reproducibility.

When reached, Sprint 3 produces, under this folder, the proposals and matrices
required by the audit:

- `WP_N2_STARTUP_BVP_PROPOSAL.md`, `WP_N2_STARTUP_CHANNEL_MATRIX.csv`
- `WP_N3_PF1000_GEOMETRY_MASK_PROPOSAL.md`, `WP_N3_GEOMETRY_SOURCE_DIMENSION_TABLE.csv`
- `WP_N5_CLOSURE_REGISTRY_PROPOSAL.md`, `WP_N5_CLOSURE_PACKET_MATRIX.csv`
- `WP_N6_NEUTRON_AUTHORITY_PROPOSAL.md`, `WP_N6_NEUTRON_MECHANISM_MATRIX.csv`
- `WP_N7_COMPARATOR_UQ_CERTIFICATE_PROPOSAL.md`
- `NUMERICAL_ACCEPTANCE_PROPOSAL.md`

Every local-source claim will carry `KnowledgeReference/` paths and line ranges;
every external lead will be isolated from authority claims; every inferred
parameter will be logged. No Sprint 3 work is started in this submission.
