# Sprint 3 — Status Ledger

Date: 2026-05-19
Branch: `codex/corpus`

## Delivery State

```
research_packet_delivered=true
runtime_foundation_delivered=true
accepted_physics_delivered=false
validation_delivered=false
```

## Delivered Research Packets

Every packet below is a source-grounded research document. These packets do not
promote validation or mark any channel accepted.

| Packet | Path |
| --- | --- |
| WP-N2 Startup BVP Channel Matrix | `sprint_3/WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md` |
| WP-N3 Geometry Source Packet | `sprint_3/WP_N3_GEOMETRY_SOURCE_PACKET.md` |
| WP-N3 Sigma_p Runtime Interface Spec | `sprint_3/WP_N3_SIGMA_P_RUNTIME_INTERFACE_SPEC.md` |
| WP-N4 Performance and Run Plan | `sprint_3/WP_N4_PERFORMANCE_AND_RUN_PLAN.md` |
| WP-N5 Closure Registry Source Audit | `sprint_3/WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md` |
| WP-N6 Neutron Authority Packet | `sprint_3/WP_N6_NEUTRON_AUTHORITY_PACKET.md` |
| WP-N7 Comparator / UQ / Certificate Spec | `sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_SPEC.md` |

## Delivered Runtime Foundations

The following implementation artifacts satisfy the Sprint 3 runtime-foundation
contract. They are implemented candidates only; all accepted-physics and
validation claims remain blocked until the named evidence packets pass.

| Work Package | Runtime Artifact | Acceptance Blocker |
| --- | --- | --- |
| S3.2 PF-1000/Akel geometry masks | `src/dpf/fields/source_geometry.py` — `PF1000GeometryPacket`, `PF1000MaskManifest`, material masks | Reviewed geometry still blocked by target extraction, conflict resolution, and same-scope 3D evidence. |
| S3.3 Sigma_p surface packet | `src/dpf/fields/source_geometry.py` — `SigmaPSurfacePacket`; consumed by `src/dpf/first_principles/power_port.py` terms II/IV/V/VI | Terms II/IV/V/VI remain blocked until reviewed Sigma_p moving-boundary geometry, sign convention, velocity, and resistivity operands are available. |
| S3.4 Startup BVP runtime packet | `src/dpf/first_principles/startup_bvp.py` — typed `StartupBVPPacket` with 13 WP-N2 channels | No startup channel has accepted same-scope source evidence; authority remains blocked. |
| S3.5 Closure registry and regime gates | `src/dpf/first_principles/closure_packet.py` — all active closures registered, sourced, candidate, or blocked | EOS/radiation/ablation/anomalous/restrike/electron-inertia/stopping and sensitivity/UQ remain blocked or candidate-only. |
| S3.6 Neutron authority packet | `src/dpf/first_principles/neutron_authority.py` — mechanism-separated interface | Beam-target authority, spectrum, anisotropy, detector response, activation, and UQ remain blocked. |
| S3.7 Numerical acceptance harness | `src/dpf/first_principles/segmented_whole_shot.py` and `src/dpf/first_principles/segmented_whole_shot_combine.py` — small-horizon tests, manifest gates, restart ledger merge | 12 us production-horizon evidence, convergence tolerances, backend parity, limiter-zero proof, and restart reproducibility remain blocked. |
| S3.8 Comparator/UQ/certificate scaffold | `src/dpf/first_principles/certificate_gate.py` — blocked dossier with exact missing packets | Same-scope waveform/phase/spatial/neutron/field-coupling/UQ packets remain absent. |
| S3.9 SRS/RTM traceability and audit | `docs/SRS_TRACEABILITY_MATRIX.csv`, `docs/SRS_TRACEABILITY_MATRIX.json`, `CHANGELOG.md`, full periodic audit | Traceability is an implemented control gate, not acceptance evidence. |

## Status Classification for Upstream Packets

| Work Package | research_packet_delivered | runtime_foundation_delivered | accepted_packet_not_delivered |
| --- | --- | --- | --- |
| WP-N2 startup BVP | true | true | true |
| WP-N3 geometry / Sigma_p | true | true | true |
| WP-N4 performance / run plan | true | true | true |
| WP-N5 closure registry | true | true | true |
| WP-N6 neutron authority | true | true | true |
| WP-N7 comparator / UQ / certificate | true | true | true |

## Predecessor File

`sprint_3/PENDING.md` has been superseded by this ledger. The placeholder text
("Sprint 3 is deferred") was correct at the time of Sprint 1 submission;
research packets have since been delivered under this directory and the
placeholder is no longer accurate. This file replaces it.
