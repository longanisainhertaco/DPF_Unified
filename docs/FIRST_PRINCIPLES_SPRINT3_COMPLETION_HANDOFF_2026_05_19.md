# First-Principles DPF Sprint 3 Completion Handoff

Date: 2026-05-19

Repository: `/Users/anthonyzamora/dpf-unified`

Branch at handoff: `codex/corpus`

Prepared by: Codex audit lead

Authoritative audit packet:
`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_PACKET_2026_05_19.md`

Latest audit evidence before this handoff:
`/private/tmp/dpf-unified-audit-logs/20260519T183831Z/summary.md`

## Handoff Verdict

The packet is ready to hand over only with this completion handoff attached.

The audit packet alone is enough to tell the team what was wrong and what the
next implementation slice is. It is not strict enough by itself for an
autonomous "continue until all of Sprint 3 is complete" assignment. This file is
the controlling sprint contract.

Sprint 3 completion does not mean a validated full DPF shot. It means the
project has converted the Sprint 3 research/spec packets into source-tagged,
fail-closed runtime foundations that unblock later full-shot work:

- packet hygiene is clean and machine-tested;
- PF-1000/Akel geometry and material masks are source-tagged in runtime code;
- `Sigma_p` surface inventory exists or fails closed with exact blockers;
- startup BVP, closure registry, neutron authority, comparator/UQ/certificate,
  and numerical-acceptance paths have implementation-ready runtime packets or
  explicit blocked packets with tests;
- every claim is traceable to `KnowledgeReference/`, tracked verified extracts,
  code, tests, and requirements.

## Non-Negotiable Source Rules

1. `KnowledgeReference/` and tracked verified extracts under
   `docs/external_team_submissions/.../sprint_2/` are the only scientific
   authorities.
2. External tools, PlasmaPy, online papers, pretrained models, and AI answers
   may be used only as engineering aids or cross-checks. They cannot be used as
   source authority.
3. Every implemented physics equation, coefficient, numeric geometry field,
   material property, source-supported threshold, or source-backed default must
   cite exact local paths and line ranges.
4. Shorthand citations such as `[KR: same file ...]` or
   `[KR: ...formulary...]` are not acceptable in actionable content.
5. No inferred value may become a default. If a value is inferred, it must be
   labeled `inferred_candidate`, isolated from acceptance, and paired with a
   fail-closed blocker.
6. Same-scope PF-1000/Akel shot-12581 16 kV evidence must not be mixed with
   full-energy PF-1000 or other-device evidence as if the scopes are identical.
7. Reduced Lee, snowplow, or scalar-yield-only models remain baselines or
   comparators. They cannot become first-principles authority.
8. Validation remains blocked unless same-scope evidence and the certificate
   gates pass. Sprint 3 is not a validation sprint.

## Sprint 3 Completion Definition

Sprint 3 is complete only when all rows in this table are either implemented
with passing tests or explicitly blocked with source-backed, machine-tested
blocker packets.

| Work package | Requirement IDs | Sprint 3 completion state |
| --- | --- | --- |
| S3.1 packet hygiene and status control | DPF-REL-002, DPF-VV-003 | implemented; packet tests reject stale status and shorthand citations |
| S3.2 PF-1000/Akel geometry packet and material masks | DPF-PHYS-022, DPF-PHYS-014 | implemented or fail-closed per missing source dimensions; mask hashes emitted |
| S3.3 `Sigma_p` surface packet plumbing | DPF-PHYS-020, DPF-PHYS-022 | implemented or fail-closed; terms II/IV/V/VI cannot be closure-derived |
| S3.4 startup BVP packet | DPF-PHYS-010, DPF-PHYS-017, DPF-PHYS-021 | runtime packet exists; unsupported channels block startup authority |
| S3.5 closure registry and regime gates | DPF-PHYS-018, DPF-PHYS-024 | every active/bounded-out closure is registered, sourced, candidate, or blocked |
| S3.6 neutron authority packet | DPF-PHYS-013, DPF-PHYS-025, DPF-VV-010 | mechanism-separated interface exists; scalar yield cannot pass authority |
| S3.7 numerical acceptance harness | DPF-PHYS-023, DPF-DATA-007 | small-horizon tests and manifest gates pass; production 12 us may remain blocked |
| S3.8 comparator/UQ/certificate scaffold | DPF-PHYS-026, DPF-VV-017, DPF-DATA-004 | certificate path reports exact missing packets and rejects validation claims |
| S3.9 traceability and audit evidence | DPF-REL-002 | SRS/RTM, packet changelog, tests, and periodic audit all pass |

If any package remains open, Sprint 3 is not complete. If a package cannot be
implemented from local sources, completion requires a typed fail-closed blocker
with exact missing fields, tests, and requirement links.

## Required Working Sequence

The team must work in this order unless a later package is documentation-only
and does not touch the same files.

1. S3.1 packet hygiene.
2. S3.2 PF-1000/Akel geometry and material masks.
3. S3.3 `Sigma_p` surface packet plumbing.
4. S3.4 startup BVP packet.
5. S3.5 closure registry and regime gates.
6. S3.6 neutron authority packet.
7. S3.7 numerical acceptance harness.
8. S3.8 comparator/UQ/certificate scaffold.
9. S3.9 traceability, changelog, and full periodic audit.

Do not compute Auluck `Sigma_p` terms II/IV/V/VI before S3.2 and S3.3 are in
place. Do not claim startup authority before S3.4 is in place. Do not claim
yield authority before S3.6 is in place. Do not claim validation at all during
Sprint 3.

## S3.1 Packet Hygiene And Status Control

Goal: make the external team packet self-consistent and machine-auditable.

Allowed files:

- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/sprint_3/*.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/README.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/THREE_SPRINT_FINAL_SUMMARY.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CHANGELOG.md`
- `tests/test_external_team_submission_package.py`

Required edits:

1. Replace `sprint_3/PENDING.md` with `SPRINT_3_STATUS_LEDGER.md`, or rewrite
   it in place as a status ledger. It must state:
   - `research_packets_delivered=true`;
   - `runtime_implementation_delivered=false` until code lands;
   - `first_principles_acceptance=false`;
   - every delivered packet path;
   - every not-yet-delivered runtime artifact.
2. Update stale WP-N4 and WP-N7 language:
   - no `Sprint 2.2 open`;
   - no unqualified `WP-N2 not delivered` after the research packet exists;
   - no unqualified `WP-N5 closure registry not delivered` after the research
     packet exists;
   - distinguish `research_packet_delivered`,
     `runtime_packet_not_delivered`, and `accepted_packet_not_delivered`.
3. Normalize citations in WP-N2 and WP-N5:
   - exact local path;
   - exact line range;
   - equation/table/figure identifier when available.
4. Replace ambiguous words:
   - use `candidate_comparator_only`, not `accepted as comparator only`;
   - use `source_backed_candidate`, not `validated`;
   - use `blocked_by_missing_local_source`, not `unknown` when the missing
     source has been searched.
5. Add tests that reject the stale and shorthand states.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py -q -rx
```

Done criteria:

- no packet-status contradiction remains;
- no actionable shorthand citations remain;
- packet changelog accounts for all non-HEAD commits in its covered range;
- test failure messages identify exact paths and lines.

## S3.2 PF-1000/Akel Geometry And Material Masks

Goal: replace candidate projection-only geometry with a source-tagged runtime
geometry packet and deterministic material partition.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `tests/test_source_geometry_packet.py`
- `tests/test_first_principles_geometry.py`
- `docs/DPF_REQUIREMENTS_BASELINE.md`
- `docs/SRS_TRACEABILITY_MATRIX.csv`
- `docs/SRS_TRACEABILITY_MATRIX.json`
- Sprint 3 packet docs for traceability updates only.

Required runtime structures:

- `PF1000GeometryPacket`
- `PF1000GeometryField`
- `PF1000GeometryConflict`
- `PF1000MaskManifest`

Each geometry field must include:

- field name;
- numeric value or `None`;
- units;
- status: `source_supported`, `candidate`, `conflict`, or `blocked`;
- exact source reference if value is present;
- blocker ID if value is missing;
- conflict group if sources disagree;
- scope tag such as `pf1000_akel_16kv_1p2torr_shot_12581` or
  `pf1000_full_energy_revision`.

Required constructors:

- `PF1000GeometryPacket.krauz_2012()`
- `PF1000GeometryPacket.akel_shot_12581()`
- `PF1000GeometryPacket.scholz_gribkov_revision()` if and only if the source
  packet can keep revision conflicts explicit.

Forbidden:

- averaging conflicting dimensions;
- using other-machine hollow-anode dimensions as PF-1000 bore dimensions;
- using probe positions as bore radius;
- treating PML/open boundary as a sourced machine dimension;
- silently falling back to the old single `wall_material_faces` class when
  source-tagged material masks are requested.

Required masks:

- `omega_volume_cells`;
- `terminal_source_interface_faces`;
- `wall_material_faces`;
- `open_pml_faces`;
- `anode_material_faces`;
- `cathode_rod_faces`;
- `insulator_material_faces`;
- `chamber_wall_faces`;
- `backplate_source_interface_faces`;
- `pml_or_open_boundary_faces`.

Required manifest fields:

- geometry packet ID;
- geometry source tag;
- source refs;
- conflict groups;
- blocked fields;
- grid shape and spacing;
- per-mask SHA-256 hashes;
- mask cell counts;
- under-resolution flags;
- `can_support_first_principles_acceptance=false`.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_source_geometry_packet.py tests/test_first_principles_geometry.py -q -rx
```

Minimum tests:

- constructors preserve source conflicts without averaging;
- missing bore/insulator/backplate fields remain blocked;
- all source references exist locally;
- material masks are mutually disjoint where required;
- the Auluck top-level partition remains exhaustive;
- `Omega` and source interface are disjoint;
- mask hashes are stable;
- under-resolved rods or insulator surfaces fail closed;
- missing mask hash fails manifest validation.

Done criteria:

- requirement `DPF-PHYS-022` can move from projection-only partial to
  source-tagged runtime partial;
- all unsupported PF-1000 dimensions are explicit blockers;
- no validation or acceptance claim is made.

## S3.3 `Sigma_p` Surface Packet Plumbing

Goal: create the data contract needed for Auluck eq. (6) moving-boundary surface
terms without fabricating missing operands.

Allowed files:

- `src/dpf/fields/source_geometry.py`
- `src/dpf/first_principles/power_port.py`
- `tests/test_first_principles_power_port.py`
- `tests/test_source_geometry_packet.py`
- SRS/RTM files and Sprint 3 docs for traceability only.

Required runtime structure:

- `SigmaPSurfacePacket`

Required fields:

- source geometry packet ID;
- source geometry hash;
- face IDs;
- oriented area vectors `dS`;
- face areas;
- outward-normal convention;
- face material class;
- moving/stationary classification;
- `Omega` side;
- excluded-interface side;
- field-sampler status for `B`, `E`, `J`;
- material velocity status for `v`;
- resistivity status for `eta`;
- centering and quadrature metadata;
- blockers for every absent operand.

Required power-port behavior:

- terms I and III remain independently computed from stored magnetic/electric
  energy deltas;
- terms II, IV, V, VI consume `SigmaPSurfacePacket` only;
- no term may be computed as `I*V` minus other terms;
- residual is `None` until all six terms are independently computed;
- stationary faces contribute exactly zero to motional terms;
- missing `Sigma_p` blocks II, IV, V, VI;
- missing `v` blocks II, IV, VI;
- missing `eta` blocks V;
- missing sign convention blocks the residual.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_power_port.py tests/test_source_geometry_packet.py -q -rx
```

Done criteria:

- `power_port.py` accepts the packet shape and fails closed on missing operands;
- `DPF-PHYS-020` remains partial, not accepted;
- Sprint 4 can implement the actual surface integrals using this packet.

## S3.4 Startup BVP Packet

Goal: replace vague seeded-startup assumptions with a typed startup packet that
either computes source-supported channels or blocks the startup authority.

Allowed files:

- `src/dpf/first_principles/startup_breakdown.py`
- `src/dpf/first_principles/startup_bvp.py` if present, or a new
  `src/dpf/first_principles/startup_packet.py`
- `tests/test_first_principles_startup_bvp.py`
- `tests/test_cli_first_principles_3d.py`
- SRS/RTM and Sprint 3 docs for traceability.

Required channels:

- gas and fill conditions;
- breakdown/Paschen or source-supported alternative;
- preionization;
- flashover;
- secondary emission;
- photoemission if sourced, otherwise blocked;
- surface plasma;
- initial `E`, `B`, `J`;
- species and charge state;
- ionization/recombination status;
- `T_e` and `T_i`;
- sheath surface/liftoff;
- handoff interval into the 3D solver.

Required packet fields:

- channel ID;
- status: `computed`, `candidate`, `blocked`;
- source refs;
- units;
- symbol map;
- input dependencies;
- output fields;
- blocker reason;
- effect on first-principles claim.

Forbidden:

- arbitrary seed density as accepted startup;
- using published end-state results to back-solve an initial condition and then
  calling it source-backed;
- silent fallback to engineering defaults in first-principles mode.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_startup_bvp.py tests/test_cli_first_principles_3d.py -q -rx
```

Done criteria:

- `DPF-PHYS-021` has a runtime packet path;
- missing channels block startup authority with exact IDs;
- first-principles CLI output reports startup packet status.

## S3.5 Closure Registry And Regime Gates

Goal: every active or bounded-out physical closure has an explicit source,
candidate, or blocked packet and no silent runtime closure remains.

Allowed files:

- `src/dpf/first_principles/closure_packet.py`
- `src/dpf/first_principles/plasmapy_audit.py`
- closure implementation files only when adding telemetry or fail-closed gates;
- `tests/test_first_principles_closures.py`
- SRS/RTM and Sprint 3 docs for traceability.

Required closure records:

- EOS/thermodynamics;
- ionization/charge state;
- single-fluid/two-temperature energy;
- electrical/thermal transport;
- radiation losses;
- impurity/electrode ablation;
- Hall/FLR/kinetic scope;
- 3D instabilities;
- restrike/anomalous resistance;
- beam-target coupling;
- stopping/collisions.

Each record must include:

- classification;
- implemented flag;
- exact source equations or explicit source absence;
- symbol map;
- units;
- validity regime;
- implementation reference;
- verification tests;
- sensitivity/UQ status;
- claim impact;
- review status.

PlasmaPy rule:

- PlasmaPy may be a cross-check only.
- A missing PlasmaPy audit cannot promote or reject a local-source closure.
- A PlasmaPy disagreement outside tolerance must set review-required telemetry.
- Acceptance remains false until local-source review resolves the discrepancy.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_closures.py -q -rx
```

Done criteria:

- no active closure is uncategorized;
- candidate closures can run engineering cases but cannot support acceptance;
- every missing closure blocks the relevant claim explicitly.

## S3.6 Neutron Authority Packet

Goal: make neutron output mechanism-separated and fail-closed for authority.

Allowed files:

- `src/dpf/first_principles/neutron_authority.py`
- `src/dpf/diagnostics/neutron_yield.py`
- `src/dpf/diagnostics/beam_target.py`
- `src/dpf/diagnostics/neutron_tof.py`
- `tests/test_first_principles_neutron_authority.py`
- SRS/RTM and Sprint 3 docs for traceability.

Required channels:

- thermonuclear history;
- beam-target history;
- ion energy distribution;
- stopping/transport;
- neutron spectrum;
- anisotropy;
- detector response;
- activation response;
- scatter/background handling;
- UQ.

Forbidden:

- scalar total yield as mechanism authority;
- beam-target authority without ion distribution and stopping;
- detector or activation authority without response/scatter packet;
- mechanism separation by naming only.

Required tests:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_neutron_authority.py -q -rx
```

Done criteria:

- scalar yield can be comparator-only;
- mechanism authority remains blocked until all mechanism and detector packets
  exist;
- `DPF-PHYS-025` has a runtime packet path and exact blockers.

## S3.7 Numerical Acceptance Harness

Goal: keep whole-shot runtime claims honest while making the available small
horizon checks reproducible.

Allowed files:

- `src/dpf/first_principles/segmented_whole_shot.py`
- `src/dpf/first_principles/segmented_whole_shot_combine.py`
- `src/dpf/first_principles/manifest.py`
- `tests/test_first_principles_segmented_whole_shot.py`
- `tests/test_first_principles_segmented_whole_shot_combine.py`
- `tests/test_first_principles_long_run_integrity.py`
- SRS/RTM and Sprint 3 docs for traceability.

Required gates:

- restart reproducibility at small horizon;
- N-restart ledger merge;
- gap/overlap rejection;
- manifest source-packet hashes;
- cumulative ledgers for circuit, field, particle, energy, ionization,
  kinetic-yield, limiter, power-port, and PML removed energy;
- explicit wall-clock blocker for 12 us if not completed;
- no `horizon_complete=true` unless the run actually reaches the requested
  horizon.

Required tests:

```bash
.venv312/bin/python -m pytest \
  tests/test_first_principles_segmented_whole_shot.py \
  tests/test_first_principles_segmented_whole_shot_combine.py \
  tests/test_first_principles_long_run_integrity.py \
  -q -rx
```

Done criteria:

- 12 us may remain blocked by compute wall and production-grid size;
- the blocker is reported by manifest, not hidden;
- small-horizon restart and ledger mechanics pass.

## S3.8 Comparator, UQ, And Certificate Scaffold

Goal: make the engineering review bundle fail closed with exact missing packets.

Allowed files:

- `src/dpf/first_principles/certificate.py` or existing certificate module;
- `src/dpf/first_principles/source_targets.py`;
- `src/dpf/first_principles/manifest.py`;
- `tests/test_first_principles_certificate_negative_controls.py`;
- `tests/test_first_principles_source_targets.py`;
- SRS/RTM and Sprint 3 docs for traceability.

Required certificate channels:

- run manifest hash;
- source packet hashes;
- same-scope source packet status;
- waveform/phase packet status;
- spatial field/temperature packet status;
- power-port packet status;
- startup packet status;
- closure packet status;
- neutron authority packet status;
- numerical fidelity packet status;
- comparator/UQ packet status;
- reviewer metadata;
- commands and versions.

Required behavior:

- blocked channels block certificate acceptance;
- cross-scope evidence blocks acceptance;
- synthetic positive fixture can prove certificate wiring only;
- no real PF-1000/Akel validation certificate is emitted in Sprint 3;
- all comparator-only channels are labeled `candidate_comparator_only`.

Required tests:

```bash
.venv312/bin/python -m pytest \
  tests/test_first_principles_certificate_negative_controls.py \
  tests/test_first_principles_source_targets.py \
  -q -rx
```

Done criteria:

- certificate scaffold reports every missing packet;
- external engineering review bundle can be generated as a blocked dossier;
- validation remains blocked.

## S3.9 Traceability, Changelog, And Audit Evidence

Goal: leave the repo in a state that Codex can audit without reconstructing the
team's intent.

Required files to update:

- `docs/DPF_REQUIREMENTS_BASELINE.md`
- `docs/SRS_TRACEABILITY_MATRIX.csv`
- `docs/SRS_TRACEABILITY_MATRIX.json`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CHANGELOG.md`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/TEST_MAP.csv`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/CLAIMS_LEDGER.csv`
- `docs/external_team_submissions/2026_05_18_three_sprint_blocker_packet/BLOCKER_MATRIX.csv`

Required commands:

```bash
.venv312/bin/python -m pytest \
  tests/test_external_team_submission_package.py \
  tests/test_source_geometry_packet.py \
  tests/test_first_principles_geometry.py \
  tests/test_first_principles_power_port.py \
  tests/test_first_principles_startup_bvp.py \
  tests/test_first_principles_closures.py \
  tests/test_first_principles_neutron_authority.py \
  tests/test_first_principles_certificate_negative_controls.py \
  tests/test_first_principles_source_targets.py \
  tests/test_srs_traceability_export.py \
  -q -rx

.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

Done criteria:

- every changed behavior is mapped to a requirement ID;
- every requirement status is still honest;
- every non-HEAD commit in the packet coverage window is listed in the packet
  changelog;
- periodic audit passes;
- `git status --short --branch` is clean.

## Prohibited Completion Claims

The team must not say any of the following unless the exact gate listed is
actually satisfied:

| Prohibited claim | Required gate before claim is allowed |
| --- | --- |
| "Sprint 3 validates PF-1000/Akel" | same-scope certificate accepted by WP-N7, not expected in Sprint 3 |
| "The simulator runs a validated 12 us shot" | manifest `horizon_complete=true` for 12 us plus numerical and same-scope gates |
| "Power-port acceptance is closed" | all six Auluck terms independently computed with reviewed `Sigma_p` and residual gate |
| "Startup is first-principles accepted" | all startup BVP channels computed or source-bounded with review |
| "Closure registry is accepted" | every active/bounded-out closure has source equation, validity, tests, UQ, and review |
| "Neutron yield is authoritative" | thermonuclear and beam-target mechanisms separated with spectrum/anisotropy/detector/UQ |
| "PF-1000 geometry is accepted" | source-tagged masks, conflicts, missing fields, and under-resolution gate reviewed |

## Submission Format For The Other Team

The team must deliver one final Sprint 3 submission with:

1. concise summary of changed code and docs;
2. exact list of files changed;
3. requirement IDs changed and new statuses;
4. tests run with full commands;
5. periodic audit path;
6. blocker ledger with remaining blockers and why they could not be closed;
7. explicit statement that validation and full-shot acceptance remain blocked
   unless they have actually passed the required gates;
8. no hidden work outside the allowed file scopes unless justified in the
   submission summary.

## Exact Prompt To Hand To The Team

Use this prompt verbatim:

> Continue DPF-Unified Sprint 3 to completion using
> `docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md` as the
> controlling contract and
> `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_PACKET_2026_05_19.md` as the audit
> basis. Treat `KnowledgeReference/` and tracked verified extracts as the only
> scientific authority. Work through S3.1 through S3.9 in order. Implement code
> only when the local source packet supports it; otherwise create typed
> fail-closed blocker packets with tests. Do not claim validation, accepted
> first-principles authority, or 12 us full-shot completion unless the listed
> gates actually pass. Sprint 3 is complete only when packet hygiene, PF-1000
> geometry masks, `Sigma_p` packet plumbing, startup BVP packet, closure
> registry, neutron authority packet, numerical harness, comparator/UQ/certificate
> scaffold, SRS/RTM traceability, packet changelog, focused tests, and the full
> periodic audit all pass or fail closed with exact source-backed blockers.

