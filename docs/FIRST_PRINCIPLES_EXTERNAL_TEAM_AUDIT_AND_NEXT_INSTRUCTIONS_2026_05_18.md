# First-Principles External Team Audit And Next Instructions

Date: 2026-05-18

Auditor: Codex

Repo: `/Users/anthonyzamora/dpf-unified`

Audited submission:

- `e7bedea feat(first-principles): SSR implementation pass -- audit, integrity fixes, negative controls`
- `cd64b8e fix(validation): clear pre-push hook blockers (pre-existing test debt)`
- `docs/FIRST_PRINCIPLES_SSR_SUBMISSION_2026_05_18.md`
- `docs/ssr_audit_2026_05_18/*.md`
- submitted `results/*.json` first-principles artifacts

Governing spec:

- `docs/FIRST_PRINCIPLES_SIMULATOR_SSR_AND_IMPLEMENTATION_AUDIT_SPEC_2026_05_18.md`
- `KnowledgeReference/` as the only scientific source of truth

## Verdict

Verdict: **request changes, accept engineering progress**.

The submission is useful and mostly honest. The code now has better fail-closed
startup gating, better conservation wording, density-floor telemetry, source-sign
Auluck `J.E / I` candidate mode, source-truth gates, and a large negative-control
test suite. It does **not** deliver a full first-principles whole-shot simulator.
It also does **not** yet provide an audit-clean evidence package.

Do not claim:

- accepted first-principles simulation;
- full first-principles whole-shot DPF simulation;
- validated PF-1000/Akel predictive result;
- solved startup, power-port, closure, neutron, geometry, or whole-shot authority.

Allowed claim:

- package-native 3-D first-principles engineering candidate with fail-closed
  packet discipline and improved negative controls.

## Commands Run During This Audit

```bash
git status --short
git log --oneline -8
git show --stat --oneline e7bedea
git show --stat --oneline cd64b8e

.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py

.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py -q
.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py -q -rx

.venv312/bin/python -m ruff check src/dpf/first_principles src/dpf/fields tests/test_first_principles_*.py tests/test_hybrid_3d_*.py
.venv312/bin/python -m ruff check scripts/verify_first_principles_source_truth_exhaustion.py src/dpf/cli/main.py src/dpf/fields/hybrid_loop.py src/dpf/fields/hybrid_simulator.py src/dpf/first_principles/deck.py src/dpf/first_principles/experimental_numerics.py src/dpf/first_principles/limiter_readiness.py src/dpf/first_principles/numerical_fidelity.py src/dpf/first_principles/power_port.py src/dpf/first_principles/runner.py src/dpf/first_principles/startup_bvp.py tests/test_cli_first_principles_3d.py tests/test_first_principles_certificate_negative_controls.py tests/test_first_principles_closures.py tests/test_first_principles_geometry.py tests/test_first_principles_long_run_integrity.py tests/test_first_principles_neutron_authority.py tests/test_first_principles_runner.py tests/test_first_principles_startup_bvp.py tests/test_hybrid_3d_simulator.py

.venv312/bin/dpf first-principles-3d --deck-preset pf1000_akel_16kv --steps 2 --output /private/tmp/dpf_audit_cli_smoke_2026_05_18.json
```

## Verification Results

| Gate | Result | Audit interpretation |
| --- | --- | --- |
| Initial worktree | clean before verification scripts | Submission was committed. |
| Source-truth exhaustion | `exhausted: true`, `open_issue_count: 0` | Pass. The scanner writes timestamps; I restored those audit-only timestamp changes. |
| Module source vetting | `strict_passed: true`, 288 modules, 0 active physics unvetted | Pass. |
| Focused tests | `180 passed, 4 xfailed, 4 warnings` | Pass with intentional xfails that remain real blockers. |
| CLI smoke | completed 2-step PF-1000/Akel run; `scientific_status: engineering_candidate_not_validation`; certificate blocked | Pass for package-native engineering candidate. |
| Broad ruff | failed on pre-existing and current style issues | Not audit-clean. |
| Touched-file ruff | failed on `scripts/verify_first_principles_source_truth_exhaustion.py:691` | Request changes. |
| Submitted runtime artifacts | mixed: newer audit artifacts are current; older 100 ns, 1 ns, and 12 us artifacts are stale | Request changes. |

The four xfails are:

- `tests/test_first_principles_geometry.py::test_conductor_mask_packet_emits_mask_hash`
- `tests/test_first_principles_geometry.py::test_conductor_mask_packet_emits_projection_error`
- `tests/test_first_principles_geometry.py::test_reviewed_rod_mask_requires_resolved_rods`
- `tests/test_first_principles_long_run_integrity.py::test_checkpoint_load_into_mismatched_grid_fails_attributably`

## Audit Findings

### A-1 High: Submitted Evidence Artifacts Are Mixed-Generation

The code-level conservation fix is present. `src/dpf/first_principles/runner.py:2202-2240`
now emits `finite_state` and
`energy_conservation_assessed: not_assessed_no_accepted_tolerance`, without the
old conservation `passed` key.

However, the submitted older artifacts still contain the old conservation field
and lack the new Stage-0 power-port packets:

- `results/experimental_limiter_proof_pf1000_auluck_power_port_100ns_2026_05_18.json:119`
- `results/experimental_limiter_proof_pf1000_auluck_power_port_1ns_2026_05_18.json:119`
- `results/experimental_limiter_proof_pf1000_seeded_power_domain_12us_2026_05_18.json:119`

Those files also lack `stage0_packet_scaffolds` and `deck_diff_packet`, while the
newer audit artifacts do include them:

- `results/audit_first_principles_3d_smoke.json:1728`
- `results/audit_first_principles_3d_smoke.json:6925`
- `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json:1936`
- `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json:8465`

Correction required:

- Regenerate or quarantine every first-principles result artifact generated
  before the conservation and Stage-0 packet changes.
- Add `artifact_generation_commit`, exact CLI `argv`, `generated_at_utc`, and
  package version to every new runtime artifact.
- Add an artifact linter that rejects top-level `conservation_telemetry.passed`
  in first-principles artifacts.

### A-2 High: The Power Port Is Still Candidate-Only

The source-sign Auluck mode is a good engineering step. The runtime can compute
`U_DPF = -P_JE / I` in `src/dpf/fields/hybrid_simulator.py:540-565`, and the
low-current `P/I` singularity is disclosed in
`src/dpf/fields/hybrid_simulator.py:568-612`.

The accepted power-port problem is not solved. Current blockers are explicit in
`src/dpf/first_principles/power_port.py`:

- domain is still `interface_surface_or_volume_domain: not_declared`
  at `power_port.py:181`;
- poynting power and electrode work are `None` at `power_port.py:182-187`;
- residual tolerance is `not_attached` at `power_port.py:252-259`;
- domain review remains `blocked_domain_packet_not_available` at
  `power_port.py:528-540`;
- the Stage-0 energy ledger marks wall Poynting and electrode/interface work
  as missing.

The 1 us source-sign artifact is useful, but it is not the requested accepted
12 us whole-shot power-port evidence. Its own packet says the domain is
`not_declared` and the certificate gate is blocked.

Correction required:

- Implement a named runtime integration domain for Auluck `Omega`, excluding the
  source interface specified by the local Auluck source.
- Implement wall/interface Poynting flux accounting and electrode/interface work
  terms.
- Add a reviewed residual definition with sign and time-centering, not a percent
  threshold invented from convenience.
- Run 100 ns, 1 us, and 12 us artifacts from the same current commit and schema.

### A-3 High: Startup BVP Gating Is Fixed, But Startup Physics Is Not Solved

The fail-open gate identified by the other team is fixed:

- `startup_payload_review["channel_acceptance_eligible"]` is merged only when
  eligible at `src/dpf/first_principles/startup_bvp.py:194-200`;
- `can_support` requires `payload_acceptance_eligible` at
  `startup_bvp.py:214-223`.

That is good fail-closed engineering. It does not provide the first-principles
startup model. The current PF-1000/Akel run still reports
`rejected_startup_mode_for_first_principles`. The required startup channels
remain unsolved:

- gas breakdown;
- preionization;
- insulator flashover;
- electrode/insulator boundary emission;
- initial current-density distribution;
- electron and ion temperatures from source-backed startup state;
- ionization/species state;
- initial electric and magnetic fields;
- sheath liftoff and field/PIC handoff interval.

Correction required:

- Produce one startup source packet per channel from `KnowledgeReference/`.
- Implement the channel only if the source packet contains formulas, variables,
  units, validity range, and scope.
- Keep `seeded_layer` rejected for acceptance until a real startup BVP/handoff
  packet exists.

### A-4 High: Geometry Is Still A Coarse Candidate Projection

The rod-resolution disclosure is a useful fix:

- `_conductor_mask_packet` is at `src/dpf/first_principles/runner.py:2531`;
- `cathode_rod_diameter_grid_cells` is emitted at `runner.py:2570-2574`;
- `cathode_rods_resolution_reviewed: False` is emitted at `runner.py:2575`;
- `insulator_material_surface_resolved: False` is emitted at `runner.py:2581`.

The required geometry packet is still incomplete. The tests explicitly xfail
missing mask hash, projection error, and reviewed-mask resolution gate:

- `tests/test_first_principles_geometry.py:32-57`;
- `tests/test_first_principles_geometry.py:125-152`.

Correction required:

- Add deterministic conductor/material/PML/open-boundary mask hashes.
- Add projection-error metrics against source dimensions.
- Reject `reviewed_same_scope_geometry_mask` when rod, hollow-anode, insulator,
  or material-surface resolution is below declared criteria.
- Represent PF-1000 rods, hollow anode, insulator, chamber boundaries, and
  material surfaces as separate masks, not one undifferentiated conductor mask.

### A-5 High: Long-Run And Restart Are Not Yet Engineering-Review Ready

The code now has useful checkpoint/restart and split-continuation tests, but one
critical test remains xfailed:

- `tests/test_first_principles_long_run_integrity.py:293-344` marks the missing
  checkpoint grid/deck shape guard.

The loader writes checkpoint arrays into a newly built session without validating
grid/deck shape first:

- `src/dpf/first_principles/state_checkpoint.py:115-197`.

The submitted 12 us artifact is not suitable as evidence for the fixed code. It
is stale, still contains the old `passed` key, and records a 58% tracked-energy
drop while carrying the older conservation wording:

- `results/experimental_limiter_proof_pf1000_seeded_power_domain_12us_2026_05_18.json:119`.

Correction required:

- Add shape, spacing, species, closure-policy, and circuit-mode checks before
  loading a checkpoint into a session.
- Add segmented 12 us run orchestration that preserves `lagged_field_work`,
  circuit state, previous current state, particle species, electron/ion energy,
  ionization state, kinetic-yield state, and cumulative ledgers.
- Regenerate the 12 us artifact from current code and current schema.
- Do not call a run "whole-shot-ready" unless it reaches 12 us with source-sign
  power-port mode, no hidden repairs, and explicit conservation residual policy.

### A-6 High: Closure Matrix Is Honest But Incomplete

The closure packet is correctly non-promoting:

- required effects are listed at `src/dpf/first_principles/closure_packet.py:37-48`;
- EOS, radiation losses, impurity/electrode ablation, restrike/anomalous
  resistance, and beam-target acceptance are blocked or candidate-only at
  `closure_packet.py:80-182`;
- every effect hard-codes `can_support_first_principles_acceptance: False` at
  `closure_packet.py:270-291`;
- the top-level packet cannot accept at `closure_packet.py:231-267`.

That is not a solved physics closure system. The PlasmaPy/Spitzer warnings seen
in the focused tests also show that the collisional formula lane needs explicit
validity handling for strong-coupling regimes instead of silent use.

Correction required:

- Build source packets for EOS, ionization, electron/ion energy, heat flux,
  radiation, ablation/impurity, anomalous resistance/restrike, collisions, and
  stopping.
- For every active formula, include units, symbol map, validity regime,
  code line, tests, nondominance/sensitivity/UQ, and claim impact.
- Add a bounded-out declaration for omitted electron inertia or implement it
  from a local source packet.
- Turn PlasmaPy coupling warnings into telemetry and closure-regime gates.

### A-7 High: Neutron Authority Is Structurally Blocked

The neutron packet is fail-closed by design:

- blocking neutron channels are reinserted into missing channels at
  `src/dpf/first_principles/neutron_authority.py:199-200`;
- `can_support_total_yield_acceptance` and
  `can_support_first_principles_acceptance` are hard-coded false at
  `neutron_authority.py:258-259`.

The runner passes candidate kinetic-yield telemetry into the packet at
`src/dpf/first_principles/runner.py:1040-1051`, but there is no accepted
mechanism-separated neutron history. This remains a blocker for any neutron
prediction claim.

Correction required:

- Implement mechanism-separated thermonuclear and beam-target histories.
- Add ion distribution, stopping, spectrum, anisotropy, detector-response, and
  uncertainty packets.
- Connect `kinetic_neutron_yield_authority_status` or an equivalent authority
  gate into the runner only after the upstream mechanism/detector/UQ packets
  exist.

### A-8 Medium: Manifest And Artifact Provenance Are Not Sufficient

The manifest is fail-closed and hashes itself:

- `src/dpf/first_principles/manifest.py:39-84`.

But the manifest does not yet carry enough provenance for external engineering
reproduction. It lacks required command `argv`, environment/config hash, source
packet hashes, artifact schema version, and exact source-truth index hash as
first-class fields.

Correction required:

- Add `command_argv`, `git_commit`, `dirty_worktree`, `source_truth_index_sha256`,
  `source_packet_hashes`, `input_deck_sha256`, `artifact_schema_version`, and
  `artifact_generation_commit`.
- Include these fields in every CLI-produced JSON artifact.
- Add tests proving missing provenance blocks certificate acceptance.

### A-9 Medium: Lint Gate Is Not Audit-Clean

The submission report states ruff was clean on touched files. My touched-file
ruff command failed:

- `scripts/verify_first_principles_source_truth_exhaustion.py:691`
  has unused loop variable `tag`.

The broader first-principles/fields/test lint command also fails on pre-existing
import and Python 3.12 style issues. Do not mix the two. The touched-file error
is directly in this submission and must be fixed.

Correction required:

- Rename `tag` to `_tag` or use it.
- Add an audit command list that exactly matches the team's "ruff clean" claim.
- Either fix broad lint or add a checked, explicit pre-existing-lint allowlist.

### A-10 Medium: The WP Audit Docs Are Stale Relative To The Fixes

The detailed WP docs are valuable but several of them still describe conditions
before the fixes were applied. For example, WP-2's report says request changes
for the startup accept gate; the code now has the payload-eligibility guard.
The submission report partly supersedes those docs, but the audit package is not
self-consistent.

Correction required:

- Add a single `docs/ssr_audit_2026_05_18/POST_FIX_RECONCILIATION.md`.
- For every WP finding, mark it `fixed`, `still_open`, `superseded`, or
  `rejected_after_review`.
- Link each fixed finding to code lines and tests.
- Link each still-open finding to the next work package below.

## Next Team Instructions

The outside team must work in the following order. Do not skip earlier hygiene
work to chase physics features; otherwise the next audit cannot trust the
submitted evidence.

### WP-N0 Evidence Hygiene And Submission Reconciliation

Research:

- Inventory every result artifact under `results/` that is referenced by
  `docs/FIRST_PRINCIPLES_SSR_SUBMISSION_2026_05_18.md` or the WP audit docs.
- For each artifact, record generation command, current schema fields,
  originating commit if known, and whether it contains stale fields such as
  top-level `conservation_telemetry.passed`.

Solve:

- Define a first-principles artifact schema with a required `artifact_schema_version`.
- Define a stale-artifact rejection rule.

Apply/follow:

- Do not modify physics to make old artifacts look current.
- Either regenerate artifacts from current code or move stale artifacts into a
  clearly named archival folder with a `stale_do_not_use_for_audit` marker.

Correct/implement:

- Add `scripts/audit_first_principles_artifacts.py`.
- The script must fail if any submitted artifact has:
  - top-level `conservation_telemetry.passed`;
  - no `artifact_generation_commit`;
  - no `command_argv`;
  - no `telemetry_packets.power_port.stage0_packet_scaffolds`;
  - no manifest `candidate_evidence.deck_diff_packet` for PF-1000/Akel runs;
  - `can_support_first_principles_acceptance: true`.

Expected result:

- `python scripts/audit_first_principles_artifacts.py results/*.json` passes
  only for current-schema artifacts.

### WP-N1 Power-Port Closure

Research:

- Re-open these local sources and produce a new source packet:
  - `KnowledgeReference/auluck-2021-dpf-circuit-element.md:151-209`
  - `KnowledgeReference/auluck-2021-dpf-circuit-element.md:235-262`
  - `KnowledgeReference/2019nrlplasma-formulary-037290d4.md:1880-1888`
  - `KnowledgeReference/fully-electromagnetic-hybrid-pic-fluid-dpf-neutron-yield-acb71fa9.md:741-789`
- Identify the exact runtime equivalent of Auluck `Omega`, the excluded source
  interface, and the Poynting surface.

Solve:

- Derive one signed, time-centered terminal power balance.
- Define which cells/faces are in the volume, which faces are the terminal
  interface, and which faces are walls/material boundaries.
- Define how the first-step and low-current `P/I` singularity are handled without
  hidden current floors.

Apply/follow:

- Keep Sigma/quasi-TEM line voltage as exploratory-only unless a local
  `KnowledgeReference/` source explicitly supports it as a DPF circuit driver.
- Keep `L_field = 2 E_B / I^2` diagnostic-only.

Correct/implement:

- Add domain masks and surface labels to runtime geometry.
- Emit per-step and cumulative:
  - `terminal_port_work_J`;
  - `volume_j_dot_e_work_J`;
  - `wall_poynting_flux_excluding_declared_port_J`;
  - `electrode_interface_work_J`;
  - `stored_em_energy_delta_J`;
  - `power_port_residual_J`;
  - `power_port_residual_policy`.
- Add negative tests for sign reversal, domain corruption, time-centering
  downgrade, omitted electrode work, residual-policy failure, and low-current
  fallback.

Expected result:

- 100 ns, 1 us, and 12 us source-sign artifacts generated from the same commit.
- Power-port packet still may be candidate, but it must have complete ledger
  channels and an explicit residual policy.

### WP-N2 Startup BVP And Handoff

Research:

- Search `KnowledgeReference/` one startup channel at a time:
  1. breakdown;
  2. preionization;
  3. flashover;
  4. secondary emission/material surface;
  5. current-density distribution;
  6. electron/ion temperatures;
  7. ionization/species;
  8. electric/magnetic fields;
  9. sheath liftoff and handoff interval.
- For each channel, write source packet fields: formula, variables, units,
  validity, device scope, missing parameters, and implementation decision.

Solve:

- Decide whether the source supports a true startup BVP or only a candidate
  initialized sheath.
- If only initialized sheath support exists, define it as a non-whole-shot
  `imported_sheath_state` mode and keep breakdown blocked.

Apply/follow:

- Never accept `seeded_layer`.
- Never accept a text-only startup channel list.
- Never accept startup without payload review and handoff interval.

Correct/implement:

- Implement startup payload classes with typed channel data.
- Add `startup_handoff_interval_s`, initial `E`, `B`, `J`, density, ionization,
  `Te`, `Ti`, and sheath position/velocity fields.
- Add tests proving payload omission, cross-scope payload, missing handoff, or
  missing surface channel blocks acceptance.

Expected result:

- Default PF-1000/Akel can still be blocked, but every startup blocker must map
  to a specific source packet or source absence.

### WP-N3 PF-1000 Geometry And Materials

Research:

- Re-open PF-1000/Akel geometry sources:
  - `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:108-142`
  - `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md:262-270`
  - `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md:340-356`
- Resolve source-dimension disagreements explicitly instead of silently choosing
  one geometry.

Solve:

- Define source dimensions for rods, hollow anode, insulator, chamber, material
  surfaces, open boundaries, PML, and current-injection surface.
- Define minimum resolution criteria for each object.

Apply/follow:

- Reviewed geometry status requires mask hash, source dimensions, projection
  errors, and resolution evidence.
- Candidate coarse masks may run, but must never claim acceptance.

Correct/implement:

- Add separate masks for cathode rods, anode/hollow anode, insulator, chamber,
  material surfaces, wall, source interface, PML, and active plasma domain.
- Add `mask_sha256`, `projection_error`, and `resolution_review` packets.
- Make `reviewed_same_scope_geometry_mask` raise before runtime if any required
  geometry object is unresolved or under-resolved.

Expected result:

- The three geometry xfails become passing tests.

### WP-N4 Long-Run Numerics, Restart, And 12 us Execution

Research:

- Identify all state required for exact restart and split continuation:
  fields, particles, previous current, circuit, electron energy, ionization,
  kinetic yield, lagged field work, cumulative ledgers, and random/state seeds.
- Identify timestep and limiter constraints for 12 us runs.

Solve:

- Define a segmented-run protocol that is equivalent to a single uninterrupted
  run at the same step sequence.
- Define a runtime budget and artifact thinning policy that does not truncate
  cumulative ledgers.

Apply/follow:

- History capping may cap retained samples only, never cumulative counters.
- Checkpoint mismatch must fail before state is written into a session.

Correct/implement:

- Add checkpoint metadata for grid shape, spacing, deck hash, circuit mode,
  closure policy, particle species, and state-channel hashes.
- Add loader validation before assigning arrays.
- Add a CLI `--segment-steps` or equivalent for segmented 12 us runs.
- Add 12 us source-sign artifact generation with current schema.

Expected result:

- Checkpoint mismatch xfail becomes passing.
- 12 us source-sign run reaches target time with current artifact schema and no
  stale conservation `passed` field.

### WP-N5 Physics Closures

Research:

- For each closure, search `KnowledgeReference/` and write a closure packet:
  EOS, ionization/recombination, conductivity, heat flux, electron-ion coupling,
  radiation, ablation/impurities, anomalous resistance/restrike, collisions,
  stopping, and electron inertia.

Solve:

- Decide whether each closure is implemented, bounded out, candidate, or
  blocked.
- Define validity gates. PlasmaPy warnings and strong-coupling regimes must
  become telemetry, not console-only warnings.

Apply/follow:

- Source formulas must carry units and symbol maps.
- Active closures must have nondominance or sensitivity/UQ tests.
- Empirical/reduced closures may be comparators only unless promoted through
  `KnowledgeReference/`.

Correct/implement:

- Add closure-specific packets and tests.
- Add warning capture for PlasmaPy coupling limits.
- Add source-backed radiation/ablation/impurity/restrike blockers or candidate
  implementations, with explicit energy-ledger coupling.

Expected result:

- Closure packet no longer says only "candidate" globally; it must identify
  each closure's exact accepted/candidate/blocked/bounded-out reason.

### WP-N6 Neutron Mechanism And Detector Authority

Research:

- Gather local source packets for thermonuclear D-D history, beam-target
  history, ion distributions, stopping, spectrum, anisotropy, detector response,
  and uncertainty.

Solve:

- Define separate histories for thermonuclear and beam-target production.
- Define detector response and UQ as required authority gates, not optional
  post-processing.

Apply/follow:

- Scalar total yield can only be comparator/baseline.
- Lee/RADPF neutron formula text can only be baseline context.
- Cross-scope neutron data can only be schema/requirements material unless a
  reviewed transfer rule exists.

Correct/implement:

- Wire mechanism-separated histories into `runner.py`.
- Wire detector/UQ evidence into the neutron authority packet.
- Add negative tests for scalar-only, cross-scope, missing detector, missing UQ,
  missing ion distribution, and reduced-model leak.

Expected result:

- Neutron authority may remain blocked, but it must be blocked because specific
  mechanism/detector/UQ evidence is missing, not because the accept path is
  structurally inert.

### WP-N7 Comparator, UQ, Certificate, And Provenance

Research:

- Identify all same-scope PF-1000/Akel targets available in
  `KnowledgeReference/`.
- Identify which are scalar, waveform, spatial, temperature, neutron, detector,
  or uncertainty targets.

Solve:

- Define the comparator scope and UQ schema for each observable.
- Define certificate requirements as explicit packet dependencies.

Apply/follow:

- Certificate acceptance must require current-schema artifacts, source packet
  hashes, tests, command provenance, and no stale blockers.
- Do not make certificate accept-path tests artificial by bypassing real packet
  dependencies.

Correct/implement:

- Add command/argv, source packet hashes, deck hash, and source-index hash to
  manifest and output JSON.
- Add source-target binding for same-scope waveform and diagnostic targets.
- Add certificate negative and positive-path tests. Positive path may use
  synthetic packet fixtures, but must prove every dependency is checked.

Expected result:

- External engineers can reproduce exactly what code, deck, sources, and command
  produced every artifact.

### WP-N8 Multi-Machine Candidate Decks

Research:

- Review GV, Soto, IPFS, May15, and May16 machine decks and decide which parts
  can be promoted into `KnowledgeReference/` source packets.

Solve:

- Define a machine-deck schema that distinguishes source value, inferred value,
  fitted engineering comparator, and blocked unknown.

Apply/follow:

- Do not use cross-machine fitted parameters as PF-1000/Akel closure authority.
- Do not infer hidden parameters into "source truth" without a reviewed
  inference packet.

Correct/implement:

- Add candidate decks with fail-closed packets.
- Add inverse-parameter packets only as non-promoting engineering inference.
- Add tests proving inferred values cannot become first-principles acceptance
  inputs.

Expected result:

- Multi-machine runs are useful engineering probes, not authority leaks.

## Required Next Submission Format

The team's next submission must include:

1. `docs/FIRST_PRINCIPLES_NEXT_SUBMISSION_<date>.md`
2. Changed-files table with one row per file.
3. Requirement/work-package table with `fixed`, `still_open`, `rejected`, or
   `superseded`.
4. Source packet table with exact `KnowledgeReference/` paths and line ranges.
5. Artifact table with path, SHA256, `artifact_generation_commit`, command argv,
   schema version, and whether it is current or stale.
6. Test/lint command table with exact command and result.
7. Xfail table. Each xfail must have an owner, blocker ID, and removal condition.
8. AI/tool disclosure. Every external AI or web suggestion must be classified as
   rejected, source-ingested, or candidate-only.

## Next Codex Audit Procedure

I will audit the next submission in this order:

1. `git status --short`; reject unexplained dirty worktree.
2. `git diff --stat HEAD~1..HEAD` or the submitted branch diff.
3. Run source-truth exhaustion and module-vetting scripts.
4. Run artifact linter over submitted result files.
5. Run exact test commands from the submission.
6. Run `pytest -q -rx` for first-principles tests and inspect all xfails.
7. Run ruff on touched files and a broad first-principles path. If broad lint is
   not clean, require a pre-existing-debt allowlist.
8. Run at least one fresh CLI smoke artifact to `/private/tmp` and inspect:
   - `scientific_status`;
   - `certificate_gate.status`;
   - top-level `can_support_first_principles_acceptance`;
   - `conservation_telemetry` keys;
   - power-port Stage-0 packets;
   - deck-diff packet;
   - startup packet;
   - geometry packet;
   - closure packet;
   - neutron packet;
   - manifest provenance.
9. Search for overclaims in docs, CLI output, README, server/UI surfaces, and
   runtime artifacts.
10. Review every new formula or physics claim against local `KnowledgeReference/`
    line ranges. Any uncited physics claim fails.

## Immediate Blockers To Close Before Another Full Physics Pass

1. Fix ruff on `scripts/verify_first_principles_source_truth_exhaustion.py:691`.
2. Add artifact stale-schema linter.
3. Regenerate or quarantine stale 100 ns, 1 ns, and 12 us artifacts.
4. Add manifest command/source/hash provenance.
5. Remove or close geometry xfails with mask hash, projection error, and
   resolution gate.
6. Close checkpoint mismatch xfail.
7. Add a segmented current-schema 12 us source-sign run.
8. Reconcile the WP audit docs after fixes.

Only after those are done should the team spend the next major block on new
physics implementation.
