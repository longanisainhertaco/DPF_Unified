# First-Principles Experimental DPF Simulator Sprint Plan

Date: 2026-05-18

Repo: `/Users/anthonyzamora/dpf-unified`

Scope: working experimental, source-grounded, true-physics dense plasma focus
simulator. This is an engineering-candidate path for later independent review.
It is not a validation claim.

Inputs:

- Six managed Codex subagent reviews: physics, architecture/runtime,
  requirements/story planning, QA/artifact gates, source-resource readiness, and
  Six Sigma process.
- Local repository state and current docs under `docs/`.
- `KnowledgeReference/` and explicitly user-verified staged sources as the only
  scientific source authority.
- Claude Code was launched for an independent read-only planning review. The
  first attempt failed with `ENOTFOUND`; a sandbox-escalated retry was started
  but had not returned output when this plan was written. This plan therefore
  does not rely on a Claude result.

## Executive Verdict

The project is ready to continue as an experimental engineering-candidate
first-principles simulator. It is not ready to claim a complete accepted
first-principles DPF whole-shot simulator.

The strongest assets are now in place:

- Source-truth exhaustion is clean: 1395 indexed files, 1395 actual files,
  zero open issues.
- Module source-vetting is clean: 288 modules reviewed, zero active physics
  modules needing source vetting, zero unsafe claim surfaces.
- The runtime is package-native and first-principles scoped: 3-D EM/PIC/fluid,
  circuit state, field updates, particle push/deposition, generalized-Ohm
  current, candidate electron-energy and ionization closures, and fail-closed
  telemetry packets.
- The other team has already repaired several earlier integrity defects:
  startup fail-open gate, conservation false-green wording, density-floor
  telemetry, geometry mask hash/projection tests, and checkpoint mismatch
  guard tests.

The controlling blockers are still open:

- Active first-principles result artifacts fail the artifact linter because
  they lack top-level `artifact_generation_commit` and `command_argv`.
- The power port is still candidate-only; it lacks a reviewed named runtime
  domain, Poynting/electrode work terms, time-centering, sign convention, and
  accepted residual policy.
- Startup is still not a solved breakdown/flashover/preionization/liftoff BVP.
- PF-1000 geometry is improved but not complete: insulator, hollow anode,
  backplate/source interface, chamber wall, material surfaces, and same-scope
  resolution review remain.
- Physics closures are incomplete: EOS, radiation, ablation/impurity,
  anomalous resistance, restrike, electron inertia handling, collisions,
  stopping, and beam-target coupling remain candidate, blocked, or bounded out.
- Neutron authority remains structurally blocked until thermonuclear and
  beam-target histories, spectra, anisotropy, detector response, and UQ are
  actually consumed by the runtime.
- Same-scope comparator/UQ/certificate paths are fail-closed by design and
  remain incomplete.

## Definition Of "Working Experimental True Physics"

For this project, "working experimental true physics" means:

1. Every active physics equation used by the runtime has a local source packet
   with equation, variables, units, validity range, scope, and source hash.
2. Every active physics closure emits status: `accepted`, `candidate`,
   `blocked`, or `bounded_out`.
3. The simulator can execute a whole PF-1000/Akel-style shot request through the
   package-native 3-D path without reduced-model authority.
4. Candidate closures may run, but their status must be visible and must block
   any acceptance or certificate claim.
5. Each output artifact records schema version, git commit, dirty-worktree
   state, command argv, deck hash, source-index hash, source-packet hashes, and
   artifact hash.
6. A team of engineers can reproduce the run, inspect all candidate/blocking
   packets, and challenge the physics without finding hidden assumptions,
   stale artifacts, or silent numerical repairs.

This is different from scientific validation. Validation comes later and
requires same-scope experimental data, UQ, comparator gates, and independent
review.

## Wire Diagram

Current experimental flow:

```text
KnowledgeReference source packets
  -> PF-1000/Akel deck lock
  -> candidate startup state
  -> RLC circuit state
  -> magnetic source boundary
  -> 3-D Maxwell field step
  -> PIC particle push
  -> charge/current deposition
  -> generalized Ohm current domain
  -> electron/ion energy and ionization candidate closures
  -> circuit power-port telemetry
  -> conservation, limiter, closure, geometry, neutron, comparator packets
  -> JSON engineering artifact
```

Required whole-shot flow:

```text
reviewed machine source packet
  -> reviewed geometry/material/boundary masks
  -> first-principles startup BVP and handoff packet
  -> time-centered circuit/field power port
  -> restartable segmented 3-D EM/PIC/fluid/kinetic runtime
  -> accepted or explicitly blocked physics closure packets
  -> mechanism-separated neutron and detector packets
  -> same-scope comparator and UQ packets
  -> engineering review bundle
  -> later validation certificate only when all dependencies pass
```

## Sprint Cadence And Controls

Use one-week sprints until the artifact/control layer is clean, then two-week
sprints for physics-heavy work. Every sprint must end with a review packet, not
just a chat summary.

Required sprint artifacts:

- Sprint goal and out-of-scope list.
- Story list with owner, source packets, code paths, tests, acceptance criteria,
  and rollback plan.
- Commands run, exact output summary, and artifact hashes.
- Updated blocker ledger.
- Updated RTM row or Doorstop item for each story.
- Updated source-packet ledger if any physics equation or closure changes.

Review rule:

- No story closes unless code, tests, docs, and artifact gates close together.
- No physics story closes unless the source packet and negative tests close.
- No run artifact is reviewable unless the artifact linter passes.

## Six Sigma DMAIC Plan

### Define

Goal: a reproducible experimental 3-D first-principles DPF whole-shot simulator
that stays honest about candidate physics and is ready for engineering review.

Critical-to-quality outputs:

- Reproducible artifact provenance.
- No stale or mixed-generation evidence.
- No reduced-model authority in first-principles runs.
- No hidden floors, clamps, limiters, or repair paths.
- Source-grounded physics packets for every active equation.
- Explicit residual ledgers for power, field energy, particles, and source
  interfaces.

### Measure

Track these metrics per sprint:

- Artifact linter pass rate: target 100 percent.
- Non-provenant first-principles artifacts in `results/`: target 0.
- Active physics modules lacking source packet: target 0.
- Candidate/blocking physics packets: burn down by named blocker, not by count
  alone.
- Xfail count with owner/removal condition: target 0 unowned.
- Focused first-principles pytest result.
- Touched-file ruff result and broad-lint delta.
- Restart equivalence drift for segmented runs.
- Conservation and power-port residual magnitudes with status labels.

### Analyze

Classify every defect into one of these buckets:

- Provenance defect.
- Stale evidence defect.
- Overclaim or fail-open claim defect.
- Source-scope defect.
- Missing negative test.
- Hidden limiter/floor/repair defect.
- Physics closure absent.
- Numerical method unresolved.
- Documentation/code divergence.

### Improve

Improve in this order:

1. Make the evidence pipeline trustworthy.
2. Make the runtime domain boundaries explicit.
3. Close the power-port ledger.
4. Close startup handoff.
5. Close geometry/material masks.
6. Extend runtime duration with restartable segmentation.
7. Close or explicitly bound physics closures.
8. Add neutron mechanism and detector packets.
9. Add comparator/UQ/certificate positive and negative paths.

### Control

Add these gates to CI or a required local pre-review script:

- `git diff --check`.
- Touched-file ruff.
- Focused first-principles tests with `-rx`.
- Source-truth exhaustion.
- Module source-vetting.
- Artifact linter over active `results/*.json`.
- Xfail/blocker inventory.
- Generated artifact manifest/provenance check.

## Backlog

### Sprint 0: Evidence Hygiene And Provenance

Goal: make the project audit-clean enough that engineers can trust the evidence
package before reviewing physics.

Stories:

- WP-N0.1: Regenerate or archive every stale first-principles artifact.
  Acceptance: `scripts/audit_first_principles_artifacts.py 'results/*.json'`
  reports zero failed first-principles artifacts.
- WP-N0.2: Add top-level artifact provenance to all generated runtime JSON.
  Acceptance: every submitted artifact includes `artifact_schema_version`,
  `artifact_generation_commit`, `command_argv`, deck hash, source-index hash,
  dirty-worktree flag, generated timestamp, and artifact hash.
- WP-N0.3: Add CI or pre-review gate for artifact linting.
  Acceptance: a missing provenance field fails the gate.
- WP-N0.4: Reconcile stale WP docs with current code state.
  Acceptance: no active instruction document cites resolved geometry/checkpoint
  xfails as live blockers.

Exit criteria:

- Artifact linter passes.
- Current focused tests pass.
- Touched-file ruff passes.
- Review packet lists all dirty worktree changes and ownership.

### Sprint 1: Runtime Domain Model And Power Port

Goal: replace the candidate active-load behavior with a source-grounded,
auditable power-port ledger.

Stories:

- WP-N1.1: Implement a named Auluck runtime integration domain.
  Acceptance: the runtime emits domain mask hash, source-interface exclusion,
  control-volume bounds, cell count, and source-packet references.
- WP-N1.2: Implement the five-term power ledger.
  Acceptance: terminal work, volume `J.E`, wall Poynting excluding declared
  port, electrode/interface work, and stored-field delta are emitted with units
  and signs.
- WP-N1.3: Define time-centering and residual status.
  Acceptance: residual is computed from declared terms, but remains
  non-accepting until a source-backed tolerance or review policy exists.
- WP-N1.4: Add negative tests.
  Acceptance: sign reversal, corrupted domain, omitted electrode work,
  time-centering downgrade, low-current `P/I`, and default-mode leakage all
  fail closed.
- WP-N1.5: Produce 100 ns, 1 us, and 12 us source-sign runs from one commit.
  Acceptance: artifacts are linter-clean and comparable against fallback mode.

Exit criteria:

- No active-load placeholder path can masquerade as accepted physics.
- `U_DPF = - integral(J.E)dV / I` is clearly labeled candidate or accepted
  according to packet completeness.

### Sprint 2: Startup BVP And Handoff

Goal: replace seeded startup with a first-principles startup packet or keep it
explicitly blocked with exact missing channels.

Stories:

- WP-N2.1: Build a startup channel matrix.
  Acceptance: breakdown, preionization, flashover, secondary emission, surface
  plasma, initial fields, current-density distribution, Te/Ti, ionization, and
  sheath liftoff each map to a source packet or explicit source absence.
- WP-N2.2: Implement accepted imported-PIC or BVP handoff contract.
  Acceptance: imported state must include E, B, J, density, species,
  ionization, Te, Ti, sheath surface, and handoff interval.
- WP-N2.3: Add startup negative controls.
  Acceptance: text-only startup descriptions, missing payload, seeded layer,
  missing handoff interval, and missing temperature fields block acceptance.

Exit criteria:

- The runtime can start from a reviewed startup packet or truthfully report
  `blocked_startup_bvp_packet_not_available`.

### Sprint 3: Reviewed PF-1000 Geometry And Materials

Goal: produce deterministic reviewed masks for the actual machine geometry.

Stories:

- WP-N3.1: Add separate material masks.
  Acceptance: rods, hollow anode, alumina insulator, backplate/source
  interface, chamber wall, PML/open boundary, and plasma domain each have
  separate hashes and source dimensions.
- WP-N3.2: Resolve source-dimension disagreements.
  Acceptance: Krauz/Akel/PF-1000 dimensional disagreements are surfaced in the
  deck-diff packet with selected value, source, and reason.
- WP-N3.3: Add resolution-review gates.
  Acceptance: reviewed geometry cannot pass when rods, bore, insulator, or
  sheath-relevant surfaces are under-resolved.

Exit criteria:

- Geometry artifacts are deterministic, source-cited, and cannot overclaim a
  coarse projection as reviewed geometry.

### Sprint 4: Segmented Whole-Shot Runtime

Goal: make 12 us whole-shot experiments repeatable, restartable, and auditable.

Stories:

- WP-N4.1: Promote session/checkpoint orchestration into the CLI.
  Acceptance: a whole-shot run can segment, checkpoint, resume, and merge
  telemetry without changing physics state.
- WP-N4.2: Preserve cumulative ledgers across segments.
  Acceptance: circuit state, previous current, lagged field work, particle
  state, electron/ion energy, ionization, kinetic-yield state, limiter
  inventory, and power-port ledgers are continuous.
- WP-N4.3: Add restart equivalence gates.
  Acceptance: split continuation matches uninterrupted run for the same step
  sequence within declared numerical comparison rules.
- WP-N4.4: Add PML removed-energy ledger.
  Acceptance: PML damping either emits removed energy or is explicitly zero and
  inactive in the artifact.

Exit criteria:

- A 12 us source-sign engineering run is reproducible from the same commit and
  can be resumed from checkpoints.

### Sprint 5: Closure Matrix

Goal: convert closure gaps into implemented, candidate, accepted, or bounded-out
source packets.

Stories:

- WP-N5.1: Add closure packet schema.
  Acceptance: every closure includes formula, units, symbol map, validity
  range, source hash, implementation lines, tests, sensitivity/UQ status, and
  claim impact.
- WP-N5.2: Close or bound EOS, radiation, ablation/impurity, anomalous
  resistance, restrike, collisions, stopping, and electron inertia.
  Acceptance: no active runtime closure is silent or uncategorized.
- WP-N5.3: Add PlasmaPy cross-checks where useful.
  Acceptance: PlasmaPy may verify community-standard formulas and constants,
  but local `KnowledgeReference/` remains the authority for claim status.
- WP-N5.4: Reconcile floor/limiter policy.
  Acceptance: every floor/clip/limiter has a named parameter, raw value,
  activation count, source or numerical justification, and claim impact.

Exit criteria:

- Closure gaps are no longer ambiguous. They are either implemented from local
  source packets, candidate-only, blocked by missing source, or bounded out.

### Sprint 6: Neutron Mechanisms And Diagnostics

Goal: replace scalar-yield-only neutron output with mechanism-separated packets.

Stories:

- WP-N6.1: Split thermonuclear and beam-target histories.
  Acceptance: scalar-only total yield cannot pass mechanism separation.
- WP-N6.2: Add ion distribution, stopping, spectrum, anisotropy, and detector
  response packet contracts.
  Acceptance: missing spectrum, anisotropy, detector response, or UQ blocks the
  neutron packet.
- WP-N6.3: Reconcile neutron authority code paths.
  Acceptance: the runner consumes one neutron-authority interface rather than
  divergent inert and diagnostic paths.

Exit criteria:

- Neutron telemetry becomes useful for engineering inspection while remaining
  fail-closed for acceptance until all mechanism and detector packets exist.

### Sprint 7: Comparator, UQ, Certificate, And Engineering Review Bundle

Goal: make the engineering package reviewable and keep validation blocked until
same-scope evidence exists.

Stories:

- WP-N7.1: Add source-packet hashes and environment/config hashes to manifest.
  Acceptance: missing provenance blocks certificate tests.
- WP-N7.2: Add positive-path certificate fixture using synthetic accepted
  packets.
  Acceptance: certificate dependency logic is tested without claiming real
  PF-1000 validation.
- WP-N7.3: Bind same-scope comparator targets.
  Acceptance: waveform, phase, spatial fields, density, temperature, neutron,
  detector, and UQ targets require packet hashes and scope match.
- WP-N7.4: Produce external engineering review bundle.
  Acceptance: bundle contains commit, dirty status, deck, source index,
  artifacts, tests, linter output, blocker ledger, and reproduction commands.

Exit criteria:

- Engineering firm can run and audit the experimental simulator.
- Scientific validation remains blocked until evidence and independent review
  pass.

## Immediate Next Actions

1. Finish Sprint 0 before further physics claims.
2. Wire top-level artifact provenance through CLI, runner, manifest, and result
   writers.
3. Regenerate the three active audit artifacts that currently fail C2/C3.
4. Add artifact linter to CI or a required pre-review script.
5. Start Sprint 1 with the named Auluck domain and five-term power ledger.

## Audit Method For Future Team Work

Every external team submission will be audited in this order:

1. Confirm worktree state and commit identity.
2. Run source-truth exhaustion and module source-vetting.
3. Run artifact linter over active first-principles artifacts.
4. Run touched-file ruff and focused first-principles pytest with `-rx`.
5. Inspect changed physics modules for source-packet compliance.
6. Inspect changed result artifacts for top-level provenance.
7. Confirm every candidate/blocking physics packet stays fail-closed.
8. Confirm reduced models remain baselines/comparators only.
9. Reconcile docs against current code so stale blockers are not repeated and
   resolved blockers are not promoted beyond evidence.
10. Return a verdict: `accept_engineering_progress`, `request_changes`, or
    `reject_overclaim`.

The audit will reject any submission that has a finite run but lacks provenance,
uses stale artifacts, hides numerical repairs, promotes candidate packets,
or cites a physics equation without a local source packet.

