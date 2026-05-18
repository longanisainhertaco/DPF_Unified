# Codex Audit: WP-N1 / WP-N4 Engineering Submission

Date: 2026-05-18

Auditor: Codex, with six delegated audit lanes:
provenance/CI, runtime architecture, physics/source authority, QA, SRS/RTM,
and external-team handoff.

Audited repository state: `76480b0` on `codex/corpus`.

Audited submission:
`docs/FIRST_PRINCIPLES_WP_N1_N4_SUBMISSION_2026_05_18.md`

## Verdict

`request_changes_accept_engineering_progress`

The engineering team closed real engineering gaps. The package now has a
candidate WP-N1 Auluck-domain power-port ledger, a candidate WP-N4 segmented
whole-shot runner, passing focused tests, cleaner source-truth hygiene, and
fail-closed artifact controls.

This is not an accepted first-principles DPF simulator. It is not ready to claim
accepted power-port authority, whole-shot startup, reviewed PF-1000 geometry,
accepted closures, neutron mechanism authority, or a completed 12 us
source-sign whole shot.

Allowed claim:

`package-native 3-D first-principles engineering candidate with fail-closed source-truth, artifact, power-port, and segmented-run controls`

Forbidden claims:

- Accepted first-principles simulation.
- Validated PF-1000/Akel prediction.
- Accepted power-port authority.
- Solved breakdown/startup/sheath-liftoff BVP.
- Reviewed PF-1000 rods, hollow anode, insulator, and material masks.
- Accepted EOS/radiation/ablation/restrike/anomalous-resistance/2T/neutron closures.
- Completed 12 us source-sign whole-shot run.

## What Passed

The following gates passed locally under `.venv312` / Python 3.12:

| Gate | Result |
| --- | --- |
| Focused WP-N1/WP-N4/manifest/artifact tests | `57 passed in 17.00s` |
| Broad first-principles/hybrid lane | `260 passed, 9 warnings in 33.66s` |
| Artifact linter, active root | `45 scanned, 3 first-principles, 3 passed, 0 failed, 11 exempt` |
| Artifact linter, recursive | `81 scanned, 39 first-principles, 3 passed, 0 failed, 47 exempt` |
| Source-truth exhaustion | `exhausted=true`, `open_issue_count=0` |
| Module source-vetting | `strict_passed=true`, `total_modules=289`, `active_physics_unvetted_count=0` |
| Ruff on round-touched/gate files | `All checks passed` |
| `git diff --check` | clean |

The broad runtime suite emitted PlasmaPy Coulomb-log warnings in strong-coupling
regimes. Those warnings are not test failures, but they are physics-regime
warnings and must be tracked in WP-N5 closure policy before any accepted
transport or collision claim.

## What Was Accepted As Engineering Progress

### WP-N1 Power Port

The runtime now emits a candidate Auluck-style five-term ledger:

- terminal port work;
- Omega-volume `J.E` work;
- wall Poynting flux excluding the declared port;
- stored EM energy delta;
- electrode/interface work.

The code correctly keeps this non-accepting. In
`src/dpf/first_principles/power_port.py`, term 4 is explicitly labeled
`electrode_interface_work_J__closure_estimate_not_independent`, and
`can_support_first_principles_acceptance` remains `False`.

### WP-N4 Segmented Whole-Shot Runner

The runtime now has:

- a static segment planner;
- segmented execution;
- checkpoint roundtrip through the fail-closed loader;
- per-segment manifests;
- cumulative ledgers;
- small-horizon restart-equivalence evidence;
- explicit 12 us compute-wall blocker reporting.

This is useful engineering infrastructure. It does not prove a completed 12 us
shot.

### Source Truth And Module Vetting

The source-truth and module-vetting reports now pass their strict checks. This
means the current source-truth inventory and active physics module citation
surface are cleaner. It does not mean the missing physics has been solved.

## Request Changes

### A-1: Manifest Provenance Is Still Too Weak

`src/dpf/first_principles/manifest.py` has a `source_packet_hashes` field, but
`source_packet_hashes` is not part of `REQUIRED_PROVENANCE_FIELDS`.

Direct probe result:

```text
{'provenance_complete': True, 'missing': [], 'source_packet_hashes': {}}
```

A manifest can therefore report complete provenance with no source packet
hashes. That is not acceptable for source-truth-controlled first-principles
work.

Required correction:

- Add `source_packet_hashes` to required provenance.
- Treat an empty dict as missing.
- Add a negative test where all other fields are populated but
  `source_packet_hashes={}`; expected result is
  `provenance_complete=false`.
- Update artifact linter C7 so it checks the actual manifest fields, not only
  `manifest.provenance_complete is True`.

### A-2: Active Result Artifacts Are Stale Relative To Current HEAD

Current HEAD is `76480b0`.

The three active first-principles result artifacts embed generation commit
`466a0a54e992acf61a9dd0f2d12e7e15fd23e9af`:

| Artifact | Embedded generation commit | Matches HEAD |
| --- | --- | --- |
| `results/audit_first_principles_3d_smoke.json` | `466a0a54e992acf61a9dd0f2d12e7e15fd23e9af` | false |
| `results/audit_experimental_whole_shot_smoke.json` | `466a0a54e992acf61a9dd0f2d12e7e15fd23e9af` | false |
| `results/audit_limiter_proof_auluck_power_port_1us_2026_05_18.json` | `466a0a54e992acf61a9dd0f2d12e7e15fd23e9af` | false |

Required correction:

- Regenerate these artifacts from a clean `76480b0` or newer HEAD; or
- move them into a quarantined archive and mark them unable to support any
  active first-principles claim.

### A-3: CI Gates Are Not Read-Only

These commands rewrite dated docs when run:

- `scripts/verify_first_principles_source_truth_exhaustion.py --strict`
- `scripts/verify_first_principles_module_source_vetting.py --strict`

During this audit they changed only timestamps in:

- `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.json`
- `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.md`
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.json`
- `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_EXHAUSTION_2026_05_18.md`

Required correction:

- Add `--check` or `--no-write` modes to both scripts.
- CI must use the no-write mode.
- CI must run `git diff --exit-code` after the audit gates.
- Report-regeneration tests must write only to `tmp_path` or be excluded from
  read-only CI globs.

### A-4: CI Artifact Scope Does Not Exercise Archive Policy

The artifact linter supports recursive scanning and explicit archive
exemptions, but CI currently exercises only active root artifacts. Archive
exemption logic can drift without a CI failure.

Required correction:

- Either run `scripts/audit_first_principles_artifacts.py 'results/**/*.json'`
  in CI; or
- add a separate archive-inventory gate proving that archived artifacts remain
  quarantined, visibly exempted, and unable to support first-principles
  acceptance.

### A-5: Broad Ruff Still Fails

Round-touched files pass ruff. The broader first-principles/fields/test lint
slice still fails with 7 issues:

- `src/dpf/fields/__init__.py`: unsorted imports.
- `src/dpf/fields/maxwell_3d.py`: two quoted type annotations.
- `src/dpf/fields/particle_boundaries.py`: quoted type annotation.
- `src/dpf/first_principles/gv_waveforms.py`: nested `with`, and `zip()`
  without explicit `strict=`.
- `tests/test_first_principles_mhd.py`: unsorted imports.

Required correction:

- Fix the broad lint slice or create an explicit, dated allowlist explaining
  why each item is outside the WP-N1/WP-N4 acceptance scope.

### A-6: Resume Ledger Continuity Is Broken

The resumed segmented runner can complete the horizon while the cumulative
ledger covers only the post-resume segment.

Direct probe:

```json
{
  "covers_executed_horizon": false,
  "ledger_steps": 2,
  "resume_started_at_step": 2,
  "total_steps_completed": 4
}
```

The current test verifies that resume completes and that fingerprints match,
but does not assert `cumulative_ledgers.covers_executed_horizon is True` after
resume.

Required correction:

- Persist cumulative ledger counters in checkpoint metadata, or load the prior
  run manifest and rehydrate `_CumulativeLedgers` on resume.
- Add a regression test where a 4-step run resumes from step 2 and requires:
  - `total_steps_completed == 4`;
  - `cumulative_ledgers.limiter_steps_observed == 4`;
  - `cumulative_ledgers.covers_executed_horizon is True`;
  - cumulative `J.E` and active-port step counts cover all executed steps.

### A-7: SRS/RTM Status Is Stale

`DPF-PHYS-020` still describes the named Auluck domain and five-term ledger as
blocked, even though a candidate implementation now exists.

`DPF-PHYS-023` still describes staged restart-equivalence as incomplete, even
though small-horizon evidence now exists.

The statuses should not be promoted to accepted. They should be updated to
partial/candidate with explicit acceptance blockers.

Required correction:

- Update `docs/DPF_REQUIREMENTS_BASELINE.md`.
- Regenerate `docs/SRS_TRACEABILITY_MATRIX.csv`.
- Regenerate `docs/SRS_TRACEABILITY_MATRIX.json`.
- Update the sprint plan so Sprint 0 is marked closed-with-debt rather than
  still broadly pending.
- Either initialize Doorstop from `DPF-PHYS-020..026`, or add a dated decision
  that CSV/JSON remain the temporary review baseline.

## Physics Blockers That Remain Open

### P-1: Power-Port Acceptance

WP-N1 is candidate-only. Acceptance remains blocked by:

- non-independent electrode/interface work;
- no legible accepted Auluck eq. 5/6 implementation for moving-boundary
  electrode work;
- no source-backed residual tolerance;
- only step-consistent time-centering, not accepted high-order time-centering;
- candidate geometry masks;
- no same-scope power-port review packet.

The current residual closes by construction because term 4 is derived from the
other terms. That residual is useful debugging telemetry. It cannot be used as
acceptance evidence.

### P-2: Startup BVP

Whole-shot startup remains blocked. The simulator still lacks accepted
source-backed channel packets for:

- breakdown;
- preionization;
- flashover;
- secondary emission;
- surface plasma;
- initial E/B/J;
- density and species;
- ionization;
- electron and ion temperatures;
- sheath surface and liftoff;
- handoff interval.

Seeded startup must remain rejected for first-principles claims.

### P-3: PF-1000 Geometry And Materials

Runtime masks are useful candidates, but not reviewed PF-1000 material
authority. Required masks still include:

- 12 cathode rods;
- hollow anode bore;
- alumina insulator;
- backplate/source interface;
- chamber wall;
- PML/open boundary;
- plasma domain.

Each must carry source dimensions, projection errors, resolution criteria, and
mask hashes.

### P-4: Physics Closures

WP-N5 remains open for:

- EOS/thermodynamics;
- ionization validity ranges;
- conductivity regimes;
- electron/ion energy coupling;
- radiation losses;
- ablation and impurities;
- anomalous resistance;
- restrike;
- electron inertia;
- stopping/collisions;
- beam-target coupling;
- floor/limiter policy reconciliation.

The PlasmaPy Coulomb-log warnings in current tests must be routed into this
closure-regime gate.

### P-5: Neutron Mechanism Authority

Scalar neutron yield is not enough. WP-N6 remains blocked until the runtime
separates:

- thermonuclear histories;
- beam-target histories;
- ion distribution functions;
- stopping;
- spectrum;
- anisotropy;
- detector response;
- UQ;
- same-scope comparator binding.

## Next Required Submission

The engineering team should submit a control-plane correction packet before any
new physics-promotion claim.

The next three sprints are not allowed to end with informal notes. They must
end with a reviewable research-and-implementation bundle that lets Codex audit
all changed code, all generated artifacts, and all proposed physics closures
without guessing what is claimed.

Each sprint must separate four things:

1. `implemented_engineering_candidate`: code/tests/artifacts that now run but
   remain non-accepting.
2. `source_backed_physics_proposal`: equations or algorithms supported by local
   `KnowledgeReference/` records, with exact file paths and line ranges.
3. `external_candidate_not_authority`: outside papers, repositories, web pages,
   or AI-derived suggestions that are useful leads but not authority until
   ingested into `KnowledgeReference/`.
4. `blocked`: anything still missing, illegible, cross-scope, under-reviewed,
   or not implemented.

Do not collapse those categories. A proposal may be high quality and still be
blocked if the source packet is missing or incomplete.

## Required Three-Sprint Package Structure

All three sprint submissions must live under one dated folder:

`docs/external_team_submissions/YYYY_MM_DD_three_sprint_blocker_packet/`

The folder must contain these top-level files:

| File | Required content |
| --- | --- |
| `README.md` | One-page index of every sprint, every claim, every blocker, and every artifact. |
| `CLAIMS_LEDGER.csv` | One row per claim, with claim type, source status, implementation status, artifact path, and promotion status. |
| `BLOCKER_MATRIX.csv` | One row per blocker listed below, with current status and sprint owner. |
| `SOURCE_PACKET_INDEX.csv` | One row per local source packet, with `KnowledgeReference/` path, line range, source hash, claim supported, and proposal ID. |
| `EXTERNAL_LEADS_NOT_AUTHORITY.md` | Any external leads researched but not yet ingested into `KnowledgeReference/`. |
| `UNKNOWN_AND_INFERENCE_LOG.md` | Every inferred parameter, algebraic reconstruction, missing value, or unresolved ambiguity. |
| `AUDIT_COMMANDS.md` | Exact commands run, environment, pass/fail result, and generated output paths. |
| `CHANGELOG.md` | Commit list, changed paths, deleted/quarantined artifacts, and generated artifacts. |
| `THREE_SPRINT_FINAL_SUMMARY.md` | Final team summary using only allowed claim language. |

If code is changed, the submission must also include:

- `PATCH_SCOPE.md`: why each changed file was necessary.
- `TEST_MAP.csv`: changed behavior mapped to tests and commands.
- `ARTIFACT_HASHES.csv`: every generated runtime artifact with path, SHA256,
  artifact generation commit, dirty-worktree flag, and whether it is active or
  quarantined.
- `RTM_DELTA.md`: every changed requirement row and whether Doorstop, CSV/JSON,
  or both are the current review baseline.

## Proposal Format Required For Every Blocker

Every physics or control blocker proposal must use this exact section order.
Missing sections are audit failures.

```markdown
# Proposal: <proposal ID> <short title>

Status: proposed | implemented_candidate | blocked | rejected
Sprint: 1 | 2 | 3
Blocker IDs: <for example WP-N1B, DPF-PHYS-020, A-1>
Claim allowed: <exact allowed runtime/doc claim>
Claim forbidden: <exact claim that remains blocked>

## 1. Scope

Define the device, shot, phase interval, geometry, solver path, and observable.
Say whether this is PF-1000/Akel-only or general DPF. General claims are not
allowed until the PF-1000/Akel path is complete.

## 2. Local Source Authority

Table with one row per local source:

| Source path | Lines | Source hash | Supports | Limits |
| --- | --- | --- | --- | --- |

Only `KnowledgeReference/` paths may be used as authority. External papers,
web pages, GitHub repositories, and AI output go in `External Leads`, not here.

## 3. Equations And Symbol Map

List every equation, every symbol, units, sign convention, coordinate system,
and any nondimensionalization. If a formula is reconstructed algebraically,
label it `inference_from_local_sources` and name the equations used.

## 4. Validity Regime

State density, temperature, coupling, magnetization, collisionality, geometry,
timestep, mesh, boundary, and phase limits. Include known warning regimes such
as PlasmaPy Coulomb-log strong-coupling warnings.

## 5. Proposed Numerical Method

Define discretization, centering, solver ordering, conservation terms, boundary
handling, limiter/floor behavior, and failure modes. If this is only research,
give pseudocode and expected code paths.

## 6. Implementation Plan

List exact files to change, new files to add, interfaces to expose, and
artifacts to emit. Separate required implementation from optional cleanup.

## 7. Test Plan

List positive tests, negative tests, regression tests, property/conservation
tests, and artifact-linter checks. Name the expected test files.

## 8. Runtime Artifacts

Name every artifact the code must emit. Include required fields, hashes,
manifest fields, and fail-closed labels.

## 9. Acceptance And Rejection Criteria

Define what would make Codex accept engineering progress, reject it, or keep it
blocked. Do not use the word validated unless same-scope validation evidence is
actually present.

## 10. Open Questions

List every unresolved source gap or engineering ambiguity. Each item must have
an owner and a proposed next action.

## 11. AI And External Tool Disclosure

List AI tools, search tools, external repositories, and generated code/prose.
State exactly what was human-reviewed and what was rejected.
```

## Claims Ledger Format

`CLAIMS_LEDGER.csv` must have these columns:

```text
claim_id,
sprint,
work_package,
claim_text,
claim_type,
source_status,
local_source_paths,
implementation_status,
code_paths,
test_paths,
artifact_paths,
can_support_first_principles_acceptance,
can_support_validation_claims,
blocked_by,
allowed_public_language,
forbidden_language
```

Valid `claim_type` values:

- `control_gate`
- `runtime_engineering`
- `physics_equation`
- `numerical_method`
- `geometry`
- `startup`
- `closure`
- `neutron`
- `comparator_uq_certificate`
- `external_lead_not_authority`

Valid `source_status` values:

- `local_authority_packet_complete`
- `local_authority_packet_partial`
- `local_source_illegible`
- `local_source_cross_scope`
- `external_candidate_not_ingested`
- `ai_suggestion_not_authority`
- `blocked_no_source`

Any row with `can_support_first_principles_acceptance=true` will be rejected
unless the proposal also contains same-scope source packets, code, tests,
artifacts, comparator/UQ evidence, and a certificate path. The expected value
for the next three sprints is almost always `false`.

## Blocker Matrix Required Rows

`BLOCKER_MATRIX.csv` must include at least these rows:

| Blocker ID | Required sprint treatment |
| --- | --- |
| A-1 manifest provenance | Implement and test non-empty source packet hashes. |
| A-2 stale active artifacts | Regenerate from HEAD or quarantine. |
| A-3 read-only CI gates | Add no-write/check mode and `git diff --exit-code`. |
| A-4 archive policy CI coverage | Recursive linter or archive inventory gate. |
| A-5 broad ruff failures | Fix or explicit allowlist. |
| A-6 resume ledger continuity | Fix cumulative ledger rehydration on resume. |
| A-7 stale SRS/RTM | Update requirement status and regenerate RTM. |
| WP-N1B power-port acceptance | Research and propose independent electrode/interface work, residual tolerance, and time-centering. |
| WP-N4B 12 us segmented runtime | Propose production orchestration, wall-time slicing, ledger merge, and artifact combiner. |
| WP-N2 startup BVP | Research every startup channel packet and propose runtime handoff. |
| WP-N3 PF-1000 geometry/material masks | Research and propose reviewed masks and resolution gates. |
| WP-N5 EOS and closure registry | Research closure packets and regime gates. |
| WP-N5 radiation/ablation/impurity | Research packet and runtime integration proposal. |
| WP-N5 anomalous resistance/restrike | Research packet and runtime trigger criteria proposal. |
| WP-N5 collision/stopping/electron inertia | Research packet and PlasmaPy/regime-warning policy. |
| WP-N6 neutron mechanism authority | Research thermonuclear/beam-target separation, spectra, anisotropy, detector response, and UQ. |
| WP-N7 comparator/UQ/certificate | Propose same-scope comparator bundle and certificate gating. |
| numerical acceptance | Propose convergence, limiter-zero, backend parity, and restart reproducibility gates. |

No blocker row may be deleted. If the team believes a blocker is obsolete, mark
it `proposed_obsolete` and justify it with code paths, tests, and source
packets. Codex will decide whether it is actually obsolete.

## Three-Sprint Work Plan

### Sprint 1: Control Gate Hardening And Review Baseline

Goal: make the repository audit-stable before more physics claims are added.

Required implementation:

1. Fix manifest provenance so `source_packet_hashes={}` cannot pass.
2. Fix artifact linter C7 to inspect actual required fields.
3. Add read-only mode to source-truth and module-vetting scripts.
4. Make CI use read-only gates and fail on dirty diffs.
5. Exercise archive policy in CI.
6. Regenerate or quarantine active stale result artifacts.
7. Fix broad ruff failures or create a dated allowlist.
8. Fix segmented resume cumulative-ledger continuity.
9. Update `DPF-PHYS-020` and `DPF-PHYS-023` to candidate/partial status with
   explicit blockers, then regenerate RTM exports.

Required research/proposal output:

- `sprint_1/CONTROL_GATE_PROPOSAL.md`
- `sprint_1/RESUME_LEDGER_CONTINUITY_PROPOSAL.md`
- `sprint_1/ARTIFACT_REGENERATION_OR_QUARANTINE_PLAN.md`
- `sprint_1/SRS_RTM_BASELINE_DECISION.md`

Sprint 1 exit criteria:

- `git status --short` clean except intentional committed files.
- read-only gate commands do not change files.
- active artifacts match current HEAD or are quarantined.
- resumed segmented runs cover all executed ledger steps.
- broad ruff slice is clean or explicitly allowlisted.
- claims ledger exists and has no acceptance promotion.

### Sprint 2: WP-N1B Power-Port And WP-N4B Long-Run Orchestration

Goal: burn down the first physics authority blocker and make the 12 us path
operationally honest.

Required research/proposal output:

- `sprint_2/WP_N1B_POWER_PORT_ACCEPTANCE_PROPOSAL.md`
- `sprint_2/WP_N1B_AULUCK_EQ_5_6_SOURCE_STATUS.md`
- `sprint_2/WP_N1B_RESIDUAL_TOLERANCE_SOURCE_STATUS.md`
- `sprint_2/WP_N1B_TIME_CENTERING_PROPOSAL.md`
- `sprint_2/WP_N4B_12US_ORCHESTRATION_PROPOSAL.md`
- `sprint_2/WP_N4B_LEDGER_MERGE_AND_ARTIFACT_COMBINER_PROPOSAL.md`

WP-N1B must answer these questions:

1. Is Auluck eq. 5/6 legible in local `KnowledgeReference/`?
2. If yes, what is the exact equation, symbol map, unit map, sign convention,
   and discrete implementation?
3. If no, what local-source-backed alternative exists?
4. If no alternative exists, what external candidate source must be ingested
   before implementation?
5. How will electrode/interface work be computed independently rather than
   solved as a residual closure?
6. What residual tolerance is source-backed, or what review packet will define
   it?
7. What negative tests prove the implementation fails closed when the domain,
   sign, centering, electrode term, or low-current behavior is wrong?

WP-N4B must answer these questions:

1. What job plan is required for a 12 us source-sign run at the selected `dt`?
2. How many steps, segments, checkpoints, and estimated wall-clock hours?
3. What artifacts prove partial, resumed, and completed runs?
4. How are cumulative ledgers merged across restarts?
5. What field/circuit/particle/ionization/kinetic/power-port data are retained,
   summarized, or chunked?
6. What makes a run `horizon_complete=true`, and what keeps partial runs from
   being mislabeled?
7. What is the minimum smaller production ladder before attempting 12 us
   (for example 10 ns, 100 ns, 1 us, then 12 us)?

Sprint 2 exit criteria:

- WP-N1B has a source-status verdict for every required term.
- If code is changed, all existing WP-N1 negative tests still pass and new
  tests cover any new term.
- WP-N4B has an executable orchestration proposal and at least one small
  artifact-combiner or ledger-merge proof if implementation is attempted.
- No proposal claims accepted power-port authority unless all required source,
  implementation, residual, geometry, and review packets exist.

### Sprint 3: Remaining Physics Blockers Research Packet

Goal: produce reviewable proposals for every remaining physics blocker, even
where implementation remains blocked.

Required research/proposal output:

- `sprint_3/WP_N2_STARTUP_BVP_PROPOSAL.md`
- `sprint_3/WP_N2_STARTUP_CHANNEL_MATRIX.csv`
- `sprint_3/WP_N3_PF1000_GEOMETRY_MASK_PROPOSAL.md`
- `sprint_3/WP_N3_GEOMETRY_SOURCE_DIMENSION_TABLE.csv`
- `sprint_3/WP_N5_CLOSURE_REGISTRY_PROPOSAL.md`
- `sprint_3/WP_N5_CLOSURE_PACKET_MATRIX.csv`
- `sprint_3/WP_N6_NEUTRON_AUTHORITY_PROPOSAL.md`
- `sprint_3/WP_N6_NEUTRON_MECHANISM_MATRIX.csv`
- `sprint_3/WP_N7_COMPARATOR_UQ_CERTIFICATE_PROPOSAL.md`
- `sprint_3/NUMERICAL_ACCEPTANCE_PROPOSAL.md`

WP-N2 startup proposal must include one row per required channel:

- breakdown;
- preionization;
- flashover;
- secondary emission;
- surface plasma;
- initial E/B/J;
- density and species;
- ionization;
- electron temperature;
- ion temperature;
- sheath surface;
- sheath liftoff;
- handoff interval.

Each row must state source status, equation or data requirement, runtime state
field, units, implementation path, test plan, and blocker status.

WP-N3 geometry proposal must include masks for:

- 12 cathode rods;
- hollow anode outer radius and bore;
- alumina insulator;
- backplate/source interface;
- chamber wall;
- PML/open boundary;
- plasma domain.

Each mask must have source dimensions, source line ranges, projection method,
resolution metric, under-resolution behavior, mask SHA256 field, and tests.

WP-N5 closure proposal must include at least:

- EOS/thermodynamics;
- ionization/recombination;
- conductivity/resistivity;
- Hall/electron inertia;
- electron-ion energy exchange;
- electron and ion heat flux;
- radiation;
- ablation/impurity;
- anomalous resistance;
- restrike;
- Coulomb collisions and stopping;
- beam-target coupling;
- floor/limiter policy.

Each closure must be classified as:

- `active_source_backed_candidate`;
- `active_blocked`;
- `bounded_out_with_source`;
- `not_simulated_and_claim_blocking`;
- `external_candidate_not_authority`.

WP-N6 neutron proposal must separate:

- thermonuclear production;
- beam-target production;
- ion distribution function;
- stopping;
- neutron spectrum;
- anisotropy;
- detector response;
- activation/TOF response;
- UQ;
- same-scope comparator targets.

WP-N7 proposal must define the review bundle an external engineering firm
would receive:

- deck and geometry packet;
- source packet index;
- runtime manifest;
- environment lock;
- command transcript;
- active/quarantined artifact list;
- convergence/numerical packet;
- comparator packet;
- UQ packet;
- certificate gate output;
- forbidden-claim statement.

Sprint 3 exit criteria:

- Every blocker in `BLOCKER_MATRIX.csv` has a proposal, implemented candidate,
  or explicit blocked verdict.
- Every proposal uses the required section order.
- Every local-source claim has `KnowledgeReference/` paths and line ranges.
- Every external lead is isolated from authority claims.
- Every inferred parameter is logged in `UNKNOWN_AND_INFERENCE_LOG.md`.

## How Codex Will Audit All Three Sprints

Codex will audit the final three-sprint bundle in this order:

1. **Worktree and provenance audit**
   - `git status --short`
   - commit list and changed files
   - dirty-worktree checks
   - active artifact generation commit equals HEAD
   - source packet hashes are non-empty and required

2. **Claim safety audit**
   - scan every artifact and doc for acceptance claims;
   - verify `can_support_first_principles_acceptance` remains false unless the
     full same-scope gate is actually present;
   - verify reduced models are only baselines/comparators;
   - verify external leads are not treated as authority.

3. **Source packet audit**
   - sample every source packet against local `KnowledgeReference/`;
   - verify path, line range, hash, and claimed equation;
   - mark proposals blocked if sources are missing, cross-scope, illegible, or
     merely inferred without an inference label.

4. **Implementation audit**
   - inspect changed code paths;
   - run targeted tests;
   - run broad first-principles/hybrid tests;
   - run artifact linter active and recursive scans;
   - verify no generated timestamp churn.

5. **Physics blocker audit**
   - evaluate WP-N1B through WP-N7 proposal by proposal;
   - classify each as accepted engineering progress, request changes, blocked
     by source, blocked by implementation, or rejected for overclaiming.

6. **SRS/RTM audit**
   - verify requirement statuses match code and source evidence;
   - verify RTM exports are regenerated;
   - verify Doorstop decision is explicit;
   - reject any implemented status without a test, artifact, or inspection
     path.

7. **Final instruction audit**
   - produce a new Codex audit document;
   - produce a new next-instructions document;
   - identify which sprint outputs may be implemented next and which remain
     research-only.

Codex will not accept "we researched it" as closure. Research closes a blocker
only when it identifies a local authority source, a concrete algorithm, a
runtime interface, tests, artifacts, and remaining claim limits. Otherwise it
is useful research, but the blocker remains open.

### Submission 1: Control Gate Hardening

Required deliverables:

1. Manifest provenance fix requiring non-empty `source_packet_hashes`.
2. Linter C7 checks actual manifest required fields.
3. No-write modes for both source-truth verification scripts.
4. CI uses no-write gates and fails on `git diff --exit-code`.
5. CI exercises recursive artifact/archive policy or a dedicated archive
   inventory gate.
6. Active `results/audit_*.json` artifacts regenerated from current clean HEAD
   or quarantined.
7. Broad ruff slice fixed or explicitly allowlisted.
8. Resume-ledger continuity fixed and covered by regression tests.
9. SRS/RTM updated for `DPF-PHYS-020` and `DPF-PHYS-023`.

Minimum commands expected in the submission:

```bash
git status --short
git diff --check
.venv312/bin/python -m ruff check scripts/verify_first_principles_source_truth_exhaustion.py scripts/audit_first_principles_artifacts.py src/dpf/first_principles src/dpf/fields tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py
.venv312/bin/python scripts/audit_first_principles_artifacts.py 'results/*.json'
.venv312/bin/python scripts/audit_first_principles_artifacts.py 'results/**/*.json'
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict --check
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --strict --check
.venv312/bin/python -m pytest tests/test_first_principles_manifest.py tests/test_first_principles_artifact_linter.py tests/test_first_principles_segmented_whole_shot.py tests/test_srs_traceability_export.py -q -rx
.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py -q -rx
git diff --exit-code
```

If the team chooses `--no-write` instead of `--check`, use that exact option in
the command list and document the semantics.

### Submission 2: WP-N1B Power-Port Acceptance Burn-Down

This should not begin until Submission 1 is clean.

Required deliverables:

1. A legible, source-backed implementation or reviewed replacement for Auluck
   eq. 5/6 electrode/moving-boundary work.
2. Independent electrode/interface work computation, not a closure residual.
3. Source-backed residual tolerance or a review packet explicitly defining the
   residual criterion.
4. Time-centering policy with tests for mismatch/reversal.
5. Runtime domain masks tied to reviewed PF-1000 geometry packets.
6. Negative tests proving the port fails when any required term is missing,
   sign-reversed, source-interface-contaminated, or singular at low current.

### Submission 3: WP-N2 / WP-N3 Physics Foundation

After WP-N1B is clean, proceed in this order:

1. WP-N2 startup BVP packets and runtime handoff.
2. WP-N3 reviewed PF-1000 geometry/material masks.
3. WP-N5 closure registry and regime gates.
4. WP-N6 neutron mechanism authority.
5. WP-N7 comparator/UQ/certificate bundle.

Do not start WP-N8 generalization until PF-1000/Akel has a complete,
same-scope, fail-closed review bundle.

## Audit Method I Will Use On The Next Submission

The next audit will be rejected if any of the following is true:

- `git status --short` is dirty from generated timestamp churn.
- Active artifacts embed an `artifact_generation_commit` older than HEAD.
- `manifest.provenance_complete` can be true with empty
  `source_packet_hashes`.
- Source-truth verification scripts write files in CI mode.
- Archived artifacts are not either recursively linted or inventory-gated.
- Resume from checkpoint completes while cumulative ledgers cover only the
  post-resume steps.
- Any artifact contains `can_support_first_principles_acceptance: true`.
- Any new physics claim lacks a local `KnowledgeReference/` source packet.
- Any reduced model output is used as first-principles authority rather than
  a comparator or baseline.

The next audit will accept engineering progress only when each changed behavior
has:

- implementation path;
- source packet path;
- test path;
- runtime artifact path;
- current HEAD artifact generation hash;
- blocker state;
- explicit non-validation label unless all same-scope gates are actually met.
