# First-Principles Codex-Agent Audit And Next Instructions

Date: 2026-05-18

Repo: `/Users/anthonyzamora/dpf-unified`

Audit mode: six managed Codex agents plus local verification. This audit uses
`KnowledgeReference/` and explicitly user-verified staged sources as scientific
authority. It does not invoke the validation workflow and does not promote any
engineering run into accepted science.

## Verdict

Verdict: **accept Sprint 0 hygiene progress, request changes for runtime and
physics readiness**.

The other engineering team materially improved the repo. Active root
first-principles artifacts now pass the artifact linter, stale pre-SSR artifacts
are quarantined, source-truth exhaustion is clean, module source-vetting is
clean, and the focused first-principles test lane is green.

The simulator is still an **engineering candidate**, not a complete accepted
first-principles whole-shot DPF simulator. The next work must focus on runtime
authority: power port, 12 us segmented source-sign execution, startup BVP,
reviewed PF-1000 masks, closure packets, and neutron mechanisms.

## What Codex Fixed During This Audit

- Added a committed-CI candidate gate in `.github/workflows/ci.yml`:
  `first-principles-audit` now runs source-truth exhaustion, strict module
  source-vetting, and artifact linting before smoke tests.
- Fixed import-time failure risk in `scripts/verify_kr_pdf_parity.py`: missing
  PyMuPDF no longer calls `sys.exit()` during module import.
- Aligned PF-1000/Akel preset and server validation-scope labels with canonical
  constants from `src/dpf/validation/first_principles_mhd.py`.
- Removed the unused `sys` import from
  `scripts/verify_first_principles_module_source_vetting.py`.
- Added direct unit coverage for `scripts/audit_first_principles_artifacts.py`.

These changes remain local until committed.

## Audit Evidence

Commands run:

```bash
git status --short
git log --oneline -8
.venv312/bin/python scripts/audit_first_principles_artifacts.py 'results/*.json'
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --strict
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py
.venv312/bin/python -m pytest tests/test_first_principles_*.py tests/test_hybrid_3d_*.py tests/test_cli_first_principles_3d.py -q -rx
.venv312/bin/python -m pytest tests/test_validation_ci.py -q -rx
.venv312/bin/python -m pytest tests/test_first_principles_artifact_linter.py tests/test_first_principles_module_source_vetting.py tests/test_preset_source_scope.py tests/test_akel_digitization_source_integrity.py -q -rx
.venv312/bin/python -m ruff check scripts/verify_kr_pdf_parity.py scripts/verify_first_principles_module_source_vetting.py tests/test_first_principles_module_source_vetting.py tests/test_first_principles_artifact_linter.py src/dpf/presets.py src/dpf/server/app.py
.venv312/bin/python -c 'import yaml, pathlib; yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text())'
git diff --check
```

Observed results:

- Artifact linter: 45 files scanned, 3 first-principles artifacts, 3 passed, 0
  failed.
- Source-truth exhaustion: `exhausted=true`, `open_issue_count=0`.
- Module source-vetting: `strict_passed=true`, 288 modules, 0 active physics
  unvetted.
- Focused first-principles lane: `214 passed`, 9 PlasmaPy strong-coupling
  warnings.
- Validation CI lane: `27 passed`, 1 skipped.
- Targeted regression lane: `23 passed`.
- Touched Python ruff: passed.
- YAML parse for CI: passed.
- `git diff --check`: passed.

## Do Not Claim

- accepted first-principles DPF simulator;
- solved full PF-1000/Akel whole-shot prediction;
- solved startup BVP;
- accepted power-port authority;
- reviewed PF-1000 material geometry;
- accepted closure matrix;
- accepted neutron mechanism or detector authority;
- validated current waveform, neutron yield, or same-scope PF-1000 result;
- Lee, snowplow, GV, or current-fit authority inside the first-principles path.

Allowed claim:

- package-native 3-D first-principles **engineering candidate** with fail-closed
  packets, source-truth-vetted modules, and improved evidence hygiene.

## Remaining Audit Findings

### A-1: Manifest Provenance Is Still Incomplete

Active artifacts now carry top-level provenance, but manifests still report
incomplete source provenance in current generated artifacts:

- `source_truth_index_sha256` is not populated in the manifest;
- `source_packet_hashes` is empty;
- linter checks do not yet fail on `manifest.provenance_complete=false`.

Instruction:

- Populate manifest `source_truth_index_sha256` from
  `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json`.
- Populate `source_packet_hashes` for every cited source packet used by the run.
- Extend artifact linter checks to fail when a first-principles artifact has
  `manifest.provenance_complete` missing or false.
- Regenerate the three active audit artifacts from a clean committed HEAD.

### A-2: Artifact Scope Policy Is Still Too Narrow

The linter correctly passes active root first-principles artifacts and correctly
fails stale archived artifacts when pointed at the archive. It still skips older
experimental PF-1000 evidence surfaces such as checkpoint/restart,
reproducibility, split-continuation, and numerical-family artifacts.

Instruction:

- Decide and implement one policy:
  - archive or remove every non-provenant PF-1000 engineering evidence artifact,
    then lint recursively; or
  - explicitly exempt archived/non-authority evidence in code and tests, with a
    status reason proving it cannot support first-principles acceptance.
- Add direct tests for the chosen archive/non-authority policy.

### A-3: SRS/RTM Story Mapping Is Stale

The Sprint 0 artifact failure statement was stale and has been corrected in the
sprint plan. The SRS/RTM layer still does not map WP-N1 through WP-N7 into
stable requirement IDs, owners, verification methods, evidence artifacts, and
blocker IDs.

Instruction:

- Add WP-N1 through WP-N7 rows to the RTM or Doorstop-ready table.
- Link them to existing physics requirements such as power-port, startup,
  geometry, closures, neutron authority, numerical fidelity, and certificate
  blockers.
- Do not start WP-N8 multi-machine generalization until PF-1000/Akel authority
  issues are closed or explicitly bounded.

### A-4: 12 us Source-Sign Whole-Shot Path Is Not Feasible Yet

The current package-native 12 us source-sign path is step-budget blocked. With
the current `dt=1e-13` planning, a 12 us run requires 120,000,000 steps. Existing
segmentation/checkpoint tests cover tiny step counts, not a reproducible 12 us
source-sign engineering run.

Instruction:

- Build a segmented whole-shot runner that plans target time, segment size,
  checkpoint cadence, wall-time cap, and resume manifest.
- Carry cumulative ledgers across segments: circuit state, `lagged_field_work`,
  particle state, electron/ion energy, ionization, kinetic-yield state, limiter
  inventory, power-port ledgers, and artifact provenance.
- Prove restart equivalence at staged horizons before attempting 12 us.
- Emit a run directory containing deck, command, commit, dirty flag, source
  hashes, segment manifests, checkpoint hashes, and blocker verdicts.

### A-5: Power Port Is The First Physics Blocker

Local sources support the direction of work, but the runtime still lacks an
accepted field/circuit power port. Auluck provides the DPF circuit-element
relation using a volume `J.E` power integral and excludes the power-source
interface. The NRL formulary provides the Poynting theorem basis.

Instruction:

- Implement named Auluck `Omega` domain with source-interface exclusion.
- Emit domain mask hash, cell count, bounds, and source refs.
- Emit terminal work, volume `J.E`, wall Poynting excluding declared port,
  electrode/interface work, and stored-field delta.
- Declare sign convention and time-centering.
- Add negative tests for sign reversal, wrong domain, omitted electrode work,
  low-current `P/I`, first-step fallback, and default-mode leakage.
- Keep acceptance blocked until residual policy is reviewed and source-backed.

### A-6: Startup BVP Remains Blocked

Startup fail-open behavior is fixed, but the physics is not. Seeded startup and
CIV/Paschen telemetry remain non-promoting.

Instruction:

- Build one startup channel packet per required channel: breakdown,
  preionization, flashover, secondary emission, surface plasma, initial E/B/J,
  density/species, ionization, Te/Ti, sheath surface, and handoff interval.
- Accept only a typed imported-PIC sheath state or a solved surface-breakdown
  BVP packet.
- Missing channels must block before runtime or produce an explicit fail-closed
  startup artifact.

### A-7: PF-1000 Geometry Masks Remain Candidate

Mask hash and projection metrics are present. The actual material geometry is
not complete.

Instruction:

- Add separate masks and hashes for 12 cathode rods, hollow anode bore, alumina
  insulator, backplate/source interface, chamber wall, PML/open boundary, and
  plasma domain.
- Reconcile PF-1000 geometry source differences in the deck-diff packet.
- Reviewed masks must fail if rod, bore, insulator, or sheath-relevant material
  surfaces are under-resolved.

### A-8: Closure And Neutron Authority Remain Candidate/Blocked

Instruction:

- For every active or bounded-out closure, emit source equation, units, symbol
  map, validity range, implementation path, tests, sensitivity/UQ status, and
  claim impact.
- Keep EOS, radiation, ablation/impurity, anomalous resistance, restrike,
  electron inertia, stopping/collisions, and beam-target coupling explicitly
  accepted, candidate, blocked, or bounded out.
- Split neutron history into thermonuclear and beam-target channels.
- Block neutron authority without spectrum, anisotropy, detector response, and
  UQ consumed by the runtime.

## Next Submission Required From Engineering Team

The next submission should be a **WP-N1/WP-N4 runtime authority packet**, not a
new broad plan.

Required contents:

1. A clean commit with CI first-principles audit gates active.
2. Updated artifact linter with manifest provenance checks and archive policy.
3. Regenerated active artifacts from clean HEAD.
4. Named Auluck power-port domain and five-term ledger implementation.
5. Staged segmented runner design and first restart-equivalence evidence.
6. Updated RTM rows for WP-N1 through WP-N7.
7. Test/lint table with exact commands and outputs.

Acceptance for the next review:

- `git status --short` is clean or every dirty file is explained.
- Source-truth and module-vetting gates pass.
- Artifact linter passes according to the new stricter policy.
- Focused first-principles tests pass with `-rx`.
- The new power-port packet emits complete candidate ledger terms.
- The simulator still reports `engineering_candidate_not_validation` until the
  source-backed residual/power-port review is actually closed.

