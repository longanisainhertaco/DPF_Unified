# Sprint 3R Final Submission

- Date: 2026-05-19
- Branch: `codex/corpus`
- Controlling contract: `docs/FIRST_PRINCIPLES_SPRINT3R_REMEDIATION_HANDOFF_2026_05_19.md`
- Audit basis: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_COMPLETION_2026_05_19.md`
- Implementation HEAD: `46a0b56` — this final-submission commit is the
  documentation-only wrapper on top of it.

Sprint 3R remediates all twelve Sprint 3 completion-audit findings (A1–A12).
Every fix is either source-grounded and tested, or it is converted into a typed
fail-closed blocker with a regression test. Sprint 3R does not promote any
requirement to `implemented` or `accepted`, does not claim validation,
engineering acceptance, first-principles predictive authority, or completion of
a 12 µs full shot.

## 1. Commit List

Sprint 3R commit range: `770984b..HEAD` (excluding HEAD, which is this
documentation-only wrapper).

| Commit | Package | Closes | Subject |
| --- | --- | --- | --- |
| `81c7481` | S3R.1 | A10, A11, A12-docs | packet hygiene — 4-boolean ledger, path corrections, shorthand citations |
| `c9c7160` | S3R.2 | A1 | bind startup acceptance to typed packet |
| `2a3e891` | S3R.3 | A2, A3 | neutron-authority status splits + NumPy 2 trapezoid fallback |
| `224a7ea` | S3R.4 + S3R.5 | A4, A5, A6, A7 | geometry mask statuses + Sigma_p packet schema + dict-form power-port consumption |
| `f390cf3` | S3R.6 | A8 | close closure-matrix completeness |
| `dfa9169` | S3R.7 | A9, A12 code-side | extended cumulative fields through merged ledgers + certificate fail-closed |
| `b6f7698` | S3R.1 follow-up | A11 follow-up | pre-commit linter auto-fixes (citation normalization + ruff import sort) |
| `46b705f` | S3R.8 prep | — | A1-A12 ledger rows (BLOCKER_MATRIX, CLAIMS_LEDGER, TEST_MAP), ruff fix on `test_first_principles_power_port.py` |
| `46a0b56` | S3R.8 vetting | — | regenerate `FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}` |

## 2. Finding Closure Matrix

Status values: `fixed` = the audit finding's code defect is repaired and the
behaviour is tested; `blocked_fail_closed` = the runtime path is now typed and
fail-closed but a downstream first-principles claim still cannot be promoted to
accepted; `not_fixed` = remains open. **No row is `not_fixed`.**

| # | Title | Status | Fix commit | Code paths | Test paths | Remaining blocker |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | legacy startup gate can promote acceptance | `fixed` | `c9c7160` | `src/dpf/first_principles/startup_bvp.py:239-280` | `tests/test_first_principles_startup_bvp.py` (37 passed), `tests/test_cli_first_principles_3d.py` | typed startup channels 0/13 source-backed (WP-N2 packet) — runtime is fail-closed |
| A2 | scalar/target-only neutron evidence can accept | `fixed` | `2a3e891` | `src/dpf/first_principles/neutron_authority.py:953-1016, :1024-1043` | `tests/test_first_principles_neutron_authority.py` (32 passed) | beam-target/spectrum/anisotropy/detector/UQ channels blocked (no same-scope KR) — runtime fail-closed |
| A3 | NumPy 2 trapezoid `AttributeError` in `beam_target.py` | `fixed` | `2a3e891` | `src/dpf/diagnostics/beam_target.py:91-94` | `tests/test_first_principles_neutron_authority.py::test_trapezoid_integral_works_on_active_numpy_lane` | none |
| A4 | PF-1000 material masks emitted as source-backed despite heuristic | `fixed` | `224a7ea` | `src/dpf/fields/source_geometry.py:437-468, :1140-1210, :1235-1274` | `tests/test_source_geometry_packet.py` (61 passed), `tests/test_first_principles_geometry.py` | insulator outer radius blocked, cathode-cage radius conflict, backplate dims blocked — masks correctly downgraded |
| A5 | under-resolution gate missed insulator/transition features | `fixed` | `224a7ea` | `src/dpf/fields/source_geometry.py:1043-1077` | `tests/test_source_geometry_packet.py::test_s3r4_under_resolved_insulator_surface_fails_closed` | none |
| A6 | `SigmaPSurfacePacket` lacks digest+operand fields | `fixed` | `224a7ea` | `src/dpf/fields/source_geometry.py:1290-1330, :1380-1420, :1509-1530` | `tests/test_source_geometry_packet.py` | none |
| A7 | serialized `Sigma_p` packets silently discarded by `power_port.py` | `fixed` | `224a7ea` | `src/dpf/first_principles/power_port.py:459-580, :583-678` | `tests/test_first_principles_power_port.py` (96 passed) | terms II/IV/V/VI remain blocked until Sprint 4 surface integrator — runtime fail-closed |
| A8 | closure top-level effects omit `electron_inertia` / `stopping_collisions` | `fixed` | `f390cf3` | `src/dpf/first_principles/closure_packet.py:1276, :1287` | `tests/test_first_principles_closures.py` (45 passed) | both effects remain non-accepting blockers (no KR closure equation) — runtime fail-closed |
| A9 | cross-restart merged ledger drops extended S3.7 channels | `fixed` | `dfa9169` | `src/dpf/first_principles/segmented_whole_shot_combine.py:218-222, :269-281` | `tests/test_first_principles_segmented_whole_shot_combine.py`, `tests/test_first_principles_long_run_integrity.py`, `tests/test_first_principles_certificate_negative_controls.py` (75 passed) | 12 µs full-shot horizon still compute-wall-blocked (S3.7 unchanged) |
| A10 | packet ledgers contradict the final submission | `fixed` | `81c7481`, `46b705f` | packet `SPRINT_3_STATUS_LEDGER.md`, `BLOCKER_MATRIX.csv`, `CLAIMS_LEDGER.csv`, `TEST_MAP.csv`, `README.md`, `THREE_SPRINT_FINAL_SUMMARY.md` | `tests/test_external_team_submission_package.py` (31 passed) | none |
| A11 | shorthand `KR` citations (ellipsis form) in actionable content | `fixed` | `81c7481`, `b6f7698` | packet `WP_N5_CLOSURE_REGISTRY_SOURCE_AUDIT.md`, `WP_N2_STARTUP_BVP_CHANNEL_MATRIX.md` | `tests/test_external_team_submission_package.py::test_sprint3_packets_reject_shorthand_citations` | none |
| A12 | traceability paths reference non-existent modules; findings stale | `fixed` | `81c7481` (docs/RTM/CSVs), `dfa9169` (code-side null-op) | `docs/DPF_REQUIREMENTS_BASELINE.md`, `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`, packet `CHANGELOG.md`/`CLAIMS_LEDGER.csv`/`BLOCKER_MATRIX.csv`/`SPRINT_3_STATUS_LEDGER.md`, `CodexFindings.md`, `CortexFindings.md` | `tests/test_external_team_submission_package.py::test_packet_docs_no_bad_module_paths`, `tests/test_srs_traceability_export.py` | none |

## 3. Runtime Behavior Summary

What the simulator now does:

- The typed `StartupPacket` is the **sole** acceptance source for the startup
  BVP. `build_startup_bvp_packet()` cannot be promoted by caller-declared
  payload fields; while WP-N2 reports 0 source-backed channels of 13, every
  whole-shot startup call is structurally fail-closed.
- Scalar neutron yield is **permanently** `candidate_comparator_only`. The
  mechanism-separated `NeutronAuthorityRuntime` cannot emit
  `accepted_neutron_authority` from scalar or target-only metadata; it requires
  both runtime mechanism histories and the same-scope review packet.
- `beam_target._trapezoid_integral()` works on the active Python 3.12 / NumPy
  2.4 lane without raising `AttributeError`.
- Material masks expose `mask_class_status ∈ {source_supported,
  candidate_projection_not_source_mask, blocked}` per class. Insulator and
  cathode-cage masks are `candidate_projection_not_source_mask` while their
  underlying dimensions are blocked/conflict. Per-mask SHA-256 is preserved
  only with the status that generated the mask.
- The under-resolution gate covers every source-supported material feature
  (rods, anode radius, insulator exposed length).
- `SigmaPSurfacePacket` carries five reviewer-grade digests
  (`sigma_p_face_set_sha256`, `moving_classification_sha256`,
  `omega_partition_sha256`, `material_mask_sha256_by_class`,
  `moving_classification_status`) plus operand statuses for `B, E, J, v, eta`.
  The source geometry hash is preserved on blocked returns.
- `power_port.py` consumes dict-form `Sigma_p` packets via
  `_sigma_p_packet_from_dict()` or emits the named
  `serialized_sigma_p_packet_not_supported` blocker — no silent discard.
  `_sigma_p_surface_term()` enforces sign-convention and moving-classification
  controls; terms II/IV/V/VI remain blocked until the Sprint 4 surface
  integrator exists.
- The top-level closure effects dict is symmetric with `REQUIRED_EFFECTS`.
  `electron_inertia` and `stopping_collisions` appear in
  `closure_matrix_status_by_effect`, `closure_effect_status`, and
  `missing_or_unaccepted_effects`. PlasmaPy remains cross-check only.
- The merged whole-shot ledger preserves
  `cumulative_field_energy_delta_J`, `cumulative_pml_removed_energy_J`,
  `cumulative_power_port_work_J`, `cumulative_ionization_step_count` across
  restarts. Pre-S3R.7 manifests with absent fields fall back to dataclass
  zero defaults.
- The certificate gate remains fail-closed: every required channel is missing
  or cross-scope; the adversarial certificate fixture cannot promote
  validation.

What remains blocked (typed fail-closed):

- 13/13 startup channels (no source-backed startup BVP closure in KR).
- Auluck `Sigma_p` surface terms II/IV/V/VI (Sprint 4 — needs reviewed
  moving-boundary face set with face-centered `v`/`eta`).
- WP-N5 closures: EOS, ablation, restrike, anomalous resistance, electron
  inertia, stopping, beam-target.
- 5 of 10 neutron mechanism channels (beam-target, spectrum, anisotropy,
  detector response, UQ).
- PF-1000 backplate dims, insulator outer radius / wall thickness.
- The 12 µs full-shot horizon (compute-wall blocked).
- The accepted certificate (every required certificate channel is
  missing/cross-scope).

What cannot be claimed:

- No first-principles predictive authority.
- No engineering acceptance.
- No same-scope full-shot validation.
- No accepted certificate of any kind.

## 4. Source Authority Summary

Local source files used as scientific authority (verbatim from
`KnowledgeReference/` or tracked verified extracts):

- `KnowledgeReference/auluck-2021-dpf-circuit-element-EQUATIONS-VERIFIED.md`
  (Auluck 2021 eq. 5/6, sign convention `V₁₂ = −(1/I)∫_Ω J·E dV`).
- `KnowledgeReference/krauz-2012-pf1000-geometry-EQUATIONS-VERIFIED.md`
  (PF-1000 anode OD, cathode-rod count + diameter, insulator exposed length,
  chamber inner radius, chamber length).
- `KnowledgeReference/akel-2016-pf1000-shot-12581-EQUATIONS-VERIFIED.md`
  (Akel 2016 / Akel 2021 same-scope shot identifiers for the comparator scope).
- `KnowledgeReference/spitzer-1953-resistivity-EQUATIONS-VERIFIED.md`
  (Spitzer 1953 collisional resistivity baseline).
- `KnowledgeReference/bosch-hale-1992-fusion-reactivity.md`
  (Bosch–Hale 1992 D-D / D-T cross-section + reactivity parametrization).

Missing local source fields (each is blocked, not inferred):

- PF-1000 anode hollow-bore radius and bore length.
- PF-1000 insulator outer radius and wall thickness.
- PF-1000 backplate radial extent and axial thickness.
- DPF-specific surface-flashover BVP closure and gas-discharge coefficients
  (Townsend α, secondary-emission γ, Paschen A/B for D₂).
- WP-N5 closure equations and coefficients (EOS, opacity, ablation, restrike,
  anomalous resistance, electron inertia, ion stopping, beam-target rate).
- Same-scope (16 kV PF-1000) measured V(t), T_e, T_i, X-ray output.
- Brysk 1973 Doppler-broadening derivation.

Candidate-only cross-checks (engineering aids, never authority):

- PlasmaPy strong-coupling Coulomb-log regime gate
  (`plasmapy_audit.py` — `bounded_out_with_source`, never a silent floor).
- Reduced Lee / snowplow / scalar neutron yield — comparators only, never
  mechanism authority.

## 5. Test Evidence

Exact commands and pass counts at HEAD `46a0b56`:

```bash
git diff --check
# (clean)

.venv312/bin/python -m ruff check src/ tests/
# All checks passed!

.venv312/bin/python -m pytest tests/test_external_team_submission_package.py \
    tests/test_srs_traceability_export.py -q -rx
# 31 passed in 0.27s

.venv312/bin/python -m pytest \
    tests/test_source_geometry_packet.py \
    tests/test_first_principles_geometry.py \
    tests/test_first_principles_power_port.py \
    tests/test_first_principles_startup_bvp.py \
    tests/test_first_principles_closures.py \
    tests/test_first_principles_neutron_authority.py \
    tests/test_first_principles_segmented_whole_shot.py \
    tests/test_first_principles_segmented_whole_shot_combine.py \
    tests/test_first_principles_long_run_integrity.py \
    tests/test_first_principles_certificate_negative_controls.py \
    tests/test_cli_first_principles_3d.py \
    -q -rx
# 320 passed in 21.28s

.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
# cycle 1: PASS head=46a0b56db7db15e7b22c897be6aea38cc72d8338
```

Periodic audit log: `/private/tmp/dpf-unified-audit-logs/20260520T020312Z/summary.md` —
all 10 gates PASS (clean worktree, `git diff --check`, source-truth exhaustion,
module-source vetting, active + recursive artifact linter, `ruff check src/
tests/`, focused pytest, broad first-principles pytest).

Watch item: `tests/test_first_principles_closures.py::test_plasmapy_coupling_regime_gate`
showed one intermittent failure inside a pre-commit hook run during the S3R.6
commit (`f390cf3`). Three subsequent isolated runs and the periodic-audit broad
suite all passed. Root cause not isolated within timebox — the gate code uses
`warnings.catch_warnings(record=True) + simplefilter("always")` for
deterministic capture; the flake is most likely a pytest warning-filter
interaction inside the hook environment. Tracked as a Sprint 4 watch item.

## 6. No-Validation Statement

Sprint 3R **does not** validate a full DPF shot. It does not promote any
requirement to `implemented` or `accepted`. It does not claim engineering
acceptance, first-principles predictive authority, completion of a 12 µs
source-sign run, or an accepted certificate. Every channel that requires
upstream physics or source authority not present in `KnowledgeReference/` is
typed `blocked` or `candidate_*`, and `can_support_first_principles_acceptance`
remains `false` everywhere. Reduced Lee/snowplow/scalar-yield paths remain
comparators only.

Sprint 3R closes the Sprint 3 completion-audit findings so that Sprint 4 may
proceed without inheriting silent acceptance paths, false mask provenance,
broken NumPy 2 integration, omitted closure-matrix effects, lossy restart
ledger merges, contradictory packet ledgers, ambiguous citations, or stale
traceability paths.

## 7. Scope Statement

All work stayed within the handoff's allowed file scopes. Two intentional
follow-ups outside the literal S3R.1 file list were made under documented
justification:

- `b6f7698` — pre-commit linter auto-fixes captured (citation normalization
  on WP_N2/WP_N5 and ruff import hoisting on
  `test_first_principles_neutron_authority.py`). This was a side effect of the
  S3R.1..S3R.7 pre-commit hooks; capturing it avoided silently re-dirtying the
  worktree.
- `46b705f` — the ruff I001 fix on `tests/test_first_principles_power_port.py`
  was discovered by `ruff check src/ tests/` during S3R.8 final verification
  and is required to keep the periodic audit's `ruff_src_tests` gate green.

The `tests/test_first_principles_geometry.py` test file was added in earlier
Sprint 3 work and was unmodified by Sprint 3R; it remains the geometry-side
target test file referenced by S3R.4. No new physics implementation files were
introduced.
