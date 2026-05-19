# Sprint 3 Final Submission

- Date: 2026-05-19
- Branch: `codex/corpus`
- Controlling contract: `docs/FIRST_PRINCIPLES_SPRINT3_COMPLETION_HANDOFF_2026_05_19.md`
- Audit basis: `docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT3_PACKET_2026_05_19.md`
- Implementation HEAD: `e9b3c20` — this final-submission commit is the
  documentation-only wrapper on top of it.

## 1. Summary of changed code and docs

Sprint 3 converted the seven Sprint 3 research/spec packets into source-tagged,
fail-closed runtime foundations (handoff S3.1–S3.9). Every package is
fail-closed: where a local `KnowledgeReference/` source or tracked verified
extract supports a value it is cited with exact path and line range; where no
local source exists, a typed blocker is emitted. No physics is validated and no
requirement is promoted to `implemented` or `accepted`.

- **S3.1** packet hygiene — `sprint_3/PENDING.md` → `SPRINT_3_STATUS_LEDGER.md`;
  stale Sprint-2.2/WP-N2/WP-N5 language and shorthand citations fixed;
  consistency tests added.
- **S3.2/S3.3** — `PF1000GeometryPacket` / `PF1000GeometryField` /
  `PF1000GeometryConflict` / `PF1000MaskManifest` with three source-tagged
  constructors and 10 material masks with per-mask SHA-256; `SigmaPSurfacePacket`
  plumbing consumed by `power_port.py`. Auluck terms II/IV/V/VI stay fail-closed
  (the surface integrals are Sprint 4); terms I/III stay independently computed.
- **S3.4** — typed `StartupPacket` covering 13 startup channels; startup
  authority is fail-closed (0 source-supported channels per WP-N2).
- **S3.5** — closure registry (12 closures + 2 sub-closures), each classified;
  PlasmaPy strong-coupling Coulomb-log regime gate (`bounded_out_with_source`,
  never a silent floor).
- **S3.6** — mechanism-separated `NeutronAuthorityRuntime` (10 channels); scalar
  yield is `candidate_comparator_only`, never mechanism authority.
- **S3.7/S3.8** — numerical acceptance gates (extended cumulative ledgers,
  run-manifest SHA-256, 12 us wall-clock blocker reported by manifest) and a
  fail-closed certificate scaffold that emits no accepted certificate.
- **S3.9** — SRS/RTM and packet-ledger traceability.

## 2. Files changed

| Commit | Package | Key files |
| --- | --- | --- |
| `100d87d` | S3.1 | `sprint_3/SPRINT_3_STATUS_LEDGER.md` (+ `PENDING.md` removed), `README.md`, `THREE_SPRINT_FINAL_SUMMARY.md`, 3 `sprint_3/WP_N*` docs, `tests/test_external_team_submission_package.py` |
| `0e91f08` | S3.2/S3.3 | `src/dpf/fields/source_geometry.py`, `src/dpf/first_principles/power_port.py`, `tests/test_source_geometry_packet.py`, `tests/test_first_principles_geometry.py`, `tests/test_first_principles_power_port.py` |
| `06744fd` | S3.4 | `src/dpf/first_principles/startup_bvp.py`, `src/dpf/cli/main.py`, `tests/test_first_principles_startup_bvp.py`, `tests/test_cli_first_principles_3d.py` |
| `7dd1994` | S3.5 | `src/dpf/first_principles/closure_packet.py`, `src/dpf/first_principles/plasmapy_audit.py`, `tests/test_first_principles_closures.py` |
| `d1dc17c` | S3.6 | `src/dpf/first_principles/neutron_authority.py`, `src/dpf/diagnostics/neutron_yield.py`, `beam_target.py`, `neutron_tof.py`, `tests/test_first_principles_neutron_authority.py` |
| `6660eb9` | S3.7/S3.8 | `src/dpf/first_principles/segmented_whole_shot.py`, `source_targets.py`, `same_scope.py`, `tests/test_first_principles_long_run_integrity.py`, `test_first_principles_certificate_negative_controls.py`, `test_first_principles_source_targets.py` |
| `f7bb9f8` | vetting | `docs/FIRST_PRINCIPLES_MODULE_SOURCE_VETTING_2026_05_18.{json,md}` |
| `e9b3c20` | S3.9 | `docs/DPF_REQUIREMENTS_BASELINE.md`, `docs/SRS_TRACEABILITY_MATRIX.{csv,json}`, packet `CHANGELOG.md`, `CLAIMS_LEDGER.csv`, `BLOCKER_MATRIX.csv`, `TEST_MAP.csv` |

## 3. Requirement IDs and new statuses

14 rows updated in `docs/DPF_REQUIREMENTS_BASELINE.md`. **No requirement promoted
to `implemented` or `accepted`.**

| Requirement | Old → New | Note |
| --- | --- | --- |
| DPF-PHYS-014 | blocked → partial | source-tagged geometry packet; bore/insulator/backplate still blocked |
| DPF-PHYS-010 / -017 / -021 | blocked → partial | typed startup packet; 0 source-supported channels |
| DPF-PHYS-018 / -024 | blocked → partial | closure registry + PlasmaPy gate; advanced closures blocked |
| DPF-PHYS-013 / -025 / DPF-VV-010 | blocked → partial | mechanism-separated neutron packet; advanced channels blocked |
| DPF-DATA-007 | implemented → partial | honest correction — small-horizon gates pass, production-horizon blocked |
| DPF-DATA-004 | implemented → partial | honest correction — certificate scaffold exists, no accepted certificate possible |
| DPF-PHYS-026 / DPF-VV-017 | blocked → partial | certificate scaffold; all channels fail-closed |
| DPF-PHYS-022 / -020 / -023 | partial (unchanged) | evidence text updated to cite Sprint 3 commits |

Regenerated RTM `summary.status_counts`: `blocked=4, partial=33, implemented=26,
planned=6` (69 total).

## 4. Tests run

```bash
.venv312/bin/python -m pytest tests/test_first_principles_*.py -q          # 363 passed
.venv312/bin/python -m pytest tests/test_external_team_submission_package.py \
  tests/test_source_geometry_packet.py tests/test_first_principles_geometry.py \
  tests/test_srs_traceability_export.py tests/test_cli_first_principles_3d.py \
  tests/test_hybrid_3d_*.py -q                                             # 129 passed
.venv312/bin/python -m ruff check src/ tests/                              # All checks passed
.venv312/bin/python scripts/run_codex_periodic_audit.py --timeout-seconds 900
```

## 5. Periodic audit

`/private/tmp/dpf-unified-audit-logs/20260519T202645Z/summary.md` — all 10 gates
PASS at HEAD `e9b3c20` (clean worktree, `git diff --check`, source-truth
exhaustion, module-source vetting, active + recursive artifact linter,
`ruff check src/ tests/`, focused pytest, broad first-principles/hybrid pytest).

## 6. Blocker ledger — remaining blockers and why they could not be closed

| Blocker | Why it cannot close from local sources |
| --- | --- |
| PF-1000 anode bore / insulator / backplate / chamber-wall-thickness dimensions | No `KnowledgeReference/` source provides numeric values; emitted as typed `blocked` geometry fields. |
| Auluck power-port terms II/IV/V/VI | Require a reviewed `Sigma_p` moving-boundary face set with face-centered `v`/`eta`; the `SigmaPSurfacePacket` plumbing exists but is fail-closed. The surface integrals are Sprint 4. |
| Startup BVP — all 13 channels | No DPF-specific flashover BVP closure or gas coefficients in the local corpus; every channel is `candidate` or `blocked`. |
| WP-N5 closures (EOS, ablation, restrike, anomalous resistance, electron inertia, stopping, beam-target) | No source equations/coefficients in `KnowledgeReference/`; registered as `active_blocked` / `not_simulated_and_claim_blocking`. |
| Neutron mechanism authority | Ion-distribution, stopping, detector-response, and scatter packets have no same-scope (Akel 16 kV) local source; mechanism authority fail-closed. |
| 12 us whole-shot run | Compute-wall blocked (~120 M steps); manifest reports the blocker rather than hiding it. |
| Certificate acceptance | Every required certificate channel is missing/cross-scope; no accepted certificate is emitted. |
| `tests/test_first_principles_closures.py::test_plasmapy_coupling_regime_gate` | One intermittent failure observed in a pre-commit hook run; 13+ subsequent isolated runs and the periodic-audit broad suite passed. Root cause not isolated within the timebox — flagged as a watch-item; the gate code itself uses `simplefilter("always")` for deterministic capture. |

## 7. Validation and acceptance statement

Validation and full-shot acceptance remain **BLOCKED**. Sprint 3 delivered
fail-closed runtime foundations only. `can_support_first_principles_acceptance`
is `false` everywhere. No PF-1000/Akel validation certificate is emitted; no
12 us source-sign run is claimed; the Auluck power-port closure is not computed;
startup, closure, and neutron authority are all fail-closed pending source
packets. Reduced Lee/snowplow/scalar-yield models remain comparators only.

## 8. Scope statement

All work stayed within the handoff's allowed file scopes, with one justified
exception: S3.4 modified `src/dpf/cli/main.py` (not in S3.4's literal allowed-file
list) so the first-principles CLI reports the startup-packet status — the S3.4
done-criterion "first-principles CLI output reports startup packet status"
requires it. SRS/RTM edits were deferred from S3.2–S3.8 to S3.9 to avoid
concurrent-edit conflict. No other out-of-scope edits were made.
