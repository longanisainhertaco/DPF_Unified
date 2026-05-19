# Three-Sprint Final Summary

## Status

The Codex Sprint 1 audit accepted Sprint 1 engineering progress. Sprint 1.1
hygiene (RC-1 through RC-7) is complete. Sprint 2 proposals (WP-N1B,
WP-N4B) are delivered and Sprint 2 implementation has fail-closed runtime code
for WP-N1B/WP-N4B. Sprint 3 remains
pending per the audit's submission gating.

## What this submission is

A package-native 3-D first-principles engineering candidate with fail-closed
source-truth, artifact, power-port, and segmented-run controls.

## What this submission is NOT

It is not an accepted first-principles simulation, not a validated PF-1000/Akel
prediction, not accepted power-port authority, not a solved startup BVP, not
reviewed PF-1000 geometry, not accepted physics closures, and not a completed
12 us source-sign whole-shot run. No artifact, manifest, requirement, or
document promotes anything to accepted or validated status;
`can_support_first_principles_acceptance` is `false` everywhere.

## Sprint 1 — Control Gate Hardening (accepted)

The Codex Sprint 1 audit
(`docs/FIRST_PRINCIPLES_CODEX_AUDIT_SPRINT1_CONTROL_GATE_2026_05_19.md`) verdict:
`accept_sprint_1_engineering_progress_request_changes_before_sprint_2`. All seven
control-gate blockers A-1 through A-7 are closed:

- A-1: run-manifest provenance fails closed on empty `source_packet_hashes`;
  artifact-linter C7 re-derives provenance independently.
- A-2: three stale, dirty-worktree audit artifacts quarantined; no active
  result artifact predates HEAD.
- A-3: the source-truth and module-vetting gates are deterministic and expose a
  read-only `--check` mode.
- A-4: CI exercises archive-exemption policy via a recursive artifact scan.
- A-5: the audit-named ruff slice is clean.
- A-6: a resumed segmented run reconstructs cumulative ledgers over the full
  executed horizon.
- A-7: `DPF-PHYS-020` and `DPF-PHYS-023` moved to `partial` with explicit
  acceptance blockers; RTM regenerated; Sprint 0 closed-with-debt.

Verification: broad first-principles/hybrid suite 275 passed at Sprint 1 baseline; ruff slice clean;
artifact linter 0 failed on both active and recursive scans; read-only gates
exit 0 with no worktree writes; worktree clean. Full transcript in
`AUDIT_COMMANDS.md`.

## Sprint 1.1 — Hygiene (closed)

Seven RC items gating Sprint 2 acceptance, all closed:

- RC-2: repository-wide `ruff check src/ tests/` clean (69 errors resolved).
- RC-3: submission CSV files parse with stable row widths;
  `tests/test_external_team_submission_package.py` added.
- RC-4: current-HEAD transcript block appended to `AUDIT_COMMANDS.md`;
  `BLOCKER_MATRIX.csv` RC rows finalized.
- RC-5: artifact linter check C8 rejects dirty-worktree artifacts and validates
  `artifact_generation_commit` against live HEAD.
- RC-6: positive current-HEAD fixture proves linter C1–C8 on a real PASS
  artifact.
- RC-7: C7 required-provenance tuple has a drift test pinned to
  `manifest.REQUIRED_PROVENANCE_FIELDS`.
- RC-1: Sprint 2 proposal docs delivered (see Sprint 2 below).

Verification: 75 focused package/control tests passed; 283 broad tests passed;
ruff clean; artifact linter 0 failed on both scans; worktree clean.

## Sprint 2 — WP-N1B / WP-N4B (implementation delivered as fail-closed runtime code)

Six proposal and source-status documents delivered under `sprint_2/`. Implementation
commits `4b080eb` (WP-N1B) and `4c8dac1` (WP-N4B) followed:

- **WP-N1B six-term Auluck ledger contract (commits `4b080eb`, `65c477f`).** The Auluck eq. (6)
  six-term power-port ledger is implemented as fail-closed runtime code in
  `src/dpf/first_principles/power_port.py`. The "electrode/interface work" term
  is absent — it was a category error (Auluck excludes the electrode interface
  from the integration domain Omega). Terms I/III are now independently computed
  from runtime split telemetry (`stored_magnetic_energy_delta_J` and
  `stored_electric_energy_delta_J`); terms II/IV/V/VI fail closed pending the
  reviewed Sigma_p moving-boundary geometry from WP-N3. The residual remains
  blocked until all six terms are independently computed. 27 new tests pass,
  with follow-up coverage for the split-telemetry path.
- **WP-N4B cross-restart combiner (commit `4c8dac1` plus Sprint 2.2 follow-up).** `merge_cumulative_ledgers()`
  and `combine_whole_run_artifacts()` are implemented in
  `src/dpf/first_principles/segmented_whole_shot_combine.py` with fail-closed
  gap/overlap/missing-manifest/empty-input/malformed-step checks. The combiner
  treats `total_steps_completed` as the cumulative terminal step and has
  synthetic plus live three-restart coverage. 12 focused tests pass. The 12 us
  compute-wall, production-grid wall-clock, and long-run restart evidence remain
  blocked.

No physics blocker is closed; no acceptance is claimed. `can_support_first_principles_acceptance`
is `false` for all Sprint 2 code and follow-up split-telemetry work.

Key verified finding (retained): the audit's WP-N1B "electrode/interface work" term is a
category error — Auluck eq. (6) is a six-term balance (stored magnetic, motional
magnetic, stored electric, motional electric, resistive, anomalous) with no
electrode-contact-work term; Auluck excludes the electrode interface from the
domain. This verdict is now encoded in the implementation.

## Sprint 3 — physics blockers (pending)

WP-N2 startup BVP, WP-N3 PF-1000 geometry/material masks, WP-N5 closure
registry, WP-N6 neutron mechanism authority, WP-N7 comparator/UQ/certificate,
and numerical acceptance. Deferred until WP-N1B implementation produces a
reviewable runtime artifact. See `sprint_3/PENDING.md`.

## Remaining physics blockers

Every physics blocker P-1 through P-5 from the audit remains open and is carried
in `BLOCKER_MATRIX.csv`. Sprint 1 changed no physics; Sprint 2 delivers
fail-closed engineering candidates only. The control plane is now hardened
enough for the physics work in Sprint 3 to be reviewable.
