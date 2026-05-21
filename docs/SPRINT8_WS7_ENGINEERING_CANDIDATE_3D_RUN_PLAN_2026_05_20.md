# Sprint 8 WS7 — Engineering-Candidate 3-D Runtime Probe Run Plan

**Date:** 2026-05-20
**Branch:** codex/corpus
**HEAD:** bd5be3a feat(s8-phaseA): Sprint 8 P0 foundation — ledger repair, channel contract, scope lock

---

## CLASSIFICATION — NOT VALIDATION

**This document and every run produced under this plan are ENGINEERING-CANDIDATE
ARTIFACTS ONLY.**  They do not constitute validation, do not satisfy the
first-principles acceptance protocol, and must not be cited as acceptance
evidence.  The guardrails in
`docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md` §Non-Negotiable-Guardrails
apply without exception:

- `accepted_runtime_claim = false` on all outputs.
- `can_support_first_principles_acceptance = false` on all outputs.
- Candidate telemetry is engineering evidence only.
- Runtime success is not validation.

---

## Selected Scope

| Field | Value |
|-------|-------|
| Label | `pf1000_full_energy_27_to_40_kv` |
| Source | `SELECTED_SCOPE_LABEL` from `dpf.first_principles.runtime_demonstrator_scope` |
| Deck preset | `pf1000_scholz_2001_24rod_full_energy` |
| Function | `pf1000_scholz_2001_24rod_full_energy_deck()` in `src/dpf/first_principles/deck.py` |
| CLI route | `dpf experimental-segmented-whole-shot --deck-preset pf1000_scholz_2001_24rod_full_energy` |
| WS3 status | Engineering candidate; geometry acceptance blocked (five fields; see below) |

---

## Source Hashes (at plan date)

| File | SHA-256 |
|------|---------|
| `src/dpf/first_principles/segmented_whole_shot.py` | `c07d0255d01daec1840540ce6c773ae23e5e73aa5a8247adcff7b48188c3e62b` |
| `src/dpf/first_principles/segmented_whole_shot_combine.py` | `7627058317c7b045cbb6dc213d7ea0cda455fa00b870c35584a3f2b5351170b4` |
| `src/dpf/first_principles/deck.py` | `b1d30a7d8bffe031cfb0e457062df0f6211f1092faea2f4e53ebed20b05a8a0a` |
| `src/dpf/cli/main.py` | `918f2c4cfa2dbaece172ad6daf216c5199fc8423a1155c4a6e633d984ea88a15` |
| `src/dpf/first_principles/runner.py` | `f01346cbb335044f42ab93c7d94c706ead8b7f498ad75a642f4bd49194f83839` |
| `docs/FIRST_PRINCIPLES_SOURCE_TRUTH_INDEX.json` | `261b55df2ef60c63b096a5cf043ce36fd463e116780eb3e1e3ff4da8fefb9514` |

Hashes are over committed-tree bytes at HEAD bd5be3a.  WS7 code changes (this
sprint) are uncommitted at plan time; a reviewer MUST re-hash after commit.

---

## Deck Parameters (Source-Supported Values)

All values consumed verbatim from `PF1000GeometryPacket.scholz_2001_24rod_large_electrode()`.

| Parameter | Value | Source |
|-----------|-------|--------|
| Anode radius | 0.122 m | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Anode length | 0.600 m | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Cathode cage radius | 0.200 m | `[KR: pf-1000-device-a2d6bc15.md:129-154]` |
| Cathode rod count | 24 | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Cathode rod diameter | 0.032 m | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Cathode rod length | 0.600 m | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Insulator exposed length | 0.113 m | `[KR: scholz-2007-pf1000-part2-jphysd.md:191-225]` |
| Insulator outer radius | 0.1145 m | `[KR: pf-1000-device-a2d6bc15.md:129-154]` |
| Capacitance | 1.332 mF | `[KR: pf-1000-device-a2d6bc15.md:129-154]` |
| Circuit inductance | 33.5 nH | `[KR: pf-1000-device-a2d6bc15.md:129-154]` |
| Circuit resistance | 6.1 mΩ | `[KR: pf-1000-device-a2d6bc15.md:129-154]` |
| Voltage (nominal) | 27.0 kV | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |
| Gas | D₂ at 3.5 Torr | `[KR: recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md:90-98]` |

---

## Limiter Ledger

| Channel | Status | Blocker ID | Reason |
|---------|--------|------------|--------|
| `anode_hollow_bore_length_m` | `blocked` | `PF1000-BLK-010` | No same-scope KR source for hollow bore in 24-rod revision |
| `insulator_wall_thickness_m` | `blocked` | `PF1000-BLK-016` | No same-scope KR source for wall thickness |
| `backplate_radial_extent_m` | `blocked` | `PF1000-BLK-017` | No same-scope KR source for backplate radial extent |
| `backplate_axial_thickness_m` | `blocked` | `PF1000-BLK-018` | No same-scope KR source for backplate axial thickness |
| `same_scope_reviewed_geometry_mask` | `blocked` | `PF1000-BLK-WS3-same-scope-reviewed-geometry-mask-no-reviewed-transfer-rule-sprint8` | No reviewed transfer rule exists |
| `anode_inner_radius_m` | `not_claimed` | — | Hollow anode NOT declared; `anode_inner_radius_m=None` in deck |
| `breakdown_model` | `blocked_missing_source` | — | Startup channel missing: listed in `startup.missing_channels` |
| `preionization_state` | `blocked_missing_source` | — | Startup channel missing |
| `surface_flashover_closure` | `blocked_missing_source` | — | Startup channel missing |
| `initial_current_density_distribution` | `blocked_missing_source` | — | Startup channel missing |
| `sheath_liftoff` | `blocked_missing_source` | — | Startup channel missing |
| `same_scope_startup_bvp` | `blocked_missing_source` | — | Startup channel missing |
| `hybrid_pic_3d_readiness` | `blocked` | — | `kinetic_ion_pic_push_deposition` and other capabilities missing (see WS1/channel_state) |
| `same_scope_source_status` | `blocked` | — | Same-scope source packet not available for this scope |

Limiter activations per run are recorded in `run_manifest.json:cumulative_ledgers.limiter_total_activations`.
Limiter steps observed per run are recorded in `run_manifest.json:cumulative_ledgers.limiter_steps_observed`.

---

## Conservation Residual

The segmented runner emits the following per-run conservation channels into `run_manifest.json:cumulative_ledgers`:

| Channel | Description | Units |
|---------|-------------|-------|
| `cumulative_j_dot_e_work_J` | Cumulative resistive J·E work across all segments | J |
| `cumulative_active_port_work_J` | Cumulative circuit→plasma port work | J |
| `cumulative_field_energy_delta_J` | Net change in stored EM field energy (magnetic + electric) | J |
| `cumulative_pml_removed_energy_J` | Energy removed by the PML absorber | J |
| `cumulative_power_port_work_J` | Power-port Poynting-flux term I (separate from J·E) | J |

A simple conservation residual estimate is:

```
energy_residual_J = cumulative_j_dot_e_work_J
                  + cumulative_field_energy_delta_J
                  - cumulative_active_port_work_J
                  - cumulative_pml_removed_energy_J
```

This is an engineering indicator only.  Acceptance-quality conservation
verification requires the six-term Auluck power-port ledger (WS6), which is
blocked pending `sigma_p_face_set` availability.

---

## Power Residual

Power-port telemetry is emitted by `build_engineering_power_port_packet()` (WS6).
The six Auluck terms (stored magnetic energy rate, motional magnetic Σp surface
integral, stored electric energy rate, motional electric Σp surface integral,
resistive Σp surface integral, anomalous/poloidal Σp surface integral) are
available only when `sigma_p_face_set` is exposed.  Current status:

| Power-port term | Status |
|-----------------|--------|
| Stored magnetic energy rate | `blocked_missing_sigma_p` |
| Motional magnetic Σp surface integral | `blocked_missing_sigma_p` |
| Stored electric energy rate | `blocked_missing_sigma_p` |
| Motional electric Σp surface integral | `blocked_missing_sigma_p` |
| Resistive Σp surface integral | `blocked_missing_sigma_p` |
| Anomalous/poloidal Σp surface integral | `blocked_missing_sigma_p` |

The accepted power-port authority remains blocked until all six terms and
residual tolerance are source-backed.  Per WS6 exit criteria, the power-port
telemetry is useful for candidate runtime review only.

---

## Duration Status

A full 12 μs source-sign whole shot requires approximately 1.2 × 10⁸ steps at
dt = 10⁻¹³ s.  This is a **known compute-wall blocker** explicitly recorded in
every run manifest under `blocker_verdicts.B-WPN4-12US-COMPUTE-WALL`.

For the engineering-candidate probe, the recommended short-horizon run
parameters are:

```bash
dpf experimental-segmented-whole-shot \
    --deck-preset pf1000_scholz_2001_24rod_full_energy \
    --segment-steps 2 \
    --explicit-total-steps 6 \
    --run-dir /tmp/ws7_probe_run \
    --no-verify-restart-equivalence \
    -o /tmp/ws7_probe_run/manifest.json
```

Or with dt-policy and auto-step-budget (parity path):

```bash
dpf experimental-segmented-whole-shot \
    --deck-preset pf1000_scholz_2001_24rod_full_energy \
    --dt-policy vacuum-cfl \
    --vacuum-cfl 0.95 \
    --auto-step-budget \
    --max-auto-steps 10 \
    --target-time-s 1e-12 \
    --segment-steps 2 \
    --run-dir /tmp/ws7_probe_cfl \
    --no-verify-restart-equivalence \
    -o /tmp/ws7_probe_cfl/manifest.json
```

Duration request satisfaction is always explicit in the run manifest:

| Field | Location |
|-------|----------|
| `horizon_complete` | `run_manifest.json` top-level |
| `total_steps_completed` vs `planned_total_steps` | `run_manifest.json` top-level |
| Compute-wall blocker | `run_manifest.json:blocker_verdicts.verdicts[id=B-WPN4-12US-COMPUTE-WALL]` |
| Wall-time cap status | `run_manifest.json:blocker_verdicts.verdicts[id=B-WPN4-WALL-TIME-CAP]` |

---

## Restart-Equivalence Evidence

Segmented and uninterrupted deterministic short runs are proven bit-identical
(state fingerprint + tracked observables) by `verify_restart_equivalence=True`
(the default).  Equivalence is reported in `run_manifest.json:restart_equivalence`.

Equivalence is intentionally not asserted on wall-time-truncated partial runs
(the comparison against a full uninterrupted run is undefined for a partial
horizon).

---

## Combine-Whole-Run CLI Route

When a long run is split across multiple OS-level process invocations, the
partial run directories can be merged into one combined whole-run manifest:

```bash
dpf combine-whole-run \
    /path/to/run_dir_part0 \
    /path/to/run_dir_part1 \
    -o /path/to/combined_manifest.json
```

This route wraps `combine_whole_run_artifacts()` from
`src/dpf/first_principles/segmented_whole_shot_combine.py`.  It validates
contiguity, merges cumulative ledgers, and re-indexes segments globally.
The output is labelled `experimental_whole_run_combined_manifest_not_validation`
and `can_support_first_principles_acceptance = false`.

---

## Hybrid PIC 3-D Readiness

`hybrid_pic_3d_readiness` is emitted by every `first-principles-3d` /
`experimental-*` CLI run.  It continues to list missing capabilities until all
evidence is accepted:

```
hybrid_pic_3d_readiness_status: "blocked"
can_support_first_principles_acceptance: false
missing_capabilities: ["kinetic_ion_pic_push_deposition", ...]
```

Acceptance of this channel stays blocked.  Runtime success does not unlock it.

---

## Summary of Active Blockers

| Blocker | Status |
|---------|--------|
| 12 μs compute wall | `blocked` — not attempted; honest verdict in every run manifest |
| Geometry mask acceptance | `blocked` — five fields with explicit IDs |
| Startup BVP (whole-shot) | `blocked` — eight missing channels |
| Same-scope source packet | `blocked` — scope `pf1000_full_energy_27_to_40_kv` lacks reviewed same-scope packet |
| `hybrid_pic_3d_readiness` | `blocked` — missing kinetic ion capabilities |
| Power-port six-term ledger | `blocked` — `sigma_p_face_set` unavailable |
| Accepted runtime claim | `false` — guardrail enforced in all output dicts |

---

## References

- `docs/SPRINT8_SUPER_SPRINT_SOURCE_TO_RUNTIME_INSTRUCTIONS_2026_05_20.md` §Workstream 7
- `src/dpf/first_principles/segmented_whole_shot.py`
- `src/dpf/first_principles/segmented_whole_shot_combine.py`
- `src/dpf/first_principles/deck.py` — `pf1000_scholz_2001_24rod_full_energy_deck()`
- `src/dpf/cli/main.py` — `experimental-segmented-whole-shot`, `combine-whole-run`
- `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md`
- `KnowledgeReference/pf-1000-device-a2d6bc15.md`
