# Post-Conflict-Fix Validation Baseline — 2026-05-04

**Captured:** 2026-05-04
**Branch:** `fix/mjolnir-l0-petrov-2022` @ `4e6b8cd` (1 commit ahead of `origin/main` `92a6316`)
**Tool:** `cortana-dpf-validate`
**Purpose:** Pre-Wave-7-fix snapshot after `experimental_waveforms.py` conflict resolution; reference state before O1 MJOLNIR L0, O2 back_emf, S1 UNU V0, S2 POSEIDON-60kV t_peak land.
**Conflict status:** No merge-conflict markers present in `src/dpf/validation/experimental_waveforms.py` (182 lines, clean).
**Note:** Branch already carries the MJOLNIR L0 = 46.7 nH / R0 = 6.3 mOhm Petrov-2022 re-anchor (commit `4e6b8cd`).

---

## Per-Device Metrics

| Device         | I_peak dev% | t_peak dev% | Yn log-ratio | NRMSE | Status |
|----------------|------------:|------------:|-------------:|------:|--------|
| PF-1000        |      +5.0%  |     +28.1%  |       +0.58  | 0.223 | PASS   |
| NX2            |     +27.0%  |      +4.9%  |       +0.39  |   N/A | FAIL   |
| UNU-ICTP       |     +12.8%  |     +29.7%  |       +1.00  | 0.067 | FAIL   |
| poseidon       |       ERROR |        --   |          --  |    -- | ERROR  |
| POSEIDON-60kV  |     +15.4%  |     +70.5%  |       +0.40  | 0.115 | FAIL   |
| MJOLNIR        |     +18.8%  |     +23.8%  |       +2.00  | 0.220 | FAIL   |
| FAETON-I       |     +60.4%  |     +73.5%  |       +1.76  | 0.077 | FAIL   |

**Overall:** 1/7 PASS (thresholds: I_peak < 15%, Yn < 1 decade)

---

## Delta vs Pre-Merge Baseline (PR_B_PRE_MERGE_BASELINE.md, 2026-04-30)

S15 (`POST_MERGE_BASELINE_2026_05_01.md`) is not on disk; closest reference is `docs/PR_B_PRE_MERGE_BASELINE.md`.

| Device         | I_peak Δ pp | t_peak Δ pp | Yn Δ dec | Status Δ           |
|----------------|------------:|------------:|---------:|--------------------|
| PF-1000        |       +1.2  |       +8.5  |    -0.61 | FAIL → **PASS**    |
| NX2            |        0.0  |        0.0  |     0.00 | FAIL → FAIL        |
| UNU-ICTP       |       -0.8  |      +10.9  |    +0.01 | PASS → **FAIL**    |
| poseidon       |          -- |          -- |       -- | ERROR → ERROR      |
| POSEIDON-60kV  |        0.0  |        0.0  |    -1.11 | FAIL → FAIL        |
| MJOLNIR        |      +11.3  |      +12.7  |    +1.75 | PASS → **FAIL**    |
| FAETON-I       |        0.0  |        0.0  |     0.00 | FAIL → FAIL        |

### Recoveries / Regressions

- **No device recovered from ERROR.** `poseidon` remains ERROR ("no validation data registered") — the conflict resolution did not introduce poseidon waveform data. This is a missing-fixture issue, not a conflict artifact.
- **PF-1000 → PASS:** Yn improved from +1.19 → +0.58 dec (under 1-decade gate). I_peak +5.0% well under 15% gate.
- **MJOLNIR → FAIL:** Expected regression — the L0/R0 Petrov-2022 re-anchor (commit `4e6b8cd`) raises I_peak deviation from +7.5% → +18.8% and Yn from +0.25 → +2.00 dec. Per `papers-are-truth`, this is a real solver gap, not a parameter knob to retune.
- **UNU-ICTP → FAIL:** Yn slid from +0.99 → +1.00 dec (now exactly at gate); t_peak grew +18.8% → +29.7%. Targeted by Wave-7 S1 (UNU V0).
- **POSEIDON-60kV:** I_peak/t_peak unchanged; Yn improved -1.11 dec but still FAIL on t_peak +70.5% (Wave-7 S2 target).

---

## CI Gate Status

```
python3 -m pytest tests/test_validation_ci.py --no-header --tb=line -q
....................x..x....                                             [100%]
26 passed, 2 xfailed in 3.79s
```

Gates: **26 passed, 2 xfailed, 0 failed** — CI green on this state.

---

## Logs

- Validation log: `/tmp/wave7_baseline_validate.log`

## VERDICT

**Pre-Wave-7-fix baseline established.** Mixed delta vs pre-merge:
- PF-1000 recovered to PASS (Yn improved).
- MJOLNIR regressed to FAIL as expected from the published-parameter re-anchor — Wave-7 O1 owns this.
- UNU-ICTP, POSEIDON-60kV, NX2, FAETON-I unchanged or marginal — Wave-7 S1/S2 targets remain valid.
- poseidon ERROR persists — separate fixture-missing issue, unrelated to the conflict fix.

CI gates pass. Snapshot ready as the comparison anchor for Wave-7 O1/O2/S1/S2 landings.
