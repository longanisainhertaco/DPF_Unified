# PR-B Post-Merge Watch Metrics (24h Oncall)

**Tag:** `pr-b-physics-recalibration-apr23`
**Watch window:** T+0 to T+24h after merge to main
**Primary tool:** `cortana-dpf-validate`

---

## Oncall Checklist

| # | Metric | Command | Pass Gate | Fail Action |
|---|---|---|---|---|
| 1 | Mean I_peak error (all devices) | `cortana-dpf-validate` | ≤ pre-PR-B baseline | Open P1 issue, block next merge |
| 2 | PF-1000 27 kV I_peak error | `cortana-dpf-validate --device PF-1000` | ≤ 12% | Regression — rollback candidate |
| 3 | MJOLNIR pass/fail | `cortana-dpf-validate --device MJOLNIR` | PASS | Verify V0=100kV applied; if fail, rollback |
| 4 | UNU-ICTP I_peak shift | `cortana-dpf-validate --device UNU-ICTP` | Document delta; flag if > 15% | Hold as known residual, log to ARCHITECTURAL_DEBT |
| 5 | Energy conservation (256² cyl) | `pytest tests/verification/ -k cyl` | No regression vs pre-merge | Investigate `/mu_0` fix interaction |
| 6 | Python vs MLX I(t) NRMSE (PF-1000) | `pytest tests/ -k parity` | NRMSE ≤ 1% | Float32 brem floor issue — do not rollback, file bug |
| 7 | CI test count | `pytest --collect-only \| grep "test session"` | ≥ 3990 | Audit deletions; restore if accidental |

---

## Detailed Notes

**#1 — Mean I_peak error baseline**
Pre-PR-B reference: Apr 10 run showed 2.8% on PF-1000 (best). Apr 18 showed 11.5%
(regression from empirical knob removal). Post-PR-B target is stabilization at ≤ 12%
across all active devices with a path to < 5% after RADPF regen.

**#2 — PF-1000 regression watch**
This is the highest-signal metric. R0_mOhm 6.43 → 2.3 removes the empirical cushion.
If I_peak error exceeds 12%, the physics is wrong somewhere — do not re-add the
empirical knob. Run 5 Whys before any parameter touch.

**#3 — MJOLNIR**
First run with V0=100kV (Schmidt 2021 §III.A). Expected: significant I_peak shift vs
prior 60kV runs. A FAIL here means either the geometry pins weren't cascaded (`f89ebb1`)
or the test threshold needs updating for the new operating point.

**#4 — UNU-ICTP**
V0, fill pressure, and I_peak all changed in this PR. Net accuracy shift is unknown
until post-merge run. Document the delta in `docs/calibration_results/` regardless of
pass/fail. Do not gate the merge on this device.

**#5 — Energy conservation (cylindrical)**
The `/mu_0` pressure fix (`3c6d3cf`) is the highest-risk physics change. Verify that
cyl convergence tests pass and energy is not drifting. A new regression guard is in
place (`91675ed`) but run the full verification suite.

**#6 — Python vs MLX parity**
MLX brem rtol was relaxed to 10% (`4706cc7`). If NRMSE degrades beyond 1% on I(t),
the float32 log-space floor is the root cause — this is a known issue, not a rollback
trigger.

**#7 — CI test count**
Zombie module deletion and fixture migration may reduce count by ≤ 10. Anything below
3990 requires audit. A drop > 50 indicates accidental test file deletion.

---

## Rollback

```bash
# Identify merge SHA
git log --oneline main | grep "Merge.*fix/audit-cleanup-apr23"

# Revert the merge commit (keep history clean)
git revert <merge-sha> -m 1 -m "revert: PR-B physics recalibration — regression in <metric>"

# Push and open emergency PR
git push origin main
```

**Tag for reference:** `pr-b-physics-recalibration-apr23`
Tag the merge SHA immediately after it lands:
```bash
git tag pr-b-physics-recalibration-apr23 <merge-sha>
git push origin pr-b-physics-recalibration-apr23
```

---

## Rollback Triggers (any one sufficient)

- PF-1000 I_peak error > 12% AND no root cause within 2h
- Energy conservation failure on cyl 256² (non-trivial drift)
- CI test count drops below 3950 with no explanation
- MJOLNIR fails AND Schmidt 2021 §III.A parameters are confirmed correct in code

**Do NOT rollback for:** UNU-ICTP shift (expected), MLX float32 precision gap,
test count reduction ≤ 10 from zombie module deletion.
