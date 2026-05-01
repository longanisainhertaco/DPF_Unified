# PR-B Pre-Merge Baseline — cortana-dpf-validate Snapshot

**Captured:** 2026-04-30
**Branch:** pre-merge (fix/audit-cleanup-apr23 open, not yet merged to main)
**Tool:** `cortana-dpf-validate`
**Purpose:** Reference state for post-merge regression comparison

---

## Per-Device Metrics

| Device | I_peak dev% | t_peak dev% | Yn log-ratio | NRMSE | Status |
|---|---|---|---|---|---|
| PF-1000 | +3.8% | +19.6% | +1.19 | 0.223 | FAIL |
| NX2 | +27.0% | +4.9% | +0.39 | N/A | FAIL |
| UNU-ICTP | +13.6% | +18.8% | +0.99 | 0.067 | PASS |
| poseidon | ERROR | — | — | — | ERROR |
| POSEIDON-60kV | +15.4% | +70.5% | +1.51 | 0.115 | FAIL |
| MJOLNIR | +7.5% | +11.1% | +0.25 | 0.160 | PASS |
| FAETON-I | +60.4% | +73.5% | +1.76 | 0.077 | FAIL |

**Overall:** 2/7 pass (thresholds: I_peak < 15%, Yn < 1 decade)

---

## PF-1000 vs CRITICAL_BLOCKER Comparison

CRITICAL_BLOCKER.md was not present at capture time. Reference is PR_B_CHANGELOG.md §Known Residuals.

| Metric | Documented Expected | Observed (this snapshot) | Delta | Assessment |
|---|---|---|---|---|
| I_peak error | ~7.6% (post-knob-removal residual) | +3.8% | -3.8 pp | BETTER than documented residual |
| t_peak error | not specified | +19.6% | — | NO GATE in changelog |
| Yn log-ratio | < 1 decade gate | +1.19 decades | +0.19 over threshold | FAIL (exceeds Yn gate) |
| NRMSE | not specified | 0.223 | — | informational |

The changelog documents PF-1000 I_peak at ~7.6% as the known residual post-empirical-knob removal. This snapshot shows 3.8% — within the ≤ 12% gate and better than the documented residual. This may indicate the pre-merge branch is at a different parameter state than the documented residual, or the validation tool uses a different comparison dataset.

PF-1000 FAILS the overall cortana-dpf-validate run due to Yn log-ratio (1.19 decades > 1.0 threshold) and t_peak error (19.6%, no explicit gate but large).

---

## Blockers

1. **PF-1000 Yn > 1 decade** — log-ratio 1.19 exceeds the 1-decade pass threshold. Not gated in PR_B_CHANGELOG validation criteria (only I_peak ≤ 12% is stated), but fails the tool's composite pass logic.

2. **poseidon ERROR** — no validation data registered for the `poseidon` device. Excluded from pass/fail count but signals a missing fixture.

3. **NX2 I_peak +27%** — well above the 15% threshold. PR_B_CHANGELOG does not document this as a known residual; root cause unclear.

4. **FAETON-I I_peak +60%, t_peak +73%** — extreme deviations. Not documented as known residuals.

5. **POSEIDON-60kV t_peak +70.5%** — large timing error. Possibly related to MJOLNIR voltage correction cascading into related devices.

---

## Post-Merge Comparison Instructions

After merging PR-B to main:

```bash
cortana-dpf-validate 2>&1 | tee /tmp/wave5_post_merge_validate.log
diff /tmp/wave5_baseline_validate.log /tmp/wave5_post_merge_validate.log
```

Regression: any device whose I_peak error or Yn ratio increases vs this baseline.
Gate: PF-1000 I_peak must remain ≤ 12% per PR_B_CHANGELOG §Known Residuals.
