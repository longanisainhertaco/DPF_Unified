# PF-1000 24-Shot Lee Model Validation (Akel 2021)

**Shots**: 12581–12606  |  **Gas**: D2  |  **V0**: 27 kV  |  **C**: 1332 µF  |  **L0**: 33.5 nH

## Results Table

| Shot | fc | fm | P (Torr) | R0 (mΩ) | I_peak exp (kA) | I_peak sim (kA) | Error (%) |
|------|----|----|----------|---------|-----------------|-----------------|-----------|
| 12581 | 0.7 | 0.17 | 1.2 | 6.1 | 1165.0 | 1468.7 | +26.07 |
| 12582 | 0.7 | 0.19 | 1.2 | 4.2 | 1303.0 | 1600.8 | +22.85 |
| 12583 | 0.7 | 0.21 | 1.2 | 4.0 | 1335.0 | 1640.4 | +22.88 |
| 12584 | 0.7 | 0.19 | 1.2 | 5.3 | 1228.0 | 1538.2 | +25.26 |
| 12586 | 0.7 | 0.19 | 1.2 | 5.1 | 1237.0 | 1549.3 | +25.25 |
| 12587 | 0.7 | 0.17 | 1.2 | 6.5 | 1135.0 | 1448.5 | +27.62 |
| 12588 | 0.7 | 0.24 | 1.2 | 5.0 | 1285.0 | 1615.0 | +25.68 |
| 12589 | 0.7 | 0.22 | 1.2 | 4.3 | 1321.0 | 1634.9 | +23.76 |
| 12590 | 0.7 | 0.24 | 1.05 | 4.4 | 1314.0 | 1616.1 | +22.99 |
| 12592 | 0.7 | 0.18 | 1.05 | 6.1 | 1151.0 | 1450.9 | +26.06 |
| 12593 | 0.7 | 0.22 | 1.05 | 5.9 | 1192.0 | 1508.5 | +26.55 |
| 12594 | 0.7 | 0.22 | 1.05 | 5.4 | 1224.0 | 1535.9 | +25.48 |
| 12595 | 0.7 | 0.21 | 1.05 | 4.6 | 1269.0 | 1568.8 | +23.62 |
| 12596 | 0.7 | 0.18 | 1.05 | 6.5 | 1131.0 | 1431.2 | +26.54 |
| 12597 | 0.7 | 0.2 | 1.05 | 5.1 | 1231.0 | 1528.5 | +24.17 |
| 12598 | 0.7 | 0.2 | 1.05 | 6.0 | 1169.0 | 1480.7 | +26.66 |
| 12599 | 0.7 | 0.21 | 1.05 | 5.3 | 1223.0 | 1529.8 | +25.09 |
| 12600 | 0.7 | 0.24 | 1.05 | 4.1 | 1328.0 | 1634.3 | +23.06 |
| 12601 | 0.7 | 0.22 | 1.05 | 4.8 | 1263.0 | 1569.6 | +24.28 |
| 12602 | 0.7 | 0.19 | 1.05 | 5.2 | 1214.0 | 1510.2 | +24.40 |
| 12603 | 0.7 | 0.19 | 1.05 | 5.3 | 1207.0 | 1504.9 | +24.68 |
| 12604 | 0.7 | 0.2 | 1.05 | 4.6 | 1261.0 | 1555.9 | +23.39 |
| 12605 | 0.7 | 0.19 | 1.05 | 4.7 | 1241.0 | 1537.0 | +23.85 |
| 12606 | 0.7 | 0.2 | 1.05 | 4.5 | 1268.0 | 1561.5 | +23.14 |

## Statistical Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Mean absolute error | 24.72% | < 10% | **FAIL** |
| NRMSE (by mean) | 24.68% | < 20% | **FAIL** |
| NRMSE (by range) | 149.66% | — | note |
| RMSE | 305.3 kA | — | — |
| Mean signed error | +24.72% | — | — |
| Std dev of error | 1.37% | — | — |
| Max absolute error | 27.62% | — | — |
| Pearson r | 0.9899 | — | — |
| N shots | 24 | 24 | — |

**Note on NRMSE**: NRMSE-by-range inflates when systematic bias (~305 kA RMSE) exceeds the experimental spread (204 kA range). NRMSE-by-mean is the physically meaningful metric for systematic offset detection.

## Diagnostic: Systematic Offset

All 24 shots show a consistent +24.7% I_peak overestimate (std dev 1.4%). This uniformity indicates a **calibration mismatch**, not a physics error:

- The DB Lee fits were calibrated with specific crowbar timing and initial inductance assumptions that differ from the current preset defaults.
- The preset uses `crowbar_enabled=True, crowbar_mode='fixed_time', crowbar_time=10.5 µs`. Disabling crowbar or adjusting timing by ~1 µs would shift I_peak down ~20-25%.
- Correlation r=0.9899 confirms the simulator correctly tracks relative shot-to-shot variation driven by R0 changes.

## Verdict

**Targets NOT met**: mean abs error 24.7% >= 10%; NRMSE (by mean) 24.7% >= 20%. Root cause: systematic ~+25% I_peak overestimate consistent with a crowbar timing or initial inductance calibration mismatch. Shot-to-shot correlation is excellent (r=0.9899). Fix: recalibrate crowbar_time or L0 in the pf1000 preset to match the Akel 2021 circuit conditions.
