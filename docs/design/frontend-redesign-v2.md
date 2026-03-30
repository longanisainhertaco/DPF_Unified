# DPF-Unified Frontend Redesign v2: Three-Mode Architecture

## Context
The current app.py (1381 LOC, 12 tabs) tries to serve students, engineers, and researchers
in one flat tab bar. This creates cognitive overload for all three users. An engineer
a DPF device operator validated that the operational workflow needs:
CSV upload → pass/fail → anomaly diagnosis → shot log → PDF report — in one page,
not spread across tabs.

## Architecture: Three Modes

```
Landing → [Student Mode] | [Engineer Mode] | [Research Mode]
```

### Student Mode (4 tabs)
Tabs: Physics Narrative, Waveforms, 3D Visualization, Energy Balance
Purpose: Learn DPF physics with guided simulations
Sidebar: Preset selector, V0/pressure sliders, Run button
Hidden: All MHD backends, advanced physics, calibration

### Engineer Mode (single page, 3 sections)
Purpose: Operate and verify real DPF devices
Layout: Top-to-bottom flow in one scrollable page

**Section 1: Pre-Shot (accordion, collapsed by default)**
- Device preset + V0 + pressure + gas inputs
- "Target I_peak" and "Target t_peak" inputs
- [Calculate] → shows recommended settings
- [Generate Ramp Plan] → voltage ramp-up with acceptance windows per step

**Section 2: Shot Evaluation (main area, always visible)**
- [Upload Rogowski CSV] file upload
- [Evaluate] button
- Color-coded PASS/MARGINAL/FAIL banner
- 5 acceptance gates table (I_peak, t_peak, dip, NRMSE, dI/dt)
- Waveform overlay plot (measured vs reference)
- Anomaly detection results (if deviations found)
- Diagnosis text with suggested causes

**Section 3: Shot Log (accordion, auto-expands after evaluation)**
- [Add to Log] button + notes field
- Shot history table with trend sparklines
- [Export CSV] [Export PDF] buttons
- Degradation alert if I_peak trends downward

### Research Mode (6 tabs)
Tabs: Compare Runs, Parameter Sweep, 2D MHD Fields, Electron Heating (2T),
      Phase Portrait, Backend Guide
Purpose: Deep analysis and publication-quality results
Sidebar: Full backend selector (all 9), grid resolution, advanced physics toggles

## New Files (6 modules, ~1,740 LOC total)

| File | LOC | Purpose |
|------|-----|---------|
| app_engineer.py | ~350 | Engineer mode page: layout, section orchestration |
| app_shot_acceptance.py | ~250 | 5-gate acceptance evaluation, diagnosis logic |
| app_preshot_calc.py | ~180 | Inverse parameter calculator, ramp plan generator |
| app_anomaly.py | ~220 | Temporal anomaly detection, cause lookup |
| app_shot_log.py | ~200 | SQLite shot database, trend tracking |
| app_report.py | ~250 | ReportLab PDF generation |

## Modified Files

| File | Changes |
|------|---------|
| app.py | Replace flat tab bar with 3-mode gr.Tabs. Import 6 new modules. ~100 LOC changed. |

## Acceptance Gate Definitions

| Gate | Metric | PASS | MARGINAL | FAIL |
|------|--------|------|----------|------|
| G1 | I_peak vs reference | <10% off | 10-20% | >20% |
| G2 | t_peak vs reference | <10% off | 10-20% | >20% |
| G3 | NRMSE (waveform shape) | <8% | 8-15% | >15% |
| G4 | Current dip depth | >5% | 2-5% | <2% |
| G5 | Rise rate dI/dt | <15% off | 15-30% | >30% |

## Anomaly Detection Algorithm
1. Resample both waveforms to common time grid
2. Compute residual R(t) = I_meas(t) - I_ref(t)
3. Rolling 1-us window standard deviation
4. Flag regions where |R(t)| > sigma_threshold * std(R) for > 0.2 us
5. Classify by temporal position and residual shape
6. Map to cause lookup table (low_peak, late_rise, missing_dip, excess_dip, oscillation)

## Pre-Shot Calculator
- Grid search over (V0, pressure) using Lee model: 100 evals in ~2s
- Find (V0, P) minimizing |I_peak - target|/target + |t_peak - target_t|/target_t
- Compute sensitivity via finite differences on the grid
- Ramp plan: linspace(V_start, V_target, n_steps), run Lee at each, ±15% acceptance windows

## Shot Log Persistence
- SQLite at ~/.dpf-unified/shot_log.db
- HF Spaces: /tmp/dpf_shot_log.db (ephemeral)
- Schema: id, timestamp, preset, V0, pressure, gas, I_peak, t_peak, dip_pct, nrmse, status, notes

## PDF Report (ReportLab)
- One page, landscape, letter size
- Header: device, date, operator, parameters
- Gates table: color-coded PASS/FAIL per gate
- Waveform plot: matplotlib embedded PNG
- Notes field + signature line

## Build Order
1. app_shot_acceptance.py (core gate logic, used by everything else)
2. app_anomaly.py (builds on acceptance infrastructure)
3. app_preshot_calc.py (independent, uses Lee model)
4. app_engineer.py (orchestrates sections 1-3, imports above)
5. app_shot_log.py (persistence layer)
6. app_report.py (output formatting)
7. app.py refactor (three-mode tabs)

## Dependencies
- All modules depend on: app_engine.run_simulation_core, app_plots.parse_experimental_csv
- app_shot_log depends on: app_shot_acceptance.evaluate_shot
- app_report depends on: app_shot_acceptance, reportlab, matplotlib
- No circular dependencies

## Testing
6 test files, ~80-120 tests total:
- test_app_shot_acceptance.py — gate computation, diagnosis
- test_app_anomaly.py — detection on synthetic traces
- test_app_preshot_calc.py — inverse calculation
- test_app_shot_log.py — SQLite CRUD, trends
- test_app_report.py — PDF generation
- test_app_engineer.py — integration
