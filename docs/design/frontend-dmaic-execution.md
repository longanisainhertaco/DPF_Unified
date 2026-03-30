# Frontend Redesign: DMAIC Execution Plan

## Sprint Schedule

| Sprint | Deliverables | Tests | Est. Hours |
|--------|-------------|-------|------------|
| 1 | app_shot_acceptance.py (250 LOC) + app_preshot_calc.py (180 LOC) | 37 | 3-4 |
| 2 | app_anomaly.py (220 LOC) + app_shot_log.py (200 LOC) | 33 | 3-4 |
| 3 | app_report.py (250 LOC) | 10 | 1-2 |
| 4 | app_engineer.py (350 LOC) + app.py refactor (100 LOC) | 20 | 4-5 |
| **Total** | **6 new + 1 modified + 6 test files** | **100** | **11-15** |

## Critical Path
app_shot_acceptance → app_anomaly + app_shot_log → app_report → app_engineer → app.py
app_preshot_calc is independent (parallel with any sprint)

## Sprint 1 Entry: No Prerequisites
## Sprint 1 Exit: 37 tests pass, zero stubs, ruff clean

## Sprint 2 Entry: Sprint 1 passes
## Sprint 2 Exit: 33 tests pass, SQLite CRUD works, anomaly detection verified

## Sprint 3 Entry: Sprint 2 passes
## Sprint 3 Exit: 10 tests pass, PDF generates valid file

## Sprint 4 Entry: Sprints 1-3 pass
## Sprint 4 Exit: 20 tests pass, app.py launches with 3 mode tabs, all existing tests still pass

## Zero-Stub Controls
- Grep test: `test_no_stubs_in_all_new_modules` checks all 6 files
- CI gate: grep for NotImplementedError/TODO/FIXME blocks merge
- Every function has a test — no untested code paths

## Gate Thresholds (per-device overrides via ExperimentalDevice uncertainties)
| Gate | PASS | MARGINAL | FAIL |
|------|------|----------|------|
| G1 I_peak | <10% | 10-20% | >20% |
| G2 t_peak | <10% | 10-20% | >20% |
| G3 NRMSE | <8% | 8-15% | >15% |
| G4 Dip | >5% | 2-5% | <2% |
| G5 dI/dt | <15% | 15-30% | >30% |

## Key Reuse Points
- parse_experimental_csv() → CSV handling (app_plots.py)
- validate_against_published() → reference values (app_validation.py)
- nrmse_peak() → waveform comparison (dpf.validation.experimental)
- run_simulation_core() → Lee model runner (app_engine.py)
- assess_quality() → pass/fail pattern (quality_assessment.py)
