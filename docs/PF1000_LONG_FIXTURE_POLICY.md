# PF-1000 Long Fixture Policy

This policy separates PF-1000 MLX scientific gates from engineering endurance
and regression evidence.

## Scientific Gate

- File: `tests/test_mlx_pf1000.py`
- Status: `blocked_by_s1_s2_source_closure`
- Pytest behavior: `xfail(run=False)`
- Reason: S1/S2 waveform and current-dip acceptance still require accepted
  same-scope Akel digitized current traces with uncertainty.

Do not re-enable the long PF-1000 classes as passing scientific gates until the
same-scope S1/S2 source-closure blocker is removed by accepted evidence.

## Endurance And Regression Path

- Files: `tests/test_mlx_pf1000_probe.py`,
  `scripts/run_mlx_pf1000_probe.py`
- Status: `non_scientific`
- Opt-in switch: `DPF_MLX_RUN_ENDURANCE=1`

Endurance runs may show runtime stability, target-time reach, phase traversal,
memory behavior, and cap exhaustion. They do not close scientific validation.
Every endurance run must report:

- target time, or `-1` when no target was requested
- step cap
- final time
- cap-exhaustion status
- MLX memory telemetry, or an explicit unavailable marker
- `scientific_status=non_scientific`
- source status for the S1/S2 blocker

Cap exhaustion is a failure condition for target-time endurance runs. A run that
hits the step cap before reaching the requested target must report
`CAP_EXHAUSTED` rather than silently passing as a partial fixture.
