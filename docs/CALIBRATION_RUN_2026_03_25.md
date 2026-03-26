# Calibration Run Record — 2026-03-25/26

**Solver**: MLX HLL+PLM SSP-RK2, 32x64 grid, lee_only handoff mode
**Method**: Optuna TPE, 20 trials per device, fc∈[0.50,0.85], fm∈[0.03,0.25]
**Boris correction**: NOT applied (run started before commit bd02b80)
**HLLS entropy solver**: NOT applied (not yet implemented)

## Completed Results

| Device | fc | fm | J (objective) | Status |
|--------|----|----|---------------|--------|
| UNU-ICTP (14 kV, 169 kA) | 0.574 | 0.176 | 0.0477 | CLEAN — no NaN |
| FAETON-I (100 kV, 1.0 MA) | 0.503 | 0.240 | 0.0502 | CLEAN — no NaN |

## Terminated Run

| Device | Trials Done | CPU Time | NaN Count | Status |
|--------|-------------|----------|-----------|--------|
| POSEIDON-60kV (60 kV, 3.19 MA) | ~4 of 20 | 345 min | 4 | KILLED — float32 HLL cancellation |
| PF-1000-Akel (27 kV, 1.87 MA) | 0 of 20 | — | — | NEVER STARTED — blocked by POSEIDON |

## Why POSEIDON Was Terminated

1. **Float32 NaN in HLL flux**: `SR*FL - SL*FR` cancellation at vacuum interfaces
   where v_A = 1.74×10^10 m/s (58× speed of light). RuntimeWarning at
   mlx_riemann.py:182.

2. **Lax-Friedrichs fallback contamination**: NaN cells use `F_LF = 0.5*(FL+FR) -
   0.5*S_max*(QR-QL)`, which is more diffusive than HLL. The calibrated fc/fm would
   compensate for this artificial diffusion, making them physically meaningless.

3. **80 min/trial**: 34,756× slower than necessary. Boris correction (implemented
   after this run started) reduces vacuum v_A from 1.74×10^10 to 5×10^5, bringing
   trial time to ~2-5 min.

4. **Opportunity cost**: 6-10 more hours for contaminated params vs implementing
   HLLS + Boris and rerunning all devices in ~40-120 min total.

## Lessons Learned

1. **Multi-device calibration must use Boris correction by default.** A solver that
   works for UNU-ICTP (169 kA) produces NaN for POSEIDON (3.19 MA).

2. **Calibration should be device-parallel, not sequential.** The sequential loop
   meant POSEIDON blocked pf1000_akel from ever starting.

3. **Float32 cancellation is current-dependent.** The SR*FL-SL*FR cancellation scales
   with I^2. Any device above ~2 MA at our grid resolution triggers it.

4. **In-memory Optuna studies are fragile.** No checkpoint → no recovery. Future
   calibrations should use SQLite storage for persistence.

## Raw Data

Full output log preserved at: `docs/calibration_run_2026_03_25_raw.log`

## Superseded By

This calibration will be rerun with Boris + HLLS entropy solver after implementation.
Expected results: zero NaN, ~2-5 min/trial, physically meaningful fc/fm.
