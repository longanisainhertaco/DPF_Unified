# Pre-Push Hook Audit — Wave 7

**Date:** 2026-04-30  
**Hook path:** `.git/hooks/pre-push` (managed in `.git/`, not project-level)

---

## Hook Content

Three sequential pytest stages, each with `-x -q --tb=line -m "not slow"` (fail-fast, suppress slow-marked tests):

| Stage | Files | Exit condition |
|-------|-------|----------------|
| Infrastructure/physics | `test_infrastructure_consolidated.py`, `test_snowplow_consolidated.py`, `test_two_temperature.py` | `BLOCKED: Infrastructure tests failed.` |
| MLX solver | `test_mlx_kernels.py`, `test_mlx_primitives.py`, `test_mlx_reconstruction.py`, `test_mlx_riemann.py`, `test_mlx_timestepper.py`, `test_mlx_solver.py`, `test_mlx_divb_and_shocks.py` | `BLOCKED: MLX solver tests failed.` |
| Metal + MHD | `test_metal_gpu_consolidated.py`, `test_mhd_solver_consolidated.py` | `BLOCKED: Metal/MHD solver tests failed.` |

All 10 referenced test files exist in `tests/`.  
Hook prints `Pre-push gate: PASS (~2,100 tests)` on success.

stdin is consumed silently (`cat > /dev/null`) — correct behavior.

---

## Test Selection Logic

- **Scope:** explicit file list (no glob, no `tests/` sweep)
- **Marker filter:** `-m "not slow"` — skips `@pytest.mark.slow` tests
- **Execution:** sequential stages (not parallel); `-x` aborts each stage on first failure
- **No coverage flags** — hook runs tests only, no `--cov`

---

## Runtime / Exit Conditions

- Exit 1 on any stage failure; exit 0 on full pass
- No timeout set in hook itself (relies on shell/git timeout)
- Claimed count "~2,100 tests" is a comment, not a validated assertion — no `--co` count check

---

## Drift vs CI (`.github/workflows/ci.yml`)

| Dimension | Pre-push hook | CI (`test` job) |
|-----------|--------------|-----------------|
| Runner | macOS local (MLX/Metal available) | ubuntu-latest (no MLX/Metal) |
| MLX tests | Included | Excluded via `-k "not metal and not Metal and not athena and not mlx"` |
| Metal tests | `test_metal_gpu_consolidated.py` included | `--ignore=tests/test_metal_gpu_consolidated.py` |
| Scope | 10 specific files | All `tests/` minus ignored files |
| Parallelism | None | `-n auto` (pytest-xdist) |
| Coverage | None | `--cov=dpf --cov-report=term-missing` |
| Count guard | None (comment only) | Hard floor: `< 2500` exits 1 |
| Python matrix | System python3 | 3.10 / 3.11 / 3.12 |
| Validation gate | Not run | Separate `validation` job (`test_validation_ci.py`) |
| Smoke tests | Not run | Separate `smoke-test` job (CLI + server) |
| Lint | Not run | Separate `lint` job (ruff) |

**Structural drift:** The hook runs GPU-backend tests (MLX + Metal) that CI *cannot* run on ubuntu runners. Conversely, CI runs the full test suite width (minus GPU files) while the hook targets only 10 files. The two test sets are **intentionally different**, not misaligned by accident — hook compensates for CI's inability to exercise MLX/Metal.

---

## Blockers / Findings

1. **No count assertion in hook.** The "~2,100 tests" claim is a comment. If test collection silently drops files (import error, missing dep), the hook can pass with far fewer tests. CI has a `< 2500` floor guard; the hook has none.

2. **No lint or ruff stage.** A push can fail CI lint after passing the hook. Adding `ruff check src/ tests/` as stage 0 would close this gap at near-zero runtime cost.

3. **No validation gate.** `test_validation_ci.py` is a separate CI job not exercised by the hook. A bad coupling-function change can push and fail CI's validation job without local warning.

4. **No timeout guard.** If Metal GPU hangs (e.g., driver issue), the push blocks indefinitely. `timeout 300 python3 -m pytest ...` on the Metal stage would cap exposure.

5. **MLX/Metal stages unavailable on non-macOS dev machines.** If a contributor pushes from Linux, the hook will fail at stage 2 (no MLX). Hook has no platform guard — `[[ $(uname) == "Darwin" ]] || skip_gpu_stages` would make it portable.
