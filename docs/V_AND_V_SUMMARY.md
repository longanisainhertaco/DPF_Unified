# Verification and Validation Summary

Executive summary for reviewers. Updated 2026-04-08.

## Verification (Does the code solve the equations correctly?)

| Test | Method | Result | Reference |
|------|--------|--------|-----------|
| Sod shock tube | L1(rho) at N=256 | < 0.02 | Analytic solution |
| Brio-Wu MHD shock | No NaN, float32 | PASS | Brio & Wu 1988 |
| Deterministic reproducibility | 50 identical runs | std = 2.2e-16 | Machine epsilon |
| Mass conservation | Full discharge | < 5% relative | — |
| Energy conservation | Full discharge | < 10% relative | — |
| B_theta propagation | Alfven speed check | Correct | Electrode BC injection |
| WENO5-Z weights | Borges smoothness | Correct | Borges et al. 2008 |
| HLLS entropy flux | Popovas criterion | Correct | Popovas et al. 2025 |
| Constrained transport | div(B) magnitude | Controlled | Evans & Hawley 1988 |
| SSP-RK3 stability | TVD preservation | PASS | Shu & Osher 1988 |
| Spitzer resistivity | NRL Formulary | PASS | Braginskii 1965 |
| Bremsstrahlung coefficient | NRL Formulary p.58 | 1.42e-40 W m^3 | Rybicki & Lightman 1979 |

**39/40 V&V requirements verified.** See `docs/VV_PLAN.md` for full traceability matrix.

## Validation (Does the code match experiment?)

### Circuit-Level (Lee Snowplow Model) — VALIDATED

| Device | I_peak Error | t_peak Error | NRMSE | Reference |
|--------|-------------|-------------|-------|-----------|
| PF-1000 (27 kV) | **2.8%** | 13.6% | 0.182 | Scholz 2006, Gribkov 2007 |
| PF-1000 (24 shots) | **1.27% MAE** | — | 0.015 | Akel et al. 2021 |
| POSEIDON (60 kV) | 0.5% | 0.8% | 0.114 | IPFS archive |
| FAETON-I (100 kV) | 3.6% | 4.6% | 0.026 | Damideh et al. 2025 |
| MJOLNIR (60 kV) | 10.5% | 11.6% | 0.162 | Offermann et al. 2021 |
| UNU-ICTP (13.5 kV) | 6.5% | 9.0% | 0.089 | IPFS archive |
| NX2 | **EXCLUDED** | — | — | Not experimental data |

**6/7 devices PASS.** Zero calibration — published RADPF parameters only.

### MHD-Level (Spatially-Resolved Solver) — VERIFIED, NOT YET VALIDATED

The MHD solver passes numerical verification tests (Sod, Brio-Wu, convergence)
but has **not yet been compared to spatially-resolved experimental measurements**.

Target: Malir et al. (2024, Phys. Plasmas) density profiles from laser
interferometry. Comparison infrastructure built (`src/dpf/validation/spatial_comparison.py`).

See [SCOPE.md](SCOPE.md) for regime of validity and known limitations.

## Known Limitations

1. **t_peak structural error**: 10-14% (snowplow propagation speed). See `TIMING_ERROR_RCA.md`.
2. **POSEIDON geometry**: 14.7% error at 40 kV (electrode gap effect).
3. **Ideal EOS**: gamma = 5/3. Saha EOS available (`enable_saha_eos=True`) but not default.
4. **No impurity radiation**: See `IMPURITY_LIMITATIONS.md` for bounding estimate.
5. **2D axisymmetric**: No m=1 kink instabilities. See `SCOPE.md`.
6. **Backend parity**: Not all physics available on all backends. See `BACKEND_PARITY.md`.

## Test Infrastructure

| Metric | Value |
|--------|-------|
| Total tests | ~4,100 |
| CI gate | >= 4,000 non-slow |
| Validation tests | 145 (test_validation_ci.py) |
| Saha EOS tests | 11 (test_saha_eos.py) |
| Spatial validation tests | 7 (test_spatial_validation.py) |
| MLX solver tests | 562 (test_mlx_*.py) |
| xfail tests | 18 (triaged, documented) |

## Document Map

| Document | What It Covers |
|----------|---------------|
| [SCOPE.md](SCOPE.md) | Claims, regime of validity, error budget |
| [BACKEND_PARITY.md](BACKEND_PARITY.md) | Physics feature matrix by backend |
| [IMPURITY_LIMITATIONS.md](IMPURITY_LIMITATIONS.md) | Impurity physics gap analysis |
| [VV_PLAN.md](VV_PLAN.md) | 40-item verification traceability matrix |
| [TIMING_ERROR_RCA.md](TIMING_ERROR_RCA.md) | Six Sigma root cause analysis on t_peak |
