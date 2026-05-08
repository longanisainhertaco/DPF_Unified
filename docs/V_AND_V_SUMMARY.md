# Verification and Validation Summary

Executive summary for reviewers. Updated 2026-05-05.

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

**39/40 verification requirements verified.** See `docs/VV_PLAN.md` for the historical traceability matrix.

## Validation (Does the code match experiment?)

### Circuit-Level (Lee Snowplow Model) — SOURCE-GATED, PARTIAL

Under the current KnowledgeReference-only rule, a device can support tier-1
circuit evidence only when the device parameters and waveform trace are both
KR-verified, the waveform is measured, and the record reliability is measured.

| Device | Source Status | Reason |
|--------|---------------|--------|
| PF-1000 | validation-ready | KR-verified measured waveform from Scholz record |
| POSEIDON-60kV | waveform KR-unverified | parameters are KR-supported, waveform is an external archive trace |
| UNU-ICTP | waveform KR-unverified | parameters are KR-supported, waveform is an external archive trace |
| PF-1000-Gribkov | KR-unverified | measured waveform from external archive, source not yet in KR |
| PF-1000-16kV | reconstructed-only | waveform reconstructed from 27 kV trace |
| FAETON-I | reconstructed-only | waveform reconstructed from circuit parameters |
| MJOLNIR | reconstructed-only | waveform reconstructed phenomenologically |
| NX2 | reference-only | no measured waveform available |

Current source status: **1/9 registered devices validation-ready**. A passing
PF-1000 waveform comparison is tier-1 circuit evidence only; it is not spatial
MHD validation and does not validate neutron production.

### MHD-Level (Spatially-Resolved Solver) — VERIFIED, NOT YET VALIDATED

The MHD solver passes numerical verification tests (Sod, Brio-Wu, convergence)
but has **not yet been compared to spatially-resolved experimental measurements**.

Target comparison paths exist for source-backed spatial components, but a
same-scope density/B-field/temperature validation bundle has not been produced.

See [SCOPE.md](SCOPE.md) for regime of validity and known limitations.

## Known Limitations

1. **Tier-1 is narrow**: PF-1000 waveform comparison can support circuit evidence only.
2. **Most device waveforms are blocked**: reconstructed, reference-only, and external archive traces cannot support validation claims by default.
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
