# Formulary Code Audit - 2026-05-11

This audit used the local source-of-truth corpus only. The primary formulary
source was `KnowledgeReference/plasma-formulary.md`. Additional MHD/circuit
fixes below are separated because they use other local `KnowledgeReference/`
material and standard conservative-MHD identities, not the NRL formulary alone.

This document is an implementation audit, not scientific validation evidence.
Correcting a formula does not validate a DPF run, close Akel S1/S2, or promote
any candidate/draft digitization packet.

## Source Scope

NRL formulary rows used in this pass:

- Plasma parameters, Debye length, Alfven speed, beta, Bohm diffusion,
  transverse Spitzer resistivity, and magnetic pressure:
  `KnowledgeReference/plasma-formulary.md:3128-3168`.
- Coulomb logarithm definitions and electron-electron versus electron-ion
  branch formulas: `KnowledgeReference/plasma-formulary.md:3506-3560`.
- Braginskii heat flux and electron thermal conductivities, including
  `kappa_e_parallel` and `kappa_e_perp = 4.7 n k T /(m omega_ce^2 tau_e)`:
  `KnowledgeReference/plasma-formulary.md:3775-3795`.
- Bennett pinch condition:
  `KnowledgeReference/plasma-formulary.md:4385-4387`.
- Ionization/recombination and radiation formulas, especially Eq. 13, Eq. 17,
  Eq. 30, Eq. 33, and Eq. 34:
  `KnowledgeReference/plasma-formulary.md:4938-5145`.

Additional local-KR/context rows used:

- Poynting theorem / electromagnetic energy flux context:
  `KnowledgeReference/plasma-formulary.md:2402-2412`.
- Lee/RADPF current-factor and circuit-loading context:
  `KnowledgeReference/lee_radpf_theory.md` and
  `KnowledgeReference/a-course-on-plasma-focus-numerical-experiments-s-lee-and-s-h-saw-part-1-basic-course.md`.
- Cylindrical conservative-MHD source-term context already represented in the
  local module/test notes for Skinner/Ostriker cylindrical equations.

## Fixed NRL Formulary Mismatches

| Area | Files | Finding | Fix |
| --- | --- | --- | --- |
| Bremsstrahlung Eq. 30 | `src/dpf/fluid/ionization.py` | `coronal_radiation_power()` treated the NRL coefficient as `erg/cm^3/s`, applied an extra `0.1`, and used `Z_eff^2` for a quasi-neutral single-effective-charge plasma. | Use Eq. 30 as `W/cm^3`, convert to `1.69e-38` for SI with `ne` in `m^-3`, and use the single-species reduction `sum Z^2 N(Z) = Z_eff * Ne`. |
| Recombination radiation Eq. 33 | `src/dpf/radiation/line_radiation.py`, `src/dpf/radiation/improved_radiation.py` | `C_REC = 1.13e-37` did not match the Eq. 33 SI reduction used by the code. | Use `C_REC = 1.69e-38 * sqrt(13.6) = 6.23241205313e-38` and keep line-cooling fits explicitly unknown-provenance. |
| Cyclotron radiation Eq. 34 | `src/dpf/radiation/improved_radiation.py` | Negative supplied `B_mag` values were clamped to zero even though Eq. 34 depends on `B^2`. | Use `abs(B_mag)` before squaring so the power is sign-invariant. |
| Radiative recombination Eq. 13 | `src/dpf/atomic/ionization.py` | The helper used a simplified Seaton-style square-root form and omitted the NRL bracket term. | Implement the full Eq. 13 bracket and keep the stated `Te/Z^2 <= 400 eV` validity note in the docstring. |
| Braginskii perpendicular conductivity | `src/dpf/collision/spitzer.py`, `src/dpf/fluid/anisotropic_conduction.py`, `src/dpf/metal/metal_transport.py` | The generic `kappa_perp = kappa_parallel/(1+x^2)` closure did not preserve the NRL high-field coefficient `4.7`; two direct high-field helpers used `4.66`. | Preserve the unmagnetized limit while forcing the high-field limit to the NRL `4.7` coefficient. |
| Magnetic Reynolds number | `src/dpf/diagnostics/plasma_regime.py` | The diagnostic duplicated an uncorrected classical Spitzer resistivity expression. | Use the centralized corrected `spitzer_resistivity()` implementation. |
| Regime classifier Coulomb log/resistivity | `src/dpf/diagnostics/regime_classifier.py` | The classifier used an electron-electron-like Coulomb-log expression and a separate resistivity derivation for an electron-ion diagnostic. | Use the NRL electron-ion Coulomb-log branches and the corrected diagnostic Spitzer resistivity expression. |
| Pinch mean-free-path Coulomb log | `src/dpf/validation/pinch_physics.py` | The default Coulomb log used the NRL electron-electron expression in an electron-ion mean-free-path helper. | Use the NRL electron-ion branches by default and retain an explicit `ln_lambda` override. |

## Fixed Additional Local-KR / MHD-Circuit Mismatches

| Area | Files | Finding | Fix |
| --- | --- | --- | --- |
| SI conservative MHD energy flux | `src/dpf/fluid/mhd_solver.py`, `src/dpf/fluid/cylindrical_mhd.py` | The magnetic energy-flux term omitted `/mu_0` in SI units. | Use `F_E = (E + p_total)v - B(v dot B)/mu_0`. |
| Cylindrical geometric source sign/form | `src/dpf/geometry/cylindrical.py`, `src/dpf/metal/metal_riemann.py`, `src/dpf/metal/mlx_timestepper.py`, `src/dpf/metal/mlx_riemann.py`, `src/dpf/metal/mlx_sources.py` | Radial source terms missed `p_total` or had hoop-stress/tension signs that made pure toroidal-field hoop stress outward. Some MLX paths also density-multiplied a conserved source a second time. | Use the r-weighted finite-volume source form with inward toroidal hoop stress and treat source arrays as conserved momentum-density sources. |
| Circuit inductive EMF double counting | `src/dpf/circuit/coupler.py` | The coupler returned both `dLp_dt` and `back_emf = I*dLp_dt`, while `RLCSolver` already includes `-I*dLp/dt`. | Clamp `dLp_dt` by equivalent inductive voltage and return `back_emf=0.0` unless a future distinct motional-EMF model is added. |
| Lee axial and radial current-factor loading | `src/dpf/validation/lee_model_comparison.py` | The comparison path omitted Lee current-factor loading from circuit-facing axial terms and did not keep radial `fcr` separate from axial `fc` in radial/frozen circuit terms. | Apply `fc` to axial circuit inductance and `dLp_dt`; apply radial `fcr` to radial inductance, radial `dLp/dt`, radial/reflected force, and frozen/post-crowbar radial inductance. |

## Verified / Left Unchanged

The sweep did not find a formulary mismatch in these areas:

- NRL free-free coefficient surfaces in `src/dpf/radiation/bremsstrahlung.py`,
  `src/dpf/diagnostics/xray_imaging.py`, and the Metal/MLX radiation helpers
  that already use the Eq. 30 coefficient with the correct unit interpretation.
- Debye length, Alfven speed, beta, and simple magnetic-pressure helpers where
  the local code matches the NRL definitions and current tests are
  implementation checks only.
- Saha equilibrium and Bennett-condition helper surfaces, with the caveat that
  these remain analytic diagnostics, not DPF validation by themselves.
- `nu_ee` was not changed in this pass. The local formulary contains multiple
  electron-electron collision/relaxation conventions, while current public
  tests expect `nu_ee = sqrt(2) * nu_ei` for `Z=1`. The API needs an explicit
  convention before this should be edited.

## Logged Blockers / Not Promoted

These modules still cannot be treated as source-closed by the NRL formulary:

- `src/dpf/radiation/line_radiation.py` piecewise line-cooling fits:
  unknown-provenance empirical coefficients remain non-validation evidence.
- `src/dpf/radiation/transport.py` opacity/FLD/Kramers/Rosseland behavior:
  not directly source-closed from visible NRL rows in this pass.
- QMF suppression, p-B11 reactivity/yield, PIC yield, synthetic nTOF,
  high-Z opacity/EOS, ablation, impurity mixing, and detector-response paths:
  require separate local source packets before promotion.

## Same-Day Physics Follow-Up

The later 2026-05-11 physics pass closed two implementation blockers that were
left open by the initial formulary audit:

- `MLX-010`: field-aware MLX thermal conduction now computes the NRL
  electron-ion Coulomb log and Braginskii high-field perpendicular conductivity
  instead of using a fixed cross-field ratio.
- `CIR-010`: the validation Lee comparison helper now separates axial `fc` from
  radial `fcr` and applies `fcr` to radial circuit-facing inductance, radial
  `dLp/dt`, radial/reflected force, frozen/post-crowbar radial inductance, and
  metadata.

The same pass added fail-closed source-status metadata for radiation transport
and p-B11 diagnostics. Those guardrails do not source-close FLD/Rosseland/
Kramers opacity, p-B11 reactivity/yield, QMF, or line cooling.

## Verification

Commands run after the fixes:

```bash
python3 -m pytest tests/test_formulary_radiation_audit.py tests/test_formulary_transport_audit.py tests/test_formulary_mhd_circuit_audit.py tests/test_ionization.py tests/test_improved_radiation.py tests/test_bremsstrahlung_nrl.py tests/test_radiation_model_metadata.py tests/test_physics.py::TestBraginskii tests/test_physics.py::TestBraginskiiKappaZDependent tests/test_physics.py::TestBraginskiiKappa tests/test_physics.py::TestRecombinationRates tests/test_mhd_solver_consolidated.py::TestBraginskiiConduction tests/test_regime_classifier.py tests/test_research_consolidated.py::TestMagneticReynoldsNumber tests/test_research_consolidated.py::TestRegimeValidity tests/test_circuit_coupler.py tests/test_mlx_kernels.py::test_cyl_source_numpy_pressure_gradient tests/test_mlx_kernels.py::test_cyl_source_numpy_centrifugal tests/test_mlx_sources.py::TestGeometricSources tests/test_cylindrical_energy_source.py tests/test_mlx_riemann.py tests/test_mlx_timestepper.py tests/test_metal_gpu_consolidated.py::TestCylindricalSources -q
```

Result: `202 passed`.

The same-day focused follow-up command also passed:

```bash
python3 -m pytest tests/test_mlx_transport.py::TestThermalConduction tests/test_formulary_transport_audit.py tests/test_lee_model_radial_fcr.py tests/test_lee_model_comparison_audit.py tests/test_snowplow_consolidated.py::TestRadialDLDT tests/test_snowplow_consolidated.py::TestSingleStepRadial tests/test_snowplow_consolidated.py::TestTwoStepRadialTransition tests/test_mlx_snowplow.py::test_radial_current_fraction_defaults_to_current_fraction tests/test_mlx_snowplow.py::test_radial_current_fraction_is_separate_from_axial_fraction tests/test_radiation_model_metadata.py tests/test_qmf_suppression.py tests/test_pb11_yield.py tests/test_formulary_radiation_audit.py -q
```

Result: `80 passed`.

An earlier full `tests/test_research_consolidated.py` run exposed an unrelated
environment/API issue: `src/dpf/diagnostics/beam_target.py` calls
`np.trapezoid`, which is unavailable in the current NumPy. That failure is not
part of the formulary fixes above, but it should be cleaned up before broad
research-suite claims are made.
