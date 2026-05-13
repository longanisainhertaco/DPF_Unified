# A14 Axis-Calibration Drafts

Generated UTC: `2026-05-11T22:18:51.838680+00:00`

These packets record source-bound crop hashes and draft axis/frame metadata for the first three A14 figure extraction candidates. They contain no digitized series arrays, no residuals, and no independent review acceptance.

## Summary

- Draft packets: 3
- Accepted for validation: 0

| Task | Figure | Source lines | Status | Visible series |
| --- | --- | --- | --- | --- |
| `a14_cikhardtova_2015_fig6_axis_calibration_draft` | Fig. 6 | `200-222` | `axis_calibration_draft_no_series` | -5 ns, +25 ns, +55 ns, +85 ns, +95 ns |
| `a14_klir_2011_fig2_axis_calibration_draft` | Fig. 2 | `172-209` | `axis_calibration_draft_no_series` | FWHM, Rise time |
| `a14_springham_2021_fig5_axis_calibration_draft` | Fig. 5 | `546-616` | `axis_calibration_draft_no_series` | mono-energetic neutrons, Gaussian peak neutrons (200 keV FWHM), Gaussian peak neutrons (400 keV FWHM) |

## Guardrails

- These packets are calibration scaffolds only.
- Pixel ranges are approximate raster-frame metadata and must be refined during numeric extraction.
- Hidden/occluded curve segments must not be synthesized.
- Validation use requires digitized arrays, residual evidence, and accepted independent review through `digitization_verification_evidence()`.
