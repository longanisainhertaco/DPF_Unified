# A14 Cikhardtova Fig. 6 Extraction Blocker

Generated UTC: `2026-05-11T22:58:56.151107+00:00`

This report records why Cikhardtova 2015 Fig. 6 was not converted into a numeric draft packet in this pass.

## Status

- Task: `a14_cikhardtova_2015_fig6_linear_density_extraction_blocker`
- Source: `KnowledgeReference/cikhardtova-plazma-indd-9dfed6c0.md`
- Source lines: `200-222`
- Figure crop: `KnowledgeReference/figures/target-extraction/2026-05-11/cikhardtova-2015-linear-density/crops/page-03-fig-6.png`
- Draft extraction status: `blocked_manual_curve_separation_required`
- Accepted for validation: False

## Blocker

Five monochrome line styles overlap and nearly merge across the same z-axis intervals. A quick point-pick pass could mislabel series and would not be defensible as a draft numeric packet.

## Required Next Steps

- perform manual or vector-assisted curve separation for all five series
- record per-series pixel picks with line-style labels
- measure round-trip residuals for every extracted series
- document uncertainty from overlapping/merged curve regions
- submit the resulting packet for independent review before validation use
