# User-Supplied Paper Intake (2026-05-20)

Generated: 2026-05-20

The user supplied nine local PDFs. This intake promotes only new
source material into `KnowledgeReference/` and records exact SHA
duplicates without creating duplicate KR records.

Guardrail: source availability only. `accepted_runtime_claim` and
`can_support_first_principles_acceptance` remain `False`.

## Summary

- Files scanned: 9
- Promoted into `KnowledgeReference/`: 0
- Skipped existing KR source: 9
- Failed: 0
- accepted_runtime_claim: `False`
- can_support_first_principles_acceptance: `False`

## Promoted Sources

| source | title | journal | pages | sha12 | KR md | KR json | priority | scope | candidate support | parity |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |

No new KR records were created by this idempotent pass. The
sources below were already represented in `KnowledgeReference/`
and are target-extracted separately by
`src/dpf/first_principles/sprint6_user_target_extractions.py`.

## Skipped Existing KR Sources

| source | title | sha12 | reason |
| --- | --- | --- | --- |
| /Users/anthonyzamora/Downloads/scholz_Recent progress.pdf | Recent progress in 1 MJ Plasma-Focus research | d3e51f6c56f7 | source SHA already appears in source-level KR metadata: KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.json |
| /Users/anthonyzamora/Downloads/The_need_of_using_anomalous_resisti.pdf | The need of using anomalous resistivity due to Lower Hybrid Instabilities in plasma-magnetic field interfaces | 73668d0e9860 | source SHA already appears in source-level KR metadata: KnowledgeReference/the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-instabilities-in-plasma-magnet-73668d0e.json |
| /Users/anthonyzamora/Downloads/scholz_PF-1000 device.pdf | PF-1000 device | a2d6bc151ee1 | source SHA already appears in source-level KR metadata: KnowledgeReference/pf-1000-device-a2d6bc15.json |
| /Users/anthonyzamora/Downloads/herold1989.pdf | Comparative analysis of large plasma focus experiments performed at IPF, Stuttgart, and IPJ, Swierk | 51a546954db9 | source SHA already appears in source-level KR metadata: KnowledgeReference/comparative-analysis-of-large-plasma-focus-experiments-performed-at-ipf-stuttgart-and-ipj-51a54695.json |
| /Users/anthonyzamora/Downloads/scholz1999.pdf | Foam liner driven by a plasma focus current sheath | 8324d6194993 | source SHA already appears in source-level KR metadata: KnowledgeReference/foam-liner-driven-by-a-plasma-focus-current-sheath-8324d619.json |
| /Users/anthonyzamora/Downloads/loarer2007.pdf | Gas balance and fuel retention in fusion devices | 09d09d6a8ecb | source SHA already appears in source-level KR metadata: KnowledgeReference/gas-balance-and-fuel-retention-in-fusion-devices-09d09d6a.json |
| /Users/anthonyzamora/Downloads/chouhan,+Artical-8.pdf | Comparison of Plasma Dynamics in Plasma Focus Devices PF1000 and PF400 | 9094f12f0ead | source SHA already appears in source-level KR metadata: KnowledgeReference/comparison-of-plasma-dynamics-in-plasma-focus-devices-pf1000-and-pf400-9094f12f.json |
| /Users/anthonyzamora/Downloads/gribkov2007.pdf | Plasma dynamics in the PF-1000 device under full-scale energy storage: II. Fast electron and ion characteristics versus neutron emission parameters and gun optimization perspectives | 80b44cd62c07 | DOI already appears in source-level KR metadata: 10.1088/0022-3727/40/12/008 (KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md) |
| /Users/anthonyzamora/Downloads/Dense_magnetized_plasma_and_its_app.pdf | Dense magnetized plasma and its applications: review of the 3-year activity of the IAEA Co-ordinated Research Programme | cca325c9ab3b | source SHA already appears in source-level KR metadata: KnowledgeReference/dense-magnetized-plasma-and-its-applications-review-of-the-3-year-activity-of-the-iaea-co-cca325c9.json |

## Failures

| source | title | reason |
| --- | --- | --- |
