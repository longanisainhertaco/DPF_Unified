# Sprint 6 KR Promotion (2026-05-20)

Generated: 2026-05-20

Sprint 6 (`/goal`) WS1 + WS2: three free Nukleonika open-access PDFs
downloaded with SHA-256 verification and ingested into
`KnowledgeReference/` as text-parity records.

Source guardrail: text-parity ingestion only. Figures, tables, plotted
curves, numeric validation targets, runtime closures, and first-
principles claims are NOT accepted until separately reviewed and
target-extracted. `accepted_runtime_claim` and
`can_support_first_principles_acceptance` both remain `False` on every
promoted record.

## Summary

- Files scanned: 3
- Promoted into `KnowledgeReference/`: 3
- Skipped because already represented: 0
- Failed or not promoted: 0
- accepted_runtime_claim: `False`
- can_support_first_principles_acceptance: `False`

## Promoted Sources

| source | title | authors | journal | URL | pages | sha12 | KR md | KR json | priority | scope | resolves | parity |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| bruzzone_bernal_2001_nukleonika_v46n2p059.pdf | The need of using anomalous resistivity due to Lower Hybrid Instabilities in plasma-magnetic field interfaces | Bruzzone, H.; Bernal, L. | Nukleonika 46(2):59-61 (2001) | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46n2p059f.pdf | 3 | 73668d0e9860 | KnowledgeReference/the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-instabilities-in-plasma-magnet-73668d0e.md | KnowledgeReference/the-need-of-using-anomalous-resistivity-due-to-lower-hybrid-instabilities-in-plasma-magnet-73668d0e.json | P1 | dpf_lhi_anomalous_resistivity_quantitative_candidate | CLOSURE-BLK-ANOM-001 (after target extraction + review) | True |
| bruzzone_2001_nukleonika_v46s1p003.pdf | The role of anomalous resistivities in Plasma Focus discharges | Bruzzone, H. | Nukleonika 46 suppl.1:S3-S7 (2001) | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p003f.pdf | 5 | 66678097f945 | KnowledgeReference/the-role-of-anomalous-resistivities-in-plasma-focus-discharges-66678097.md | KnowledgeReference/the-role-of-anomalous-resistivities-in-plasma-focus-discharges-66678097.json | P1 | dpf_anomalous_resistivity_dpf_scope_candidate | CLOSURE-BLK-ANOM-001 (after target extraction + review) | True |
| szydlowski_miklaszewski_2001_nukleonika_v46s1p061.pdf | Neutron and fast ion emission from PF-1000 facility equipped with new large electrodes | Szydlowski, A.; Scholz, M.; Karpinski, L.; Sadowski, M.; Tomaszewski, K.; Paduch, M.; Miklaszewski, R. | Nukleonika 46 suppl.1:S61-S64 (2001) | http://www.ichtj.waw.pl/ichtj/nukleon/back/full/vol46_2001/v46s1p061f.pdf | 4 | dc61e78e8c97 | KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.md | KnowledgeReference/neutron-and-fast-ion-emission-from-pf-1000-facility-equipped-with-new-large-electrodes-dc61e78e.json | P2 | pf1000_large_electrodes_geometry_neutron_emission_candidate | PF1000-BLK-009 hardware-scope hollow-bore (after target extraction + review) | True |

## Skipped Existing KR Coverage

| source | title | sha12 | reason |
| --- | --- | --- | --- |

## Failures / Not Promoted

| source | reason |
| --- | --- |
