# Scholz 2001 PF-1000 Hardware Target Extraction (2026-05-20)

Source: `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md`

Source PDF SHA-256:
`d3e51f6c56f734e871f657f950486be441f75df9b75660e4524675738b002c75`

Status: `source_available_not_target_extracted` -> target-extracted for
hardware context only. Runtime acceptance remains false.

## Guardrail

This packet extracts PF-1000 2001 large-electrode hardware dimensions and
diagnostic context. It does not accept a reviewed 3-D material mask, an
Akel-scope validation comparator, anomalous-resistivity closure, neutron
authority, or whole-shot first-principles certificate.

Acceptance flags:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`

## Extracted Targets

| target | value | units | source |
| --- | ---: | --- | --- |
| cathode rod count | 24 | count | KR lines 90-92 |
| cathode rod length | 0.600 | m | KR lines 90-92 |
| cathode rod diameter | 0.032 | m | KR lines 90-92 |
| outer electrode diameter | 0.400 | m | KR lines 91-93 |
| inner electrode material | copper | flag | KR lines 93-94 |
| inner electrode diameter | 0.244 | m | KR lines 93-94 |
| end-face hole diameter | 0.030 | m | KR lines 94-95 |
| interelectrode gap | 0.062 | m | KR lines 95-96 |
| insulator material | alumina | flag | KR lines 96-98 |
| insulator outer diameter | 0.229 | m | KR lines 96-98 |
| insulator length | 0.113 | m | KR lines 96-98 |
| bank modules | 12 | count | KR lines 98-100 |
| capacitors per module | 24 | count | KR lines 98-100 |
| per-capacitor voltage | 50 | kV | KR lines 98-100 |
| per-capacitor capacitance | 4.625 | uF | KR lines 98-100 |
| charging-voltage range | 20-40 | kV | KR lines 101-103 |
| bank-energy range | 266-1064 | kJ | KR lines 101-104 |
| quarter discharge time | 5.4 | us | KR lines 101-104 |
| PIN1 observation offset | 0.020 | m | KR lines 116-119 |
| PIN2 pinhole diameter | 100e-6 | m | KR lines 120-122 |
| PIN2 Be filter thickness | 20e-6 | m | KR lines 120-122 |
| scintillator distance | 15 | m | KR lines 149-158 |
| good-shot energy | 1070 | kJ | KR lines 166-170 |
| good-shot neutron yield | 2.06e11 | neutrons | KR lines 166-170 |
| empirical scaling exponent | 3.3 | exponent | KR lines 245-254 |
| no reported yield saturation below | 2.3 | MA | KR lines 251-254 |

## Blocker Impact

Resolved as source targets after review:

- `PF1000-BLK-004` cathode rod length for the 2001 24-rod PF-1000 large-electrode configuration.
- `PF1000-BLK-015` insulator outer radius for the 2001 24-rod PF-1000 large-electrode configuration.

Candidate context only:

- `PF1000-BLK-009`: the paper gives a 30 mm diameter end-face hole, but that
  is not a full hollow-bore radius/length runtime mask. Full hollow-anode bore
  authority remains blocked.

Still blocked:

- `PF1000-BLK-010` anode bore length.
- `PF1000-BLK-016` insulator wall thickness.
- `PF1000-BLK-017` backplate radial extent.
- `PF1000-BLK-018` backplate axial thickness.

## Render Evidence

Page 2 / journal page 36 was rendered to verify Figure 1 geometry labels:

- Artifact: `docs/extractions/scholz_2001_recent_progress_render_evidence/pdf_p002_journal_p036-2.png`
- SHA-256: `4c9657f07a2a2caf6949e677f426d0ede55a3700238214e687c874b34ae60c84`

The render confirms the visible Figure 1 labels for the 24-rod electrode,
400 mm outer-electrode diameter, 32 mm rod diameter, 244 mm inner-electrode
diameter, 30 mm end-face hole, 447 mm inner-electrode axial label, and
113 mm insulator axial label. The line-referenced text extraction already
captures the targets listed above, so this render is retained as visual
cross-check evidence rather than an acceptance artifact.

## Machine-Readable Packet

Implemented in:
`src/dpf/first_principles/sprint6_user_target_extractions.py`

Tests:
`tests/test_sprint6_user_supplied_extractions.py`
