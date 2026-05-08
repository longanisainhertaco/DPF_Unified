# Local PDF Source Audit

Updated: 2026-05-06

Scope: local PDFs under the DPF-Unified tree were checked for the candidate
documents in `docs/SCIENTIFIC_CLOSURE_SOURCE_QUEUE.md`. Matching was done by
filename search, PDF metadata, DOI/title text extraction, and SHA-256 duplicate
checks.

Source-of-truth rule: a PDF found outside `KnowledgeReference/` is not
source-of-truth evidence. It is only available for ingestion.

## Found

| Document | Local PDF match | KR status |
| --- | --- | --- |
| Akel et al. 2021, DOI `10.1016/j.radphyschem.2021.109633` | `archive_reference_OLD/references/papers/core-dpf/akel-2021-pf1000-neutron-yield.pdf`; duplicate same hash under `archive_reference_OLD/references/papers/archive/` | In KR as `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md`; filename is generic but title/DOI match. |
| Gribkov et al. 2007 Part I, DOI `10.1088/0022-3727/40/7/021` | `archive_reference_OLD/references/papers/core-dpf/gribkov-2007-pf1000-jphysd-part2.pdf` | In KR as `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md`; filename says part2 but content is Part I. |
| Gribkov et al. 2007 Part II, DOI `10.1088/0022-3727/40/12/008` | `archive_reference_OLD/references/papers/core-dpf/scholz-2007-pf1000-part2-jphysd.pdf` | In KR as `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`; filename uses Scholz but content first author is Gribkov. |
| Schmidt et al. 2022 MJOLNIR, DOI `10.1063/5.0089121` | `archive_reference_OLD/references/papers/core-dpf/goyon-2022-mjolnir-high-low.pdf` | In KR as `KnowledgeReference/goyon-2022-mjolnir-high-low.md`; filename is author-misleading. Accepted-manuscript/preprint copy also exists as `petrov-2022-mjolnir-high-low-discharges.pdf` and KR markdown. |
| Malir et al. 2024, DOI `10.1063/5.0193268` | `archive_reference_OLD/references/papers/core-dpf/malir-2024-interferometry-dpf.pdf` | In KR as `KnowledgeReference/malir-2024-interferometry-dpf.md`. |
| Goyon et al. 2025, DOI `10.1063/5.0253547` | `archive_reference_OLD/references/papers/core-dpf/goyon-2025-ma-class-dpf-neutron.pdf`; duplicate identical hash at `archive_reference_OLD/references/papers/core-dpf/Neutron_generation_dynamics_inside_a_MA-class_dens.pdf` and under `archive_reference_OLD/references/papers/archive/` | In KR under long title filenames `neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch*.md`. |

## Not Found

No exact local PDF match was found for:

- Cikhardtova et al. 2015, DOI `10.1515/nuka-2015-0065`
- Sadowski/Scholz/PF-1000 team 2004, DOI `10.1016/j.vacuum.2004.07.040`
- Catenacci et al. 2020, DOI `10.1109/TPS.2020.3012104`
- Springham et al. 2021, DOI `10.1016/j.nima.2020.164830`
- Klir et al. 2011, DOI `10.1063/1.3559548`
- Jednorog et al. 2017, DOI `10.1515/nuka-2017-0003`

Some local PDFs contain these candidates only in bibliography/reference lists.
Those are false positives, not local copies of the candidate papers.

## Parity Verification

I verified the exact local PDF matches against their existing
`KnowledgeReference` markdown/JSON pairs with
`scripts/verify_kr_pdf_parity.py`. The verifier requires PDF page count to
match the KR JSON page count, every PDF page's extracted text to match the KR
JSON `pages[].text`, and every PDF page's extracted text to be present in the
KR markdown after normalization.

| Document | KR markdown | PDF SHA-256 | Text parity |
| --- | --- | --- | --- |
| Akel et al. 2021 | `KnowledgeReference/radiation-physics-and-chemistry-188-2021-109633.md` | `9a762bc36bc1f5c175a0ec8dc07b69c48ad956d0c6a382882daf4e24677dcb3b` | passed, 6/6 pages |
| Gribkov et al. 2007 Part I | `KnowledgeReference/gribkov-2007-pf1000-jphysd-part2.md` | `7acfb46d1db6ee5894978f70e1372edda7efaa5171d8e7c3bdf0baf7025eff43` | passed, 13/13 pages |
| Gribkov et al. 2007 Part II | `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md` | `c4d62f5015bc6040aa85070e43f3cb6e7e4a8329e5d2baf33fa4d38f828caa4f` | passed, 16/16 pages |
| Schmidt et al. 2022 MJOLNIR article | `KnowledgeReference/goyon-2022-mjolnir-high-low.md` | `89877f5c880dcd9c4454925984398cf51984f95d2ff78ac4437f5f755e98fe6a` | passed, 29/29 pages |
| Schmidt/Goyon accepted-manuscript copy | `KnowledgeReference/petrov-2022-mjolnir-high-low-discharges.md` | `d9674bd39b12c3a87e7549c540384f56722d739f5b85a693fab73c24b2d32623` | passed, 16/16 pages |
| Malir et al. 2024 | `KnowledgeReference/malir-2024-interferometry-dpf.md` | `fafc32261c9172702b1c8dfdc92bcc33b1a32aeeb4cb9680d535478191db46c9` | passed, 14/14 pages |
| Goyon et al. 2025 canonical KR record | `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch.md` | `9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c` | passed, 10/10 pages using the matching duplicate PDF filename |
| Goyon et al. 2025 short-name KR duplicate | `KnowledgeReference/neutron-generation-dynamics-inside-a-ma-class-dense-plasma-focus-z-pinch-5.md` | `9c0bc58d72ced9c914914aabdab63937a2b9c7820950eb0fa2412be9fd9d0f8c` | passed, 10/10 pages |

This is text parity. Figure pixels and plotted curves are not numeric data until
they pass `digitization_verification_evidence()`.

## Next Action

No new `KnowledgeReference` markdown file was created in this pass because all
exact local PDF matches already have KR markdown/JSON pairs and passed parity.
The next action is not paper ingestion. Akel et al. 2021 Tables 1 and 2 are
now typed as `pf1000_16kv_shot_table_2021_akel`; the remaining Akel work is
verified digitization of the waveform/yield figures through
`digitization_verification_evidence()`.
