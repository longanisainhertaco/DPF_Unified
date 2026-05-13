# PDF Source Inventory

Generated: 2026-05-11

Source guardrail: this is a file inventory only. A PDF-like file is not scientific evidence until it is reviewed into `KnowledgeReference/`, hashed, and mapped to source/target records.

## Summary

- Total scanned PDF-like files: 1298
- Total unique SHA-256 payloads: 651
- Project PDF-like files excluding `KnowledgeReference/`: 1159
- Project unique SHA-256 payloads excluding `KnowledgeReference/`: 583
- Downloads PDF-like files scanned: 139
- Downloads unique SHA-256 payloads: 130
- Active research-paper intake files: 91
- Active research-paper intake unique SHA-256 payloads: 91
- Duplicate SHA-256 groups across scanned scopes: 457

The previous 91-unique count was only the active `downloaded_books_papers/Research Papers` intake scope. The broader local source inventory is much larger and should be triaged before bulk KR promotion.

## Scope Counts

| scope | files | unique SHA-256 | bytes |
| --- | ---: | ---: | ---: |
| active_research_papers_intake | 91 | 91 | 296607872 |
| archive_reference_old_other | 114 | 107 | 176972208 |
| archive_reference_old_papers | 952 | 508 | 2981647377 |
| downloaded_books_papers_other | 1 | 1 | 1716723 |
| downloads | 139 | 130 | 370122432 |
| external_vendor_or_backend_docs | 1 | 1 | 761252 |

## Top Directories

| directory | files | unique SHA-256 |
| --- | ---: | ---: |
| `./archive_reference_OLD/references/papers/archive` | 292 | 258 |
| `./archive_reference_OLD/references/papers/core-dpf` | 115 | 101 |
| `./archive_reference_OLD/references/papers/textbooks` | 110 | 104 |
| `./archive_reference_OLD/references/papers/adjacent-fields` | 102 | 95 |
| `/Users/anthonyzamora/Downloads` | 89 | 86 |
| `./archive_reference_OLD/references/papers/plasma-physics` | 89 | 81 |
| `./archive_reference_OLD/references/papers/datasets` | 57 | 57 |
| `./archive_reference_OLD/references/papers/mhd-numerics` | 54 | 53 |
| `./archive_reference_OLD/references/papers/legacy-code-docs` | 52 | 42 |
| `./archive_reference_OLD/reference/legacy-simulators/HeliosMatrix/gpt` | 33 | 33 |
| `./downloaded_books_papers/Research Papers/2026-05-11-user-ingest` | 30 | 30 |
| `./downloaded_books_papers/Research Papers` | 28 | 28 |
| `./archive_reference_OLD/references/papers/z-pinch` | 27 | 25 |
| `./archive_reference_OLD/references/papers/machine-learning` | 21 | 21 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots/inference` | 19 | 19 |
| `./downloaded_books_papers/Research Papers/Wave2` | 19 | 19 |
| `/Users/anthonyzamora/Downloads/AmberDeVan` | 17 | 17 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots/wd-scan_big` | 16 | 16 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots/beta-NLL_wd-scan` | 12 | 12 |
| `./archive_reference_OLD/references/papers/nuclear-radiation` | 12 | 12 |
| `./downloaded_books_papers/Research Papers/Wave 5` | 10 | 10 |
| `/Users/anthonyzamora/Downloads/Business-RaizYRazon` | 9 | 9 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots` | 8 | 8 |
| `/Users/anthonyzamora/Downloads/OneDrive_1_4-2-2026` | 7 | 7 |
| `/Users/anthonyzamora/Downloads/Political-NBMP` | 6 | 6 |
| `./archive_reference_OLD/references/papers/circuit-engineering` | 6 | 6 |
| `./archive_reference_OLD/references/papers/diagnostics` | 5 | 5 |
| `./archive_reference_OLD/references/papers/uq` | 5 | 5 |
| `/Users/anthonyzamora/Downloads/05bbe3814c3007af4211fab963fbddc9ab43fbdbc9f2b90b6705b2a48be589ba-2026-03-19-14-58-23-68e16b5fd01341ba85ca451cf68b6a0f` | 4 | 4 |
| `/Users/anthonyzamora/Downloads/OneDrive_2_4-1-2026` | 4 | 4 |
| `./archive_reference_OLD/reference/legacy-simulators/HeliosMatrix` | 4 | 4 |
| `./archive_reference_OLD/reference/legacy-simulators/KelixDPFV1.0/documentation` | 4 | 1 |
| `./downloaded_books_papers/Research Papers/Wave4` | 4 | 4 |
| `/Users/anthonyzamora/Downloads/OneDrive_1_4-1-2026-2` | 3 | 3 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468` | 3 | 3 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots/beta-NLL-compare` | 3 | 3 |
| `./archive_reference_OLD/reference/datasets/physicistphil-lapd-isat-predict-390c468/code/plots/cv-tests` | 3 | 3 |
| `./archive_reference_OLD/reference/legacy-simulators/picongpu-dev/docs/logo` | 3 | 3 |
| `./archive_reference_OLD/references/papers/computing` | 3 | 3 |
| `./archive_reference_OLD/reference/legacy-simulators/HeliosMatrixDPF1.0/research_paper` | 2 | 1 |

## Next Action

- Keep the active research-paper intake as the reviewed promotion surface.
- Triage `archive_reference_OLD/references/papers` before copying into active intake.
- Do not bulk-promote vendor docs, generated plots, logos, or stale simulator artifacts.
- Promote textbooks with chunked Markdown indexes so full-book context remains readable.
