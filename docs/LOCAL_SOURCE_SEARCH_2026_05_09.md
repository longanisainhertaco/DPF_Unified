# Local Source Search - 2026-05-09

Scope: searched for the current `docs/SOURCE_ACQUISITION_NEEDED.md`
paper and method-reference targets outside
`/Users/anthonyzamora/dpf-unified/KnowledgeReference/`.

Source-of-truth boundary: every hit listed here started as an acquisition
candidate only. None of these files can support physics, validation, readiness,
or requirements-closure claims until it is reviewed into `KnowledgeReference/`,
hashed, mapped to typed targets or digitization packets, and accepted. LeVeque
2002 was later converted and promoted under that process on 2026-05-09.

## Search Coverage

Primary roots checked:

- `/Users/anthonyzamora/Downloads`
- `/Users/anthonyzamora/Downloads/OneDrive_1_4-2-2026`
- `/Users/anthonyzamora/Downloads/OneDrive_2_4-1-2026`
- `/Users/anthonyzamora/Downloads/Code/2025/gpt`
- `/Users/anthonyzamora/DPF-U2`
- `/Users/anthonyzamora/dpf-unified/archive_reference_OLD`
- `/Users/anthonyzamora/dpf-unified/downloaded_books_papers`
- `/Users/anthonyzamora/tools/claude-memory-db/memory-stage/dpf-papers`

Excluded from content scans:

- `KnowledgeReference/`
- `node_modules/`, `.web/`, `.next/`, `.git/`, `build/`, `dist/`, caches
- clearly unrelated personal/download subtrees such as 3D model packs and
  business/personnel document folders

Methods used:

- Spotlight metadata search with `mdfind` for target titles/DOIs.
- Filename search with `find` plus exact DOI/title/author fragments.
- Text search with `rg` over Markdown, JSON, BibTeX-like, and extracted text
  artifacts, with build/cache exclusions and `--max-filesize` limits.
- PDF first-page checks with `pdftotext -f 1 -l 4` for the six target papers
  across top-level Downloads, OneDrive paper drops, GPT paper downloads,
  DPF-U2 paper pools, and old project paper archives.
- PDF first-page checks with `pdftotext -f 1 -l 6` for textbook/method targets.
- PDF metadata and SHA-256 checks with `pdfinfo` and `shasum -a 256` for
  local textbook candidates.

## Blocking Paper Results

No exact local PDF copy was found outside `KnowledgeReference/` for the six
blocking or secondary acquisition papers:

- Klir et al. 2011, DOI `10.1063/1.3559548`
- Sadowski/Scholz/PF-1000 team 2004, DOI `10.1016/j.vacuum.2004.07.040`
- Catenacci et al. 2020, DOI `10.1109/TPS.2020.3012104`
- Springham et al. 2021, DOI `10.1016/j.nima.2020.164830`
- Jednorog et al. 2017, DOI `10.1515/nuka-2017-0003`
- Cikhardtova et al. 2015, DOI `10.1515/nuka-2015-0065`

Observed non-source hits:

- Klir et al. 2011 appears only as reference-list text in the 2026 hybrid
  X-pinch material, for example
  `/Users/anthonyzamora/DPF-U2/organized-papers/07-Diagnostics/converted/deuterium-hybrid-x-pinch-driven-by-small-dense-pla.json`
  and corresponding converted Markdown in
  `/Users/anthonyzamora/tools/claude-memory-db/memory-stage/dpf-papers/`.
  This is a citation pointer, not the Klir paper.
- Several filename/text false positives were caused by generic terms such as
  `vacuum`, `Ideal MHD`, or DOI-number fragments in package/cache paths. These
  did not match target title/DOI first-page checks.

Conclusion: the six paper acquisitions remain open. The local machine search
did not turn up an ingestible exact copy outside `KnowledgeReference/`.

## Textbook And Method-Reference Results

### Local Candidate: LeVeque

- Path:
  `/Users/anthonyzamora/dpf-unified/archive_reference_OLD/references/papers/textbooks/leveque-2002-finite-volume-hyperbolic.pdf`
- Metadata title: `Finite Volume Methods for Hyperbolic Problems`
- Metadata author: `RANDALL J.LEVEQUE`
- Pages: `580`
- SHA-256:
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`
- Status: promoted on 2026-05-09 to
  `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` and
  `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.json`.
  Schema validation and PDF-text parity passed across all 580 pages. Usable
  scope is finite-volume/hyperbolic-conservation-law method authority for
  numerical verification only.

### Local Partial Candidate: Toro

- Path:
  `/Users/anthonyzamora/dpf-unified/archive_reference_OLD/references/papers/textbooks/toro-2009-riemann-solvers-excerpt.pdf`
- Metadata title:
  `Riemann Solvers and Numerical Methods for Fluid Dynamics - ReadingSample`
- Metadata subject:
  `Riemann Solvers and Numerical Methods for Fluid Dynamics, A Practical Introduction`
- Pages: `47`
- SHA-256:
  `78144939eadb0f7382c222f49a9a11ce9bae3e19c4f866b94e4aa6de1f39d73f`
- Status: reading sample/excerpt only; full book still not found.

### Local Partial Candidate: Rybicki And Lightman

- Path:
  `/Users/anthonyzamora/dpf-unified/archive_reference_OLD/references/papers/textbooks/rybicki-lightman-1979-radiative-processes.pdf`
- First-page title text:
  `RADIATIVE PROCESSES IN ASTROPHYSICS`
- Pages: `63`
- SHA-256:
  `fcff04d2c6c1c77855192cd107ad144497cc7637706a66278658af1a5f23a08d`
- Status: partial local candidate with frontmatter and Chapter 1 material; full
  book still not found.

### Citation-Only Or Not Found

The following method references were not found as exact local books in the
searched roots:

- Hutchinson, *Principles of Plasma Diagnostics*
- Freidberg, *Ideal MHD*
- Goedbloed, Keppens, Poedts, *Magnetohydrodynamics of Laboratory and
  Astrophysical Plasmas*
- Birdsall and Langdon, *Plasma Physics via Computer Simulation*
- Griem, *Principles of Plasma Spectroscopy*
- Atzeni and Meyer-ter-Vehn, *The Physics of Inertial Fusion*

Some of these appear in bibliography/reference-list text or implementation
comments only. Those are not local copies of the source.

## Intake Recommendations

1. LeVeque ingestion is complete for method-reference scope. Use only the new
   `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.*`
   records for source-backed method references.
2. Keep Toro listed as still needing the full source; the local file is only a
   reading sample.
3. Keep Rybicki and Lightman listed as still needing the full source; the local
   file is only a partial candidate with frontmatter and Chapter 1 material.
4. Leave all six blocking paper acquisitions open until exact source documents
   are acquired and reviewed into `KnowledgeReference/`.
5. After any acquisition, rerun:
   - `scientific_closure_source_acquisition_queue()`
   - `kr_validation_same_scope_target_report()`
   - `scientific_accuracy_gap_report()`
   - predictive/high-fidelity readiness checks
