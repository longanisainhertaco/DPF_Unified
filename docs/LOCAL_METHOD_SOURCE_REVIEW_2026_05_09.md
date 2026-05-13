# Local Method Source Review - 2026-05-09

Scope: intake review of the local method-source candidates found outside
`KnowledgeReference/` during `docs/LOCAL_SOURCE_SEARCH_2026_05_09.md`.

Source-of-truth boundary: this document began as an intake review, not a
`KnowledgeReference/` promotion. Since that intake, LeVeque 2002 has been
converted and promoted to `KnowledgeReference/` method-reference records. On
2026-05-11, the full Toro 2009 source was also promoted to
`KnowledgeReference/toro2009-433cd861.md` / `.json` and chunked into readable
page-range Markdown under `KnowledgeReference/chunks/toro2009-433cd861/`.
Rybicki-Lightman remains a full-source acquisition candidate.

## Review Method

Commands and checks used:

- `pdfinfo` for metadata, page count, encryption status, and file size.
- `shasum -a 256` for candidate identity.
- `pdftotext -layout` for table-of-contents and content scans.
- `pdftoppm -png` plus spot visual inspection for representative pages.
- Repository search against `src/dpf`, `tests`, `scripts`, and SRS/RTM docs to
  map source coverage to current finite-volume and verification surfaces.

Temporary extraction/rendering files were written under
`tmp/pdfs/method_source_review_2026_05_09/`.

## Candidate 1: LeVeque 2002

Local file:
`archive_reference_OLD/references/papers/textbooks/leveque-2002-finite-volume-hyperbolic.pdf`

Metadata:

- Title: `Finite Volume Methods for Hyperbolic Problems`
- Author: `RANDALL J.LEVEQUE`
- Pages: `580`
- File size: `16182581` bytes
- Encrypted: `no`
- SHA-256:
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`

Completeness assessment: promoted full local candidate. The extracted text
contains front matter, full table of contents, chapters 1-23, references, and
index material. Spot-rendered pages show normal page layout. The PDF was
converted to `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md`
and `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.json` on
2026-05-09.

Promotion evidence:

- `scripts/validate_kr_schema.py` passed for the JSON record.
- `scripts/verify_kr_pdf_parity.py` passed against all `580` PDF pages with no
  JSON page-text mismatches and no Markdown missing pages.
- Original PDF SHA-256:
  `b3adec0d3616dbde57a5522cfce1861890887d7c03a2232d2136cb94c9bac1d5`

Relevant coverage found:

- Conservation laws and differential equations.
- Characteristics and Riemann problems for linear hyperbolic equations.
- Finite-volume methods, CFL condition, upwind methods, Godunov method,
  numerical fluxes, Roe method, and flux splitting.
- High-resolution methods, TVD/flux limiters, slope limiters, nonuniform grids,
  and boundary/ghost-cell handling.
- Convergence, accuracy, stability, and local truncation error.
- Nonlinear scalar conservation laws, entropy conditions, Lax-Wendroff theorem,
  nonlinear systems, shallow-water examples, gas dynamics, Euler equations,
  shock tubes, approximate Riemann solvers, source terms, balance laws, and
  multidimensional finite-volume methods.
- Multidimensional systems, transverse Riemann solvers, and quadrilateral-grid
  finite-volume treatment.

Fit to current DPF-Unified work:

- Strong method-source fit for Tier-3 code numerical verification language
  around finite-volume conservation-law discretization, CFL-limited explicit
  stepping, shock-capturing tests, convergence/error studies, boundary handling,
  and source-term splitting review.
- Useful for tightening the documented basis of current finite-volume checks in
  `tests/test_mhd_numerical_fidelity.py`,
  `scripts/build_mhd_tier3_numerical_packet.py`, and related evidence builders.
- Useful for reviewing claims around generic analytic tests such as Sod shock
  tubes and nonlinear hyperbolic method behavior.

Guardrails:

- This book is method authority, not DPF experimental evidence.
- It is not a same-device PF-1000 source and cannot close Tier 4 spatial
  validation, Tier 5 neutron validation, predictive readiness, or
  high-fidelity scientific readiness.
- It is not specific enough by itself to replace local DPF/MHD references
  already used for cylindrical DPF scope limits, circuit coupling, and
  same-scope validation.
- It may now be cited as local `KnowledgeReference/` method authority only for
  finite-volume and hyperbolic-conservation-law numerical verification.

Recommendation: use the promoted KR records only for method-verification
support around finite-volume and hyperbolic-conservation-law verification. Do
not map it to experimental DPF validation.

## Candidate 2: Toro 2009

2026-05-11 supersession: the 47-page reading sample below is no longer the
best local Toro source. The full 738-page Springer source was promoted as
`KnowledgeReference/toro2009-433cd861.md` and
`KnowledgeReference/toro2009-433cd861.json`, with 30 page-range Markdown chunks
under `KnowledgeReference/chunks/toro2009-433cd861/`. The original caution
still applies to any code claim that has not yet been mapped to a specific
reviewed chapter/page target.

Local file:
`archive_reference_OLD/references/papers/textbooks/toro-2009-riemann-solvers-excerpt.pdf`

Metadata:

- Title: `Riemann Solvers and Numerical Methods for Fluid Dynamics - ReadingSample`
- Subject:
  `Riemann Solvers and Numerical Methods for Fluid Dynamics, A Practical Introduction`
- Author: `Eleuterio F Toro - Springer-Verlag Berlin Heidelberg`
- Pages: `47`
- File size: `754410` bytes
- Encrypted: `no`
- SHA-256:
  `78144939eadb0f7382c222f49a9a11ce9bae3e19c4f866b94e4aa6de1f39d73f`

Completeness assessment: not a full source. Visual inspection confirms it is a
beck-shop/Springer reading sample. The text begins with Chapter 2, "Notions on
Hyperbolic Partial Differential Equations", and the excerpt itself points to
later chapters for the exact Euler Riemann problem and approximate Riemann
solvers.

Relevant coverage found:

- Introductory hyperbolic PDE definitions.
- Conservation-law notation.
- Eigenvalue/eigenvector definitions.
- Cauchy and Riemann problem basics.
- Linearized gas-dynamics example.
- Elementary shock/contact/rarefaction wave conditions.

Fit to current DPF-Unified work:

- Useful as a pointer and sanity check for basic terminology, but insufficient
  for authoritative review of HLL/HLLD/Roe-family solver implementation,
  approximate solver selection, full Euler equations treatment, or production
  Riemann-solver acceptance criteria.

Guardrails:

- Do not cite this excerpt as the full Toro method source.
- Do not use it to close Riemann-solver method authority.
- Superseded 2026-05-11: use the promoted full Toro KR record instead of this
  excerpt for future method review.

Recommendation: keep this excerpt as a historical local partial-source pointer
only. Use the promoted full Toro KR record before promoting Riemann-solver and
shock-tube tests from implementation checks to documented source-backed
verification.

## Candidate 3: Rybicki And Lightman

Local file:
`archive_reference_OLD/references/papers/textbooks/rybicki-lightman-1979-radiative-processes.pdf`

Metadata:

- Title: `Frontmatter`
- Pages: `63`
- File size: `2558129` bytes
- Encrypted: `no`
- SHA-256:
  `fcff04d2c6c1c77855192cd107ad144497cc7637706a66278658af1a5f23a08d`

Completeness assessment: partial candidate, not a full source. The extracted
text includes title/copyright/preface material, a table of contents for the
full book, and Chapter 1 material on fundamentals of radiative transfer,
ending with Chapter 1 problems/references. The table of contents references an
index around page 375, while this local PDF has only 63 physical pages.

Relevant coverage found:

- Specific intensity, flux, radiative transfer equation, optical depth, source
  function, mean free path, radiation force, thermal radiation, Einstein
  coefficients, scattering, radiative diffusion, Eddington approximation, and
  related Chapter 1 problems.

Fit to current DPF-Unified work:

- Could support only a limited future review of basic radiative-transfer
  definitions after KR ingestion.
- It does not include the full book content needed for bremsstrahlung,
  synchrotron, Compton, or broader radiation-process implementation review.

Guardrails:

- Do not mark Rybicki-Lightman as acquired in full.
- Do not use this partial PDF to source-back current radiation-loss,
  bremsstrahlung, or full radiative-process claims.
- Keep the full Rybicki-Lightman source on the acquisition list unless the
  planned radiation review is explicitly limited to Chapter 1 definitions.

Recommendation: keep as a partial local candidate. It is not sufficient to
close radiation method authority.

## Current Actionable Status

| Source | Review result | Recommended next action |
| --- | --- | --- |
| LeVeque 2002 | Promoted KR method-reference record | Use for finite-volume method authority only; do not use as DPF experimental evidence. |
| Toro 2009 | Full source promoted 2026-05-11; old reading sample remains partial only | Use `KnowledgeReference/toro2009-433cd861.md` and its chunks for source-backed Riemann-solver review; still extract chapter/page method targets before closure. |
| Rybicki-Lightman | Partial Chapter 1 candidate | Acquire full source before source-backed radiation-process closure. |

## Planned Follow-Up

1. Update method-verification source mappings to cite
   `KnowledgeReference/finite-volume-methods-for-hyperbolic-problems.md` where
   LeVeque method authority is used.
2. Keep Rybicki-Lightman as an acquisition blocker; keep Toro as a
   method-target extraction blocker rather than a source-acquisition blocker.
3. Rerun the SRS trace audit and finite-volume/Tier-3 evidence tests after any
   method-source mapping changes.
