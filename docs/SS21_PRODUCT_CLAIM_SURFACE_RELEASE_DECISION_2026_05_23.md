# SS21 Product Claim Surface and Release Decision — 2026-05-23

- Task: `t_ca789d9b`
- Fix/reverify task: `t_98d289df`
- Sprint: SS21 product claim surface and release decision
- Parent dry-run: `docs/SS20_FULL_INTEGRATION_ACCEPTANCE_DRY_RUN_2026_05_23.md`
- Decision basis: SS20 fix/reverify parent reports commit `490d967f840ba9f34cece2f9c972bde0f55fd499`, focused suite `181 passed`, periodic audit PASS, acceptance flags false, and independent focused review PASS.
- Post-review fix/reverify status: independent focused review PASS for SS21 claim surface; no claim-surface wording fixes were required by the reviewer.

## Release decision

Release decision: HONEST-BLOCKED / SOURCE-GATED PREVIEW.

The project may be described as a dense plasma focus simulation workbench with source-gated validation infrastructure, fail-closed evidence packets, and engineering probes. It must not be described as an accepted production first-principles simulator, an accepted full-3D simulator, or an end-to-end predictive DPF simulator.

The public claim boundary remains:

- `accepted_runtime_claim=false`
- `can_support_first_principles_acceptance=false`
- `promotes_acceptance=false`

This decision is a successful honest blocked release posture: the certificate stack did not authorize production runtime acceptance, and the product surface should say that plainly.

## SS20 dry-run outcome carried into SS21

SS20 proved that the fail-closed gates and claim guardrails can run together, but it did not create a public/runtime acceptance claim. The current release posture is therefore:

| Surface | Allowed wording | Disallowed wording |
| --- | --- | --- |
| README / landing copy | "not yet an end-to-end predictive DPF simulator"; "source-gated preview" | "end-to-end predictive simulator" without the negation or gate |
| CLI / app output | "engineering probe"; "not validation evidence"; "fail-closed readiness" | production certificate, accepted first-principles runtime, accepted full-3D |
| V&V docs | source-gated, partial, KR-only authority | broad validated-device counts, publication quality, zero-calibration validation |
| Release notes | honest blocked release; remaining blockers named | certificate accepted, scientific validation accepted, publication-grade validation |

## Product claim surface scan

SS21 treats existing claim-hygiene tests as the enforced product surface, then adds this release-decision guardrail so future README/release edits cannot silently drift.

Existing claim guardrails retained from SS20:

- `tests/test_readme_claims.py`
- `tests/test_scope_claims.py`
- `tests/test_v_and_v_summary_claims.py`
- `tests/test_gradio_claims.py`
- `tests/test_joss_draft_claims.py`
- `tests/test_ai_disclosure_claims.py`
- `tests/test_mlx_claim_guardrails.py`

New SS21 guardrail:

- `tests/test_ss21_product_claim_surface.py`

The guardrail requires this memo, README release posture language, and fail-closed acceptance flags. It rejects broad product overclaims such as publication-grade validation, accepted production certificate language, and full-3D acceptance language.

## README / UI guardrails

README:

- Adds a release posture section pointing to this memo.
- Keeps "not yet an end-to-end predictive DPF simulator" as the lead public claim.
- Names the fail-closed acceptance flags directly.

UI and app surfaces:

- Continue to use preview/source-gated/fail-closed language enforced by `tests/test_gradio_claims.py` and related claim tests.
- Existing app language may expose first-principles commands as engineering probe paths only.
- UI language must not treat engineering probes, synthetic fixtures, or successful packet validators as scientific validation.

## Release notes language to use

Recommended release note:

> DPF-Unified is released as a source-gated preview and engineering-probe workbench. The SS20/SS21 gates preserved fail-closed acceptance: `accepted_runtime_claim=false`, `can_support_first_principles_acceptance=false`, and `promotes_acceptance=false`. No production first-principles, full-3D, or end-to-end predictive DPF claim is authorized.

## Remaining blockers before any accepted claim

1. Complete same-scope source packets for every claimed observable.
2. Close uncertainty propagation and comparator mapping with reviewed production evidence.
3. Pass the certificate stack in one commit with `accepted_runtime_claim`, `can_support_first_principles_acceptance`, and `promotes_acceptance` reviewed explicitly.
4. Obtain independent review approval for any proposed accepted public/runtime claim.
5. Re-scan README/UI/docs/release notes after any certificate-status change.

## Evaluate / Learn / Continue

Evaluate:

- SS20 dry-run outcome remains honest blocked / source-gated preview.
- Claim-surface tests cover README, UI/app, V&V, JOSS draft, AI disclosure, MLX, and the SS21 release memo.
- Acceptance flags remain false on the product surface.

Learn:

- The right product outcome after partial technical success is not silence; it is explicit release posture.
- A release can be useful and shippable as a research workbench while still refusing production scientific acceptance.
- Claim hygiene needs its own regression test because overclaim risk is mostly prose drift, not solver failure.

Continue:

- Route this SS21 output through post-review fix/reverify before treating the release wording as approved.
- Keep README/UI/release language in source-gated preview mode until a reviewer explicitly approves a complete certificate stack and public claim change.
- If a future sprint changes certificate outcome, update this memo and rerun the claim guardrail suite before changing public copy.

## Review status

review-approved: SS21 product claim surface and release decision received independent focused review PASS, and the post-review fix/reverify pass found no wording fix required. This approval does not authorize any accepted production/runtime/full-3D/end-to-end predictive public claim; it approves only the HONEST-BLOCKED / SOURCE-GATED PREVIEW posture with all three acceptance flags false.
