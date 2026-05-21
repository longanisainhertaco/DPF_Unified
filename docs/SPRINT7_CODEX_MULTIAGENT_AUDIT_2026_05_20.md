# Sprint 7 Codex Multi-Agent Audit (2026-05-20)

Audit target: Sprint 7 first-principles runtime contract work at HEAD
`35bb1a9`.

Audit posture: first-principles source-authority review only. This is not a
reduced-model validation pass, not a validation-certificate review, and not a
whole-shot acceptance review.

## Verdict

Sprint 7 is accepted as a fail-closed runtime-contract sprint with required
corrections before the next physics wiring sprint.

It correctly:

- exposes `hybrid_pic_3d_readiness` through the package-native runner, CLI
  telemetry, manifest candidate evidence, and validation packet;
- keeps candidate 3-D evidence non-promoting;
- rejects Te/Ti caveat/model/manual evidence for same-scope acceptance;
- keeps reduced Lee/snowplow models as comparator baselines only;
- adds a revision-specific 2000/2001 24-rod PF-1000 geometry constructor
  without mutating Akel/Krauz constructors.

It does not:

- produce an accepted first-principles runtime claim;
- produce an engineering-firm validation certificate;
- provide a reviewed full 3-D same-scope validation packet;
- provide an accepted 12 us source-sign whole-shot artifact;
- make Bennett 2017 KR-authoritative;
- synchronize every ledger/export after the Sprint 7 transitions.

All current acceptance flags remain false. This is correct.

## Audit Findings

### S7-A1 - Bennett 2017 Over-Promoted In WS-E Text

Severity: High.

`docs/extractions/SPRINT7_WSE_NEXT_PHYSICS_SOURCE_PACKETS_2026_05_20.md`
states that both primary-source packets are `target_extracted_source_supported`,
but the same packet also says Bennett 2017 still must be ingested as
`KnowledgeReference/bennett-2017-kinetic-dpf-breakdown.md` and that this was
not done in Sprint 7. The source-acquisition ledger also has
`bennett_2017_startup.already_in_kr=false`.

Correct state:

- Bennett 2017 is source-available on disk and line/page verified.
- Bennett 2017 is not yet KR-authoritative.
- Bennett 2017 must not be consumed by runtime startup code until canonical KR
  ingestion plus target extraction exist.

Required correction:

- Rewrite Sprint 7 WS-E wording from `target_extracted_source_supported` to
  `on_disk_line_page_verified_kr_promotion_required` for Bennett.
- Add a regression test that fails when a source packet claims target-extracted
  status while the source ledger says `already_in_kr=false`.

### S7-A2 - Braginskii Ledgers Are Stale Against Target Extraction

Severity: High.

The blocker ledger still says `CLOSURE-BLK-BRAG-001` is
`pdf_present_needs_rendered_page_or_ocr_verification`, while the repository
contains `docs/extractions/BRAGINSKII_1965_TABLE_2_TARGET_EXTRACTION_2026_05_20.md`
and the Sprint 7 packet re-confirms render evidence. The source ledger still
says to render/OCR before target extraction.

Correct state:

- Braginskii Table 2 Z=1 cells are render-verified and target-extracted.
- Eqs. 4.30-4.45 still need equation-level extraction before full transport
  runtime consumption.
- Five review-required cells remain blocked.
- No runtime coefficient is accepted.

Required correction:

- Update the blocker and source ledgers to reflect
  `target_extracted_source_supported_pending_equation_extraction_and_review`.
- Replace tests that expect the old `pdf_present_needs_rendered_page_or_ocr_verification`
  status.
- Keep `accepted_runtime_claim=false` and
  `can_support_first_principles_acceptance=false`.

### S7-A3 - Ledger Tests Lock Stale Commit And Status Values

Severity: Medium.

`tests/test_first_principles_v2_handoff_ledgers.py` still requires
`last_verified_commit == 8e6b5e9` and still expects one
`pdf_present_needs_rendered_page_or_ocr_verification` row. The focused test
suite passes, but the tests are proving the stale ledger shape rather than the
Sprint 7 state.

Required correction:

- Stop hardcoding stale commit IDs in ledger tests unless the row is explicitly
  testing a historical artifact.
- Add tests for expected status transitions keyed by blocker ID, not only
  aggregate status counts.
- Require Sprint 7 audit memo status claims to agree with normalized ledger
  rows.

### S7-A4 - Same-Scope Comparator Decision Row Is Not KR Authority

Severity: Medium.

`SAME-SCOPE-COMPARATOR-DECISION` is marked
`existing_kr_source_supported`, but it cites an in-repo decision memo rather
than a KR scientific source. A decision memo can lock project scope; it cannot
be scientific authority.

Correct state:

- The scope memo is a governance/control-plane artifact.
- Same-scope comparator evidence remains blocked until a selected scope has
  source packets for the required observables.

Required correction:

- Change the row to a control-plane status such as
  `scope_governance_decision_pending`.
- Keep it out of scientific source-supported counts.

### S7-A5 - Traceability Exports Drifted

Severity: High.

`docs/DPF_REQUIREMENTS_BASELINE.md` and `docs/SRS_TRACEABILITY_MATRIX.csv`
contain Sprint 7 geometry wording, but `docs/SRS_TRACEABILITY_MATRIX.json`
still has stale rows for `DPF-PHYS-014` and `DPF-PHYS-022`. The current tests
render RTM exports to a temporary directory and do not compare committed
exports against a fresh render.

Required correction:

- Regenerate both committed RTM exports.
- Add a read-only drift test that compares a fresh export to
  `docs/SRS_TRACEABILITY_MATRIX.csv` and
  `docs/SRS_TRACEABILITY_MATRIX.json`.
- Ensure `DPF-PHYS-022` evidence refs include the Scholz extraction doc and
  the Stepniewski 0.015 m simulation-context caveat.

### S7-A6 - Findings Docs Tail Is Incomplete

Severity: Medium.

`CodexFindings.md` and `CortexFindings.md` top summaries mention Sprint 7
WS-B through WS-D, but the latest dated tail entry records only WS-A. Readers
using the tail as the current ratchet will miss geometry, 3-D readiness, and
Te/Ti rejection changes.

Required correction:

- Add dated tail entries for the Sprint 7 multi-agent audit and Super-Sprint 8
  handoff.
- Link this audit and the Super-Sprint 8 instructions.

### S7-A7 - Acceptance Channel Internals Are Contradictory

Severity: Medium.

Some packets can mark a channel as accepted while also unconditionally adding
all required channels back into `missing_acceptance_channels`. This is safe
today because top-level acceptance remains false, but it will block future
coherent acceptance.

Observed surfaces:

- `src/dpf/first_principles/same_scope.py`
- `src/dpf/first_principles/numerical_fidelity.py`
- `src/dpf/first_principles/certificate_gate.py`

Required correction:

- Replace the current mixed `accepted` plus unconditional `missing` pattern
  with per-channel states:
  `accepted`, `blocked_missing_source`, `blocked_wrong_scope`,
  `blocked_missing_review`, `blocked_missing_uncertainty`,
  `excluded_not_validated`, and `not_claimed`.
- Certificate-level acceptance may pass only when every claimed required
  channel is accepted and every excluded channel is explicitly excluded from
  the claim.

### S7-A8 - Manual Same-Scope Channel Injection Can Mislead

Severity: Low.

Te/Ti manual injection is rejected, but non-Te/Ti entries in
`accepted_same_scope_channels` can still appear as `accepted_same_scope` inside
the packet while the packet remains blocked.

Required correction:

- Treat manual same-scope channels as `requested_manual_channel_not_evidence`
  unless they are backed by a validation target with review and uncertainty.
- Keep manual channels useful for tests and synthetic fixtures only when the
  packet declares `synthetic_fixture=true`.

### S7-A9 - 12 us Whole-Shot State Is Engineering-Candidate Only

Severity: High for product claims; expected for Sprint 7.

The system can run candidate whole-shot probes. The prior dossier records a
conservative seeded-domain 12 us candidate run, but source-sign 12 us operation
did not produce a practical completed artifact. This is not a reviewed
first-principles whole-shot simulation.

Remaining runtime blockers:

- reviewed same-scope PF-1000 mask;
- hollow anode bore length and active hollow mask;
- insulator wall thickness and backplate dimensions;
- accepted startup BVP;
- Sigma-p face/operator closure and time-centered power residual;
- source-sign 12 us artifact;
- convergence, restart, backend parity, limiter-zero proof;
- same-scope waveform/UQ, spatial diagnostics, neutron mechanism authority,
  and certificate gates.

## Verification Run By Codex

Passed:

```bash
.venv312/bin/python -m pytest tests/test_first_principles_v2_handoff_ledgers.py tests/test_source_geometry_packet.py tests/test_first_principles_runner.py tests/test_cli_first_principles_3d.py tests/test_first_principles_certificate_negative_controls.py -q
```

Result: `158 passed in 26.73s`.

Passed:

```bash
.venv312/bin/python scripts/verify_first_principles_source_truth_exhaustion.py --date 2026_05_20 --check
```

Result: `exhausted=true`, `open_issue_count=0`.

Passed:

```bash
.venv312/bin/python scripts/verify_first_principles_module_source_vetting.py --date 2026_05_20 --check
```

Result: `strict_passed=true`, `total_modules=293`.

Important caveat: the focused test suite passes, but S7-A2 and S7-A3 show that
some tests currently enforce stale ledger state. Passing tests are therefore
not sufficient for Sprint 7 acceptance without the required corrections.

## Audit Decision

Accept Sprint 7 with corrections. Do not start accepted-physics promotion from
Sprint 7 artifacts until Super-Sprint 8 Workstream 0 is complete.

