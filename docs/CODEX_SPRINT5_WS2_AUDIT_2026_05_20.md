# Codex Audit: Sprint 5 WS2 Target Extractions And X-Ray Pass

Date: 2026-05-20

Audited HEAD: `558de6fc0ed1689a58fec84ed7621fb4d3d92fbd`

Branch state observed by Codex: `codex/corpus` ahead of `origin/codex/corpus`
by 72 commits; worktree clean.

## Verdict

Sprint 5 WS2 is accepted as a fail-closed source-availability and target-
extraction pass, not as runtime physics acceptance.

The team's central safety claim is true: the new extraction packets,
acquisition memo, x-ray docstring fixes, and regenerated module-vetting outputs
do not promote a first-principles runtime claim. Every audited packet still
reports `accepted_runtime_claim = false` and
`can_support_first_principles_acceptance = false`.

## Independent Checks

Codex ran:

```text
.venv312/bin/python -m pytest \
  tests/test_sprint5_target_extractions.py \
  tests/test_first_principles_physics_acceptance_protocol.py \
  tests/test_first_principles_v2_handoff_ledgers.py \
  tests/test_external_team_submission_package.py -q
```

Result: `52 passed in 1.11s`.

Codex also reviewed the latest periodic audit record:
`/private/tmp/dpf-unified-audit-logs/20260520T161836Z/summary.md`.

Result: 10/10 periodic audit gates passed at
`558de6fc0ed1689a58fec84ed7621fb4d3d92fbd`.

## Confirmed Work

1. `src/dpf/first_principles/sprint5_target_extractions.py` contains the seven
   claimed Sprint 5 WS2 packets and retains fail-closed acceptance flags.
2. `tests/test_sprint5_target_extractions.py` contains the claimed 17 tests and
   exercises the row-6/7/8 corrections plus the negative findings.
3. The Braginskii 1965 PDF exists at
   `archive_reference_OLD/references/papers/mhd-numerics/braginskii_1965.pdf`.
   Codex independently rendered PDF page 26 and verified that journal page 251
   contains Table 2 with the Z-dependent coefficient columns and the spot-check
   families encoded in the packet. This closes the previous OCR/layout concern,
   but does not make the PDF a promoted KR target extract.
4. The Bennett 2017 PDF exists at
   `archive_reference_OLD/references/papers/core-dpf/schmidt-2017-kinetic-dpf-breakdown.pdf`.
   Text extraction confirms the filename mislabel: the paper is Bennett et al.,
   Phys. Plasmas 24, 062705 (2017), DOI `10.1063/1.4985313`.
5. The UCSD/Beg line-range correction is supported by the local KR file
   `KnowledgeReference/effect-of-current-sheath-initiation-on-the-radial-collapse-and-energetic-particle-accelera-b2e95b88.md`.
   The cited ranges cover the mass-sweeping fit, Paschen-regime caveat, few-eV
   startup temperature assumption, and `Liz/Li` ratio context.
6. The Scholz/Gribkov PF-1000 full-energy neutron anisotropy extraction is
   supported by `KnowledgeReference/scholz-2007-pf1000-part2-jphysd.md`.
7. Bernard 1977 remains correctly classified as historical/wrong-scope
   filament-phase Ti evidence, and the thermonuclear `1/4` volumetric prefactor
   remains not found there.
8. The x-ray docstring edits in `closure_packet.py`, `neutron_authority.py`, and
   `startup_bvp.py` are consistent with the fail-closed posture.

## Findings To Fix Before Relying On WS3

### A1 - Bennett Packet Has A CH01 Mapping Ambiguity

Severity: Medium

`BENNETT_2017_STARTUP_EXTRACTION["resolves_blockers"]` lists only
`STARTUP-BVP-CH03`, `CH04`, `CH07`, and `CH08`. Inside the `targets` map,
`fill_pressure_baseline` lists `resolves = ("STARTUP-BVP-CH01",)` while also
calling the target "corroborative only".

This is not a physics acceptance failure because the whole packet remains
fail-closed. It is still a bookkeeping defect: a downstream extractor could
incorrectly infer that Bennett closes CH01.

Required fix:

- Either remove `STARTUP-BVP-CH01` from the per-target `resolves` tuple and keep
  it as a note-only/corroborative target, or add an explicit
  `corroborative_only = true` field and tests proving per-target
  corroborative blockers do not enter `resolves_blockers`.

Acceptance test:

- Add a test that the union of non-corroborative per-target `resolves` is a
  subset of the top-level `resolves_blockers`.

### A2 - Te/Ti Gap Wording Is Still Too Broad In The New Memo

Severity: Medium

`docs/SPRINT5_FREE_ACQUISITIONS_2026_05_20.md` states that no DPF literature
publishes pinch-phase spectroscopic Te/Ti. That wording is broader than the
source-backed finding already recorded in the handoff plan.

The accepted narrow statement is:

- no accepted same-scope PF-1000 bulk pinch Te/Ti history exists for the
  selected certificate scope;
- Bernard 1977 contains direct historical filament-phase Ti evidence, but it is
  wrong-scope for PF-1000 pinch validation;
- Plasma Focus Update 2021 contains PF-1000 local hot-spot Te method context,
  but it is text-only/method context and not accepted as bulk same-scope Te
  validation.

Required fix:

- Replace the broad sentence in the Sprint 5 free-acquisition memo with the
  narrow statement above.

Acceptance test:

- Add a doc-lint or regression test that rejects broad
  `no DPF in any literature` Te/Ti wording outside explicitly scoped evidence
  discussions.

### A3 - Free Acquisition Memo Uses Some Closure Language Too Aggressively

Severity: Low/Medium

The memo is careful at the top that the URLs are acquisition pointers, but later
uses phrases like "closes blockers" for LXCat, SRIM, Munro, and PlasmaPy. Those
items may close source-availability gaps after acquisition and review; they do
not close blocker acceptance.

Required fix:

- Change "closes" wording in `docs/SPRINT5_FREE_ACQUISITIONS_2026_05_20.md` to
  "may close source availability after acquisition, KR ingestion, target
  extraction, and review".
- Keep SRIM, Munro, and PlasmaPy labeled as substitutes or cross-checks, not
  source-equivalent replacements, until source review accepts the substitution.

### A4 - Handoff Count Drift

Severity: Low

The handoff says "74 total unpushed"; local Git reports the branch is ahead by
72 commits at the audited HEAD. This does not affect physics, but it should be
corrected before another team treats the summary as a release manifest.

## Structural Blockers Remain

The team's two structural blockers are real in the codebase:

1. `src/dpf/validation/first_principles_mhd.py` still only treats the
   non-fallback `python*` backend path as acceptance-eligible for the legacy MHD
   readiness gate.
2. `src/dpf/first_principles/same_scope.py` still forces blocking same-scope
   channels into the missing channel set.

However, the accepted resolution must remain claim-limited:

- do not add a generic `caveat_accepted` Te/Ti state;
- do add explicit observable exclusion only where the certificate clearly says
  the excluded observable is not validated;
- do not count excluded Te/Ti channels as same-scope comparator evidence.

## Next Direction For The Other Team

The next team sprint should proceed only after A1-A3 are corrected or carried
as explicit sprint preconditions.

Recommended order:

1. Fix the Bennett CH01 mapping ambiguity and Te/Ti/free-acquisition wording.
2. Acquire the three directly free Nukleonika PDFs and ingest them into KR as
   fail-closed source records.
3. Target-extract Braginskii Table 2 from the locally rendered PDF into a KR
   record, including render artifact path/hash or an equivalent reproducible
   evidence trail.
4. Treat LXCat, SRIM/NIST/IAEA, Munro, and PlasmaPy as candidate substitution
   or cross-check lanes until Codex and the external team both accept the
   source-equivalence argument.
5. Keep all runtime acceptance flags false until source packet, code
   consumption, numerical acceptance, same-scope comparator, and certificate
   gates all pass at the same commit.
