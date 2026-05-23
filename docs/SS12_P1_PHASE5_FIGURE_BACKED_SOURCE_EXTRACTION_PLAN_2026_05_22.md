# SS12 P1 Phase 5 Figure-Backed Source Extraction Plan

> **For Hermes:** Use subagent-driven-development or direct TDD execution task-by-task. Keep every figure-derived value staged, not accepted.

**Goal:** Build a controlled, reproducible workflow for extracting PF-1000 figure-backed current, density, EM-field, and neutron candidates from local source files while preserving fail-closed acceptance boundaries.

**Architecture:** Phase 5 adds extraction manifests and validators first, then optional digitized candidate packets. Local `KnowledgeReference` and `/pdfs` remain evidence authority; HeliosMatrix_KB retrieval is discovery support only. Every extracted number must carry figure id, source path, line range, extraction method, digitization hash, uncertainty, scope classification, reviewer, and review state.

**Tech Stack:** Python 3.12, pytest, ruff, pathlib/json, existing `figure_candidate_staging.py`, local markdown/PDF corpus.

---

## Source candidates from Phase 3/4

Initial high-value local sources:

1. `KnowledgeReference/recent-progress-in-1-mj-plasma-focus-research-d3e51f6c.md`
   - Target: PF-1000 Rogowski / dI/dt / PIN traces.
   - Known figure candidate: Fig. 6 region referenced around source lines 169-178.

2. `KnowledgeReference/scholz-2006-pf1000-mega-joule.md`
   - Target: computed plasma density distributions, current evolution, PF-1000 operation summaries.

3. `KnowledgeReference/experimental-study-of-the-structure-of-the-plasma-current-sheath-on-the-pf-1000-facility-705bcc83.md`
   - Target: magnetic-probe / dB/dt / current-sheath structure candidates.

## Non-negotiable acceptance boundary

Phase 5 outputs are staged candidates only.

Forbidden:

- `accepted_*_claim=true`
- `promotes_acceptance=true`
- `can_support_first_principles_acceptance=true`
- using transfer candidates as same-source accepted targets
- accepting figure-derived values without independent review certificate

## Task 1: Add figure-source manifest schema test

**Objective:** Define the JSON shape for figure-source extraction manifests.

**Files:**

- Create: `tests/test_ss12_phase5_figure_source_manifest.py`
- Create later: `docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json`

**Test requirements:**

- manifest has `manifest_id`, `validation_scope`, `acceptance_boundary`, and `figure_sources`;
- acceptance boundary flags are all false;
- each figure source has source path, line range, figure id, channel, scope classification, extraction priority, and review state;
- source path resolves under repo root;
- line range exists;
- no row has accepted/reviewed-as-accepted status.

**Command:**

```bash
.venv312/bin/python -m pytest tests/test_ss12_phase5_figure_source_manifest.py -q
```

Expected RED: manifest missing.

## Task 2: Create minimal figure-source manifest

**Objective:** Add a manifest of candidate figures without numeric extraction.

**Files:**

- Create: `docs/SS12_P1_PHASE5_FIGURE_SOURCE_MANIFEST_2026_05_22.json`

**Required rows:**

- current waveform / Rogowski candidate from `recent-progress...` Fig. 6;
- density distribution candidate from `scholz-2006-pf1000-mega-joule.md`;
- magnetic-probe / EM-field candidate from `experimental-study...pf-1000...md`;
- neutron timing/spectrum candidate only if line-citable local evidence is found, otherwise explicit blocked row.

**Command:**

```bash
.venv312/bin/python -m pytest tests/test_ss12_phase5_figure_source_manifest.py -q
```

Expected GREEN.

## Task 3: Add manifest validator script

**Objective:** Provide a CLI validator equivalent to the Phase 2 source-packet validator, but scoped to figure-source candidates.

**Files:**

- Create: `scripts/validate_ss12_phase5_figure_source_manifest.py`
- Extend test: `tests/test_ss12_phase5_figure_source_manifest.py`

**Behavior:**

- reads manifest path;
- checks required fields;
- checks source path and line range;
- checks all acceptance flags false;
- fails if candidate status implies accepted;
- emits JSON summary with `passed`, `issue_count`, and issue list.

**Command:**

```bash
.venv312/bin/python -m pytest tests/test_ss12_phase5_figure_source_manifest.py -q
.venv312/bin/python scripts/validate_ss12_phase5_figure_source_manifest.py --repo-root .
```

## Task 4: Add staged candidate packet builder test

**Objective:** Convert manifest rows into `stage_figure_observable_candidate(...)` packets without extracting numeric values yet.

**Files:**

- Create: `tests/test_first_principles_phase5_figure_packet_builder.py`
- Create later: `src/dpf/first_principles/figure_source_manifest.py`

**Behavior:**

- loads manifest;
- maps rows to staged packets;
- every packet has `accepted_observable_claim=false`;
- transfer candidates remain blocked by `scope_not_same_source_accepted`;
- missing review certificate remains blocker.

## Task 5: Implement figure packet builder

**Objective:** Add the minimal loader/builder for manifest rows.

**Files:**

- Create: `src/dpf/first_principles/figure_source_manifest.py`

**Command:**

```bash
.venv312/bin/python -m pytest tests/test_first_principles_phase5_figure_packet_builder.py -q
```

Expected GREEN.

## Task 6: Integrated Phase 5 verification

**Objective:** Prove Phase 5 manifest and staged packets do not disturb Phase 4 shields.

**Command:**

```bash
.venv312/bin/python -m pytest \
  tests/test_ss12_phase5_figure_source_manifest.py \
  tests/test_first_principles_phase5_figure_packet_builder.py \
  tests/test_first_principles_acceptance_shield_phase4d.py \
  tests/test_first_principles_figure_candidate_phase4c.py \
  tests/test_first_principles_circuit_power_port_phase4b.py \
  tests/test_first_principles_numerical_fidelity_phase4a.py -q

ruff check \
  scripts/validate_ss12_phase5_figure_source_manifest.py \
  src/dpf/first_principles/figure_source_manifest.py \
  tests/test_ss12_phase5_figure_source_manifest.py \
  tests/test_first_principles_phase5_figure_packet_builder.py
```

## Task 7: Evaluate / learn / continue report

**Objective:** Write Phase 5 ELC report and decide whether to digitize figures or continue source search.

**Files:**

- Create: `docs/SS12_P1_PHASE5_FIGURE_SOURCE_EXTRACTION_EVALUATE_LEARN_CONTINUE_2026_05_22.md`

**Required conclusion:**

- candidate figures staged;
- no numeric target accepted;
- blockers listed by channel;
- next step is digitization with reproducible hash/uncertainty or source search for missing same-scope channels.
