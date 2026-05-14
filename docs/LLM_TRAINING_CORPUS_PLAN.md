# LLM Training Corpus Plan

Date: 2026-05-14

Status: planning baseline; no corpus records generated yet

Scope: build `data/llm_corpus/training.jsonl` from local `KnowledgeReference/`
papers, books, source-fidelity sidecars, and accepted repository evidence. This
file defines the training corpus only. Evaluation, holdout, and validation
prompts belong in `docs/LLM_VALIDATION_CORPUS_PLAN.md`.

## Provenance Basis

- The local source authority is `KnowledgeReference/`; the root agent contract
  forbids external sources for DPF physics claims. [KB: AGENTS.md]
- The current corpus has 513 Markdown records under `KnowledgeReference/` at
  max depth 2, 517 JSON sidecars, and a reference image index. [KB:
  KnowledgeReference inventory, 2026-05-14]
- A prior intake report promoted 32 papers into `KnowledgeReference/`, skipped
  59 already represented records, and explicitly warned that text parity does
  not accept figures, tables, plotted curves, numeric targets, or validation
  claims. [KB: docs/RESEARCH_PAPERS_KR_PROMOTION_2026_05_11.md]
- A source-fidelity audit found 91 checked intake records, 10,767 recovered
  secondary-extraction items, 14,554 formula-like lines, 9,533 numeric target
  contexts, and 2,143 uncertainty contexts, but it also says this is source-copy
  fidelity only, not validation acceptance. [KB:
  docs/KR_SOURCE_FIDELITY_AUDIT_2026_05_11.md]
- Together AI currently accepts JSONL text datasets with `messages`,
  `prompt`/`completion`, or `text` fields, and a validation file can be supplied
  separately at job creation. [DOC: Together AI fine-tuning data preparation
  2026-05-14] [DOC: Together AI fine-tuning CLI 2026-05-14]

## Goal

Produce a supervised fine-tuning corpus that teaches the model how to answer,
classify, extract, and cite within the DPF-Unified evidence regime:

- retrieve from local sources before asserting physics;
- distinguish source text, candidate extraction, accepted validation evidence,
  engineering probes, synthetic-only data, and non-validation scaffolding;
- answer with explicit provenance tags and uncertainty limits;
- refuse or mark unsupported physics claims instead of inventing values;
- preserve the PF-1000/Akel, Lee/snowplow, MHD, diagnostics, radiation,
  pulsed-power, and first-principles vocabulary already used by the repo.

The training corpus should improve style, routing, extraction discipline, and
domain terminology. It must not train the model to treat unreviewed source text
as accepted validation evidence.

## May 2026 Best-Practice Update

The target architecture is hybrid, not "stuff every paper into SFT and hope the
model remembers." Use four separate adaptation layers:

1. Retrieval layer: index the corpus for inference-time lookup with local
   source paths, hashes, page/line spans, evidence states, and source-scope
   metadata.
2. Domain adaptive pretraining layer: locally continue training on permitted
   source text so the model internalizes DPF/plasma terminology, equations,
   acronyms, and prose patterns. Keep this local unless license review clears
   cloud use.
3. RAG-aware SFT layer: train instruction examples where the model receives
   retrieved source snippets plus distractors, answers from the useful snippets,
   cites the source, and rejects unsupported prompts.
4. Preference/alignment layer: train or score preferred grounded answers against
   dispreferred hallucinated, cross-scope, uncited, or over-promoted answers.

This split reflects the current evidence from provider docs and 2026 domain
adaptation studies:

- Together's May 2026 docs say fine-tuning is appropriate when prompting alone
  fails, when labeled examples exist, or when the model needs domain terminology
  and output formats; they still recommend RAG first when the need is only
  factual grounding. [DOC:
  https://docs.together.ai/docs/fine-tuning-overview 2026-05-14]
- Together supports JSONL `messages`, `prompt`/`completion`, `text`,
  preference, tool-call, reasoning, and tokenized Parquet formats. Use JSONL for
  the first pass; reserve Parquet/tokenized data for custom masking, repeated
  experiments, or custom tokenizer work. [DOC:
  https://docs.together.ai/docs/fine-tuning-data-preparation 2026-05-14]
- Together's March 2026 update added tool-call, reasoning, and vision-language
  fine-tuning support and expanded dataset scale, so future corpus rows can
  include retrieval/tool-call behavior once the local retriever API is stable.
  [DOC: https://www.together.ai/blog/fine-tuning-update 2026-05-14]
- Axolotl's current dataset guide separates pretraining, SFT, and
  preference-based post-training. Mirror that separation locally so DAPT text,
  chat/instruction SFT, and DPO-style preference pairs are not mixed into one
  ambiguous file. [DOC: https://docs.axolotl.ai/docs/dataset-formats/
  2026-05-14]
- NeMo's current SFT/PEFT guide keeps full-parameter SFT separate from
  parameter-efficient tuning and notes that PEFT can approach full-SFT accuracy
  with much lower hardware cost. Default local experiments should start with
  LoRA/QLoRA before full fine-tuning. [DOC:
  https://docs.nvidia.com/nemo-framework/user-guide/24.07/sft_peft/index.html
  2026-05-14]
- RAFT trains the model in an open-book setting with retrieved documents and
  distractors, explicitly teaching it to ignore irrelevant retrieved text and
  cite the useful document. That pattern maps directly onto this corpus.
  [DOC: https://arxiv.org/abs/2403.10131 2026-05-14]
- A 2026 engineering-domain study found RAG beat baseline strongly while naive
  raw-text fine-tuning reduced performance on most expert-judged answers. Treat
  raw-text DAPT as terminology adaptation, not as a replacement for retrieval.
  [DOC: https://arxiv.org/abs/2605.12516 2026-05-14]

For this project, that means the local model should learn the source corpus in
two ways:

- Parametric knowledge: local-only DAPT on cleaned, deduplicated source text and
  equation-heavy chunks, with short training runs and regression evals to catch
  forgetting or overconfidence.
- Non-parametric knowledge: RAG at inference using the exact current corpus,
  because accepted evidence state, digitization review status, and source
  manifests change over time.

The model should be rewarded for saying "I know the concept, but I need the
source span before claiming this value." That is not hesitation. That is the
machine acting like a scientist.

## Target Model Architectures

### A. Open-Source DPF-Focused LLM

Purpose: a public DPF specialist that understands DPF terminology and operating
discipline, but still retrieves local evidence before making source-specific
scientific claims.

Training stack:

1. `DPF-DAPT`: local domain adaptive pretraining on cleaned,
   license-reviewed DPF/plasma/pulsed-power text chunks.
2. `DPF-SFT`: supervised rows for source-grounded Q/A, evidence
   classification, target extraction, formula mapping, refusal/gap handling,
   and RAFT-style retrieved-context answers.
3. `DPF-STYLE`: the user's 500 personality/style/tone Q/A examples, converted
   into a separate style lane and optionally a separate LoRA adapter.
4. `DPF-PREF`: preference pairs where the chosen answer is cited, scoped,
   evidence-state-aware, and stylistically aligned, while rejected answers are
   uncited, overconfident, cross-scope, or bland.
5. `DPF-RAG`: required runtime retrieval over the current `KnowledgeReference/`
   index and project evidence manifests.

Release shape:

- open-source base recipe, corpus builder, validators, manifests, and adapter
  config;
- publish only source rows and derived examples that pass license/privacy gates;
- ship the model with a default system policy: "retrieve before source-specific
  claims; cite local source paths; mark missing evidence instead of guessing."

### B. Repository-Grounded Science/Engineering LLM

Purpose: a reusable model pattern for any user-provided science/engineering
repository. The model should not search the web by default, ingest poisoned web
material, or rely on parametric memory for repository-specific conclusions.

Training stack:

1. `REPO-CATALOG`: inventory user files, hashes, licenses, source type,
   trust tier, and metadata.
2. `REPO-INDEX`: build local lexical, vector, and metadata-filtered retrieval
   over the user's repository.
3. `REPO-SFT`: train source-use behavior on generic science/engineering tasks:
   retrieve, cite, extract, compare, classify, refuse, and report uncertainty.
4. `REPO-TOOLS`: optional tool-call fine-tuning for `repo_search`,
   `repo_open`, `repo_extract_table`, and `repo_cite`.
5. `REPO-EVAL`: score retrieval recall, citation fidelity, unsupported-claim
   refusal, conflict detection, and source-trust handling.

Default safety rule:

- no web retrieval unless the user explicitly enables it for acquisition
  planning;
- repository sources are cited by file hash/path/span;
- model memory may explain stable background concepts, but repository-specific
  claims require retrieval.

The two architectures share tooling. The DPF model is the first domain-specific
implementation; the science/engineering model is the generalized product
pattern.

## Non-Goals

- Do not upload raw copyrighted book or paper text to Together AI unless license
  review permits it. Cloud training should use derived Q/A, short excerpts within
  lawful limits, metadata, and source pointers by default.
- Do not include validation holdout questions in the training file.
- Do not use generated simulation artifacts as scientific truth labels.
- Do not train hidden chain-of-thought. Use concise rationales, citation
  checklists, and observable evidence steps instead.
- Do not include secrets, local absolute private paths in cloud files, API keys,
  user notes, or unredacted operational memory.

## Corpus Lanes

| Lane | Purpose | Source | Training shape | Inclusion status |
| --- | --- | --- | --- | --- |
| `domain_qa` | Teach DPF/plasma explanations with source-scoped answers | DPF core papers, textbooks, formula sources | conversational JSONL | Include after excerpt/citation review |
| `evidence_classification` | Teach accepted vs candidate vs blocked vs probe labels | `CodexFindings.md`, `CortexFindings.md`, pipeline docs, sidecars | instruction JSONL | Include |
| `target_extraction` | Teach extraction of device, shot, units, observable, uncertainty, and scope | source-fidelity JSON sidecars and reviewed markdown windows | instruction JSONL with structured completion | Include only when source span is recorded |
| `formula_mapping` | Teach formula citation, symbol mapping, unit checks, and code mapping | NRL, Lee/RADPF, MHD/numerics sources | instruction JSONL | Include as draft unless independently accepted |
| `refusal_and_gap` | Teach fail-closed behavior when evidence is absent | blocker docs and missing-source cases | conversational JSONL | Include heavily |
| `style_and_ops` | Teach terse provenance-tagged reports | AGENTS contract, findings docs | conversational JSONL | Include |
| `personality_style_qa` | Teach personality, tone, and interaction style without adding scientific claims | user-provided 500 Q/A style set | conversational JSONL or separate LoRA | Include with science masking and holdout |
| `off_scope_filter` | Teach noise rejection for unrelated corpus items | climate, biomedical, generic AI, unrelated math records | classification JSONL | Include small balanced sample |

## Source Stratification

Build the training set by stratified sampling rather than dumping all text:

| Stratum | Current local signal | Training priority |
| --- | ---: | --- |
| DPF core | 135 markdown files matched DPF/PF-1000/Akel/Lee/RADPF terms | Highest |
| Z-pinch and pinch physics | 146 matched Z-pinch, Bennett, instability, or shear-flow terms | High |
| MHD and numerical methods | 423 matched MHD, finite-volume, CT, WENO, ALE, ALEGRA, FLASH, or related numerics | High but deduplicate aggressively |
| Transport/collisions/radiation | 256 matched Spitzer, Braginskii, Coulomb, bremsstrahlung, ionization, opacity, conductivity | High for formula and source-blocker lessons |
| Diagnostics/neutrons/beams | 301 matched neutron, detector, anisotropy, nTOF, beam, x-ray, interferometer | High for validation and uncertainty prompts |
| Pulsed power/electrical | 247 matched capacitor, bank, circuit, inductance, breakdown, switch, transmission-line terms | Medium-high |
| ML/data-driven plasma | 61 matched ML/data-driven/surrogate/foundation-model terms | Medium, mostly method and dataset hygiene |
| Likely off-scope/noise | 126 matched broad unrelated terms | Low, use only for rejection training |

These counts are keyword triage, not review acceptance. Every generated record
must carry its exact source path and source span.

## Dataset Format

Canonical internal row:

```json
{
  "id": "train-domain-000001",
  "lane": "domain_qa",
  "messages": [
    {
      "role": "system",
      "content": "Answer only from provided local-source excerpts. Preserve evidence state."
    },
    {
      "role": "user",
      "content": "What can the local corpus support about PF-1000 neutron timing?"
    },
    {
      "role": "assistant",
      "content": "Answer with [KB: ...] tags, uncertainty, and blocker status."
    }
  ],
  "metadata": {
    "source_paths": ["KnowledgeReference/scholz-2006-pf1000-mega-joule.md"],
    "source_spans": [{"path": "KnowledgeReference/scholz-2006-pf1000-mega-joule.md", "start_line": 1, "end_line": 220}],
    "evidence_state": "source_available_not_target_extracted",
    "license_gate": "local_ok_cloud_review_required",
    "cloud_safe": false,
    "split": "train",
    "created_by": "build_llm_corpus.py",
    "created_at": "2026-05-14"
  }
}
```

Together-compatible projection:

```json
{"messages":[{"role":"system","content":"Answer only from provided local-source excerpts. Preserve evidence state."},{"role":"user","content":"What can the local corpus support about PF-1000 neutron timing?"},{"role":"assistant","content":"Answer with [KB: ...] tags, uncertainty, and blocker status."}]}
```

Keep the rich canonical record locally. Export stripped provider files only after
hashing and storing a manifest that maps each stripped row back to the local
canonical row.

Personality/style row:

```json
{
  "id": "style-000001",
  "lane": "personality_style_qa",
  "messages": [
    {
      "role": "system",
      "content": "Answer in the target assistant style. Do not make scientific claims unless sources are provided."
    },
    {
      "role": "user",
      "content": "User's style/tone training question."
    },
    {
      "role": "assistant",
      "content": "User-approved style/tone answer."
    }
  ],
  "metadata": {
    "source_paths": [],
    "evidence_state": "style_only_not_scientific_evidence",
    "cloud_safe": "review_required",
    "split": "train"
  }
}
```

Style rows must never carry `[KB:]`, `[PAPER:]`, or accepted-evidence labels
unless the user-provided answer also includes valid source metadata. Their job
is voice, not truth.

## Record Types

### 0. Domain Adaptive Pretraining Text

Use only for local continued pretraining or cloud-cleared generic text training:

```json
{"text": "Cleaned source chunk with document title, section heading, page/line metadata, and no answer label."}
```

Rules:

- keep by-source hash split isolation;
- deduplicate near-identical chunks;
- preserve equations, units, tables, and captions only when extraction quality is
  acceptable;
- mask or remove bibliography-only, copyright-page, OCR-garbage, and unrelated
  chunks;
- do not train the model to answer from this row format; this lane is for
  terminology and latent domain familiarity only.

### 1. Source-Grounded Q/A

Use short local excerpts as context and ask narrow questions:

- "According to the provided local source, what phase or observable is
  discussed?"
- "Which claim can be made, and which claim remains unsupported?"
- "What would need independent review before this can validate the simulator?"

Assistant answers must include:

- a direct answer;
- source path tag;
- evidence state;
- explicit unsupported items.

### 2. Structured Target Extraction

Prompt with a source window and require JSON output:

```json
{
  "observable_group": "neutron",
  "observable_name": "yield",
  "device": "PF-1000",
  "shot": "unknown",
  "raw_value": "3.5e11",
  "raw_units": "neutrons/shot",
  "source_path": "KnowledgeReference/scholz-2006-pf1000-mega-joule.md",
  "evidence_state": "typed_target_draft",
  "validation_support": false,
  "blockers": ["independent_review_missing", "uncertainty_missing", "same_scope_comparator_missing"]
}
```

Only use values when the generator records the exact source line range. If the
line range is unknown, generate a refusal/gap record instead.

### 3. Evidence-State Classification

Create examples where the correct label is one of:

- `missing`
- `candidate`
- `blocked_by_review`
- `accepted`
- `engineering_probe`
- `synthetic_only`
- `not_validation_evidence`

Use active blocker examples from findings and pipeline docs so the model learns
that "text exists" is weaker than "validated evidence exists."

### 4. Formula Mapping

Use formula windows from local sources, then ask for:

- equation transcription in ASCII;
- symbol map;
- unit/dimensional check;
- source citation;
- validation state.

Do not include formula records in cloud training until copyright and excerpt
length are reviewed.

### 5. Refusal And Gap Examples

These are mandatory. At least 20 percent of training rows should teach the model
to say:

- "unsupported by local source";
- "source exists but target extraction is not accepted";
- "engineering probe only";
- "same-scope evidence missing";
- "cloud upload blocked pending license review."

This is the antidote to confident nonsense. Small mercy.

### 6. Retrieval-Augmented Fine-Tuning Rows

Use RAFT-style rows for the main "retrieve and use it" behavior:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Use only the retrieved local-source snippets. Ignore distractors. Cite source paths."
    },
    {
      "role": "user",
      "content": "Question: Can this source validate PF-1000/Akel neutron yield?\n\nRetrieved snippets:\n[1] useful local source span...\n[2] distractor from another device or off-scope source...\n[3] candidate-only source span..."
    },
    {
      "role": "assistant",
      "content": "No. The relevant source is candidate/source-available only and lacks accepted same-scope UQ and comparator binding. [KB: KnowledgeReference/...]"
    }
  ],
  "metadata": {
    "lane": "raft_rag",
    "useful_sources": ["KnowledgeReference/..."],
    "distractor_sources": ["KnowledgeReference/..."],
    "evidence_state": "candidate"
  }
}
```

Ratios for the first full build:

- 30 percent source-grounded Q/A;
- 20 percent RAFT-style retrieved-context rows;
- 20 percent refusal/gap/evidence-state rows;
- 15 percent structured extraction/formula mapping;
- 5-10 percent personality/style rows;
- 5-10 percent tool-call or retriever-routing rows after the retriever CLI/API
  is stable.

### 7. Retriever Tool-Use Rows

For models that support tool-call fine-tuning, add rows where the model must
call a retrieval tool before answering. Keep this lane out of the first cloud
upload until the local retrieval contract is stable.

Expected tool schema:

```json
{
  "name": "kr_search",
  "arguments": {
    "query": "PF-1000 Akel current waveform uncertainty",
    "filters": {
      "domain": "dpf",
      "evidence_state": ["accepted", "blocked_by_review", "candidate"]
    },
    "top_k": 8
  }
}
```

Training objective:

- choose retrieval before answering source-specific questions;
- use metadata filters when scope matters;
- refuse if retrieval returns no same-scope evidence;
- cite returned source paths only.

### 8. Personality/Style Q/A Rows

Use the user's 500 Q/A examples to teach tone, conversational posture, and
interaction style. Keep them separate from source-grounded science rows.

Processing rules:

- store the raw import at `data/llm_corpus/style/personality_qa_raw.jsonl`;
- normalize into `data/llm_corpus/style/personality_qa.canonical.jsonl`;
- tag every row as `style_only_not_scientific_evidence`;
- remove or rewrite any answer that asserts uncited physics, medical, legal, or
  safety-critical claims;
- split by stable hash into train/dev/validation, for example 400/50/50;
- optionally train the style set as a separate adapter so it can be merged,
  weighted, disabled, or revised without retraining the science behavior.

Recommended use:

- first pass: include a light 5-10 percent style mixture in SFT batches;
- safer pass: train a separate style LoRA on the 400-row train split and merge
  it with the DPF behavior adapter only after validation;
- never let style loss dominate refusal, citation, or evidence-state behavior.

## Split Policy

- Use deterministic document-level splitting, not random row-level splitting.
- Hold out entire source documents for validation to prevent source leakage.
- Keep all rows derived from one source hash in one split.
- Reserve PF-1000/Akel accepted-or-candidate evidence and all active blockers
  for validation unless a duplicate training version is explicitly allowed.
- Suggested split: 80 percent training, 10 percent development, 10 percent
  validation by source hash, with additional handpicked validation stress sets.

## Local Build Pipeline

Target files:

- `scripts/build_llm_corpus.py`
- `data/llm_corpus/catalog.jsonl`
- `data/llm_corpus/training.canonical.jsonl`
- `data/llm_corpus/training.together.jsonl`
- `data/llm_corpus/training.local.jsonl`
- `data/llm_corpus/style/personality_qa.canonical.jsonl`
- `data/llm_corpus/manifests/training_manifest.json`

Pipeline:

1. Inventory all `KnowledgeReference/**/*.md` and `KnowledgeReference/**/*.json`
   records.
2. Pair markdown with JSON sidecars by stem.
3. Compute SHA-256 hashes for every input file.
4. Classify source domain with keyword, title, sidecar, and path features.
5. Reject exact duplicate hashes.
6. Segment text into source windows with line ranges.
7. Exclude bibliography-only, cover-page, OCR-noise, and license-problem
   windows.
8. Generate candidate rows by lane.
9. Import and normalize the 500 personality/style Q/A rows.
10. Run deterministic split by source hash or row hash.
11. Validate every row against schema, provenance, size, and leakage rules.
12. Export canonical local JSONL and stripped Together JSONL.
13. Write manifest with source counts, row counts, token estimates, hashes,
    license gates, and split assignment.

## Validation Checks For The Training File

Run before using `training.together.jsonl` or `training.local.jsonl`:

```bash
python3 scripts/build_llm_corpus.py --input KnowledgeReference --out data/llm_corpus --split train
python3 scripts/validate_llm_corpus.py data/llm_corpus/training.canonical.jsonl --mode training
python3 scripts/validate_llm_corpus.py data/llm_corpus/training.together.jsonl --mode together
python3 -m py_compile scripts/build_llm_corpus.py scripts/validate_llm_corpus.py
```

Required checks:

- JSONL parses one object per line.
- Exactly one provider format is present per row: `messages`, `prompt` and
  `completion`, or `text`.
- Every canonical row has source path, source hash, source span, lane, split,
  cloud-safety flag, and evidence state.
- Training and validation splits share no source hash.
- No cloud export contains rows with `cloud_safe=false`.
- No row contains API keys, home-directory private paths, or hidden operational
  memory.
- No assistant answer cites a source not present in metadata.
- No answer uses `accepted` unless the referenced evidence packet is accepted.
- No `personality_style_qa` row is counted as scientific evidence.
- No style row overrides refusal, citation, or same-scope evidence behavior in
  the held-out validation set.

## Together AI Export Notes

Use `messages` for the main SFT corpus because the desired behavior is
instruction-following with citations and refusal discipline. Together's current
docs say conversational JSONL uses a `messages` field and that the service
formats conversations into the target model chat template when available. [DOC:
Together AI fine-tuning data preparation 2026-05-14]

Use separate `training.together.jsonl` and `validation.together.jsonl` files.
Together's current CLI supports local `--training-file` and `--validation-file`
paths and uploads them automatically. [DOC: Together AI fine-tuning CLI
2026-05-14]

Keep `--train-on-inputs` at the default/auto setting unless a deliberate
reasoning or extraction experiment needs prompt-token loss. For conversational
and instruction data, Together currently masks inputs by default under `auto`.
[DOC: Together AI fine-tuning CLI 2026-05-14]

## Acceptance Criteria

The training corpus is ready when:

- `training.canonical.jsonl` and provider projections pass schema validation.
- The manifest reports zero train/validation source-hash overlap.
- At least seven corpus lanes are represented.
- At least 20 percent of rows are refusal/gap/evidence-state examples.
- The 500-row personality/style set is imported, deduplicated, safety-reviewed,
  tagged as style-only, and split into train/dev/validation.
- Every row is source-traceable back to local `KnowledgeReference/` or repo
  status docs, unless it is explicitly tagged `style_only_not_scientific_evidence`.
- Cloud export contains only rows cleared by license and privacy gates.
- A 50-row manual audit finds no unsupported physics promotion.

## First Sprint

1. Create `scripts/build_llm_corpus.py` and catalog only; no generated Q/A yet.
2. Emit `data/llm_corpus/catalog.jsonl` with source hash, title, page count,
   domain tags, sidecar path, markdown path, and cloud gate.
3. Create `scripts/validate_llm_corpus.py` with schema, source-span, citation,
   and split-leak checks.
4. Import the 500 personality/style Q/A examples and produce a 400/50/50 split.
5. Generate 100 pilot canonical rows: 40 DPF Q/A, 20 extraction, 20 evidence
   classification, 20 refusal/gap.
6. Add 25 style rows to the pilot mix and verify they do not change scientific
   refusal/citation behavior.
7. Manually review all 125 pilot rows before scaling.
