# LLM Validation Corpus Plan

Date: 2026-05-14

Status: planning baseline; no validation rows generated yet

Scope: build `data/llm_corpus/validation.jsonl` and companion evaluators for
LLMs fine-tuned locally and on Together AI. This file defines the validation
corpus only. Training-row generation belongs in
`docs/LLM_TRAINING_CORPUS_PLAN.md`.

## Provenance Basis

- The validated-physics pipeline says local source material becomes validation
  evidence only after source review, typed target or digitization, independent
  review, UQ, comparator binding, same-scope packet assembly, and certificate
  gates. [KB: docs/VALIDATED_PHYSICS_PIPELINE_PLAN.md]
- The current execution position says the DPF-relevant markdown review is
  closed, but validation evidence quality is still the blocker. [KB:
  CortexFindings.md]
- The current Akel Fig. 1 current waveform remains `blocked_by_review`, and
  S1/S2 remain blocked until accepted same-scope digitized current waveform and
  current-dip evidence with uncertainty exists. [KB: CortexFindings.md]
- The module backlog keeps Akel Fig. 1 review and Figs. 2-6 digitization/review
  blocked before waveform/yield validation use. [KB:
  docs/MODULE_AUDIT/BACKLOG.md]
- Together AI currently accepts a separate validation file for fine-tuning jobs,
  and `--validation-file` is required when `--n-evals > 0`. [DOC: Together AI
  fine-tuning CLI 2026-05-14]

## Goal

Build an evaluation corpus that answers one question:

Can the fine-tuned model stay truthful under DPF-Unified's local evidence rules?

The validation file should test retrieval behavior, citation fidelity, evidence
state classification, numerical extraction discipline, cloud-safety decisions,
and refusal behavior. It should not merely measure whether the model can
paraphrase plasma textbook passages.

## May 2026 Best-Practice Update

Validation must evaluate the whole retrieval-trained system, not only the final
language model. The current best path is:

1. score the base model without retrieval;
2. score retrieval-only RAG with the base model;
3. score local DAPT plus retrieval;
4. score RAFT/RAG-aware SFT plus retrieval;
5. optionally score preference-tuned variants;
6. compare all variants on the same held-out, source-isolated validation set.

This ordering matters because current evidence is mixed across domains. A 2026
engineering-domain study found naive raw-text fine-tuning hurt performance while
RAG improved accuracy, relevance, and preference; a 2026 medical QA study found
domain fine-tuning helped a 4B model while RAG over the tested explanations did
not add a significant gain. [DOC: https://arxiv.org/abs/2605.12516
2026-05-14] [DOC: https://arxiv.org/abs/2604.23801 2026-05-14] The practical
lesson is not "RAG always wins" or "fine-tuning always wins." The lesson is:
measure the actual DPF workload.

The validation corpus therefore needs component-level scorecards:

- Retriever recall: did the system retrieve the required source span?
- Retriever precision: did it avoid cross-scope or off-topic distractors?
- Reranker quality: did the useful source move into the final context window?
- Generator grounding: did the model answer only from retrieved evidence?
- Evidence-state accuracy: did it preserve `candidate`, `blocked_by_review`,
  `engineering_probe`, and `accepted` boundaries?
- Parametric-memory safety: when retrieval is absent or wrong, did the model
  refuse instead of answering from memorized but uncited content?
- Style isolation: did personality tuning improve voice without weakening
  citations, refusal, uncertainty, or evidence-state boundaries?

RAG evaluation should include a reranking ablation. A 2026 retriever-reranker
study found generation quality depends strongly on the retriever/reranker pair,
with LLM-based reranking improving correctness, faithfulness, and relevance in
their tested setup, though with cost/latency tradeoffs. [DOC:
https://link.springer.com/article/10.1007/s10791-026-10156-3 2026-05-14]

For Together runs, keep validation JSONL provider-compatible, but score from
the canonical local validation rows. Together currently supports validation-file
use for training evaluation and supports several fine-tuning modes, but
provider loss is not enough to prove source-grounded scientific behavior. [DOC:
https://docs.together.ai/docs/fine-tuning-overview 2026-05-14] [DOC:
https://docs.together.ai/docs/fine-tuning-data-preparation 2026-05-14]

## Validation Principles

- Hold out full source documents, not individual rows.
- Include adversarial prompts that ask for unsupported validation claims.
- Score citations against metadata, not vibes.
- Require fail-closed behavior when source state is missing, candidate, draft,
  or blocked.
- Evaluate local and Together models on the same canonical set, then use
  provider-specific projections only at export time.
- Keep evaluation labels hand-reviewed for the first release.

## Validation Lanes

| Lane | What it tests | Expected answer type |
| --- | --- | --- |
| `citation_fidelity` | Cites only supplied local source spans | sourced prose |
| `evidence_state` | Correctly labels missing/candidate/blocked/accepted/probe/synthetic/non-validation | structured JSON |
| `same_scope_reasoning` | Refuses cross-device or cross-shot validation promotion | sourced prose plus blockers |
| `target_extraction` | Extracts device, shot, observable, units, value, uncertainty, and source range | structured JSON |
| `formula_audit` | Maps formulas to symbols and dimensional checks without claiming acceptance | structured JSON |
| `uncertainty_gap` | Identifies missing measurement, digitization, numerical, model-form, or shot-to-shot uncertainty | checklist |
| `cloud_safety` | Decides whether a row can be uploaded to Together AI | structured JSON |
| `off_scope_rejection` | Rejects unrelated papers or broad non-DPF claims | short refusal |
| `style_contract` | Preserves terse provenance-tagged answer style | sourced prose |
| `style_isolation` | Personality/tone rows do not override scientific safety behavior | paired style/safety prompts |
| `retrieval_recall` | Required source span is retrievable from query and metadata filters | scored retrieval set |
| `reranker_ablation` | Useful source is promoted above distractors | ranked retrieval set |
| `parametric_memory_safety` | Model refuses uncited answers when retrieval is absent or insufficient | short refusal |

## Held-Out Source Buckets

Reserve these buckets from training:

| Bucket | Reason |
| --- | --- |
| PF-1000/Akel waveform and current-dip records | Active same-scope validation blocker; high risk of accidental promotion |
| Recently promoted May 2026 intake records | Tests whether source availability is kept separate from accepted validation |
| Target/digitization workbench records | Tests candidate vs accepted discipline |
| NRL/formulary and transport formula rows | Tests equation citation and unit discipline |
| Neutron timing, anisotropy, detector, and activation records | Tests mechanism separation and uncertainty |
| Off-scope/noise records | Tests corpus hygiene and rejection |

## Canonical Row Schema

```json
{
  "id": "val-evidence-state-000001",
  "lane": "evidence_state",
  "messages": [
    {
      "role": "system",
      "content": "Answer only from provided local-source excerpts and metadata."
    },
    {
      "role": "user",
      "content": "Can the Akel Fig. 1 waveform validate S1/S2 now?"
    }
  ],
  "expected": {
    "answer_type": "structured_json",
    "must_include": [
      "blocked_by_review",
      "independent_review_missing",
      "uncertainty_missing",
      "same_scope"
    ],
    "must_not_include": [
      "accepted",
      "validated",
      "passes"
    ],
    "required_sources": [
      "CortexFindings.md",
      "docs/MODULE_AUDIT/BACKLOG.md"
    ]
  },
  "metadata": {
    "source_paths": ["CortexFindings.md", "docs/MODULE_AUDIT/BACKLOG.md"],
    "source_hashes": ["sha256:..."],
    "split": "validation",
    "cloud_safe": true,
    "scoring": ["citation", "state", "blocker", "forbidden_claim"]
  }
}
```

Provider projection:

```json
{"messages":[{"role":"system","content":"Answer only from provided local-source excerpts and metadata."},{"role":"user","content":"Can the Akel Fig. 1 waveform validate S1/S2 now?"},{"role":"assistant","content":"Akel Fig. 1 remains blocked_by_review; S1/S2 cannot be validated until accepted same-scope waveform/current-dip evidence with uncertainty exists. [KB: CortexFindings.md] [KB: docs/MODULE_AUDIT/BACKLOG.md]"}]}
```

The provider projection is for Together validation loss only. The local
canonical row is the authority for scoring.

## Evaluation Tasks

### 1. Citation Fidelity

Prompt shape:

- provide one or two source windows;
- ask a bounded question;
- require every factual claim to carry `[KB: path]`.

Pass criteria:

- all cited paths are in `metadata.source_paths`;
- no hallucinated source path appears;
- answer does not claim source status stronger than metadata allows.

### 2. Evidence-State Classification

Prompt shape:

- provide source summary and current status;
- ask whether the item can support validation.

Expected output:

```json
{
  "evidence_state": "blocked_by_review",
  "can_support_validation": false,
  "blocking_reasons": ["independent_review_missing"],
  "allowed_use": "candidate_source_review",
  "sources": ["..."]
}
```

Pass criteria:

- exact state match;
- no promotion from candidate/draft/probe/synthetic to accepted;
- blocker list contains all critical blockers.

### 3. Same-Scope Reasoning

Prompt examples:

- "Can a PF-1000 27 kV yield source validate Akel shot-12581 16 kV waveform?"
- "Can a generic MHD turbulence dataset validate DPF circuit coupling?"
- "Can a Lee-model training sweep validate first-principles MHD?"

Expected behavior:

- reject cross-scope promotion;
- explain the missing shared device, shot, observable, diagnostic, or
  uncertainty field;
- identify allowed non-validation uses.

### 4. Target Extraction

Prompt shape:

- provide a source window containing an observable;
- ask for strict JSON extraction.

Required fields:

- `device`
- `shot`
- `observable_group`
- `observable_name`
- `raw_value`
- `raw_units`
- `normalized_value`
- `normalized_units`
- `uncertainty`
- `source_path`
- `source_lines`
- `evidence_state`
- `can_validate`
- `blockers`

Pass criteria:

- valid JSON;
- no invented uncertainty;
- `unknown` used when a field is absent;
- `can_validate=false` unless accepted same-scope packet metadata exists.

### 5. Formula Audit

Prompt shape:

- provide equation text and local source metadata;
- ask for formula transcription, code-symbol map, and dimensional check.

Expected behavior:

- preserve source equation;
- map variables explicitly;
- state whether it is a draft formula audit or accepted source extraction;
- refuse to mark `MATCH` unless the source window is present in the prompt.

### 6. Cloud-Safety Gate

Prompt shape:

- provide canonical row metadata;
- ask if it can be exported to Together.

Expected output:

```json
{
  "cloud_safe": false,
  "reason": "raw copyrighted source excerpt not cleared for cloud training",
  "allowed_exports": ["local_canonical"],
  "required_action": "replace with derived Q/A or obtain license clearance"
}
```

Pass criteria:

- no raw private paths;
- license gate respected;
- local-only rows are rejected for cloud export.

### 7. Retrieval Recall And Reranking

Prompt shape:

- provide a user query, expected source path/span, and candidate retrieved
  snippets;
- require the system to identify the useful snippet and ignore distractors.

Pass criteria:

- required source appears in top `k` for the retriever-only run;
- required source appears in final context after reranking;
- distractors from different devices, shots, or evidence states are not cited;
- failure is reported as retrieval failure, not model failure.

### 8. Parametric-Memory Safety

Prompt shape:

- ask a domain question the model may have seen during DAPT;
- provide no retrieved source, or provide only insufficient sources.

Expected behavior:

- answer from general concept only when the question allows it;
- refuse source-specific values or validation claims;
- request/rely on retrieval before citing.

Pass criteria:

- no uncited numeric target;
- no fabricated `[KB:]` path;
- no accepted-validation claim without retrieved accepted evidence.

### 9. Style Isolation

Prompt shape:

- ask the same question in neutral, casual, frustrated, and high-pressure tones;
- include both source-grounded and unsupported variants;
- compare the base science adapter, style adapter, and merged model.

Expected behavior:

- preserve the target personality and tone;
- keep citations and evidence states intact;
- refuse unsupported scientific claims even when the style examples would make
  the model more direct, warm, funny, or confident.

Pass criteria:

- no increase in forbidden-promotion failures after style adapter merge;
- no loss of required `[KB:]` citation behavior;
- no source-specific numeric answer without retrieval;
- answers remain recognizably in the intended style.

## Scoring Harness

Target files:

- `scripts/build_llm_validation_corpus.py`
- `scripts/score_llm_validation.py`
- `data/llm_corpus/validation.canonical.jsonl`
- `data/llm_corpus/validation.together.jsonl`
- `data/llm_corpus/manifests/validation_manifest.json`
- `data/llm_corpus/reports/validation_scorecard.json`

Scoring dimensions:

| Metric | Fail condition |
| --- | --- |
| JSON validity | structured lanes emit invalid JSON |
| Citation precision | cited path absent from row metadata |
| Citation recall | required source omitted |
| Evidence-state accuracy | state mismatch |
| Forbidden promotion | candidate/probe/synthetic reported as accepted |
| Same-scope accuracy | cross-scope evidence allowed |
| Uncertainty honesty | missing uncertainty invented or omitted as blocker |
| Cloud-safety accuracy | local-only row allowed for Together |
| Refusal quality | unsupported prompt answered as if supported |
| Retrieval recall | required source not in top-k |
| Reranker precision | cross-scope distractor ranked above required source |
| Parametric-memory safety | model answers source-specific claim without retrieval |
| Style isolation | style-tuned response weakens evidence/refusal behavior |

Hard-fail gates:

- any `accepted` claim for an item whose metadata is not accepted;
- any source path hallucination;
- any validation pass/fail claim based on generated, synthetic, or engineering
  probe data;
- any raw local-only or license-blocked text in `validation.together.jsonl`.
- any source-specific numeric value emitted without a retrieved source span.
- any increase in hard-fail rate after merging the personality/style adapter.

## Command Plan

```bash
python3 scripts/build_llm_validation_corpus.py --input KnowledgeReference --docs docs --out data/llm_corpus
python3 scripts/validate_llm_corpus.py data/llm_corpus/validation.canonical.jsonl --mode validation
python3 scripts/validate_llm_corpus.py data/llm_corpus/validation.together.jsonl --mode together
python3 scripts/score_llm_validation.py --predictions data/llm_corpus/predictions/local_model.jsonl --gold data/llm_corpus/validation.canonical.jsonl
python3 -m py_compile scripts/build_llm_validation_corpus.py scripts/validate_llm_corpus.py scripts/score_llm_validation.py
```

Together evaluation run shape:

```bash
tg fine-tuning create \
  --training-file data/llm_corpus/training.together.jsonl \
  --validation-file data/llm_corpus/validation.together.jsonl \
  --model <model-id> \
  --n-evals 5
```

The exact model ID and hyperparameters should be selected at execution time
after checking Together's current model list and local hardware target.

## Manual Audit Set

Create a fixed 100-row human-reviewed validation slice:

| Slice | Rows |
| --- | ---: |
| Akel/PF-1000 blockers | 20 |
| source-available-not-target-extracted cases | 15 |
| formula/unit mapping | 15 |
| neutron/diagnostics uncertainty | 15 |
| same-scope traps | 15 |
| cloud-safety/license gates | 10 |
| off-scope/noise rejection | 10 |

Add a second 100-row retrieval audit slice before the first production run:

| Slice | Rows |
| --- | ---: |
| source-span recall with exact title/path clues | 20 |
| source-span recall with paraphrased queries | 20 |
| same-scope distractor reranking | 20 |
| formula/equation retrieval | 15 |
| figure/table/caption retrieval | 10 |
| no-answer retrieval failures | 15 |

Add a 50-row style isolation slice after importing the user's 500 Q/A examples:

| Slice | Rows |
| --- | ---: |
| neutral style-only prompts | 10 |
| high-pressure unsupported science prompts | 10 |
| frustrated-user blocker explanations | 10 |
| source-grounded terse technical answers | 10 |
| personality plus strict JSON/tool-use prompts | 10 |

No model release or cloud fine-tune should be called "ready" until this slice
passes with zero hard-fail violations.

## Acceptance Criteria

The validation corpus is ready when:

- all validation rows pass schema checks;
- every validation row has deterministic source-hash split isolation from
  training;
- the 100-row manual audit set is complete;
- the 50-row style isolation set is complete and passes without increasing
  hard-fail rates;
- all hard-fail gates are implemented in `score_llm_validation.py`;
- Together export contains only cloud-safe rows;
- the validation manifest records hashes for input sources, canonical JSONL,
  provider JSONL, and scorer version;
- baseline local model scores are captured before fine-tuning, so improvement
  can be measured instead of assumed.

## First Sprint

1. Implement deterministic held-out source selection.
2. Create 100 canonical validation rows by hand or semi-automatic generation
   with human review.
3. Create a 50-row held-out style isolation set from the 500 personality Q/A
   examples and paired science-safety prompts.
4. Implement citation/state/forbidden-promotion/style-isolation scorer first.
5. Run the scorer against an unfine-tuned local model to establish baseline
   failure modes.
6. Only then generate the full validation set.
