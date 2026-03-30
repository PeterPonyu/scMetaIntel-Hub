# scMetaIntel-Hub Benchmark Design

## Overview

**51 models × 16 families × 35 evaluation tasks × 4 ablation dimensions**

The benchmark has two layers:
- **Layer 1: Domain-specific** — 8 tasks (A–H) on scMetaIntel ground truth (the paper's contribution)
- **Layer 2: General capability** — 27 public datasets across 6 categories (contextualizes model quality)

All evaluation runs locally via Ollama on RTX 5090 Laptop (24.5 GB VRAM).

---

## Layer 1: Domain-Specific Tasks (scMetaIntel)

| Task | What | Data | Input → Output | Metrics | Scale |
|------|------|------|-----------------|---------|-------|
| **A** Parse | NL query → structured JSON | eval_queries.json | query → `{organism, tissue, ...}` | **Exact match**, per-field accuracy | 171 queries |
| **B** Extract | GEO text → metadata | ground_truth/*.json | title+summary → `{tissues[], diseases[], cell_types[]}` | **P/R/F1** per field | 2189 docs |
| **C** Ontology | Terms → ontology IDs | ground_truth + OBO | raw terms → `{ontology_id, label}` | **Accuracy, Recall, F1** | 2189 docs |
| **D** Answer | Retrieved studies → cited answer | eval_queries.json | context+query → answer | **Citation P/R**, grounding rate | 171 queries |
| **E** Speed | Throughput | 5 built-in prompts | prompt → response | **tok/s**, per-prompt breakdown | 5 prompts |
| **F** Relevance | Query-doc binary classification | queries × ground_truth | query+doc → `{relevant: bool}` | **Accuracy, P/R/F1**, confusion matrix | ~1340 pairs |
| **G** Domain | Study → domain label | ground_truth/*.json | title+summary → `{domain}` | **Accuracy**, per-domain accuracy | 1649 docs |
| **H** Org+Mod | Study → organism & modality | ground_truth/*.json | title+summary → `{organism, modalities[]}` | Organism **accuracy**, modality **set F1** | 2106 docs |

---

## Layer 2: Public Benchmark Datasets

### Category 1: General LLM Knowledge (4 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **MMLU** | 14,042 | 4-choice MCQ | `answer` (int 0-3) | **Accuracy** | Prompt: "Q: ... A/B/C/D: ... Answer:" → extract letter → map to index |
| **HellaSwag** | 10,042 | 4-choice completion | `label` (str "0"-"3") | **Accuracy** | Prompt: "Complete: {ctx} A/B/C/D: {endings}" → extract choice |
| **WinoGrande** | 1,267 | Binary choice | `answer` (str "1"/"2") | **Accuracy** | Prompt: "{sentence} option1: ... option2: ..." → extract 1 or 2 |
| **ARC-Challenge** | 1,172 | 4-choice MCQ | `answerKey` (str "A"-"D") | **Accuracy** | Same as MMLU but letter answer |

**Unified evaluator:** `mcq_accuracy(pred_letter, gold_letter)` — exact match on extracted letter/index.

### Category 2: Reasoning (3 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **GSM8K** | 1,319 | Math word problem | `answer` (str "...#### N") | **Accuracy** (final number) | Extract number after "####" from both pred and gold |
| **TruthfulQA** | 817 | MCQ with truth labels | `mc1_targets.labels` | **MC1 accuracy** | Prompt MCQ, check if selected answer has label=1 |
| **ARC-Easy** | 2,376 | 4-choice MCQ | `answerKey` (str) | **Accuracy** | Same as ARC-Challenge |

**GSM8K evaluator:** `gsm8k_accuracy(pred_text, gold_text)` — extract final number from `#### N` pattern.
**TruthfulQA evaluator:** `truthfulqa_mc1(pred_index, labels)` — check if `labels[pred_index] == 1`.

### Category 3: Biomedical (11 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **PubMedQA** | 1,000 | Yes/No/Maybe | `final_decision` | **Accuracy** | Extract yes/no/maybe from response |
| **MedQA** | 1,273 | MCQ (A-D) | `answer_idx` (str) | **Accuracy** | Standard MCQ extraction |
| **MedMCQA** | 4,183 | MCQ (0-3) | `cop` (int) | **Accuracy** | Standard MCQ extraction |
| **SciQ** | 1,000 | MCQ (text answer) | `correct_answer` (str) | **Accuracy** | Match against correct_answer vs distractors |
| **BioASQ-mini** | 4,719 | Free text QA | `answer` (str) | **Token F1**, exact match | Token-level F1 between pred and gold |
| **MMLU-anatomy** | 135 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |
| **MMLU-clinical** | 265 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |
| **MMLU-college-bio** | 144 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |
| **MMLU-college-med** | 173 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |
| **MMLU-med-genetics** | 100 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |
| **MMLU-prof-med** | 272 | MCQ (0-3) | `answer` (int) | **Accuracy** | Standard MCQ |

**Bio aggregate:** Average accuracy across all 11 bio datasets (macro-average).

### Category 4: Structured Output (3 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **IFEval** | 541 | Instruction constraints | `instruction_id_list` | **Strict accuracy**, loose accuracy | Check each constraint (word count, format, etc.) |
| **JSON-mode-eval** | 100 | JSON schema | `completion` + `schema` | **Schema validity**, field accuracy | Parse output → validate against schema |
| **SQuAD-v2** | 1,000 | Extractive QA | `answers.text[]` | **Token F1**, exact match | Standard SQuAD F1: overlap between pred and gold spans |

**IFEval evaluator:** Per-constraint binary check (programmatic, no LLM judge).
**JSON evaluator:** `json_schema_accuracy(pred_json, gold_schema)` — parse + validate.

### Category 5: Tool-use / Function Calling (3 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **NexusRaven-FC** | 318 | API function call | `python_function_name` + `python_args_dict` | **Function name accuracy**, argument F1 | Exact match on func name + JSON arg comparison |
| **Glaive-FC** | 1,000 | `<functioncall>` JSON | embedded in `chat` | **Call accuracy**, parse success rate | Extract `<functioncall>` → validate JSON → match name/args |
| **ToolACE** | 1,000 | `[func(param=val)]` | embedded in `conversations` | **Call accuracy**, parse success rate | Extract function call → match name + params |

**Unified FC evaluator:**
- `fc_name_accuracy`: Does the model call the correct function?
- `fc_args_f1`: How many arguments are correct? (key match + value match)
- `fc_parse_rate`: Can the output be parsed as a valid function call at all?

### Category 6: Commonsense (3 datasets)

| Dataset | Samples | Format | Label Field | Metric | How to Evaluate |
|---------|---------|--------|-------------|--------|-----------------|
| **SIQA** | 1,954 | 3-choice | `label` (str "1"-"3") | **Accuracy** | Extract choice number |
| **OpenBookQA** | 500 | 4-choice MCQ | `answerKey` (str) | **Accuracy** | Standard MCQ extraction |
| **BoolQ** | 3,270 | Boolean QA | `answer` (bool) | **Accuracy** | Extract yes/no → map to bool |

---

## Ablation Dimensions

### Dimension 1: Think Mode

| Condition | Models | How |
|-----------|--------|-----|
| think=False | All 51 models | Baseline direct answering |
| think=True | 19 API-toggle models (Qwen3/3.5, Gemma3, Granite3.3) | Ollama `{"think": true}` |
| always-think | 4 DeepSeek-R1 models | Cannot disable, `<think>` always in response |

**Metric:** Δ accuracy (think=True − think=False) per task.
**Total configs:** 66 (51 base + 15 +think variants; DeepSeek-R1 only stores +think).

### Dimension 2: Quantization

| Pair | Q4_K_M | Q8_0 | Size |
|------|--------|------|------|
| Qwen3-8B | qwen3-8b (5.2GB) | qwen3-8b-q8 (9.0GB) | 8B |
| Gemma3-12B | gemma3-12b (8.1GB) | gemma3-12b-q8 (13GB) | 12B |
| Llama3.1-8B | llama3.1-8b-q4 (4.9GB) | llama3.1-8b (8.5GB) | 8B |
| Mistral-7B | mistral-7b-q4 (4.4GB) | mistral-7b (7.7GB) | 7B |

**Metric:** Δ accuracy (Q8_0 − Q4_K_M) per task, VRAM/speed tradeoff.

### Dimension 3: KV Cache Type (runtime, no download)

| Setting | Env Var | Effect |
|---------|---------|--------|
| f16 (default) | `OLLAMA_KV_CACHE_TYPE=f16` | Best quality, most VRAM |
| q8_0 | `OLLAMA_KV_CACHE_TYPE=q8_0` | ~25% less KV VRAM |
| q4_0 | `OLLAMA_KV_CACHE_TYPE=q4_0` | ~50% less KV VRAM |

**Test on:** 5 models (qwen3-8b, llama3.1-8b, phi4-14b-q8, gemma3-12b-q8, qwen3.5-9b-q8) × 3 KV types × Tasks A/D/F (most context-sensitive).
**Metric:** Δ accuracy across KV types, VRAM reduction measured.

### Dimension 4: Context Length (runtime, no download)

| Setting | Env Var | Effect |
|---------|---------|--------|
| 2048 | `SCMETA_NUM_CTX=2048` | Minimal context, fastest |
| 4096 (default) | `SCMETA_NUM_CTX=4096` | Standard context |
| 8192 | `SCMETA_NUM_CTX=8192` | Extended context |
| 16384 | `SCMETA_NUM_CTX=16384` | Maximum tested context |

**Test on:** Same 5 models × 4 context lengths × Tasks D/E (answer quality + speed).
**Metric:** Throughput (tok/s) and answer quality vs context length.

---

## Metric Summary by Evaluator Type

| Evaluator | Datasets Using It | Implementation |
|-----------|-------------------|----------------|
| **MCQ Accuracy** | MMLU (all), ARC, HellaSwag, WinoGrande, MedQA, MedMCQA, SciQ, SIQA, OpenBookQA, TruthfulQA | Extract letter/index → exact match |
| **Boolean Accuracy** | BoolQ, PubMedQA | Extract yes/no → compare |
| **Numeric Extraction** | GSM8K | Extract `#### N` → compare numbers |
| **Token F1** | SQuAD-v2, BioASQ-mini | Word overlap between pred and gold |
| **JSON Schema Valid** | JSON-mode-eval, Tasks A/B/G/H | Parse → validate against schema |
| **Instruction Compliance** | IFEval | Programmatic constraint checking |
| **Function Call Match** | NexusRaven, Glaive-FC, ToolACE | Parse call → match name + args |
| **Set F1** | Task B (fields), Task H (modalities) | Set intersection / union |
| **Citation Extraction** | Task D | Regex GSE\d+ → precision/recall |
| **Fuzzy Field Match** | Task A (parse), Task C (ontology) | Synonym normalization + containment |
| **Binary Classification** | Task F (relevance) | Accuracy, P/R/F1, confusion matrix |
| **Speed** | Task E | tokens/sec, latency |

---

## Composite Scoring

### Domain composite (implemented in `generate_article_figures.py`)

```
parse_score   = task_a.exact_match
extract_score = mean(task_b.tissues_f1, task_b.diseases_f1, task_b.cell_types_f1)
onto_score    = task_c.f1
answer_score  = mean(task_d.citation_recall, task_d.grounding_rate)
speed_score   = min(task_e.tokens_per_sec / 100, 1.0)
relev_score   = task_f.f1
domain_score  = task_g.accuracy
orgmod_score  = mean(task_h.organism_accuracy, task_h.modality_f1)

composite = (0.15 × parse_score   +
             0.15 × extract_score +
             0.15 × onto_score    +
             0.20 × answer_score  +
             0.05 × speed_score   +
             0.10 × relev_score   +
             0.10 × domain_score  +
             0.10 × orgmod_score)
```

### Public benchmark categories (per-category averages, reported in Fig 7)

```text
general_score     = mean(mmlu, hellaswag, winogrande, arc_challenge)
reasoning_score   = mean(gsm8k, truthfulqa, arc_easy)
biomedical_score  = mean(pubmedqa, medqa, medmcqa, sciq, bioasq, 6×mmlu_bio)
structured_score  = mean(ifeval, json_mode, squad_v2)
tooluse_score     = mean(nexusraven, glaive, toolace)
commonsense_score = mean(siqa, openbookqa, boolq)
```
