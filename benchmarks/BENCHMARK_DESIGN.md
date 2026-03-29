# scMetaIntel-Hub Benchmark Design

## Overview

**51 models × 15 families × 35 evaluation tasks × 3 ablation dimensions**

The benchmark has two layers:
- **Layer 1: Domain-specific** — 8 tasks (A–H) on scMetaIntel ground truth (the paper's contribution)
- **Layer 2: General capability** — 27 public datasets across 6 categories (contextualizes model quality)

All evaluation runs locally via Ollama on RTX 5090 Laptop (24.5 GB VRAM).

---

## Layer 1: Domain-Specific Tasks (scMetaIntel)

| Task | What | Data | Input → Output | Metrics | Scale |
|------|------|------|-----------------|---------|-------|
| **A** Parse | NL query → structured JSON | eval_queries.json | query → `{organism, tissue, ...}` | **Exact match**, per-field accuracy | 163 queries |
| **B** Extract | GEO text → metadata | ground_truth/*.json | title+summary → `{tissues[], diseases[], cell_types[]}` | **P/R/F1** per field | 2109 docs |
| **C** Ontology | Terms → ontology IDs | ground_truth + OBO | raw terms → `{ontology_id, label}` | **Accuracy, Recall, F1** | 2109 docs |
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

## Ablation Dimensions (No Extra Downloads)

### Dimension 1: Think Mode

| Condition | Models | How |
|-----------|--------|-----|
| think=False | All 51 models | Baseline direct answering |
| think=True | 14 API-toggle models (Qwen3/3.5, Gemma3, Granite3.3) | Ollama `{"think": true}` |
| always-think | 4 DeepSeek-R1 models | Cannot disable, `<think>` always in response |

**Metric:** Δ accuracy (think=True − think=False) per task.

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

**Test on:** Top 5 models × 3 KV types × Tasks A/D/F (most context-sensitive).
**Metric:** Δ accuracy across KV types, VRAM reduction measured.

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

### Per-category composite (for radar charts):

```
general_score     = mean(mmlu_acc, hellaswag_acc, winogrande_acc, arc_acc)
reasoning_score   = mean(gsm8k_acc, truthfulqa_acc, arc_easy_acc)
biomedical_score  = mean(pubmedqa_acc, medqa_acc, medmcqa_acc, sciq_acc, mmlu_bio_avg)
structured_score  = mean(ifeval_strict, json_schema_acc, squad_f1)
tooluse_score     = mean(nexus_name_acc, glaive_parse_rate, toolace_parse_rate)
commonsense_score = mean(siqa_acc, openbookqa_acc, boolq_acc)
domain_score      = mean(task_a_em, task_b_f1, task_c_f1, task_d_cite_recall,
                         task_f_f1, task_g_acc, task_h_org_acc)
speed_score       = normalized(tok/s)
```

### Overall composite:
```
composite = weighted_mean(
    domain_score     × 0.35,   # Our paper's contribution (most weight)
    biomedical_score × 0.15,   # Domain-adjacent public benchmarks
    general_score    × 0.10,   # General capability baseline
    reasoning_score  × 0.10,   # Reasoning (important for think ablation)
    structured_score × 0.10,   # JSON/structured output (core pipeline need)
    tooluse_score    × 0.05,   # Function calling
    commonsense_score× 0.05,   # Commonsense
    speed_score      × 0.10,   # Practical deployment consideration
)
```
