# scMetaIntel-Hub: Project Status Report

**Date**: 2026-04-13 (updated)
**Scope**: Benchmarking analysis, blockers, bugs found & fixed, current state

---

## 1. Where We Are Now

### 1.1 Infrastructure (Complete)

| Component | Status | Details |
|-----------|--------|---------|
| Ground truth corpus | **Done** | 2,189 enriched GSE JSONs, 42 organisms, 100% document text |
| Evaluation queries | **Done** | 171 hand-curated queries, 7 categories, 540 expected GSEs |
| Public datasets | **Done** | 27 datasets across 6 categories |
| Benchmark scripts | **Done** | 9 scripts (01-09) covering all benchmark levels |
| Figure generation | **Done** | 7 main + 2 supplementary figures, 9 tables |
| Model configs | **Done** | 51 base models, 16 families, 66 total configs |
| Article draft | **Done** | sn-nature format, 10 pages, compiles cleanly |
| Article plan | **Updated** | `ARTICLE_PLAN.tex` v1.0.0 matches 66 configs × 8 tasks |

### 1.2 Benchmark Runs

| Benchmark | Models | Sample Size | Full Scale | Completion |
|-----------|--------|-------------|------------|------------|
| **LLM domain (04)** | 51 base configs | Full (163-2106/task) | 171q / 2,189d | **Done (base)** |
| **LLM think-mode (04)** | 15 think configs | Not started | 171q / 2,189d | **Pending** |
| **Public bench (05)** | 6 models | 50/dataset | 15 models × 200 | **Partial (~40%)** |
| **Embedding (02)** | 14 models | Full | Full | **Done** |
| **Retrieval (03)** | 6 strategies | Full | Full | **Done** |
| **Ablation (08)** | 5 models | 20-30/task | Full | **Partial (~15%)** |
| **Context (06/07)** | Done | Done | Done | **Done** |
| **E2E (09)** | 4 pipelines | Full (90q) | Full | **Done** |

### 1.3 Current Top-5 (Full-Scale, 51 Base Models)

| Rank | Model | Composite | Parse EM | Extract F1 | Cite Recall |
|------|-------|-----------|----------|------------|-------------|
| 1 | llama3.1-8b-q4 | 0.776 | 0.967 | 0.273 | 0.883 |
| 2 | llama3.1-8b | 0.768 | 0.833 | 0.300 | 0.933 |
| 3 | aya-expanse-8b | 0.767 | 1.000 | 0.278 | 0.950 |
| 4 | qwen3.5-4b | 0.756 | 0.633 | 0.312 | 1.000 |
| 5 | qwen3-1.7b+think | 0.755 | 0.700 | 0.311 | 0.917 |

### 1.4 Remaining Work Before Submission

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Regenerate fig4 heatmap + table3 from current data | Critical | ~1 hour |
| Bootstrap CIs for all 51 models | Critical | ~2 hours |
| 15 think-mode configs re-bench | High | ~8 hours GPU |
| Expand public benchmarks (15 models, n=200) | High | ~12 hours GPU |
| Full-scale ablation re-run | Medium | ~4 hours GPU |
| Fill real affiliation in article/main.tex | Medium | Manual |

---

## 2. Bugs Found & Fixed

### Bug 1 (Critical): Gemma3 + Granite Think-Mode Total Failure

**Symptom**: All 7 Gemma3+think and Granite+think configs scored **0.0 on every task**. Error messages: "no successful extractions", "model not available".

**Root cause**: When Ollama think mode is enabled for these models, the answer content field comes back empty while all output goes into the `thinking` field. The `extract_json()` function sees empty text and returns None.

**Fix** (`scmetaintel/answer.py`): Added fallback in `ollama_generate()` — if think mode is on, content is empty, but thinking is non-empty, use thinking text as the response. This recovers the JSON answer from models that put everything in the thinking field.

```python
if think and not response_text.strip() and thinking_text.strip():
    response_text = thinking_text  # Fallback: extract answer from thinking
```

**Impact**: 7 configs × 8 tasks = 56 task results now recoverable.

### Bug 2 (Major): Tissue F1 Floor Effect — 79% of Gold Annotations Discarded

**Symptom**: 27% of models scored exactly 0.1319 on tissue_F1. Only 4 out of 24 benchmark docs had non-empty tissue gold after cleaning.

**Root cause**: `clean_extraction_gold()` required exact substring match of tissue names in the title+summary. Sample-level metadata terms (e.g., "cadaveric pancreatic islet") rarely appear verbatim in study abstracts, so 79% of tissue annotations were silently discarded.

**Fix** (`scmetaintel/evaluate.py`): Relaxed the text-presence filter. New logic: keep a term if (a) exact substring match in text, OR (b) all significant words (len >= 3) from the term appear in the text.

**Measured improvement** (first 200 docs):
- OLD: 54/200 docs with tissues, 66 terms total
- NEW: 76/200 docs with tissues, 131 terms total (+41% docs, +98% terms)

Also added `NOT_TISSUE` set to filter out condition labels ("normal", "control", "healthy", etc.) that leaked from sample metadata into tissue lists.

### Bug 3 (Major): grounding_rate == citation_precision (Duplicate Metric)

**Symptom**: `grounding_rate` and `citation_precision` were numerically identical for all 59 models (59.3% scoring 1.0).

**Root cause**: In `task_d_answer_generation()`, the context was built from `expected_gse`, and `citation_accuracy(cited, expected_gse, expected_gse)` passed the same set as both `relevant_gse` and `retrieved_gse`. By construction: `cited ∩ relevant / |cited| == cited ∩ retrieved / |cited|`.

**Fix** (`benchmarks/04_bench_llm.py`): Now passes `all_expected` (full expected_gse set from query, including GSEs not in corpus) as `relevant_gse`, and only corpus-available GSEs as `retrieved_gse`. This differentiates "did the model cite relevant studies?" from "did the model cite studies that were in the context?".

### Bug 4 (Minor): Non-Tissue Terms in Gold ("normal", "control")

**Symptom**: GSE112302 had `"normal"` as a tissue term, penalizing every model.

**Fix** (`scmetaintel/evaluate.py`): Added `NOT_TISSUE` filter set in `clean_tissue_list()` with 18 common non-tissue condition labels.

---

## 3. Metric Quality Analysis

### Metrics That Discriminate Well (useful for ranking)

| Metric | Range | Stdev | Notes |
|--------|-------|-------|-------|
| **parse_EM** | 0.0–1.0 | 0.323 | Wide bimodal distribution |
| **onto_F1** | 0.0–0.97 | 0.287 | Good spread |
| **cite_recall** | 0.03–1.0 | 0.223 | 31 unique values, continuous |
| **relev_F1** | 0.0–0.93 | 0.302 | Good once zeros excluded |
| **tok/s** | 29–324 | 70.6 | Wide, continuous |

### Metrics With Ceiling/Floor Effects (poor discrimination)

| Metric | Issue | Details |
|--------|-------|---------|
| **grounding** | Ceiling | 59% score 1.0, median=1.0 |
| **cite_prec** | Ceiling (was duplicate) | Identical to grounding; now fixed |
| **tissue_F1** | Floor | 27% at 0.1319; now fixed with relaxed filter |
| **mod_F1** | Narrow | stdev=0.063, range 0.65–1.0 |
| **domain_acc** | Narrow (excluding zeros) | Working models cluster 0.52–0.76 |
| **org_acc** | Narrow (excluding zeros) | Working models cluster 0.72–0.84 |

### Think-Mode Analysis

| Family | Think Effect | Details |
|--------|-------------|---------|
| **Gemma3** (5 models) | **Broken** (all 0s) | Fixed: content→thinking fallback |
| **Granite** (2 models) | **Broken** (all 0s) | Fixed: content→thinking fallback |
| **Qwen3** (6 models) | **Mixed** | Helps onto (+2-3x), domain (+20-30%); hurts parsing (-30% for larger models) |
| **Qwen3.5** (2 models) | **Hurts badly** | cite_recall drops from 1.0 to 0.2-0.3; many invalid outputs |
| **DeepSeek-R1** (4 models) | **Baseline only** | No non-think counterpart; decent scores (14b best) |

**Key insight**: Think mode helps **reasoning-heavy tasks** (ontology normalization, domain classification) but hurts **format-sensitive tasks** (parsing, structured extraction) because the reasoning chain consumes token budget and may corrupt JSON output. The Qwen3.5 regression on cite_recall is severe enough to warrant investigation — think mode may cause these models to "overthink" and not produce citations.

---

## 4. Code Improvements Made

### Per-Task Checkpoint/Resume (`04_bench_llm.py`)

The benchmark script now saves results after **every individual task** (not just per-model). If interrupted mid-run:
- Completed tasks are preserved
- On restart, only un-finished tasks for each model are re-run
- No GPU time wasted re-running completed tasks

Previously: interrupt → lose all 8 tasks for the current model.
Now: interrupt → lose at most 1 task.

### Re-Bench Preparation Script (`scripts/prep_rebench.py`)

New utility script that:
1. Archives current results (timestamped backup)
2. Identifies broken think-mode entries (7 configs)
3. Identifies small-sample entries (all 66 configs)
4. Clears them to force re-bench at full scale
5. Supports `--dry-run`, `--clear-all`, `--clear-think-only`

Current dry-run result: **all 66 entries need re-benching** (all used small samples).

---

## 5. Re-Bench Plan

### Phase 1: Clear & Archive (5 minutes)
```bash
python scripts/prep_rebench.py  # Archives + clears all 66 entries
```

### Phase 2: Full-Scale LLM Re-Bench (~12-24 hours GPU)
```bash
conda run -n dl python benchmarks/04_bench_llm.py
# No --max-* flags → defaults to 0 (all data)
# 66 configs × 8 tasks at full scale
# Per-task checkpointing: safe to Ctrl+C and resume
```

**Sample sizes at full scale**:
- Task A: 171 queries (was 30)
- Task B: ~2,000+ docs (was 24)
- Task C: ~1,500+ docs (was 16)
- Task D: 171 queries (was 20)
- Task F: ~1,000+ pairs (was ~30)
- Task G: ~1,649 docs (was 50)
- Task H: ~2,106 docs (was 50)

### Phase 3: Regenerate Figures (~30 min)
```bash
python scripts/generate_article_figures.py
```

### Phase 4: Expand Public Benchmarks (~8-12 hours GPU)
```bash
conda run -n dl python benchmarks/05_bench_public.py
# Expand from 6 to 15-20 models, 50 to 200+ samples
```

### Resumability

The re-bench is **fully resumable at per-task granularity**:
- `llm_bench.json` is saved after every task completion
- On restart, completed tasks are detected and skipped
- Safe to interrupt with Ctrl+C at any time
- If Ollama crashes, just restart `ollama serve` and re-run the script
- Each model is unloaded from VRAM before loading the next one

---

## 6. Expected Outcomes After Re-Bench

1. **Rankings will change significantly** — n=30 → n=171 queries removes statistical noise
2. **Tissue F1 will improve** — from 0.13 floor to meaningful discrimination (2x more gold terms)
3. **Gemma3/Granite think-mode will have real scores** — 7 previously broken configs now testable
4. **grounding vs cite_prec will diverge** — different semantics now correctly computed
5. **Extraction F1 may still be low** — if so, investigate prompt engineering next
6. **Think-mode patterns will be clearer** — at full scale, think-mode benefits vs costs become statistically significant

---

## 7. Remaining Blockers After Fixes

| Blocker | Status | Action |
|---------|--------|--------|
| Small sample sizes | **Fixed** (code ready) | Run full-scale re-bench |
| Gemma3/Granite think failures | **Fixed** (fallback added) | Will re-test in re-bench |
| Tissue F1 floor effect | **Fixed** (filter relaxed) | Will improve in re-bench |
| grounding == cite_prec | **Fixed** (different semantics) | Will diverge in re-bench |
| Article plan outdated | **Not yet fixed** | Update after re-bench results |
| Public bench coverage minimal | **Not yet fixed** | Expand after LLM re-bench |
| Extraction F1 low overall | **Partially fixed** (better gold) | Monitor in re-bench; may need prompt work |
| Speed metric think-mode bias | **Not yet fixed** | Minor; document as limitation |

---

## 8. Architecture Refactoring (2026-03-31)

### Problem

Per-family behavior was handled correctly via `MODEL_FAMILY_CONFIG` table, but:
- Token budgets were hardcoded as `4096 if think else <base>` in 9 places across 4 files
- Model key lookup was O(n) per `ollama_generate()` call (scanning LLM_MODELS by ollama_name)
- New families required knowing to set all config fields (no defaults)
- Output quirks (markdown wrapping, language mixing) were implicit in json_hint strings

### Solution

1. **`_FAMILY_DEFAULTS` dict** — New families only override what differs. Fields auto-filled:
   - `think_api`, `think_method`, `always_thinks`, `multimodal`
   - `json_hint`, `think_token_overhead` (default: 4096, deepseek: 6144)
   - `output_quirks` (set of: `markdown_wrap`, `preamble`, `language_mix`, `verbose_think`)

2. **`get_family_config(family)`** — Single accessor returning complete config with defaults merged.

3. **`think_token_budget(base, think, family)`** — Replaces all 9 hardcoded `4096 if think else N`.
   Centralized so changing think overhead per-family is one config change, not a code hunt.

4. **`resolve_model_key(ollama_name)`** — O(1) reverse index replacing O(n) scan in `ollama_generate()`.

5. **`output_quirks` per family** — Documented quirks (mistral: `markdown_wrap`, falcon: `preamble + markdown_wrap`, glm/exaone: `language_mix`, deepseek: `verbose_think`). Future: can auto-select post-processing or extra prompt hints.

### Adding a New Model Family (Before vs After)

**Before**: Add entry to `MODEL_FAMILY_CONFIG` with all 6+ fields, grep for `4096 if think` to find all token budget locations, test.

**After**: Add minimal entry to `MODEL_FAMILY_CONFIG` (only fields that differ from defaults). Everything else auto-inherits. Token budget, think dispatch, and output handling just work.

```python
# Minimal new family — only override what differs
"new_family": {
    "json_hint": "Return JSON only.",
    "output_quirks": {"markdown_wrap"},
    "note": "New model family description.",
}
# All other fields (think_api=False, think_method=None, ...) inherit from _FAMILY_DEFAULTS
```

---

## 9. Files Changed (Complete)

| File | Change |
|------|--------|
| `scmetaintel/config.py` | `_FAMILY_DEFAULTS`, `get_family_config()`, `think_token_budget()`, `resolve_model_key()`, `_OLLAMA_NAME_INDEX`, `output_quirks` per family, `think_token_overhead` per family |
| `scmetaintel/answer.py` | Think-mode fallback, refactored `ollama_generate()` to use `get_family_config()` + O(1) index, replaced hardcoded token budgets |
| `scmetaintel/evaluate.py` | Relaxed tissue filter (word-level matching), `NOT_TISSUE` set, cleaned condition labels |
| `benchmarks/04_bench_llm.py` | Per-task checkpoint/resume, grounding fix, `think_token_budget()` |
| `benchmarks/05_bench_public.py` | `think_token_budget()` |
| `benchmarks/08_bench_ablation.py` | `think_token_budget()` |
| `scripts/prep_rebench.py` | New: re-bench preparation utility |
