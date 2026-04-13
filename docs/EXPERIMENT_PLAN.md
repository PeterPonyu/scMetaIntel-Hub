# Experiment Plan: Full-Scale Re-Bench

**Date**: 2026-03-31
**Hardware**: RTX 5090 Laptop (24.5 GB VRAM), Linux 6.14
**Prerequisites**: All bugs fixed, code centralized, benchmark scripts updated

---

## Current State

| Asset | Count | Status |
|-------|-------|--------|
| Ground truth docs | 2,189 | Ready |
| Eval queries | 171 | Ready |
| LLM configs | 66 (51 base + 15 think) | All at small-sample |
| Public bench models | 6 / 66 | 50 samples each |
| Ablation configs | 34 (15 KV + 19 ctx) | Small-sample |

---

## Phase 1: Prep & Archive (5 min)

**Goal**: Archive pilot results, clear for full re-bench.

```bash
ollama serve &                           # Start Ollama
python scripts/prep_rebench.py           # Archive + clear all 66 entries
```

**Checkpoint**: `llm_bench.json` should have 0 entries after clearing.

---

## Phase 2: Full-Scale LLM Domain Bench (12-24 hours)

**Goal**: Run all 66 configs on 8 tasks at full scale.

```bash
conda run -n dl python benchmarks/04_bench_llm.py
```

**Scale** (per config):
- Task A: 171 queries (was 30)
- Task B: ~2,000 docs (was 24) — with relaxed tissue filter
- Task C: ~1,500 docs (was 16)
- Task D: 171 queries (was 20) — with fixed grounding metric
- Task E: 5 prompts (unchanged)
- Task F: ~1,000 pairs (was ~30)
- Task G: ~1,649 docs (was 50)
- Task H: ~2,106 docs (was 50)

**Total**: 66 configs x 8 tasks = 528 task runs

**Estimated time per config**:
- Small models (1-4B): ~5-8 min
- Medium models (7-9B): ~10-15 min
- Large models (12-32B): ~15-25 min
- Average: ~15 min/config x 66 = ~16 hours

**Resumability**: Saves after every individual task. Safe to Ctrl+C.

**Monitoring**:
```bash
# Check progress (in another terminal)
python3 -c "
import json
with open('benchmarks/results/llm_bench.json') as f:
    data = json.load(f)
done = sum(1 for v in data.values()
           if all(f'task_{t}' in str(v) for t in 'abcdefgh'))
print(f'{done}/{66} configs complete, {len(data)} partially started')
"
```

**Checkpoint**: 66 entries in llm_bench.json, all with n_queries >= 100.

---

## Phase 3: Regenerate Figures (30 min)

**Goal**: Update all article figures with full-scale results.

```bash
python scripts/generate_article_figures.py
```

**Outputs**: 7 main figures + 2 supplementary + 8 tables in `article_figures/`

**Checkpoint**: `article_summary.json` updated with new top-5 models.

---

## Phase 4: Expanded Public Benchmarks (8-12 hours)

**Goal**: Test representative models on standard benchmarks at meaningful sample sizes.

**Model selection** (15 models — one per family + size extremes):

| Model | Family | Size | Rationale |
|-------|--------|------|-----------|
| qwen2.5-0.5b | qwen | 0.5B | Floor baseline |
| qwen3-8b | qwen | 8B | Mid-range Qwen3 |
| qwen3.5-9b-q8 | qwen | 9B | Frontier Qwen |
| qwen3-32b | qwen | 32B | Ceiling Qwen |
| llama3.1-8b | llama | 8B | Top composite |
| llama3.2-3b | llama | 3B | Small Llama |
| gemma3-12b-q8 | gemma | 12B | Mid Gemma |
| gemma3-1b | gemma | 1B | Tiny Gemma |
| mistral-7b | mistral | 7B | Mistral baseline |
| phi4-14b-q8 | phi | 14B | Microsoft |
| deepseek-r1-7b+think | deepseek | 7B | Always-think |
| falcon3-7b | falcon | 7B | Falcon baseline |
| aya-expanse-8b | aya | 8B | Multilingual |
| granite3.3-8b | granite | 8B | IBM structured |
| starcoder2-7b | starcoder | 7B | Negative control |

```bash
conda run -n dl python benchmarks/05_bench_public.py \
    --models qwen2.5-0.5b qwen3-8b qwen3.5-9b-q8 qwen3-32b \
             llama3.1-8b llama3.2-3b gemma3-12b-q8 gemma3-1b \
             mistral-7b phi4-14b-q8 deepseek-r1-7b \
             falcon3-7b aya-expanse-8b granite3.3-8b starcoder2-7b \
    --max-samples 200
```

**Checkpoint**: 15 models in public_bench.json, each with n_samples=200.

---

## Phase 5: Ablation Re-Bench (4-6 hours)

**Goal**: Re-run KV cache + context length ablations at larger sample sizes.

```bash
conda run -n dl python benchmarks/08_bench_ablation.py
```

**Checkpoint**: ablation_bench.json updated with larger n per task.

---

## Phase 6: Update Article Plan (2-3 hours, manual)

**Goal**: Rewrite ARTICLE_PLAN.tex to match actual scale.

Key updates needed:
- Central claim: 21 → 66 configs, 5 → 8 tasks, 90 → 171 queries
- Add 5th contribution: public benchmark contextualization
- Update methods with Tasks F/G/H
- Update results with actual top-5 from re-bench
- Add ablation section (KV cache + context length)
- Add think-mode analysis section

---

## Phase 7: Final Figure Regeneration + Quality Check (1 hour)

```bash
python scripts/generate_article_figures.py
```

**Quality checks**:
- [ ] Top-5 rankings changed from pilot? (expected)
- [ ] Tissue F1 no longer clustered at 0.1319?
- [ ] Gemma3/Granite think-mode has real scores?
- [ ] grounding != citation_precision?
- [ ] All 66 configs in heatmap?

---

## Dependency Graph

```
Phase 1 (5 min)
    |
    v
Phase 2 (12-24h) -----> Phase 3 (30 min) -----> Phase 7 (1h)
    |                         |
    v                         v
Phase 4 (8-12h)          Phase 6 (manual)
    |
    v
Phase 5 (4-6h)
```

Phases 2 and 4 can overlap if Phase 2 finishes first batch.
Phase 6 can happen in parallel with GPU work.

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| Ollama OOM on 32B models | `--include-spill` flag; num_ctx=4096 proven |
| Script crashes mid-run | Per-task checkpoint; just re-run same command |
| Power loss / reboot | Results saved after each task; full resume |
| Rankings change dramatically | Expected — pilot rankings were statistically unreliable |
| Extraction F1 still low | Already fixed gold truth filter; may need prompt work later |

---

## Success Criteria

- [ ] All 66 LLM configs benchmarked at full scale
- [ ] At least 15 models on public benchmarks (n>=200)
- [ ] Ablation re-run with fixed code
- [ ] All figures regenerated
- [ ] No hardcoded Ollama URLs or magic numbers remain
- [ ] Article plan updated to match reality
