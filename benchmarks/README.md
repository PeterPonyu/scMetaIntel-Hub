# Benchmarks in scMetaIntel-Hub

This folder is the merged benchmark workspace for the integrated project.

## Current state

Phase 1 keeps the heavy benchmark implementation inside the original `GEO-DataHub/benchmarks` tree while the merged project establishes:

- shared evaluation utilities in `scmetaintel/evaluate.py`
- CLI evaluation wrapper in `scmetaintel/eval.py`
- model-by-subsystem documentation in `docs/MODEL_SUMMARY.md`
- local benchmark asset seeds such as `eval_queries.json`

## Next migration step

Port or wrap these GEO-DataHub benchmark scripts here:

1. `01_build_ground_truth.py`
2. `02_bench_embeddings.py`
3. `03_bench_retrieval.py`
4. `04_bench_llm.py`
5. `05_bench_context.py`
6. `06_bench_finetune.py`
7. `07_bench_e2e.py`

## Immediate usage

```bash
python -m scmetaintel eval --run-all
python benchmarks/model_subsection_summary.py
```
