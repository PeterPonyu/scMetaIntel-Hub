<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/scMetaIntel-Hub.svg" width="64" alt="ZF Lab · scMetaIntel-Hub">
  </a>
</div>

# scMetaIntel-Hub

[![Repo health](https://github.com/PeterPonyu/scMetaIntel-Hub/actions/workflows/repo-health.yml/badge.svg)](https://github.com/PeterPonyu/scMetaIntel-Hub/actions/workflows/repo-health.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A unified project that merges:

- **GEO-DataHub** — acquisition backbone for GEO search, download, verification, conversion, and organization
- **scMetaIntel** — metadata intelligence layer for enrichment, ontology normalization, embedding, hybrid retrieval, reranking, evaluation, and chat

## Integrated architecture

```text
User Query
   │
   ├─ GEO acquisition bridge (`python -m scmetaintel geo ...`)
   │    └─ delegates to the proven GEO-DataHub CLI
   │
   └─ Intelligence layer (`python -m scmetaintel ...`)
        ├─ enrich      → GEO + PubMed metadata enrichment
        ├─ ontology    → CL / UBERON / MONDO normalization
        ├─ embed       → build vector index
        ├─ retrieve    → hybrid dense+sparse retrieval + reranking
        ├─ answer      → grounded answer generation with citations
        ├─ chat        → interactive dataset search REPL
        └─ eval        → retrieval / ontology evaluation
```

## Project layout

```text
scMetaIntel-Hub/
├── scmetaintel/          # unified intelligence package
├── geodh/                # GEO acquisition bridge package
├── benchmarks/           # benchmark scripts, ground truth, eval queries, results
│   ├── ground_truth/     # 2189 enriched GSE JSON files
│   ├── public_datasets/  # 27 public benchmark dataset configs
│   └── results/          # benchmark output JSONs (gitignored)
├── article_figures/      # generated figures and tables for publication
├── scripts/              # figure generation, model download, data augmentation
├── configs/              # shared YAML configuration
├── docs/                 # architecture notes
├── tests/                # repository health tests
├── data/                 # conventional home for downloads + h5ad outputs
├── enriched_metadata/    # rich study JSONs (written on demand)
├── ontologies/           # CL / UBERON / MONDO
├── qdrant_data/          # local vector store (created on demand)
├── reports/              # generated reports (created on demand)
└── logs/                 # runtime logs (created on demand)
```

## Benchmark suite

The project includes a comprehensive benchmark evaluating the full RAG pipeline:

| Level | Script(s) | Scope |
| ----- | --------- | ----- |
| Data foundation | `01_build_ground_truth.py` | 2189 enriched GSE documents |
| Representation | `02_bench_embeddings.py` | 14 embedding models × 4 sub-tasks |
| Retrieval | `03_bench_retrieval.py` | 6 strategies (dense/sparse/hybrid/rerank) |
| Intelligence | `04_bench_llm.py` | 51 LLMs × 8 domain tasks (66 configs with think ablation) |
| Intelligence | `05_bench_public.py` | 6 models × 27 public datasets across 6 categories |
| Context | `06_bench_context.py`, `07_bench_context_mgmt.py` | Context window curves + 15 management strategies |
| Ablation | `08_bench_ablation.py` | KV cache quantization + context length effects |
| End-to-end | `09_bench_e2e.py` | Full pipeline latency and quality |

All evaluation runs locally via Ollama on RTX 5090 Laptop (24.5 GB VRAM).

See `benchmarks/BENCHMARK_DESIGN.md` for full task definitions and dataset details.

## Repository standards

- **governance**: `LICENSE`, `CONTRIBUTING.md`, `CODEOWNERS`, issue forms, and a pull request template
- **automation**: GitHub Actions workflow at `.github/workflows/repo-health.yml`
- **developer consistency**: `.editorconfig`, `.gitattributes`, `.env.example`, and `pyproject.toml`

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

### 2. Create a local environment file

```bash
cp .env.example .env
```

### 3. Pull models via Ollama

```bash
# Pull all 51 LLM models (parallel, batched)
bash scripts/ollama_parallel_pull.sh

# Or see the download plan for selective pulling
python scripts/model_download_plan.py
```

### 4. Enrich studies

```bash
python -m scmetaintel enrich --gse-list GSE185224 GSE175975
```

### 5. Build ontology cache

```bash
python -m scmetaintel ontology --build-index
```

### 6. Build vector index

```bash
python -m scmetaintel embed --input enriched_metadata
```

### 7. Search or chat

```bash
python -m scmetaintel retrieve --query "human lung fibrosis scRNA-seq"
python -m scmetaintel chat
```

### 8. Use GEO acquisition from the merged project

```bash
python -m scmetaintel geo search --query "lung cancer scRNA-seq" --organism human --max-results 10
```

## Model strategy

### Benchmark-validated recommendations

| Role | Model | Evidence |
| ---- | ----- | -------- |
| Best LLM (composite) | `llama3.1-8b` | Top domain composite score (0.776) across 8 tasks |
| Best embedding (retrieval) | `mxbai-embed-large` | Best R@50 and nDCG@10 |
| Best embedding (ontology) | `sapbert` | Highest ontology recall@1 |
| Reranker | `bge-reranker-v2-m3` | Used in all hybrid+rerank strategies |

### Practical defaults

- **LLM**: Any model from the 51-model Ollama registry (see `scmetaintel/config.py`)
- **Embedding**: `BAAI/bge-m3`
- **Ontology embedding**: `FremyCompany/BioLORD-2023`
- **Reranker**: `BAAI/bge-reranker-v2-m3`

## Contributing and local validation

Before opening a pull request, run the lightweight repository checks:

```bash
python -m compileall scmetaintel geodh benchmarks scripts tests
python -m unittest discover -s tests -p 'test_repository_health.py' -v
```

For contribution expectations, see `CONTRIBUTING.md`.

