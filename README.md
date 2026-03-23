# scMetaIntel-Hub

A unified project that merges:

- **GEO-DataHub** → acquisition backbone for GEO search, download, verification, conversion, and organization
- **scMetaIntel** → metadata intelligence layer for enrichment, ontology normalization, embedding, hybrid retrieval, reranking, evaluation, and chat

## Why this merged project exists

On this desktop there are two strong but separate codebases:

- `/home/zeyufu/Desktop/GEO-DataHub`
- `/home/zeyufu/Desktop/scMetaIntel`

`scMetaIntel-Hub` is the integration layer that turns them into one coherent system.

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
├── benchmarks/           # merged benchmark workspace
├── configs/              # shared YAML configuration
├── docs/                 # architecture and merge notes
├── scripts/              # helper scripts
├── data/                 # downloads + h5ad outputs
├── enriched_metadata/    # rich study JSONs
├── ontologies/           # CL / UBERON / MONDO
├── qdrant_data/          # local vector store
├── reports/              # generated reports
└── logs/                 # runtime logs
```

## What is fully integrated now

- unified `scmetaintel` package with:
  - shared config + model registry
  - dataclass models
  - enrichment
  - ontology normalization
  - embedding + indexing
  - hybrid retrieval
  - grounded answer generation
  - chat REPL
  - evaluation CLI
- `geodh` bridge so acquisition commands are accessible from the merged project immediately
- merged dependency list and shared YAML config

## What is intentionally phase-1 bridged

The GEO acquisition CLI is exposed through `geodh/cli.py`, which currently delegates to the external `GEO-DataHub` repo instead of duplicating every `geo_*.py` file immediately. This keeps the merged project usable now while preserving the stable acquisition pipeline.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull small local models first

```bash
bash scripts/pull_models.sh
```

### 3. Enrich studies

```bash
python -m scmetaintel enrich --gse-list GSE185224 GSE175975
```

### 4. Build ontology cache

```bash
python -m scmetaintel ontology --build-index
```

### 5. Build vector index

```bash
python -m scmetaintel embed --input enriched_metadata
```

### 6. Search or chat

```bash
python -m scmetaintel retrieve --query "human lung fibrosis scRNA-seq"
python -m scmetaintel chat
```

### 7. Use GEO acquisition from the merged project

```bash
python -m scmetaintel geo search --query "lung cancer scRNA-seq" --organism human --max-results 10
```

## Model strategy

### Practical defaults

- **LLM**: `qwen2.5:1.5b`
- **Embedding**: `BAAI/bge-m3`
- **Ontology embedding**: `FremyCompany/BioLORD-2023`
- **Reranker**: `BAAI/bge-reranker-v2-m3`

These defaults are chosen to work on the current machine without requiring the biggest frontier downloads first.

### Recommended frontier upgrades

- **LLM**: `qwen3.5:27b`
- **Fast LLM**: `qwen3.5:9b`
- **Dense embedding**: `Qwen/Qwen3-Embedding-8B`
- **Reranker**: `Qwen/Qwen3-Reranker-4B`

## Next refinement steps

1. Vendor the `geo_*.py` acquisition modules into `geodh/`
2. Port the full benchmark script suite into this repo’s `benchmarks/`
3. Add a proper `pyproject.toml`
4. Add end-to-end smoke tests for acquisition → enrichment → retrieval → answer
5. Add a lightweight web UI on top of the chat and search APIs
