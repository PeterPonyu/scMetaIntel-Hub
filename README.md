<div align="center">
  <a href="https://peterponyu.github.io/">
    <img src="https://peterponyu.github.io/assets/badges/scMetaIntel-Hub.svg" width="64" alt="ZF Lab · scMetaIntel-Hub">
  </a>
</div>

# scMetaIntel-Hub

[![Repo health](https://github.com/PeterPonyu/scMetaIntel-Hub/actions/workflows/repo-health.yml/badge.svg)](https://github.com/PeterPonyu/scMetaIntel-Hub/actions/workflows/repo-health.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`scMetaIntel-Hub` is a local-first software toolkit for GEO acquisition, metadata enrichment, ontology normalization, vector indexing, retrieval, and grounded answering.

## What is public in this repository

- `geodh/`: GEO search, download, verification, conversion, and manifest tooling
- `scmetaintel/`: enrichment, ontology, embedding, retrieval, answer generation, and chat
- `benchmarks/05_bench_public.py`: optional public-dataset evaluation harness
- `benchmarks/public_datasets/`: public benchmark dataset adapters
- `configs/`, `docs/`, `tests/`: public software configuration, architecture notes, and repo health checks

Only public runtime assets are kept in this repository.

## Project layout

```text
scMetaIntel-Hub/
├── scmetaintel/          # metadata intelligence package
├── geodh/                # GEO acquisition package
├── benchmarks/           # public evaluation harnesses and public datasets
├── scripts/              # public utility scripts
├── configs/              # shared YAML configuration
├── docs/                 # public architecture notes
├── tests/                # repository and public-surface tests
├── data/                 # local downloads and outputs (created on demand)
├── enriched_metadata/    # generated study JSONs (created on demand)
├── ontologies/           # local ontology cache (created on demand)
├── qdrant_data/          # local vector store (created on demand)
├── reports/              # generated reports (created on demand)
└── logs/                 # runtime logs (created on demand)
```

## Prerequisites

- **Python 3.10+**.
- **Ollama server.** The `enrich`, `embed`, `retrieve`, and `chat` steps call a
  local LLM through Ollama, which defaults to `http://localhost:11434`. Install
  Ollama and keep it running (`ollama serve`) before invoking those commands,
  then pull the models you intend to use (see step 3). Without a running Ollama
  server the first command fails with a bare connection error.
- **Compute device.** The default device is CUDA (`SCMETA_DEVICE=cuda`). If you
  do not have a CUDA-capable GPU, override it to CPU, for example
  `export SCMETA_DEVICE=cpu` or set `SCMETA_DEVICE=cpu` in your `.env` file.

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

### 3. Pull the models you need

```bash
bash scripts/ollama_parallel_pull.sh
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

> **Upgrading an existing index?** The default embedding model is `bge-m3`. If you
> have a `qdrant_data/` collection built with a different embedding model, re-index it
> (delete the stale collection or re-run `embed` against a fresh store) — embeddings
> from different models are not comparable even when their vector dimensions match.

### 7. Search or chat

```bash
python -m scmetaintel retrieve --query "human lung fibrosis scRNA-seq"
python -m scmetaintel chat
```

### 8. Use GEO acquisition from the unified CLI

```bash
python -m scmetaintel geo search --query "lung cancer scRNA-seq" --organism human --max-results 10
```

## Public benchmark harness

This repo keeps one public evaluation lane for generally available datasets:

```bash
python benchmarks/05_bench_public.py --max-samples 50
```

The public harness is optional and is not required to use the runtime packages.

## Contributing and local validation

Before opening a pull request, run the lightweight repository checks:

The repository health workflow lives at `.github/workflows/repo-health.yml`.

```bash
python -m compileall scmetaintel geodh benchmarks scripts tests
python -m unittest discover -s tests -p 'test_*.py' -v
```

For contribution expectations, see `CONTRIBUTING.md`.
