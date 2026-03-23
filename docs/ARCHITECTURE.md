# scMetaIntel-Hub Architecture

## Integration goal

The merged project unifies two previously separate responsibilities:

- **data acquisition**: GEO search / scan / verify / download / convert / organize
- **metadata intelligence**: enrich / normalize / embed / retrieve / rerank / answer / evaluate

## Phase-1 integration design

### 1. Acquisition remains stable

The existing GEO-DataHub codebase already provides a mature CLI and file-processing pipeline. Instead of copying every module prematurely, the merged project exposes it through an internal bridge:

- `geodh/cli.py`
- command path: `python -m scmetaintel geo ...`

This preserves stability while letting the merged project become the new umbrella entrypoint.

### 2. Intelligence is first-class and local

The `scmetaintel/` package in this repo is the canonical merged implementation.

It combines:

- dataclass models and interactive UX from the standalone `scMetaIntel`
- richer model registry and benchmark-oriented utilities from `GEO-DataHub/scmetaintel`

## Core subsystems

### Enrichment

Input:

- GEO SOFT metadata
- GSM sample characteristics
- PubMed abstracts / MeSH terms
- optional external `dataset_meta.json` from GEO-DataHub downloads

Output:

- rich `EnrichedStudy` JSONs for chat/retrieval
- simplified benchmark-style documents for evaluation

### Ontology normalization

Three-layer normalization:

1. exact / synonym lookup
2. embedding similarity with BioLORD-2023
3. future LLM disambiguation hook

Targets:

- CL
- UBERON
- MONDO

### Embedding and indexing

Two APIs coexist intentionally:

- `Embedder` → benchmark-friendly low-level interface
- `StudyEmbedder` → project-friendly high-level interface

Storage:

- local on-disk Qdrant

### Retrieval

Two styles coexist intentionally:

- `RetrievalPipeline` → benchmark-style dense / sparse / hybrid / rerank experiments
- `HybridRetriever` → user-facing search and chat retrieval wrapper

### Answer generation

Two styles coexist intentionally:

- helper functions (`generate_answer`, `parse_query`, `extract_metadata`)
- class API (`AnswerGenerator`)

This was deliberate to support both scripted evaluation and interactive usage.

## Model policy

### Recommended models

These are the architectural targets for best quality:

- LLM: `qwen3.5-27b`
- fast LLM: `qwen3.5-9b`
- embedding: `qwen3-embed-8b`
- reranker: `qwen3-reranker-4b`

### Practical defaults

These are chosen to run immediately on the current desktop setup:

- LLM: `qwen2.5-1.5b`
- embedding: `bge-m3`
- ontology embedding: `biolord-2023`
- reranker: `bge-reranker-v2-m3`

## Migration roadmap

### Phase 1 (done here)

- create merged folder
- unify config and model registry
- merge core intelligence package
- expose GEO CLI via bridge
- add shared docs, config, scripts

### Phase 2

- vendor `geo_*.py` modules into `geodh/`
- port benchmark scripts into this repo
- standardize evaluation assets

### Phase 3

- add tests and packaging
- add web/API layer
- add CI and reproducible environment definitions
