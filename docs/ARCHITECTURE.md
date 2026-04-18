# scMetaIntel-Hub Architecture

## Scope

This repository is the public software surface for two integrated capabilities:

- GEO acquisition and local dataset management
- metadata intelligence for enrichment, normalization, retrieval, and grounded answering

This document covers only the public runtime architecture.

## High-level flow

```text
User Query
   │
   ├─ GEO acquisition bridge (`python -m scmetaintel geo ...`)
   │    └─ delegates to the GEO-DataHub-compatible CLI
   │
   └─ Intelligence layer (`python -m scmetaintel ...`)
        ├─ enrich      → GEO + PubMed metadata enrichment
        ├─ ontology    → CL / UBERON / MONDO normalization
        ├─ embed       → vector index construction
        ├─ retrieve    → hybrid retrieval and reranking
        ├─ answer      → grounded answers with citations
        └─ chat        → interactive dataset exploration
```

## Packages

### `geodh/`

Public acquisition utilities for:

- GEO search
- download and verification
- format conversion
- manifest generation
- dataset organization

### `scmetaintel/`

Public metadata-intelligence utilities for:

- metadata enrichment
- ontology normalization
- embedding and indexing
- retrieval and reranking
- answer generation
- interactive chat
- evaluation helpers used by the public runtime surface

## Storage and local state

- `data/`: downloads and local outputs
- `enriched_metadata/`: generated study JSONs
- `ontologies/`: ontology cache
- `qdrant_data/`: local vector store
- `reports/`: generated runtime reports
- `logs/`: local execution logs

These are local working directories, not part of the committed public artifact set.

## Public evaluation surface

The public repository keeps a lightweight benchmark harness for public datasets:

- `benchmarks/05_bench_public.py`
- `benchmarks/public_datasets/`

This surface is optional and separate from the core runtime packages.

## Quality gates

Public release hygiene is enforced by:

- `tests/test_repository_health.py`
- `.github/workflows/repo-health.yml`
- `pyproject.toml`
- `CONTRIBUTING.md`
