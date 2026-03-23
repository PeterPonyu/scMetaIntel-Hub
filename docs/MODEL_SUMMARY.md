# Model Summary by Subsystem

This summary reflects the merged `scMetaIntel-Hub` design and the current desktop state.

## 1. Query parsing and lightweight metadata extraction

### Recommended parsing model

- `qwen3.5-9b`
  - role: fast structured extraction
  - strengths: large context, strong instruction following

### Practical parsing default in this merged project

- `qwen2.5-1.5b`
  - role: local smoke-test / lightweight parsing
  - reason: already small enough for fast local use

### Tiny fallback

- `qwen2.5-0.5b`
  - role: wiring tests only
  - caution: accuracy is limited

### Confirmed local Ollama availability on this desktop

- `qwen2.5:1.5b`
- `qwen2.5:0.5b`

## 2. Grounded answer generation

### Recommended answer-generation model

- `qwen3.5-27b`
  - role: primary answer generation
  - strengths: best reasoning quality among configured local options

### Good medium option

- `qwen3-14b`
  - role: answer generation with lower VRAM cost

### Practical answer-generation default in this merged project

- `qwen2.5-1.5b`
  - role: immediate local runtime default
  - reason: lets the merged project run now without waiting for large downloads

## 3. Main retrieval embeddings

### Recommended frontier dense models

- `qwen3-embed-8b`
  - role: highest-quality dense embedding
- `qwen3-embed-4b`
  - role: balanced quality / VRAM
- `qwen3-embed-0.6b`
  - role: lightweight dense fallback

### Practical retrieval default

- `bge-m3`
  - role: primary retrieval model in this merged project
  - strengths: dense + sparse + ColBERT-style support in one model
  - why default: strongest practical hybrid-search choice

### Additional general baselines

- `nomic-embed`
- `gte-large`

### Confirmed general-embedding cache presence on this desktop

- `Qwen/Qwen3-Embedding-8B`
- `Qwen/Qwen3-Embedding-4B`
- `BAAI/bge-m3`
- `nomic-ai/nomic-embed-text-v1.5`
- `Alibaba-NLP/gte-large-en-v1.5`

## 4. Biomedical / ontology-focused embeddings

### Primary ontology model

- `biolord-2023`
  - role: ontology normalization and biomedical concept matching
  - target ontologies: CL, UBERON, MONDO

### Additional biomedical baselines

- `biomedbert-base`
- `biomedbert-large`
- `medcpt-query`
- `medcpt-article`

### Scientific retrieval specialist

- `specter2`
- `specter2-query`

### Confirmed biomedical-embedding cache presence on this desktop

- `FremyCompany/BioLORD-2023`
- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`
- `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`

## 5. Reranking

### Recommended frontier rerankers

- `qwen3-reranker-4b`
  - role: primary high-quality reranker
- `qwen3-reranker-0.6b`
  - role: lighter reranker fallback

### Practical reranker default

- `bge-reranker-v2-m3`
  - role: current default reranker in the merged project
  - reason: stable, well understood, already aligned with `bge-m3`

### Additional alternative

- `bge-reranker-v2-gemma`

### Confirmed reranker cache presence on this desktop

- `BAAI/bge-reranker-v2-m3`

## 6. End-to-end recommended stacks

### A. Best-quality target stack

- query parsing: `qwen3.5-9b`
- answer generation: `qwen3.5-27b`
- embedding: `qwen3-embed-8b`
- ontology matching: `biolord-2023`
- reranker: `qwen3-reranker-4b`

### B. Practical hybrid stack for this desktop right now

- query parsing: `qwen2.5-1.5b`
- answer generation: `qwen2.5-1.5b`
- embedding: `bge-m3`
- ontology matching: `biolord-2023`
- reranker: `bge-reranker-v2-m3`

### C. Minimal smoke-test stack

- query parsing: `qwen2.5-0.5b`
- answer generation: `qwen2.5-0.5b`
- embedding: `bge-m3`
- ontology matching: `biolord-2023`
- reranker: optional / skip

## 7. Recommendation for immediate next use

If you want the merged project to be useful immediately without more downloads:

- keep default LLM = `qwen2.5-1.5b`
- keep default embedding = `bge-m3`
- keep ontology encoder = `biolord-2023`
- keep reranker = `bge-reranker-v2-m3`

If you want the best eventual local quality:

- pull `qwen3.5:9b`
- pull `qwen3.5:27b`
- evaluate whether `qwen3-embed-8b` and `qwen3-reranker-4b` are worth the extra VRAM/runtime cost in your workflow
