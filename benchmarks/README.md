# Benchmarks in scMetaIntel-Hub

## Task Taxonomy

The benchmark suite evaluates the full discovery pipeline across **4 levels**,
each with named sub-tasks. Every sub-task has defined inputs, metrics, and a
clear pass/fail threshold.

```
Level 1 – Data Foundation   (01_build_ground_truth.py)
Level 2 – Representation    (02_bench_embeddings.py)
Level 3 – Retrieval         (03_bench_retrieval.py)
Level 4 – Intelligence      (04_bench_llm.py)
Level 5 – Context & E2E     (05/05b/07)
```

---

### Level 1 – Data Foundation (`01_build_ground_truth.py`)

| Item | Detail |
|------|--------|
| **Goal** | Enrich GSE accessions with structured metadata from GEO + PubMed |
| **Input** | `discovery_index.json` (1357 GSE IDs), GEO/PubMed APIs |
| **Output** | 1 JSON per GSE in `ground_truth/` |
| **Status** | **1994 files built.** 1943 have tissue annotations. 51 fully empty (not used by eval queries). |

---

### Level 2 – Representation (`02_bench_embeddings.py`)

Compares embedding models on 4 sub-tasks:

| Sub-task | What | Metrics |
|----------|------|---------|
| **2A – Semantic Clustering** | Embed all docs, silhouette by organism/series_type | `silhouette_organism`, `silhouette_series_type` |
| **2B – Retrieval Accuracy** | Embed queries + docs, rank by cosine sim | `P@5`, `P@10`, `R@50`, `MRR`, `nDCG@10` |
| **2C – Ontology Matching** | Match raw tissue/cell terms to CL/UBERON canonical names | `recall@1`, `recall@5` |
| **2D – Encoding Speed** | Encode 100 sentences | `tokens/sec` |

**Required data:** ground truth docs, eval queries, ontology OBO files (`cl.obo`, `uberon-basic.obo`)

---

### Level 3 – Retrieval (`03_bench_retrieval.py`)

Compares 6 retrieval strategies over the same eval queries:

| Strategy | Description |
|----------|-------------|
| `dense` | Dense vector search only |
| `sparse` | Bag-of-words overlap only |
| `hybrid` | Dense + sparse via RRF |
| `hybrid+filter` | + payload filter (organism) |
| `hybrid+rerank` | + neural reranker |
| `hybrid+filter+rerank` | Full pipeline |

**Metrics per strategy:** `P@5`, `P@10`, `R@50`, `MRR`, `nDCG@10`, `avg_latency_ms`
**Broken down by:** difficulty (easy/medium/hard)
**Required data:** ground truth docs, eval queries, Qdrant, embedding model, reranker

---

### Level 4 – Intelligence (`04_bench_llm.py`)

Compares all LLMs (17 models x 2 think modes where applicable) across 5 sub-tasks:

| Sub-task | What | Input | Gold Standard | Metrics |
|----------|------|-------|---------------|---------|
| **4A – Query Parsing** | Parse NL query → structured JSON (organism/tissue/disease/assay/cell_type) | eval queries | `expected_constraints` | `exact_match`, `field_accuracy` |
| **4B – Metadata Extraction** | Extract tissues/diseases/cell_types from GEO title+summary | ground truth docs | `tissues`, `diseases`, `cell_types` arrays | per-field `P`, `R`, `F1` |
| **4C – Ontology Normalization** | Map raw terms → UBERON/CL/MONDO IDs | ground truth + OBO files | OBO term→ID lookup | `accuracy`, `recall`, `F1` |
| **4D – Answer Generation** | Generate cited answer from query + context | eval queries + docs | `expected_gse` | `citation_precision`, `citation_recall`, `grounding_rate` |
| **4E – Inference Speed** | Raw throughput measurement | fixed prompt | wall-clock | `tokens/sec` |

**Required data:** Ollama server + models, eval queries, ground truth docs, ontology OBO files

---

### Level 5 – Context & End-to-End

| Script | What | Metrics |
|--------|------|---------|
| `05_bench_context.py` | Context window management strategies | context utilization, answer quality |
| `05b_bench_context_management.py` | MMR diversity, recency reorder, ontology expansion, multi-step retrieval | quality lift per strategy |
| `07_bench_e2e.py` | Full pipeline: query → parse → retrieve → rerank → answer | end-to-end latency, quality |

---

## Eval Queries

**90 queries** in `eval_queries.json`, categorized:

| Category | Count | Difficulty | Example |
|----------|-------|------------|---------|
| `basic` | 19 | easy | "human brain scRNA-seq" |
| `disease` | 23 | medium | "atherosclerosis single-cell RNA-seq" |
| `cell_type` | 12 | medium | "T cell single-cell RNA-seq" |
| `multi_constraint` | 15 | hard | "mouse myocardial infarction single-cell RNA-seq" |
| `natural_language` | 8 | hard | "What datasets study the tumor microenvironment?" |
| `modality` | 7 | medium | "Perturb-seq CRISPR screen single-cell" |
| `organism` | 6 | easy–medium | "Xenopus single-cell RNA-seq" |

Each query has `expected_gse` (2–5 GSE IDs) for measuring retrieval precision/recall.
All 227 unique expected GSEs are present in `ground_truth/`.

## Ground Truth Corpus

- **2189 enriched GSE JSON files** in `ground_truth/`
- **1943** have non-empty tissue annotations
- **1064** have non-empty cell_type annotations
- **186** have non-empty disease annotations
- Schema: `gse_id`, `title`, `summary`, `organism`, `tissues`, `diseases`, `cell_types`, `document_text`, `pubmed`, `characteristics_summary`, `samples`

## Running Benchmarks

```bash
# Requires: dl conda env for GPU, Ollama running for LLM tasks
conda activate dl

# Level 1 – rebuild ground truth (slow, hits GEO/PubMed APIs)
python benchmarks/01_build_ground_truth.py

# Level 2 – embedding comparison
python benchmarks/02_bench_embeddings.py

# Level 3 – retrieval strategy comparison
python benchmarks/03_bench_retrieval.py

# Level 4 – LLM comparison (start Ollama first)
ollama serve &
python benchmarks/04_bench_llm.py

# Quick smoke test
python benchmarks/quick_smoke_test.py
```

## Results

Benchmark results are saved to `results/` as JSON files. Previous runs:

- `embedding_bench.json` – embedding model comparison
- `retrieval_bench.json` – retrieval strategy comparison
- `llm_bench.json` – LLM comparison (all 17 models, think ablation)
- `context_bench.json` / `context_management_bench.json` – context strategies
