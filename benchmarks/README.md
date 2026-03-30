# Benchmarks in scMetaIntel-Hub

## Task Taxonomy

The benchmark suite evaluates the full discovery pipeline across **5 levels**,
each with named sub-tasks. Every sub-task has defined inputs, metrics, and a
clear pass/fail threshold.

```text
Level 1 – Data Foundation   (01_build_ground_truth.py)
Level 2 – Representation    (02_bench_embeddings.py)
Level 3 – Retrieval         (03_bench_retrieval.py)
Level 4 – Intelligence      (04_bench_llm.py, 05_bench_public.py)
Level 5 – Context & E2E     (06/07/08/09)
```

---

### Level 1 – Data Foundation (`01_build_ground_truth.py`)

| Item | Detail |
|------|--------|
| **Goal** | Enrich GSE accessions with structured metadata from GEO + PubMed |
| **Input** | `discovery_index.json` (GSE IDs), GEO/PubMed APIs |
| **Output** | 1 JSON per GSE in `ground_truth/` |
| **Status** | **2189 files built.** |

---

### Level 2 – Representation (`02_bench_embeddings.py`)

Compares 14 embedding models on 4 sub-tasks:

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

### Level 4 – Intelligence

#### Domain tasks (`04_bench_llm.py`)

Compares 51 LLMs (66 configs with think variants) across 8 sub-tasks:

| Sub-task | What | Input | Gold Standard | Metrics |
|----------|------|-------|---------------|---------|
| **A – Query Parsing** | Parse NL query → structured JSON | eval queries | `expected_constraints` | `exact_match`, `field_accuracy` |
| **B – Metadata Extraction** | Extract tissues/diseases/cell_types from GEO text | ground truth docs | `tissues`, `diseases`, `cell_types` arrays | per-field `P`, `R`, `F1` |
| **C – Ontology Normalization** | Map raw terms → UBERON/CL/MONDO IDs | ground truth + OBO | OBO term→ID lookup | `accuracy`, `recall`, `F1` |
| **D – Answer Generation** | Generate cited answer from query + context | eval queries + docs | `expected_gse` | `citation_precision`, `citation_recall`, `grounding_rate` |
| **E – Inference Speed** | Raw throughput measurement | fixed prompts | wall-clock | `tokens/sec` |
| **F – Relevance Judgment** | Binary query-document classification | queries × docs | ground truth labels | `accuracy`, `P`, `R`, `F1` |
| **G – Domain Classification** | Assign domain label to study | ground truth docs | curated domain labels | `accuracy`, per-domain breakdown |
| **H – Organism & Modality** | Extract organism and assay modality | ground truth docs | curated org/modality | organism `accuracy`, modality `set F1` |

**Required data:** Ollama server + models, eval queries, ground truth docs, ontology OBO files

#### Public benchmarks (`05_bench_public.py`)

Evaluates models on 27 public datasets across 6 categories:
General (4), Reasoning (3), Biomedical (11), Structured (3), Tool-use (3), Commonsense (3).

See `BENCHMARK_DESIGN.md` for full dataset details and evaluation methods.

---

### Level 5 – Context, Ablation & End-to-End

| Script | What | Metrics |
|--------|------|---------|
| `06_bench_context.py` | Context window management (k × format curves) | context utilization, answer quality |
| `07_bench_context_mgmt.py` | MMR diversity, recency reorder, ontology expansion, multi-step retrieval (15 strategies) | quality lift per strategy |
| `08_bench_ablation.py` | KV cache quantization (f16/q8_0/q4_0) + context length (2K–16K) | Δ accuracy, Δ speed, VRAM |
| `09_bench_e2e.py` | Full pipeline: query → parse → retrieve → rerank → answer | end-to-end latency, quality |

---

## Eval Queries

**171 queries** in `eval_queries.json`, categorized:

| Category | Count | Difficulty | Example |
|----------|-------|------------|---------|
| `basic` | 35 | easy | "human brain scRNA-seq" |
| `disease` | 33 | medium | "atherosclerosis single-cell RNA-seq" |
| `multi_constraint` | 31 | hard | "mouse myocardial infarction single-cell RNA-seq" |
| `cell_type` | 29 | medium | "T cell single-cell RNA-seq" |
| `modality` | 18 | medium | "Perturb-seq CRISPR screen single-cell" |
| `natural_language` | 13 | hard | "What datasets study the tumor microenvironment?" |
| `organism` | 12 | easy–medium | "Xenopus single-cell RNA-seq" |

Each query has `expected_gse` (2–5 GSE IDs) for measuring retrieval precision/recall.
All 540 unique expected GSEs are present in `ground_truth/`.

## Ground Truth Corpus

- **2189 enriched GSE JSON files** in `ground_truth/`
- Schema: `gse_id`, `title`, `summary`, `organism`, `tissues`, `diseases`, `cell_types`, `document_text`, `pubmed`, `characteristics_summary`, `samples`

## Running Benchmarks

```bash
# Requires: dl conda env for GPU, Ollama running for LLM tasks
conda activate dl

# Level 1 – rebuild ground truth (slow, hits GEO/PubMed APIs)
python benchmarks/01_build_ground_truth.py

# Level 2 – embedding comparison (14 models)
python benchmarks/02_bench_embeddings.py

# Level 3 – retrieval strategy comparison (6 strategies)
python benchmarks/03_bench_retrieval.py

# Level 4 – LLM domain tasks (51 models × 8 tasks)
ollama serve &
python benchmarks/04_bench_llm.py

# Level 4 – public benchmarks (27 datasets)
python benchmarks/05_bench_public.py --no-think-ablation

# Level 5 – context, ablation, e2e
python benchmarks/06_bench_context.py
python benchmarks/07_bench_context_mgmt.py
python benchmarks/08_bench_ablation.py --models qwen3-8b llama3.1-8b phi4-14b-q8 gemma3-12b-q8 qwen3.5-9b-q8
python benchmarks/09_bench_e2e.py

# Generate all article figures and tables
python scripts/generate_article_figures.py
```

## Results

Benchmark results are saved to `results/` as JSON files:

| File | Content |
|------|---------|
| `embedding_bench.json` | 14 embedding models comparison |
| `retrieval_bench.json` | 6 retrieval strategy comparison |
| `llm_bench.json` | 66 LLM configs (51 base + 15 think), 8 domain tasks |
| `public_bench.json` | 6 models × 27 public datasets |
| `context_bench.json` | Context window optimisation curves |
| `context_management_bench.json` | 15 context management strategies |
| `ablation_bench.json` | KV cache (3 types) + context length (4 sizes) × 5 models |
| `e2e_report.json` | End-to-end pipeline comparison (4 configs) |
