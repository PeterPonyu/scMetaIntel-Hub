# scMetaIntel-Hub audit — 2026-03-19

## 1. Functional merge status

### What is fully present inside the merged project

The merged `scmetaintel/` package now contains the full file-level coverage of both source `scmetaintel` packages:

- `__init__.py`
- `__main__.py`
- `answer.py`
- `chat.py`
- `config.py`
- `embed.py`
- `enrich.py`
- `eval.py`
- `evaluate.py`
- `models.py`
- `ontology.py`
- `retrieve.py`

Coverage check result:

- legacy `scMetaIntel/scmetaintel`: no missing files in merged package
- `GEO-DataHub/scmetaintel`: no missing files in merged package
- merged package adds `evaluate.py` as an extra compatibility layer

### What is not yet fully vendored from GEO-DataHub

These GEO-DataHub top-level acquisition modules are **not copied as standalone source files** into `scMetaIntel-Hub` root yet:

- `geo_cancer_gap.py`
- `geo_classifier.py`
- `geo_convert.py`
- `geo_download.py`
- `geo_manifest.py`
- `geo_search.py`
- `geo_theme.py`
- `geo_verify.py`
- `geodh.py`
- `pipeline.py`

However, their functionality is still available through the bridge:

- `geodh/cli.py`
- `python -m scmetaintel geo ...`

So the answer is:

- **full intelligence-layer merge:** yes
- **full GEO source vendoring:** not yet
- **full GEO functionality access:** yes, via bridge

## 2. Benchmark infrastructure status

### Newly present in the merged project

The merged `benchmarks/` folder now contains:

- `01_build_ground_truth.py`
- `02_bench_embeddings.py`
- `03_bench_retrieval.py`
- `04_bench_llm.py`
- `05_bench_context.py`
- `06_bench_finetune.py`
- `07_bench_e2e.py`
- `model_cache_check.py`
- `model_subsection_summary.py`
- `eval_queries.json`

### Benchmark assets now configured

- ontology files copied into merged project
  - `cl.obo`
  - `mondo.obo`
  - `uberon-basic.obo`
- ground-truth corpus seeded
  - `53` study JSON files currently present in `benchmarks/ground_truth/`

## 3. Runtime and hardware readiness

From `reports/runtime_readiness.json`:

- GPU: `NVIDIA GeForce RTX 5090 Laptop GPU`
- VRAM: `25.12 GB`
- CUDA: `13.0`
- PyTorch: `2.9.0+cu130`
- Ollama models pulled:
  - `qwen2.5:1.5b`
  - `qwen2.5:0.5b`

Benchmark dependency stack now present:

- `pronto`
- `trl`
- `peft`
- `bitsandbytes`
- `accelerate`
- `datasets`

## 4. Model availability vs operational readiness

### LLMs

Configured in merged config:

- `qwen3.5-27b`
- `qwen3.5-9b`
- `qwen3.5-9b-q8`
- `qwen3-32b`
- `qwen3-14b`
- `qwen3-14b-q8`
- `qwen3-8b`
- `qwen2.5-7b`
- `qwen2.5-1.5b`
- `qwen2.5-0.5b`

Actually pulled in Ollama right now:

- `qwen2.5:1.5b`
- `qwen2.5:0.5b`

Operational probe status:

- `qwen2.5-0.5b`: operational
- `qwen2.5-1.5b`: operational
- `qwen3.5-9b`: attempted pull, not completed
- `qwen3.5-27b`: configured but not pulled

### Embeddings

Cache availability after refresh:

- cached `9 / 13`
- notable cached models:
  - `qwen3-embed-8b`
  - `qwen3-embed-4b`
  - `qwen3-embed-0.6b`  ← newly downloaded
  - `bge-m3`
  - `nomic-embed`
  - `gte-large`
  - `biolord-2023`
  - `biomedbert-base`
  - `biomedbert-large`

Operational embedding probe:

Operational:

- `qwen3-embed-0.6b`
- `bge-m3`
- `nomic-embed`
- `biolord-2023`
- `biomedbert-base`

Not operational in current loader environment:

- `qwen3-embed-4b`
- `gte-large`

### Rerankers

Cache availability after refresh:

- cached `2 / 4`
- cached:
  - `qwen3-reranker-0.6b`  ← newly downloaded
  - `bge-reranker-v2-m3`

Operational reranker probe:

- `qwen3-reranker-0.6b`: operational
- `bge-reranker-v2-m3`: operational

## 5. Benchmark-readiness by subsystem

### Ready now

#### LLM benchmarking

Verified by running:

- `benchmarks/04_bench_llm.py --models qwen2.5-0.5b`

This completed successfully and wrote:

- `benchmarks/results/llm_bench.json`

#### Fine-tuning benchmark scaffold

Verified by running:

- `benchmarks/06_bench_finetune.py --task llm`

This completed successfully and wrote:

- `benchmarks/finetuned/llm_training_data.jsonl`
- `benchmarks/results/finetune_bench.json`

#### Ontology normalization stack

Ready now because:

- ontology assets are present locally
- `pronto` is installed
- `biolord-2023` is cached and operational

#### Local inference stack

Ready now because:

- Ollama server is running
- two local LLMs are pulled and responsive
- `answer.py` operational probe succeeded

### Partially ready / constrained

#### Retrieval / RAG benchmarking

Status: **partially ready**

What is ready:

- merged retrieval code exists
- Qdrant local path works
- rerankers probe successfully
- several embedding backbones probe successfully

What is still unstable:

- `benchmarks/03_bench_retrieval.py` with `bge-m3` is still tripping over upstream loader behavior involving `FlagEmbedding` vs current `transformers`, plus noisy Hugging Face conversion/network behavior in benchmark execution context

Interpretation:

- the **RAG stack itself is configured**
- the **benchmark script path for dense/hybrid retrieval still needs one more stabilization pass** if you want a clean one-command run with `bge-m3`

#### Context benchmarking

Status: **software-ready, not yet smoke-run in this audit**

Why:

- uses the same LLM answer path that is already operational
- depends on retrieval context and local LLM stack
- should be runnable with the pulled Qwen2.5 models, but was not executed in this audit to save time

#### Embedding fine-tuning and reranker fine-tuning

Status: **framework-ready, not yet executed in this audit**

Why:

- packages are installed
- scripts are present
- cached backbone models exist for fine-tuning
- not run end-to-end during this pass

## 6. What was newly fixed/configured during this audit

- copied missing benchmark scripts into merged project
- copied ontology assets into merged project
- seeded merged ground-truth benchmark corpus
- installed actual missing fine-tune/ontology deps:
  - `pronto`
  - `trl`
- downloaded missing lightweight benchmark models:
  - `Qwen/Qwen3-Embedding-0.6B`
  - `Qwen/Qwen3-Reranker-0.6B`
- improved benchmark reporting scripts
- improved runtime fallbacks for embedding/reranker loading
- patched `07_bench_e2e.py` to choose best available pulled LLM fallback if requested models are absent

## 7. Bottom-line assessment

### Is the merged project already complete relative to both source repos?

**Not completely.**

- intelligence layer: **yes, fully merged**
- benchmark layer: **now largely merged**
- GEO acquisition source files: **not fully vendored, still bridged**

### What is ready for benchmarking right now?

#### Benchmark-ready now

- local LLM benchmarking on pulled Qwen2.5 models
- ontology normalization benchmarking stack
- fine-tuning data-prep / QLoRA scaffold
- reranker benchmarking with cached operational rerankers

#### Configured and mostly ready, but not fully stabilized

- dense / hybrid retrieval benchmark path
- context-efficiency benchmark path
- end-to-end benchmark path

### Highest-value next step

If you want the merged project to move from "practical baseline benchmark ready" to "full planned benchmark ready", the next most valuable step is:

1. stabilize `03_bench_retrieval.py` for `bge-m3`
2. then run `05_bench_context.py`
3. then pull at least one medium LLM such as `qwen3.5:9b`

That would unlock a much stronger benchmark story than the current small-model-only LLM baseline.
