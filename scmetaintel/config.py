"""
Unified configuration for scMetaIntel-Hub.

This merges:
- the richer model registry and benchmark settings from `GEO-DataHub/scmetaintel`
- the path/runtime singleton pattern from the standalone `scMetaIntel` project

Design principle:
- keep *recommended* frontier models in the registry
- default to *locally runnable* models for first-run smoke tests
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESKTOP_ROOT = PROJECT_ROOT.parent
EXTERNAL_GEO_DATAHUB = PROJECT_ROOT / "geodh"  # vendored; downloads/ may not exist
EXTERNAL_SCMETAINTEL = None  # legacy, fully superseded

HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# Compatibility path constants (used by benchmark-style modules)
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADS_DIR = DATA_DIR / "downloads"
H5AD_DIR = DATA_DIR / "h5ad_output"
ONTOLOGY_DIR = PROJECT_ROOT / "ontologies"
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
GROUND_TRUTH_DIR = BENCHMARK_DIR / "ground_truth"
RESULTS_DIR = BENCHMARK_DIR / "results"
QDRANT_DIR = PROJECT_ROOT / "qdrant_data"
ACCESSION_INDEX = PROJECT_ROOT / "local_accession_index.json"

QDRANT_COLLECTION = "scmetaintel_studies"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: Dict[str, dict] = {
    # frontier dense embedders
    "qwen3-embed-8b": {
        "name": "Qwen/Qwen3-Embedding-8B",
        "dim": 4096,
        "max_tokens": 32768,
        "type": "dense",
        "tier": "general",
        "mrl": True,
        "vram_gb": 16,
        "note": "Best-quality dense embedder; use when VRAM allows.",
    },
    "qwen3-embed-4b": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "max_tokens": 32768,
        "type": "dense",
        "tier": "general",
        "mrl": True,
        "vram_gb": 8,
        "note": "Balanced quality / VRAM.",
    },
    "qwen3-embed-0.6b": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dim": 1024,
        "max_tokens": 32768,
        "type": "dense",
        "tier": "general",
        "mrl": True,
        "vram_gb": 1.2,
        "note": "Lightweight dense fallback.",
    },

    # hybrid retrieval workhorse
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "dim": 1024,
        "max_tokens": 8192,
        "type": "dense+sparse+colbert",
        "tier": "general",
        "mrl": False,
        "vram_gb": 1.1,
        "note": "Primary practical default: dense+sparse hybrid retrieval in one model.",
    },
    "nomic-embed": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "max_tokens": 8192,
        "type": "dense",
        "tier": "general",
        "mrl": True,
        "vram_gb": 0.3,
        "note": "Very small long-context general baseline.",
    },
    "gte-large": {
        "name": "Alibaba-NLP/gte-large-en-v1.5",
        "dim": 1024,
        "max_tokens": 8192,
        "type": "dense",
        "tier": "general",
        "mrl": False,
        "vram_gb": 1.6,
        "device": "cpu",
        "disabled": True,
        "note": "Broken with transformers>=5.0 (rotary embedding IndexError). Re-enable after fix.",
    },

    # biomedical/scientific specialists
    "biolord-2023": {
        "name": "FremyCompany/BioLORD-2023",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "biomedical",
        "mrl": False,
        "vram_gb": 0.3,
        "note": "Best ontology / biomedical concept matcher.",
    },
    "specter2": {
        "name": "allenai/specter2_base",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "scientific",
        "mrl": False,
        "vram_gb": 0.3,
        "note": "Scientific paper retrieval specialist (base model).",
    },
    "specter2-query": {
        "name": "allenai/specter2_adhoc_query",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "scientific",
        "mrl": False,
        "vram_gb": 0.3,
        "device": "cpu",
        "base_model": "allenai/specter2_base",
        "disabled": True,
        "note": "Uses adapter-transformers format (not PEFT). Needs `adapters` library.",
    },
    "medcpt-query": {
        "name": "ncbi/MedCPT-Query-Encoder",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "biomedical",
        "mrl": False,
        "vram_gb": 0.3,
        "note": "Biomedical asymmetric query encoder.",
    },
    "medcpt-article": {
        "name": "ncbi/MedCPT-Article-Encoder",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "biomedical",
        "mrl": False,
        "vram_gb": 0.3,
        "note": "Biomedical asymmetric document encoder.",
    },
    "biomedbert-base": {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        "dim": 768,
        "max_tokens": 512,
        "type": "dense",
        "tier": "biomedical",
        "mrl": False,
        "vram_gb": 0.3,
        "note": "Compact PubMed baseline.",
    },
    "biomedbert-large": {
        "name": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "dim": 1024,
        "max_tokens": 512,
        "type": "dense",
        "tier": "biomedical",
        "mrl": False,
        "vram_gb": 1.2,
        "note": "Larger PubMed baseline.",
    },

    # --- NEW: biomedical specialists (Tier 1) ---
    "sapbert": {
        "name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "dim": 768, "max_tokens": 512, "type": "dense", "tier": "biomedical",
        "mrl": False, "vram_gb": 0.3,
        "note": "UMLS-aligned entity embeddings.",
    },
    "pubmedbert-embed": {
        "name": "NeuML/pubmedbert-base-embeddings",
        "dim": 768, "max_tokens": 512, "type": "dense", "tier": "biomedical",
        "mrl": False, "vram_gb": 0.3,
        "note": "Best PubMed sentence embeddings at BERT-base size.",
    },
    "pubmedncl": {
        "name": "malteos/PubMedNCL",
        "dim": 768, "max_tokens": 512, "type": "dense", "tier": "biomedical",
        "mrl": False, "vram_gb": 0.3,
        "note": "Citation-graph contrastive learning on PubMed.",
    },
    "s-pubmedbert-marco": {
        "name": "pritamdeka/S-PubMedBert-MS-MARCO",
        "dim": 768, "max_tokens": 512, "type": "dense", "tier": "biomedical",
        "mrl": False, "vram_gb": 0.3,
        "note": "PubMedBERT fine-tuned on MS-MARCO.",
    },

    # --- NEW: general-purpose (Tier 2) ---
    "mxbai-embed-large": {
        "name": "mixedbread-ai/mxbai-embed-large-v1",
        "dim": 1024, "max_tokens": 512, "type": "dense", "tier": "general",
        "mrl": True, "vram_gb": 0.4,
        "note": "Strong general BERT-large baseline.",
    },
}

RERANKER_MODELS: Dict[str, dict] = {
    "qwen3-reranker-4b": {
        "name": "Qwen/Qwen3-Reranker-4B",
        "vram_gb": 8,
        "note": "Recommended reranker when available.",
    },
    "qwen3-reranker-0.6b": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "vram_gb": 1.2,
        "note": "Lightweight reranker fallback.",
    },
    "bge-reranker-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "vram_gb": 1.1,
        "note": "Loaded via CrossEncoder (FlagReranker bypassed). Works with transformers>=5.3.",
    },
    "bge-reranker-v2-gemma": {
        "name": "BAAI/bge-reranker-v2-gemma",
        "vram_gb": 6,
        "note": "Gemma-based multilingual reranker.",
    },

    # --- NEW: biomedical + lightweight rerankers ---
    "medcpt-cross-encoder": {
        "name": "ncbi/MedCPT-Cross-Encoder",
        "vram_gb": 0.3,
        "note": "Biomedical reranker trained on PubMed search logs. Completes MedCPT pipeline.",
    },
    "ms-marco-minilm-l6": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "vram_gb": 0.05,
        "note": "Ultralight reranker (22M params).",
    },
    "mxbai-rerank-large": {
        "name": "mixedbread-ai/mxbai-rerank-large-v1",
        "vram_gb": 0.6,
        "note": "DeBERTa-v2 reranker.",
    },
}

LLM_MODELS: Dict[str, dict] = {
    "qwen3.5-27b": {
        "ollama_name": "qwen3.5:27b",
        "size_b": 27,
        "quant": "Q4_K_M",
        "vram_gb": 17,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Recommended primary reasoning model.",
    },
    "qwen3.5-9b": {
        "ollama_name": "qwen3.5:9b",
        "size_b": 9,
        "quant": "Q4_K_M",
        "vram_gb": 6.6,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Recommended fast reasoning model.",
    },
    "qwen3.5-9b-q8": {
        "ollama_name": "qwen3.5:9b-q8_0",
        "size_b": 9,
        "quant": "Q8_0",
        "vram_gb": 11,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Higher precision 9B ablation model.",
    },
    "qwen3-32b": {
        "ollama_name": "qwen3:32b",
        "size_b": 32,
        "quant": "Q4_K_M",
        "vram_gb": 20,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "High-end dense reasoning model.",
    },
    "qwen3-14b": {
        "ollama_name": "qwen3:14b",
        "size_b": 14,
        "quant": "Q4_K_M",
        "vram_gb": 9.3,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "LoRA-friendly medium model.",
    },
    "qwen3-14b-q8": {
        "ollama_name": "qwen3:14b-q8_0",
        "size_b": 14,
        "quant": "Q8_0",
        "vram_gb": 16,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Higher precision 14B.",
    },
    "qwen3-8b": {
        "ollama_name": "qwen3:8b",
        "size_b": 8,
        "quant": "Q4_K_M",
        "vram_gb": 5.2,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Lightweight modern baseline.",
    },
    "qwen2.5-7b": {
        "ollama_name": "qwen2.5:7b-instruct-q8_0",
        "size_b": 7,
        "quant": "Q8_0",
        "vram_gb": 8,
        "ctx": 32768,
        "family": "qwen",
        "think": False,
        "note": "Legacy benchmark baseline.",
    },
    "qwen2.5-1.5b": {
        "ollama_name": "qwen2.5:1.5b",
        "size_b": 1.5,
        "quant": "Q4_K_M",
        "vram_gb": 1.0,
        "ctx": 32768,
        "family": "qwen",
        "think": False,
        "note": "Locally runnable smoke-test LLM on this machine right now.",
    },
    "qwen2.5-0.5b": {
        "ollama_name": "qwen2.5:0.5b",
        "size_b": 0.5,
        "quant": "Q4_K_M",
        "vram_gb": 0.4,
        "ctx": 32768,
        "family": "qwen",
        "think": False,
        "note": "Tiny fallback for wiring tests only.",
    },

    # --- NEW: Non-Qwen model families for cross-architecture benchmarking ---

    # Google Gemma
    "gemma3-27b-qat": {
        "ollama_name": "gemma3:27b-it-qat",
        "size_b": 27,
        "quant": "QAT",
        "vram_gb": 18,
        "ctx": 128000,
        "family": "gemma",
        "think": True,
        "note": "Quantization-aware trained Gemma 3. Top non-Qwen dense model.",
    },
    "gemma3-12b-q8": {
        "ollama_name": "gemma3:12b-it-q8_0",
        "size_b": 12,
        "quant": "Q8_0",
        "vram_gb": 13,
        "ctx": 128000,
        "family": "gemma",
        "think": True,
        "note": "Mid-tier Gemma for scaling comparison.",
    },

    # Mistral
    "mistral-small-24b": {
        "ollama_name": "mistral-small:24b-instruct-2501-q4_K_M",
        "size_b": 24,
        "quant": "Q4_K_M",
        "vram_gb": 14,
        "ctx": 32000,
        "family": "mistral",
        "think": False,
        "note": "Strong structured output and function calling.",
    },
    "mistral-nemo-12b": {
        "ollama_name": "mistral-nemo:12b-instruct-2407-q8_0",
        "size_b": 12,
        "quant": "Q8_0",
        "vram_gb": 13,
        "ctx": 1000000,
        "family": "mistral",
        "think": False,
        "note": "1M context window. Mistral + NVIDIA collaboration.",
    },

    # Microsoft Phi
    "phi4-14b-q8": {
        "ollama_name": "phi4:14b-q8_0",
        "size_b": 14,
        "quant": "Q8_0",
        "vram_gb": 16,
        "ctx": 16000,
        "family": "phi",
        "think": False,
        "note": "Best structured output / JSON parsing among small models.",
    },

    # Meta Llama
    "llama3.1-8b": {
        "ollama_name": "llama3.1:8b-instruct-q8_0",
        "size_b": 8,
        "quant": "Q8_0",
        "vram_gb": 8.5,
        "ctx": 128000,
        "family": "llama",
        "think": False,
        "note": "De facto standard. Best LoRA/PEFT ecosystem support.",
    },

    # Cohere Command-R (RAG-specialized)
    "command-r-35b": {
        "ollama_name": "command-r:35b-08-2024-q4_K_M",
        "size_b": 35,
        "quant": "Q4_K_M",
        "vram_gb": 20,
        "ctx": 128000,
        "family": "command-r",
        "think": False,
        "note": "Purpose-built for RAG + grounded generation with citations.",
    },
}

# Recommended frontier defaults (design intent)
RECOMMENDED_LLM = "qwen3.5-27b"
RECOMMENDED_LLM_FAST = "qwen3.5-9b"
RECOMMENDED_EMBEDDING = "qwen3-embed-8b"
RECOMMENDED_RERANKER = "qwen3-reranker-4b"

# Practical locally-runnable defaults (execution intent)
DEFAULT_LLM = os.getenv("SCMETA_LLM_MODEL", "qwen2.5-1.5b")
DEFAULT_LLM_FAST = os.getenv("SCMETA_LLM_FAST_MODEL", "qwen2.5-1.5b")
DEFAULT_EMBEDDING = os.getenv("SCMETA_EMBED_MODEL", "medcpt-query")
DEFAULT_EMBED_HYBRID = "bge-m3"
DEFAULT_EMBED_BIO = os.getenv("SCMETA_BIO_EMBED_MODEL", "biolord-2023")
DEFAULT_RERANKER = os.getenv("SCMETA_RERANK_MODEL", "ms-marco-minilm-l6")


# ---------------------------------------------------------------------------
# Runtime config singletons
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    desktop_root: Path = field(default_factory=lambda: DESKTOP_ROOT)
    geo_datahub_root: Path = field(default_factory=lambda: EXTERNAL_GEO_DATAHUB)
    legacy_scmetaintel_root: Path | None = None  # deprecated, fully superseded

    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    downloads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "downloads")
    h5ad_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "h5ad_output")

    ontologies_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "ontologies")
    enriched_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "enriched_metadata")
    qdrant_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "qdrant_data")
    benchmark_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "benchmarks")
    ground_truth_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "benchmarks" / "ground_truth")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "benchmarks" / "results")
    config_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "configs")
    docs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "docs")
    scripts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "scripts")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")

    def accession_index(self) -> Path:
        """Return local accession index. Fall back to GEO-DataHub only if local missing."""
        local = self.project_root / "local_accession_index.json"
        return local

    def ensure_dirs(self):
        for d in [
            self.data_dir,
            self.downloads_dir,
            self.h5ad_dir,
            self.ontologies_dir,
            self.enriched_dir,
            self.qdrant_dir,
            self.benchmark_dir,
            self.ground_truth_dir,
            self.results_dir,
            self.config_dir,
            self.docs_dir,
            self.scripts_dir,
            self.logs_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMRuntimeConfig:
    model_key: str = DEFAULT_LLM
    fallback_model_key: str = DEFAULT_LLM_FAST
    base_url: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    temperature: float = 0.1
    max_tokens: int = 2048
    context_budget: int = 12000

    @property
    def model(self) -> str:
        return LLM_MODELS[self.model_key]["ollama_name"]

    @property
    def fallback_model(self) -> str:
        return LLM_MODELS[self.fallback_model_key]["ollama_name"]


@dataclass
class EmbeddingRuntimeConfig:
    dense_model_key: str = DEFAULT_EMBEDDING
    bio_model_key: str = DEFAULT_EMBED_BIO
    reranker_model_key: str = DEFAULT_RERANKER
    batch_size: int = 32
    device: str = os.getenv("SCMETA_DEVICE", "cuda")

    @property
    def dense_model(self) -> str:
        return EMBEDDING_MODELS[self.dense_model_key]["name"]

    @property
    def bio_model(self) -> str:
        return EMBEDDING_MODELS[self.bio_model_key]["name"]

    @property
    def reranker_model(self) -> str:
        return RERANKER_MODELS[self.reranker_model_key]["name"]

    @property
    def dense_dim(self) -> int:
        return EMBEDDING_MODELS[self.dense_model_key]["dim"]


@dataclass
class RetrievalRuntimeConfig:
    collection_name: str = "scmetaintel_studies"
    top_k_retrieve: int = 50
    top_k_rerank: int = 10
    top_k_context: int = 10
    use_sparse: bool = True
    use_payload_filter: bool = True


@dataclass
class ServiceConfig:
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    hf_endpoint: str = HF_ENDPOINT
    ncbi_eutils_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    geo_query_url: str = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    rate_limit_delay: float = 0.35


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LLMRuntimeConfig = field(default_factory=LLMRuntimeConfig)
    embedding: EmbeddingRuntimeConfig = field(default_factory=EmbeddingRuntimeConfig)
    retrieval: RetrievalRuntimeConfig = field(default_factory=RetrievalRuntimeConfig)
    services: ServiceConfig = field(default_factory=ServiceConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG
