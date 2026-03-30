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
    # qwen3-embed-8b: REMOVED — 16GB VRAM, redundant with qwen3-embed-4b.
    "qwen3-embed-4b": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "dim": 2560,
        "max_tokens": 32768,
        "type": "dense",
        "tier": "general",
        "mrl": True,
        "vram_gb": 8,
        "note": "Qwen3 4B dense embedder. High quality, long context.",
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
    # gte-large: REMOVED — broken with transformers>=5.0, redundant with 8 general embedders.

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
    # specter2-query: REMOVED — needs adapter-transformers lib, specter2 base is sufficient.
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
    "biomedbert-fulltext": {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "dim": 768, "max_tokens": 512, "type": "dense", "tier": "biomedical",
        "mrl": False, "vram_gb": 0.3,
        "note": "PubMed abstract+fulltext trained. Complement to abstract-only variant.",
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
    "stella-en-v5": {
        "name": "dunzhang/stella_en_400M_v5",
        "dim": 1024, "max_tokens": 8192, "type": "dense", "tier": "general",
        "mrl": True, "vram_gb": 0.5,
        "note": "MTEB top-5 at 400M params. Strong general-purpose.",
    },
    "e5-large-v2": {
        "name": "intfloat/e5-large-v2",
        "dim": 1024, "max_tokens": 512, "type": "dense", "tier": "general",
        "mrl": False, "vram_gb": 1.3,
        "note": "Microsoft E5 large. Strong retrieval baseline.",
    },
    "gte-qwen2-1.5b": {
        "name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "dim": 1536, "max_tokens": 32768, "type": "dense", "tier": "general",
        "mrl": True, "vram_gb": 3.0,
        "note": "Qwen2-based 1.5B embedder. Instruction-tuned, long context.",
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
    "jina-reranker-v2": {
        "name": "jinaai/jina-reranker-v2-base-multilingual",
        "vram_gb": 1.1,
        "note": "Jina multilingual reranker. 8k context.",
    },
    "ms-marco-minilm-l12": {
        "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "vram_gb": 0.1,
        "note": "MiniLM 34M params. Stronger than L-6 variant.",
    },
    "ms-marco-tinybert": {
        "name": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "vram_gb": 0.02,
        "note": "TinyBERT 4M params. Floor baseline — smallest possible reranker.",
    },
    "bge-reranker-minicpm": {
        "name": "BAAI/bge-reranker-v2-minicpm-layerwise",
        "vram_gb": 2.7,
        "note": "MiniCPM 2.7B layerwise reranker. Different architecture from BGE-M3.",
    },
}

# ---------------------------------------------------------------------------
# Model family configuration
# ---------------------------------------------------------------------------
# Family-level settings shared by all models of the same architecture.
# Used by answer.py and benchmarks for think-mode, multimodal awareness, etc.

MODEL_FAMILY_CONFIG: Dict[str, dict] = {
    "qwen": {
        "think_api": True,           # Ollama /api/chat accepts "think" param
        "think_method": "api",       # Think triggered via API parameter
        "multimodal": False,         # Text-only in Ollama (no vision projector)
        "generations": ["2.5", "3", "3.5"],
        # Inference: Qwen3+ default is temp=0.7 but 0.0 best for structured tasks.
        # Qwen2.5 instruction-tuned models are stable at 0.0.
        "json_hint": "Return ONLY the JSON object. No explanation, no markdown.",
        "note": "Alibaba Qwen. Qwen3+ supports think API; Qwen2.5 does not.",
    },
    "gemma": {
        "think_api": True,           # Ollama ≥0.9 supports think for Gemma3
        "think_method": "api",
        "multimodal": True,          # Gemma3 bundles vision projector
        "multimodal_overhead_pct": 8,  # ~8% VRAM overhead for unused vision
        "generations": ["2", "3"],
        # Inference: Gemma3 is sensitive to temperature — use 0.0 for JSON tasks.
        # Vision projector loaded but unused for text-only workloads.
        "json_hint": "Return ONLY the JSON object. No explanation, no markdown.",
        "note": "Google Gemma. Gemma3 is multimodal — vision projector loaded "
                "but unused for text-only tasks (~8% VRAM overhead).",
    },
    "llama": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,         # llama3.2 base is text-only
        "generations": ["3.1", "3.2", "3.3"],
        "json_hint": "Return ONLY the JSON object. No explanation, no markdown.",
        "note": "Meta Llama. No native think mode. De facto standard baseline.",
    },
    "mistral": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["0.3", "nemo", "small"],
        # Inference: Mistral often wraps JSON in ```json``` markdown blocks.
        # The json_hint tells it not to; extract_json() handles it if it does.
        "json_hint": "Output raw JSON only. Do NOT wrap in markdown code blocks.",
        "note": "Mistral AI. Strong structured output and function calling.",
    },
    "phi": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,         # phi4 text-only variant
        "generations": ["3.5", "4", "4-mini"],
        # Inference: Phi excels at structured JSON output.
        "json_hint": "Output JSON only. No markdown. No explanation.",
        "note": "Microsoft Phi. Compact but strong on structured/JSON tasks.",
    },
    "deepseek": {
        "think_api": False,
        "think_method": "embedded",  # <think>...</think> always in response
        "always_thinks": True,       # Cannot disable thinking
        "multimodal": False,
        "generations": ["r1"],
        # Inference: R1 models always produce <think>...</think> before answer.
        # Token budget must be large enough for both reasoning + final answer.
        # answer.py extracts think text automatically.
        "json_hint": "After reasoning, return ONLY the JSON object as your final answer.",
        "note": "DeepSeek-R1 distilled. CoT reasoning always embedded in "
                "response as <think>...</think> tags. Distilled into Qwen2.5 "
                "or Llama3 base architectures.",
    },
    "granite": {
        "think_api": True,           # Granite3.3 supports think in Ollama
        "think_method": "api",
        "multimodal": False,
        "generations": ["3.3"],
        # Inference: Granite3.3 trained with tool-use / structured output focus.
        "json_hint": "Return ONLY the JSON object. Strict schema compliance required.",
        "note": "IBM Granite. Strong at structured output and tool use.",
    },
    "falcon": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["3"],
        # Inference: Falcon may add preamble before JSON.
        "json_hint": "Output raw JSON only. No preamble, no markdown code blocks.",
        "note": "TII Falcon3. Trained on curated web+code data.",
    },
    "command-r": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["r"],
        "json_hint": "Return ONLY the JSON object.",
        "note": "Cohere Command-R. RAG-specialized with grounded generation.",
    },
    "aya": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["expanse"],
        # Inference: Aya is multilingual but prompts are English for consistency.
        "json_hint": "Output raw JSON only. No markdown code blocks.",
        "note": "Cohere Aya. Massively multilingual (23 languages).",
    },
    "glm": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["4"],
        "json_hint": "Output raw JSON only. No markdown code blocks. No Chinese.",
        "note": "Zhipu GLM4. Chinese-English bilingual, strong reasoning.",
    },
    "internlm": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["2"],
        "json_hint": "Return ONLY the JSON object. No explanation, no markdown.",
        "note": "Shanghai AI Lab InternLM2. Strong bilingual Chinese+English.",
    },
    "yi": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["1.5"],
        "json_hint": "Return ONLY the JSON object. No explanation.",
        "note": "01.AI Yi. Strong reasoning and bilingual.",
    },
    "solar": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["10.7"],
        "json_hint": "Return ONLY the JSON object. No explanation, no markdown.",
        "note": "Upstage Solar. Depth-upscaled Llama architecture.",
    },
    "exaone": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["3.5"],
        "json_hint": "Output raw JSON only. No markdown. No Korean.",
        "note": "LG AI Research EXAONE. Korean+English bilingual.",
    },
    "starcoder": {
        "think_api": False,
        "think_method": None,
        "multimodal": False,
        "generations": ["2"],
        "json_hint": "Return ONLY the JSON object.",
        "note": "BigCode StarCoder2. Code-specialized — negative control.",
    },
}


def family_supports_think_api(family: str) -> bool:
    """Check if a model family supports the Ollama 'think' API parameter."""
    return MODEL_FAMILY_CONFIG.get(family, {}).get("think_api", False)


def family_always_thinks(family: str) -> bool:
    """Check if a model family always produces CoT (e.g. DeepSeek-R1)."""
    return MODEL_FAMILY_CONFIG.get(family, {}).get("always_thinks", False)


def family_think_method(family: str) -> str | None:
    """Return how thinking is triggered: 'api', 'embedded', or None."""
    return MODEL_FAMILY_CONFIG.get(family, {}).get("think_method")


def family_json_hint(family: str) -> str:
    """Return family-specific JSON formatting instruction to append to prompts."""
    return MODEL_FAMILY_CONFIG.get(family, {}).get(
        "json_hint", "Return ONLY the JSON object."
    )


def resolve_model_family(model_key: str) -> str:
    """Look up the family name for a given model key."""
    return LLM_MODELS.get(model_key, {}).get("family", "")


LLM_MODELS: Dict[str, dict] = {
    # qwen3.5-27b: REMOVED from disk — CPU spill, excluded from benchmarks.
    "qwen3.5-9b-q8": {
        "ollama_name": "qwen3.5:9b-q8_0",
        "size_b": 9,
        "quant": "Q8_0",
        "vram_gb": 11,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Recommended fast reasoning model. Q8_0 replaces Q4_K_M.",
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
    # --- WAVE 1: Fill gaps in Qwen scaling curve ---
    "qwen3.5-4b": {
        "ollama_name": "qwen3.5:4b",
        "size_b": 4,
        "quant": "Q4_K_M",
        "vram_gb": 2.8,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "wave": 1,
        "note": "Smallest Qwen3.5 available (no 3B exists). Intra-family scaling.",
    },
    "qwen3-4b": {
        "ollama_name": "qwen3:4b",
        "size_b": 4,
        "quant": "Q4_K_M",
        "vram_gb": 2.7,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "wave": 1,
        "note": "Mid-small Qwen3 fills gap between 1.7B and 8B.",
    },
    "qwen3-1.7b": {
        "ollama_name": "qwen3:1.7b",
        "size_b": 1.7,
        "quant": "Q4_K_M",
        "vram_gb": 1.2,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "wave": 1,
        "note": "Tiny Qwen3 with think — tests whether CoT helps sub-2B models.",
    },
    "qwen2.5-14b": {
        "ollama_name": "qwen2.5:14b-instruct-q8_0",
        "size_b": 14,
        "quant": "Q8_0",
        "vram_gb": 15,
        "ctx": 32768,
        "family": "qwen",
        "think": False,
        "wave": 1,
        "note": "Large Qwen2.5 baseline for generation-over-generation comparison.",
    },
    "qwen2.5-3b": {
        "ollama_name": "qwen2.5:3b",
        "size_b": 3,
        "quant": "Q4_K_M",
        "vram_gb": 2.0,
        "ctx": 32768,
        "family": "qwen",
        "think": False,
        "wave": 1,
        "note": "Mid-small Qwen2.5 for scaling curve.",
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
    # --- WAVE 1: Fill gaps in Gemma scaling curve ---
    "gemma3-4b-q8": {
        "ollama_name": "gemma3:4b-it-q8_0",
        "size_b": 4,
        "quant": "Q8_0",
        "vram_gb": 4.3,
        "ctx": 128000,
        "family": "gemma",
        "think": True,
        "wave": 1,
        "note": "Small Gemma3 Q8 for scaling curve. Multimodal overhead ~8%.",
    },
    "gemma3-1b": {
        "ollama_name": "gemma3:1b",
        "size_b": 1,
        "quant": "Q8_0",
        "vram_gb": 1.0,
        "ctx": 128000,
        "family": "gemma",
        "think": True,
        "wave": 1,
        "note": "Tiny Gemma3 for floor-performance analysis.",
    },
    "gemma2-9b": {
        "ollama_name": "gemma2:9b-instruct-q8_0",
        "size_b": 9,
        "quant": "Q8_0",
        "vram_gb": 10,
        "ctx": 8192,
        "family": "gemma",
        "think": False,
        "wave": 2,
        "note": "Previous-gen Gemma for generation-over-generation comparison.",
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
    # --- WAVE 2: Classic Mistral for 7B-class comparison ---
    "mistral-7b": {
        "ollama_name": "mistral:7b-instruct-v0.3-q8_0",
        "size_b": 7,
        "quant": "Q8_0",
        "vram_gb": 7.7,
        "ctx": 32768,
        "family": "mistral",
        "think": False,
        "wave": 2,
        "note": "Classic Mistral 7B v0.3 for 7B-class cross-family comparison.",
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
    # --- WAVE 2: Smaller Phi for scaling ---
    "phi4-mini": {
        "ollama_name": "phi4-mini:latest",
        "size_b": 3.8,
        "quant": "Q4_K_M",
        "vram_gb": 2.5,
        "ctx": 16000,
        "family": "phi",
        "think": False,
        "wave": 2,
        "note": "Phi4-mini 3.8B. Compact Microsoft model for small-model tier.",
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
    # --- WAVE 1: Llama 3.2 scaling points ---
    "llama3.2-3b": {
        "ollama_name": "llama3.2:3b",
        "size_b": 3,
        "quant": "Q4_K_M",
        "vram_gb": 2.0,
        "ctx": 128000,
        "family": "llama",
        "think": False,
        "wave": 1,
        "note": "Llama 3.2 small — lightweight text-only variant.",
    },
    "llama3.2-1b": {
        "ollama_name": "llama3.2:1b",
        "size_b": 1,
        "quant": "Q4_K_M",
        "vram_gb": 1.3,
        "ctx": 128000,
        "family": "llama",
        "think": False,
        "wave": 1,
        "note": "Llama 3.2 tiny for floor-performance analysis.",
    },

    # Cohere Command-R (RAG-specialized)
    # command-r-35b: REMOVED from disk — CPU spill, excluded from benchmarks.

    # --- Q3_K_M requantized models (100% GPU at num_ctx=4096 on RTX 5090 Laptop) ---
    "qwen3-32b": {
        "ollama_name": "qwen3:32b",
        "size_b": 32,
        "quant": "Q4_K_M",
        "vram_gb": 19,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "note": "Official qwen3:32b Q4_K_M. Replaces custom Q3_K_M for reproducibility.",
    },
    # Q3_K_M tested but not kept:
    # - qwen3.5-27b: quality preserved (0.800 composite) but no speed gain over Q4_K_M@4k ctx
    # - command-r-35b: ontology destroyed (65%→0%), -0.150 composite. REJECTED.

    # ===================================================================
    # WAVE 3: DeepSeek-R1 — always-reasoning distilled models
    # ===================================================================
    # NOTE: DeepSeek-R1 distillations embed <think>...</think> in every
    # response. The "think" flag below means the *base architecture*
    # supports think — but R1 models ALWAYS produce CoT regardless.
    # Family config `always_thinks=True` handles this in answer.py.

    "deepseek-r1-1.5b": {
        "ollama_name": "deepseek-r1:1.5b",
        "size_b": 1.5,
        "quant": "Q4_K_M",
        "vram_gb": 1.1,
        "ctx": 65536,
        "family": "deepseek",
        "think": True,
        "wave": 3,
        "note": "DeepSeek-R1 distilled into Qwen2.5-1.5B. Tiny reasoning model.",
    },
    "deepseek-r1-7b": {
        "ollama_name": "deepseek-r1:7b",
        "size_b": 7,
        "quant": "Q4_K_M",
        "vram_gb": 4.7,
        "ctx": 65536,
        "family": "deepseek",
        "think": True,
        "wave": 3,
        "note": "DeepSeek-R1 distilled into Qwen2.5-7B. Mid-tier reasoning.",
    },
    "deepseek-r1-8b": {
        "ollama_name": "deepseek-r1:8b",
        "size_b": 8,
        "quant": "Q4_K_M",
        "vram_gb": 4.9,
        "ctx": 65536,
        "family": "deepseek",
        "think": True,
        "wave": 3,
        "note": "DeepSeek-R1 distilled into Llama3.1-8B base. Cross-arch reasoning.",
    },
    "deepseek-r1-14b": {
        "ollama_name": "deepseek-r1:14b",
        "size_b": 14,
        "quant": "Q4_K_M",
        "vram_gb": 9.0,
        "ctx": 65536,
        "family": "deepseek",
        "think": True,
        "wave": 3,
        "note": "DeepSeek-R1 distilled into Qwen2.5-14B. Largest R1 that fits GPU.",
    },

    # ===================================================================
    # WAVE 4: IBM Granite — structured output specialists
    # ===================================================================

    "granite3.3-8b": {
        "ollama_name": "granite3.3:8b",
        "size_b": 8,
        "quant": "Q4_K_M",
        "vram_gb": 5.0,
        "ctx": 128000,
        "family": "granite",
        "think": True,
        "wave": 4,
        "note": "IBM Granite 3.3. Strong tool-use and structured output.",
    },
    "granite3.3-2b": {
        "ollama_name": "granite3.3:2b",
        "size_b": 2,
        "quant": "Q4_K_M",
        "vram_gb": 1.5,
        "ctx": 128000,
        "family": "granite",
        "think": True,
        "wave": 4,
        "note": "IBM Granite 3.3 small. Tests structured output at small scale.",
    },

    # ===================================================================
    # WAVE 5: Additional families for cross-architecture diversity
    # ===================================================================

    # TII Falcon3
    "falcon3-10b": {
        "ollama_name": "falcon3:10b",
        "size_b": 10,
        "quant": "Q4_K_M",
        "vram_gb": 6.0,
        "ctx": 32768,
        "family": "falcon",
        "think": False,
        "wave": 5,
        "note": "TII Falcon3 10B. Trained on curated web+code data.",
    },
    "falcon3-7b": {
        "ollama_name": "falcon3:7b",
        "size_b": 7,
        "quant": "Q4_K_M",
        "vram_gb": 4.3,
        "ctx": 32768,
        "family": "falcon",
        "think": False,
        "wave": 5,
        "note": "TII Falcon3 7B for 7B-class cross-family comparison.",
    },
    "falcon3-3b": {
        "ollama_name": "falcon3:3b",
        "size_b": 3,
        "quant": "Q4_K_M",
        "vram_gb": 1.9,
        "ctx": 32768,
        "family": "falcon",
        "think": False,
        "wave": 5,
        "note": "TII Falcon3 3B for small-model tier comparison.",
    },

    # Cohere Aya (multilingual)
    "aya-expanse-8b": {
        "ollama_name": "aya-expanse:8b",
        "size_b": 8,
        "quant": "Q4_K_M",
        "vram_gb": 4.8,
        "ctx": 8192,
        "family": "aya",
        "think": False,
        "wave": 5,
        "note": "Cohere Aya Expanse 8B. 23-language multilingual baseline.",
    },

    # Zhipu GLM4
    "glm4-9b": {
        "ollama_name": "glm4:9b",
        "size_b": 9,
        "quant": "Q4_K_M",
        "vram_gb": 5.5,
        "ctx": 131072,
        "family": "glm",
        "think": False,
        "wave": 5,
        "note": "Zhipu GLM4 9B. Chinese-English bilingual, strong reasoning.",
    },

    # ===================================================================
    # WAVE 6: Quantization ablation pairs — same model, different quant
    # ===================================================================
    # Purpose: isolate the effect of quantization on task quality.
    # Each pairs with an existing model at a different quant level.

    "qwen3-8b-q8": {
        "ollama_name": "qwen3:8b-q8_0",
        "size_b": 8,
        "quant": "Q8_0",
        "vram_gb": 9.0,
        "ctx": 256000,
        "family": "qwen",
        "think": True,
        "wave": 6,
        "quant_pair": "qwen3-8b",  # Q4_K_M counterpart
        "note": "Q8_0 quant ablation pair for qwen3-8b (Q4_K_M, 5.2GB).",
    },
    "gemma3-12b": {
        "ollama_name": "gemma3:12b-it-q4_K_M",
        "size_b": 12,
        "quant": "Q4_K_M",
        "vram_gb": 8.0,
        "ctx": 128000,
        "family": "gemma",
        "think": True,
        "wave": 6,
        "quant_pair": "gemma3-12b-q8",  # Q8_0 counterpart
        "note": "Q4_K_M quant ablation pair for gemma3-12b-q8 (Q8_0, 13GB).",
    },
    "llama3.1-8b-q4": {
        "ollama_name": "llama3.1:8b-instruct-q4_K_M",
        "size_b": 8,
        "quant": "Q4_K_M",
        "vram_gb": 5.0,
        "ctx": 128000,
        "family": "llama",
        "think": False,
        "wave": 6,
        "quant_pair": "llama3.1-8b",  # Q8_0 counterpart
        "note": "Q4_K_M quant ablation pair for llama3.1-8b (Q8_0, 8.5GB).",
    },
    "mistral-7b-q4": {
        "ollama_name": "mistral:7b-instruct-v0.3-q4_K_M",
        "size_b": 7,
        "quant": "Q4_K_M",
        "vram_gb": 4.0,
        "ctx": 32768,
        "family": "mistral",
        "think": False,
        "wave": 6,
        "quant_pair": "mistral-7b",  # Q8_0 counterpart
        "note": "Q4_K_M quant ablation pair for mistral-7b (Q8_0, 7.7GB).",
    },

    # ===================================================================
    # WAVE 7: New families — architecturally distinct models
    # ===================================================================

    # Shanghai AI Lab InternLM2
    "internlm2-7b": {
        "ollama_name": "internlm2:7b",
        "size_b": 7,
        "quant": "Q4_K_M",
        "vram_gb": 4.5,
        "ctx": 32768,
        "family": "internlm",
        "think": False,
        "wave": 7,
        "note": "Shanghai AI Lab InternLM2. Strong bilingual Chinese+English.",
    },

    # 01.AI Yi
    "yi-9b": {
        "ollama_name": "yi:9b",
        "size_b": 9,
        "quant": "Q4_K_M",
        "vram_gb": 5.0,
        "ctx": 4096,
        "family": "yi",
        "think": False,
        "wave": 7,
        "note": "01.AI Yi 9B. Strong reasoning, bilingual.",
    },

    # Upstage Solar
    "solar-10.7b": {
        "ollama_name": "solar:10.7b",
        "size_b": 10.7,
        "quant": "Q4_K_M",
        "vram_gb": 6.0,
        "ctx": 4096,
        "family": "solar",
        "think": False,
        "wave": 7,
        "note": "Upstage Solar 10.7B. Instruction-tuned, depth-upscaled Llama.",
    },

    # LG AI Research EXAONE
    "exaone3.5-7.8b": {
        "ollama_name": "exaone3.5:7.8b",
        "size_b": 7.8,
        "quant": "Q4_K_M",
        "vram_gb": 4.5,
        "ctx": 32768,
        "family": "exaone",
        "think": False,
        "wave": 7,
        "note": "LG AI EXAONE 3.5 7.8B. Korean+English bilingual.",
    },

    # BigCode StarCoder2 (code-specialized control)
    "starcoder2-7b": {
        "ollama_name": "starcoder2:7b",
        "size_b": 7,
        "quant": "Q4_K_M",
        "vram_gb": 4.0,
        "ctx": 16384,
        "family": "starcoder",
        "think": False,
        "wave": 7,
        "note": "BigCode StarCoder2. Code-specialized — negative control for bio tasks.",
    },

    # ===================================================================
    # WAVE 8: Second size point for single-model families
    # ===================================================================
    # Each family needs ≥2 models to plot a scaling curve.
    # Kept ≤24GB VRAM to avoid CPU spill.

    # Aya: 8B → add 32B (multilingual scaling)
    "aya-expanse-32b": {
        "ollama_name": "aya-expanse:32b-q4_K_M",
        "size_b": 32,
        "quant": "Q4_K_M",
        "vram_gb": 19.0,
        "ctx": 8192,
        "family": "aya",
        "think": False,
        "wave": 8,
        "note": "Aya Expanse 32B. Large multilingual for scaling vs 8B.",
    },

    # EXAONE: 7.8B → add 2.4B (small scaling point)
    "exaone3.5-2.4b": {
        "ollama_name": "exaone3.5:2.4b",
        "size_b": 2.4,
        "quant": "Q4_K_M",
        "vram_gb": 1.7,
        "ctx": 32768,
        "family": "exaone",
        "think": False,
        "wave": 8,
        "note": "EXAONE 3.5 2.4B. Small scaling point vs 7.8B.",
    },

    # InternLM: 7B → add 1.8B (small scaling point)
    "internlm2-1.8b": {
        "ollama_name": "internlm2:1.8b",
        "size_b": 1.8,
        "quant": "Q4_K_M",
        "vram_gb": 1.1,
        "ctx": 32768,
        "family": "internlm",
        "think": False,
        "wave": 8,
        "note": "InternLM2 1.8B. Small scaling point vs 7B.",
    },

    # Yi: 9B → add 6B (nearby size for precision comparison)
    "yi-6b": {
        "ollama_name": "yi:6b",
        "size_b": 6,
        "quant": "Q4_K_M",
        "vram_gb": 3.5,
        "ctx": 4096,
        "family": "yi",
        "think": False,
        "wave": 8,
        "note": "Yi 6B. Scaling point vs 9B.",
    },

    # StarCoder2: 7B → add 3B (small code control)
    "starcoder2-3b": {
        "ollama_name": "starcoder2:3b",
        "size_b": 3,
        "quant": "Q4_K_M",
        "vram_gb": 1.7,
        "ctx": 16384,
        "family": "starcoder",
        "think": False,
        "wave": 8,
        "note": "StarCoder2 3B. Small code-model negative control.",
    },

    # Solar: 10.7B is the only size. solar-pro:22b is a different architecture.
    # Add it as the large scaling point.
    "solar-pro-22b": {
        "ollama_name": "solar-pro:22b",
        "size_b": 22,
        "quant": "Q4_K_M",
        "vram_gb": 13.0,
        "ctx": 4096,
        "family": "solar",
        "think": False,
        "wave": 8,
        "note": "Upstage Solar Pro 22B. Large scaling point vs 10.7B.",
    },

    # GLM4: only 9B exists in Ollama. No other GLM4 size available.
    # Already at 1 model — accept this limitation.
}

# Recommended frontier defaults (design intent)
RECOMMENDED_LLM = "qwen3.5-9b-q8"
RECOMMENDED_LLM_FAST = "qwen3.5-9b-q8"
RECOMMENDED_EMBEDDING = "qwen3-embed-8b"
RECOMMENDED_RERANKER = "qwen3-reranker-4b"

# Practical locally-runnable defaults (execution intent)
DEFAULT_LLM = os.getenv("SCMETA_LLM_MODEL", "qwen2.5-1.5b")
DEFAULT_LLM_FAST = os.getenv("SCMETA_LLM_FAST_MODEL", "qwen2.5-1.5b")
DEFAULT_EMBEDDING = os.getenv("SCMETA_EMBED_MODEL", "mxbai-embed-large")
DEFAULT_EMBED_HYBRID = "bge-m3"
DEFAULT_EMBED_BIO = os.getenv("SCMETA_BIO_EMBED_MODEL", "biolord-2023")
DEFAULT_RERANKER = os.getenv("SCMETA_RERANK_MODEL", "ms-marco-minilm-l6")


# ---------------------------------------------------------------------------
# Centralized system prompts (used by pipeline and benchmarks)
# ---------------------------------------------------------------------------

PROMPTS = {
    "answer": (
        "You are a scientific dataset search assistant. Answer the user's query "
        "about single-cell datasets based ONLY on the provided study information.\n\n"
        "Rules:\n"
        "1. Cite specific GSE accessions (e.g., GSE123456) for every claim\n"
        "2. If the context doesn't contain relevant studies, say so\n"
        "3. Be concise and factual\n"
        "4. Never fabricate GSE IDs or study details\n"
    ),
    "parse": (
        "You are a biomedical search query parser. Extract structured constraints "
        "from the user's natural language query about single-cell datasets.\n"
        "Return ONLY valid JSON with these fields (use null or empty string if not mentioned):\n"
        '{"organism": "", "tissue": "", "disease": "", '
        '"cell_type": "", "assay": "", "treatment": "", "free_text": ""}'
    ),
    "extract": (
        "You are a biomedical metadata extractor. Given a GEO dataset title and "
        "summary, extract structured metadata.\n"
        "Return ONLY valid JSON with:\n"
        '{"tissues": [str], "diseases": [str], "cell_types": [str], '
        '"modalities": [str], "organism": str}'
    ),
    "ontology": (
        "You are a biomedical ontology normalizer. Given raw tissue, disease, or "
        "cell type terms from GEO metadata, map them to standard ontology terms.\n"
        "Use these ontologies:\n"
        "- Tissues: UBERON (e.g., UBERON:0000955 for brain)\n"
        "- Cell types: CL (e.g., CL:0000540 for neuron)\n"
        "- Diseases: MONDO (e.g., MONDO:0005015 for diabetes)\n"
        'Return JSON: {"normalized": [{"raw": str, "ontology_id": str, '
        '"ontology_label": str, "confidence": float}]}'
    ),
    "chat": (
        "You are scMetaIntel, a single-cell genomics dataset intelligence assistant.\n\n"
        "Your role is to help researchers find, compare, and understand single-cell datasets.\n\n"
        "RULES:\n"
        "1. Answer ONLY based on the retrieved dataset metadata provided below.\n"
        "2. For each claim, cite the source GSE accession in brackets, e.g. [GSE185224].\n"
        "3. If the information is not in the retrieved data, say "
        '"I don\'t have enough data to answer that."\n'
        "4. Never fabricate dataset details, cell types, or sample counts.\n"
        "5. Present results in a clear, structured format.\n"
        "6. When comparing datasets, highlight compatible platforms, shared cell types, "
        "and complementary coverage.\n"
        "7. Be concise but thorough."
    ),
    # --- Benchmark-specific prompts (Tasks D, F) ---
    "answer_bench": (
        "You are a scientific dataset search assistant. "
        "Answer the user's query about single-cell datasets based ONLY on the "
        "provided study information.\n\n"
        "Rules:\n"
        "1. Cite specific GSE accessions (e.g., GSE123456) for every claim\n"
        "2. If the context doesn't contain relevant studies, say so\n"
        "3. Be concise and factual\n"
        "4. Never fabricate GSE IDs or study details\n"
        "5. Cover ALL provided studies that are relevant to the query"
    ),
    "relevance": (
        "You are a biomedical dataset relevance judge. "
        "Given a user query and a GEO dataset description, determine if the dataset "
        "is RELEVANT to the query.\n"
        "A dataset is relevant if it could help answer the query based on organism, "
        "tissue, disease, cell types, or experimental modality.\n"
        'Return ONLY valid JSON: {"relevant": true} or {"relevant": false}'
    ),
    "classify_domain": (
        "You are a biomedical dataset classifier. Given a GEO dataset title and summary, "
        "classify the primary research domain.\n"
        "Choose EXACTLY ONE domain from: cancer, development, immunology, "
        "neurodegeneration, infectious_disease, cardiovascular, metabolic, other\n"
        'Return ONLY valid JSON: {"domain": "<chosen_domain>"}'
    ),
    "extract_organism_modality": (
        "You are a biomedical metadata extractor. Given a GEO dataset title and summary, "
        "extract the organism and experimental modalities.\n"
        "For organism, use the standard binomial name (e.g., Homo sapiens, Mus musculus).\n"
        "For modalities, choose from: scRNA-seq, scATAC-seq, CITE-seq, Multiome, Spatial, "
        "ChIP-seq/ATAC-seq, Bisulfite-seq, other\n"
        'Return ONLY valid JSON: {"organism": "<binomial name>", "modalities": ["<modality1>"]}'
    ),
}

# Default Ollama context window (tokens). Models may support more but
# larger windows increase VRAM usage. Override via SCMETA_NUM_CTX env var.
DEFAULT_NUM_CTX = int(os.getenv("SCMETA_NUM_CTX", "4096"))


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
        """Create the conventional project/output directories on demand.

        This helper is intentionally opt-in. Runtime code should prefer to
        create directories immediately before writing files so importing the
        config module does not leave empty folders behind.
        """
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


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG
