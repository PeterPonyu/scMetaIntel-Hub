"""
Task-specific model routing for scMetaIntel-Hub.

Different LLMs excel at different tasks. This module routes each pipeline
stage to its best-performing model based on benchmark results.

Benchmark source: benchmarks/results/llm_bench.json (04_bench_llm.py)
"""

from __future__ import annotations

import logging
from typing import Dict

from .config import LLM_MODELS, DEFAULT_LLM

logger = logging.getLogger(__name__)

# Best model per task based on 04_bench_llm.py results (think=False):
#   Task A (query parsing):        command-r-35b = 0.545 field_accuracy
#   Task B (metadata extraction):  qwen2.5-0.5b  = 0.326 avg-F1
#   Task C (ontology norm):        phi4-14b      = 0.286 F1 (only non-zero)
#   Task D (answer generation):    command-r-35b = 1.0 citation_precision
#   Task E (speed):                qwen2.5-0.5b  = 317 tok/s
TASK_MODEL_MAP: Dict[str, str] = {
    "query_parsing": "command-r-35b",
    "metadata_extraction": "qwen2.5-0.5b",
    "ontology_normalization": "phi4-14b-q8",
    "answer_generation": "command-r-35b",
    "relevance_judgment": "qwen3.5-9b",
    "fast_utility": "qwen2.5-0.5b",
}

# Performance tier presets for pipeline configurations
PIPELINE_TIERS: Dict[str, Dict[str, str]] = {
    "quality": {
        "description": "Best quality, ~7-10s per query",
        "parse_model": "command-r-35b",
        "answer_model": "command-r-35b",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter+rerank",
        "context_format": "structured",
        "context_k": 5,
    },
    "fast": {
        "description": "Fastest, ~1.5-2s per query",
        "parse_model": "qwen2.5-1.5b",
        "answer_model": "qwen2.5-1.5b",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter",
        "context_format": "structured",
        "context_k": 3,
    },
    "balanced": {
        "description": "Good balance, ~3s per query",
        "parse_model": "qwen3-8b",
        "answer_model": "qwen3-8b",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter+rerank",
        "context_format": "structured",
        "context_k": 5,
    },
}


def get_task_model(task: str, fallback: str | None = None) -> str:
    """Return the best model key for a given task.

    Falls back to DEFAULT_LLM if the task model is not available in LLM_MODELS.
    """
    model_key = TASK_MODEL_MAP.get(task, fallback or DEFAULT_LLM)
    if model_key not in LLM_MODELS:
        logger.warning(f"Task model {model_key} not in LLM_MODELS, using {DEFAULT_LLM}")
        return DEFAULT_LLM
    return model_key


def get_tier_config(tier: str) -> Dict[str, str]:
    """Return pipeline configuration for a performance tier."""
    if tier not in PIPELINE_TIERS:
        logger.warning(f"Unknown tier '{tier}', using 'balanced'")
        tier = "balanced"
    return PIPELINE_TIERS[tier].copy()


def update_task_model(task: str, model_key: str):
    """Update the model for a specific task (e.g., after fine-tuning)."""
    if model_key not in LLM_MODELS:
        logger.warning(f"Model {model_key} not in LLM_MODELS")
        return
    old = TASK_MODEL_MAP.get(task, "none")
    TASK_MODEL_MAP[task] = model_key
    logger.info(f"Updated {task} model: {old} -> {model_key}")
