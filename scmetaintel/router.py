"""
Task-specific model routing for scMetaIntel-Hub.

Different LLMs excel at different tasks. This module routes each pipeline
stage to its best-performing model based on benchmark results.

Benchmark source: benchmarks/results/ (02-05_bench_*.py, 2026-03-23)
"""

from __future__ import annotations

import logging
from typing import Dict

from .config import LLM_MODELS, DEFAULT_LLM

logger = logging.getLogger(__name__)

# Best model per task based on bench 04 results (fixed fuzzy eval, 2026-03-23):
#   Task A (query parsing):        qwen2.5-7b  = 91.4% cell_type, 82.8% tissue (best overall)
#   Task B (metadata extraction):  qwen2.5-7b  = 0.269 avg-F1 (best extraction)
#   Task C (ontology norm):        all 0% with ID-matching — redesigned with label eval
#   Task D (answer generation):    qwen2.5-1.5b = 91.1% citation_recall, 100% grounding (BEST)
#   Task E (speed):                qwen2.5-1.5b = 112.2 tok/s (fastest)
# Embedding (bench 02): medcpt-query R@50=0.392 | mxbai-embed-large R@50=0.410
# Retrieval (bench 03): hybrid+filter R@50=0.414, MRR=0.183, 181ms
# Context (bench 05):   k=3 structured = 100% precision, 91.1% recall, 168 tokens
TASK_MODEL_MAP: Dict[str, str] = {
    "query_parsing": "qwen2.5-7b",
    "metadata_extraction": "qwen2.5-7b",
    "ontology_normalization": "qwen2.5-7b",
    "answer_generation": "qwen2.5-1.5b",
    "relevance_judgment": "qwen2.5-7b",
    "fast_utility": "qwen2.5-1.5b",
}

# Performance tier presets for pipeline configurations
PIPELINE_TIERS: Dict[str, Dict[str, str]] = {
    "quality": {
        "description": "Best quality, ~2-3s per query",
        "parse_model": "qwen2.5-7b",
        "answer_model": "qwen2.5-1.5b",
        "embedding": "medcpt-query",
        "strategy": "hybrid+filter",
        "context_format": "structured",
        "context_k": 3,
    },
    "fast": {
        "description": "Fastest, ~0.5-1s per query",
        "parse_model": "qwen2.5-1.5b",
        "answer_model": "qwen2.5-1.5b",
        "embedding": "medcpt-query",
        "strategy": "dense",
        "context_format": "structured",
        "context_k": 3,
    },
    "balanced": {
        "description": "Good balance, ~1.5-2s per query",
        "parse_model": "qwen2.5-1.5b",
        "answer_model": "qwen2.5-1.5b",
        "embedding": "medcpt-query",
        "strategy": "hybrid+filter",
        "context_format": "structured",
        "context_k": 3,
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
