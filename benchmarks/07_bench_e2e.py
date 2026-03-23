#!/usr/bin/env python3
"""
Benchmark 07 — End-to-End Pipeline Evaluation
===============================================
Compare 3 pipeline configurations:
  1. Baseline: BGE-M3 off-shelf, dense-only, Qwen2.5-7B, full text k=10
  2. Optimized: best from benchmarks 02-05
  3. Fine-tuned: fine-tuned models from benchmark 06

Metrics: P@10, citation accuracy, hallucination rate, latency, token efficiency.

Usage:
    conda run -n dl python benchmarks/07_bench_e2e.py
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    GROUND_TRUTH_DIR, BENCHMARK_DIR, QDRANT_DIR,
    EMBEDDING_MODELS, LLM_MODELS,
)
from scmetaintel.embed import Embedder, build_index_from_ground_truth, get_qdrant_client
from scmetaintel.retrieve import RetrievalPipeline, Reranker
from scmetaintel.answer import generate_answer, parse_query
from scmetaintel.evaluate import (
    compute_retrieval_metrics, citation_accuracy,
    aggregate_metrics, load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("07_bench_e2e")


# Pipeline configurations — informed by benchmarks 02-05 results:
#   Embedding: bge-m3 best overall (hybrid+sparse support)
#   Retrieval: hybrid+filter+rerank best (nDCG 0.138 vs 0.064 dense-only)
#   Context: structured format, k=3-5 best precision/recall balance
#   LLM: command-r-35b best query parsing & citation; qwen2.5-1.5b fastest
CONFIGS = {
    "baseline": {
        "description": "Dense-only, full context, small LLM — minimal pipeline",
        "embedding": "bge-m3",
        "strategy": "dense",
        "reranker": False,
        "llm": "qwen2.5-1.5b",
        "parse_model": None,  # same as llm
        "context_format": "full",
        "context_k": 10,
    },
    "optimized_quality": {
        "description": "Best quality — task-routed models, reranker, structured context",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter+rerank",
        "reranker": True,
        "llm": "command-r-35b",
        "parse_model": "command-r-35b",
        "context_format": "structured",
        "context_k": 5,
    },
    "optimized_fast": {
        "description": "Fast pipeline — hybrid+filter, no reranker, small context",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter",
        "reranker": False,
        "llm": "qwen2.5-1.5b",
        "parse_model": None,
        "context_format": "structured",
        "context_k": 3,
    },
    "finetuned": {
        "description": "Fine-tuned model — update llm after 06 completes",
        "embedding": "bge-m3",
        "strategy": "hybrid+filter+rerank",
        "reranker": True,
        "llm": "qwen3-8b",  # replace with fine-tuned Ollama tag after 06
        "parse_model": None,
        "context_format": "structured",
        "context_k": 5,
    },
}


def get_pulled_ollama_models() -> set[str]:
    """Return currently pulled Ollama model tags."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}
    except Exception:
        return set()


def model_is_pulled(model_key: str, pulled: set[str]) -> bool:
    cfg = LLM_MODELS.get(model_key)
    if not cfg:
        return False
    tag = cfg["ollama_name"]
    if tag in pulled:
        return True
    base = tag.split(":", 1)[0]
    return any(m.startswith(base + ":") for m in pulled)


def pick_best_available_llm(candidates: list[str], pulled: set[str]) -> str | None:
    for key in candidates:
        if model_is_pulled(key, pulled):
            return key
    return None


def resolve_configs(configs: dict) -> dict:
    """Replace unavailable target LLMs with the best currently pulled fallback."""
    pulled = get_pulled_ollama_models()
    resolved = {}
    for name, cfg in configs.items():
        cfg = dict(cfg)
        desired = cfg.get("llm")
        if desired and not model_is_pulled(desired, pulled):
            if name == "optimized":
                fallback = pick_best_available_llm(
                    ["qwen3.5-27b", "qwen3.5-9b", "qwen3-14b", "qwen2.5-7b", "qwen2.5-1.5b", "qwen2.5-0.5b"],
                    pulled,
                )
            elif name == "finetuned":
                fallback = pick_best_available_llm(
                    ["qwen3-14b", "qwen3.5-9b", "qwen2.5-7b", "qwen2.5-1.5b", "qwen2.5-0.5b"],
                    pulled,
                )
            else:
                fallback = pick_best_available_llm([desired, "qwen2.5-1.5b", "qwen2.5-0.5b"], pulled)

            if fallback:
                logger.warning(
                    "Config '%s' requested unavailable LLM '%s'; using pulled fallback '%s'",
                    name, desired, fallback,
                )
                cfg["llm_requested"] = desired
                cfg["llm"] = fallback
                cfg["llm_fallback_applied"] = True
            else:
                cfg["llm_requested"] = desired
                cfg["llm_fallback_applied"] = False
        resolved[name] = cfg
    return resolved


def run_e2e_pipeline(config_name: str, config: dict,
                     queries: list, docs: list) -> dict:
    """Run full pipeline for one configuration."""
    logger.info(f"\n--- Config: {config_name} ---")
    logger.info(f"  {config}")

    # Build index
    collection = f"e2e_{config_name}"
    qdrant_path = QDRANT_DIR / f"e2e_{config_name}"

    embedder = Embedder(model_key=config["embedding"])
    client = get_qdrant_client(qdrant_path)

    from scmetaintel.embed import create_collection, index_studies
    create_collection(client, collection, embedder.dim)
    index_studies(client, collection, docs, embedder)

    reranker = Reranker() if config["reranker"] else None

    pipeline = RetrievalPipeline(
        embedder=embedder,
        qdrant_client=client,
        collection_name=collection,
        reranker=reranker,
        strategy=config["strategy"],
        top_k=50,
        rerank_k=config["context_k"],
    )

    per_query_results = []
    for q in queries:
        relevant = set(q.get("expected_gse", []))
        if not relevant:
            continue

        t0 = time.time()

        # Step 1: Parse query (use parse_model if specified, else llm)
        parse_model = config.get("parse_model") or config["llm"]
        try:
            parsed = parse_query(q["query"], model_key=parse_model)
        except Exception:
            parsed = {"free_text": q["query"]}

        # Step 2: Retrieve
        filters = None
        constraints = q.get("expected_constraints", {})
        if "filter" in config["strategy"] and constraints.get("organism"):
            filters = {"organism": constraints["organism"]}

        retrieved = pipeline.retrieve(q["query"], filters=filters)
        retrieved_gse = [r["gse_id"] for r in retrieved]

        # Step 3: Generate answer
        try:
            answer_result = generate_answer(
                q["query"], retrieved,
                model_key=config["llm"],
                context_format=config["context_format"],
            )
        except Exception as e:
            logger.warning(f"    Answer generation failed: {e}")
            answer_result = {"answer": "", "cited_gse": [],
                             "eval_tokens": 0, "prompt_tokens": 0}

        total_time = time.time() - t0

        # Step 4: Compute metrics
        ret_metrics = compute_retrieval_metrics(retrieved_gse, relevant)
        cite_metrics = citation_accuracy(
            answer_result.get("cited_gse", []), relevant, retrieved_gse)

        # Hallucination check: cited GSE not in retrieved context
        cited = set(answer_result.get("cited_gse", []))
        hallucinated = cited - set(retrieved_gse)

        qr = {
            "query": q["query"],
            "difficulty": q.get("difficulty", "unknown"),
            **ret_metrics,
            **cite_metrics,
            "hallucination_rate": len(hallucinated) / len(cited) if cited else 0,
            "latency_sec": round(total_time, 2),
            "total_tokens": (answer_result.get("eval_tokens", 0) +
                             answer_result.get("prompt_tokens", 0)),
        }
        per_query_results.append(qr)

    # Aggregate
    import numpy as np
    avg = {}
    for key in per_query_results[0] if per_query_results else []:
        if isinstance(per_query_results[0][key], (int, float)):
            avg[key] = round(np.mean([r[key] for r in per_query_results]), 4)

    return {
        "config": config,
        "average": avg,
        "per_query": per_query_results,
        "n_queries": len(per_query_results),
    }


def main():
    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    if not docs:
        logger.error("No ground truth docs. Run 01_build_ground_truth.py first.")
        return

    logger.info(f"Loaded {len(queries)} queries, {len(docs)} docs")

    active_configs = resolve_configs(CONFIGS)

    all_results = {}
    for config_name, config in active_configs.items():
        try:
            result = run_e2e_pipeline(config_name, config, queries, docs)
            all_results[config_name] = result
            logger.info(f"  {config_name} average: {result['average']}")
        except Exception as e:
            logger.error(f"  {config_name} failed: {e}")
            all_results[config_name] = {"error": str(e)}

    # Comparison summary
    logger.info("\n" + "="*60)
    logger.info("E2E COMPARISON SUMMARY")
    logger.info("="*60)
    for name, res in all_results.items():
        avg = res.get("average", {})
        logger.info(f"  {name:15s} | P@10={avg.get('p_at_10', 'N/A'):>6} | "
                     f"MRR={avg.get('mrr', 'N/A'):>6} | "
                     f"CitPrec={avg.get('citation_precision', 'N/A'):>6} | "
                     f"Halluc={avg.get('hallucination_rate', 'N/A'):>6} | "
                     f"Latency={avg.get('latency_sec', 'N/A'):>6}s")

    save_results(all_results, "e2e_report")
    logger.info("End-to-end benchmark complete.")


if __name__ == "__main__":
    main()
