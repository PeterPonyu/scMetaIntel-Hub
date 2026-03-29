#!/usr/bin/env python3
"""
Benchmark 05 — Context Efficiency
==================================
Optimize how much and what context to inject into the LLM.

Variables tested:
  - Number of studies in context: 3, 5, 10, 15, 20
  - Context format: full, structured, minimal
  - System prompt length: minimal, standard, detailed

Usage:
    conda run -n dl python benchmarks/05_bench_context.py
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import GROUND_TRUTH_DIR, BENCHMARK_DIR, LLM_MODELS, DEFAULT_LLM
from scmetaintel.answer import generate_answer, format_context, ANSWER_SYSTEM
from scmetaintel.evaluate import (
    citation_accuracy, load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("05_bench_ctx")


SYSTEM_PROMPTS = {
    "minimal": "Answer the query using the provided studies. Cite GSE IDs.",
    "standard": ANSWER_SYSTEM,
    "detailed": (
        "You are a scientific dataset search assistant specializing in single-cell "
        "genomics. Your role is to help researchers find relevant datasets.\n\n"
        "Instructions:\n"
        "1. Carefully read ALL provided study descriptions\n"
        "2. Identify studies that match the user's query\n"
        "3. Cite specific GSE accessions (e.g., GSE123456) for every claim\n"
        "4. Explain WHY each study is relevant (organism, tissue, disease match)\n"
        "5. If no studies match, clearly state that\n"
        "6. Organize your answer with the most relevant studies first\n"
        "7. Never fabricate or hallucinate GSE IDs\n"
        "8. Be concise but comprehensive\n"
    ),
}

K_VALUES = [3, 5, 10, 15, 20]
FORMATS = ["full", "structured", "minimal"]


def load_retrieval_results():
    """Load pre-computed retrieval results or simulate from ground truth."""
    ret_path = BENCHMARK_DIR / "results" / "retrieval_bench.json"
    if ret_path.exists():
        with open(ret_path) as f:
            return json.load(f)
    # Fallback: use ground truth docs as mock retrieval results
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))
    return docs


def run_context_experiment(query: dict, studies: list,
                           k: int, fmt: str, system_key: str,
                           model_key: str) -> dict:
    """Run a single context configuration and measure quality."""
    system = SYSTEM_PROMPTS[system_key]

    t0 = time.time()
    result = generate_answer(
        query["query"], studies[:k],
        model_key=model_key,
        context_format=fmt,
        system_prompt=system,
    )
    elapsed = time.time() - t0

    # Measure citation quality
    relevant = set(query.get("expected_gse", []))
    retrieved = [s.get("gse_id", "") for s in studies[:k]]
    cite_metrics = citation_accuracy(
        result.get("cited_gse", []), relevant, retrieved)

    # Estimate token usage
    context_text = format_context(studies[:k], fmt=fmt)
    approx_context_tokens = len(context_text) // 4

    return {
        "k": k,
        "format": fmt,
        "system_prompt": system_key,
        "context_tokens_approx": approx_context_tokens,
        "duration_ms": round(elapsed * 1000, 1),
        "output_tokens": result.get("eval_tokens", 0),
        "n_cited": len(result.get("cited_gse", [])),
        **cite_metrics,
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

    # Build GSE -> doc index for fast lookup
    doc_by_gse = {d["gse_id"]: d for d in docs}

    # Use subset of queries for context experiments (expensive with LLM calls)
    test_queries = [q for q in queries if q.get("expected_gse")][:15]
    logger.info(f"Testing {len(test_queries)} queries × {len(K_VALUES)} k-values "
                f"× {len(FORMATS)} formats")

    # Build per-query study lists: relevant docs first, then distractors
    import random
    random.seed(42)
    query_studies = {}
    for q in test_queries:
        qid = q.get("id", q["query"][:30])
        expected = q.get("expected_gse", [])
        # Relevant docs that exist in ground truth
        relevant = [doc_by_gse[gse] for gse in expected if gse in doc_by_gse]
        # Distractors: other docs not in expected_gse
        distractors = [d for d in docs if d["gse_id"] not in set(expected)]
        random.shuffle(distractors)
        # Context = relevant first, then distractors to fill up to max(K_VALUES)
        query_studies[qid] = relevant + distractors[:max(K_VALUES)]

    all_results = []

    # Test k-values × formats (with standard system prompt)
    for k in K_VALUES:
        for fmt in FORMATS:
            logger.info(f"  k={k}, format={fmt}, system=standard")
            for q in test_queries:
                qid = q.get("id", q["query"][:30])
                try:
                    result = run_context_experiment(
                        q, query_studies[qid], k=k, fmt=fmt,
                        system_key="standard", model_key=DEFAULT_LLM)
                    result["query_id"] = qid
                    all_results.append(result)
                except Exception as e:
                    logger.warning(f"    Failed: {e}")
            # Intermediate save after each k×fmt combo
            _save_intermediate(all_results)

    # Test system prompt lengths (with k=10, structured format)
    for sys_key in SYSTEM_PROMPTS:
        logger.info(f"  k=10, format=structured, system={sys_key}")
        for q in test_queries[:5]:  # fewer queries for prompt comparison
            qid = q.get("id", q["query"][:30])
            try:
                result = run_context_experiment(
                    q, query_studies[qid], k=10, fmt="structured",
                    system_key=sys_key, model_key=DEFAULT_LLM)
                result["query_id"] = qid
                all_results.append(result)
            except Exception as e:
                logger.warning(f"    Failed: {e}")

    # Summarize
    import numpy as np
    summary = {}
    for k in K_VALUES:
        for fmt in FORMATS:
            subset = [r for r in all_results
                      if r["k"] == k and r["format"] == fmt
                      and r.get("system_prompt") == "standard"]
            if subset:
                key = f"k={k}_{fmt}"
                summary[key] = {
                    "avg_grounding_rate": round(
                        np.mean([r["grounding_rate"] for r in subset]), 4),
                    "avg_citation_precision": round(
                        np.mean([r["citation_precision"] for r in subset]), 4),
                    "avg_citation_recall": round(
                        np.mean([r["citation_recall"] for r in subset]), 4),
                    "avg_duration_ms": round(
                        np.mean([r["duration_ms"] for r in subset]), 1),
                    "avg_context_tokens": round(
                        np.mean([r["context_tokens_approx"] for r in subset]), 0),
                    "n_queries": len(subset),
                }

    # System prompt comparison summary
    for sys_key in SYSTEM_PROMPTS:
        subset = [r for r in all_results
                  if r["k"] == 10 and r["format"] == "structured"
                  and r.get("system_prompt") == sys_key]
        if subset:
            summary[f"sysprompt_{sys_key}"] = {
                "avg_grounding_rate": round(
                    np.mean([r["grounding_rate"] for r in subset]), 4),
                "avg_citation_precision": round(
                    np.mean([r["citation_precision"] for r in subset]), 4),
                "avg_citation_recall": round(
                    np.mean([r["citation_recall"] for r in subset]), 4),
                "avg_duration_ms": round(
                    np.mean([r["duration_ms"] for r in subset]), 1),
                "n_queries": len(subset),
            }

    save_results({"summary": summary, "details": all_results}, "context_bench")
    logger.info("Context efficiency benchmark complete.")


def _save_intermediate(results):
    """Save intermediate results for crash safety."""
    try:
        save_results({"details": results}, "context_bench")
    except Exception:
        pass


if __name__ == "__main__":
    main()
