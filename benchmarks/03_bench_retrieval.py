#!/usr/bin/env python3
"""
Benchmark 03 — Retrieval Strategy Comparison
=============================================
Compare 6 retrieval strategies:
  1. Dense-only
  2. Sparse-only (BM25-like)
  3. Hybrid (dense+sparse RRF)
  4. Hybrid + payload filter
  5. Hybrid + rerank
  6. Hybrid + filter + rerank

Uses the best embedding model from bench 02 (configurable).

Usage:
    conda run -n dl python benchmarks/03_bench_retrieval.py
    conda run -n dl python benchmarks/03_bench_retrieval.py --embedding bge-m3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import GROUND_TRUTH_DIR, QDRANT_DIR, BENCHMARK_DIR
from scmetaintel.embed import Embedder, build_index_from_ground_truth, get_qdrant_client
from scmetaintel.retrieve import RetrievalPipeline, Reranker
from scmetaintel.evaluate import (
    compute_retrieval_metrics, aggregate_metrics,
    load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("03_bench_ret")

STRATEGIES = [
    "dense",
    "sparse",
    "hybrid",
    "hybrid+filter",
    "hybrid+rerank",
    "hybrid+filter+rerank",
]


def run_strategy(strategy: str, embedder: Embedder, client, queries: list,
                 reranker=None, collection: str = "bench_retrieval") -> dict:
    """Run a single retrieval strategy on all queries."""
    pipeline = RetrievalPipeline(
        embedder=embedder,
        qdrant_client=client,
        collection_name=collection,
        reranker=reranker if "rerank" in strategy else None,
        strategy=strategy,
        top_k=50,
        rerank_k=10,
    )

    per_query = []
    latencies = []

    for q in queries:
        query_text = q["query"]
        relevant = set(q.get("expected_gse", []))
        if not relevant:
            continue

        # Build filters from expected constraints if strategy uses filters
        filters = None
        if "filter" in strategy:
            constraints = q.get("expected_constraints", {})
            if constraints.get("organism"):
                filters = {"organism": constraints["organism"]}

        t0 = time.time()
        results = pipeline.retrieve(query_text, filters=filters)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        retrieved_gse = [r["gse_id"] for r in results]
        metrics = compute_retrieval_metrics(retrieved_gse, relevant)
        metrics["difficulty"] = q.get("difficulty", "unknown")
        metrics["latency_ms"] = round(lat, 1)
        per_query.append(metrics)

    avg = aggregate_metrics(per_query)
    avg["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0

    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [m for m in per_query if m.get("difficulty") == diff]
        if subset:
            by_diff[diff] = aggregate_metrics(subset)

    return {
        "strategy": strategy,
        "average": avg,
        "by_difficulty": by_diff,
        "n_queries": len(per_query),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", default="bge-m3",
                        help="Embedding model key to use")
    args = parser.parse_args()

    queries = load_eval_queries()
    logger.info(f"Loaded {len(queries)} eval queries")

    collection = "bench_retrieval"

    # Build index
    logger.info(f"Building Qdrant index with {args.embedding} ...")
    qdrant_path = QDRANT_DIR / "bench_retrieval"
    client, embedder = build_index_from_ground_truth(
        model_key=args.embedding,
        qdrant_path=qdrant_path,
        collection_name=collection,
    )

    # Load reranker
    reranker = None
    try:
        reranker = Reranker()
        logger.info("Reranker loaded")
    except Exception as e:
        logger.warning(f"Reranker not available: {e}")

    # Run each strategy
    all_results = {"embedding_model": args.embedding}
    for strategy in STRATEGIES:
        logger.info(f"\n--- Strategy: {strategy} ---")
        needs_rerank = "rerank" in strategy
        if needs_rerank and reranker is None:
            logger.warning(f"  Skipping {strategy} (no reranker)")
            continue

        try:
            result = run_strategy(strategy, embedder, client, queries,
                                  reranker=reranker, collection=collection)
            all_results[strategy] = result
            logger.info(f"  Average: {result['average']}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            all_results[strategy] = {"strategy": strategy, "error": str(e)}

    save_results(all_results, "retrieval_bench")
    logger.info("Retrieval benchmark complete.")


if __name__ == "__main__":
    main()
