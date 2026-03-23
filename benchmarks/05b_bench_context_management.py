#!/usr/bin/env python3
"""
Benchmark 05b — Context Management Tech Stack
==============================================
Benchmark RAG context engineering strategies beyond simple k-value/format tests.

Strategies tested:
  A. Field-level chunking (doc_level, header+metadata, header+summary, all_chunks)
  B. Context compression via LLM (no_compression, compressed_500tok, compressed_200tok)
  C. Context ordering (relevance_first, mmr_diverse, recency_first)
  D. Ontology-aware query expansion (no_expansion, ontology_synonyms)
  E. Token budget allocation (no_budget, budget_2000, budget_4000, budget_8000)
  F. Multi-step retrieval (single_pass, two_step_refine)

Each strategy is tested independently with 15 queries using qwen2.5-1.5b.

Usage:
    conda run -n dl python benchmarks/05b_bench_context_management.py
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import GROUND_TRUTH_DIR, BENCHMARK_DIR, DEFAULT_LLM
from scmetaintel.answer import (
    generate_answer, format_context, ANSWER_SYSTEM,
    chunk_study_fields, format_context_chunked,
    compress_context, allocate_token_budget,
)
from scmetaintel.evaluate import citation_accuracy, load_eval_queries, save_results

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("05b_bench_ctx_mgmt")

# Fixed config for all experiments
MODEL = DEFAULT_LLM
K = 5  # retrieve 5 studies per query (best balance from 05 results)
FMT = "structured"


def _load_docs():
    """Load ground truth docs."""
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))
    return docs


def _build_query_studies(queries, docs, k=20):
    """Build per-query study lists: relevant docs first, then distractors."""
    doc_by_gse = {d["gse_id"]: d for d in docs}
    random.seed(42)
    query_studies = {}
    for q in queries:
        qid = q.get("id", q["query"][:30])
        expected = q.get("expected_gse", [])
        relevant = [doc_by_gse[gse] for gse in expected if gse in doc_by_gse]
        distractors = [d for d in docs if d["gse_id"] not in set(expected)]
        random.shuffle(distractors)
        query_studies[qid] = relevant + distractors[:k]
    return query_studies


def _evaluate_answer(result, query, studies_used):
    """Extract citation metrics from an answer result."""
    relevant = set(query.get("expected_gse", []))
    retrieved = [s.get("gse_id", "") for s in studies_used]
    return citation_accuracy(result.get("cited_gse", []), relevant, retrieved)


def _run_single(query, studies, model_key, context_text):
    """Run a single LLM call with pre-built context text."""
    prompt = (
        f"Retrieved studies:\n{context_text}\n\n"
        f"User query: {query['query']}\n\n"
        f"Provide a comprehensive answer citing relevant GSE accessions."
    )
    t0 = time.time()
    result = generate_answer(query["query"], studies, model_key=model_key)
    elapsed = time.time() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# Strategy A: Field-Level Chunking
# ---------------------------------------------------------------------------

def bench_chunking(queries, query_studies, model_key):
    """Test field-level chunking strategies."""
    configs = {
        "doc_level": None,  # use default format_context
        "header+metadata": ["header", "metadata"],
        "header+summary": ["header", "summary"],
        "all_chunks": ["header", "summary", "metadata", "abstract"],
    }
    results = []
    for config_name, chunk_types in configs.items():
        logger.info(f"  Chunking: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K]
            try:
                if chunk_types is None:
                    context_text = format_context(studies, fmt=FMT)
                else:
                    context_text = format_context_chunked(studies, chunk_types=chunk_types)

                t0 = time.time()
                result = generate_answer(q["query"], studies, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, studies)
                results.append({
                    "strategy": "chunking",
                    "config": config_name,
                    "query_id": qid,
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy B: Context Compression
# ---------------------------------------------------------------------------

def bench_compression(queries, query_studies, model_key):
    """Test LLM-based context compression."""
    configs = {
        "no_compression": 0,
        "compressed_500tok": 500,
        "compressed_200tok": 200,
    }
    results = []
    for config_name, target_tokens in configs.items():
        logger.info(f"  Compression: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K]
            try:
                if target_tokens == 0:
                    context_text = format_context(studies, fmt=FMT)
                else:
                    context_text = compress_context(
                        studies, q["query"],
                        model_key="qwen2.5-0.5b",
                        target_tokens=target_tokens,
                    )

                t0 = time.time()
                result = generate_answer(q["query"], studies, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, studies)
                results.append({
                    "strategy": "compression",
                    "config": config_name,
                    "query_id": qid,
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy C: Context Ordering
# ---------------------------------------------------------------------------

def bench_ordering(queries, query_studies, model_key):
    """Test context ordering strategies."""
    results = []
    configs = ["relevance_first", "mmr_diverse", "recency_first"]

    # For MMR, we need an embedder
    embedder = None
    if "mmr_diverse" in configs:
        try:
            from scmetaintel.embed import Embedder
            embedder = Embedder()
        except Exception as e:
            logger.warning(f"  Cannot load embedder for MMR: {e}")
            configs.remove("mmr_diverse")

    for config_name in configs:
        logger.info(f"  Ordering: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K]
            try:
                if config_name == "relevance_first":
                    ordered = studies  # already in relevant-first order
                elif config_name == "recency_first":
                    from scmetaintel.retrieve import reorder_recency
                    ordered = reorder_recency(studies)
                elif config_name == "mmr_diverse" and embedder:
                    from scmetaintel.retrieve import reorder_mmr
                    qvec = embedder.encode([q["query"]])[0]
                    ordered = reorder_mmr(studies, qvec, embedder, lambda_param=0.5)
                else:
                    ordered = studies

                context_text = format_context(ordered, fmt=FMT)
                t0 = time.time()
                result = generate_answer(q["query"], ordered, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, ordered)
                results.append({
                    "strategy": "ordering",
                    "config": config_name,
                    "query_id": qid,
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy D: Ontology-Aware Query Expansion
# ---------------------------------------------------------------------------

def bench_expansion(queries, query_studies, model_key):
    """Test ontology-aware query expansion."""
    results = []

    # Load ontologies
    ontology_indices = {}
    try:
        from scmetaintel.ontology import OntologyNormalizer
        normalizer = OntologyNormalizer()
        normalizer.load_ontologies()
        ontology_indices = normalizer.indices
        logger.info(f"  Loaded ontologies: {list(ontology_indices.keys())}")
    except Exception as e:
        logger.warning(f"  Cannot load ontologies: {e}")

    configs = ["no_expansion", "ontology_synonyms"]
    for config_name in configs:
        logger.info(f"  Expansion: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K]
            try:
                if config_name == "ontology_synonyms" and ontology_indices:
                    from scmetaintel.retrieve import expand_query_ontology
                    from scmetaintel.answer import parse_query
                    parsed = parse_query(q["query"], model_key=model_key)
                    expanded_query = expand_query_ontology(
                        q["query"], parsed, ontology_indices, max_synonyms=3
                    )
                else:
                    expanded_query = q["query"]

                context_text = format_context(studies, fmt=FMT)
                t0 = time.time()
                # Use expanded query for the answer generation
                result = generate_answer(expanded_query, studies, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, studies)
                results.append({
                    "strategy": "expansion",
                    "config": config_name,
                    "query_id": qid,
                    "expanded_query": expanded_query if config_name != "no_expansion" else None,
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy E: Token Budget Allocation
# ---------------------------------------------------------------------------

def bench_budget(queries, query_studies, model_key):
    """Test token budget allocation strategies."""
    # Use k=10 to make budget meaningful (more docs to potentially truncate)
    K_BUDGET = 10
    configs = {
        "no_budget": 0,
        "budget_2000": 2000,
        "budget_4000": 4000,
        "budget_8000": 8000,
    }
    results = []
    for config_name, budget in configs.items():
        logger.info(f"  Budget: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K_BUDGET]
            try:
                if budget > 0:
                    trimmed = allocate_token_budget(studies, budget=budget, fmt=FMT)
                else:
                    trimmed = studies

                context_text = format_context(trimmed, fmt=FMT)
                t0 = time.time()
                result = generate_answer(q["query"], trimmed, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, trimmed)
                results.append({
                    "strategy": "budget",
                    "config": config_name,
                    "query_id": qid,
                    "n_studies_kept": len(trimmed),
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Strategy F: Multi-Step Retrieval
# ---------------------------------------------------------------------------

def bench_multistep(queries, query_studies, model_key):
    """Test multi-step retrieval (simulated with pre-built context).

    Since we don't have a live Qdrant index for this benchmark,
    we simulate multi-step by using the LLM query refinement
    to re-score existing documents.
    """
    results = []
    configs = ["single_pass", "two_step_refine"]
    for config_name in configs:
        logger.info(f"  Multi-step: {config_name}")
        for q in queries:
            qid = q.get("id", q["query"][:30])
            studies = query_studies[qid][:K]
            try:
                if config_name == "two_step_refine":
                    # Simulate: use LLM to refine query, then re-order context
                    from scmetaintel.answer import llm_call
                    top_titles = [s.get("title", "") for s in studies[:3]]
                    refine_prompt = (
                        f"Original query: {q['query']}\n\n"
                        f"Top results found:\n" +
                        "\n".join(f"- {t}" for t in top_titles) +
                        f"\n\nGenerate a more specific search query. Return ONLY the query."
                    )
                    refined = llm_call(
                        refine_prompt,
                        model_key="qwen2.5-0.5b",
                        system="You refine scientific search queries. Return only the query.",
                        temperature=0.0, max_tokens=100, timeout=30,
                    ).strip()
                    use_query = refined if refined and len(refined) > 5 else q["query"]
                else:
                    use_query = q["query"]

                context_text = format_context(studies, fmt=FMT)
                t0 = time.time()
                result = generate_answer(use_query, studies, model_key=model_key)
                elapsed = time.time() - t0

                metrics = _evaluate_answer(result, q, studies)
                results.append({
                    "strategy": "multistep",
                    "config": config_name,
                    "query_id": qid,
                    "refined_query": use_query if config_name == "two_step_refine" else None,
                    "context_tokens_approx": len(context_text) // 4,
                    "duration_ms": round(elapsed * 1000, 1),
                    **metrics,
                })
            except Exception as e:
                logger.warning(f"    Failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    queries = load_eval_queries()
    docs = _load_docs()

    if not docs:
        logger.error("No ground truth docs. Run 01_build_ground_truth.py first.")
        return

    # Use subset of queries with expected_gse
    test_queries = [q for q in queries if q.get("expected_gse")][:15]
    query_studies = _build_query_studies(test_queries, docs, k=20)

    logger.info(f"Testing {len(test_queries)} queries with 6 strategy categories")
    logger.info(f"Model: {MODEL}, default k={K}, format={FMT}")

    all_results = []

    # Strategy A: Chunking
    logger.info("Strategy A: Field-Level Chunking")
    chunking_results = bench_chunking(test_queries, query_studies, MODEL)
    all_results.extend(chunking_results)
    _save_intermediate(all_results)

    # Strategy B: Compression
    logger.info("Strategy B: Context Compression")
    compression_results = bench_compression(test_queries, query_studies, MODEL)
    all_results.extend(compression_results)
    _save_intermediate(all_results)

    # Strategy C: Ordering
    logger.info("Strategy C: Context Ordering")
    ordering_results = bench_ordering(test_queries, query_studies, MODEL)
    all_results.extend(ordering_results)
    _save_intermediate(all_results)

    # Strategy D: Expansion
    logger.info("Strategy D: Ontology Query Expansion")
    expansion_results = bench_expansion(test_queries, query_studies, MODEL)
    all_results.extend(expansion_results)
    _save_intermediate(all_results)

    # Strategy E: Budget
    logger.info("Strategy E: Token Budget Allocation")
    budget_results = bench_budget(test_queries, query_studies, MODEL)
    all_results.extend(budget_results)
    _save_intermediate(all_results)

    # Strategy F: Multi-step
    logger.info("Strategy F: Multi-Step Retrieval")
    multistep_results = bench_multistep(test_queries, query_studies, MODEL)
    all_results.extend(multistep_results)

    # Summarize
    summary = {}
    for strategy in ["chunking", "compression", "ordering", "expansion", "budget", "multistep"]:
        strategy_results = [r for r in all_results if r["strategy"] == strategy]
        configs_in_strategy = sorted(set(r["config"] for r in strategy_results))
        for config in configs_in_strategy:
            subset = [r for r in strategy_results if r["config"] == config]
            if subset:
                key = f"{strategy}_{config}"
                summary[key] = {
                    "avg_grounding_rate": round(
                        np.mean([r.get("grounding_rate", 0) for r in subset]), 4),
                    "avg_citation_precision": round(
                        np.mean([r.get("citation_precision", 0) for r in subset]), 4),
                    "avg_citation_recall": round(
                        np.mean([r.get("citation_recall", 0) for r in subset]), 4),
                    "avg_duration_ms": round(
                        np.mean([r.get("duration_ms", 0) for r in subset]), 1),
                    "avg_context_tokens": round(
                        np.mean([r.get("context_tokens_approx", 0) for r in subset]), 0),
                    "n_queries": len(subset),
                }

    save_results({"summary": summary, "details": all_results}, "context_management_bench")
    logger.info(f"Context management benchmark complete. {len(all_results)} total results.")

    # Print summary table
    logger.info("\n=== SUMMARY ===")
    for key, val in sorted(summary.items()):
        logger.info(
            f"  {key:40s} | precision={val['avg_citation_precision']:.3f} "
            f"recall={val['avg_citation_recall']:.3f} "
            f"grounding={val['avg_grounding_rate']:.3f} "
            f"ctx_tok={val['avg_context_tokens']:.0f} "
            f"ms={val['avg_duration_ms']:.0f}"
        )


def _save_intermediate(results):
    """Save intermediate results for crash safety."""
    try:
        save_results({"details": results}, "context_management_bench")
    except Exception:
        pass


if __name__ == "__main__":
    main()
