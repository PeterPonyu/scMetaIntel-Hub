"""Backward-compatible evaluation CLI wrapper."""

from __future__ import annotations

import json
import logging

from .config import get_config
from .evaluate import (
    load_eval_queries,
    load_ontology_eval,
    save_results,
)
from .retrieve import HybridRetriever
from .ontology import OntologyNormalizer

logger = logging.getLogger(__name__)


def evaluate_retrieval(retriever: HybridRetriever, queries=None) -> dict:
    from .evaluate import compute_retrieval_metrics, aggregate_metrics

    queries = queries or load_eval_queries()
    per_query = []
    for q in queries:
        expected = set(q.get("expected_gse", []))
        if not expected:
            continue
        parsed, results = retriever.search(q["query"], top_k=50, use_reranker=True)
        retrieved_ids = [r.study.gse_id for r in results]
        metrics = compute_retrieval_metrics(retrieved_ids, expected)
        metrics["id"] = q.get("id", q["query"][:20])
        metrics["query"] = q["query"]
        metrics["hit"] = any(x in expected for x in retrieved_ids[:10])
        per_query.append(metrics)
    summary = aggregate_metrics(per_query)
    summary["hit_rate_at_10"] = round(sum(1 for q in per_query if q.get("hit")) / len(per_query), 4) if per_query else 0.0
    return {"summary": summary, "per_query": per_query}


def evaluate_ontology(normalizer: OntologyNormalizer, test_cases=None) -> dict:
    test_cases = test_cases or load_ontology_eval()
    rows = []
    correct = 0
    for tc in test_cases:
        mapping = normalizer.normalize(tc["raw"], tc["category"])
        ok = mapping.ontology_id == tc["expected_id"]
        correct += int(ok)
        rows.append({
            "raw": tc["raw"],
            "category": tc["category"],
            "expected": tc["expected_id"],
            "got": mapping.ontology_id,
            "got_name": mapping.ontology_name,
            "method": mapping.method,
            "confidence": mapping.confidence,
            "correct": ok,
        })
    total = len(rows)
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_case": rows,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluation framework for scMetaIntel-Hub")
    parser.add_argument("--run-retrieval", action="store_true")
    parser.add_argument("--run-ontology", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not (args.run_retrieval or args.run_ontology or args.run_all):
        parser.print_help()
        return

    all_results = {}
    if args.run_retrieval or args.run_all:
        retriever = HybridRetriever()
        all_results["retrieval"] = evaluate_retrieval(retriever)

    if args.run_ontology or args.run_all:
        normalizer = OntologyNormalizer()
        normalizer.load_ontologies()
        normalizer.build_embedding_indices()
        all_results["ontology"] = evaluate_ontology(normalizer)

    save_results(all_results, "eval_results")
    print(json.dumps(all_results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
