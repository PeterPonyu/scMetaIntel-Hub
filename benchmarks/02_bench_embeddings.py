#!/usr/bin/env python3
"""
Benchmark 02 — Embedding Model Comparison
==========================================
Compare all configured embedding models on:
  1. Semantic clustering (Silhouette score by domain/tissue)
  2. Retrieval accuracy (P@5, P@10, R@50, MRR, nDCG@10)
  3. Ontology term matching (Recall@1)
  4. Encoding speed (tokens/sec)

Usage:
    conda run -n dl python benchmarks/02_bench_embeddings.py
"""

import gc
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import EMBEDDING_MODELS, GROUND_TRUTH_DIR, BENCHMARK_DIR
from scmetaintel.embed import Embedder
from scmetaintel.evaluate import (
    compute_retrieval_metrics, aggregate_metrics,
    load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("02_bench_emb")


def _batch_size_for_model(model_key: str) -> int:
    """Choose batch size based on model VRAM needs.

    Qwen3-Embedding models use decoder attention which allocates
    O(seq_len²) memory per sample.  Large batches of long texts OOM
    on a 24 GB GPU.
    """
    if model_key.startswith("qwen3-embed"):
        return 2
    vram = EMBEDDING_MODELS[model_key].get("vram_gb", 0.5)
    if vram >= 8:
        return 2
    if vram >= 1:
        return 8
    return 32


def load_ground_truth_docs():
    """Load all enriched study JSONs."""
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))
    return docs


def bench_clustering(model_key: str, docs: list, embedder: Embedder,
                     batch_size: int = 32) -> dict:
    """Task 1: Semantic clustering quality."""
    from sklearn.metrics import silhouette_score

    texts = [d.get("document_text", d.get("title", "")) for d in docs]
    vectors = embedder.encode(texts, batch_size=batch_size)

    results = {}
    # Cluster by organism
    organisms = [d.get("organism", "unknown") for d in docs]
    unique_orgs = list(set(organisms))
    if len(unique_orgs) >= 2:
        labels = [unique_orgs.index(o) for o in organisms]
        results["silhouette_organism"] = round(
            silhouette_score(vectors, labels), 4)

    # Cluster by series_type
    types = [d.get("series_type", "unknown").split(";")[0].strip() for d in docs]
    unique_types = list(set(types))
    if len(unique_types) >= 2:
        labels = [unique_types.index(t) for t in types]
        results["silhouette_series_type"] = round(
            silhouette_score(vectors, labels), 4)

    return results


def bench_retrieval(model_key: str, docs: list, embedder: Embedder,
                    queries: list, batch_size: int = 32) -> dict:
    """Task 2: Retrieval accuracy on eval queries."""
    texts = [d.get("document_text", d.get("title", "")) for d in docs]
    doc_vectors = embedder.encode(texts, batch_size=batch_size)
    gse_ids = [d["gse_id"] for d in docs]

    # Normalize for cosine similarity
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    doc_vectors = doc_vectors / norms

    per_query_metrics = []
    for q in queries:
        query_text = q["query"]
        relevant = set(q.get("expected_gse", []))
        if not relevant:
            continue

        q_vec = embedder.encode([query_text])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)

        similarities = doc_vectors @ q_vec
        ranked_idx = np.argsort(-similarities)
        ranked_gse = [gse_ids[i] for i in ranked_idx]

        metrics = compute_retrieval_metrics(ranked_gse, relevant)
        metrics["difficulty"] = q.get("difficulty", "unknown")
        per_query_metrics.append(metrics)

    avg = aggregate_metrics(per_query_metrics)
    # Also compute by difficulty
    by_diff = {}
    for diff in ["easy", "medium", "hard"]:
        subset = [m for m in per_query_metrics if m.get("difficulty") == diff]
        if subset:
            by_diff[diff] = aggregate_metrics(subset)

    return {"average": avg, "by_difficulty": by_diff,
            "n_queries": len(per_query_metrics)}


def bench_speed(model_key: str, embedder: Embedder,
                batch_size: int = 32) -> dict:
    """Task 4: Encoding speed."""
    # Warm up
    embedder.encode(["warm up sentence"], batch_size=1)

    test_texts = [f"This is test sentence number {i} about single cell genomics."
                  for i in range(100)]

    t0 = time.time()
    embedder.encode(test_texts, batch_size=batch_size)
    elapsed = time.time() - t0

    # Rough token count (4 chars per token)
    total_chars = sum(len(t) for t in test_texts)
    approx_tokens = total_chars / 4
    tok_per_sec = approx_tokens / elapsed

    return {
        "sentences": len(test_texts),
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tok_per_sec, 1),
    }


def _parse_obo_with_synonyms(obo_path: Path, prefix: str) -> dict:
    """Parse an OBO file and return {canonical_name: {"id": str, "synonyms": [str]}}.

    Extracts both the primary name and all synonym fields from each [Term].
    Only EXACT and RELATED synonyms are included (BROAD/NARROW are too loose).
    """
    import re
    terms = {}
    current_id = current_name = None
    current_synonyms = []
    in_term = False
    syn_re = re.compile(r'^synonym:\s+"([^"]+)"\s+(EXACT|RELATED)')

    with open(obo_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                # Save previous term
                if current_id and current_name:
                    terms[current_name] = {
                        "id": current_id,
                        "synonyms": current_synonyms,
                    }
                in_term = True
                current_id = current_name = None
                current_synonyms = []
                continue
            if line.startswith("[") and line.endswith("]"):
                if in_term and current_id and current_name:
                    terms[current_name] = {
                        "id": current_id,
                        "synonyms": current_synonyms,
                    }
                in_term = False
                continue
            if not in_term:
                continue
            if line.startswith("id: ") and line[4:].strip().startswith(prefix + ":"):
                current_id = line[4:].strip()
            elif line.startswith("name: ") and current_id:
                current_name = line[6:].strip()
            elif line.startswith("synonym: ") and current_id:
                m = syn_re.match(line)
                if m:
                    syn_text = m.group(1).strip()
                    current_synonyms.append(syn_text)

    # Save last term
    if in_term and current_id and current_name:
        terms[current_name] = {"id": current_id, "synonyms": current_synonyms}

    return terms


def bench_ontology_matching(model_key: str, docs: list, embedder: Embedder,
                            batch_size: int = 32) -> dict:
    """Task 3: Ontology term matching via embedding similarity.

    Tests whether the embedding model can match synonym/variant terms
    to their canonical ontology names using vector similarity.
    Uses real synonym pairs from OBO files for a challenging test.
    """
    ontology_dir = Path(__file__).resolve().parent.parent / "ontologies"

    # Build ontology term bank with synonyms
    ont_terms = {}  # canonical_name -> {"id": str, "synonyms": [str]}
    for obo_file, prefix in [("cl.obo", "CL"), ("uberon-basic.obo", "UBERON")]:
        obo_path = ontology_dir / obo_file
        if not obo_path.exists():
            continue
        ont_terms.update(_parse_obo_with_synonyms(obo_path, prefix))

    if len(ont_terms) < 100:
        return {"error": "insufficient ontology terms loaded"}

    # Build test pairs from synonym -> canonical name
    # Only use terms that have at least one synonym different from the name
    synonym_pairs = []  # (synonym_text, canonical_name)
    for canonical, info in ont_terms.items():
        for syn in info["synonyms"]:
            if syn.lower().strip() != canonical.lower().strip() and len(syn) >= 3:
                synonym_pairs.append((syn, canonical))

    rng = np.random.RandomState(42)
    if len(synonym_pairs) > 500:
        idx = rng.choice(len(synonym_pairs), 500, replace=False)
        synonym_pairs = [synonym_pairs[i] for i in idx]

    # Also collect raw terms from docs that match ontology (metadata-based pairs)
    canonical_lower = {n.lower(): n for n in ont_terms}
    doc_pairs = []
    for d in docs:
        for field in ["tissues", "cell_types"]:
            for term in (d.get(field, []) or []):
                if not term or len(term.strip()) < 3:
                    continue
                t_lower = term.lower().strip()
                if t_lower in canonical_lower:
                    doc_pairs.append((term, canonical_lower[t_lower]))
        if len(doc_pairs) >= 200:
            break
    if len(doc_pairs) > 200:
        idx = rng.choice(len(doc_pairs), 200, replace=False)
        doc_pairs = [doc_pairs[i] for i in idx]

    # Combine: synonym pairs (hard) + metadata pairs (easier)
    test_pairs = synonym_pairs + doc_pairs
    if len(test_pairs) < 10:
        return {"error": "too few test pairs", "n_pairs": len(test_pairs)}

    # Build search bank: all gold targets + distractors up to 1000 total
    all_canonical = list(ont_terms.keys())
    gold_names = set(gn for _, gn in test_pairs)
    remaining = [n for n in all_canonical if n not in gold_names]
    bank_size = 1000
    n_extra = max(0, min(bank_size - len(gold_names), len(remaining)))
    if n_extra > 0:
        idx = rng.choice(len(remaining), n_extra, replace=False)
        bank_names = sorted(gold_names) + [remaining[i] for i in idx]
    else:
        bank_names = sorted(gold_names)

    bank_vecs = embedder.encode(bank_names, batch_size=batch_size)
    bank_norms = np.linalg.norm(bank_vecs, axis=1, keepdims=True)
    bank_norms = np.where(bank_norms == 0, 1, bank_norms)
    bank_vecs = bank_vecs / bank_norms

    # Evaluate: for each test pair, embed the query and find the gold name
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_10 = 0
    mrr_sum = 0.0
    for query_term, gold_name in test_pairs:
        q_vec = embedder.encode([query_term])[0]
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
        sims = bank_vecs @ q_vec
        top_idx = np.argsort(-sims)[:10]
        top_names = [bank_names[i] for i in top_idx]

        gold_lower = gold_name.lower()
        if top_names[0].lower() == gold_lower:
            correct_at_1 += 1
        if any(n.lower() == gold_lower for n in top_names[:5]):
            correct_at_5 += 1
        if any(n.lower() == gold_lower for n in top_names):
            correct_at_10 += 1
        # MRR
        for rank, n in enumerate(top_names, 1):
            if n.lower() == gold_lower:
                mrr_sum += 1.0 / rank
                break

    n = len(test_pairs)
    n_syn = len(synonym_pairs)
    n_doc = len(doc_pairs)
    return {
        "recall_at_1": round(correct_at_1 / n, 4),
        "recall_at_5": round(correct_at_5 / n, 4),
        "recall_at_10": round(correct_at_10 / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "n_test_pairs": n,
        "n_synonym_pairs": n_syn,
        "n_metadata_pairs": n_doc,
        "n_bank_terms": len(bank_names),
    }


def main():
    docs = load_ground_truth_docs()
    if not docs:
        logger.error("No ground truth docs found. Run 01_build_ground_truth.py first.")
        return

    queries = load_eval_queries()
    logger.info(f"Loaded {len(docs)} docs and {len(queries)} eval queries")

    all_results = {}
    for model_key in EMBEDDING_MODELS:
        info = EMBEDDING_MODELS[model_key]
        if info.get("disabled"):
            logger.info(f"Skipping disabled model: {model_key}")
            continue
        logger.info(f"\n{'='*60}\nBenchmarking: {model_key}\n{'='*60}")
        try:
            embedder = Embedder(model_key=model_key)
            bs = _batch_size_for_model(model_key)

            results = {"model": model_key,
                       "model_name": EMBEDDING_MODELS[model_key]["name"]}

            logger.info(f"  Task 1: Clustering ... (batch_size={bs})")
            results["clustering"] = bench_clustering(model_key, docs, embedder, bs)

            logger.info(f"  Task 2: Retrieval ...")
            results["retrieval"] = bench_retrieval(model_key, docs, embedder, queries, bs)

            logger.info(f"  Task 3: Ontology matching ...")
            results["ontology"] = bench_ontology_matching(model_key, docs, embedder, bs)

            logger.info(f"  Task 4: Speed ...")
            results["speed"] = bench_speed(model_key, embedder, bs)

            all_results[model_key] = results
            logger.info(f"  Results: {json.dumps(results, indent=2)[:500]}")

            # Free memory aggressively between models
            del embedder
            gc.collect()
            gc.collect()  # Second pass catches ref cycles
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"  Failed: {e}")
            all_results[model_key] = {"model": model_key, "error": str(e)}

        # Save intermediate results after each model (crash-safe)
        if all_results:
            save_results(all_results, "embedding_bench")

    save_results(all_results, "embedding_bench")
    logger.info("Embedding benchmark complete.")


if __name__ == "__main__":
    main()
