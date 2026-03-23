#!/usr/bin/env python3
"""
Comprehensive Model Verification Test Suite
============================================
Tests every model in the registry for basic functionality:
  1. Embedding models: encode a test sentence, verify dimensions
  2. Reranker models: score a query-document pair
  3. LLM models: run each of the 5 pipeline tasks
  4. Integration: full pipeline smoke test

Usage:
    conda run -n dl python tests/test_all_models.py
    conda run -n dl python tests/test_all_models.py --section embeddings
    conda run -n dl python tests/test_all_models.py --section rerankers
    conda run -n dl python tests/test_all_models.py --section llms
    conda run -n dl python tests/test_all_models.py --section integration
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    EMBEDDING_MODELS, RERANKER_MODELS, LLM_MODELS,
    GROUND_TRUTH_DIR, DEFAULT_LLM, DEFAULT_RERANKER,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_all_models")

# Test data
TEST_QUERY = "single-cell RNA-seq of human lung cancer with immune cell types"
TEST_DOC_TITLE = "Single-cell transcriptomic atlas of human lung adenocarcinoma"
TEST_DOC_SUMMARY = (
    "We performed scRNA-seq on 42 tissue samples from 18 patients with lung "
    "adenocarcinoma. We identified 52 cell subtypes including epithelial, "
    "stromal, and immune cells. Tissues: lung. Organism: Homo sapiens. "
    "Cell types: T cells, B cells, macrophages, fibroblasts, epithelial cells."
)
TEST_GSE = "GSE999999"

RESULTS = {"embeddings": [], "rerankers": [], "llms": [], "integration": []}


def _record(section, name, status, detail="", duration=0):
    """Record a test result."""
    entry = {"model": name, "status": status, "detail": detail,
             "duration_sec": round(duration, 2)}
    RESULTS[section].append(entry)
    icon = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
    logger.info(f"  [{icon}] {name:35s} | {detail} ({duration:.1f}s)")


# =========================================================================
# Section 1: Embedding Models
# =========================================================================

def test_embeddings():
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 1: EMBEDDING MODELS")
    logger.info("=" * 60)

    from scmetaintel.embed import Embedder

    for key, cfg in EMBEDDING_MODELS.items():
        if cfg.get("disabled"):
            _record("embeddings", key, "skip", f"disabled: {cfg.get('disabled_reason', '?')}")
            continue

        t0 = time.time()
        try:
            embedder = Embedder(model_key=key)
            vecs = embedder.encode([TEST_QUERY, TEST_DOC_TITLE])
            elapsed = time.time() - t0

            assert vecs.shape[0] == 2, f"Expected 2 vectors, got {vecs.shape[0]}"
            assert vecs.shape[1] == cfg["dim"], \
                f"Expected dim={cfg['dim']}, got {vecs.shape[1]}"
            assert not np.isnan(vecs).any(), "NaN in embeddings"
            assert not np.isinf(vecs).any(), "Inf in embeddings"
            assert np.linalg.norm(vecs[0]) > 0.1, "Zero-norm embedding"

            # cosine similarity sanity check
            sim = np.dot(vecs[0], vecs[1]) / (
                np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))

            _record("embeddings", key, "pass",
                    f"dim={vecs.shape[1]}, sim={sim:.3f}", elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            _record("embeddings", key, "fail", str(e)[:100], elapsed)

        # Free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del embedder
        except Exception:
            pass


# =========================================================================
# Section 2: Reranker Models
# =========================================================================

def test_rerankers():
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 2: RERANKER MODELS")
    logger.info("=" * 60)

    from scmetaintel.retrieve import Reranker

    passages = [
        f"{TEST_DOC_TITLE}. {TEST_DOC_SUMMARY}",
        "Weather forecast for tomorrow in New York City.",
    ]

    for key, cfg in RERANKER_MODELS.items():
        if cfg.get("disabled"):
            _record("rerankers", key, "skip",
                    f"disabled: {cfg.get('disabled_reason', '?')}")
            continue

        t0 = time.time()
        try:
            reranker = Reranker(model_key=key)
            scores = reranker.score(TEST_QUERY, passages)
            elapsed = time.time() - t0

            assert len(scores) == 2, f"Expected 2 scores, got {len(scores)}"
            assert all(isinstance(s, (int, float, np.floating)) for s in scores), "Non-numeric scores"
            assert not any(np.isnan(s) for s in scores), "NaN in scores"

            # Relevant doc should score higher than irrelevant
            relevant_higher = scores[0] > scores[1]

            _record("rerankers", key, "pass",
                    f"relevant={float(scores[0]):.4f} vs irrelevant={float(scores[1]):.4f} "
                    f"{'(correct order)' if relevant_higher else '(WRONG order)'}",
                    elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            _record("rerankers", key, "fail", str(e)[:120], elapsed)

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del reranker
        except Exception:
            pass


# =========================================================================
# Section 3: LLM Models (5 tasks)
# =========================================================================

def test_llms():
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 3: LLM MODELS — 5 TASKS")
    logger.info("=" * 60)

    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        pulled = {m["name"] for m in resp.json().get("models", [])}
        logger.info(f"  Ollama running, {len(pulled)} models pulled")
    except Exception:
        logger.error("  Ollama not running! Skipping all LLM tests.")
        for key in LLM_MODELS:
            _record("llms", f"{key}/all_tasks", "skip", "Ollama not running")
        return

    from scmetaintel.answer import parse_query, extract_metadata, llm_call

    # Check which models are pulled
    def is_pulled(key):
        cfg = LLM_MODELS.get(key, {})
        tag = cfg.get("ollama_name", key)
        if tag in pulled:
            return True
        base = tag.split(":", 1)[0]
        return any(m.startswith(base + ":") for m in pulled)

    # Sample ground truth doc for extraction tasks
    gt_docs = sorted(GROUND_TRUTH_DIR.glob("GSE*.json"))
    sample_doc = None
    if gt_docs:
        with open(gt_docs[0]) as f:
            sample_doc = json.load(f)

    for key, cfg in LLM_MODELS.items():
        ollama_name = cfg.get("ollama_name", key)
        if not is_pulled(key):
            _record("llms", f"{key}", "skip", f"{ollama_name} not pulled")
            continue

        logger.info(f"\n  --- Testing LLM: {key} ({ollama_name}) ---")

        # Task A: Query Parsing
        t0 = time.time()
        try:
            parsed = parse_query(TEST_QUERY, model_key=key)
            elapsed = time.time() - t0
            fields_found = [f for f in ["organism", "tissue", "disease", "assay", "cell_type"]
                           if parsed.get(f)]
            _record("llms", f"{key}/task_a_parsing", "pass",
                    f"fields={fields_found}", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            _record("llms", f"{key}/task_a_parsing", "fail", str(e)[:100], elapsed)

        # Task B: Metadata Extraction
        if sample_doc:
            t0 = time.time()
            try:
                meta = extract_metadata(
                    sample_doc.get("title", ""), sample_doc.get("summary", ""),
                    model_key=key)
                elapsed = time.time() - t0
                extracted = {k: len(v) for k, v in meta.items() if isinstance(v, list)}
                _record("llms", f"{key}/task_b_extraction", "pass",
                        f"fields={extracted}", elapsed)
            except Exception as e:
                elapsed = time.time() - t0
                _record("llms", f"{key}/task_b_extraction", "fail", str(e)[:100], elapsed)
        else:
            _record("llms", f"{key}/task_b_extraction", "skip", "no ground truth docs")

        # Task C: Ontology Normalization (via llm_call)
        t0 = time.time()
        try:
            onto_prompt = (
                "Normalize these terms to standard ontology terms:\n"
                "- 'lung' -> UBERON term\n"
                "- 'T cell' -> CL term\n"
                "- 'breast cancer' -> MONDO term\n"
                "Return JSON with keys: uberon, cl, mondo"
            )
            onto_result = llm_call(
                onto_prompt, model_key=key,
                system="You normalize biomedical terms to ontology identifiers.",
                temperature=0.0, max_tokens=256, timeout=60)
            elapsed = time.time() - t0
            has_content = len(onto_result.strip()) > 5
            _record("llms", f"{key}/task_c_ontology", "pass",
                    f"len={len(onto_result)}, has_content={has_content}", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            _record("llms", f"{key}/task_c_ontology", "fail", str(e)[:100], elapsed)

        # Task D: Answer Generation (via llm_call with context)
        t0 = time.time()
        try:
            context = (
                f"[{TEST_GSE}] {TEST_DOC_TITLE}\n"
                f"  Organism: Homo sapiens\n"
                f"  Summary: {TEST_DOC_SUMMARY[:200]}"
            )
            answer_prompt = (
                f"Retrieved studies:\n{context}\n\n"
                f"User query: {TEST_QUERY}\n\n"
                f"Provide a concise answer citing GSE accessions."
            )
            answer = llm_call(
                answer_prompt, model_key=key,
                system="You are a scientific dataset search assistant. "
                       "Cite specific GSE accessions for every claim.",
                temperature=0.0, max_tokens=512, timeout=90)
            elapsed = time.time() - t0
            import re
            cited = re.findall(r"GSE\d+", answer)
            has_citation = len(cited) > 0
            _record("llms", f"{key}/task_d_answer", "pass",
                    f"citations={cited[:5]}, len={len(answer)}", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            _record("llms", f"{key}/task_d_answer", "fail", str(e)[:100], elapsed)

        # Task E: Speed (3-run average)
        t0 = time.time()
        try:
            from scmetaintel.answer import ollama_generate
            speed_prompt = "Extract the tissue type from: 'scRNA-seq of human lung fibrosis'"
            tokens_list = []
            times_list = []
            for _ in range(3):
                st = time.time()
                result = ollama_generate(speed_prompt, model=ollama_name,
                                        max_tokens=100)
                et = time.time() - st
                times_list.append(et)
                tokens_list.append(result.get("eval_count", 0))
            elapsed = time.time() - t0
            avg_tps = np.mean(tokens_list) / np.mean(times_list) if np.mean(times_list) > 0 else 0
            _record("llms", f"{key}/task_e_speed", "pass",
                    f"avg_tps={avg_tps:.1f}, avg_time={np.mean(times_list):.2f}s", elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            _record("llms", f"{key}/task_e_speed", "fail", str(e)[:100], elapsed)


# =========================================================================
# Section 4: Integration Tests
# =========================================================================

def test_integration():
    logger.info("\n" + "=" * 60)
    logger.info("SECTION 4: INTEGRATION TESTS")
    logger.info("=" * 60)

    # Test 1: Config loading
    t0 = time.time()
    try:
        from scmetaintel.config import (
            PROJECT_ROOT, GROUND_TRUTH_DIR, BENCHMARK_DIR,
            QDRANT_DIR, EXTERNAL_GEO_DATAHUB,
        )
        assert PROJECT_ROOT.exists(), f"PROJECT_ROOT missing: {PROJECT_ROOT}"
        assert GROUND_TRUTH_DIR.exists(), f"GROUND_TRUTH_DIR missing: {GROUND_TRUTH_DIR}"
        assert BENCHMARK_DIR.exists(), f"BENCHMARK_DIR missing: {BENCHMARK_DIR}"
        assert EXTERNAL_GEO_DATAHUB == PROJECT_ROOT / "geodh"
        elapsed = time.time() - t0
        _record("integration", "config_loading", "pass",
                f"PROJECT_ROOT={PROJECT_ROOT}", elapsed)
    except Exception as e:
        _record("integration", "config_loading", "fail", str(e)[:100], time.time() - t0)

    # Test 2: Geodh vendored import
    t0 = time.time()
    try:
        from geodh import geodh_main, search_geo, GEOSeriesInfo
        assert callable(geodh_main)
        assert callable(search_geo)
        _record("integration", "geodh_import", "pass", "all imports OK", time.time() - t0)
    except Exception as e:
        _record("integration", "geodh_import", "fail", str(e)[:100], time.time() - t0)

    # Test 3: Eval queries loading
    t0 = time.time()
    try:
        from scmetaintel.evaluate import load_eval_queries
        queries = load_eval_queries()
        assert len(queries) > 0, "No eval queries loaded"
        assert all("query" in q for q in queries), "Queries missing 'query' field"
        _record("integration", "eval_queries", "pass",
                f"{len(queries)} queries loaded", time.time() - t0)
    except Exception as e:
        _record("integration", "eval_queries", "fail", str(e)[:100], time.time() - t0)

    # Test 4: Ground truth loading
    t0 = time.time()
    try:
        gt_files = sorted(GROUND_TRUTH_DIR.glob("GSE*.json"))
        assert len(gt_files) > 0, "No ground truth files"
        with open(gt_files[0]) as f:
            doc = json.load(f)
        assert "gse_id" in doc, "Missing gse_id"
        assert "title" in doc, "Missing title"
        _record("integration", "ground_truth", "pass",
                f"{len(gt_files)} docs, sample: {doc['gse_id']}", time.time() - t0)
    except Exception as e:
        _record("integration", "ground_truth", "fail", str(e)[:100], time.time() - t0)

    # Test 5: Router module
    t0 = time.time()
    try:
        from scmetaintel.router import get_task_model, get_tier_config, TASK_MODEL_MAP
        for task in TASK_MODEL_MAP:
            model = get_task_model(task)
            assert model in LLM_MODELS, f"Router model {model} not in LLM_MODELS"
        for tier in ["quality", "fast", "balanced"]:
            cfg = get_tier_config(tier)
            assert "parse_model" in cfg
            assert "answer_model" in cfg
        _record("integration", "router", "pass",
                f"{len(TASK_MODEL_MAP)} tasks mapped", time.time() - t0)
    except Exception as e:
        _record("integration", "router", "fail", str(e)[:100], time.time() - t0)

    # Test 6: Ontology loading
    t0 = time.time()
    try:
        from scmetaintel.ontology import OntologyNormalizer
        norm = OntologyNormalizer()
        norm.load_ontologies()
        n_ontologies = len(norm.indices)
        total = sum(len(idx.terms) for idx in norm.indices.values())
        _record("integration", "ontology", "pass",
                f"{n_ontologies} ontologies, {total} entries", time.time() - t0)
    except Exception as e:
        _record("integration", "ontology", "fail", str(e)[:100], time.time() - t0)

    # Test 7: Embedder + Qdrant pipeline
    t0 = time.time()
    try:
        from scmetaintel.embed import Embedder, get_qdrant_client, create_collection, index_studies
        embedder = Embedder(model_key="bge-m3")
        test_qdrant_path = QDRANT_DIR / "test_verify"
        client = get_qdrant_client(test_qdrant_path)
        create_collection(client, "test_verify", embedder.dim)

        # Index a few docs
        test_docs = []
        for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json"))[:3]:
            with open(p) as f:
                test_docs.append(json.load(f))
        if test_docs:
            index_studies(client, "test_verify", test_docs, embedder)

        # Query
        from scmetaintel.retrieve import RetrievalPipeline
        pipeline = RetrievalPipeline(
            embedder=embedder, qdrant_client=client,
            collection_name="test_verify", strategy="dense", top_k=3)
        results = pipeline.retrieve(TEST_QUERY)
        _record("integration", "retrieval_pipeline", "pass",
                f"indexed {len(test_docs)} docs, retrieved {len(results)}", time.time() - t0)

        # Cleanup
        import shutil
        shutil.rmtree(test_qdrant_path, ignore_errors=True)
    except Exception as e:
        _record("integration", "retrieval_pipeline", "fail", str(e)[:100], time.time() - t0)

    # Test 8: Context management utilities
    t0 = time.time()
    try:
        from scmetaintel.answer import (
            chunk_study_fields, format_context_chunked,
            allocate_token_budget, format_context,
        )
        test_study = {
            "gse_id": TEST_GSE, "title": TEST_DOC_TITLE,
            "organism": "Homo sapiens", "summary": TEST_DOC_SUMMARY,
            "tissues": ["lung"], "diseases": ["lung adenocarcinoma"],
            "cell_types": ["T cell", "B cell"],
        }
        chunks = chunk_study_fields(test_study)
        assert len(chunks) > 0, "No chunks produced"

        chunked_ctx = format_context_chunked([test_study], chunk_types=["header", "summary"])
        assert len(chunked_ctx) > 0, "Empty chunked context"

        trimmed = allocate_token_budget([test_study] * 5, budget=500)
        assert len(trimmed) <= 5, "Budget didn't trim"

        ctx = format_context([test_study], fmt="structured")
        assert TEST_GSE in ctx, "GSE ID missing from context"

        _record("integration", "context_utilities", "pass",
                f"chunks={len(chunks)}, trimmed={len(trimmed)}/5", time.time() - t0)
    except Exception as e:
        _record("integration", "context_utilities", "fail", str(e)[:100], time.time() - t0)

    # Test 9: Evaluate module
    t0 = time.time()
    try:
        from scmetaintel.evaluate import (
            compute_retrieval_metrics, citation_accuracy,
            calibration_metrics,
        )
        ret_m = compute_retrieval_metrics(
            ["GSE1", "GSE2", "GSE3"], {"GSE1", "GSE4"})
        assert "p_at_10" in ret_m or "precision_at_10" in ret_m or "mrr" in ret_m

        cite_m = citation_accuracy(["GSE1", "GSE2"], {"GSE1", "GSE3"}, ["GSE1", "GSE2"])
        assert "citation_precision" in cite_m

        cal_m = calibration_metrics(
            [{"score": 0.9}, {"score": 0.1}],
            [{"relevant": True}, {"relevant": False}])
        assert "ece" in cal_m or "brier" in cal_m

        _record("integration", "evaluate_module", "pass",
                f"ret_metrics OK, cite_metrics OK, cal_metrics OK", time.time() - t0)
    except Exception as e:
        _record("integration", "evaluate_module", "fail", str(e)[:100], time.time() - t0)


# =========================================================================
# Main
# =========================================================================

def print_summary():
    logger.info("\n" + "=" * 70)
    logger.info("FULL TEST SUMMARY")
    logger.info("=" * 70)

    total_pass, total_fail, total_skip = 0, 0, 0
    for section, results in RESULTS.items():
        if not results:
            continue
        n_pass = sum(1 for r in results if r["status"] == "pass")
        n_fail = sum(1 for r in results if r["status"] == "fail")
        n_skip = sum(1 for r in results if r["status"] == "skip")
        total_pass += n_pass
        total_fail += n_fail
        total_skip += n_skip
        logger.info(f"  {section:15s}: {n_pass} pass, {n_fail} fail, {n_skip} skip")

        if n_fail > 0:
            for r in results:
                if r["status"] == "fail":
                    logger.info(f"    FAIL: {r['model']} — {r['detail']}")

    logger.info(f"\n  TOTAL: {total_pass} pass, {total_fail} fail, {total_skip} skip")
    total_time = sum(r["duration_sec"] for rs in RESULTS.values() for r in rs)
    logger.info(f"  Total test time: {total_time:.1f}s")

    return total_fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["embeddings", "rerankers", "llms", "integration", "all"],
                        default="all")
    args = parser.parse_args()

    sections = {
        "embeddings": test_embeddings,
        "rerankers": test_rerankers,
        "llms": test_llms,
        "integration": test_integration,
    }

    if args.section == "all":
        for name, fn in sections.items():
            fn()
    else:
        sections[args.section]()

    n_fail = print_summary()

    # Save results
    out_path = Path(__file__).parent.parent / "benchmarks" / "results" / "model_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    return n_fail


if __name__ == "__main__":
    sys.exit(main())
