#!/usr/bin/env python3
"""
Benchmark 04 — LLM Comparison
==============================
Compare all local LLMs on 5 tasks, with thinking-mode ablation:
  A. Query parsing (structured extraction)
  B. Metadata extraction (from GEO text)
  C. Ontology normalization (term matching)
  D. Answer generation quality
  E. Inference speed

Thinking-capable models (Qwen3/3.5, Gemma3) are tested in two modes:
  - think=True  (reasoning chain visible, higher quality expected)
  - think=False (direct answer, faster)
Non-thinking models are tested once (think=False).

Requires Ollama server running with models pulled.

Usage:
    conda run -n dl python benchmarks/04_bench_llm.py
    conda run -n dl python benchmarks/04_bench_llm.py --models qwen3.5-9b qwen3-8b
    conda run -n dl python benchmarks/04_bench_llm.py --no-think-ablation
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import LLM_MODELS, GROUND_TRUTH_DIR, BENCHMARK_DIR
from scmetaintel.answer import (
    parse_query, extract_metadata, generate_answer, llm_call,
)
from scmetaintel.evaluate import (
    field_f1, extraction_metrics, citation_accuracy,
    query_parsing_metrics, ontology_metrics,
    load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("04_bench_llm")


def check_ollama():
    """Check if Ollama server is running."""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama models available: {models}")
        return models
    except Exception:
        logger.error("Ollama server not running. Start with: ollama serve")
        return []


def unload_ollama_models():
    """Unload all models from Ollama VRAM to free GPU memory between runs."""
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/ps", timeout=5)
        loaded = resp.json().get("models", [])
        for m in loaded:
            name = m["name"]
            logger.info(f"  Unloading {name} from VRAM ...")
            requests.post(
                "http://localhost:11434/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=30,
            )
        if loaded:
            import time
            time.sleep(2)  # brief pause for VRAM release
            logger.info(f"  Unloaded {len(loaded)} model(s)")
    except Exception as e:
        logger.warning(f"  Failed to unload models: {e}")


def task_a_query_parsing(model_key: str, queries: list, think: bool = False) -> dict:
    """Task A: Parse natural language queries into structured JSON."""
    from scmetaintel.evaluate import query_parsing_metrics
    # Limit sample size for think mode: thinking chains are very long per call
    sample = queries[:15] if think else queries
    correct, total = 0, 0
    field_scores = {"organism": [], "tissue": [], "disease": [],
                    "assay": [], "cell_type": []}

    for q in sample:
        gold = q.get("expected_constraints", {})
        if not gold:
            continue
        total += 1
        try:
            pred = parse_query(q["query"], model_key=model_key, think=think)
        except Exception as e:
            logger.warning(f"  Parse failed: {e}")
            continue

        # Use evaluate.py fuzzy matching for per-field scoring
        metrics = query_parsing_metrics(pred, gold)
        for field in field_scores:
            field_scores[field].append(metrics.get(f"f_{field}", 0.0))

        # Exact match via evaluate.py
        if metrics["exact_match"] == 1.0:
            correct += 1

    import numpy as np
    return {
        "exact_match": round(correct / total, 4) if total else 0,
        "field_accuracy": {f: round(np.mean(s), 4) if s else 0
                           for f, s in field_scores.items()},
        "n_queries": total,
    }


def task_b_metadata_extraction(model_key: str, docs: list, think: bool = False) -> dict:
    """Task B: Extract metadata from GEO title + summary."""
    # Limit sample size for think mode: 15 docs instead of 50
    limit = 15 if think else 50
    all_metrics = []
    for doc in docs[:limit]:
        # Build gold from terms that are actually extractable from the input text.
        # Raw sample-level metadata (donor IDs, cell-line codes) cannot be expected
        # from title+summary extraction — filter to terms present in the text.
        text = (doc.get("title", "") + " " + doc.get("summary", "")).lower()
        gold = {}
        for field in ["tissues", "diseases", "cell_types"]:
            raw = doc.get(field, []) or []
            gold[field] = [t for t in raw if t and len(t) >= 3 and t.lower() in text]
        if not any(gold.values()):
            continue
        try:
            pred = extract_metadata(
                doc.get("title", ""), doc.get("summary", ""),
                model_key=model_key, think=think)
            metrics = extraction_metrics(pred, gold,
                                         fields=["tissues", "diseases", "cell_types"])
            all_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"  Extract failed for {doc.get('gse_id')}: {e}")

    # Average across docs
    if not all_metrics:
        return {"error": "no successful extractions"}

    import numpy as np
    avg = {}
    for field in ["tissues", "diseases", "cell_types"]:
        vals = [m[field] for m in all_metrics if field in m]
        if vals:
            avg[field] = {
                "precision": round(np.mean([v["precision"] for v in vals]), 4),
                "recall": round(np.mean([v["recall"] for v in vals]), 4),
                "f1": round(np.mean([v["f1"] for v in vals]), 4),
            }
    return {"average": avg, "n_docs": len(all_metrics)}


def task_e_speed(model_key: str, think: bool = False) -> dict:
    """Task E: Inference speed."""
    prompt = "Extract the tissue type from: 'Single-cell RNA-seq of human lung fibrosis'"

    # Warm up
    try:
        llm_call(prompt, model_key=model_key, max_tokens=50)
    except Exception:
        return {"error": "model not available"}

    # Timed runs
    times, tokens = [], []
    for _ in range(3):
        t0 = time.time()
        from scmetaintel.answer import ollama_generate
        cfg = LLM_MODELS[model_key]
        result = ollama_generate(prompt, model=cfg["ollama_name"],
                                 max_tokens=200, think=think)
        elapsed = time.time() - t0
        times.append(elapsed)
        tokens.append(result.get("eval_count", 0))

    import numpy as np
    avg_time = np.mean(times)
    avg_tokens = np.mean(tokens)
    tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "avg_time_sec": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(tok_per_sec, 1),
        "think_enabled": think,
    }


# Ontology normalization system prompt (same as 06_bench_finetune.py)
ONTOLOGY_SYSTEM = (
    "You are a biomedical ontology normalizer. Given raw tissue, disease, or "
    "cell type terms from GEO metadata, map them to standard ontology terms.\n"
    "Use these ontologies:\n"
    "- Tissues: UBERON (e.g., UBERON:0000955 for brain)\n"
    "- Cell types: CL (e.g., CL:0000540 for neuron)\n"
    "- Diseases: MONDO (e.g., MONDO:0005015 for diabetes)\n"
    "Return JSON: {\"normalized\": [{\"raw\": str, \"ontology_id\": str, "
    "\"ontology_label\": str, \"confidence\": float}]}"
)


def task_c_ontology_normalization(model_key: str, docs: list, think: bool = False) -> dict:
    """Task C: Ontology normalization — map raw biomedical terms to ontology IDs."""
    import re

    # Load ontology lookups for gold comparison
    ontology_dir = Path(__file__).resolve().parent.parent / "ontologies"
    # Build gold_lookup: map lowercased name -> (ontology_id, canonical_name)
    gold_lookup = {}
    for obo_file, prefix in [("cl.obo", "CL"), ("uberon-basic.obo", "UBERON"), ("mondo.obo", "MONDO")]:
        obo_path = ontology_dir / obo_file
        if not obo_path.exists():
            continue
        current_id = current_name = None
        in_term = False
        with open(obo_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "[Term]":
                    in_term = True
                    current_id = current_name = None
                    continue
                if line.startswith("[") and line.endswith("]"):
                    in_term = False
                    continue
                if not in_term:
                    continue
                if line.startswith("id: ") and line[4:].strip().startswith(prefix + ":"):
                    current_id = line[4:].strip()
                elif line.startswith("name: ") and current_id:
                    current_name = line[6:].strip()
                    gold_lookup[current_name.lower()] = (current_id, current_name)
                elif line.startswith("synonym: ") and current_id and current_name:
                    # Also index synonyms for broader test coverage
                    syn_match = re.match(r'synonym:\s*"([^"]+)"', line)
                    if syn_match:
                        syn = syn_match.group(1).lower()
                        if syn not in gold_lookup:
                            gold_lookup[syn] = (current_id, current_name)

    all_scores = []
    test_docs = [d for d in docs if d.get("tissues") or d.get("cell_types")][:30]
    for doc in test_docs:
        raw_terms = []
        gold_items = []
        for field in ["tissues", "cell_types"]:
            for term in (doc.get(field, []) or [])[:3]:
                if term and len(term) >= 3:
                    key = term.lower().strip()
                    if key in gold_lookup:
                        ont_id, ont_label = gold_lookup[key]
                        raw_terms.append(term)
                        gold_items.append({
                            "raw": term,
                            "ontology_id": ont_id,
                            "ontology_label": ont_label,
                        })
        if not raw_terms:
            continue

        prompt = f"Normalize these biomedical terms from {doc.get('gse_id', 'unknown')}:\n{json.dumps(raw_terms)}"
        try:
            max_tok = 4096 if think else 1024
            raw = llm_call(prompt, model_key=model_key, system=ONTOLOGY_SYSTEM,
                           temperature=0.0, max_tokens=max_tok, think=think, timeout=300)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                pred_items = parsed.get("normalized", [])
            else:
                pred_items = []
            score = ontology_metrics(pred_items, gold_items)
            all_scores.append(score)
        except Exception as e:
            logger.warning(f"  Ontology failed for {doc.get('gse_id')}: {e}")

    if not all_scores:
        return {"error": "no successful ontology normalizations"}

    import numpy as np
    return {
        "accuracy": round(np.mean([s["accuracy"] for s in all_scores]), 4),
        "recall": round(np.mean([s["recall"] for s in all_scores]), 4),
        "f1": round(np.mean([s["f1"] for s in all_scores]), 4),
        "n_docs": len(all_scores),
    }


def task_d_answer_generation(model_key: str, queries: list, docs: list, think: bool = False) -> dict:
    """Task D: Answer generation quality — evaluate cited answers."""
    doc_by_gse = {d["gse_id"]: d for d in docs}
    all_scores = []

    for q in queries[:15]:  # limit for speed
        expected_gse = [g for g in q.get("expected_gse", []) if g in doc_by_gse]
        if not expected_gse:
            continue

        # Build context from expected GSEs
        context_parts = []
        for gse_id in expected_gse:
            d = doc_by_gse[gse_id]
            context_parts.append(
                f"[{gse_id}] {d.get('title', '')}\n"
                f"  Organism: {d.get('organism', 'N/A')}\n"
                f"  Summary: {d.get('summary', '')[:300]}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            f"Retrieved studies:\n{context}\n\n"
            f"User query: {q['query']}\n\n"
            f"Provide a comprehensive answer citing relevant GSE accessions."
        )
        try:
            max_tok = 4096 if think else 1024
            answer = llm_call(prompt, model_key=model_key,
                              system="You are a scientific dataset search assistant. "
                                     "Cite specific GSE accessions for every claim. "
                                     "Be concise and factual.",
                              temperature=0.0, max_tokens=max_tok, think=think, timeout=300)
            # Extract cited GSE IDs
            import re
            cited = re.findall(r"GSE\d+", answer)
            cited = list(dict.fromkeys(cited))  # deduplicate preserving order
            metrics = citation_accuracy(cited, set(expected_gse), expected_gse)
            metrics["n_cited"] = len(cited)
            all_scores.append(metrics)
        except Exception as e:
            logger.warning(f"  Answer generation failed for query {q.get('id')}: {e}")

    if not all_scores:
        return {"error": "no successful answer generations"}

    import numpy as np
    return {
        "citation_precision": round(np.mean([s["citation_precision"] for s in all_scores]), 4),
        "citation_recall": round(np.mean([s["citation_recall"] for s in all_scores]), 4),
        "grounding_rate": round(np.mean([s["grounding_rate"] for s in all_scores]), 4),
        "avg_citations": round(np.mean([s["n_cited"] for s in all_scores]), 1),
        "n_queries": len(all_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="LLM model keys to test")
    parser.add_argument("--no-think-ablation", action="store_true",
                        help="Skip thinking-mode ablation (only test think=False)")
    args = parser.parse_args()

    available = check_ollama()
    if not available:
        logger.error("No Ollama models available. Exiting.")
        return

    model_keys = args.models or list(LLM_MODELS.keys())
    # Filter to installed models (exact ollama_name match)
    active = []
    for mk in model_keys:
        if mk in LLM_MODELS:
            ollama_name = LLM_MODELS[mk]["ollama_name"]
            if ollama_name in available:
                active.append(mk)
            else:
                logger.warning(f"Model {mk} ({ollama_name}) not pulled in Ollama")

    if not active:
        logger.error("No matching models found in Ollama. Pull models first.")
        return

    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    # Build test configurations: (model_key, think_mode, run_label)
    runs = []
    for mk in active:
        can_think = LLM_MODELS[mk].get("think", False)
        # Always test with think=False (direct answering)
        runs.append((mk, False, mk))
        # Also test with think=True if model supports it and ablation enabled
        if can_think and not args.no_think_ablation:
            runs.append((mk, True, f"{mk}+think"))

    logger.info(f"Testing {len(active)} models in {len(runs)} configurations")
    logger.info(f"  Thinking-capable: {[mk for mk in active if LLM_MODELS[mk].get('think')]}")
    logger.info(f"  Non-thinking: {[mk for mk in active if not LLM_MODELS[mk].get('think')]}")
    logger.info(f"Loaded {len(queries)} queries, {len(docs)} docs")

    # Load existing results to support incremental runs (skip already-tested)
    results_path = BENCHMARK_DIR / "results" / "llm_bench.json"
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} existing results from {results_path}")

    prev_model_key = None
    for model_key, think_mode, run_label in runs:
        # Skip if already benchmarked
        if run_label in all_results and "error" not in str(all_results[run_label]):
            logger.info(f"Skipping {run_label} (already in results)")
            continue

        # Unload previous model from VRAM before loading new one
        if prev_model_key != model_key:
            logger.info(f"GPU memory management: unloading before {model_key}")
            unload_ollama_models()
            prev_model_key = model_key

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {run_label} (think={think_mode})")
        logger.info(f"{'='*60}")

        info = LLM_MODELS[model_key]
        results = {
            "model": model_key,
            "think_enabled": think_mode,
            "family": info.get("family", "unknown"),
            "size_b": info.get("size_b"),
            "quant": info.get("quant"),
        }

        logger.info("  Task A: Query parsing ...")
        results["task_a_parsing"] = task_a_query_parsing(model_key, queries, think=think_mode)

        logger.info("  Task B: Metadata extraction ...")
        results["task_b_extraction"] = task_b_metadata_extraction(model_key, docs, think=think_mode)

        logger.info("  Task C: Ontology normalization ...")
        results["task_c_ontology"] = task_c_ontology_normalization(model_key, docs, think=think_mode)

        logger.info("  Task D: Answer generation ...")
        results["task_d_answer"] = task_d_answer_generation(model_key, queries, docs, think=think_mode)

        logger.info("  Task E: Speed ...")
        results["task_e_speed"] = task_e_speed(model_key, think=think_mode)

        all_results[run_label] = results
        logger.info(f"  Done: {run_label}")

        # Save intermediate results after each config (crash-safe)
        save_results(all_results, "llm_bench")

    save_results(all_results, "llm_bench")
    unload_ollama_models()  # free VRAM after benchmark
    logger.info("LLM benchmark complete.")


if __name__ == "__main__":
    main()
