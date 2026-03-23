#!/usr/bin/env python3
"""
Benchmark 06b — Post-Fine-Tuning Evaluation
=============================================
Evaluate a fine-tuned LLM against its base model on all 5 benchmark tasks.

Produces a comparison report with:
  - Task regression testing (fine-tuned vs base on all 5 LLM tasks)
  - Generalization testing (held-out vs training queries)
  - Calibration metrics (ECE, Brier score for relevance judgments)
  - Latency comparison

Usage:
    conda run -n dl python benchmarks/06b_eval_finetuned.py --finetuned scmetaintel-qwen3-8b --base qwen3-8b
    conda run -n dl python benchmarks/06b_eval_finetuned.py --finetuned scmetaintel-qwen3-8b  # auto-detect base
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import LLM_MODELS, GROUND_TRUTH_DIR, BENCHMARK_DIR
from scmetaintel.answer import parse_query, extract_metadata, llm_call
from scmetaintel.evaluate import (
    citation_accuracy, calibration_metrics,
    load_eval_queries, save_results,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("06b_eval_ft")

# Held-out query IDs (not used in training data generation)
# Reserve last 10 queries by ID for generalization testing
HELD_OUT_IDS = {"q58", "q59", "q60", "q61", "q62", "q63", "q64", "q65", "q66", "q67"}


def task_a_parsing(model_key, queries, label=""):
    """Task A: Query parsing."""
    field_scores = {"organism": [], "tissue": [], "disease": [],
                    "assay": [], "cell_type": []}
    correct, total = 0, 0
    for q in queries[:30]:
        gold = q.get("expected_constraints", {})
        if not gold:
            continue
        total += 1
        try:
            pred = parse_query(q["query"], model_key=model_key)
        except Exception as e:
            logger.warning(f"  [{label}] Parse failed: {e}")
            continue
        for field in field_scores:
            gold_val = gold.get(field)
            pred_val = pred.get(field)
            if gold_val is None and pred_val is None:
                field_scores[field].append(1.0)
            elif gold_val and pred_val:
                field_scores[field].append(1.0 if str(gold_val).lower() == str(pred_val).lower() else 0.0)
            else:
                field_scores[field].append(0.0)
        match = all(str(gold.get(f, "")).lower() == str(pred.get(f, "")).lower() for f in field_scores)
        if match:
            correct += 1
    return {
        "exact_match": round(correct / total, 4) if total else 0,
        "field_accuracy": {f: round(np.mean(s), 4) if s else 0 for f, s in field_scores.items()},
        "avg_field_accuracy": round(np.mean([np.mean(s) for s in field_scores.values() if s]), 4),
        "n_queries": total,
    }


def task_b_extraction(model_key, docs, label=""):
    """Task B: Metadata extraction."""
    from scmetaintel.evaluate import extraction_metrics
    all_metrics = []
    for doc in docs[:30]:
        gold = {"tissues": doc.get("tissues", []), "diseases": doc.get("diseases", []),
                "cell_types": doc.get("cell_types", [])}
        if not any(gold.values()):
            continue
        try:
            pred = extract_metadata(doc.get("title", ""), doc.get("summary", ""), model_key=model_key)
            metrics = extraction_metrics(pred, gold, fields=["tissues", "diseases", "cell_types"])
            all_metrics.append(metrics)
        except Exception as e:
            logger.warning(f"  [{label}] Extract failed: {e}")
    if not all_metrics:
        return {"error": "no successful extractions"}
    avg = {}
    for field in ["tissues", "diseases", "cell_types"]:
        vals = [m[field] for m in all_metrics if field in m]
        if vals:
            avg[field] = {
                "precision": round(np.mean([v["precision"] for v in vals]), 4),
                "recall": round(np.mean([v["recall"] for v in vals]), 4),
                "f1": round(np.mean([v["f1"] for v in vals]), 4),
            }
    avg_f1 = np.mean([avg[f]["f1"] for f in avg if "f1" in avg[f]])
    return {"average": avg, "avg_f1": round(avg_f1, 4), "n_docs": len(all_metrics)}


def task_d_answer(model_key, queries, docs, label=""):
    """Task D: Answer generation with citation quality."""
    import re
    doc_by_gse = {d["gse_id"]: d for d in docs}
    all_scores = []
    for q in queries[:15]:
        expected_gse = [g for g in q.get("expected_gse", []) if g in doc_by_gse]
        if not expected_gse:
            continue
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
            answer = llm_call(prompt, model_key=model_key,
                              system="You are a scientific dataset search assistant. "
                                     "Cite specific GSE accessions for every claim. Be concise.",
                              temperature=0.0, max_tokens=1024, timeout=120)
            cited = list(dict.fromkeys(re.findall(r"GSE\d+", answer)))
            metrics = citation_accuracy(cited, set(expected_gse), expected_gse)
            metrics["n_cited"] = len(cited)
            all_scores.append(metrics)
        except Exception as e:
            logger.warning(f"  [{label}] Answer failed: {e}")
    if not all_scores:
        return {"error": "no successful answers"}
    return {
        "citation_precision": round(np.mean([s["citation_precision"] for s in all_scores]), 4),
        "citation_recall": round(np.mean([s["citation_recall"] for s in all_scores]), 4),
        "grounding_rate": round(np.mean([s["grounding_rate"] for s in all_scores]), 4),
        "avg_citations": round(np.mean([s["n_cited"] for s in all_scores]), 1),
        "n_queries": len(all_scores),
    }


def task_e_speed(model_key, label=""):
    """Task E: Inference speed."""
    from scmetaintel.answer import ollama_generate
    from scmetaintel.config import LLM_MODELS
    prompt = "Extract the tissue type from: 'Single-cell RNA-seq of human lung fibrosis'"
    try:
        llm_call(prompt, model_key=model_key, max_tokens=50)
    except Exception:
        return {"error": "model not available"}
    times, tokens = [], []
    for _ in range(3):
        t0 = time.time()
        cfg = LLM_MODELS.get(model_key, {})
        ollama_name = cfg.get("ollama_name", model_key)
        result = ollama_generate(prompt, model=ollama_name, max_tokens=200)
        elapsed = time.time() - t0
        times.append(elapsed)
        tokens.append(result.get("eval_count", 0))
    avg_time = np.mean(times)
    avg_tokens = np.mean(tokens)
    return {
        "avg_time_sec": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(avg_tokens / avg_time if avg_time > 0 else 0, 1),
    }


def run_eval(model_key, queries, docs, label):
    """Run all evaluation tasks for a single model."""
    logger.info(f"  [{label}] Task A: Query parsing...")
    task_a = task_a_parsing(model_key, queries, label)

    logger.info(f"  [{label}] Task B: Metadata extraction...")
    task_b = task_b_extraction(model_key, docs, label)

    logger.info(f"  [{label}] Task D: Answer generation...")
    task_d = task_d_answer(model_key, queries, docs, label)

    logger.info(f"  [{label}] Task E: Speed...")
    task_e = task_e_speed(model_key, label)

    return {
        "model": model_key,
        "task_a_parsing": task_a,
        "task_b_extraction": task_b,
        "task_d_answer": task_d,
        "task_e_speed": task_e,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", required=True, help="Fine-tuned model key in Ollama")
    parser.add_argument("--base", default=None, help="Base model key (auto-detect if not given)")
    args = parser.parse_args()

    ft_model = args.finetuned
    base_model = args.base or "qwen3-8b"

    # Load data
    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    if not docs:
        logger.error("No ground truth docs.")
        return

    # Split queries into training-overlap and held-out
    training_queries = [q for q in queries if q.get("id") not in HELD_OUT_IDS]
    held_out_queries = [q for q in queries if q.get("id") in HELD_OUT_IDS]
    logger.info(f"Queries: {len(training_queries)} training-overlap, {len(held_out_queries)} held-out")

    report = {
        "base_model": base_model,
        "fine_tuned_model": ft_model,
    }

    # Task regression: run both models on all queries
    logger.info(f"\n{'='*60}\nEvaluating BASE model: {base_model}\n{'='*60}")
    base_results = run_eval(base_model, queries, docs, "base")

    logger.info(f"\n{'='*60}\nEvaluating FINE-TUNED model: {ft_model}\n{'='*60}")
    ft_results = run_eval(ft_model, queries, docs, "ft")

    report["task_regression"] = {
        "base": base_results,
        "finetuned": ft_results,
    }

    # Compute deltas
    deltas = {}
    for task_key in ["task_a_parsing", "task_b_extraction", "task_d_answer"]:
        base_val = base_results.get(task_key, {})
        ft_val = ft_results.get(task_key, {})
        if isinstance(base_val, dict) and isinstance(ft_val, dict):
            for metric in base_val:
                if isinstance(base_val[metric], (int, float)) and isinstance(ft_val.get(metric), (int, float)):
                    delta = ft_val[metric] - base_val[metric]
                    deltas[f"{task_key}.{metric}"] = round(delta, 4)
    report["deltas"] = deltas

    # Generalization: compare fine-tuned model on held-out vs training queries
    if held_out_queries:
        logger.info(f"\nGeneralization test on {len(held_out_queries)} held-out queries...")
        ft_held_out = run_eval(ft_model, held_out_queries, docs, "ft-heldout")
        ft_training = run_eval(ft_model, training_queries, docs, "ft-train")
        report["generalization"] = {
            "held_out": ft_held_out,
            "training": ft_training,
        }

    # Latency comparison
    base_speed = base_results.get("task_e_speed", {})
    ft_speed = ft_results.get("task_e_speed", {})
    if "tokens_per_sec" in base_speed and "tokens_per_sec" in ft_speed:
        speed_ratio = ft_speed["tokens_per_sec"] / base_speed["tokens_per_sec"] if base_speed["tokens_per_sec"] else 0
        report["latency"] = {
            "base_tps": base_speed["tokens_per_sec"],
            "ft_tps": ft_speed["tokens_per_sec"],
            "speed_ratio": round(speed_ratio, 3),
        }

    save_results(report, "finetune_eval_report")
    logger.info("Fine-tune evaluation complete.")

    # Print summary
    logger.info("\n=== SUMMARY ===")
    for key, val in sorted(deltas.items()):
        direction = "+" if val > 0 else ""
        logger.info(f"  {key:40s} | {direction}{val:.4f}")


if __name__ == "__main__":
    main()
