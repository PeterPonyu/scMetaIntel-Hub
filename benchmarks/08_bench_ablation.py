#!/usr/bin/env python3
"""
Benchmark 08 — Runtime Ablation Studies
=========================================
Test how runtime parameters affect LLM quality and speed WITHOUT
changing the model weights. Two dimensions:

  A. KV Cache Quantization: f16 (baseline) vs q8_0 vs q4_0
     - Requires Ollama server restart per KV type
     - Tests Tasks A (parse), D (answer), F (relevance)

  B. Context Length: 2048 vs 4096 (default) vs 8192 vs 16384
     - No restart needed — num_ctx is per-request
     - Tests Tasks D (answer) and E (speed)

Usage:
    conda run -n dl python benchmarks/08_bench_ablation.py
    conda run -n dl python benchmarks/08_bench_ablation.py --models qwen3-8b llama3.1-8b
    conda run -n dl python benchmarks/08_bench_ablation.py --skip-kv   # context only
    conda run -n dl python benchmarks/08_bench_ablation.py --skip-ctx  # KV only
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    LLM_MODELS, BENCHMARK_DIR, GROUND_TRUTH_DIR,
    family_always_thinks, family_json_hint, resolve_model_family,
    think_token_budget,
    OLLAMA_API_TAGS, OLLAMA_API_PS, OLLAMA_API_GENERATE,
    OLLAMA_KEEP_ALIVE_UNLOAD, TIMEOUT_OLLAMA_CHECK, TIMEOUT_OLLAMA_MGMT,
    BENCH_TEMPERATURE, TIMEOUT_LLM_LONG, GSE_PATTERN,
)
from scmetaintel.evaluate import load_eval_queries, save_results

logger = logging.getLogger("08_bench_ablation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

RESULTS_DIR = BENCHMARK_DIR / "results"
DEFAULT_MODELS = ["qwen3-8b", "llama3.1-8b", "phi4-14b-q8", "gemma3-12b-q8", "qwen3.5-9b-q8"]
KV_TYPES = ["f16", "q8_0", "q4_0"]
CTX_LENGTHS = [2048, 4096, 8192, 16384]


# ---------------------------------------------------------------------------
# Ollama server management
# ---------------------------------------------------------------------------

def kill_ollama():
    """Kill all Ollama server processes."""
    subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
    subprocess.run(["pkill", "-f", "ollama runner"], capture_output=True)
    time.sleep(3)


def start_ollama(kv_cache_type: str = "f16") -> bool:
    """Start Ollama server with specific KV cache type."""
    env = os.environ.copy()
    env["OLLAMA_KV_CACHE_TYPE"] = kv_cache_type
    env["OLLAMA_FLASH_ATTENTION"] = "1"  # required for KV cache quant
    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for health
    for _ in range(30):
        try:
            resp = requests.get(OLLAMA_API_TAGS, timeout=TIMEOUT_OLLAMA_CHECK)
            if resp.status_code == 200:
                logger.info(f"  Ollama started (kv_cache={kv_cache_type})")
                return True
        except Exception:
            pass
        time.sleep(1)
    logger.error("  Ollama failed to start")
    return False


def get_vram_mb() -> float:
    """Get current GPU VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def unload_models():
    """Unload all models from VRAM."""
    try:
        resp = requests.get(OLLAMA_API_PS, timeout=TIMEOUT_OLLAMA_CHECK)
        for m in resp.json().get("models", []):
            requests.post(OLLAMA_API_GENERATE,
                          json={"model": m["name"], "keep_alive": OLLAMA_KEEP_ALIVE_UNLOAD},
                          timeout=TIMEOUT_OLLAMA_MGMT)
        time.sleep(2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task runners (lightweight versions of 04_bench_llm.py tasks)
# ---------------------------------------------------------------------------

def run_task_a(model_key: str, queries: list, think: bool, max_q: int = 20) -> dict:
    """Task A: Query parsing (reduced)."""
    from scmetaintel.answer import parse_query
    from scmetaintel.evaluate import query_parsing_metrics
    sample = queries[:max_q]
    correct, total = 0, 0
    for q in sample:
        gold = q.get("expected_constraints", {})
        if not gold:
            continue
        total += 1
        try:
            pred = parse_query(q["query"], model_key=model_key, think=think)
            metrics = query_parsing_metrics(pred, gold)
            if metrics["exact_match"] == 1.0:
                correct += 1
        except Exception:
            pass
    return {"exact_match": round(correct / total, 4) if total else 0, "n": total}


def run_task_d(model_key: str, queries: list, docs: list,
               think: bool, max_q: int = 10) -> dict:
    """Task D: Answer generation (reduced)."""
    import re as _re
    from scmetaintel.answer import llm_call
    from scmetaintel.evaluate import citation_accuracy
    doc_by_gse = {d["gse_id"]: d for d in docs}
    scores = []
    for q in queries[:max_q]:
        expected = [g for g in q.get("expected_gse", []) if g in doc_by_gse]
        if not expected:
            continue
        context = "\n\n".join(
            f"[{g}] {doc_by_gse[g].get('title','')}\n  Organism: {doc_by_gse[g].get('organism','')}\n"
            f"  Summary: {doc_by_gse[g].get('summary','')[:300]}"
            for g in expected
        )
        prompt = f"Retrieved studies:\n{context}\n\nQuery: {q['query']}\n\nCite relevant GSE accessions."
        try:
            family = resolve_model_family(model_key)
            max_tok = think_token_budget(1024, think, family)
            raw = llm_call(prompt, model_key=model_key, temperature=0.0,
                           max_tokens=max_tok, think=think, timeout=300)
            cited = list(dict.fromkeys(_re.findall(r"GSE\d+", raw)))
            m = citation_accuracy(cited, set(expected), expected)
            scores.append(m["citation_recall"])
        except Exception:
            pass
    return {"cite_recall": round(float(np.mean(scores)), 4) if scores else 0, "n": len(scores)}


def run_task_e(model_key: str, think: bool) -> dict:
    """Task E: Speed (single prompt)."""
    from scmetaintel.answer import ollama_generate
    prompt = "Extract the tissue type from: 'Single-cell RNA-seq of human lung fibrosis'"
    try:
        ollama_generate(prompt, model=LLM_MODELS[model_key]["ollama_name"],
                        max_tokens=50, think=think)  # warmup
        t0 = time.time()
        result = ollama_generate(prompt, model=LLM_MODELS[model_key]["ollama_name"],
                                 max_tokens=200, think=think)
        elapsed = time.time() - t0
        tokens = result.get("eval_count", 0)
        return {"tok_per_sec": round(tokens / elapsed, 1) if elapsed > 0 else 0,
                "tokens": tokens, "time_sec": round(elapsed, 3)}
    except Exception:
        return {"error": "failed"}


def run_task_f(model_key: str, queries: list, docs: list,
               think: bool, max_pairs: int = 30) -> dict:
    """Task F: Relevance judgment (reduced)."""
    import random as _rnd
    from scmetaintel.answer import llm_call, extract_json
    from scmetaintel.config import PROMPTS
    _rnd.seed(42)

    doc_by_gse = {d["gse_id"]: d for d in docs}
    all_gses = list(doc_by_gse.keys())
    system = PROMPTS["relevance"]

    tp, fp, tn, fn, tested = 0, 0, 0, 0, 0
    for q in queries:
        if tested >= max_pairs:
            break
        expected = set(q.get("expected_gse", []))
        if not expected:
            continue
        positives = [g for g in expected if g in doc_by_gse][:2]
        negatives = _rnd.sample([g for g in all_gses if g not in expected],
                                min(len(positives), 2))
        for gse, is_pos in [(g, True) for g in positives] + [(g, False) for g in negatives]:
            if tested >= max_pairs:
                break
            tested += 1
            d = doc_by_gse[gse]
            prompt = (f"Query: {q['query']}\nDataset [{gse}]: {d.get('title','')}\n"
                      f"  Organism: {d.get('organism','')}\n"
                      f"  Summary: {d.get('summary','')[:300]}\nIs this relevant?")
            try:
                raw = llm_call(prompt, model_key=model_key, system=system,
                               temperature=0.0, max_tokens=256, think=think, timeout=60)
                parsed = extract_json(raw)
                pred = parsed.get("relevant", False) if parsed else False
                if is_pos and pred: tp += 1
                elif is_pos and not pred: fn += 1
                elif not is_pos and pred: fp += 1
                else: tn += 1
            except Exception:
                if is_pos: fn += 1
                else: tn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    return {"accuracy": round(acc, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": total}


# ---------------------------------------------------------------------------
# KV Cache Ablation
# ---------------------------------------------------------------------------

def run_kv_ablation(models: list, queries: list, docs: list) -> dict:
    """Run KV cache ablation: f16 vs q8_0 vs q4_0."""
    results = {}
    for kv_type in KV_TYPES:
        logger.info(f"\n{'='*60}")
        logger.info(f"KV Cache: {kv_type}")
        logger.info(f"{'='*60}")

        kill_ollama()
        if not start_ollama(kv_type):
            logger.error(f"Failed to start Ollama with kv_cache={kv_type}")
            continue

        for mk in models:
            if mk not in LLM_MODELS:
                continue
            label = f"{mk}@kv={kv_type}"
            if label in results:
                continue

            logger.info(f"  {mk} (kv={kv_type})...")
            unload_models()

            think = False  # baseline, no think for ablation
            vram_before = get_vram_mb()

            r = {
                "model": mk, "kv_cache_type": kv_type,
                "task_a": run_task_a(mk, queries, think),
                "task_d": run_task_d(mk, queries, docs, think),
                "task_f": run_task_f(mk, queries, docs, think),
            }

            vram_after = get_vram_mb()
            r["vram_mb"] = round(vram_after, 0)
            results[label] = r
            logger.info(f"    A={r['task_a']} D={r['task_d']} F={r['task_f']} VRAM={vram_after:.0f}MB")

    # Restore default
    kill_ollama()
    start_ollama("f16")
    return results


# ---------------------------------------------------------------------------
# Context Length Ablation
# ---------------------------------------------------------------------------

def run_ctx_ablation(models: list, queries: list, docs: list) -> dict:
    """Run context length ablation: 2048, 4096, 8192, 16384."""
    import scmetaintel.config as cfg_module

    results = {}
    for ctx_len in CTX_LENGTHS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Context Length: {ctx_len}")
        logger.info(f"{'='*60}")

        # Set num_ctx globally (read by answer.py)
        cfg_module.DEFAULT_NUM_CTX = ctx_len
        os.environ["SCMETA_NUM_CTX"] = str(ctx_len)

        for mk in models:
            if mk not in LLM_MODELS:
                continue
            # Skip if model's max ctx is less than test value
            model_ctx = LLM_MODELS[mk].get("ctx", 32768)
            if ctx_len > model_ctx:
                logger.info(f"  Skip {mk} (max ctx={model_ctx} < {ctx_len})")
                continue

            label = f"{mk}@ctx={ctx_len}"
            if label in results:
                continue

            logger.info(f"  {mk} (num_ctx={ctx_len})...")
            unload_models()

            think = False
            r = {
                "model": mk, "num_ctx": ctx_len,
                "task_d": run_task_d(mk, queries, docs, think),
                "task_e": run_task_e(mk, think),
            }
            results[label] = r
            logger.info(f"    D={r['task_d']} E={r['task_e']}")

    # Restore default
    cfg_module.DEFAULT_NUM_CTX = 4096
    os.environ["SCMETA_NUM_CTX"] = "4096"
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Runtime ablation studies")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--skip-kv", action="store_true", help="Skip KV cache ablation")
    parser.add_argument("--skip-ctx", action="store_true", help="Skip context length ablation")
    args = parser.parse_args()

    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    logger.info(f"Models: {args.models}")
    logger.info(f"Queries: {len(queries)}, Docs: {len(docs)}")

    all_results = {}

    # Load existing
    results_path = RESULTS_DIR / "ablation_bench.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    if not args.skip_kv:
        logger.info("\n" + "=" * 60)
        logger.info("PART A: KV Cache Ablation")
        logger.info("=" * 60)
        kv_results = run_kv_ablation(args.models, queries, docs)
        all_results["kv_cache"] = kv_results

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if not args.skip_ctx:
        logger.info("\n" + "=" * 60)
        logger.info("PART B: Context Length Ablation")
        logger.info("=" * 60)
        ctx_results = run_ctx_ablation(args.models, queries, docs)
        all_results["context_length"] = ctx_results

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    unload_models()
    logger.info("Ablation studies complete.")


if __name__ == "__main__":
    main()
