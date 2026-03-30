#!/usr/bin/env python3
"""
Benchmark 04 — LLM Comparison
==============================
Compare all local LLMs on 8 tasks, with thinking-mode ablation:
  A. Query parsing (structured extraction from natural language)
  B. Metadata extraction (tissues/diseases/cell_types from GEO text)
  C. Ontology normalization (term → ontology ID mapping)
  D. Answer generation (cited answers from retrieved studies)
  E. Inference speed (diverse prompt types)
  F. Relevance judgment (binary query-doc classification)
  G. Domain classification (cancer/development/immunology/... from text)
  H. Organism & modality extraction (binomial name + assay type)

Think-mode handling per model family:
  - API-triggered (Qwen3/3.5, Gemma3, Granite3.3): tested with think=True and think=False
  - Always-on (DeepSeek-R1): tested once (CoT always embedded in response)
  - No think mode (Llama, Mistral, Phi, Falcon, Aya, GLM): tested once (think=False)

Requires Ollama server running with models pulled.

Usage:
    conda run -n dl python benchmarks/04_bench_llm.py
    conda run -n dl python benchmarks/04_bench_llm.py --models qwen3.5-9b qwen3-8b
    conda run -n dl python benchmarks/04_bench_llm.py --no-think-ablation
"""

import argparse
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

from scmetaintel.config import (
    LLM_MODELS, GROUND_TRUTH_DIR, BENCHMARK_DIR, family_always_thinks,
)
from scmetaintel.answer import (
    parse_query, extract_metadata, generate_answer, llm_call, extract_json,
)
from scmetaintel.evaluate import (
    field_f1, extraction_metrics, citation_accuracy,
    query_parsing_metrics, ontology_metrics,
    load_eval_queries, save_results,
    clean_extraction_gold,
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


def task_a_query_parsing(model_key: str, queries: list, think: bool = False,
                         max_queries: int = 0) -> dict:
    """Task A: Parse natural language queries into structured JSON."""
    from scmetaintel.evaluate import query_parsing_metrics
    sample = queries[:max_queries] if max_queries > 0 else queries
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

    return {
        "exact_match": round(correct / total, 4) if total else 0,
        "field_accuracy": {f: round(np.mean(s), 4) if s else 0
                           for f, s in field_scores.items()},
        "n_queries": total,
    }


def task_b_metadata_extraction(model_key: str, docs: list, think: bool = False,
                               max_docs: int = 100) -> dict:
    """Task B: Extract metadata from GEO title + summary."""
    limit = max_docs if max_docs > 0 else len(docs)
    all_metrics = []
    for doc in docs[:limit]:
        # Build clean gold truth: text-presence filter + cell-line removal +
        # disease migration + non-disease removal + text-based gap filling.
        gold = clean_extraction_gold(doc)
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


# Diverse speed-test prompts covering the 5 pipeline workload types.
# Each prompt is representative of what the model actually does in production.
SPEED_PROMPTS = [
    # Short structured extraction (Task A-like)
    'Extract structured constraints from this query: "human lung cancer scRNA-seq"\n'
    'Return JSON: {"organism": "", "tissue": "", "disease": ""}',
    # Medium extraction (Task B-like)
    'Title: Single-cell RNA-seq reveals heterogeneity in pancreatic ductal adenocarcinoma\n'
    'Summary: We performed scRNA-seq on surgically resected PDAC tumors and adjacent '
    'normal pancreas from 5 patients to characterize the tumor microenvironment.\n'
    'Return JSON: {"tissues": [], "diseases": [], "cell_types": [], "organism": ""}',
    # Ontology mapping (Task C-like)
    'Normalize these biomedical terms to ontology IDs:\n["brain", "T cells", "melanoma"]\n'
    'Return JSON: {"normalized": [{"raw": "", "ontology_id": "", "ontology_label": ""}]}',
    # Answer generation (Task D-like)
    '[GSE175975] Single-cell RNA-seq of human melanoma\n  Organism: Homo sapiens\n'
    '  Summary: scRNA-seq profiling of melanoma tumors and matched normal skin.\n\n'
    'Query: What single-cell datasets study melanoma immune infiltration?\n'
    'Provide a comprehensive answer citing GSE accessions.',
    # Binary judgment (Task F-like)
    'Query: mouse brain Alzheimer scRNA-seq\n'
    'Dataset [GSE142858]: Single cell RNA-seq of microglia in 3XTg-AD mice\n'
    '  Organism: Mus musculus\n  Summary: scRNA-seq of microglia from AD mouse model.\n'
    'Is this dataset relevant? Return JSON: {"relevant": true/false}',
]


def task_e_speed(model_key: str, think: bool = False) -> dict:
    """Task E: Inference speed across diverse prompt types."""
    from scmetaintel.answer import ollama_generate

    # Warm up with short prompt (retry once on failure)
    for attempt in range(2):
        try:
            llm_call(SPEED_PROMPTS[0], model_key=model_key, max_tokens=50,
                     timeout=180)
            break
        except Exception as e:
            if attempt == 1:
                return {"error": f"model not available: {e}"}

    cfg = LLM_MODELS[model_key]
    all_times, all_tokens = [], []
    per_type = []

    for prompt in SPEED_PROMPTS:
        t0 = time.time()
        result = ollama_generate(prompt, model=cfg["ollama_name"],
                                 max_tokens=200, think=think)
        elapsed = time.time() - t0
        n_tok = result.get("eval_count", 0)
        all_times.append(elapsed)
        all_tokens.append(n_tok)
        per_type.append({
            "time_sec": round(elapsed, 3),
            "tokens": n_tok,
            "tok_per_sec": round(n_tok / elapsed, 1) if elapsed > 0 else 0,
        })

    avg_time = np.mean(all_times)
    avg_tokens = np.mean(all_tokens)
    tok_per_sec = avg_tokens / avg_time if avg_time > 0 else 0

    return {
        "avg_time_sec": round(avg_time, 3),
        "avg_tokens": round(avg_tokens, 1),
        "tokens_per_sec": round(tok_per_sec, 1),
        "think_enabled": think,
        "n_prompts": len(SPEED_PROMPTS),
        "per_prompt": per_type,
    }


# Use centralized prompts from config — single source of truth
from scmetaintel.config import PROMPTS, family_json_hint, resolve_model_family
ONTOLOGY_SYSTEM = PROMPTS["ontology"]
ANSWER_BENCH_SYSTEM = PROMPTS["answer_bench"]
RELEVANCE_SYSTEM_BASE = PROMPTS["relevance"]


def task_c_ontology_normalization(model_key: str, docs: list, think: bool = False,
                                  max_docs: int = 100) -> dict:
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

    def _normalize_plural(term: str) -> str:
        """Strip common English plurals and abbreviations for ontology matching."""
        t = term.lower().strip()
        # Common bio abbreviations
        abbrevs = {"pbmcs": "peripheral blood mononuclear cell",
                   "pbmc": "peripheral blood mononuclear cell",
                   "hscs": "hematopoietic stem cell",
                   "hsc": "hematopoietic stem cell",
                   "mscs": "mesenchymal stem cell",
                   "msc": "mesenchymal stem cell",
                   "nk cells": "natural killer cell",
                   "nk cell": "natural killer cell",
                   "ipsc": "induced pluripotent stem cell",
                   "ipscs": "induced pluripotent stem cell"}
        if t in abbrevs:
            return abbrevs[t]
        # Strip plural suffixes
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
        return t

    def _lookup_term(key: str) -> tuple:
        """Try exact match, then plural-normalised match."""
        if key in gold_lookup:
            return gold_lookup[key]
        normed = _normalize_plural(key)
        if normed in gold_lookup:
            return gold_lookup[normed]
        return None

    all_scores = []
    filtered = [d for d in docs if d.get("tissues") or d.get("cell_types")]
    test_docs = filtered[:max_docs] if max_docs > 0 else filtered
    for doc in test_docs:
        raw_terms = []
        gold_items = []
        for field in ["tissues", "cell_types"]:
            for term in (doc.get(field, []) or [])[:3]:
                if term and len(term) >= 3:
                    key = term.lower().strip()
                    result = _lookup_term(key)
                    if result:
                        ont_id, ont_label = result
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
            family = resolve_model_family(model_key)
            onto_system = ONTOLOGY_SYSTEM + "\n" + family_json_hint(family)
            raw = llm_call(prompt, model_key=model_key, system=onto_system,
                           temperature=0.0, max_tokens=max_tok, think=think, timeout=300)
            parsed = extract_json(raw)
            if parsed:
                pred_items = parsed.get("normalized", [])
            else:
                pred_items = []
            score = ontology_metrics(pred_items, gold_items)
            all_scores.append(score)
        except Exception as e:
            logger.warning(f"  Ontology failed for {doc.get('gse_id')}: {e}")

    if not all_scores:
        return {"error": "no successful ontology normalizations"}

    return {
        "accuracy": round(np.mean([s["accuracy"] for s in all_scores]), 4),
        "recall": round(np.mean([s["recall"] for s in all_scores]), 4),
        "f1": round(np.mean([s["f1"] for s in all_scores]), 4),
        "n_docs": len(all_scores),
    }


def task_f_relevance_judgment(model_key: str, queries: list, docs: list,
                              think: bool = False, max_pairs: int = 0) -> dict:
    """Task F: Relevance judgment — binary classification of query-doc relevance.

    For each query, tests positive pairs (expected_gse) and negative pairs
    (random non-matching GSEs). Measures precision, recall, F1, and accuracy.
    """
    doc_by_gse = {d["gse_id"]: d for d in docs}
    all_gse_ids = list(doc_by_gse.keys())

    family = resolve_model_family(model_key)
    system = RELEVANCE_SYSTEM_BASE + "\n" + family_json_hint(family)

    tp, fp, tn, fn = 0, 0, 0, 0
    n_tested = 0

    for q in queries:
        if max_pairs > 0 and n_tested >= max_pairs:
            break
        expected = set(q.get("expected_gse", []))
        if not expected:
            continue

        # Positive pairs: expected GSEs that exist in corpus
        positives = [g for g in expected if g in doc_by_gse]
        # Negative pairs: random GSEs NOT in expected (same count as positives)
        neg_pool = [g for g in all_gse_ids if g not in expected]
        random.shuffle(neg_pool)
        negatives = neg_pool[:len(positives)]

        for gse_id, is_positive in (
            [(g, True) for g in positives] + [(g, False) for g in negatives]
        ):
            if max_pairs > 0 and n_tested >= max_pairs:
                break
            n_tested += 1
            d = doc_by_gse[gse_id]
            prompt = (
                f"Query: {q['query']}\n\n"
                f"Dataset [{gse_id}]:\n"
                f"  Title: {d.get('title', '')}\n"
                f"  Organism: {d.get('organism', 'N/A')}\n"
                f"  Summary: {d.get('summary', '')[:300]}\n\n"
                f"Is this dataset relevant to the query?"
            )
            try:
                max_tok = 4096 if think else 256
                raw = llm_call(prompt, model_key=model_key, system=system,
                               temperature=0.0, max_tokens=max_tok, think=think,
                               timeout=120)
                parsed = extract_json(raw)
                predicted = parsed.get("relevant", False) if parsed else False

                if is_positive and predicted:
                    tp += 1
                elif is_positive and not predicted:
                    fn += 1
                elif not is_positive and predicted:
                    fp += 1
                else:
                    tn += 1
            except Exception as e:
                logger.warning(f"  Relevance failed for {gse_id}: {e}")
                if is_positive:
                    fn += 1
                else:
                    tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "n_pairs": total,
    }


def task_d_answer_generation(model_key: str, queries: list, docs: list,
                             think: bool = False, max_queries: int = 0) -> dict:
    """Task D: Answer generation quality — evaluate cited answers."""
    doc_by_gse = {d["gse_id"]: d for d in docs}
    all_scores = []

    q_sample = queries[:max_queries] if max_queries > 0 else queries
    for q in q_sample:
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
                              system=ANSWER_BENCH_SYSTEM,
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

    return {
        "citation_precision": round(np.mean([s["citation_precision"] for s in all_scores]), 4),
        "citation_recall": round(np.mean([s["citation_recall"] for s in all_scores]), 4),
        "grounding_rate": round(np.mean([s["grounding_rate"] for s in all_scores]), 4),
        "avg_citations": round(np.mean([s["n_cited"] for s in all_scores]), 1),
        "n_queries": len(all_scores),
    }


DOMAIN_SYSTEM = PROMPTS["classify_domain"]
ORG_MOD_SYSTEM = PROMPTS["extract_organism_modality"]

# Valid domain labels for Task G scoring
VALID_DOMAINS = {"cancer", "development", "immunology", "neurodegeneration",
                 "infectious_disease", "cardiovascular", "metabolic", "other"}


def task_g_domain_classification(model_key: str, docs: list, think: bool = False,
                                 max_docs: int = 0) -> dict:
    """Task G: Classify research domain from title + summary.

    Tests the model's ability to understand study context and assign the correct
    research domain label. Uses ground truth 'domain' field (1649 docs have it).
    """
    family = resolve_model_family(model_key)
    system = DOMAIN_SYSTEM + "\n" + family_json_hint(family)

    filtered = [d for d in docs if d.get("domain") and d["domain"] in VALID_DOMAINS]
    sample = filtered[:max_docs] if max_docs > 0 else filtered

    correct, total, invalid = 0, 0, 0
    per_domain = {}  # domain -> {"correct": n, "total": n}

    for doc in sample:
        gold = doc["domain"]
        prompt = f"Title: {doc.get('title', '')}\n\nSummary: {doc.get('summary', '')[:500]}"
        try:
            max_tok = 4096 if think else 256
            raw = llm_call(prompt, model_key=model_key, system=system,
                           temperature=0.0, max_tokens=max_tok, think=think, timeout=120)
            parsed = extract_json(raw)
            pred = (parsed.get("domain", "") or "").lower().strip() if parsed else ""

            if pred not in VALID_DOMAINS:
                invalid += 1
                pred = ""

            total += 1
            if gold not in per_domain:
                per_domain[gold] = {"correct": 0, "total": 0}
            per_domain[gold]["total"] += 1

            if pred == gold:
                correct += 1
                per_domain[gold]["correct"] += 1
        except Exception as e:
            logger.warning(f"  Domain classify failed for {doc.get('gse_id')}: {e}")

    accuracy = correct / total if total else 0.0
    domain_acc = {d: round(v["correct"] / v["total"], 4) if v["total"] else 0
                  for d, v in per_domain.items()}

    return {
        "accuracy": round(accuracy, 4),
        "per_domain": domain_acc,
        "n_docs": total,
        "n_invalid": invalid,
    }


def task_h_organism_modality(model_key: str, docs: list, think: bool = False,
                             max_docs: int = 0) -> dict:
    """Task H: Extract organism and modality from title + summary.

    Tests structured extraction of two fields that exist in ALL ground truth docs.
    Organism: exact match after normalization. Modality: set F1 matching.
    """
    family = resolve_model_family(model_key)
    system = ORG_MOD_SYSTEM + "\n" + family_json_hint(family)

    filtered = [d for d in docs if d.get("organism") and d.get("modalities")]
    sample = filtered[:max_docs] if max_docs > 0 else filtered

    org_correct, org_total = 0, 0
    mod_scores = []

    for doc in sample:
        gold_org = (doc["organism"] or "").lower().strip()
        gold_mods = {m.lower() for m in doc["modalities"]}

        prompt = f"Title: {doc.get('title', '')}\n\nSummary: {doc.get('summary', '')[:500]}"
        try:
            max_tok = 4096 if think else 256
            raw = llm_call(prompt, model_key=model_key, system=system,
                           temperature=0.0, max_tokens=max_tok, think=think, timeout=120)
            parsed = extract_json(raw)
            if not parsed:
                continue

            # Organism scoring: fuzzy match
            pred_org = (parsed.get("organism", "") or "").lower().strip()
            org_total += 1
            if pred_org == gold_org or pred_org in gold_org or gold_org in pred_org:
                org_correct += 1

            # Modality scoring: set F1
            pred_mods = parsed.get("modalities", [])
            if isinstance(pred_mods, str):
                pred_mods = [pred_mods]
            pred_set = {m.lower().strip() for m in pred_mods if m}

            if not pred_set and not gold_mods:
                mod_scores.append({"precision": 1.0, "recall": 1.0, "f1": 1.0})
            elif not pred_set or not gold_mods:
                mod_scores.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
            else:
                tp = len(pred_set & gold_mods)
                prec = tp / len(pred_set) if pred_set else 0
                rec = tp / len(gold_mods) if gold_mods else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
                mod_scores.append({"precision": prec, "recall": rec, "f1": f1})
        except Exception as e:
            logger.warning(f"  Org/mod failed for {doc.get('gse_id')}: {e}")

    org_acc = org_correct / org_total if org_total else 0
    avg_mod = {}
    if mod_scores:
        avg_mod = {
            "precision": round(np.mean([s["precision"] for s in mod_scores]), 4),
            "recall": round(np.mean([s["recall"] for s in mod_scores]), 4),
            "f1": round(np.mean([s["f1"] for s in mod_scores]), 4),
        }

    return {
        "organism_accuracy": round(org_acc, 4),
        "modality": avg_mod,
        "n_docs": org_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="LLM model keys to test")
    parser.add_argument("--no-think-ablation", action="store_true",
                        help="Skip thinking-mode ablation (only test think=False)")
    parser.add_argument("--include-spill", action="store_true",
                        help="Include models marked cpu_spill=True (skipped by default)")
    parser.add_argument("--max-parse-queries", type=int, default=0,
                        help="Max queries for Task A parsing (0=all)")
    parser.add_argument("--max-extract-docs", type=int, default=0,
                        help="Max docs for Task B extraction (0=all)")
    parser.add_argument("--max-onto-docs", type=int, default=0,
                        help="Max docs for Task C ontology normalization (0=all)")
    parser.add_argument("--max-answer-queries", type=int, default=0,
                        help="Max queries for Task D answer generation (0=all)")
    parser.add_argument("--max-relevance-pairs", type=int, default=0,
                        help="Max query-doc pairs for Task F relevance judgment (0=all)")
    parser.add_argument("--max-classify-docs", type=int, default=0,
                        help="Max docs for Task G domain classification (0=all)")
    parser.add_argument("--max-orgmod-docs", type=int, default=0,
                        help="Max docs for Task H organism/modality extraction (0=all)")
    args = parser.parse_args()

    available = check_ollama()
    if not available:
        logger.error("No Ollama models available. Exiting.")
        return

    model_keys = args.models or list(LLM_MODELS.keys())
    # Filter to installed models (exact ollama_name match)
    active = []
    # Strip :latest from available names for flexible matching
    available_base = [n.removesuffix(":latest") for n in available]
    for mk in model_keys:
        if mk in LLM_MODELS:
            cfg = LLM_MODELS[mk]
            if not cfg.get("enabled", True) and not args.models:
                logger.info(f"Skipping {mk} (disabled in config)")
                continue
            if cfg.get("cpu_spill") and not args.include_spill and not args.models:
                logger.info(f"Skipping {mk} (cpu_spill=True, use --include-spill to override)")
                continue
            ollama_name = cfg["ollama_name"]
            if ollama_name in available or ollama_name in available_base:
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
        model_family = LLM_MODELS[mk].get("family", "")
        always_thinks = family_always_thinks(model_family)

        if always_thinks:
            # DeepSeek-R1 and similar: CoT is always embedded in response.
            # Only test once — think is always on (no ablation possible).
            runs.append((mk, True, f"{mk}+think"))
        else:
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
        results["task_a_parsing"] = task_a_query_parsing(
            model_key, queries, think=think_mode,
            max_queries=args.max_parse_queries)

        logger.info("  Task B: Metadata extraction ...")
        results["task_b_extraction"] = task_b_metadata_extraction(
            model_key, docs, think=think_mode,
            max_docs=args.max_extract_docs)

        logger.info("  Task C: Ontology normalization ...")
        results["task_c_ontology"] = task_c_ontology_normalization(
            model_key, docs, think=think_mode,
            max_docs=args.max_onto_docs)

        logger.info("  Task D: Answer generation ...")
        results["task_d_answer"] = task_d_answer_generation(
            model_key, queries, docs, think=think_mode,
            max_queries=args.max_answer_queries)

        logger.info("  Task E: Speed ...")
        results["task_e_speed"] = task_e_speed(model_key, think=think_mode)

        logger.info("  Task F: Relevance judgment ...")
        results["task_f_relevance"] = task_f_relevance_judgment(
            model_key, queries, docs, think=think_mode,
            max_pairs=args.max_relevance_pairs)

        logger.info("  Task G: Domain classification ...")
        results["task_g_domain"] = task_g_domain_classification(
            model_key, docs, think=think_mode,
            max_docs=args.max_classify_docs)

        logger.info("  Task H: Organism & modality extraction ...")
        results["task_h_org_modality"] = task_h_organism_modality(
            model_key, docs, think=think_mode,
            max_docs=args.max_orgmod_docs)

        all_results[run_label] = results
        logger.info(f"  Done: {run_label}")

        # Save intermediate results after each config (crash-safe)
        save_results(all_results, "llm_bench")

    save_results(all_results, "llm_bench")
    unload_ollama_models()  # free VRAM after benchmark
    logger.info("LLM benchmark complete.")


if __name__ == "__main__":
    main()
