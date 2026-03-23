"""
Evaluation utilities for scMetaIntel-Hub.

This module merges:
- the metric implementations from GEO-DataHub/scmetaintel/evaluate.py
- the curated query/ontology test-set idea from scMetaIntel/scmetaintel/eval.py
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default evaluation assets
# ---------------------------------------------------------------------------

DEFAULT_EVAL_QUERIES = [
    {
        "id": "q01",
        "query": "human lung cancer scRNA-seq",
        "expected_constraints": {
            "organism": "Homo sapiens",
            "tissue": "lung",
            "disease": "cancer",
        },
        "expected_gse": ["GSE175975"],
        "category": "basic",
        "difficulty": "easy",
    },
    {
        "id": "q02",
        "query": "breast cancer single-cell RNA sequencing",
        "expected_constraints": {
            "tissue": "breast",
            "disease": "cancer",
        },
        "expected_gse": ["GSE110686", "GSE248214"],
        "category": "basic",
        "difficulty": "easy",
    },
    {
        "id": "q03",
        "query": "development colon human single cell",
        "expected_constraints": {
            "organism": "Homo sapiens",
            "tissue": "colon",
        },
        "expected_gse": ["GSE185224"],
        "category": "development",
        "difficulty": "medium",
    },
]

DEFAULT_ONTOLOGY_EVAL = [
    {"raw": "lung", "category": "tissue", "expected_id": "UBERON:0002048"},
    {"raw": "brain", "category": "tissue", "expected_id": "UBERON:0000955"},
    {"raw": "T cell", "category": "cell_type", "expected_id": "CL:0000084"},
    {"raw": "macrophage", "category": "cell_type", "expected_id": "CL:0000235"},
    {"raw": "melanoma", "category": "disease", "expected_id": "MONDO:0005105"},
]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def compute_retrieval_metrics(retrieved: List[str], relevant: Set[str]) -> Dict[str, float]:
    return {
        "p_at_5": round(precision_at_k(retrieved, relevant, 5), 4),
        "p_at_10": round(precision_at_k(retrieved, relevant, 10), 4),
        "r_at_50": round(recall_at_k(retrieved, relevant, 50), 4),
        "mrr": round(mrr(retrieved, relevant), 4),
        "ndcg_at_10": round(ndcg_at_k(retrieved, relevant, 10), 4),
    }


def aggregate_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = [k for k in per_query[0].keys() if isinstance(per_query[0][k], (int, float))]
    return {k: round(float(np.mean([q[k] for q in per_query])), 4) for k in keys}


# ---------------------------------------------------------------------------
# Extraction / answer quality metrics
# ---------------------------------------------------------------------------


def field_f1(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    pred_set = {str(s).lower().strip() for s in predicted}
    gold_set = {str(s).lower().strip() for s in gold}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # Fuzzy matching: a predicted value matches a gold value if either
    # contains the other or they share a synonym normalization.
    tp_pred = 0  # how many predicted match some gold
    for p in pred_set:
        if any(_fuzzy_match(p, g) for g in gold_set):
            tp_pred += 1
    tp_gold = 0  # how many gold match some predicted
    for g in gold_set:
        if any(_fuzzy_match(p, g) for p in pred_set):
            tp_gold += 1
    precision = tp_pred / len(pred_set) if pred_set else 0.0
    recall = tp_gold / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def extraction_metrics(predicted: Dict, gold: Dict, fields: Optional[List[str]] = None) -> Dict[str, Dict]:
    fields = fields or ["tissues", "diseases", "cell_types", "modalities"]
    out = {}
    for f in fields:
        pred_vals = predicted.get(f, [])
        gold_vals = gold.get(f, [])
        if isinstance(pred_vals, str):
            pred_vals = [pred_vals]
        if isinstance(gold_vals, str):
            gold_vals = [gold_vals]
        out[f] = field_f1(pred_vals, gold_vals)
    return out


def citation_accuracy(cited_gse: List[str], relevant_gse: Set[str], retrieved_gse: List[str]) -> Dict[str, float]:
    cited_set = set(cited_gse)
    retrieved_set = set(retrieved_gse)
    if not cited_set:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "grounding_rate": 0.0}
    prec = len(cited_set & relevant_gse) / len(cited_set) if cited_set else 0.0
    rec = len(cited_set & relevant_gse) / len(relevant_gse) if relevant_gse else 0.0
    grounding = len(cited_set & retrieved_set) / len(cited_set) if cited_set else 0.0
    return {
        "citation_precision": round(prec, 4),
        "citation_recall": round(rec, 4),
        "grounding_rate": round(grounding, 4),
    }


# ---------------------------------------------------------------------------
# Query parsing metrics
# ---------------------------------------------------------------------------


_SYNONYM_MAP: Dict[str, str] = {
    "human": "homo sapiens",
    "homo sapiens": "homo sapiens",
    "mouse": "mus musculus",
    "mus musculus": "mus musculus",
    "rat": "rattus norvegicus",
    "rattus norvegicus": "rattus norvegicus",
    "zebrafish": "danio rerio",
    "danio rerio": "danio rerio",
    "fruit fly": "drosophila melanogaster",
    "drosophila": "drosophila melanogaster",
    "drosophila melanogaster": "drosophila melanogaster",
    "scrna-seq": "scrna-seq",
    "single-cell rna-seq": "scrna-seq",
    "single cell rna-seq": "scrna-seq",
    "single-cell rna sequencing": "scrna-seq",
    "scatac-seq": "scatac-seq",
    "single-cell atac-seq": "scatac-seq",
    "cite-seq": "cite-seq",
    "spatial transcriptomics": "spatial",
    "10x visium": "spatial",
    "multiome": "multiome",
}


def _normalize_field(val: str) -> str:
    """Normalize a field value via synonym lookup."""
    v = val.lower().strip()
    return _SYNONYM_MAP.get(v, v)


def _fuzzy_match(pred: str, gold: str) -> bool:
    """Check if two values match, allowing containment and synonyms."""
    p = _normalize_field(pred)
    g = _normalize_field(gold)
    if not p and not g:
        return True
    if not p or not g:
        return False
    if p == g:
        return True
    # Containment: "brain" in "brain tissue" or vice versa
    if p in g or g in p:
        return True
    return False


def query_parsing_metrics(predicted: Dict, gold: Dict) -> Dict[str, float]:
    """Evaluate parsed query constraints against gold standard.

    Both predicted and gold are dicts like:
        {"organism": "Homo sapiens", "tissue": "brain", "disease": null, ...}

    Returns per-field accuracy and exact match score.
    Uses synonym normalization and containment matching for robustness.
    """
    fields = ["organism", "tissue", "disease", "cell_type", "assay"]
    correct = 0
    total = 0
    per_field = {}
    for f in fields:
        gold_val = (gold.get(f) or "").lower().strip()
        pred_val = (predicted.get(f) or "").lower().strip()
        # Both empty = correct, fuzzy match = correct
        match = _fuzzy_match(pred_val, gold_val)
        per_field[f] = 1.0 if match else 0.0
        if gold_val or pred_val:  # only count fields that are non-empty in either
            total += 1
            if match:
                correct += 1
    accuracy = correct / total if total > 0 else 1.0
    exact_match = 1.0 if all(v == 1.0 for v in per_field.values()) else 0.0
    return {
        "field_accuracy": round(accuracy, 4),
        "exact_match": exact_match,
        **{f"f_{k}": v for k, v in per_field.items()},
    }


# ---------------------------------------------------------------------------
# Ontology normalization metrics
# ---------------------------------------------------------------------------


def ontology_metrics(predicted: List[Dict], gold: List[Dict]) -> Dict[str, float]:
    """Evaluate ontology normalization against gold mappings.

    Each item is {"raw": str, "ontology_id": str, "ontology_label": str, "confidence": float}.
    Matches by raw term (case-insensitive), checks ontology_id exact match.
    """
    if not gold:
        return {"accuracy": 1.0 if not predicted else 0.0, "recall": 1.0 if not predicted else 0.0}

    gold_by_raw = {g["raw"].lower().strip(): g for g in gold}
    pred_by_raw = {p["raw"].lower().strip(): p for p in predicted}

    correct = 0
    for raw_key, gold_item in gold_by_raw.items():
        pred_item = pred_by_raw.get(raw_key)
        if pred_item and pred_item.get("ontology_id") == gold_item.get("ontology_id"):
            correct += 1

    accuracy = correct / len(pred_by_raw) if pred_by_raw else 0.0
    recall = correct / len(gold_by_raw) if gold_by_raw else 0.0
    f1 = (2 * accuracy * recall / (accuracy + recall)) if (accuracy + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_predicted": len(pred_by_raw),
        "n_gold": len(gold_by_raw),
        "n_correct": correct,
    }


# ---------------------------------------------------------------------------
# Relevance judgment metrics
# ---------------------------------------------------------------------------


def relevance_metrics(predicted: Dict, gold: Dict) -> Dict[str, float]:
    """Evaluate a single relevance judgment prediction.

    Both predicted and gold are dicts like:
        {"relevant": bool, "score": float, "reasoning": str}

    Returns classification accuracy and score error.
    """
    pred_relevant = bool(predicted.get("relevant", False))
    gold_relevant = bool(gold.get("relevant", False))
    binary_correct = 1.0 if pred_relevant == gold_relevant else 0.0

    pred_score = float(predicted.get("score", 0.5))
    gold_score = float(gold.get("score", 0.5))
    score_error = abs(pred_score - gold_score)

    return {
        "binary_accuracy": binary_correct,
        "score_abs_error": round(score_error, 4),
    }


def relevance_metrics_batch(predictions: List[Dict], golds: List[Dict]) -> Dict[str, float]:
    """Aggregate relevance metrics over a batch of predictions."""
    if not predictions or not golds:
        return {"binary_accuracy": 0.0, "mean_score_error": 1.0}
    results = [relevance_metrics(p, g) for p, g in zip(predictions, golds)]
    return {
        "binary_accuracy": round(float(np.mean([r["binary_accuracy"] for r in results])), 4),
        "mean_score_error": round(float(np.mean([r["score_abs_error"] for r in results])), 4),
    }


# ---------------------------------------------------------------------------
# Calibration metrics (for post-fine-tuning evaluation)
# ---------------------------------------------------------------------------


def calibration_metrics(
    predictions: List[Dict],
    golds: List[Dict],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Brier score.

    For relevance judgment tasks where predictions produce confidence scores.
    Each prediction has {"relevant": bool, "score": float}.
    Each gold has {"relevant": bool}.
    """
    if not predictions or not golds:
        return {"ece": 1.0, "brier": 1.0, "n_samples": 0}

    confidences = []
    correctnesses = []
    for pred, gold in zip(predictions, golds):
        conf = float(pred.get("score", 0.5))
        correct = 1.0 if bool(pred.get("relevant")) == bool(gold.get("relevant")) else 0.0
        confidences.append(conf)
        correctnesses.append(correct)

    confidences = np.array(confidences)
    correctnesses = np.array(correctnesses)

    # Brier score: mean squared error between confidence and ground truth label
    gold_labels = np.array([1.0 if bool(g.get("relevant")) else 0.0 for g in golds])
    brier = float(np.mean((confidences - gold_labels) ** 2))

    # ECE: binned calibration error
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctnesses[mask].mean()
            ece += (n_in_bin / len(confidences)) * abs(avg_acc - avg_conf)

    return {
        "ece": round(float(ece), 4),
        "brier": round(float(brier), 4),
        "n_samples": len(predictions),
    }


# ---------------------------------------------------------------------------
# Asset loading helpers
# ---------------------------------------------------------------------------


def load_eval_queries(path: Optional[Path] = None) -> List[Dict]:
    cfg = get_config()
    p = path or (cfg.paths.benchmark_dir / "eval_queries.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return DEFAULT_EVAL_QUERIES


def load_ontology_eval() -> List[Dict]:
    return DEFAULT_ONTOLOGY_EVAL


def save_results(results: Dict, name: str, results_dir: Optional[Path] = None):
    cfg = get_config()
    d = results_dir or cfg.paths.results_dir
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out}")
