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
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
        raw_gold = gold.get(f) or ""
        raw_pred = predicted.get(f) or ""
        # Coerce lists to first element (some models return lists instead of strings)
        if isinstance(raw_gold, list):
            raw_gold = raw_gold[0] if raw_gold else ""
        if isinstance(raw_pred, list):
            raw_pred = raw_pred[0] if raw_pred else ""
        gold_val = str(raw_gold).lower().strip()
        pred_val = str(raw_pred).lower().strip()
        # Both empty = correct, fuzzy match = correct
        match = _fuzzy_match(pred_val, gold_val)
        per_field[f] = 1.0 if match else 0.0
        if gold_val or pred_val:  # only count fields that are non-empty in either
            total += 1
            if match:
                correct += 1
    accuracy = correct / total if total > 0 else 1.0
    # Exact match: only require gold-specified fields to match (ignore predicted-only extras)
    gold_fields = [f for f in fields if str(gold.get(f) or "").strip()]
    exact_match = 1.0 if all(per_field[f] == 1.0 for f in gold_fields) else 0.0 if gold_fields else 1.0
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
    Matches by raw term (case-insensitive). Accepts EITHER exact ontology_id match
    OR fuzzy ontology_label match (tests concept understanding, not ID memorization).
    """
    if not gold:
        return {"accuracy": 1.0 if not predicted else 0.0, "recall": 1.0 if not predicted else 0.0}

    gold_by_raw = {g["raw"].lower().strip(): g for g in gold}
    pred_by_raw = {p["raw"].lower().strip(): p for p in predicted}

    correct_id = 0
    correct_label = 0
    for raw_key, gold_item in gold_by_raw.items():
        pred_item = pred_by_raw.get(raw_key)
        if not pred_item:
            continue
        if pred_item.get("ontology_id") == gold_item.get("ontology_id"):
            correct_id += 1
        pred_label = (pred_item.get("ontology_label") or pred_item.get("raw") or "").lower().strip()
        gold_label = (gold_item.get("ontology_label") or gold_item.get("raw") or "").lower().strip()
        if pred_label and gold_label and _fuzzy_match(pred_label, gold_label):
            correct_label += 1

    n_pred = len(pred_by_raw)
    n_gold = len(gold_by_raw)
    correct = max(correct_id, correct_label)
    accuracy = correct / n_pred if n_pred else 0.0
    recall = correct / n_gold if n_gold else 0.0
    f1 = (2 * accuracy * recall / (accuracy + recall)) if (accuracy + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_predicted": n_pred,
        "n_gold": n_gold,
        "n_correct_id": correct_id,
        "n_correct_label": correct_label,
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
# Gold-truth cleaning helpers (shared by bench 04 and 06)
# ---------------------------------------------------------------------------

# Cell line names that should NOT appear in tissue gold truth
CELL_LINE_NAMES = {
    "a172", "a549", "a13lg", "hek293", "hek293t", "hela", "hepg2",
    "jurkat", "k562", "lncap", "mcf7", "mcf-7", "thp-1", "u937",
    "pc9", "snu719", "akata", "mutu", "yccel1", "4t1",
    "huh-1", "huh-7", "huh1", "huh7", "caco-2", "caco2",
}

# Terms that are NOT diseases
NOT_DISEASE = {
    "development", "immunology", "in vitro", "healthy", "control",
    "controls", "normal", "case", "mild", "parental and differentiated",
    "differentiated", "aging", "ageing",
}

# Disease keywords — if these appear in a tissue term, move to diseases
DISEASE_KEYWORDS = {
    "cancer", "tumor", "tumour", "carcinoma", "melanoma", "lymphoma",
    "leukemia", "leukaemia", "adenoma", "sarcoma", "glioblastoma",
    "glioma", "myeloma", "neuroblastoma", "mesothelioma", "fibrosis",
    "cirrhosis", "hepatitis", "colitis", "arthritis", "diabetes",
    "alzheimer", "parkinson", "infection", "hiv", "covid",
}

# Common disease regex patterns for text extraction
_DISEASE_PATTERNS = [
    (r"\b(alzheimer(?:'?s)?(?:\s+disease)?)\b", "Alzheimer's disease"),
    (r"\b(parkinson(?:'?s)?(?:\s+disease)?)\b", "Parkinson's disease"),
    (r"\bbreast\s+(?:cancer|carcinoma|tumor)\b", "breast cancer"),
    (r"\blung\s+(?:cancer|adenocarcinoma|carcinoma)\b", "lung cancer"),
    (r"\bcolorectal\s+(?:cancer|carcinoma|adenocarcinoma)\b", "colorectal cancer"),
    (r"\bpancreatic\s+(?:cancer|carcinoma|adenocarcinoma)\b", "pancreatic cancer"),
    (r"\bprostate\s+(?:cancer|carcinoma|adenocarcinoma)\b", "prostate cancer"),
    (r"\b(?:liver|hepatocellular)\s+(?:cancer|carcinoma)\b", "liver cancer"),
    (r"\bglioblastoma\b", "glioblastoma"),
    (r"\bglioma\b", "glioma"),
    (r"\bmelanoma\b", "melanoma"),
    (r"\b(?:acute\s+)?leukemia\b", "leukemia"),
    (r"\blymphoma\b", "lymphoma"),
    (r"\bdiabetes(?:\s+mellitus)?\b", "diabetes"),
    (r"\bfibrosis\b", "fibrosis"),
    (r"\bcovid-?19\b", "COVID-19"),
    (r"\bsars-cov-2\b", "COVID-19"),
    (r"\bcarcinoma\b", "carcinoma"),
    (r"\bsarcoma\b", "sarcoma"),
    (r"\bneuroblastoma\b", "neuroblastoma"),
    (r"\bmyeloma\b", "myeloma"),
    (r"\b(?:crohn(?:'?s)?)\b", "Crohn's disease"),
    (r"\bhypertension\b", "hypertension"),
    (r"\batherosclerosis\b", "atherosclerosis"),
]

# Common cell type regex patterns for text extraction
_CELL_TYPE_PATTERNS = [
    (r"\bmacrophage[s]?\b", "macrophages"),
    (r"\bneutrophil[s]?\b", "neutrophils"),
    (r"\bt\s*cell[s]?\b", "T cells"),
    (r"\bb\s*cell[s]?\b", "B cells"),
    (r"\bfibroblast[s]?\b", "fibroblasts"),
    (r"\bneuron[s]?\b", "neurons"),
    (r"\bastrocyte[s]?\b", "astrocytes"),
    (r"\bmicroglia[l]?\b", "microglia"),
    (r"\boligodendrocyte[s]?\b", "oligodendrocytes"),
    (r"\bendothelial\s+cell[s]?\b", "endothelial cells"),
    (r"\bepithelial\s+cell[s]?\b", "epithelial cells"),
    (r"\bcardiomyocyte[s]?\b", "cardiomyocytes"),
    (r"\bhepatocyte[s]?\b", "hepatocytes"),
    (r"\bmonocyte[s]?\b", "monocytes"),
    (r"\bdendritic\s+cell[s]?\b", "dendritic cells"),
    (r"\bnk\s+cell[s]?\b", "NK cells"),
]


def clean_tissue_list(tissues: List[str], diseases: List[str]) -> Tuple[List[str], List[str]]:
    """Clean tissue list: remove cell lines, move disease terms to disease list."""
    clean_tissues = []
    for t in tissues:
        t_low = t.lower().strip()
        if t_low in CELL_LINE_NAMES or any(
            cl in t_low for cl in CELL_LINE_NAMES if len(cl) > 3
        ):
            continue
        if re.match(r'^[a-z0-9_-]+$', t_low) and any(c.isdigit() for c in t_low):
            continue
        if any(dk in t_low for dk in DISEASE_KEYWORDS):
            if t not in diseases:
                diseases.append(t)
            continue
        clean_tissues.append(t)
    return clean_tissues, diseases


def clean_disease_list(diseases: List[str]) -> List[str]:
    """Remove non-disease terms from disease list."""
    return [d for d in diseases if d.lower().strip() not in NOT_DISEASE
            and not re.match(r'^[0-9x?]+$', d.lower().strip())]


def extract_diseases_from_text(title: str, summary: str,
                               existing_diseases: List[str]) -> List[str]:
    """Extract disease terms from title/summary when diseases list is empty."""
    if existing_diseases:
        return existing_diseases
    text_low = (title + " " + summary).lower()
    found = []
    for pattern, disease_name in _DISEASE_PATTERNS:
        if re.search(pattern, text_low):
            if disease_name not in found:
                found.append(disease_name)
    return found


def extract_cell_types_from_text(title: str, summary: str,
                                 existing: List[str]) -> List[str]:
    """Extract cell type terms from title/summary when cell_types is empty."""
    if existing:
        return existing
    text_low = (title + " " + summary).lower()
    found = []
    for pattern, cell_name in _CELL_TYPE_PATTERNS:
        if re.search(pattern, text_low):
            if cell_name not in found:
                found.append(cell_name)
    return found


def clean_extraction_gold(doc: Dict) -> Dict[str, List[str]]:
    """Build clean extraction gold truth from a GSE doc.

    Applies the full cleaning pipeline:
    1. Text-presence filter (only terms extractable from title+summary)
    2. Remove cell lines from tissues
    3. Move disease terms from tissues to diseases
    4. Remove non-disease terms from diseases
    5. Extract diseases from text when list is empty
    6. Extract cell types from text when list is empty
    """
    title = doc.get("title", "")
    summary = doc.get("summary", "")
    text = (title + " " + summary).lower()

    gold = {}
    for field in ["tissues", "diseases", "cell_types"]:
        raw = doc.get(field, []) or []
        gold[field] = [t for t in raw if t and len(t) >= 3 and t.lower() in text]

    # Clean tissues: remove cell lines, migrate diseases
    gold["tissues"], gold["diseases"] = clean_tissue_list(
        gold["tissues"], gold["diseases"]
    )
    # Clean diseases: remove non-disease terms
    gold["diseases"] = clean_disease_list(gold["diseases"])
    # Fill empty diseases from text
    gold["diseases"] = extract_diseases_from_text(
        title, summary, gold["diseases"]
    )
    # Fill empty cell types from text
    gold["cell_types"] = extract_cell_types_from_text(
        title, summary, gold["cell_types"]
    )
    return gold


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
