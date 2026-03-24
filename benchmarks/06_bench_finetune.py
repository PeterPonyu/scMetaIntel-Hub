#!/usr/bin/env python3
"""
Benchmark 06 — Fine-tuning Experiments
=======================================
Test whether domain-specific fine-tuning improves performance:
  6.1 Embedding fine-tuning (contrastive, sentence-transformers)
  6.2 Reranker fine-tuning (cross-encoder)
  6.3 LLM fine-tuning (QLoRA via unsloth)

Fine-tuning policy:
  - Base model: Qwen3-8B (fits in 24 GB VRAM with 4-bit quant + LoRA)
  - Quantization: 4-bit (bnb nf4) for training, GGUF Q4_K_M for deployment
  - LoRA config: r=16, alpha=32, targeting q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
  - Training data: 3 task types from 1,357 GSE corpus
    - Task A: Query parsing (67 eval queries -> structured JSON)
    - Task B: Metadata extraction (GSE title+summary -> structured metadata)
    - Task C: Grounded answer generation (query + context -> cited answer)
  - Training strategy: SFT with chat template, 3 epochs, lr=2e-4
  - Output: LoRA adapter + optional GGUF merge for Ollama deployment

Usage:
    conda run -n dl python benchmarks/06_bench_finetune.py --task embeddings
    conda run -n dl python benchmarks/06_bench_finetune.py --task reranker
    conda run -n dl python benchmarks/06_bench_finetune.py --task llm
    conda run -n dl python benchmarks/06_bench_finetune.py --task llm --prepare-only
    conda run -n dl python benchmarks/06_bench_finetune.py --task llm --base-model qwen3-8b
    conda run -n dl python benchmarks/06_bench_finetune.py --task llm --export-gguf
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import (
    EMBEDDING_MODELS, LLM_MODELS,
    GROUND_TRUTH_DIR, BENCHMARK_DIR,
)
from scmetaintel.evaluate import load_eval_queries, save_results

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("06_bench_ft")

FINETUNE_DIR = BENCHMARK_DIR / "finetuned"

# ---------------------------------------------------------------------------
# LLM fine-tuning configuration
# ---------------------------------------------------------------------------

# Models that fit in 24 GB VRAM with 4-bit quant + LoRA gradients
# Unsloth 2026.3.8 supports: qwen3, qwen2, llama, gemma, gemma2, mistral, cohere, phi
FINETUNE_CANDIDATES = {
    # --- PRIMARY tier: comfortable fit, fast training ---
    "qwen3-8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "vram_4bit_gb": 5,
        "family": "qwen",
        "tier": "primary",
        "recommended": True,
        "note": "Best balance of quality and VRAM for 24 GB GPU.",
    },
    "qwen3.5-9b": {
        "hf_name": "Qwen/Qwen3.5-9B",
        "vram_4bit_gb": 22.9,  # WARNING: multi-modal (Qwen3_5ForConditionalGeneration)
        "family": "qwen",
        "tier": "skip",  # NOT suitable: vision encoder eats all VRAM
        "recommended": False,
        "note": "MULTI-MODAL model — vision encoder uses ~23 GB 4-bit. OOMs on 24 GB GPU. Use qwen3-8b instead.",
    },
    "qwen2.5-7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "vram_4bit_gb": 4.5,
        "family": "qwen",
        "tier": "primary",
        "recommended": False,
        "note": "Legacy baseline for A/B comparison.",
    },
    "llama3.1-8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "vram_4bit_gb": 5,
        "family": "llama",
        "tier": "primary",
        "recommended": False,
        "note": "Best LoRA ecosystem support. Cross-architecture comparison.",
    },

    # --- VIABLE tier: works with gradient checkpointing ---
    "qwen3-14b": {
        "hf_name": "Qwen/Qwen3-14B",
        "vram_4bit_gb": 9,
        "family": "qwen",
        "tier": "viable",
        "recommended": False,
        "note": "Higher quality, uses gradient_checkpointing.",
    },
    "gemma3-12b": {
        "hf_name": "google/gemma-3-12b-it",
        "vram_4bit_gb": 7.5,
        "family": "gemma",
        "tier": "viable",
        "recommended": False,
        "note": "Cross-architecture, Google's mid-tier.",
    },
    "mistral-nemo-12b": {
        "hf_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "vram_4bit_gb": 7.5,
        "family": "mistral",
        "tier": "viable",
        "recommended": False,
        "note": "1M context window. Mistral + NVIDIA collab.",
    },
    "phi4-14b": {
        "hf_name": "microsoft/phi-4",
        "vram_4bit_gb": 9,
        "family": "phi",
        "tier": "viable",
        "recommended": False,
        "note": "Best structured output / JSON parsing among small models.",
    },
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,  # effective batch = 8
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "seed": 42,
}

# System prompts for training data
PARSE_SYSTEM = (
    "You are a biomedical search query parser. Extract structured constraints "
    "from the user's natural language query about single-cell datasets.\n"
    "Return ONLY valid JSON with these fields (use null if not mentioned):\n"
    '{"organism": "", "tissue": "", "disease": "", '
    '"cell_type": "", "assay": "", "free_text": ""}'
)

EXTRACT_SYSTEM = (
    "You are a biomedical metadata extractor. Given a GEO dataset title and "
    "summary, extract structured metadata.\n"
    "Return ONLY valid JSON with:\n"
    '{"tissues": [str], "diseases": [str], "cell_types": [str], '
    '"modalities": [str], "organism": str}'
)

ANSWER_SYSTEM = (
    "You are a scientific dataset search assistant. Answer the user's query "
    "about single-cell datasets based ONLY on the provided study information.\n"
    "Rules:\n"
    "1. Cite specific GSE accessions (e.g., GSE123456) for every claim\n"
    "2. If the context doesn't contain relevant studies, say so\n"
    "3. Be concise and factual\n"
    "4. Never fabricate GSE IDs or study details\n"
)

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

RELEVANCE_SYSTEM = (
    "You are a dataset relevance judge. Given a user query and a GEO dataset "
    "description, judge whether the dataset is relevant to the query.\n"
    "Return JSON: {\"relevant\": bool, \"score\": float (0-1), "
    "\"reasoning\": str (1-2 sentences)}"
)

# ---------------------------------------------------------------------------
# Data-cleaning helpers
# ---------------------------------------------------------------------------

# Mojibake patterns (UTF-8 bytes decoded as Latin-1, then re-encoded)
_MOJIBAKE = [
    ("â\x80\x99", "\u2019"), ("â\x80\x98", "\u2018"),  # curly quotes
    ("â\x80\x9c", "\u201c"), ("â\x80\x9d", "\u201d"),
    ("â\x80\x93", "\u2013"), ("â\x80\x94", "\u2014"),  # dashes
    ("Î±", "\u03b1"), ("Î²", "\u03b2"), ("Î³", "\u03b3"),  # Greek
    ("Î´", "\u03b4"), ("Îµ", "\u03b5"), ("Î¶", "\u03b6"),
    ("Î¼", "\u03bc"), ("Î»", "\u03bb"), ("Îº", "\u03ba"),
    ("Ï\x80", "\u03c0"), ("Ï\x83", "\u03c3"), ("Ï\x84", "\u03c4"),
    ("Ã\x97", "\u00d7"),  # multiplication sign
    ("Â±", "\u00b1"), ("Â°", "\u00b0"),  # plus-minus, degree
    ("Ã©", "\u00e9"), ("Ã¼", "\u00fc"),  # accented letters
]


def fix_mojibake(text: str) -> str:
    """Fix common UTF-8 mojibake patterns in GEO metadata."""
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    # Generic fallback: try Latin-1 round-trip for remaining garbled chars
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if fixed != text:
            text = fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return text


# Cell line names that appear in tissue fields but are NOT tissues
_CELL_LINE_NAMES = {
    "a172", "a549", "a13lg", "hek293", "hek293t", "hela", "hepg2",
    "jurkat", "k562", "lncap", "mcf7", "mcf-7", "thp-1", "u937",
    "pc9", "snu719", "akata", "mutu", "yccel1", "4t1",
    "huh-1", "huh-7", "huh1", "huh7", "caco-2", "caco2",
}

# Terms that are NOT diseases
_NOT_DISEASE = {
    "development", "immunology", "in vitro", "healthy", "control",
    "controls", "normal", "case", "mild", "parental and differentiated",
    "differentiated", "aging", "ageing",
}

# Disease keywords → if these appear in a tissue term, move to disease
_DISEASE_KEYWORDS = {
    "cancer", "tumor", "tumour", "carcinoma", "melanoma", "lymphoma",
    "leukemia", "leukaemia", "adenoma", "sarcoma", "glioblastoma",
    "glioma", "myeloma", "neuroblastoma", "mesothelioma", "fibrosis",
    "cirrhosis", "hepatitis", "colitis", "arthritis", "diabetes",
    "alzheimer", "parkinson", "infection", "hiv", "covid",
}

# Organism keywords for query parsing
_ORGANISM_KEYWORDS = {
    "human": "Homo sapiens", "mouse": "Mus musculus",
    "rat": "Rattus norvegicus", "zebrafish": "Danio rerio",
    "drosophila": "Drosophila melanogaster", "fly": "Drosophila melanogaster",
    "worm": "Caenorhabditis elegans", "c. elegans": "Caenorhabditis elegans",
    "pig": "Sus scrofa", "chicken": "Gallus gallus",
    "macaque": "Macaca mulatta", "monkey": "Macaca mulatta",
}

# Known correct ontology overrides (fixing systematic wrong IDs)
_ONTOLOGY_OVERRIDES = {
    "skin": ("UBERON:0002097", "skin of body"),
    "hippocampus": ("UBERON:0002421", "hippocampal formation"),
    "bone marrow": ("UBERON:0002371", "bone marrow"),
    "peripheral blood": ("UBERON:0013756", "venous blood"),
    "peripheral blood mononuclear cells": ("CL:2000001", "peripheral blood mononuclear cell"),
    "pbmcs": ("CL:2000001", "peripheral blood mononuclear cell"),
    "pbmc": ("CL:2000001", "peripheral blood mononuclear cell"),
    "t cells": ("CL:0000084", "T cell"),
    "t cell": ("CL:0000084", "T cell"),
    "cd4+ t cells": ("CL:0000624", "CD4-positive, alpha-beta T cell"),
    "cd8+ t cells": ("CL:0000625", "CD8-positive, alpha-beta T cell"),
    "b cells": ("CL:0000236", "B cell"),
    "b cell": ("CL:0000236", "B cell"),
    "macrophages": ("CL:0000235", "macrophage"),
    "macrophage": ("CL:0000235", "macrophage"),
    "neutrophils": ("CL:0000775", "neutrophil"),
    "neutrophil": ("CL:0000775", "neutrophil"),
    "monocytes": ("CL:0000576", "monocyte"),
    "monocyte": ("CL:0000576", "monocyte"),
    "fibroblasts": ("CL:0000057", "fibroblast"),
    "fibroblast": ("CL:0000057", "fibroblast"),
    "epithelial cells": ("CL:0000066", "epithelial cell"),
    "epithelial cell": ("CL:0000066", "epithelial cell"),
    "endothelial cells": ("CL:0000115", "endothelial cell"),
    "endothelial cell": ("CL:0000115", "endothelial cell"),
    "neurons": ("CL:0000540", "neuron"),
    "neuron": ("CL:0000540", "neuron"),
    "nk cells": ("CL:0000623", "natural killer cell"),
    "natural killer cells": ("CL:0000623", "natural killer cell"),
    "dendritic cells": ("CL:0000451", "dendritic cell"),
    "dendritic cell": ("CL:0000451", "dendritic cell"),
    "cardiomyocytes": ("CL:0000746", "cardiac muscle cell"),
    "cardiomyocyte": ("CL:0000746", "cardiac muscle cell"),
    "astrocytes": ("CL:0000127", "astrocyte"),
    "astrocyte": ("CL:0000127", "astrocyte"),
    "microglia": ("CL:0000129", "microglial cell"),
    "oligodendrocytes": ("CL:0000128", "oligodendrocyte"),
    "hepatocytes": ("CL:0000182", "hepatocyte"),
    "hepatocyte": ("CL:0000182", "hepatocyte"),
    "pancreatic islets": ("UBERON:0000006", "islet of Langerhans"),
    "islets of langerhans": ("UBERON:0000006", "islet of Langerhans"),
    "glioblastoma": ("MONDO:0018177", "glioblastoma"),
    "melanoma": ("MONDO:0005105", "melanoma"),
    "breast cancer": ("MONDO:0007254", "breast cancer"),
    "lung cancer": ("MONDO:0008903", "lung cancer"),
    "colorectal cancer": ("MONDO:0005575", "colorectal cancer"),
    "pancreatic cancer": ("MONDO:0005192", "pancreatic cancer"),
    "prostate cancer": ("MONDO:0008315", "prostate cancer"),
    "liver cancer": ("MONDO:0007256", "liver cancer"),
    "leukemia": ("MONDO:0005059", "leukemia"),
    "lymphoma": ("MONDO:0005062", "lymphoma"),
    "calvarium": ("UBERON:0004339", "vault of skull"),
    "nasal epithelium": ("UBERON:0005384", "nasal cavity epithelium"),
}


def clean_tissue_list(tissues: list, diseases: list) -> tuple[list, list]:
    """Clean tissue list: remove cell lines, move diseases to disease list."""
    clean_tissues = []
    for t in tissues:
        t_low = t.lower().strip()
        # Skip cell line names
        if t_low in _CELL_LINE_NAMES or any(
            cl in t_low for cl in _CELL_LINE_NAMES if len(cl) > 3
        ):
            continue
        # Skip sample IDs (contain digits + letters, no spaces)
        if re.match(r'^[a-z0-9_-]+$', t_low) and any(c.isdigit() for c in t_low):
            continue
        # Check if this looks like a disease — move to diseases
        if any(dk in t_low for dk in _DISEASE_KEYWORDS):
            if t not in diseases:
                diseases.append(t)
            continue
        clean_tissues.append(t)
    return clean_tissues, diseases


def clean_disease_list(diseases: list) -> list:
    """Remove non-disease terms from disease list."""
    return [d for d in diseases if d.lower().strip() not in _NOT_DISEASE
            and not re.match(r'^[0-9x?]+$', d.lower().strip())]


def extract_diseases_from_text(title: str, summary: str,
                               existing_diseases: list) -> list:
    """Extract obvious disease terms from title/summary when diseases is empty."""
    if existing_diseases:
        return existing_diseases
    text_low = (title + " " + summary).lower()
    found = []
    # Check for common disease patterns
    disease_patterns = [
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
    for pattern, disease_name in disease_patterns:
        if re.search(pattern, text_low):
            if disease_name not in found:
                found.append(disease_name)
    return found


def extract_cell_types_from_text(title: str, summary: str,
                                 existing: list) -> list:
    """Extract obvious cell type terms when cell_types is empty."""
    if existing:
        return existing
    text_low = (title + " " + summary).lower()
    found = []
    cell_patterns = [
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
    for pattern, cell_name in cell_patterns:
        if re.search(pattern, text_low):
            if cell_name not in found:
                found.append(cell_name)
    return found


# ------------------------------------------------------------------
# 6.1 Embedding fine-tuning
# ------------------------------------------------------------------

def generate_triplets(queries: list, docs: list, n_negatives: int = 5) -> list:
    """Generate (query, positive, negative) triplets for contrastive training."""
    doc_by_gse = {d["gse_id"]: d for d in docs}
    triplets = []
    all_gse = list(doc_by_gse.keys())

    for q in queries:
        query_text = q["query"]
        positives = q.get("expected_gse", [])
        for pos_gse in positives:
            if pos_gse not in doc_by_gse:
                continue
            pos_text = doc_by_gse[pos_gse].get("document_text",
                       doc_by_gse[pos_gse].get("title", ""))
            neg_pool = [g for g in all_gse if g not in positives]
            for neg_gse in random.sample(neg_pool, min(n_negatives, len(neg_pool))):
                neg_text = doc_by_gse[neg_gse].get("document_text",
                           doc_by_gse[neg_gse].get("title", ""))
                triplets.append({
                    "query": query_text,
                    "positive": pos_text,
                    "negative": neg_text,
                })
    return triplets


def finetune_embeddings(base_model: str = "biolord-2023"):
    """Fine-tune an embedding model with contrastive learning."""
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from torch.utils.data import DataLoader

    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    triplets = generate_triplets(queries, docs)
    if len(triplets) < 10:
        logger.error("Too few triplets for fine-tuning. Need more annotated queries.")
        return None

    logger.info(f"Generated {len(triplets)} training triplets")

    model_cfg = EMBEDDING_MODELS[base_model]
    device = model_cfg.get("device", "cuda")
    model = SentenceTransformer(model_cfg["name"], device=device)

    examples = [
        InputExample(texts=[t["query"], t["positive"], t["negative"]])
        for t in triplets
    ]
    dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    loss = losses.TripletLoss(model=model)

    output_path = str(FINETUNE_DIR / f"{base_model}-ft")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=3,
        warmup_steps=10,
        output_path=output_path,
        show_progress_bar=True,
    )
    logger.info(f"Fine-tuned model saved to {output_path}")
    return output_path


# ------------------------------------------------------------------
# 6.2 Reranker fine-tuning
# ------------------------------------------------------------------

def generate_reranker_data(queries: list, docs: list) -> list:
    """Generate (query, passage, label) pairs for cross-encoder training."""
    doc_by_gse = {d["gse_id"]: d for d in docs}
    pairs = []

    for q in queries:
        positives = set(q.get("expected_gse", []))
        for gse, doc in doc_by_gse.items():
            text = doc.get("document_text", doc.get("title", ""))
            label = 1 if gse in positives else 0
            pairs.append({
                "query": q["query"],
                "passage": text[:512],
                "label": label,
            })
    return pairs


def finetune_reranker():
    """Fine-tune a cross-encoder reranker."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import InputExample as CEInput
    import torch

    queries = load_eval_queries()
    docs = []
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            docs.append(json.load(f))

    pairs = generate_reranker_data(queries, docs)
    logger.info(f"Generated {len(pairs)} training pairs for reranker")

    model = CrossEncoder(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        device="cpu",  # Blackwell GPU compat
    )

    examples = [
        CEInput(texts=[p["query"], p["passage"]], label=float(p["label"]))
        for p in pairs
    ]

    output_path = str(FINETUNE_DIR / "biomedbert-reranker-ft")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=torch.utils.data.DataLoader(
            examples, shuffle=True, batch_size=16),
        epochs=3,
        warmup_steps=10,
        output_path=output_path,
        show_progress_bar=True,
    )
    logger.info(f"Fine-tuned reranker saved to {output_path}")
    return output_path


# ------------------------------------------------------------------
# 6.3 LLM fine-tuning (QLoRA via unsloth)
# ------------------------------------------------------------------

def prepare_llm_training_data() -> Path:
    """
    Generate SFT training data from 1,357 GSE corpus + 67 eval queries.

    Five task types in chat format:
      A. Query parsing: user query -> structured JSON
      B. Metadata extraction: title + summary -> structured JSON
      C. Answer generation: query + context -> grounded answer with citations
      D. Ontology normalization: raw terms -> standard ontology IDs
      E. Relevance judgment: query + dataset -> relevance score + reasoning
    """
    queries = load_eval_queries()
    docs = {}
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            d = json.load(f)
            docs[d["gse_id"]] = d

    conversations = []

    # --- Task A: Query parsing (from eval queries) ---
    for q in queries:
        constraints = q.get("expected_constraints", {})
        if not constraints:
            continue
        # Fill all 6 fields — use null for unmentioned (matches system prompt)
        full_constraints = {
            "organism": constraints.get("organism") or None,
            "tissue": constraints.get("tissue") or None,
            "disease": constraints.get("disease") or None,
            "cell_type": constraints.get("cell_type") or None,
            "assay": constraints.get("assay") or None,
            "free_text": constraints.get("free_text") or q["query"],
        }
        output = json.dumps(full_constraints, indent=2)
        conversations.append({
            "messages": [
                {"role": "system", "content": PARSE_SYSTEM},
                {"role": "user", "content": q["query"]},
                {"role": "assistant", "content": output},
            ],
            "task": "query_parsing",
        })

    # --- Task B: Metadata extraction (from GSE ground truth) ---
    # Human cell line keywords for organism validation
    _HUMAN_KEYWORDS = {"hek293", "k562", "hela", "jurkat", "u937", "thp-1",
                       "a549", "mcf7", "human", "patient", "homo sapiens"}
    _TISSUE_BLOCKLIST = {"bacterial culture", "unknown", "n/a", "na", "none"}

    for gse_id, d in docs.items():
        title = d.get("title", "")
        summary = d.get("summary", "")
        if not title or not summary:
            continue

        # Fix mojibake in text
        title = fix_mojibake(title)
        summary = fix_mojibake(summary)

        # Build gold metadata from available fields
        gold = {
            "tissues": d.get("tissues", []) or [],
            "diseases": d.get("diseases", []) or [],
            "cell_types": d.get("cell_types", []) or [],
            "modalities": d.get("modalities", []) or [],
            "organism": d.get("organism", ""),
        }

        # Validate organism: if title/summary mentions human cell lines but
        # gold says mouse, fix it
        text_lower = (title + " " + summary).lower()
        if gold["organism"].lower() in ("mus musculus", "mouse"):
            if any(kw in text_lower for kw in _HUMAN_KEYWORDS):
                gold["organism"] = "Homo sapiens"

        # Filter out invalid tissue terms
        gold["tissues"] = [t for t in gold["tissues"]
                           if t and t.lower().strip() not in _TISSUE_BLOCKLIST]

        # Clean tissue list: remove cell lines, move diseases to disease list
        gold["tissues"], gold["diseases"] = clean_tissue_list(
            gold["tissues"], gold["diseases"])

        # Clean disease list: remove non-disease terms
        gold["diseases"] = clean_disease_list(gold["diseases"])

        # Use domain as disease hint when diseases list is empty
        domain = d.get("domain", "")
        if not gold["diseases"] and domain and domain not in ("unknown", "development"):
            gold["diseases"] = [domain]

        # Extract diseases from text if still empty
        gold["diseases"] = extract_diseases_from_text(
            title, summary, gold["diseases"])

        # Build tissue/celltype from characteristics_summary if available
        cs = d.get("characteristics_summary", {})
        if isinstance(cs, dict):
            if not gold["tissues"] and cs.get("tissues"):
                raw_tissues = cs["tissues"]
                raw_tissues = [t for t in raw_tissues
                               if t and t.lower().strip() not in _TISSUE_BLOCKLIST]
                raw_tissues, gold["diseases"] = clean_tissue_list(
                    raw_tissues, gold["diseases"])
                gold["tissues"] = raw_tissues
            if not gold["cell_types"] and cs.get("cell_types"):
                gold["cell_types"] = cs["cell_types"]
            if not gold["diseases"] and cs.get("diseases"):
                gold["diseases"] = clean_disease_list(cs["diseases"])

        # Extract cell types from text if still empty
        gold["cell_types"] = extract_cell_types_from_text(
            title, summary, gold["cell_types"])

        user_text = f"Title: {title}\n\nSummary: {summary[:1500]}"
        output = json.dumps(gold, indent=2)
        conversations.append({
            "messages": [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": output},
            ],
            "task": "metadata_extraction",
        })

    # --- Task C: Answer generation (query + context -> cited answer) ---
    _ANSWER_INTROS = [
        'Based on the retrieved studies, here are the relevant datasets for "{query}":\n\n',
        'The following datasets are relevant to "{query}":\n\n',
        'For your query about "{query}", I found these relevant datasets:\n\n',
        'Here are the single-cell datasets matching "{query}":\n\n',
    ]
    for qi, q in enumerate(queries):
        expected_gse = q.get("expected_gse", [])
        context_parts = []
        for gse_id in expected_gse:
            if gse_id not in docs:
                continue
            d = docs[gse_id]
            ctx_title = fix_mojibake(d.get('title', ''))
            ctx_summary = fix_mojibake(d.get('summary', '')[:300])
            context_parts.append(
                f"[{gse_id}] {ctx_title}\n"
                f"  Organism: {d.get('organism', 'N/A')}\n"
                f"  Summary: {ctx_summary}"
            )
        if not context_parts:
            continue

        context = "\n\n".join(context_parts)
        user_text = (
            f"Retrieved studies:\n{context}\n\n"
            f"User query: {q['query']}\n\n"
            f"Provide a comprehensive answer citing relevant GSE accessions."
        )
        # Build a reference answer with varied intros
        intro = _ANSWER_INTROS[qi % len(_ANSWER_INTROS)]
        answer = intro.format(query=q['query'])
        for gse_id in expected_gse:
            if gse_id not in docs:
                continue
            d = docs[gse_id]
            a_title = fix_mojibake(d.get('title', ''))
            answer += (
                f"- [{gse_id}]: {a_title}. "
                f"Organism: {d.get('organism', 'N/A')}. "
                f"Modalities: {', '.join(d.get('modalities', ['N/A']))}.\n"
            )
        gse_list = ", ".join(f"[{g}]" for g in expected_gse if g in docs)
        answer += f"\nThese datasets {gse_list} are relevant to the query."

        conversations.append({
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ],
            "task": "answer_generation",
        })

    # --- Task D: Ontology normalization (from GSE characteristics) ---
    # Map raw tissue/cell_type terms to ontology IDs using available ontology data
    ontology_dir = Path(__file__).resolve().parent.parent / "ontologies"
    cl_map = {}  # cell type name -> CL ID
    uberon_map = {}  # tissue name -> UBERON ID
    mondo_map = {}  # disease name -> MONDO ID

    for obo_file, target_map, prefix in [
        ("cl.obo", cl_map, "CL:"),
        ("uberon-basic.obo", uberon_map, "UBERON:"),
        ("mondo.obo", mondo_map, "MONDO:"),
    ]:
        obo_path = ontology_dir / obo_file
        if not obo_path.exists():
            continue
        prefix_bare = prefix.rstrip(":")
        current_id = None
        current_name = None
        in_term = False
        is_obsolete = False
        with open(obo_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "[Term]":
                    # Save previous term
                    if current_id and current_name and not is_obsolete:
                        target_map[current_name.lower()] = (current_id, current_name)
                    current_id = None
                    current_name = None
                    in_term = True
                    is_obsolete = False
                    continue
                if line.startswith("[") and line.endswith("]"):
                    if in_term and current_id and current_name and not is_obsolete:
                        target_map[current_name.lower()] = (current_id, current_name)
                    in_term = False
                    current_id = None
                    current_name = None
                    is_obsolete = False
                    continue
                if not in_term:
                    continue
                if line.startswith("id: "):
                    cid = line[4:].strip()
                    if cid.startswith(prefix_bare):
                        current_id = cid
                    else:
                        current_id = None
                elif line.startswith("name: ") and current_id:
                    current_name = line[6:].strip()
                    target_map[current_name.lower()] = (current_id, current_name)
                elif line.startswith("synonym: ") and current_id and current_name:
                    start = line.find('"')
                    end = line.find('"', start + 1)
                    if start >= 0 and end > start:
                        syn = line[start + 1:end].strip()
                        key = syn.lower()
                        if key not in target_map:
                            target_map[key] = (current_id, current_name)
                elif line.startswith("is_obsolete: true"):
                    is_obsolete = True
        # Save last term
        if in_term and current_id and current_name and not is_obsolete:
            target_map[current_name.lower()] = (current_id, current_name)

    # Apply overrides to the ontology maps (fix known wrong IDs + add missing)
    for key, (oid, olabel) in _ONTOLOGY_OVERRIDES.items():
        if oid.startswith("UBERON"):
            uberon_map[key] = (oid, olabel)
        elif oid.startswith("CL"):
            cl_map[key] = (oid, olabel)
        elif oid.startswith("MONDO"):
            mondo_map[key] = (oid, olabel)

    all_ont_keys = set(cl_map) | set(uberon_map) | set(mondo_map)

    for gse_id, d in docs.items():
        cs = d.get("characteristics_summary", {})
        if not isinstance(cs, dict):
            continue

        mapped_terms = []
        unmapped_terms = []
        for field, ont_map, ont_name in [
            ("tissues", uberon_map, "UBERON"),
            ("cell_types", cl_map, "CL"),
            ("diseases", mondo_map, "MONDO"),
        ]:
            terms = cs.get(field, []) or d.get(field, [])
            for term in terms:
                if not term or len(term) < 3:
                    continue
                key = term.lower().strip()
                # Skip cell lines and non-biological terms
                if key in _CELL_LINE_NAMES or key in _NOT_DISEASE:
                    continue
                if key in ont_map:
                    oid, olabel = ont_map[key]
                    mapped_terms.append({
                        "raw": term,
                        "ontology_id": oid,
                        "ontology_label": olabel,
                        "confidence": 1.0,
                    })
                else:
                    unmapped_terms.append({
                        "raw": term,
                        "ontology_id": None,
                        "ontology_label": None,
                        "confidence": 0.0,
                    })

        # Create mixed examples (both mapped and unmapped) when possible
        all_terms = mapped_terms + unmapped_terms
        if not all_terms:
            continue

        user_text = f"Normalize these biomedical terms from {gse_id}:\n"
        user_text += json.dumps([t["raw"] for t in all_terms])
        output = json.dumps({"normalized": all_terms}, indent=2)

        conversations.append({
            "messages": [
                {"role": "system", "content": ONTOLOGY_SYSTEM},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": output},
            ],
            "task": "ontology_normalization",
        })

    # --- Task E: Relevance judgment (query + dataset pairs) ---
    all_gse_ids = list(docs.keys())
    for q in queries:
        expected = set(q.get("expected_gse", []))
        query_lower = q['query'].lower()
        # Positive examples (relevant datasets)
        for gse_id in expected:
            if gse_id not in docs:
                continue
            d = docs[gse_id]
            d_title = fix_mojibake(d.get('title', ''))
            d_summary = fix_mojibake(d.get('summary', '')[:500])
            user_text = (
                f"Query: {q['query']}\n\n"
                f"Dataset [{gse_id}]:\n"
                f"Title: {d_title}\n"
                f"Summary: {d_summary}"
            )
            # Build specific reasoning from metadata overlap
            overlaps = []
            if d.get('organism', '').lower() in query_lower:
                overlaps.append(f"organism ({d['organism']})")
            for t in (d.get('tissues', []) or [])[:2]:
                if t.lower() in query_lower:
                    overlaps.append(f"tissue ({t})")
            for dt in (d.get('diseases', []) or [])[:2]:
                if dt.lower() in query_lower:
                    overlaps.append(f"disease ({dt})")
            overlap_str = ", ".join(overlaps) if overlaps else "the study topic"
            reasoning = (
                f"This dataset is relevant because it matches the query on "
                f"{overlap_str} and uses single-cell sequencing."
            )
            output = json.dumps({
                "relevant": True,
                "score": 0.95,
                "reasoning": reasoning,
            }, indent=2)
            conversations.append({
                "messages": [
                    {"role": "system", "content": RELEVANCE_SYSTEM},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output},
                ],
                "task": "relevance_judgment",
            })

        # Negative examples (non-relevant datasets) — sample 2 per query
        neg_pool = [g for g in all_gse_ids if g not in expected]
        for neg_gse in random.sample(neg_pool, min(2, len(neg_pool))):
            d = docs[neg_gse]
            d_title = fix_mojibake(d.get('title', ''))
            d_summary = fix_mojibake(d.get('summary', '')[:500])
            user_text = (
                f"Query: {q['query']}\n\n"
                f"Dataset [{neg_gse}]:\n"
                f"Title: {d_title}\n"
                f"Summary: {d_summary}"
            )
            # Build specific reasoning for why it's NOT relevant
            mismatches = []
            q_org = q.get('expected_constraints', {}).get('organism', '')
            if q_org and d.get('organism', '') != q_org:
                mismatches.append(
                    f"studies {d.get('organism', 'unknown')} instead of {q_org}")
            q_tissue = q.get('expected_constraints', {}).get('tissue', '')
            if q_tissue:
                d_tissues = " ".join(d.get('tissues', []) or []).lower()
                if q_tissue.lower() not in d_tissues:
                    mismatches.append(
                        f"focuses on different tissues")
            mismatch_str = ("; ".join(mismatches)
                           if mismatches
                           else "studies a different biological context")
            reasoning = (
                f"This dataset is not relevant to the query because it "
                f"{mismatch_str}."
            )
            output = json.dumps({
                "relevant": False,
                "score": 0.1,
                "reasoning": reasoning,
            }, indent=2)
            conversations.append({
                "messages": [
                    {"role": "system", "content": RELEVANCE_SYSTEM},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": output},
                ],
                "task": "relevance_judgment",
            })

    # Merge enhanced data if available
    enhanced_path = FINETUNE_DIR / "llm_data" / "sft_train_enhanced.jsonl"
    if enhanced_path.exists():
        n_enhanced = 0
        with open(enhanced_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    conversations.append(json.loads(line))
                    n_enhanced += 1
        logger.info(f"Merged {n_enhanced} enhanced samples from {enhanced_path}")
    else:
        logger.info("No enhanced data found — run scripts/enhance_training_data.py first")

    # Deduplicate by user message content (keep first occurrence)
    seen_user_msgs = set()
    deduped = []
    for conv in conversations:
        user_msg = conv["messages"][1]["content"]  # user message
        key = (conv["task"], user_msg)
        if key in seen_user_msgs:
            continue
        seen_user_msgs.add(key)
        deduped.append(conv)
    n_removed = len(conversations) - len(deduped)
    if n_removed:
        logger.info(f"Removed {n_removed} duplicate examples")
    conversations = deduped

    # Fix mojibake in ALL conversations (including enhanced data)
    for conv in conversations:
        for msg in conv["messages"]:
            msg["content"] = fix_mojibake(msg["content"])

    # Clean "development" from disease fields in enhanced extraction data
    for conv in conversations:
        if conv["task"] == "metadata_extraction":
            try:
                resp = json.loads(conv["messages"][2]["content"])
                if "development" in resp.get("diseases", []):
                    resp["diseases"] = clean_disease_list(resp["diseases"])
                    conv["messages"][2]["content"] = json.dumps(resp, indent=2)
            except (json.JSONDecodeError, KeyError):
                pass

    # Shuffle and save
    random.seed(42)
    random.shuffle(conversations)

    out_dir = FINETUNE_DIR / "llm_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    out_path = out_dir / "sft_train.jsonl"
    with open(out_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Save task distribution for logging
    from collections import Counter
    task_dist = Counter(c["task"] for c in conversations)
    logger.info(f"Training data: {len(conversations)} conversations")
    logger.info(f"  Task distribution: {dict(task_dist)}")
    logger.info(f"  Saved to {out_path}")

    # Also save a small validation split (10%)
    n_val = max(10, len(conversations) // 10)
    val_data = conversations[:n_val]
    train_data = conversations[n_val:]

    val_path = out_dir / "sft_val.jsonl"
    train_path = out_dir / "sft_train_split.jsonl"
    with open(val_path, "w") as f:
        for conv in val_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    with open(train_path, "w") as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info(f"  Train split: {len(train_data)}, Val split: {len(val_data)}")
    return out_path


def finetune_llm(base_model_key: str = "qwen3.5-9b",
                 export_gguf: bool = False,
                 prepare_only: bool = False,
                 task_filter: str | None = None,
                 epochs: int | None = None,
                 resume: bool = False) -> str:
    """
    Fine-tune an LLM with QLoRA via unsloth.

    Args:
        base_model_key: Key from FINETUNE_CANDIDATES
        export_gguf: If True, export merged model to GGUF for Ollama
        prepare_only: If True, only prepare training data without training
        task_filter: If set, only include SFT examples for this task
        epochs: Override default epoch count
        resume: If True, resume training from the latest checkpoint
    """
    # Step 1: Prepare training data
    data_path = prepare_llm_training_data()
    if prepare_only:
        return str(data_path)

    # Step 2: Load model via unsloth (fully offline — use local HF cache)
    import os
    candidate = FINETUNE_CANDIDATES[base_model_key]

    # Reject multi-modal models (vision encoder consumes whole VRAM budget)
    if candidate.get("tier") == "skip":
        raise ValueError(
            f"{base_model_key} is not suitable for text-only SFT: {candidate.get('note')}"
        )

    hf_name = candidate["hf_name"]

    # Resolve local snapshot path to avoid any network check
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    cache_model_dir = os.path.join(hf_cache, "models--" + hf_name.replace("/", "--"), "snapshots")
    if os.path.isdir(cache_model_dir):
        snapshots = sorted(os.listdir(cache_model_dir))
        if snapshots:
            local_path = os.path.join(cache_model_dir, snapshots[-1])
            logger.info(f"Using local model path: {local_path}")
            hf_name = local_path
        else:
            raise FileNotFoundError(f"No snapshots found in {cache_model_dir}")
    else:
        raise FileNotFoundError(
            f"Model '{candidate['hf_name']}' not found in HF cache at {cache_model_dir}. "
            "Download it first with: huggingface-cli download " + candidate["hf_name"]
        )

    # Prevent any network calls during model load
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logger.info(f"Loading {base_model_key} via unsloth (4-bit, offline)...")
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_name,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    # Step 3: Apply LoRA
    logger.info(f"Applying LoRA (r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=TRAINING_CONFIG["seed"],
    )

    # Step 4: Prepare dataset
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Filter by task if requested
    if task_filter:
        dataset = dataset.filter(lambda x: x["task"] == task_filter)
        logger.info(f"Filtered to {len(dataset)} {task_filter} examples")

    logger.info(f"Loaded {len(dataset)} training examples")

    # Load validation set
    val_path = Path(data_path).parent / "sft_val.jsonl"
    val_dataset = None
    if val_path.exists():
        val_dataset = load_dataset("json", data_files=str(val_path), split="train")
        if task_filter:
            val_dataset = val_dataset.filter(lambda x: x["task"] == task_filter)
        logger.info(f"Loaded {len(val_dataset)} validation examples")

    # Format into chat template
    def format_chat(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
    if val_dataset is not None:
        val_dataset = val_dataset.map(format_chat, remove_columns=val_dataset.column_names)

    # Step 5: Train with SFTTrainer
    from trl import SFTTrainer
    from transformers import TrainingArguments

    num_epochs = epochs if epochs is not None else TRAINING_CONFIG["num_train_epochs"]
    output_dir = str(FINETUNE_DIR / f"{base_model_key}-scmetaintel-ft")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=num_epochs,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG.get("bf16", False),
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_strategy=TRAINING_CONFIG["save_strategy"],
        seed=TRAINING_CONFIG["seed"],
        report_to="none",
    )
    if val_dataset is not None:
        training_args.eval_strategy = "epoch"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    logger.info("Starting training...")
    t0 = time.time()
    resume_ckpt = None
    if resume:
        ckpt_dirs = sorted(Path(output_dir).glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[-1]))
        if ckpt_dirs:
            resume_ckpt = str(ckpt_dirs[-1])
            logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        else:
            logger.warning("No checkpoint found — starting fresh")
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.0f}s")
    logger.info(f"  Loss: {train_result.training_loss:.4f}")

    # Step 6: Save LoRA adapter
    lora_path = str(Path(output_dir) / "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"LoRA adapter saved to {lora_path}")

    # Step 7: Export to GGUF (optional)
    if export_gguf:
        logger.info("Exporting to GGUF for Ollama...")
        gguf_dir = str(Path(output_dir) / "gguf")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method="q4_k_m",
        )
        logger.info(f"GGUF exported to {gguf_dir}")

        # Step 7b: Create Ollama model
        import subprocess
        gguf_files = list(Path(gguf_dir).glob("*.gguf"))
        if gguf_files:
            gguf_path = gguf_files[0]
            modelfile_path = Path(output_dir) / "Modelfile"
            ollama_name = f"scmetaintel-{base_model_key}"
            with open(modelfile_path, "w") as mf:
                mf.write(f"FROM {gguf_path.resolve()}\n")
                mf.write("PARAMETER temperature 0.0\n")
                mf.write("PARAMETER num_ctx 4096\n")
            logger.info(f"Creating Ollama model: {ollama_name}")
            try:
                result = subprocess.run(
                    ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    logger.info(f"Ollama model created: {ollama_name}")
                else:
                    logger.warning(f"Ollama create failed: {result.stderr}")
            except FileNotFoundError:
                logger.warning("ollama CLI not found — create model manually with:")
                logger.warning(f"  ollama create {ollama_name} -f {modelfile_path}")

    # Save training metadata
    metadata = {
        "base_model": hf_name,
        "base_model_key": base_model_key,
        "lora_config": LORA_CONFIG,
        "training_config": TRAINING_CONFIG,
        "train_time_sec": round(train_time, 1),
        "train_loss": round(train_result.training_loss, 4),
        "n_examples": len(dataset),
        "output_dir": output_dir,
        "lora_path": lora_path,
    }
    with open(Path(output_dir) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["embeddings", "reranker", "llm", "all"],
                        default="all")
    parser.add_argument("--base-model", default="qwen3.5-9b",
                        choices=list(FINETUNE_CANDIDATES.keys()),
                        help="Base LLM for fine-tuning")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only prepare training data, skip training")
    parser.add_argument("--export-gguf", action="store_true",
                        help="Export fine-tuned model to GGUF for Ollama")
    parser.add_argument("--task-filter", default=None,
                        choices=["query_parsing", "metadata_extraction",
                                 "ontology_normalization", "answer_generation",
                                 "relevance_judgment"],
                        help="Only train on examples from this task")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override default epoch count")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    results = {}

    if args.task in ("embeddings", "all"):
        logger.info("\n=== 6.1 Embedding Fine-tuning ===")
        try:
            path = finetune_embeddings()
            results["embedding_ft"] = {"status": "done", "path": path}
        except Exception as e:
            logger.error(f"Embedding FT failed: {e}")
            results["embedding_ft"] = {"status": "error", "error": str(e)}

    if args.task in ("reranker", "all"):
        logger.info("\n=== 6.2 Reranker Fine-tuning ===")
        try:
            path = finetune_reranker()
            results["reranker_ft"] = {"status": "done", "path": path}
        except Exception as e:
            logger.error(f"Reranker FT failed: {e}")
            results["reranker_ft"] = {"status": "error", "error": str(e)}

    if args.task in ("llm", "all"):
        logger.info("\n=== 6.3 LLM Fine-tuning (QLoRA via unsloth) ===")
        try:
            path = finetune_llm(
                base_model_key=args.base_model,
                export_gguf=args.export_gguf,
                prepare_only=args.prepare_only,
                task_filter=args.task_filter,
                epochs=args.epochs,
                resume=args.resume,
            )
            results["llm_ft"] = {"status": "done", "path": path}
        except Exception as e:
            logger.error(f"LLM FT failed: {e}")
            results["llm_ft"] = {"status": "error", "error": str(e)}

    save_results(results, "finetune_bench")
    logger.info("Fine-tuning benchmark complete.")


if __name__ == "__main__":
    main()
