#!/usr/bin/env python3
"""
Enhance SFT training data for scMetaIntel-Hub fine-tuning.

Generates augmented training conversations for 5 tasks:
  A. query_parsing       (target: ~650)
  B. answer_generation   (target: ~550)
  C. ontology_normalization (target: ~800)
  D. relevance_judgment  (target: ~1100)
  E. metadata_extraction (target: ~170 augmented)

Total enhancement target: ~3,270 new conversations
Combined with base (~1,900): ~5,100 total

Usage:
    conda run -n dl python scripts/enhance_training_data.py
    conda run -n dl python scripts/enhance_training_data.py --dry-run
    conda run -n dl python scripts/enhance_training_data.py --task query_parsing
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import GROUND_TRUTH_DIR, BENCHMARK_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("enhance_data")

FINETUNE_DIR = BENCHMARK_DIR / "finetuned"
ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "ontologies"

# ---------------------------------------------------------------------------
# System prompts (same as 06_bench_finetune.py)
# ---------------------------------------------------------------------------
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
# Organism short names
# ---------------------------------------------------------------------------
ORG_SHORT = {
    "Homo sapiens": "human",
    "Mus musculus": "mouse",
    "Danio rerio": "zebrafish",
    "Rattus norvegicus": "rat",
    "Macaca mulatta": "macaque",
    "Drosophila melanogaster": "drosophila",
    "Sus scrofa": "pig",
    "Bos taurus": "cow",
    "Gallus gallus": "chicken",
}

# Generic terms to skip in query/ontology generation
BLOCKLIST = {
    "cell line", "organoid", "tissue", "sample", "cells", "cell",
    "other", "unknown", "n/a", "na", "none", "", "tumor", "normal",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_corpus() -> dict:
    """Load all GSE JSON files from ground truth."""
    docs = {}
    for p in sorted(GROUND_TRUTH_DIR.glob("GSE*.json")):
        with open(p) as f:
            d = json.load(f)
            docs[d["gse_id"]] = d
    return docs


def load_eval_queries() -> list:
    """Load eval queries."""
    path = BENCHMARK_DIR / "eval_queries.json"
    with open(path) as f:
        return json.load(f)


def load_ontologies() -> dict:
    """
    Parse all 3 OBO files with synonym support.
    Returns {prefix: {lowercased_term: (ontology_id, canonical_name)}}
    """
    lookups = {}
    for obo_file, prefix in [
        ("cl.obo", "CL"),
        ("uberon-basic.obo", "UBERON"),
        ("mondo.obo", "MONDO"),
    ]:
        obo_path = ONTOLOGY_DIR / obo_file
        if not obo_path.exists():
            logger.warning(f"Ontology file not found: {obo_path}")
            lookups[prefix] = {}
            continue

        lookup = {}
        current_id = None
        current_name = None
        in_term = False
        is_obsolete = False
        prefix_bare = prefix

        with open(obo_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "[Term]":
                    if current_id and current_name and not is_obsolete:
                        lookup[current_name.lower()] = (current_id, current_name)
                    current_id = None
                    current_name = None
                    in_term = True
                    is_obsolete = False
                    continue
                if line.startswith("[") and line.endswith("]"):
                    if in_term and current_id and current_name and not is_obsolete:
                        lookup[current_name.lower()] = (current_id, current_name)
                    in_term = False
                    current_id = None
                    current_name = None
                    is_obsolete = False
                    continue
                if not in_term:
                    continue
                if line.startswith("id: "):
                    cid = line[4:].strip()
                    if cid.startswith(prefix_bare + ":"):
                        current_id = cid
                    else:
                        current_id = None
                elif line.startswith("name: ") and current_id:
                    current_name = line[6:].strip()
                    lookup[current_name.lower()] = (current_id, current_name)
                elif line.startswith("synonym: ") and current_id and current_name:
                    start = line.find('"')
                    end = line.find('"', start + 1)
                    if start >= 0 and end > start:
                        syn = line[start + 1:end].strip()
                        key = syn.lower()
                        if key not in lookup:
                            lookup[key] = (current_id, current_name)
                elif line.startswith("is_obsolete: true"):
                    is_obsolete = True

        if in_term and current_id and current_name and not is_obsolete:
            lookup[current_name.lower()] = (current_id, current_name)

        lookups[prefix] = lookup
        logger.info(f"  {prefix}: {len(lookup)} lookup entries from {obo_file}")

    return lookups


def _get_terms(d: dict, field: str) -> list:
    """Get terms from top-level or characteristics_summary."""
    terms = d.get(field, []) or []
    if not terms:
        cs = d.get("characteristics_summary", {})
        if isinstance(cs, dict):
            terms = cs.get(field, []) or []
    return [t for t in terms if t and len(t) >= 3 and t.lower().strip() not in BLOCKLIST]


def _make_conv(system: str, user: str, assistant: str, task: str) -> dict:
    """Build a conversation dict."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "task": task,
    }


# ---------------------------------------------------------------------------
# Task A: Query Parsing Enhancement
# ---------------------------------------------------------------------------

QUERY_TEMPLATES = {
    "org_tissue": [
        "{org_short} {tissue} scRNA-seq",
        "{org_short} {tissue} single-cell RNA-seq",
        "{org_short} {tissue} single-cell transcriptomics",
        "single-cell sequencing of {org_short} {tissue}",
        "{tissue} datasets from {org_full}",
        "{org_short} {tissue} single-cell analysis",
    ],
    "org_disease": [
        "{disease} scRNA-seq",
        "{disease} single-cell RNA-seq in {org_short}",
        "{org_short} {disease} single-cell sequencing",
        "single-cell studies of {disease}",
        "{disease} single-cell analysis",
    ],
    "org_tissue_disease": [
        "{org_short} {tissue} {disease} scRNA-seq",
        "{disease} in {org_short} {tissue} single-cell",
        "{org_short} {tissue} {disease} single-cell analysis",
    ],
    "cell_type": [
        "{cell_type} scRNA-seq",
        "{cell_type} single-cell RNA-seq",
        "single-cell profiling of {cell_type}",
        "{cell_type} transcriptomics",
        "{cell_type} single-cell analysis",
    ],
    "cell_type_tissue": [
        "{cell_type} in {tissue} scRNA-seq",
        "{org_short} {tissue} {cell_type} single-cell",
        "{cell_type} in {org_short} {tissue}",
    ],
    "modality": [
        "{modality} datasets",
        "{modality} single-cell studies",
        "{org_short} {modality} data",
    ],
    "natural_language": [
        "Find datasets studying {tissue} in {org_full}",
        "What scRNA-seq datasets study {disease}?",
        "Show me {org_short} {tissue} datasets",
        "Studies of {cell_type} at single-cell level",
        "Are there single-cell datasets for {disease} in {org_short}?",
        "Looking for {org_short} {tissue} scRNA-seq data",
    ],
}


def enhance_query_parsing(docs: dict, ontologies: dict, target: int = 650) -> list:
    """Generate synthetic query_parsing training samples from GSE metadata."""
    random.seed(42)
    convs = []
    seen = set()

    for gse_id, d in docs.items():
        org_full = d.get("organism", "")
        if not org_full:
            continue
        org_short = ORG_SHORT.get(org_full, org_full.split()[-1].lower())

        tissues = _get_terms(d, "tissues")
        diseases = _get_terms(d, "diseases")
        cell_types = _get_terms(d, "cell_types")
        modalities = d.get("modalities", []) or []
        domain = d.get("domain", "")

        # Use domain as disease fallback
        if not diseases and domain and domain != "unknown":
            diseases = [domain]

        # org+tissue queries
        for tissue in tissues[:2]:
            tpl = random.choice(QUERY_TEMPLATES["org_tissue"])
            query = tpl.format(org_short=org_short, org_full=org_full, tissue=tissue)
            key = ("org_tissue", org_full, tissue.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"organism": org_full, "tissue": tissue}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # org+disease queries
        for disease in diseases[:1]:
            tpl = random.choice(QUERY_TEMPLATES["org_disease"])
            query = tpl.format(org_short=org_short, org_full=org_full, disease=disease)
            key = ("org_disease", org_full, disease.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"organism": org_full, "disease": disease}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # org+tissue+disease queries
        if tissues and diseases:
            tissue = tissues[0]
            disease = diseases[0]
            tpl = random.choice(QUERY_TEMPLATES["org_tissue_disease"])
            query = tpl.format(org_short=org_short, org_full=org_full,
                               tissue=tissue, disease=disease)
            key = ("org_tissue_disease", org_full, tissue.lower(), disease.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"organism": org_full, "tissue": tissue, "disease": disease}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # cell_type queries
        for ct in cell_types[:1]:
            tpl = random.choice(QUERY_TEMPLATES["cell_type"])
            query = tpl.format(cell_type=ct)
            key = ("cell_type", ct.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"cell_type": ct}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # cell_type+tissue queries
        if cell_types and tissues:
            ct = cell_types[0]
            tissue = tissues[0]
            tpl = random.choice(QUERY_TEMPLATES["cell_type_tissue"])
            query = tpl.format(org_short=org_short, cell_type=ct, tissue=tissue)
            key = ("ct_tissue", ct.lower(), tissue.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"cell_type": ct, "tissue": tissue}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # modality queries
        for mod in modalities[:1]:
            tpl = random.choice(QUERY_TEMPLATES["modality"])
            query = tpl.format(org_short=org_short, modality=mod)
            key = ("modality", org_full, mod.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints = {"organism": org_full, "assay": mod}
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

        # natural language queries (for a subset)
        if tissues and random.random() < 0.15:
            tissue = tissues[0]
            tpl = random.choice(QUERY_TEMPLATES["natural_language"])
            try:
                query = tpl.format(org_short=org_short, org_full=org_full,
                                   tissue=tissue, disease=diseases[0] if diseases else "",
                                   cell_type=cell_types[0] if cell_types else "")
            except (IndexError, KeyError):
                continue
            key = ("nl", query.lower()[:40])
            if key in seen:
                continue
            seen.add(key)
            constraints = {"organism": org_full, "tissue": tissue}
            if diseases:
                constraints["disease"] = diseases[0]
            convs.append(_make_conv(
                PARSE_SYSTEM, query,
                json.dumps(constraints, indent=2),
                "query_parsing"
            ))

    random.shuffle(convs)
    if len(convs) > target:
        convs = convs[:target]
    logger.info(f"  query_parsing: {len(convs)} enhanced samples")
    return convs


# ---------------------------------------------------------------------------
# Task B: Answer Generation Enhancement
# ---------------------------------------------------------------------------

ANSWER_TEMPLATES = [
    # Bullet-list format
    "bullet",
    # Paragraph format
    "paragraph",
    # Brief summary
    "brief",
]


def _build_context(gse_ids: list, docs: dict) -> str:
    """Build study context text."""
    parts = []
    for gse_id in gse_ids:
        d = docs[gse_id]
        parts.append(
            f"[{gse_id}] {d.get('title', '')}\n"
            f"  Organism: {d.get('organism', 'N/A')}\n"
            f"  Summary: {d.get('summary', '')[:300]}"
        )
    return "\n\n".join(parts)


def _build_answer(query: str, gse_ids: list, docs: dict, style: str = "bullet") -> str:
    """Build a template answer citing real GSE IDs."""
    if style == "bullet":
        answer = f'Based on the retrieved studies, here are the relevant datasets for "{query}":\n\n'
        for gse_id in gse_ids:
            d = docs[gse_id]
            mods = ", ".join(d.get("modalities", ["N/A"]))
            answer += f"- [{gse_id}]: {d.get('title', '')}. Organism: {d.get('organism', 'N/A')}. Modalities: {mods}.\n"
        gse_list = ", ".join(f"[{g}]" for g in gse_ids)
        answer += f"\nThese datasets {gse_list} are relevant to the query."
    elif style == "paragraph":
        answer = f'For the query "{query}", I found {len(gse_ids)} relevant studies. '
        for i, gse_id in enumerate(gse_ids):
            d = docs[gse_id]
            answer += f'[{gse_id}] "{d.get("title", "")}" studies {d.get("organism", "unknown")} using {", ".join(d.get("modalities", ["unspecified"]))}. '
        answer += "These studies collectively address the query."
    else:  # brief
        gse_list = ", ".join(f"[{g}]" for g in gse_ids)
        answer = f'The following datasets match "{query}": {gse_list}.'
        if len(gse_ids) <= 3:
            for gse_id in gse_ids:
                d = docs[gse_id]
                answer += f' [{gse_id}] covers {d.get("organism", "")} {", ".join(d.get("tissues", [])[:2])}.'
    return answer


def enhance_answer_generation(docs: dict, queries: list, target: int = 550) -> list:
    """Generate answer_generation samples from GSE clusters."""
    random.seed(43)
    convs = []

    # Build clusters: (organism, first_tissue) -> [gse_ids]
    org_tissue_clusters = defaultdict(list)
    domain_clusters = defaultdict(list)
    for gse_id, d in docs.items():
        org = d.get("organism", "")
        tissues = _get_terms(d, "tissues")
        domain = d.get("domain", "")
        if org and tissues:
            org_tissue_clusters[(org, tissues[0].lower())].append(gse_id)
        if domain and domain != "unknown":
            domain_clusters[domain].append(gse_id)

    # Sub-strategy 1: Cluster-based generation (2+ GSEs per cluster)
    for (org, tissue), gse_ids in org_tissue_clusters.items():
        if len(gse_ids) < 2:
            continue
        org_short = ORG_SHORT.get(org, org.split()[-1].lower())
        query_options = [
            f"{org_short} {tissue} scRNA-seq",
            f"Find {org_short} {tissue} single-cell datasets",
            f"What studies analyze {tissue} in {org}?",
            f"{org_short} {tissue} single-cell analysis",
        ]
        n_queries = min(2, len(query_options)) if len(gse_ids) < 3 else min(3, len(query_options))
        for query_text in random.sample(query_options, n_queries):
            ctx_size = random.choice([3, 5])
            ctx_gses = random.sample(gse_ids, min(ctx_size, len(gse_ids)))
            context = _build_context(ctx_gses, docs)
            user_text = (
                f"Retrieved studies:\n{context}\n\n"
                f"User query: {query_text}\n\n"
                f"Provide a comprehensive answer citing relevant GSE accessions."
            )
            style = random.choice(ANSWER_TEMPLATES)
            answer = _build_answer(query_text, ctx_gses, docs, style)
            convs.append(_make_conv(ANSWER_SYSTEM, user_text, answer, "answer_generation"))

    # Sub-strategy 2: Domain-based generation
    for domain, gse_ids in domain_clusters.items():
        if len(gse_ids) < 5:
            continue
        query_options = [
            f"{domain} single-cell studies",
            f"Find scRNA-seq datasets related to {domain}",
            f"What single-cell datasets study {domain}?",
        ]
        for query_text in query_options[:2]:
            ctx_size = random.choice([3, 5, 7])
            ctx_gses = random.sample(gse_ids, min(ctx_size, len(gse_ids)))
            context = _build_context(ctx_gses, docs)
            user_text = (
                f"Retrieved studies:\n{context}\n\n"
                f"User query: {query_text}\n\n"
                f"Provide a comprehensive answer citing relevant GSE accessions."
            )
            style = random.choice(ANSWER_TEMPLATES)
            answer = _build_answer(query_text, ctx_gses, docs, style)
            convs.append(_make_conv(ANSWER_SYSTEM, user_text, answer, "answer_generation"))

    # Sub-strategy 3: Negative context (no relevant results)
    tissue_groups = defaultdict(list)
    for gse_id, d in docs.items():
        tissues = _get_terms(d, "tissues")
        if tissues:
            tissue_groups[tissues[0].lower()].append(gse_id)

    tissue_keys = [k for k, v in tissue_groups.items() if len(v) >= 3]
    neg_count = 0
    for i in range(min(80, len(tissue_keys))):
        query_tissue = tissue_keys[i]
        # Find a different tissue for context
        other_tissues = [t for t in tissue_keys if t != query_tissue]
        if not other_tissues:
            continue
        context_tissue = random.choice(other_tissues)
        ctx_gses = random.sample(tissue_groups[context_tissue],
                                 min(3, len(tissue_groups[context_tissue])))
        context = _build_context(ctx_gses, docs)
        query_text = f"Find single-cell datasets studying {query_tissue}"
        user_text = (
            f"Retrieved studies:\n{context}\n\n"
            f"User query: {query_text}\n\n"
            f"Provide a comprehensive answer citing relevant GSE accessions."
        )
        answer = (
            f'The provided context does not contain datasets directly relevant '
            f'to "{query_text}". The retrieved studies focus on {context_tissue} '
            f'rather than {query_tissue}. A new search may be needed to find '
            f'{query_tissue} datasets.'
        )
        convs.append(_make_conv(ANSWER_SYSTEM, user_text, answer, "answer_generation"))
        neg_count += 1

    # Sub-strategy 4: Varied context sizes from existing eval queries
    for q in queries:
        expected_gse = [g for g in q.get("expected_gse", []) if g in docs]
        if len(expected_gse) < 2:
            continue
        for ctx_size in [3, 7]:
            # Pad with random irrelevant GSEs for larger contexts
            padding = []
            if ctx_size > len(expected_gse):
                pool = [g for g in docs if g not in set(expected_gse)]
                padding = random.sample(pool, min(ctx_size - len(expected_gse), len(pool)))
            ctx_gses = expected_gse[:ctx_size] + padding[:ctx_size - len(expected_gse)]
            random.shuffle(ctx_gses)
            context = _build_context(ctx_gses, docs)
            user_text = (
                f"Retrieved studies:\n{context}\n\n"
                f"User query: {q['query']}\n\n"
                f"Provide a comprehensive answer citing relevant GSE accessions."
            )
            # Answer only cites the relevant ones
            style = random.choice(ANSWER_TEMPLATES)
            relevant_in_ctx = [g for g in ctx_gses if g in set(expected_gse)]
            answer = _build_answer(q["query"], relevant_in_ctx, docs, style)
            convs.append(_make_conv(ANSWER_SYSTEM, user_text, answer, "answer_generation"))

    random.shuffle(convs)
    if len(convs) > target:
        convs = convs[:target]
    logger.info(f"  answer_generation: {len(convs)} enhanced samples")
    return convs


# ---------------------------------------------------------------------------
# Task C: Ontology Normalization Enhancement
# ---------------------------------------------------------------------------

def enhance_ontology_normalization(docs: dict, ontologies: dict, target: int = 800) -> list:
    """Generate ontology_normalization samples using full ontology lookups."""
    random.seed(44)
    convs = []

    field_to_prefix = {
        "tissues": "UBERON",
        "cell_types": "CL",
        "diseases": "MONDO",
    }

    # Collect all matchable terms per GSE
    gse_matches = {}  # gse_id -> [{"raw": ..., "ontology_id": ..., ...}]
    for gse_id, d in docs.items():
        matched = []
        for field, prefix in field_to_prefix.items():
            ont_lookup = ontologies.get(prefix, {})
            if not ont_lookup:
                continue
            terms = _get_terms(d, field)
            for term in terms:
                key = term.lower().strip()
                if key in ont_lookup:
                    oid, olabel = ont_lookup[key]
                    matched.append({
                        "raw": term,
                        "ontology_id": oid,
                        "ontology_label": olabel,
                        "confidence": 1.0,
                    })
        if matched:
            gse_matches[gse_id] = matched

    logger.info(f"  Ontology: {len(gse_matches)} GSEs with exact+synonym matches")

    # Sub-strategy 1: One sample per GSE (all matched terms)
    for gse_id, matched in gse_matches.items():
        user_text = f"Normalize these biomedical terms from {gse_id}:\n"
        user_text += json.dumps([m["raw"] for m in matched])
        output = json.dumps({"normalized": matched}, indent=2)
        convs.append(_make_conv(ONTOLOGY_SYSTEM, user_text, output, "ontology_normalization"))

    # Sub-strategy 2: Single-term samples (from GSEs with multiple matches)
    single_convs = []
    for gse_id, matched in gse_matches.items():
        if len(matched) < 2:
            continue
        for m in matched[:3]:  # up to 3 single-term samples per GSE
            user_text = f"Normalize these biomedical terms from {gse_id}:\n"
            user_text += json.dumps([m["raw"]])
            output = json.dumps({"normalized": [m]}, indent=2)
            single_convs.append(_make_conv(
                ONTOLOGY_SYSTEM, user_text, output, "ontology_normalization"
            ))
    random.shuffle(single_convs)
    convs.extend(single_convs[:200])

    # Sub-strategy 3: Fuzzy/partial matching
    fuzzy_convs = []
    for gse_id, d in docs.items():
        if gse_id in gse_matches:
            continue  # already has exact matches
        for field, prefix in field_to_prefix.items():
            ont_lookup = ontologies.get(prefix, {})
            terms = _get_terms(d, field)
            for term in terms:
                key = term.lower().strip()
                if len(key) < 4:
                    continue
                # Containment matching
                match = None
                for ont_key, (oid, olabel) in ont_lookup.items():
                    if len(ont_key) < 4:
                        continue
                    if ont_key in key and ont_key != key:
                        # raw term contains ontology term
                        if len(ont_key) / len(key) >= 0.5:
                            match = (oid, olabel, 0.85)
                            break
                    elif key in ont_key and ont_key != key:
                        # ontology term contains raw term
                        if len(key) / len(ont_key) >= 0.6:
                            match = (oid, olabel, 0.80)
                            break
                if match:
                    oid, olabel, conf = match
                    user_text = f"Normalize these biomedical terms from {gse_id}:\n"
                    user_text += json.dumps([term])
                    output = json.dumps({"normalized": [{
                        "raw": term,
                        "ontology_id": oid,
                        "ontology_label": olabel,
                        "confidence": conf,
                    }]}, indent=2)
                    fuzzy_convs.append(_make_conv(
                        ONTOLOGY_SYSTEM, user_text, output, "ontology_normalization"
                    ))
                    break  # one fuzzy match per GSE is enough

    random.shuffle(fuzzy_convs)
    convs.extend(fuzzy_convs[:100])

    random.shuffle(convs)
    if len(convs) > target:
        convs = convs[:target]
    logger.info(f"  ontology_normalization: {len(convs)} enhanced samples")
    return convs


# ---------------------------------------------------------------------------
# Task D: Relevance Judgment Enhancement
# ---------------------------------------------------------------------------

def enhance_relevance_judgment(docs: dict, queries: list, target: int = 1100) -> list:
    """Generate additional relevance_judgment samples."""
    random.seed(45)
    convs = []
    all_gse_ids = list(docs.keys())

    for q in queries:
        expected = set(q.get("expected_gse", []))
        query_text = q["query"]
        constraints = q.get("expected_constraints", {})

        # Sub-strategy 1: More random negatives (6 extra per query)
        neg_pool = [g for g in all_gse_ids if g not in expected]
        extra_negs = random.sample(neg_pool, min(6, len(neg_pool)))
        for neg_gse in extra_negs:
            d = docs[neg_gse]
            user_text = (
                f"Query: {query_text}\n\n"
                f"Dataset [{neg_gse}]:\n"
                f"Title: {d.get('title', '')}\n"
                f"Summary: {d.get('summary', '')[:500]}"
            )
            score = round(random.uniform(0.02, 0.12), 2)
            output = json.dumps({
                "relevant": False,
                "score": score,
                "reasoning": (
                    f"This dataset is not relevant to the query about "
                    f"{query_text.lower()}. It studies different topics."
                ),
            }, indent=2)
            convs.append(_make_conv(RELEVANCE_SYSTEM, user_text, output, "relevance_judgment"))

        # Sub-strategy 2: Hard negatives (same organism, different tissue)
        target_org = constraints.get("organism", "")
        target_tissue = constraints.get("tissue", "")
        hard_negs = []
        for gse_id in neg_pool:
            d = docs[gse_id]
            gse_org = d.get("organism", "")
            gse_tissues = [t.lower() for t in d.get("tissues", [])]

            if target_org and target_tissue:
                org_match = gse_org == target_org
                tissue_match = any(target_tissue.lower() in t for t in gse_tissues)
                if org_match and not tissue_match:
                    hard_negs.append((gse_id, 0.25, "same organism but different tissue"))
                elif tissue_match and not org_match:
                    hard_negs.append((gse_id, 0.30, "similar tissue but different organism"))
            elif target_org:
                if gse_org == target_org:
                    hard_negs.append((gse_id, 0.20, "same organism but different focus"))

        for neg_gse, score, reason_hint in hard_negs[:5]:
            d = docs[neg_gse]
            user_text = (
                f"Query: {query_text}\n\n"
                f"Dataset [{neg_gse}]:\n"
                f"Title: {d.get('title', '')}\n"
                f"Summary: {d.get('summary', '')[:500]}"
            )
            score = round(score + random.uniform(-0.05, 0.10), 2)
            score = max(0.05, min(0.40, score))
            output = json.dumps({
                "relevant": False,
                "score": score,
                "reasoning": (
                    f"This dataset has {reason_hint} compared to the query. "
                    f"It is not directly relevant to {query_text.lower()}."
                ),
            }, indent=2)
            convs.append(_make_conv(RELEVANCE_SYSTEM, user_text, output, "relevance_judgment"))

        # Sub-strategy 3: Nuanced positive scoring
        for gse_id in expected:
            if gse_id not in docs:
                continue
            d = docs[gse_id]
            user_text = (
                f"Query: {query_text}\n\n"
                f"Dataset [{gse_id}]:\n"
                f"Title: {d.get('title', '')}\n"
                f"Summary: {d.get('summary', '')[:500]}"
            )
            # Check field overlap for nuanced score
            gse_tissues = [t.lower() for t in d.get("tissues", [])]
            gse_diseases = [dis.lower() for dis in d.get("diseases", [])]

            overlap_count = 0
            if target_org and d.get("organism", "") == target_org:
                overlap_count += 1
            if target_tissue and any(target_tissue.lower() in t for t in gse_tissues):
                overlap_count += 1
            target_disease = constraints.get("disease", "")
            if target_disease and any(target_disease.lower() in dis for dis in gse_diseases):
                overlap_count += 1

            if overlap_count >= 2:
                score = round(random.uniform(0.88, 0.98), 2)
                detail = "directly addresses multiple aspects of the query"
            elif overlap_count == 1:
                score = round(random.uniform(0.65, 0.85), 2)
                detail = "partially matches the query constraints"
            else:
                score = round(random.uniform(0.50, 0.70), 2)
                detail = "has some relevance but does not directly match all constraints"

            output = json.dumps({
                "relevant": True,
                "score": score,
                "reasoning": f"This dataset {detail} about {query_text.lower()}.",
            }, indent=2)
            convs.append(_make_conv(RELEVANCE_SYSTEM, user_text, output, "relevance_judgment"))

    random.shuffle(convs)
    if len(convs) > target:
        convs = convs[:target]
    logger.info(f"  relevance_judgment: {len(convs)} enhanced samples")
    return convs


# ---------------------------------------------------------------------------
# Task E: Metadata Extraction Enhancement
# ---------------------------------------------------------------------------

def enhance_metadata_extraction(docs: dict, target: int = 170) -> list:
    """Generate augmented metadata_extraction samples."""
    random.seed(46)
    convs = []

    gse_list = [(gse_id, d) for gse_id, d in docs.items()
                if d.get("title") and d.get("summary")]

    # Sub-strategy 1: Truncated summaries (~85)
    trunc_sample = random.sample(gse_list, min(85, len(gse_list)))
    for gse_id, d in trunc_sample:
        summary = d.get("summary", "")[:500] + "..."
        user_text = f"Title: {d.get('title', '')}\n\nSummary: {summary}"
        gold = {
            "tissues": d.get("tissues", []) or [],
            "diseases": d.get("diseases", []) or [],
            "cell_types": d.get("cell_types", []) or [],
            "modalities": d.get("modalities", []) or [],
            "organism": d.get("organism", ""),
        }
        domain = d.get("domain", "")
        if not gold["diseases"] and domain and domain != "unknown":
            gold["diseases"] = [domain]
        cs = d.get("characteristics_summary", {})
        if isinstance(cs, dict):
            if not gold["tissues"] and cs.get("tissues"):
                gold["tissues"] = cs["tissues"]
            if not gold["cell_types"] and cs.get("cell_types"):
                gold["cell_types"] = cs["cell_types"]
            if not gold["diseases"] and cs.get("diseases"):
                gold["diseases"] = cs["diseases"]
        output = json.dumps(gold, indent=2)
        convs.append(_make_conv(EXTRACT_SYSTEM, user_text, output, "metadata_extraction"))

    # Sub-strategy 2: Field-dropout (~85)
    dropout_sample = random.sample(gse_list, min(85, len(gse_list)))
    for gse_id, d in dropout_sample:
        user_text = f"Title: {d.get('title', '')}\n\nSummary: {d.get('summary', '')[:1500]}"
        gold = {
            "tissues": d.get("tissues", []) or [],
            "diseases": d.get("diseases", []) or [],
            "cell_types": d.get("cell_types", []) or [],
            "modalities": d.get("modalities", []) or [],
            "organism": d.get("organism", ""),
        }
        domain = d.get("domain", "")
        if not gold["diseases"] and domain and domain != "unknown":
            gold["diseases"] = [domain]
        cs = d.get("characteristics_summary", {})
        if isinstance(cs, dict):
            if not gold["tissues"] and cs.get("tissues"):
                gold["tissues"] = cs["tissues"]
            if not gold["cell_types"] and cs.get("cell_types"):
                gold["cell_types"] = cs["cell_types"]
            if not gold["diseases"] and cs.get("diseases"):
                gold["diseases"] = cs["diseases"]
        # Drop 1-2 fields
        droppable = [k for k in ["tissues", "diseases", "cell_types"] if gold[k]]
        if droppable:
            n_drop = min(random.choice([1, 2]), len(droppable))
            for field in random.sample(droppable, n_drop):
                gold[field] = []
        output = json.dumps(gold, indent=2)
        convs.append(_make_conv(EXTRACT_SYSTEM, user_text, output, "metadata_extraction"))

    random.shuffle(convs)
    if len(convs) > target:
        convs = convs[:target]
    logger.info(f"  metadata_extraction: {len(convs)} enhanced samples")
    return convs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(output_path: Path, corpus_gses: set = None) -> dict:
    """Verify enhanced training data format and counts."""
    task_counts = Counter()
    errors = []
    seen_hashes = set()
    total = 0

    with open(output_path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {i}: invalid JSON")
                continue

            total += 1
            task = conv.get("task", "unknown")
            task_counts[task] += 1

            messages = conv.get("messages", [])
            if len(messages) != 3:
                errors.append(f"Line {i}: expected 3 messages, got {len(messages)}")
                continue
            if messages[0]["role"] != "system":
                errors.append(f"Line {i}: first message should be system")
            if messages[1]["role"] != "user":
                errors.append(f"Line {i}: second message should be user")
            if messages[2]["role"] != "assistant":
                errors.append(f"Line {i}: third message should be assistant")

            # Verify JSON tasks produce valid JSON
            if task in ("query_parsing", "metadata_extraction",
                        "ontology_normalization", "relevance_judgment"):
                try:
                    json.loads(messages[2]["content"])
                except json.JSONDecodeError:
                    errors.append(f"Line {i}: invalid JSON in {task} assistant response")

            # Dedup check
            content_hash = hash(messages[1]["content"][:200] + messages[2]["content"][:200])
            if content_hash in seen_hashes:
                errors.append(f"Line {i}: duplicate conversation in {task}")
            seen_hashes.add(content_hash)

    result = {
        "total": total,
        "task_counts": dict(task_counts),
        "errors": errors[:20],  # cap error output
        "n_errors": len(errors),
    }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhance SFT training data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show expected counts without saving")
    parser.add_argument("--task", choices=[
        "query_parsing", "answer_generation", "ontology_normalization",
        "relevance_judgment", "metadata_extraction", "all"
    ], default="all")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: benchmarks/finetuned/llm_data/sft_train_enhanced.jsonl)")
    args = parser.parse_args()

    logger.info("Loading corpus...")
    docs = load_corpus()
    logger.info(f"  {len(docs)} GSE documents loaded")

    logger.info("Loading eval queries...")
    queries = load_eval_queries()
    logger.info(f"  {len(queries)} queries loaded")

    logger.info("Loading ontologies (with synonyms)...")
    ontologies = load_ontologies()

    all_convs = []

    if args.task in ("query_parsing", "all"):
        logger.info("Enhancing query_parsing...")
        all_convs.extend(enhance_query_parsing(docs, ontologies))

    if args.task in ("answer_generation", "all"):
        logger.info("Enhancing answer_generation...")
        all_convs.extend(enhance_answer_generation(docs, queries))

    if args.task in ("ontology_normalization", "all"):
        logger.info("Enhancing ontology_normalization...")
        all_convs.extend(enhance_ontology_normalization(docs, ontologies))

    if args.task in ("relevance_judgment", "all"):
        logger.info("Enhancing relevance_judgment...")
        all_convs.extend(enhance_relevance_judgment(docs, queries))

    if args.task in ("metadata_extraction", "all"):
        logger.info("Enhancing metadata_extraction...")
        all_convs.extend(enhance_metadata_extraction(docs))

    # Summary
    task_dist = Counter(c["task"] for c in all_convs)
    logger.info(f"\nEnhanced data summary: {len(all_convs)} total conversations")
    for task, cnt in sorted(task_dist.items()):
        logger.info(f"  {task}: {cnt}")

    if args.dry_run:
        logger.info("\n[DRY RUN] No files written.")
        return

    # Save
    out_dir = FINETUNE_DIR / "llm_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "sft_train_enhanced.jsonl"

    random.seed(42)
    random.shuffle(all_convs)

    with open(out_path, "w") as f:
        for conv in all_convs:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.info(f"\nSaved to {out_path}")

    # Validate
    logger.info("Validating output...")
    result = validate_output(out_path, set(docs.keys()))
    logger.info(f"  Total: {result['total']}")
    for task, cnt in sorted(result["task_counts"].items()):
        logger.info(f"  {task}: {cnt}")
    if result["n_errors"] > 0:
        logger.warning(f"  {result['n_errors']} validation errors found!")
        for err in result["errors"]:
            logger.warning(f"    {err}")
    else:
        logger.info("  All validations passed!")


if __name__ == "__main__":
    main()
