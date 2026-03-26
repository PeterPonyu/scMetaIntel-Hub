#!/usr/bin/env python3
"""
Data Augmentation Script — Expand Ground Truth from 1,357 to 2,000 Studies
==========================================================================
Searches GEO for new single-cell datasets filling coverage gaps, then
enriches each into the standard ground-truth JSON format.

Strategy:
  1. Disease-enriched studies (~250): Alzheimer's, Parkinson's, T2D, cancers, autoimmune...
  2. Underrepresented modalities (~150): spatial, CITE-seq, multiome, snRNA-seq
  3. Underrepresented tissues (~120): placenta, ovary, testis, muscle, thymus, stomach...
  4. Underrepresented organisms (~120): zebrafish, rat, NHP, Drosophila, pig, chicken

Each search term yields ~10-50 new GSE IDs. After deduplication against existing
ground truth, new IDs are enriched via NCBI GEO E-utilities and PubMed.

Usage:
    conda run -n dl python scripts/augment_ground_truth.py
    conda run -n dl python scripts/augment_ground_truth.py --search-only  # just discover, don't enrich
    conda run -n dl python scripts/augment_ground_truth.py --target 2000
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.config import GROUND_TRUTH_DIR
from scmetaintel.enrich import build_enriched_document

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("augment_gt")

# ---------------------------------------------------------------------------
# Search terms designed to fill coverage gaps
# ---------------------------------------------------------------------------

SEARCH_TERMS = [
    # =========================================================================
    # GAP 1: Autoimmune / Inflammatory diseases (15 studies → target 80+)
    # =========================================================================
    {"query": "single cell RNA-seq rheumatoid arthritis", "organism": "Homo sapiens", "max_results": 30, "category": "autoimmune"},
    {"query": "single cell RNA-seq lupus SLE systemic lupus", "organism": "Homo sapiens", "max_results": 30, "category": "autoimmune"},
    {"query": "single cell RNA-seq ulcerative colitis inflammatory bowel", "organism": "Homo sapiens", "max_results": 30, "category": "autoimmune"},
    {"query": "single cell RNA-seq Crohn disease", "organism": "Homo sapiens", "max_results": 25, "category": "autoimmune"},
    {"query": "single cell RNA-seq psoriasis dermatitis atopic", "organism": "Homo sapiens", "max_results": 25, "category": "autoimmune"},
    {"query": "single cell RNA-seq multiple sclerosis", "organism": "Homo sapiens", "max_results": 25, "category": "autoimmune"},
    {"query": "single cell RNA-seq asthma COPD", "organism": "Homo sapiens", "max_results": 25, "category": "autoimmune"},
    {"query": "single cell RNA-seq type 1 diabetes autoimmune", "organism": "Homo sapiens", "max_results": 20, "category": "autoimmune"},
    {"query": "single cell RNA-seq myasthenia gravis autoimmune", "organism": "", "max_results": 15, "category": "autoimmune"},
    {"query": "single cell RNA-seq scleroderma systemic sclerosis", "organism": "Homo sapiens", "max_results": 15, "category": "autoimmune"},

    # =========================================================================
    # GAP 2: Cardiovascular diseases (18 studies → target 60+)
    # =========================================================================
    {"query": "single cell RNA-seq heart failure", "organism": "Homo sapiens", "max_results": 30, "category": "cardiovascular"},
    {"query": "single cell RNA-seq myocardial infarction heart", "organism": "", "max_results": 30, "category": "cardiovascular"},
    {"query": "single cell RNA-seq atherosclerosis", "organism": "", "max_results": 25, "category": "cardiovascular"},
    {"query": "single cell RNA-seq cardiomyopathy dilated hypertrophic", "organism": "", "max_results": 25, "category": "cardiovascular"},
    {"query": "single cell RNA-seq aortic valve calcification", "organism": "", "max_results": 15, "category": "cardiovascular"},
    {"query": "single cell RNA-seq stroke cerebrovascular", "organism": "", "max_results": 15, "category": "cardiovascular"},
    {"query": "single cell RNA-seq pulmonary hypertension", "organism": "", "max_results": 15, "category": "cardiovascular"},

    # =========================================================================
    # GAP 3: Neurodegeneration (20 studies → target 60+)
    # =========================================================================
    {"query": "single cell RNA-seq Alzheimer disease", "organism": "Homo sapiens", "max_results": 40, "category": "neurodegeneration"},
    {"query": "single cell RNA-seq Alzheimer disease mouse model", "organism": "Mus musculus", "max_results": 30, "category": "neurodegeneration"},
    {"query": "single cell RNA-seq Parkinson disease", "organism": "Homo sapiens", "max_results": 30, "category": "neurodegeneration"},
    {"query": "single cell RNA-seq ALS amyotrophic lateral sclerosis", "organism": "", "max_results": 20, "category": "neurodegeneration"},
    {"query": "single cell RNA-seq Huntington disease neurodegeneration", "organism": "", "max_results": 15, "category": "neurodegeneration"},
    {"query": "single cell RNA-seq multiple system atrophy frontotemporal dementia", "organism": "", "max_results": 15, "category": "neurodegeneration"},

    # =========================================================================
    # GAP 4: Perturb-seq / functional genomics (15 studies → target 40+)
    # =========================================================================
    {"query": "Perturb-seq CRISPR single cell", "organism": "", "max_results": 40, "category": "modality"},
    {"query": "CROP-seq CRISPR screen single cell", "organism": "", "max_results": 30, "category": "modality"},
    {"query": "single cell CRISPR perturbation screen", "organism": "", "max_results": 30, "category": "modality"},
    {"query": "scVDJ-seq single cell VDJ immune repertoire", "organism": "", "max_results": 25, "category": "modality"},

    # =========================================================================
    # GAP 5: Non-model organisms (sparse coverage)
    # =========================================================================
    {"query": "single cell RNA-seq", "organism": "Sus scrofa", "max_results": 30, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Macaca mulatta", "max_results": 30, "category": "organism"},
    {"query": "single cell RNA-seq primate non-human", "organism": "Macaca fascicularis", "max_results": 20, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Gallus gallus", "max_results": 25, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Drosophila melanogaster", "max_results": 30, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Danio rerio", "max_results": 30, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Canis lupus familiaris", "max_results": 20, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Rattus norvegicus", "max_results": 25, "category": "organism"},
    {"query": "single cell RNA-seq", "organism": "Xenopus", "max_results": 15, "category": "organism"},

    # =========================================================================
    # EXISTING GAPS: Other diseases, tissues, modalities (maintain balance)
    # =========================================================================
    # Metabolic
    {"query": "single cell RNA-seq type 2 diabetes", "organism": "Homo sapiens", "max_results": 25, "category": "disease"},
    {"query": "single cell RNA-seq NAFLD NASH liver fibrosis", "organism": "Homo sapiens", "max_results": 20, "category": "disease"},
    {"query": "single cell RNA-seq obesity metabolic syndrome", "organism": "", "max_results": 15, "category": "disease"},
    # Infectious
    {"query": "single cell RNA-seq tuberculosis", "organism": "Homo sapiens", "max_results": 20, "category": "disease"},
    {"query": "single cell RNA-seq HIV", "organism": "Homo sapiens", "max_results": 20, "category": "disease"},
    {"query": "single cell RNA-seq malaria Plasmodium", "organism": "", "max_results": 15, "category": "disease"},
    # Underrepresented tissues
    {"query": "single cell RNA-seq placenta", "organism": "Homo sapiens", "max_results": 20, "category": "tissue"},
    {"query": "single cell RNA-seq testis spermatogenesis", "organism": "", "max_results": 20, "category": "tissue"},
    {"query": "single cell RNA-seq skeletal muscle", "organism": "", "max_results": 20, "category": "tissue"},
    {"query": "single cell RNA-seq thyroid adrenal endocrine", "organism": "", "max_results": 15, "category": "tissue"},
    {"query": "single cell RNA-seq adipose fat tissue", "organism": "", "max_results": 15, "category": "tissue"},
    {"query": "single cell RNA-seq lymph node", "organism": "Homo sapiens", "max_results": 20, "category": "tissue"},
    {"query": "single cell RNA-seq spinal cord", "organism": "", "max_results": 15, "category": "tissue"},
    # Modalities
    {"query": "spatial transcriptomics Visium MERFISH Xenium", "organism": "", "max_results": 40, "category": "modality"},
    {"query": "CITE-seq multimodal single cell protein RNA", "organism": "", "max_results": 30, "category": "modality"},
    {"query": "single cell multiome ATAC RNA 10x", "organism": "", "max_results": 30, "category": "modality"},
]


def load_existing_gse_ids() -> set:
    """Load all existing GSE IDs from ground truth directory."""
    existing = set()
    for p in GROUND_TRUTH_DIR.glob("GSE*.json"):
        gse_id = p.stem
        existing.add(gse_id)
    return existing


def search_geo_esearch(query: str, organism: str = "", max_results: int = 50) -> list:
    """Search GEO via NCBI E-utilities esearch + esummary, return GSE IDs."""
    import requests
    import xml.etree.ElementTree as ET

    # Build query
    terms = [f"({query})"]
    if organism:
        terms.append(f'"{organism}"[Organism]')
    full_query = " AND ".join(terms)

    # Step 1: esearch
    try:
        resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "gds",
                "term": full_query,
                "retmax": max_results,
                "retmode": "xml",
                "usehistory": "y",
            },
            timeout=30,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ids = [el.text for el in root.findall(".//Id") if el.text]
        count = root.findtext(".//Count", "0")
        time.sleep(0.35)
    except Exception as e:
        logger.warning(f"  esearch failed for '{full_query}': {e}")
        return []

    if not ids:
        return []

    # Step 2: esummary to get GSE accessions
    gse_ids = set()
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "gds",
                    "id": ",".join(batch),
                    "retmode": "xml",
                },
                timeout=30,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for doc in root.findall(".//DocSum"):
                # Try Accession field
                acc = doc.findtext(".//Item[@Name='Accession']", "")
                if acc.startswith("GSE"):
                    gse_ids.add(acc)
                    continue
                # Try entryType == GSE
                etype = doc.findtext(".//Item[@Name='entryType']", "")
                uid = doc.findtext("Id", "")
                if etype == "GSE" and uid:
                    gse_ids.add(f"GSE{uid}")
                    continue
                # Regex fallback
                text = ET.tostring(doc, encoding="unicode")
                for m in re.findall(r"GSE\d+", text):
                    gse_ids.add(m)
            time.sleep(0.35)
        except Exception as e:
            logger.warning(f"  esummary failed: {e}")

    return sorted(gse_ids)


def search_all_terms(existing: set, target: int) -> list:
    """Run all search terms, accumulate new GSE IDs until target is reached."""
    new_ids = []
    seen = set(existing)
    from collections import Counter
    category_counts = Counter()

    for i, term in enumerate(SEARCH_TERMS):
        if len(existing) + len(new_ids) >= target:
            logger.info(f"  Reached target {target}, stopping search.")
            break

        query = term["query"]
        organism = term.get("organism", "")
        max_results = term.get("max_results", 50)
        category = term.get("category", "unknown")

        logger.info(f"  [{i+1}/{len(SEARCH_TERMS)}] Searching: '{query}'"
                     f"{f' [{organism}]' if organism else ''} (max={max_results})")

        results = search_geo_esearch(query, organism, max_results)

        added = 0
        for gse_id in results:
            if gse_id not in seen:
                seen.add(gse_id)
                new_ids.append({"gse_id": gse_id, "category": category,
                               "search_query": query, "organism_filter": organism})
                category_counts[category] += 1
                added += 1

        remaining = target - len(existing) - len(new_ids)
        logger.info(f"    Found {len(results)} total, {added} new "
                     f"(total new: {len(new_ids)}, remaining: {remaining})")

    logger.info(f"\n  Search complete: {len(new_ids)} new GSE IDs discovered")
    logger.info(f"  By category: {dict(category_counts)}")
    return new_ids


def enrich_new_studies(new_ids: list, existing: set) -> dict:
    """Enrich new GSE IDs into ground truth JSON documents."""
    success = 0
    failed = 0
    skipped = 0
    errors = []

    for i, entry in enumerate(new_ids):
        gse_id = entry["gse_id"]
        out_path = GROUND_TRUTH_DIR / f"{gse_id}.json"

        if out_path.exists():
            skipped += 1
            continue

        logger.info(f"  [{i+1}/{len(new_ids)}] Enriching {gse_id} "
                     f"({entry['category']})...")

        try:
            doc = build_enriched_document(gse_id)
            if not doc or not doc.get("title"):
                logger.warning(f"    {gse_id}: empty or no title, skipping")
                failed += 1
                errors.append({"gse_id": gse_id, "error": "empty document"})
                continue

            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2, default=str)

            success += 1
            if success % 50 == 0:
                logger.info(f"    Progress: {success} enriched, {failed} failed, "
                             f"{skipped} skipped")

        except Exception as e:
            failed += 1
            errors.append({"gse_id": gse_id, "error": str(e)[:200]})
            logger.warning(f"    {gse_id} failed: {e}")
            time.sleep(1)  # Extra backoff on failure

    return {
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "errors": errors[:50],  # Limit error log
    }


def main():
    parser = argparse.ArgumentParser(
        description="Augment ground truth with gap-targeted GEO search + enrichment")
    parser.add_argument("--target", type=int, default=2500,
                        help="Target number of total studies")
    parser.add_argument("--search-only", action="store_true",
                        help="Only search, don't enrich")
    parser.add_argument("--enrich-from", type=str, default=None,
                        help="Path to search results JSON to enrich from")
    args = parser.parse_args()

    existing = load_existing_gse_ids()
    logger.info(f"Existing ground truth: {len(existing)} studies")
    logger.info(f"Target: {args.target}")
    needed = args.target - len(existing)

    if needed <= 0:
        logger.info(f"Already at or above target ({len(existing)} >= {args.target}). Done.")
        return

    logger.info(f"Need {needed} more studies.\n")

    # Phase 1: Search
    if args.enrich_from:
        logger.info(f"Loading search results from {args.enrich_from}")
        with open(args.enrich_from) as f:
            data = json.load(f)
        new_ids = data.get("new_gse_ids", data)
    else:
        logger.info("Phase 1: Searching GEO for new datasets...")
        new_ids = search_all_terms(existing, args.target)

        # Save search results
        search_out = GROUND_TRUTH_DIR.parent / "results" / "augmentation_search.json"
        search_out.parent.mkdir(parents=True, exist_ok=True)
        with open(search_out, "w") as f:
            json.dump({
                "existing_count": len(existing),
                "target": args.target,
                "new_gse_ids": new_ids,
                "total_new": len(new_ids),
            }, f, indent=2)
        logger.info(f"Search results saved to {search_out}")

    if args.search_only:
        logger.info("Search-only mode. Stopping before enrichment.")
        return

    # Phase 2: Enrich
    logger.info(f"\nPhase 2: Enriching {len(new_ids)} new studies...")
    result = enrich_new_studies(new_ids, existing)

    # Summary
    final_count = len(load_existing_gse_ids())
    logger.info(f"\n{'='*60}")
    logger.info(f"AUGMENTATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Previously: {len(existing)} studies")
    logger.info(f"  Enriched:   {result['success']} new")
    logger.info(f"  Failed:     {result['failed']}")
    logger.info(f"  Skipped:    {result['skipped']} (already existed)")
    logger.info(f"  Final:      {final_count} studies")
    logger.info(f"  Target:     {args.target}")

    # Save augmentation report
    report_path = GROUND_TRUTH_DIR.parent / "results" / "augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "previous_count": len(existing),
            "final_count": final_count,
            "target": args.target,
            **result,
        }, f, indent=2)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
