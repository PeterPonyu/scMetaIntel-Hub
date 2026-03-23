#!/usr/bin/env python3
"""
Scale the scMetaIntel text corpus from ~53 GSE to 500+ GSE.

Strategy:
  1. Enrich all 125 already-indexed GSE (skip existing).
  2. Discover new GSE via NCBI E-utilities esearch with diverse queries.
  3. Enrich newly discovered GSE.

All operations are text-only — no expression matrices are downloaded.

Usage:
    conda run -n dl python scripts/scale_corpus.py
    conda run -n dl python scripts/scale_corpus.py --discover-only   # just find GSE IDs
    conda run -n dl python scripts/scale_corpus.py --enrich-only     # just enrich known IDs
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.enrich import build_enriched_document, load_gse_ids
from scmetaintel.config import GROUND_TRUTH_DIR

logger = logging.getLogger("scale_corpus")

# ---------------------------------------------------------------------------
# Diverse search queries for GEO discovery
# ---------------------------------------------------------------------------

DISCOVERY_QUERIES = [
    # Cancer subtypes
    ("single cell RNA-seq lung cancer", 80),
    ("single cell RNA-seq breast cancer", 80),
    ("single cell RNA-seq colorectal cancer", 60),
    ("single cell RNA-seq pancreatic cancer", 50),
    ("single cell RNA-seq glioblastoma", 50),
    ("single cell RNA-seq liver cancer hepatocellular", 50),
    ("single cell RNA-seq melanoma", 40),
    ("single cell RNA-seq leukemia", 50),
    ("single cell RNA-seq ovarian cancer", 40),
    ("single cell RNA-seq prostate cancer", 40),
    # Organ / tissue atlas
    ("single cell RNA-seq human brain", 80),
    ("single cell RNA-seq human heart", 60),
    ("single cell RNA-seq human kidney", 60),
    ("single cell RNA-seq human liver", 60),
    ("single cell RNA-seq human lung", 60),
    ("single cell RNA-seq human intestine gut", 60),
    ("single cell RNA-seq human pancreas", 50),
    ("single cell RNA-seq human skin", 40),
    ("single cell RNA-seq human bone marrow", 50),
    ("single cell RNA-seq human retina", 40),
    # Immunology
    ("single cell RNA-seq T cell immune", 60),
    ("single cell RNA-seq tumor microenvironment", 60),
    ("single cell RNA-seq autoimmune disease", 40),
    ("single cell RNA-seq COVID-19 SARS-CoV-2", 60),
    # Development
    ("single cell RNA-seq embryo development human", 50),
    ("single cell RNA-seq fetal organogenesis", 40),
    ("single cell RNA-seq iPSC organoid", 50),
    # Neurodegeneration
    ("single cell RNA-seq Alzheimer disease", 40),
    ("single cell RNA-seq Parkinson disease", 30),
    # Technology / modality
    ("scATAC-seq human chromatin accessibility", 40),
    ("CITE-seq single cell protein", 40),
    ("spatial transcriptomics Visium human", 50),
    ("multiome single cell ATAC RNA", 40),
    # Model organisms (diversity)
    ("single cell RNA-seq mouse brain", 60),
    ("single cell RNA-seq mouse tumor", 50),
    ("single cell RNA-seq zebrafish", 30),
]


def search_geo_esearch(query: str, max_results: int = 50, retries: int = 3) -> list[str]:
    """Search GEO via NCBI E-utilities and return GSE accession IDs."""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Step 1: esearch to get GDS IDs
    for attempt in range(retries):
        try:
            full_query = f'({query}) AND "Expression profiling by high throughput sequencing"[DataSet Type]'
            params = {
                "db": "gds",
                "term": full_query,
                "retmax": max_results,
                "retmode": "json",
                "usehistory": "y",
            }
            resp = requests.get(f"{base_url}/esearch.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []
            break
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            logger.warning(f"esearch failed for '{query}': {e}")
            return []

    time.sleep(0.4)

    # Step 2: esummary to convert GDS IDs to GSE accessions
    gse_ids = set()
    batch_size = 50
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        try:
            params = {
                "db": "gds",
                "id": ",".join(batch),
                "retmode": "json",
            }
            resp = requests.get(f"{base_url}/esummary.fcgi", params=params, timeout=30)
            resp.raise_for_status()
            summary = resp.json().get("result", {})

            for uid in batch:
                entry = summary.get(uid, {})
                accession = entry.get("accession", "")
                # GDS records may reference GSE in the accession or FTPLink
                if accession.startswith("GSE"):
                    gse_ids.add(accession)
                elif accession.startswith("GDS"):
                    # Try to extract GSE from the FTPLink or GPL cross-refs
                    ftplink = entry.get("ftplink", "")
                    if "GSE" in ftplink:
                        import re
                        m = re.search(r"(GSE\d+)", ftplink)
                        if m:
                            gse_ids.add(m.group(1))
                    # Also check GPL and other fields for GSE references
                    gse_field = entry.get("gse", "")
                    if gse_field:
                        gse_ids.add(f"GSE{gse_field}" if not gse_field.startswith("GSE") else gse_field)

                    # Extract GSE from entrytype=GSE entries
                    entry_type = entry.get("entrytype", "")
                    if entry_type == "GSE":
                        gse_ids.add(accession)

            time.sleep(0.4)
        except Exception as e:
            logger.warning(f"esummary failed for batch: {e}")
            time.sleep(1)

    return sorted(gse_ids)


def discover_new_gse(existing_ids: set[str]) -> list[str]:
    """Run diverse searches and return undiscovered GSE IDs."""
    all_new = set()

    for query, max_results in DISCOVERY_QUERIES:
        logger.info(f"Searching: '{query}' (max {max_results})")
        found = search_geo_esearch(query, max_results=max_results)
        new = [g for g in found if g not in existing_ids and g not in all_new]
        all_new.update(new)
        logger.info(f"  Found {len(found)} GSE, {len(new)} new (total new: {len(all_new)})")
        time.sleep(0.5)

    return sorted(all_new)


def enrich_gse_list(gse_ids: list[str], output_dir: Path, skip_existing: bool = True) -> int:
    """Enrich a list of GSE IDs, writing ground truth JSONs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = 0
    failed = 0

    for i, gse_id in enumerate(gse_ids):
        outfile = output_dir / f"{gse_id}.json"
        if skip_existing and outfile.exists():
            continue

        try:
            doc = build_enriched_document(gse_id)
            with open(outfile, "w") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
            enriched += 1

            if enriched % 10 == 0:
                logger.info(f"  Progress: {enriched} enriched, {failed} failed, "
                            f"{i + 1}/{len(gse_ids)} processed")
        except Exception as e:
            failed += 1
            logger.warning(f"  Failed to enrich {gse_id}: {e}")

    logger.info(f"Enrichment complete: {enriched} new, {failed} failed")
    return enriched


def save_discovery_index(all_ids: list[str], output_path: Path):
    """Save the complete discovered GSE list for reference."""
    data = {
        "total_gse": len(all_ids),
        "gse_ids": all_ids,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved discovery index with {len(all_ids)} GSE to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scale scMetaIntel text corpus")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover new GSE IDs, do not enrich")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Only enrich known GSE IDs (from discovery index)")
    parser.add_argument("--output-dir", type=Path, default=GROUND_TRUTH_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_index_path = output_dir.parent / "discovery_index.json"

    # Load already-known GSE IDs
    existing_ids = set(load_gse_ids())
    existing_gt = set(p.stem for p in output_dir.glob("GSE*.json"))
    logger.info(f"Known GSE IDs: {len(existing_ids)} from accession index")
    logger.info(f"Existing ground truth: {len(existing_gt)} files")

    if not args.enrich_only:
        # Phase 1: Enrich all existing indexed GSE that aren't done yet
        unenriched = sorted(existing_ids - existing_gt)
        if unenriched:
            logger.info(f"\n=== Phase 1: Enriching {len(unenriched)} remaining indexed GSE ===")
            enrich_gse_list(unenriched, output_dir)

        # Phase 2: Discover new GSE
        logger.info(f"\n=== Phase 2: Discovering new GSE via NCBI search ===")
        all_known = existing_ids | existing_gt
        new_gse = discover_new_gse(all_known)
        logger.info(f"Discovered {len(new_gse)} new unique GSE IDs")

        # Save discovery index
        all_ids = sorted(all_known | set(new_gse))
        save_discovery_index(all_ids, discovery_index_path)

        if not args.discover_only and new_gse:
            # Phase 3: Enrich discovered GSE
            logger.info(f"\n=== Phase 3: Enriching {len(new_gse)} newly discovered GSE ===")
            enrich_gse_list(new_gse, output_dir)
    else:
        # Enrich-only mode: read discovery index
        if discovery_index_path.exists():
            with open(discovery_index_path) as f:
                idx = json.load(f)
            all_ids = idx.get("gse_ids", [])
            unenriched = [g for g in all_ids if g not in existing_gt]
            logger.info(f"Discovery index has {len(all_ids)} GSE, {len(unenriched)} not yet enriched")
            if unenriched:
                enrich_gse_list(unenriched, output_dir)
        else:
            logger.error(f"No discovery index at {discovery_index_path}. Run discovery first.")
            sys.exit(1)

    # Final stats
    final_count = len(list(output_dir.glob("GSE*.json")))
    total_size = sum(f.stat().st_size for f in output_dir.glob("GSE*.json"))
    logger.info(f"\n=== Final corpus: {final_count} GSE, {total_size / 1024:.1f} KB ===")


if __name__ == "__main__":
    main()
