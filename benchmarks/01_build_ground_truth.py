#!/usr/bin/env python3
"""
Benchmark 01 — Build Ground Truth
==================================
Enrich all indexed GSE accessions with full metadata from GEO + PubMed.
Outputs one JSON per GSE into benchmarks/ground_truth/.

Usage:
    conda run -n dl python benchmarks/01_build_ground_truth.py
    conda run -n dl python benchmarks/01_build_ground_truth.py --limit 10  # test run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scmetaintel.enrich import enrich_all, load_gse_ids
from scmetaintel.config import GROUND_TRUTH_DIR, ACCESSION_INDEX


def main():
    parser = argparse.ArgumentParser(description="Build ground truth metadata")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of GSE to enrich (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip already-enriched GSE")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger("01_build_gt")

    out_dir = Path(args.output_dir) if args.output_dir else GROUND_TRUTH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    gse_ids = load_gse_ids()
    if args.limit > 0:
        gse_ids = gse_ids[:args.limit]

    logger.info(f"Enriching {len(gse_ids)} GSE accessions → {out_dir}")
    docs = enrich_all(output_dir=out_dir, skip_existing=args.skip_existing)

    # Summary
    enriched_count = len(list(out_dir.glob("GSE*.json")))
    logger.info(f"Done. {enriched_count} enriched files in {out_dir}")

    # Quick stats
    total_pubmed = sum(len(d.get("pubmed_ids", [])) for d in docs)
    total_tissues = sum(len(d.get("tissues", [])) for d in docs)
    total_diseases = sum(len(d.get("diseases", [])) for d in docs)
    logger.info(f"Stats: {total_pubmed} PubMed refs, "
                f"{total_tissues} tissue annotations, "
                f"{total_diseases} disease annotations")


if __name__ == "__main__":
    main()
