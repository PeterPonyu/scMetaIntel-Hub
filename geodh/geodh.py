#!/usr/bin/env python3
"""
GEO-DataHub CLI (geodh)
========================
Comprehensive command-line interface for searching, filtering, verifying,
downloading, and converting single-cell datasets from NCBI GEO.

USAGE:
  python geodh.py search  [options]    Search GEO and display results
  python geodh.py scan    [options]    Scan GSE files, classify formats/modality/domain
  python geodh.py verify  [options]    Verify download links (no actual download)
  python geodh.py download [options]   Download selected files
  python geodh.py convert [options]    Convert downloaded files to h5ad
  python geodh.py run     [options]    Full pipeline: search → filter → download → convert

EXAMPLES:
  # Search for human lung cancer scRNA-seq, show what formats are available
  python geodh.py search --query "lung cancer scRNA-seq" --organism human --max-results 10

  # Scan specific GSEs to see their file types and modalities
  python geodh.py scan --gse GSE174367 GSE200997 GSE185224

  # Only find datasets that provide filtered_feature_bc_matrix.h5 (RNA)
  python geodh.py scan --gse GSE174367 GSE185224 --format-filter filtered_feature_h5

  # Search for cancer scATAC datasets with filtered peak matrix h5
  python geodh.py search --query "scATAC cancer" --domain cancer --modality atac --format-filter filtered_peak_h5

  # Verify download links without actually downloading
  python geodh.py verify --gse GSE174367 --format-filter filtered_peak_h5 filtered_feature_h5

  # Download only filtered_feature_bc_matrix.h5 files from specific GSEs
  python geodh.py download --gse GSE185224 --format-filter filtered_feature_h5

  # Full pipeline: search → filter → download → convert
  python geodh.py run --query "single cell RNA-seq breast cancer" --organism human \\
      --domain cancer --modality rna --format-filter filtered_feature_h5 --max-results 5

Author: GEO-DataHub Pipeline
"""

import os
import sys
import json
import logging
import argparse
import time
import re
from datetime import datetime
from typing import List, Optional
import shutil

import yaml

from .geo_search import (
    GEOSeriesInfo,
    search_geo,
    search_geo_direct,
    save_search_results,
    load_search_results,
    fetch_supplementary_file_list,
    fetch_sample_supplementary_urls,
    _fetch_gse_metadata,
    RATE_LIMIT_DELAY,
)
from .geo_classifier import (
    classify_file,
    classify_files,
    classify_dataset_files,
    classify_domain,
    classify_modality_from_metadata,
    filter_files_by_format,
    filter_files_by_modality,
    print_classification_report,
    get_available_format_filters,
    DatasetClassification,
    FORMAT_FILTER_MAP,
)
from .geo_verify import verify_links, print_verification_report
from .geo_download import download_file, download_all
from .geo_convert import convert_all
from .geo_manifest import (
    scan_existing_gse_locations,
    should_skip_gse_download,
    write_dataset_meta_json,
    select_best_h5ad_by_gse,
)
from .geo_theme import organize_downloads_by_theme
from .geo_cancer_gap import scan_cancer_organ_gaps


logger = logging.getLogger("geodh")


# ============================================================
# Logging
# ============================================================
def setup_logging(level: str = "INFO", log_dir: str = "./logs"):
    """Setup logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"geodh_{timestamp}.log")
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level, logging.INFO))
    ch.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    
    root.addHandler(fh)
    root.addHandler(ch)
    
    return log_file


# ============================================================
# Core: Enrich GSE with metadata + classification
# ============================================================

def _refresh_dataset_flags(dc: DatasetClassification, extra_clfs) -> DatasetClassification:
    """Update DatasetClassification boolean flags after appending extra file classifications."""
    for clf in extra_clfs:
        fmt = clf.format_type
        if fmt == "h5_10x_filtered_feature":
            dc.has_h5_10x_filtered_feature = True
        elif fmt == "h5_10x_raw_feature":
            dc.has_h5_10x_raw_feature = True
        elif fmt == "h5_10x_filtered_peak":
            dc.has_h5_10x_filtered_peak = True
        elif fmt == "h5_10x_raw_peak":
            dc.has_h5_10x_raw_peak = True
        elif fmt == "h5ad":
            dc.has_h5ad = True
        elif fmt == "h5_generic":
            pass  # could be anything
        elif fmt in ("mtx_matrix", "mtx_barcodes", "mtx_features"):
            dc.has_mtx_triplet = True
        elif fmt == "fragments":
            dc.has_fragments = True
        elif fmt == "loom":
            dc.has_loom = True
        elif fmt in ("csv", "tsv"):
            dc.has_csv_tsv = True
        elif fmt == "rds":
            dc.has_rds = True
        if clf.modality_signal != "unknown":
            dc.modalities.add(clf.modality_signal)
    return dc


def enrich_gse_list(gse_list: List[GEOSeriesInfo]) -> List[DatasetClassification]:
    """
    Enrich a list of GEOSeriesInfo with full classification:
    file format detection, modality detection, biological domain classification.
    
    Fetches metadata from GEO if not already present.
    """
    results = []
    
    for gse_info in gse_list:
        # Fetch metadata if needed
        if not gse_info.summary and not gse_info.title:
            meta = _fetch_gse_metadata(gse_info.gse_id)
            gse_info.title = meta["title"]
            gse_info.summary = meta["summary"]
            gse_info.overall_design = meta.get("overall_design", "")
            gse_info.organism = meta["organism"]
            gse_info.series_type = meta.get("series_type", "")
            time.sleep(RATE_LIMIT_DELAY)
        
        # Fetch supplementary files if needed
        if not gse_info.supplementary_files:
            gse_info.supplementary_files = fetch_supplementary_file_list(gse_info.gse_id)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Classify files
        dc = classify_dataset_files(gse_info.gse_id, gse_info.supplementary_files)
        
        # If the series-level files have no convertible matrix (e.g. only
        # a _RAW.tar + filelist.txt + maybe .rds), probe sample-level
        # supplementary files from GEO to determine what's inside the tar.
        if not dc.has_convertible_matrix() and dc.has_tar:
            sample_urls = fetch_sample_supplementary_urls(gse_info.gse_id)
            time.sleep(RATE_LIMIT_DELAY)
            if sample_urls:
                from .geo_classifier import classify_files as _clf_files
                sample_clfs = _clf_files(sample_urls)
                dc.file_classifications.extend(sample_clfs)
                # Re-derive format flags from the extended list
                dc = _refresh_dataset_flags(dc, sample_clfs)
                logger.info(
                    "%s: no convertible matrix at series level → probed %d "
                    "sample files, detected formats: %s",
                    gse_info.gse_id, len(sample_urls),
                    {c.format_type for c in sample_clfs},
                )
        
        # Classify modality from metadata
        meta_modalities = classify_modality_from_metadata(
            gse_info.title, gse_info.summary,
            gse_info.series_type, gse_info.overall_design,
        )
        dc.modalities.update(meta_modalities)
        
        # Classify biological domain
        dc.domain, dc.domain_keywords_matched = classify_domain(
            gse_info.title, gse_info.summary, gse_info.overall_design,
        )
        
        dc.organism = gse_info.organism
        dc.series_type = gse_info.series_type
        
        # Attach original GSE info
        dc._gse_info = gse_info  # internal reference
        
        results.append(dc)
    
    return results


def apply_filters(
    enriched: List[DatasetClassification],
    domain_filter: Optional[str] = None,
    modality_filter: Optional[str] = None,
    format_filters: Optional[List[str]] = None,
    organism_filter: Optional[str] = None,
    require_format: bool = False,
    require_convertible: bool = False,
) -> List[DatasetClassification]:
    """
    Apply dataset-level and file-level filters.
    
    Args:
        enriched: List of DatasetClassification objects
        domain_filter: 'cancer', 'development', 'normal', 'disease', or None
        modality_filter: 'rna', 'atac', 'multiome', 'cite', or None
        format_filters: List of format filter names, e.g. ['filtered_feature_h5']
        organism_filter: Organism name substring, e.g. 'human', 'mouse'
        require_format: If True, exclude datasets that have no matching files
        require_convertible: If True, exclude datasets whose supplementary files
            contain no format convertible to h5ad (e.g. only .rds, .bw, fragments)
    
    Returns:
        Filtered list of DatasetClassification objects
    """
    result = enriched
    
    # Convertible-matrix filter (early rejection)
    if require_convertible:
        before = len(result)
        rejected = [dc for dc in result if not dc.has_convertible_matrix()]
        result = [dc for dc in result if dc.has_convertible_matrix()]
        if rejected:
            for dc in rejected:
                fmts = {clf.format_type for clf in dc.file_classifications}
                logger.warning(
                    "Filtered out %s: no convertible matrix format "
                    "(available formats: %s)", dc.gse_id, ", ".join(sorted(fmts))
                )
            logger.info(
                "Convertible-matrix filter: %d → %d datasets (dropped %d)",
                before, len(result), len(rejected),
            )
    
    # Domain filter
    if domain_filter:
        domain_filter = domain_filter.lower()
        result = [dc for dc in result if dc.domain == domain_filter]
        logger.info(f"Domain filter '{domain_filter}': {len(result)} datasets remaining")
    
    # Modality filter (dataset-level)
    if modality_filter:
        modality_filter = modality_filter.lower()
        if modality_filter == "multiome":
            # Must have both RNA and ATAC
            result = [dc for dc in result if "multiome" in dc.modalities or
                      ("rna" in dc.modalities and "atac" in dc.modalities)]
        else:
            result = [dc for dc in result if modality_filter in dc.modalities]
        logger.info(f"Modality filter '{modality_filter}': {len(result)} datasets remaining")
    
    # Organism filter
    if organism_filter:
        organism_filter = organism_filter.lower()
        # Map common names
        organism_map = {
            "human": "homo sapiens", "mouse": "mus musculus",
            "rat": "rattus norvegicus", "zebrafish": "danio rerio",
        }
        organism_val = organism_map.get(organism_filter, organism_filter)
        result = [dc for dc in result if organism_val in dc.organism.lower()]
        logger.info(f"Organism filter '{organism_filter}': {len(result)} datasets remaining")
    
    # Format filter (file-level)
    if format_filters:
        for dc in result:
            filtered = filter_files_by_format(dc.file_classifications, format_filters)
            dc._filtered_files = filtered
        
        if require_format:
            result = [dc for dc in result if hasattr(dc, '_filtered_files') and dc._filtered_files]
            logger.info(f"Format filter {format_filters} (require_format): {len(result)} datasets remaining")
    else:
        for dc in result:
            dc._filtered_files = dc.file_classifications
    
    return result


# ============================================================
# CLI Commands
# ============================================================

def cmd_search(args):
    """Search GEO and display enriched results with classification."""
    logger.info("=" * 60)
    logger.info("COMMAND: search")
    logger.info("=" * 60)
    
    # Build search config
    if args.gse:
        gse_list = search_geo_direct(args.gse)
    else:
        search_config = {
            "query": args.query or "single cell RNA-seq",
            "organism": _resolve_organism(args.organism),
            "dataset_type": args.dataset_type or "",
            "max_results": args.max_results,
            "min_samples": args.min_samples,
        }
        gse_list = search_geo(search_config)
    
    if not gse_list:
        print("\nNo datasets found.")
        return
    
    # Enrich with classification
    enriched = enrich_gse_list(gse_list)
    
    # Apply filters
    filtered = apply_filters(
        enriched,
        domain_filter=args.domain,
        modality_filter=args.modality,
        format_filters=args.format_filter,
        organism_filter=args.organism,
        require_format=args.require_format,
    )
    
    # Display results
    _print_enriched_results(filtered, show_files=args.show_files)
    
    # Save results
    _save_enriched_results(gse_list, filtered, args.output)
    
    return filtered


def cmd_scan(args):
    """Scan specific GSEs: show file classification, modality, domain."""
    logger.info("=" * 60)
    logger.info("COMMAND: scan")
    logger.info("=" * 60)
    
    if not args.gse:
        print("Error: --gse is required for scan command.")
        sys.exit(1)
    
    gse_list = search_geo_direct(args.gse)
    
    # Enrich
    enriched = enrich_gse_list(gse_list)
    
    # Apply filters  
    filtered = apply_filters(
        enriched,
        domain_filter=args.domain,
        modality_filter=args.modality,
        format_filters=args.format_filter,
        require_format=args.require_format,
    )
    
    # Display detailed classification
    _print_enriched_results(filtered, show_files=True)
    
    return filtered


def cmd_verify(args):
    """Verify download links without downloading."""
    logger.info("=" * 60)
    logger.info("COMMAND: verify")
    logger.info("=" * 60)
    
    if not args.gse:
        print("Error: --gse is required for verify command.")
        sys.exit(1)
    
    gse_list = search_geo_direct(args.gse)
    enriched = enrich_gse_list(gse_list)
    
    filtered = apply_filters(
        enriched,
        format_filters=args.format_filter,
        modality_filter=args.modality,
    )
    
    all_ok = 0
    all_fail = 0
    
    for dc in filtered:
        files_to_verify = dc._filtered_files if hasattr(dc, '_filtered_files') and dc._filtered_files else dc.file_classifications
        urls = [clf.url for clf in files_to_verify]
        
        if not urls:
            print(f"\n  {dc.gse_id}: No files to verify")
            continue
        
        print(f"\n  Verifying {dc.gse_id} ({len(urls)} files)...")
        results = verify_links(urls)
        print_verification_report(results, dc.gse_id)
        
        all_ok += sum(1 for r in results if r.accessible)
        all_fail += sum(1 for r in results if not r.accessible)
    
    print(f"\n{'='*60}")
    print(f"  TOTAL: {all_ok} accessible, {all_fail} failed")
    print(f"{'='*60}")


def cmd_download(args):
    """Download files with filtering."""
    logger.info("=" * 60)
    logger.info("COMMAND: download")
    logger.info("=" * 60)
    
    output_dir = args.output_dir or "./downloads"
    
    if args.gse:
        gse_list = search_geo_direct(args.gse)
    elif os.path.exists("search_results.json"):
        gse_list = load_search_results("search_results.json")
    else:
        print("Error: specify --gse or run search first.")
        sys.exit(1)
    
    enriched = enrich_gse_list(gse_list)
    
    filtered = apply_filters(
        enriched,
        domain_filter=args.domain,
        modality_filter=args.modality,
        format_filters=args.format_filter,
        organism_filter=args.organism,
        require_format=bool(args.format_filter),
        require_convertible=not args.no_require_convertible,
    )
    
    if not filtered:
        print("No datasets match the filters.")
        return

    existing_gse_map = scan_existing_gse_locations(args.existing_roots)
    skipped_gse = []
    
    # Download only the filtered files
    for dc in filtered:
        gse_info = dc._gse_info
        gse_dir = os.path.join(output_dir, dc.gse_id)

        if args.skip_existing_gse and should_skip_gse_download(dc.gse_id, gse_dir, existing_gse_map):
            skipped_gse.append(dc.gse_id)
            first_existing = existing_gse_map.get(dc.gse_id, ["(unknown)"])[0]
            logger.info(f"Skipping {dc.gse_id}: already present at {first_existing}")
            continue

        os.makedirs(gse_dir, exist_ok=True)
        
        files_to_dl = dc._filtered_files if hasattr(dc, '_filtered_files') and dc._filtered_files else dc.file_classifications
        urls = [clf.url for clf in files_to_dl if clf.is_count_matrix or clf.format_type not in ("rds",)]
        
        print(f"\n{'='*60}")
        print(f"Downloading {dc.gse_id}: {len(urls)} files")
        print(f"{'='*60}")
        
        for url in urls:
            fname = url.rstrip("/").split("/")[-1]
            local_path = os.path.join(gse_dir, fname)
            
            success = download_file(
                url, local_path,
                max_retries=args.max_retries,
                timeout=args.timeout,
            )
            status = "OK" if success else "FAILED"
            logger.info(f"  {status}: {fname}")

        write_dataset_meta_json(gse_dir, dc.gse_id, include_geo_meta=True)
    
    print(f"\nDownloads saved to: {output_dir}")
    if skipped_gse:
        print(f"Skipped existing GSEs ({len(skipped_gse)}): {', '.join(sorted(set(skipped_gse)))}")


def cmd_convert(args):
    """Convert downloaded files to h5ad."""
    logger.info("=" * 60)
    logger.info("COMMAND: convert")
    logger.info("=" * 60)
    
    input_dir = args.input_dir or "./downloads"
    output_dir = args.output_dir or "./h5ad_output"
    
    convert_config = {
        "apply_qc": args.apply_qc,
        "min_genes": args.min_genes,
        "min_cells": args.min_cells,
        "max_mito_pct": args.max_mito_pct,
        "merge_samples": args.merge,
    }
    
    summary = convert_all(input_dir, output_dir, convert_config)
    
    print(f"\nConversion: {summary['success']}/{summary['total']} successful")
    if summary["outputs"]:
        print("Output files:")
        for p in summary["outputs"]:
            size_mb = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0
            print(f"  {p} ({size_mb:.1f} MB)")


def cmd_normalize(args):
    """
    Normalize existing downloads:
      1) Convert to h5ad (merged per GSE by default)
      2) Place canonical h5ad into each GSE folder
      3) Backfill unified JSON metadata in each GSE folder
    """
    logger.info("=" * 60)
    logger.info("COMMAND: normalize")
    logger.info("=" * 60)

    download_dir = args.downloads_dir or "./downloads"
    h5ad_dir = args.h5ad_dir or "./h5ad_output"

    if not os.path.isdir(download_dir):
        print(f"Error: downloads dir not found: {download_dir}")
        sys.exit(1)

    gse_dirs = []
    for item in sorted(os.listdir(download_dir)):
        item_path = os.path.join(download_dir, item)
        if os.path.isdir(item_path) and item.startswith("GSE"):
            gse_dirs.append(item_path)

    if not gse_dirs:
        print(f"No GSE folders found under: {download_dir}")
        return

    summary = {"outputs": []}
    if not args.meta_only:
        convert_config = {
            "apply_qc": args.apply_qc,
            "min_genes": args.min_genes,
            "min_cells": args.min_cells,
            "max_mito_pct": args.max_mito_pct,
            "merge_samples": True,
        }
        summary = convert_all(download_dir, h5ad_dir, convert_config)

    selected_h5ad = select_best_h5ad_by_gse(summary.get("outputs", []))

    linked = 0
    meta_written = 0
    for gse_dir in gse_dirs:
        gse_id = os.path.basename(gse_dir)

        src_h5ad = selected_h5ad.get(gse_id)
        if src_h5ad and os.path.exists(src_h5ad):
            dst_h5ad = os.path.join(gse_dir, f"{gse_id}.h5ad")
            if os.path.abspath(src_h5ad) != os.path.abspath(dst_h5ad):
                if (not os.path.exists(dst_h5ad)) or (os.path.getsize(dst_h5ad) != os.path.getsize(src_h5ad)):
                    shutil.copy2(src_h5ad, dst_h5ad)
                    linked += 1

        write_dataset_meta_json(gse_dir, gse_id, include_geo_meta=not args.no_geo_meta)
        meta_written += 1

    print(f"\nNormalization complete:")
    print(f"  GSE folders processed: {len(gse_dirs)}")
    if not args.meta_only:
        print(f"  Converted successfully: {summary.get('success', 0)}/{summary.get('total', 0)}")
        print(f"  h5ad outputs: {len(summary.get('outputs', []))}")
    print(f"  Canonical h5ad copied into GSE folders: {linked}")
    print(f"  JSON metadata written: {meta_written}")


def cmd_organize(args):
    """Organize downloaded GSE folders into thematic view (domain/organ)."""
    logger.info("=" * 60)
    logger.info("COMMAND: organize")
    logger.info("=" * 60)

    summary = organize_downloads_by_theme(
        downloads_dir=args.downloads_dir,
        thematic_root=args.thematic_root,
        mode=args.mode,
    )

    print("\nTheme organization complete:")
    print(f"  Total GSE seen: {summary.get('total_gse', 0)}")
    print(f"  Created: {summary.get('created', 0)}")
    print(f"  Skipped(existing): {summary.get('skipped', 0)}")
    print(f"  Mode: {summary.get('mode')}")
    print(f"  Root: {summary.get('thematic_root')}")


def cmd_cancer_gap(args):
    """Scan cancer organ gaps and optionally download top candidates serially."""
    logger.info("=" * 60)
    logger.info("COMMAND: cancer-gap")
    logger.info("=" * 60)

    report = scan_cancer_organ_gaps(
        downloads_dir=args.downloads_dir,
        format_filters=args.format_filter,
        modality=args.modality,
        organism=args.organism,
        max_results_per_organ=args.max_results_per_organ,
        max_candidates_per_organ=args.max_candidates_per_organ,
    )

    report_path = args.output or "./reports/cancer_organ_gap_report.json"
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nCancer organ gap scan complete:")
    print(f"  Covered organs: {len(report.get('covered_organs', []))}")
    print(f"  Missing organs: {len(report.get('missing_organs', []))}")
    print(f"  Report: {report_path}")

    if args.download_top_per_organ <= 0:
        return

    existing_gse_map = scan_existing_gse_locations(args.existing_roots)
    downloaded = 0

    for organ in report.get("missing_organs", []):
        candidates = report.get("candidates_by_organ", {}).get(organ, [])
        for candidate in candidates[: args.download_top_per_organ]:
            gse_id = candidate.get("gse_id")
            if not gse_id:
                continue

            gse_list = search_geo_direct([gse_id])
            if not gse_list:
                continue

            enriched = enrich_gse_list(gse_list)
            filtered = apply_filters(
                enriched,
                domain_filter="cancer",
                modality_filter=args.modality,
                format_filters=args.format_filter,
                organism_filter=None,
                require_format=True,
                require_convertible=not args.no_require_convertible,
            )
            if not filtered:
                continue

            dc = filtered[0]
            gse_dir = os.path.join(args.output_dir, dc.gse_id)
            if args.skip_existing_gse and should_skip_gse_download(dc.gse_id, gse_dir, existing_gse_map):
                logger.info("Skipping %s: already present", dc.gse_id)
                continue

            os.makedirs(gse_dir, exist_ok=True)
            urls = [clf.url for clf in dc._filtered_files] if hasattr(dc, "_filtered_files") else []
            logger.info("Serial download %s (%d files)", dc.gse_id, len(urls))
            for url in urls:
                fname = url.rstrip("/").split("/")[-1]
                local_path = os.path.join(gse_dir, fname)
                download_file(url, local_path, max_retries=args.max_retries, timeout=args.timeout)

            write_dataset_meta_json(gse_dir, dc.gse_id, include_geo_meta=True)
            downloaded += 1

    print(f"  Serial downloaded new GSEs: {downloaded}")


def cmd_run(args):
    """Full pipeline: search → filter → download → convert."""
    logger.info("=" * 60)
    logger.info("COMMAND: run (full pipeline)")
    logger.info("=" * 60)
    
    # Step 1: Search
    print("\n[1/4] Searching GEO...")
    if args.gse:
        gse_list = search_geo_direct(args.gse)
    else:
        search_config = {
            "query": args.query or "single cell RNA-seq",
            "organism": _resolve_organism(args.organism),
            "dataset_type": args.dataset_type or "",
            "max_results": args.max_results,
            "min_samples": args.min_samples,
        }
        gse_list = search_geo(search_config)
    
    if not gse_list:
        print("No datasets found.")
        return
    
    # Step 2: Enrich & Filter
    print(f"\n[2/4] Classifying {len(gse_list)} datasets...")
    enriched = enrich_gse_list(gse_list)
    
    filtered = apply_filters(
        enriched,
        domain_filter=args.domain,
        modality_filter=args.modality,
        format_filters=args.format_filter,
        organism_filter=args.organism,
        require_format=bool(args.format_filter),
        require_convertible=not args.no_require_convertible,
    )
    
    _print_enriched_results(filtered, show_files=True)
    
    if not filtered:
        print("No datasets pass the filters.")
        return
    
    save_search_results(
        [dc._gse_info for dc in filtered],
        "search_results.json",
    )
    
    # Step 3: Download
    output_dir = args.output_dir or "./downloads"
    print(f"\n[3/4] Downloading to {output_dir}...")

    existing_gse_map = scan_existing_gse_locations(args.existing_roots)
    skipped_gse = []
    
    for dc in filtered:
        gse_dir = os.path.join(output_dir, dc.gse_id)

        if args.skip_existing_gse and should_skip_gse_download(dc.gse_id, gse_dir, existing_gse_map):
            skipped_gse.append(dc.gse_id)
            first_existing = existing_gse_map.get(dc.gse_id, ["(unknown)"])[0]
            logger.info(f"Skipping {dc.gse_id}: already present at {first_existing}")
            continue

        os.makedirs(gse_dir, exist_ok=True)
        
        files_to_dl = dc._filtered_files if hasattr(dc, '_filtered_files') and dc._filtered_files else dc.file_classifications
        urls = [clf.url for clf in files_to_dl]
        
        logger.info(f"Downloading {dc.gse_id}: {len(urls)} files")
        
        for url in urls:
            fname = url.rstrip("/").split("/")[-1]
            local_path = os.path.join(gse_dir, fname)
            download_file(url, local_path, max_retries=args.max_retries, timeout=args.timeout)

        write_dataset_meta_json(gse_dir, dc.gse_id, include_geo_meta=True)

    if skipped_gse:
        print(f"Skipped existing GSEs ({len(skipped_gse)}): {', '.join(sorted(set(skipped_gse)))}")
    
    # Step 4: Convert
    h5ad_dir = args.h5ad_dir or "./h5ad_output"
    print(f"\n[4/4] Converting to h5ad → {h5ad_dir}...")
    
    convert_config = {
        "apply_qc": args.apply_qc,
        "min_genes": args.min_genes,
        "min_cells": args.min_cells,
        "max_mito_pct": args.max_mito_pct,
        "merge_samples": args.merge,
    }
    
    summary = convert_all(output_dir, h5ad_dir, convert_config)
    
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Datasets: {len(filtered)}")
    print(f"  h5ad files: {len(summary.get('outputs', []))}")
    if summary.get("outputs"):
        for p in summary["outputs"]:
            size_mb = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0
            print(f"    {p} ({size_mb:.1f} MB)")
    print(f"{'='*60}")


# ============================================================
# Display / Output Helpers
# ============================================================

def _print_enriched_results(
    filtered: List[DatasetClassification],
    show_files: bool = False,
):
    """Pretty-print enriched, filtered results."""
    print(f"\n{'='*80}")
    print(f" Results: {len(filtered)} datasets")
    print(f"{'='*80}")
    
    for i, dc in enumerate(filtered, 1):
        gse_info = dc._gse_info
        title = gse_info.title[:75] if gse_info.title else "(no title)"
        
        # Domain badge
        domain_badge = {
            "cancer": "🔴 CANCER",
            "development": "🟢 DEV",
            "normal": "🔵 NORMAL",
            "disease": "🟡 DISEASE",
            "unknown": "⚪ ?",
        }.get(dc.domain, dc.domain)
        
        # Modality badge
        mod_parts = []
        if "rna" in dc.modalities: mod_parts.append("RNA")
        if "atac" in dc.modalities: mod_parts.append("ATAC")
        if "multiome" in dc.modalities: mod_parts.append("MULTI")
        if "cite" in dc.modalities: mod_parts.append("CITE")
        mod_badge = "+".join(mod_parts) if mod_parts else "?"
        
        print(f"\n[{i}] {dc.gse_id}  {domain_badge}  [{mod_badge}]")
        print(f"    {title}")
        print(f"    Organism: {dc.organism or '?'} | Type: {dc.series_type or '?'}")
        print(f"    Formats:  {dc.available_formats_str()}")
        
        if show_files:
            print_classification_report(dc)
        
        # Show filtered files count if applicable
        if hasattr(dc, '_filtered_files') and dc._filtered_files:
            n_filtered = len(dc._filtered_files)
            n_total = len(dc.file_classifications)
            if n_filtered < n_total:
                print(f"    → Filtered: {n_filtered}/{n_total} files match your criteria")
    
    print(f"\n{'='*80}")


def _save_enriched_results(gse_list, enriched, output_path=None):
    """Save search results JSON."""
    output_path = output_path or "search_results.json"
    save_search_results(gse_list, output_path)
    
    # Also save enriched classification
    clf_path = output_path.replace(".json", "_classified.json")
    clf_data = []
    for dc in enriched:
        clf_data.append({
            "gse_id": dc.gse_id,
            "domain": dc.domain,
            "modalities": sorted(dc.modalities),
            "formats": dc.available_formats_str(),
            "organism": dc.organism,
            "n_files": len(dc.file_classifications),
            "files": [
                {
                    "filename": clf.filename,
                    "format_type": clf.format_type,
                    "modality_signal": clf.modality_signal,
                    "processing_level": clf.processing_level,
                    "is_count_matrix": clf.is_count_matrix,
                }
                for clf in dc.file_classifications
            ],
        })
    
    with open(clf_path, "w") as f:
        json.dump(clf_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Classification saved to {clf_path}")


def _resolve_organism(name: Optional[str]) -> str:
    """Resolve common organism names to formal names."""
    if not name:
        return ""
    mapping = {
        "human": "Homo sapiens",
        "mouse": "Mus musculus",
        "rat": "Rattus norvegicus",
        "zebrafish": "Danio rerio",
        "fly": "Drosophila melanogaster",
        "worm": "Caenorhabditis elegans",
    }
    return mapping.get(name.lower(), name)


# ============================================================
# Argument Parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    
    # ── Top-level parser ──
    parser = argparse.ArgumentParser(
        prog="geodh",
        description="GEO-DataHub: Search, filter, download & convert single-cell data from GEO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT FILTERS (--format-filter):
  filtered_feature_h5    10x filtered_feature_bc_matrix.h5 (scRNA)
  raw_feature_h5         10x raw_feature_bc_matrix.h5 (scRNA)
  filtered_peak_h5       10x filtered_peak_bc_matrix.h5 (scATAC)
  raw_peak_h5            10x raw_peak_bc_matrix.h5 (scATAC)
  h5ad                   AnnData .h5ad files
  h5                     Any HDF5 file
  mtx                    10x MTX sparse matrix triplet
  loom                   Loom files
  csv / tsv              Count matrices in CSV/TSV
  fragments              ATAC fragments.tsv.gz
  tar                    Tar archives (contain per-sample files)
  count_matrix           Any file identified as a count matrix

MODALITY FILTERS (--modality):
  rna        scRNA-seq / snRNA-seq
  atac       scATAC-seq / snATAC-seq
  multiome   Multi-omics (RNA + ATAC jointly)
  cite       CITE-seq / protein + RNA

DOMAIN FILTERS (--domain):
  cancer       Cancer / tumor datasets
  development  Development / differentiation / fetal
  normal       Healthy tissue / atlas / homeostasis
  disease      Non-cancer disease (neurodegeneration, fibrosis, etc.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLES:
  # Search human cancer scRNA with filtered 10x h5 files
  python geodh.py search --query "scRNA cancer" --organism human \\
      --domain cancer --modality rna --format-filter filtered_feature_h5

  # Scan specific GSEs for file availability
  python geodh.py scan --gse GSE174367 GSE200997

  # Verify links (no download) for ATAC peak h5 files
  python geodh.py verify --gse GSE174367 --format-filter filtered_peak_h5

  # Download only filtered feature h5 from matched datasets
  python geodh.py download --gse GSE185224 --format-filter filtered_feature_h5

  # Full pipeline: search → classify → download → convert
  python geodh.py run --query "lung cancer scATAC" --domain cancer \\
      --modality atac --format-filter filtered_peak_h5 --max-results 5
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ── Shared arguments (added to each subparser) ──
    def add_common_args(sub):
        """Add arguments shared across commands."""
        g = sub.add_argument_group("Data Source")
        g.add_argument("--gse", nargs="+", metavar="GSE_ID",
                        help="Specific GSE accession(s) to process")
        g.add_argument("--query", "-q", type=str, default=None,
                        help="GEO search query (e.g. 'scRNA-seq lung cancer human')")
        g.add_argument("--organism", "-org", type=str, default=None,
                        help="Organism filter (human, mouse, or full Latin name)")
        g.add_argument("--max-results", "-n", type=int, default=20,
                        help="Maximum search results (default: 20)")
        g.add_argument("--min-samples", type=int, default=1,
                        help="Minimum samples per series")
        g.add_argument("--dataset-type", type=str, default=None,
                        help="GEO dataset type filter")
        
        g = sub.add_argument_group("Filters")
        g.add_argument("--domain", "-d", type=str, default=None,
                        choices=["cancer", "development", "normal", "disease"],
                        help="Biological domain filter")
        g.add_argument("--modality", "-m", type=str, default=None,
                        choices=["rna", "atac", "multiome", "cite"],
                        help="Data modality filter")
        g.add_argument("--format-filter", "-f", nargs="+", default=None,
                        metavar="FMT",
                        choices=list(FORMAT_FILTER_MAP.keys()),
                        help="File format filter(s)")
        g.add_argument("--require-format", action="store_true",
                        help="Exclude datasets with no files matching format filter")
    
    def add_output_args(sub):
        sub.add_argument("--output", "-o", type=str, default=None,
                         help="Output JSON path for results")
        sub.add_argument("--log-level", type=str, default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # ── search ──
    p_search = subparsers.add_parser("search", help="Search GEO for datasets")
    add_common_args(p_search)
    add_output_args(p_search)
    p_search.add_argument("--show-files", action="store_true",
                          help="Show individual file classifications")
    
    # ── scan ──
    p_scan = subparsers.add_parser("scan", help="Scan GSEs: classify files, modality, domain")
    add_common_args(p_scan)
    add_output_args(p_scan)
    
    # ── verify ──
    p_verify = subparsers.add_parser("verify", help="Verify download links (no download)")
    add_common_args(p_verify)
    add_output_args(p_verify)
    
    # ── download ──
    p_download = subparsers.add_parser("download", help="Download files with filtering")
    add_common_args(p_download)
    add_output_args(p_download)
    p_download.add_argument("--output-dir", type=str, default="./downloads",
                            help="Download directory (default: ./downloads)")
    p_download.add_argument("--existing-roots", nargs="+",
                            default=["./downloads"],
                            help="Roots scanned for already downloaded GSEs")
    p_download.add_argument("--skip-existing-gse", action=argparse.BooleanOptionalAction, default=True,
                            help="Skip download if GSE already exists in existing-roots")
    p_download.add_argument("--max-retries", type=int, default=3)
    p_download.add_argument("--timeout", type=int, default=600,
                            help="Download timeout per file in seconds")
    p_download.add_argument("--no-require-convertible", action="store_true", default=False,
                            help="Disable the default filter that rejects datasets with "
                                 "no h5/h5ad/mtx/loom convertible files (e.g. only .rds, .bw)")
    
    # ── convert ──
    p_convert = subparsers.add_parser("convert", help="Convert downloaded files to h5ad")
    add_output_args(p_convert)
    p_convert.add_argument("--input-dir", type=str, default="./downloads")
    p_convert.add_argument("--output-dir", type=str, default="./h5ad_output")
    p_convert.add_argument("--apply-qc", action="store_true", help="Apply QC filtering")
    p_convert.add_argument("--min-genes", type=int, default=200)
    p_convert.add_argument("--min-cells", type=int, default=3)
    p_convert.add_argument("--max-mito-pct", type=float, default=20.0)
    p_convert.add_argument("--merge", action="store_true",
                           help="Merge multiple samples per GSE into one h5ad")
    
    # ── run (full pipeline) ──
    p_run = subparsers.add_parser("run", help="Full pipeline: search → filter → download → convert")
    add_common_args(p_run)
    add_output_args(p_run)
    p_run.add_argument("--output-dir", type=str, default="./downloads")
    p_run.add_argument("--h5ad-dir", type=str, default="./h5ad_output")
    p_run.add_argument("--existing-roots", nargs="+",
                       default=["./downloads"],
                       help="Roots scanned for already downloaded GSEs")
    p_run.add_argument("--skip-existing-gse", action=argparse.BooleanOptionalAction, default=True,
                       help="Skip download if GSE already exists in existing-roots")
    p_run.add_argument("--max-retries", type=int, default=3)
    p_run.add_argument("--timeout", type=int, default=600)
    p_run.add_argument("--apply-qc", action="store_true")
    p_run.add_argument("--min-genes", type=int, default=200)
    p_run.add_argument("--min-cells", type=int, default=3)
    p_run.add_argument("--max-mito-pct", type=float, default=20.0)
    p_run.add_argument("--merge", action="store_true")
    p_run.add_argument("--no-require-convertible", action="store_true", default=False,
                       help="Disable the default filter that rejects datasets with "
                            "no h5/h5ad/mtx/loom convertible files")

    # ── normalize (backfill + standardize) ──
    p_norm = subparsers.add_parser("normalize", help="Backfill metadata JSON and standardize to h5ad per GSE")
    add_output_args(p_norm)
    p_norm.add_argument("--downloads-dir", type=str, default="./downloads")
    p_norm.add_argument("--h5ad-dir", type=str, default="./h5ad_output")
    p_norm.add_argument("--meta-only", action="store_true",
                        help="Only refresh JSON metadata (skip conversion)")
    p_norm.add_argument("--no-geo-meta", action="store_true",
                        help="Skip GEO online metadata fetch and only summarize local files")
    p_norm.add_argument("--apply-qc", action="store_true")
    p_norm.add_argument("--min-genes", type=int, default=200)
    p_norm.add_argument("--min-cells", type=int, default=3)
    p_norm.add_argument("--max-mito-pct", type=float, default=20.0)

    # ── organize (theme view) ──
    p_orgz = subparsers.add_parser("organize", help="Organize downloads by domain/organ thematic folders")
    add_output_args(p_orgz)
    p_orgz.add_argument("--downloads-dir", type=str, default="./downloads")
    p_orgz.add_argument("--thematic-root", type=str, default="./downloads_by_theme")
    p_orgz.add_argument("--mode", type=str, choices=["symlink", "copy", "move"], default="symlink")

    # ── cancer-gap (coverage scan + optional serial download) ──
    p_gap = subparsers.add_parser("cancer-gap", help="Scan missing cancer organs and suggest/download candidates")
    add_output_args(p_gap)
    p_gap.add_argument("--downloads-dir", type=str, default="./downloads")
    p_gap.add_argument("--output-dir", type=str, default="./downloads")
    p_gap.add_argument("--organism", type=str, default="human")
    p_gap.add_argument("--modality", type=str, default="rna", choices=["rna", "atac", "multiome", "cite"])
    p_gap.add_argument("--format-filter", "-f", nargs="+", default=["filtered_feature_h5", "mtx"],
                       metavar="FMT", choices=list(FORMAT_FILTER_MAP.keys()))
    p_gap.add_argument("--max-results-per-organ", type=int, default=12)
    p_gap.add_argument("--max-candidates-per-organ", type=int, default=5)
    p_gap.add_argument("--download-top-per-organ", type=int, default=0,
                       help="Serially download top-N candidates per missing organ")
    p_gap.add_argument("--existing-roots", nargs="+",
                       default=["./downloads"],
                       help="Roots scanned for already downloaded GSEs")
    p_gap.add_argument("--skip-existing-gse", action=argparse.BooleanOptionalAction, default=True)
    p_gap.add_argument("--max-retries", type=int, default=3)
    p_gap.add_argument("--timeout", type=int, default=600)
    p_gap.add_argument("--no-require-convertible", action="store_true", default=False,
                       help="Disable the default filter that rejects datasets with "
                            "no h5/h5ad/mtx/loom convertible files")
    
    return parser


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Setup logging
    log_level = getattr(args, "log_level", "INFO")
    setup_logging(level=log_level)
    
    logger.info(f"geodh command={args.command}")
    start = datetime.now()
    
    # Dispatch
    commands = {
        "search": cmd_search,
        "scan": cmd_scan,
        "verify": cmd_verify,
        "download": cmd_download,
        "convert": cmd_convert,
        "run": cmd_run,
        "normalize": cmd_normalize,
        "organize": cmd_organize,
        "cancer-gap": cmd_cancer_gap,
    }
    
    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    elapsed = datetime.now() - start
    logger.info(f"Done in {elapsed}")


if __name__ == "__main__":
    main()
