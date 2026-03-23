"""
GEO-DataHub: Main Pipeline Orchestrator
========================================
Orchestrates the full pipeline: Search → Download → Convert to h5ad.

Usage:
  # Full pipeline (search + download + convert)
  python pipeline.py --config config.yaml

  # Search only
  python pipeline.py --config config.yaml --step search

  # Download only (uses previous search results)
  python pipeline.py --config config.yaml --step download

  # Convert only (uses previously downloaded files)  
  python pipeline.py --config config.yaml --step convert

  # Direct GSE accession mode (skip search)
  python pipeline.py --gse GSE123456 GSE789012

Author: GEO-DataHub Pipeline
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

import yaml

from .geo_search import search_geo, search_geo_direct, print_search_results, save_search_results, load_search_results
from .geo_download import download_all
from .geo_convert import convert_all


# ============================================================
# Logging Setup
# ============================================================
def setup_logging(config: dict) -> logging.Logger:
    """Configure logging to both file and console."""
    log_config = config.get("logging", {})
    log_dir = log_config.get("log_dir", "./logs")
    log_level = log_config.get("level", "INFO")
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level, logging.INFO))
    ch.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    
    logger = logging.getLogger("pipeline")
    logger.info(f"Log file: {log_file}")
    
    return logger


# ============================================================
# Pipeline Steps
# ============================================================
def step_search(config: dict, gse_ids: list = None) -> list:
    """Step 1: Search or lookup GEO datasets."""
    logger = logging.getLogger("pipeline")
    
    logger.info("=" * 60)
    logger.info("STEP 1: Search GEO Database")
    logger.info("=" * 60)
    
    if gse_ids:
        logger.info(f"Direct lookup mode: {gse_ids}")
        gse_list = search_geo_direct(gse_ids)
    else:
        search_config = config.get("search", {})
        gse_list = search_geo(search_config)
    
    # Display results
    print_search_results(gse_list)
    
    # Save results
    save_search_results(gse_list, "search_results.json")
    
    logger.info(f"Search complete: {len(gse_list)} datasets found")
    return gse_list


def step_download(config: dict, gse_list: list = None) -> list:
    """Step 2: Download selected datasets."""
    logger = logging.getLogger("pipeline")
    
    logger.info("=" * 60)
    logger.info("STEP 2: Download Files")
    logger.info("=" * 60)
    
    download_config = config.get("download", {})
    
    if gse_list is None:
        # Try to load from previous search
        search_file = "search_results.json"
        if os.path.exists(search_file):
            gse_list = load_search_results(search_file)
            logger.info(f"Loaded {len(gse_list)} datasets from {search_file}")
        else:
            logger.error(f"No search results found. Run search step first.")
            return []
    
    manifests = download_all(gse_list, download_config)
    
    logger.info(f"Download complete: {len(manifests)} datasets processed")
    return manifests


def step_convert(config: dict) -> dict:
    """Step 3: Convert downloaded files to h5ad."""
    logger = logging.getLogger("pipeline")
    
    logger.info("=" * 60)
    logger.info("STEP 3: Convert to h5ad")
    logger.info("=" * 60)
    
    convert_config = config.get("convert", {})
    download_config = config.get("download", {})
    
    input_dir = convert_config.get("input_dir", download_config.get("output_dir", "./downloads"))
    output_dir = convert_config.get("output_dir", "./h5ad_output")
    
    summary = convert_all(input_dir, output_dir, convert_config)
    
    logger.info(f"Conversion complete: {summary['success']}/{summary['total']} successful")
    return summary


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="GEO-DataHub: Search, Download & Convert scRNA-seq data from GEO to h5ad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python pipeline.py --config config.yaml
  
  # Search only
  python pipeline.py --config config.yaml --step search
  
  # Direct GSE lookup + download + convert
  python pipeline.py --gse GSE123456 GSE789012
  
  # Convert previously downloaded files
  python pipeline.py --config config.yaml --step convert
        """,
    )
    
    parser.add_argument("--config", default="config.yaml", help="Config file path (default: config.yaml)")
    parser.add_argument("--step", choices=["search", "download", "convert", "all"], default="all",
                        help="Which pipeline step to run (default: all)")
    parser.add_argument("--gse", nargs="+", help="Directly specify GSE accession(s) to process")
    
    args = parser.parse_args()
    
    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        config = {}
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info(f"GEO-DataHub Pipeline Started")
    logger.info(f"Step: {args.step}")
    logger.info(f"Config: {config_path}")
    if args.gse:
        logger.info(f"Direct GSE: {args.gse}")
    
    start_time = datetime.now()
    
    try:
        if args.step == "search" or args.step == "all":
            gse_list = step_search(config, gse_ids=args.gse)
        else:
            gse_list = None
        
        if args.step == "download" or args.step == "all":
            manifests = step_download(config, gse_list=gse_list)
        
        if args.step == "convert" or args.step == "all":
            summary = step_convert(config)
            
            # Final report
            if summary.get("outputs"):
                logger.info(f"\n{'='*60}")
                logger.info("FINAL OUTPUT h5ad FILES:")
                logger.info(f"{'='*60}")
                for out_path in summary["outputs"]:
                    size_mb = os.path.getsize(out_path) / 1024 / 1024 if os.path.exists(out_path) else 0
                    logger.info(f"  {out_path} ({size_mb:.1f} MB)")
    
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline error: {e}", exc_info=True)
        sys.exit(1)
    
    elapsed = datetime.now() - start_time
    logger.info(f"\nPipeline completed in {elapsed}")


if __name__ == "__main__":
    main()
