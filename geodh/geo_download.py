"""
GEO-DataHub: GEO Download Module
=================================
Download supplementary files from GEO for selected datasets.

Handles:
  - Direct HTTP/FTP download of supplementary files
  - Tar/gz extraction
  - Resume support for interrupted downloads
  - File size filtering
  - Progress reporting

Author: GEO-DataHub Pipeline
"""

import os
import gzip
import shutil
import tarfile
import logging
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import requests
import yaml

from .geo_search import (
    GEOSeriesInfo,
    load_search_results,
    fetch_supplementary_file_list,
    classify_supplementary_files,
)

logger = logging.getLogger("geo_download")


# ============================================================
# Download Functions
# ============================================================
def get_remote_file_size(url: str, timeout: int = 30) -> Optional[int]:
    """Get the size of a remote file in bytes via HEAD request."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            size = resp.headers.get("Content-Length")
            return int(size) if size else None
    except Exception:
        pass
    return None


def download_file(
    url: str,
    output_path: str,
    max_retries: int = 3,
    timeout: int = 600,
    chunk_size: int = 8192 * 4,
) -> bool:
    """
    Download a file with progress reporting and retry support.
    
    Args:
        url: Remote file URL
        output_path: Local path to save
        max_retries: Number of retry attempts
        timeout: Request timeout in seconds
        chunk_size: Download chunk size in bytes
    
    Returns:
        True if download successful, False otherwise
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(output_path):
        local_size = os.path.getsize(output_path)
        remote_size = get_remote_file_size(url)
        if remote_size and local_size == remote_size:
            logger.info(f"  Already downloaded: {os.path.basename(output_path)}")
            return True
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"  Downloading ({attempt}/{max_retries}): {os.path.basename(output_path)}")
            
            # Support resume via Range header
            headers = {}
            mode = "wb"
            initial_size = 0
            
            if os.path.exists(output_path):
                initial_size = os.path.getsize(output_path)
                headers["Range"] = f"bytes={initial_size}-"
                mode = "ab"
                logger.info(f"    Resuming from {initial_size / 1024 / 1024:.1f} MB")
            
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            
            if resp.status_code == 416:
                # Range not satisfiable = already complete
                logger.info(f"    File already complete.")
                return True
            
            resp.raise_for_status()
            
            # Get total size
            total_size = resp.headers.get("Content-Length")
            total_size = int(total_size) + initial_size if total_size else None
            
            downloaded = initial_size
            last_report = time.time()
            
            with open(output_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress reporting every 5 seconds
                        now = time.time()
                        if now - last_report > 5:
                            if total_size:
                                pct = downloaded / total_size * 100
                                mb_done = downloaded / 1024 / 1024
                                mb_total = total_size / 1024 / 1024
                                logger.info(f"    Progress: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
                            else:
                                logger.info(f"    Downloaded: {downloaded / 1024 / 1024:.1f} MB")
                            last_report = now
            
            # Final size report
            final_mb = downloaded / 1024 / 1024
            logger.info(f"    Complete: {final_mb:.1f} MB")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"    Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = attempt * 5
                logger.info(f"    Retrying in {wait}s...")
                time.sleep(wait)
    
    logger.error(f"  Failed to download after {max_retries} attempts: {url}")
    return False


def extract_tar(tar_path: str, extract_dir: str) -> List[str]:
    """
    Extract a tar/tar.gz/tgz archive.
    
    Returns list of extracted file paths.
    """
    extracted = []
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            # Security: avoid path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    logger.warning(f"Skipping suspicious tar member: {member.name}")
                    continue
                tar.extract(member, extract_dir)
                extracted.append(os.path.join(extract_dir, member.name))
        
        logger.info(f"  Extracted {len(extracted)} files from {os.path.basename(tar_path)}")
    except Exception as e:
        logger.error(f"  Failed to extract {tar_path}: {e}")
    
    return extracted


def decompress_gz(gz_path: str) -> Optional[str]:
    """
    Decompress a .gz file (non-tar).
    
    Returns path to decompressed file, or None on failure.
    """
    if not gz_path.endswith(".gz"):
        return gz_path
    
    output_path = gz_path[:-3]  # Remove .gz
    
    if os.path.exists(output_path):
        logger.info(f"  Already decompressed: {os.path.basename(output_path)}")
        return output_path
    
    try:
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(f"  Decompressed: {os.path.basename(output_path)}")
        return output_path
    except Exception as e:
        logger.error(f"  Failed to decompress {gz_path}: {e}")
        return None


# ============================================================
# Smart Download Logic
# ============================================================
def select_files_for_download(
    files: List[str],
    max_file_size_gb: float = 5.0,
) -> List[str]:
    """
    Intelligently select which files to download based on format priority.
    
    Priority order:
    1. .h5ad files (already in target format)
    2. .h5 files (10x HDF5, easy to convert)
    3. .mtx + barcodes + features (10x sparse matrix)
    4. .tar.gz archives (may contain any of the above)
    5. .csv/.tsv count matrices
    6. .loom files
    
    Files exceeding max_file_size_gb are skipped.
    """
    categories = classify_supplementary_files(files)
    
    selected = []
    
    # Priority 1: h5ad
    selected.extend(categories["h5ad"])
    
    # Priority 2: h5
    selected.extend(categories["h5"])
    
    # Priority 3: mtx triplets
    selected.extend(categories["mtx"])
    
    # Priority 4: tar archives (likely contain mtx or h5)
    selected.extend(categories["tar"])
    
    # Priority 5: csv/tsv
    selected.extend(categories["csv_tsv"])
    
    # Priority 6: loom
    selected.extend(categories["loom"])
    
    # If nothing was selected from known categories, download other files too
    if not selected and categories["other"]:
        selected.extend(categories["other"])
    
    # Filter by file size
    filtered = []
    for url in selected:
        size = get_remote_file_size(url)
        if size is not None:
            size_gb = size / (1024 ** 3)
            if size_gb > max_file_size_gb:
                fname = url.split("/")[-1]
                logger.warning(f"  Skipping {fname} ({size_gb:.2f} GB > {max_file_size_gb} GB limit)")
                continue
        filtered.append(url)
    
    return filtered


def download_gse(
    gse_info: GEOSeriesInfo,
    output_dir: str,
    max_file_size_gb: float = 5.0,
    max_retries: int = 3,
    timeout: int = 600,
) -> dict:
    """
    Download all relevant files for a single GSE series.
    
    Args:
        gse_info: GEOSeriesInfo object
        output_dir: Base output directory
        max_file_size_gb: Max file size to download
        max_retries: Download retry count
        timeout: Download timeout
    
    Returns:
        Dict with 'gse_id', 'download_dir', 'files' (downloaded paths), 'status'
    """
    gse_dir = os.path.join(output_dir, gse_info.gse_id)
    os.makedirs(gse_dir, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading {gse_info.gse_id}: {gse_info.title[:60]}")
    logger.info(f"{'='*60}")
    
    # Refresh file list if needed
    if not gse_info.supplementary_files:
        gse_info.supplementary_files = fetch_supplementary_file_list(gse_info.gse_id)
    
    if not gse_info.supplementary_files:
        logger.warning(f"No supplementary files found for {gse_info.gse_id}")
        return {"gse_id": gse_info.gse_id, "download_dir": gse_dir, "files": [], "status": "no_files"}
    
    # Select files to download
    selected_files = select_files_for_download(
        gse_info.supplementary_files,
        max_file_size_gb=max_file_size_gb,
    )
    
    logger.info(f"Selected {len(selected_files)} files for download")
    
    downloaded_files = []
    failed_files = []
    
    for url in selected_files:
        fname = url.split("/")[-1]
        local_path = os.path.join(gse_dir, fname)
        
        success = download_file(
            url, local_path,
            max_retries=max_retries,
            timeout=timeout,
        )
        
        if success:
            downloaded_files.append(local_path)
            
            # Auto-extract tar files
            if any(local_path.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz"]):
                extract_dir = os.path.join(gse_dir, "extracted")
                extracted = extract_tar(local_path, extract_dir)
                downloaded_files.extend(extracted)
        else:
            failed_files.append(url)
    
    status = "success" if not failed_files else ("partial" if downloaded_files else "failed")
    
    # Save download manifest
    manifest = {
        "gse_id": gse_info.gse_id,
        "title": gse_info.title,
        "organism": gse_info.organism,
        "download_dir": gse_dir,
        "downloaded_files": downloaded_files,
        "failed_files": failed_files,
        "status": status,
    }
    
    manifest_path = os.path.join(gse_dir, "download_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Download {status}: {len(downloaded_files)} files saved, {len(failed_files)} failed")
    
    return manifest


# ============================================================
# Batch Download
# ============================================================
def download_all(
    gse_list: List[GEOSeriesInfo],
    config: dict,
) -> List[dict]:
    """
    Download files for all GSE series in the list.
    
    Args:
        gse_list: List of GEOSeriesInfo objects
        config: Download configuration from config.yaml
    
    Returns:
        List of download manifest dicts
    """
    output_dir = config.get("output_dir", "./downloads")
    max_file_size_gb = config.get("max_file_size_gb", 5.0)
    max_retries = config.get("max_retries", 3)
    timeout = config.get("timeout", 600)
    selected_gse = config.get("selected_gse", [])
    
    # Filter to selected GSE if specified
    if selected_gse:
        gse_list = [g for g in gse_list if g.gse_id in selected_gse]
        logger.info(f"Filtered to {len(gse_list)} selected GSE series")
    
    manifests = []
    for i, gse_info in enumerate(gse_list, 1):
        logger.info(f"\n[{i}/{len(gse_list)}] Processing {gse_info.gse_id}")
        
        manifest = download_gse(
            gse_info,
            output_dir=output_dir,
            max_file_size_gb=max_file_size_gb,
            max_retries=max_retries,
            timeout=timeout,
        )
        manifests.append(manifest)
        
        # Polite delay between series
        time.sleep(1)
    
    # Summary
    success = sum(1 for m in manifests if m["status"] == "success")
    partial = sum(1 for m in manifests if m["status"] == "partial")
    failed = sum(1 for m in manifests if m["status"] == "failed")
    no_files = sum(1 for m in manifests if m["status"] == "no_files")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary: {success} success, {partial} partial, {failed} failed, {no_files} no files")
    logger.info(f"{'='*60}")
    
    # Save overall manifest
    overall_path = os.path.join(output_dir, "download_summary.json")
    with open(overall_path, "w") as f:
        json.dump(manifests, f, indent=2)
    
    return manifests


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Download GEO datasets")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--search-results", default="search_results.json", help="Search results JSON from geo_search.py")
    parser.add_argument("--gse", nargs="+", help="Directly download specific GSE accessions")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    download_config = config.get("download", {})
    
    if args.gse:
        # Direct GSE mode
        from .geo_search import search_geo_direct
        gse_list = search_geo_direct(args.gse)
    else:
        # Load from search results
        gse_list = load_search_results(args.search_results)
    
    # Download
    manifests = download_all(gse_list, download_config)
    
    return manifests


if __name__ == "__main__":
    main()
