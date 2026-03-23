"""
GEO-DataHub: Format Conversion Module
======================================
Convert downloaded GEO files to AnnData .h5ad format.

Supported input formats:
  - 10x HDF5 (.h5) — CellRanger output
  - AnnData (.h5ad) — direct copy / validation
  - 10x MTX triplet (matrix.mtx + barcodes.tsv + features/genes.tsv)
  - CSV/TSV count matrices
  - Loom (.loom)
  - Tar archives containing any of the above

Author: GEO-DataHub Pipeline
"""

import os
import re
import gzip
import json
import shutil
import tarfile
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import h5py
import anndata as ad
import yaml

logger = logging.getLogger("geo_convert")


# ============================================================
# Format Detection
# ============================================================
def detect_format(file_path: str) -> str:
    """
    Detect the data format of a file.
    
    Returns one of: 'h5ad', 'h5_10x', 'h5_other', 'mtx', 'csv', 'tsv',
                     'loom', 'tar', 'gz', 'unknown'
    """
    fp = file_path.lower()
    basename = os.path.basename(fp)
    
    if fp.endswith(".h5ad"):
        return "h5ad"
    elif fp.endswith(".h5ad.gz"):
        return "h5ad_gz"
    elif fp.endswith(".h5"):
        # Check if it's 10x format
        try:
            with h5py.File(file_path, "r") as f:
                if "matrix" in f or "GRCh38" in f or "mm10" in f:
                    return "h5_10x"
                elif "X" in f and "obs" in f:
                    return "h5ad"  # Actually h5ad without extension
                else:
                    return "h5_other"
        except Exception:
            return "h5_other"
    elif "matrix.mtx" in basename or basename == "matrix.mtx.gz":
        return "mtx"
    elif fp.endswith(".loom"):
        return "loom"
    elif fp.endswith((".tar.gz", ".tgz", ".tar")):
        return "tar"
    elif fp.endswith(".gz") and not fp.endswith(".tar.gz"):
        # Could be mtx.gz, csv.gz, tsv.gz
        inner = fp[:-3]
        if inner.endswith(".mtx"):
            return "mtx"
        elif inner.endswith(".csv"):
            return "csv"
        elif inner.endswith((".tsv", ".txt")):
            return "tsv"
        return "gz"
    elif fp.endswith(".csv"):
        return "csv"
    elif fp.endswith((".tsv", ".txt")):
        return "tsv"
    
    return "unknown"


def find_mtx_triplet(directory: str) -> Optional[Tuple[str, str, str]]:
    """
    Find 10x MTX triplet files in a directory.
    
    Returns (matrix_path, barcodes_path, features_path) or None.
    """
    matrix_files = []
    barcode_files = []
    feature_files = []

    for f in os.listdir(directory):
        fl = f.lower()
        full = os.path.join(directory, f)

        if "matrix.mtx" in fl:
            matrix_files.append(full)
        elif "barcodes.tsv" in fl:
            barcode_files.append(full)
        elif "features.tsv" in fl or "genes.tsv" in fl:
            feature_files.append(full)

    if not matrix_files or not barcode_files or not feature_files:
        return None

    def _prefix_score(matrix_name: str, other_name: str) -> int:
        matrix_stem = re.sub(r"matrix\.mtx(\.gz)?$", "", matrix_name, flags=re.IGNORECASE)
        other_stem = re.sub(r"(barcodes|features|genes)\.tsv(\.gz)?$", "", other_name, flags=re.IGNORECASE)
        matrix_tokens = [t for t in re.split(r"[_\-.]+", matrix_stem.lower()) if t]
        other_tokens = [t for t in re.split(r"[_\-.]+", other_stem.lower()) if t]
        return len(set(matrix_tokens).intersection(other_tokens))

    best = None
    best_score = -1
    for m in matrix_files:
        m_name = os.path.basename(m)
        best_b = max(barcode_files, key=lambda b: _prefix_score(m_name, os.path.basename(b)))
        best_f = max(feature_files, key=lambda f: _prefix_score(m_name, os.path.basename(f)))
        score = _prefix_score(m_name, os.path.basename(best_b)) + _prefix_score(m_name, os.path.basename(best_f))
        if score > best_score:
            best_score = score
            best = (m, best_b, best_f)

    if best:
        return best
    
    return None


def scan_directory_for_data(directory: str) -> List[dict]:
    """
    Recursively scan a directory for convertible data files.
    
    Returns list of dicts: {'path': ..., 'format': ..., 'mtx_triplet': ...}
    """
    found = []
    mtx_dirs_checked = set()
    
    for root, dirs, files in os.walk(directory):
        # Check for MTX triplet first
        if root not in mtx_dirs_checked:
            mtx_dirs_checked.add(root)
            triplet = find_mtx_triplet(root)
            if triplet:
                found.append({
                    "path": root,
                    "format": "mtx_triplet",
                    "mtx_triplet": triplet,
                })
                continue  # Don't double-count individual mtx files
        
        for fname in files:
            fpath = os.path.join(root, fname)
            fl = fname.lower()

            # Skip pipeline metadata/manifests and obvious non-expression tables
            if fl in {
                "dataset_meta.json",
                "download_manifest_standard.tsv",
                "download_manifest.csv",
                "download_manifest.json",
                "conversion_summary.json",
            }:
                continue
            if "fragments.tsv" in fl or "positions_file.csv" in fl:
                continue

            fmt = detect_format(fpath)
            
            if fmt in ("h5ad", "h5ad_gz", "h5_10x", "h5_other", "csv", "tsv", "loom"):
                found.append({
                    "path": fpath,
                    "format": fmt,
                    "mtx_triplet": None,
                })
            elif fmt == "tar":
                found.append({
                    "path": fpath,
                    "format": "tar",
                    "mtx_triplet": None,
                })
    
    return found


# ============================================================
# Conversion Functions
# ============================================================
def convert_10x_h5(file_path: str) -> Optional[ad.AnnData]:
    """Convert 10x Genomics HDF5 file to AnnData."""
    try:
        import scanpy as sc
        adata = sc.read_10x_h5(file_path)
        adata.var_names_make_unique()
        logger.info(f"  Loaded 10x H5: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    except Exception as e:
        logger.error(f"  Failed to read 10x H5 {file_path}: {e}")
        # Fallback: try reading as generic h5
        return convert_generic_h5(file_path)


def convert_generic_h5(file_path: str) -> Optional[ad.AnnData]:
    """
    Try to read a generic HDF5 file as AnnData.
    Attempts multiple reading strategies.
    """
    # Strategy 1: Try as h5ad
    try:
        adata = ad.read_h5ad(file_path)
        logger.info(f"  Loaded as h5ad: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    except Exception:
        pass
    
    # Strategy 2: Try reading 10x CellRanger HDF5 format directly with h5py
    try:
        with h5py.File(file_path, "r") as f:
            keys = list(f.keys())
            logger.info(f"  H5 top-level keys: {keys}")
            
            # 10x CellRanger v3 format: "matrix" group
            # 10x CellRanger v2 format: genome name group (e.g., "GRCh38", "mm10")
            group = None
            for key in keys:
                if isinstance(f[key], h5py.Group):
                    g = f[key]
                    if "data" in g and "indices" in g and "indptr" in g:
                        group = g
                        break
            
            if group is not None:
                data = group["data"][:]
                indices = group["indices"][:]
                indptr = group["indptr"][:]
                shape = tuple(group["shape"][:]) if "shape" in group else None
                
                if shape:
                    # 10x stores as genes x cells (CSC), need cells x genes
                    X = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape).T
                    
                    # Extract barcodes (cell names)
                    barcodes = None
                    if "barcodes" in group:
                        barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in group["barcodes"][:]]
                    
                    # Extract gene names
                    gene_names = None
                    gene_ids = None
                    if "features" in group:
                        feat_group = group["features"]
                        if "name" in feat_group:
                            gene_names = [g.decode() if isinstance(g, bytes) else str(g) for g in feat_group["name"][:]]
                        if "id" in feat_group:
                            gene_ids = [g.decode() if isinstance(g, bytes) else str(g) for g in feat_group["id"][:]]
                    elif "gene_names" in group:
                        gene_names = [g.decode() if isinstance(g, bytes) else str(g) for g in group["gene_names"][:]]
                    if "genes" in group and gene_ids is None:
                        gene_ids = [g.decode() if isinstance(g, bytes) else str(g) for g in group["genes"][:]]
                    
                    adata = ad.AnnData(X=X)
                    
                    if barcodes and len(barcodes) == X.shape[0]:
                        adata.obs_names = barcodes
                    if gene_names and len(gene_names) == X.shape[1]:
                        adata.var_names = gene_names
                    elif gene_ids and len(gene_ids) == X.shape[1]:
                        adata.var_names = gene_ids
                    if gene_ids and len(gene_ids) == X.shape[1]:
                        adata.var["gene_ids"] = gene_ids
                    
                    adata.var_names_make_unique()
                    logger.info(f"  Loaded 10x H5 (h5py fallback): {adata.shape[0]} cells × {adata.shape[1]} genes")
                    return adata
            
            # Generic dense matrix
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    if len(data.shape) == 2:
                        adata = ad.AnnData(X=data)
                        logger.info(f"  Loaded H5 matrix '{key}': {adata.shape}")
                        return adata
    except Exception as e:
        logger.error(f"  Failed to parse H5 {file_path}: {e}")
    
    return None


def convert_mtx_triplet(
    matrix_path: str,
    barcodes_path: str,
    features_path: str,
) -> Optional[ad.AnnData]:
    """Convert 10x MTX triplet to AnnData."""
    try:
        import scanpy as sc
        
        # scanpy can read mtx directory directly
        mtx_dir = os.path.dirname(matrix_path)
        
        # Check if files need decompression
        # scanpy.read_10x_mtx handles .gz automatically
        try:
            adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
        except Exception:
            # Fallback: try with gene_ids
            try:
                adata = sc.read_10x_mtx(mtx_dir, var_names="gene_ids", make_unique=True)
            except Exception:
                # Manual fallback
                return _manual_mtx_read(matrix_path, barcodes_path, features_path)
        
        logger.info(f"  Loaded MTX triplet: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
        
    except Exception as e:
        logger.error(f"  Failed to read MTX triplet: {e}")
        return _manual_mtx_read(matrix_path, barcodes_path, features_path)


def _manual_mtx_read(
    matrix_path: str,
    barcodes_path: str,
    features_path: str,
) -> Optional[ad.AnnData]:
    """Manually read MTX triplet when scanpy fails."""
    try:
        # Read matrix
        if matrix_path.endswith(".gz"):
            import gzip
            with gzip.open(matrix_path, "rb") as f:
                matrix = scipy.io.mmread(f)
        else:
            matrix = scipy.io.mmread(matrix_path)
        
        matrix = scipy.sparse.csr_matrix(matrix.T)  # Transpose: genes x cells -> cells x genes
        
        # Read barcodes
        open_func = gzip.open if barcodes_path.endswith(".gz") else open
        with open_func(barcodes_path, "rt") as f:
            barcodes = [line.strip().split("\t")[0] for line in f]
        
        # Read features/genes
        open_func = gzip.open if features_path.endswith(".gz") else open
        with open_func(features_path, "rt") as f:
            features = []
            gene_ids = []
            for line in f:
                parts = line.strip().split("\t")
                gene_ids.append(parts[0])
                features.append(parts[1] if len(parts) > 1 else parts[0])
        
        adata = ad.AnnData(X=matrix)
        adata.obs_names = barcodes[:matrix.shape[0]]
        adata.var_names = features[:matrix.shape[1]]
        adata.var["gene_ids"] = gene_ids[:matrix.shape[1]]
        adata.var_names_make_unique()
        
        logger.info(f"  Loaded MTX (manual): {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
        
    except Exception as e:
        logger.error(f"  Manual MTX read failed: {e}")
        return None


def convert_csv(file_path: str) -> Optional[ad.AnnData]:
    """Convert CSV count matrix to AnnData."""
    try:
        # Detect separator and header
        open_func = gzip.open if file_path.endswith(".gz") else open
        
        with open_func(file_path, "rt") as f:
            first_lines = [f.readline() for _ in range(3)]
        
        # Auto-detect separator
        sep = ","
        if "\t" in first_lines[0]:
            sep = "\t"
        elif ";" in first_lines[0]:
            sep = ";"
        
        # Read with pandas
        df = pd.read_csv(file_path, sep=sep, index_col=0, nrows=5)
        
        # Check orientation: if more columns than rows, it might be genes x cells
        logger.info(f"  CSV preview: {df.shape} (first 5 rows)")
        
        # Full read
        df = pd.read_csv(file_path, sep=sep, index_col=0)
        
        # Heuristic: if columns >> rows, transpose (genes x cells -> cells x genes)
        if df.shape[1] > df.shape[0] * 5:
            logger.info(f"  Transposing matrix (detected genes × cells format)")
            df = df.T
        
        adata = ad.AnnData(X=scipy.sparse.csr_matrix(df.values.astype(np.float32)))
        adata.obs_names = list(df.index.astype(str))
        adata.var_names = list(df.columns.astype(str))
        adata.var_names_make_unique()
        
        logger.info(f"  Loaded CSV: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
        
    except Exception as e:
        logger.error(f"  Failed to read CSV {file_path}: {e}")
        return None


def convert_tsv(file_path: str) -> Optional[ad.AnnData]:
    """Convert TSV/TXT count matrix to AnnData. Uses same logic as CSV."""
    # TSV is handled the same way, pandas auto-detects
    return convert_csv(file_path)


def convert_loom(file_path: str) -> Optional[ad.AnnData]:
    """Convert Loom file to AnnData."""
    try:
        import scanpy as sc
        adata = sc.read_loom(file_path)
        adata.var_names_make_unique()
        logger.info(f"  Loaded Loom: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    except Exception as e:
        logger.error(f"  Failed to read Loom {file_path}: {e}")
        return None


def convert_h5ad(file_path: str) -> Optional[ad.AnnData]:
    """Load existing h5ad file (validation)."""
    try:
        adata = ad.read_h5ad(file_path)
        logger.info(f"  Loaded h5ad: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    except Exception as e:
        logger.error(f"  Failed to read h5ad {file_path}: {e}")
        return None


def convert_h5ad_gz(file_path: str) -> Optional[ad.AnnData]:
    """Decompress and load .h5ad.gz file."""
    try:
        decompressed = file_path[:-3]
        if not os.path.exists(decompressed):
            with gzip.open(file_path, "rb") as f_in:
                with open(decompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return convert_h5ad(decompressed)
    except Exception as e:
        logger.error(f"  Failed to decompress h5ad.gz {file_path}: {e}")
        return None


# ============================================================
# Quality Control (Optional)
# ============================================================
def apply_qc_filters(adata: ad.AnnData, config: dict) -> ad.AnnData:
    """
    Apply basic QC filters to AnnData object.
    
    Only applied if config['apply_qc'] is True.
    """
    if not config.get("apply_qc", False):
        return adata
    
    import scanpy as sc
    
    n_before = adata.n_obs
    
    min_genes = config.get("min_genes", 200)
    min_cells = config.get("min_cells", 3)
    max_mito_pct = config.get("max_mito_pct", 20)
    
    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Mitochondrial gene filtering
    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    
    if "pct_counts_mt" in adata.obs:
        adata = adata[adata.obs["pct_counts_mt"] < max_mito_pct, :].copy()
    
    n_after = adata.n_obs
    logger.info(f"  QC: {n_before} → {n_after} cells ({n_before - n_after} removed)")
    
    return adata


# ============================================================
# Main Conversion Pipeline
# ============================================================
# Dispatch table for conversion functions
CONVERTERS = {
    "h5ad": convert_h5ad,
    "h5ad_gz": convert_h5ad_gz,
    "h5_10x": convert_10x_h5,
    "h5_other": convert_generic_h5,
    "csv": convert_csv,
    "tsv": convert_tsv,
    "loom": convert_loom,
}


def convert_single_file(data_info: dict, qc_config: dict = None) -> Optional[ad.AnnData]:
    """
    Convert a single data source (file or directory) to AnnData.
    
    Args:
        data_info: Dict from scan_directory_for_data
        qc_config: Optional QC configuration
    
    Returns:
        AnnData object or None
    """
    fmt = data_info["format"]
    path = data_info["path"]
    
    logger.info(f"  Converting: {os.path.basename(path)} (format: {fmt})")
    
    adata = None
    
    if fmt == "mtx_triplet":
        triplet = data_info["mtx_triplet"]
        adata = convert_mtx_triplet(*triplet)
    elif fmt in CONVERTERS:
        adata = CONVERTERS[fmt](path)
    elif fmt == "tar":
        # Extract and recurse
        extract_dir = path + "_extracted"
        if not os.path.exists(extract_dir):
            try:
                with tarfile.open(path, "r:*") as tar:
                    tar.extractall(extract_dir)
            except Exception as e:
                logger.error(f"  Failed to extract tar: {e}")
                return None
        
        # Scan extracted files
        sub_files = scan_directory_for_data(extract_dir)
        if sub_files:
            # Convert the first valid file found
            for sf in sub_files:
                adata = convert_single_file(sf, qc_config)
                if adata is not None:
                    break
    else:
        logger.warning(f"  Unsupported format: {fmt}")
    
    # Apply QC if configured
    if adata is not None and qc_config:
        adata = apply_qc_filters(adata, qc_config)
    
    return adata


def convert_gse_directory(
    gse_dir: str,
    output_dir: str,
    qc_config: dict = None,
    merge_samples: bool = False,
) -> List[str]:
    """
    Convert all data files in a GSE download directory to h5ad.
    
    Args:
        gse_dir: Path to the downloaded GSE directory
        output_dir: Output directory for h5ad files
        qc_config: QC configuration
        merge_samples: Whether to merge multiple samples
    
    Returns:
        List of output h5ad file paths
    """
    gse_id = os.path.basename(gse_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\nConverting {gse_id}...")
    
    # Scan for data files
    data_files = scan_directory_for_data(gse_dir)
    
    if not data_files:
        logger.warning(f"  No convertible files found in {gse_dir}")
        return []
    
    logger.info(f"  Found {len(data_files)} data sources")
    
    output_paths = []
    adatas = []
    
    for i, data_info in enumerate(data_files):
        adata = convert_single_file(data_info, qc_config)
        
        if adata is not None:
            if merge_samples:
                # Tag with source info
                source_name = os.path.basename(data_info["path"])
                adata.obs["source_file"] = source_name
                adatas.append(adata)
            else:
                # Save individually
                if len(data_files) == 1:
                    out_name = f"{gse_id}.h5ad"
                else:
                    source = os.path.basename(data_info["path"])
                    source_clean = re.sub(r'\.(h5|h5ad|mtx|csv|tsv|txt|loom|gz|tar)+$', '', source, flags=re.IGNORECASE)
                    out_name = f"{gse_id}_{source_clean}.h5ad"
                
                out_path = os.path.join(output_dir, out_name)
                adata.write_h5ad(out_path)
                logger.info(f"  Saved: {out_path} ({adata.shape[0]} cells × {adata.shape[1]} genes)")
                output_paths.append(out_path)
    
    # Merge if requested
    if merge_samples and adatas:
        logger.info(f"  Merging {len(adatas)} samples...")
        try:
            merged = ad.concat(adatas, join="outer", label="sample")
            out_path = os.path.join(output_dir, f"{gse_id}_merged.h5ad")
            merged.write_h5ad(out_path)
            logger.info(f"  Saved merged: {out_path} ({merged.shape[0]} cells × {merged.shape[1]} genes)")
            output_paths.append(out_path)
        except Exception as e:
            logger.error(f"  Failed to merge: {e}")
            # Save individually as fallback
            for j, adata in enumerate(adatas):
                out_path = os.path.join(output_dir, f"{gse_id}_sample{j}.h5ad")
                adata.write_h5ad(out_path)
                output_paths.append(out_path)
    
    return output_paths


def convert_all(
    download_dir: str,
    output_dir: str,
    config: dict,
) -> dict:
    """
    Convert all downloaded GSE directories to h5ad.
    
    Args:
        download_dir: Base download directory
        output_dir: Output directory for h5ad files
        config: Convert configuration from config.yaml
    
    Returns:
        Summary dict
    """
    qc_config = config if config.get("apply_qc", False) else None
    merge_samples = config.get("merge_samples", False)
    
    # Find all GSE directories
    gse_dirs = []
    for item in sorted(os.listdir(download_dir)):
        item_path = os.path.join(download_dir, item)
        if os.path.isdir(item_path) and item.startswith("GSE"):
            gse_dirs.append(item_path)
    
    if not gse_dirs:
        logger.warning(f"No GSE directories found in {download_dir}")
        return {"total": 0, "success": 0, "failed": 0, "outputs": []}
    
    logger.info(f"Found {len(gse_dirs)} GSE directories to convert")
    
    all_outputs = []
    success = 0
    failed = 0
    
    for gse_dir in gse_dirs:
        outputs = convert_gse_directory(gse_dir, output_dir, qc_config, merge_samples)
        
        if outputs:
            all_outputs.extend(outputs)
            success += 1
        else:
            failed += 1
    
    # Summary
    summary = {
        "total": len(gse_dirs),
        "success": success,
        "failed": failed,
        "outputs": all_outputs,
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Conversion Summary: {success}/{len(gse_dirs)} successful")
    logger.info(f"Total h5ad files: {len(all_outputs)}")
    logger.info(f"{'='*60}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "conversion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Convert downloaded GEO files to h5ad")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--input-dir", default=None, help="Override input directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    convert_config = config.get("convert", {})
    
    input_dir = args.input_dir or convert_config.get("input_dir", "./downloads")
    output_dir = args.output_dir or convert_config.get("output_dir", "./h5ad_output")
    
    # Run conversion
    summary = convert_all(input_dir, output_dir, convert_config)
    
    return summary


if __name__ == "__main__":
    main()
