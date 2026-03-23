"""
GEO-DataHub: Metadata & Manifest Module
=======================================
Isolated utilities for:
  - Existing GSE discovery / dedup checks
  - Canonical naming helpers
  - Unified per-GSE metadata JSON writing
  - h5ad output selection helpers
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List

from .geo_search import _fetch_gse_metadata, fetch_supplementary_file_list
from .geo_classifier import classify_file

logger = logging.getLogger("geo_manifest")


def scan_existing_gse_locations(roots: List[str]) -> Dict[str, List[str]]:
    """Scan local roots and map GSE accession -> existing directory paths."""
    gse_map: Dict[str, List[str]] = {}
    gse_pat = re.compile(r"(GSE\d+)", re.IGNORECASE)

    for root in roots:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            continue
        for dirpath, _, _ in os.walk(root_abs):
            base = os.path.basename(dirpath)
            m = gse_pat.search(base)
            if not m:
                continue
            gse_id = m.group(1).upper()
            gse_map.setdefault(gse_id, []).append(dirpath)

    return gse_map


def should_skip_gse_download(gse_id: str, target_gse_dir: str, existing_gse_map: Dict[str, List[str]]) -> bool:
    """Skip if GSE exists in local roots outside target GSE directory."""
    existing_paths = existing_gse_map.get(gse_id.upper(), [])
    if not existing_paths:
        return False

    target_abs = os.path.abspath(target_gse_dir)
    for path in existing_paths:
        path_abs = os.path.abspath(path)
        if path_abs == target_abs:
            continue
        if path_abs.startswith(target_abs + os.sep):
            continue
        if target_abs.startswith(path_abs + os.sep):
            continue
        return True
    return False


def extract_gsm_id(filename: str) -> str:
    """Extract GSM accession from filename, else NA."""
    m = re.search(r"(GSM\d+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else "NA"


def infer_modality_from_filename(filename: str) -> str:
    """Infer modality tag from filename for metadata readability."""
    name = filename.lower()
    if "peak_bc_matrix" in name or "atac" in name or "fragments" in name:
        return "atac"
    if "feature_bc_matrix" in name or "rna" in name or "gene" in name:
        return "rna"
    if "multiome" in name or "arc" in name:
        return "multiome"
    return "unknown"


def build_canonical_name(gse_id: str, filename: str) -> str:
    """Build a canonical readable name (manifest only, no file rename)."""
    gsm_id = extract_gsm_id(filename)
    modality = infer_modality_from_filename(filename)
    stem, ext = os.path.splitext(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"{gse_id.upper()}__{gsm_id}__{modality}__{stem}{ext}"


def select_best_h5ad_by_gse(output_paths: List[str]) -> Dict[str, str]:
    """Select one canonical h5ad path for each GSE from conversion outputs."""
    by_gse: Dict[str, List[str]] = {}

    for path in output_paths:
        base = os.path.basename(path)
        m = re.match(r"^(GSE\d+)", base, re.IGNORECASE)
        if not m:
            continue
        gse = m.group(1).upper()
        by_gse.setdefault(gse, []).append(path)

    selected: Dict[str, str] = {}
    for gse, paths in by_gse.items():
        exact = [p for p in paths if os.path.basename(p).lower() == f"{gse.lower()}.h5ad"]
        merged = [p for p in paths if os.path.basename(p).lower() == f"{gse.lower()}_merged.h5ad"]
        selected[gse] = exact[0] if exact else (merged[0] if merged else paths[0])

    return selected


def write_dataset_meta_json(gse_dir: str, gse_id: str, include_geo_meta: bool = True) -> str:
    """Write standardized per-GSE JSON metadata with unified schema."""
    gse_id = gse_id.upper()

    geo_meta = {
        "title": "",
        "summary": "",
        "overall_design": "",
        "organism": "",
        "series_type": "",
    }

    remote_files: List[str] = []
    remote_by_name: Dict[str, List[str]] = {}

    if include_geo_meta:
        try:
            meta = _fetch_gse_metadata(gse_id)
            geo_meta.update(
                {
                    "title": meta.get("title", ""),
                    "summary": meta.get("summary", ""),
                    "overall_design": meta.get("overall_design", ""),
                    "organism": meta.get("organism", ""),
                    "series_type": meta.get("series_type", ""),
                }
            )
        except Exception as exc:
            logger.warning(f"Failed to fetch GEO metadata for {gse_id}: {exc}")

        try:
            remote_files = fetch_supplementary_file_list(gse_id)
            for url in remote_files:
                name = url.rstrip("/").split("/")[-1]
                remote_by_name.setdefault(name, []).append(url)
        except Exception as exc:
            logger.warning(f"Failed to fetch supplementary list for {gse_id}: {exc}")

    rows = []
    for name in sorted(os.listdir(gse_dir)):
        local_path = os.path.join(gse_dir, name)
        if not os.path.isfile(local_path):
            continue
        if name in {
            "download_manifest.json",
            "download_manifest_standard.tsv",
            "download_manifest.csv",
            "dataset_meta.json",
        }:
            continue

        file_clf = classify_file(name)
        rows.append(
            {
                "gse_id": gse_id,
                "gsm_id": extract_gsm_id(name),
                "modality": infer_modality_from_filename(name),
                "file_name": name,
                "canonical_name": build_canonical_name(gse_id, name),
                "size_bytes": os.path.getsize(local_path),
                "format_type": file_clf.format_type,
                "processing_level": file_clf.processing_level,
                "is_count_matrix": file_clf.is_count_matrix,
                "source_url": (remote_by_name.get(name, [""])[0] if remote_by_name.get(name) else ""),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gse_id": gse_id,
        "geo_metadata": geo_meta,
        "summary": {
            "total_local_files": len(rows),
            "total_remote_files": len(remote_files),
            "count_matrix_files": sum(1 for r in rows if r.get("is_count_matrix")),
            "has_h5ad": any(str(r.get("file_name", "")).lower().endswith(".h5ad") for r in rows),
            "has_h5_or_mtx": any(
                (".h5" in str(r.get("file_name", "")).lower())
                or ("matrix.mtx" in str(r.get("file_name", "")).lower())
                for r in rows
            ),
        },
        "files": rows,
    }

    manifest_path = os.path.join(gse_dir, "dataset_meta.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    # Remove old mixed-format manifests
    for legacy in ["download_manifest_standard.tsv", "download_manifest.csv"]:
        legacy_path = os.path.join(gse_dir, legacy)
        if os.path.exists(legacy_path):
            try:
                os.remove(legacy_path)
            except OSError:
                logger.warning(f"Could not remove legacy manifest: {legacy_path}")

    return manifest_path
