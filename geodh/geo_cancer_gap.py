"""
GEO-DataHub: Cancer Organ Gap Scan Module
=========================================
Find missing cancer-organ coverage and suggest downloadable candidates.
"""

import os
import json
import time
import logging
from typing import Dict, List, Set

from .geo_search import search_geo, fetch_supplementary_file_list, _fetch_gse_metadata, RATE_LIMIT_DELAY
from .geo_classifier import (
    classify_dataset_files,
    classify_domain,
    classify_modality_from_metadata,
    filter_files_by_format,
)
from .geo_theme import infer_organ_from_text, ORGAN_KEYWORDS

logger = logging.getLogger("geo_cancer_gap")


DEFAULT_TARGET_ORGANS = list(ORGAN_KEYWORDS.keys())


def _collect_covered_organs(downloads_dir: str) -> Set[str]:
    covered = set()
    for item in sorted(os.listdir(downloads_dir)):
        gse_dir = os.path.join(downloads_dir, item)
        if not (item.startswith("GSE") and os.path.isdir(gse_dir)):
            continue

        meta_path = os.path.join(gse_dir, "dataset_meta.json")
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            continue

        geo = meta.get("geo_metadata", {}) if isinstance(meta, dict) else {}
        text = " ".join([
            str(geo.get("title", "")),
            str(geo.get("summary", "")),
            str(geo.get("overall_design", "")),
        ])
        organ = infer_organ_from_text(text)
        if organ != "unknown":
            covered.add(organ)

    return covered


def scan_cancer_organ_gaps(
    downloads_dir: str,
    format_filters: List[str],
    modality: str = "rna",
    organism: str = "human",
    max_results_per_organ: int = 12,
    max_candidates_per_organ: int = 5,
) -> dict:
    """Scan missing cancer organs and suggest candidate GSE datasets."""
    covered_organs = _collect_covered_organs(downloads_dir)
    targets = set(DEFAULT_TARGET_ORGANS)
    missing_organs = sorted(targets - covered_organs)

    candidates_by_organ: Dict[str, List[dict]] = {}

    for organ in missing_organs:
        query = f"single cell {organ} cancer"
        cfg = {
            "query": query,
            "organism": "Homo sapiens" if organism.lower() == "human" else organism,
            "dataset_type": "",
            "max_results": max_results_per_organ,
            "min_samples": 1,
        }

        gse_list = search_geo(cfg)
        organ_candidates = []

        for gse in gse_list:
            try:
                if not gse.summary and not gse.title:
                    meta = _fetch_gse_metadata(gse.gse_id)
                    gse.title = meta.get("title", "")
                    gse.summary = meta.get("summary", "")
                    gse.overall_design = meta.get("overall_design", "")
                    gse.organism = meta.get("organism", "")
                    gse.series_type = meta.get("series_type", "")

                if not gse.supplementary_files:
                    gse.supplementary_files = fetch_supplementary_file_list(gse.gse_id)

                domain, _ = classify_domain(gse.title, gse.summary, gse.overall_design)
                if domain != "cancer":
                    continue

                modalities = classify_modality_from_metadata(gse.title, gse.summary, gse.series_type, gse.overall_design)
                if modality and modality not in modalities and not (
                    modality == "multiome" and ("rna" in modalities and "atac" in modalities)
                ):
                    continue

                dc = classify_dataset_files(gse.gse_id, gse.supplementary_files)
                filtered_files = filter_files_by_format(dc.file_classifications, format_filters)
                if not filtered_files:
                    continue

                organ_candidates.append(
                    {
                        "gse_id": gse.gse_id,
                        "title": gse.title,
                        "organism": gse.organism,
                        "available_formats": dc.available_formats_str(),
                        "matched_files": [f.filename for f in filtered_files[:10]],
                        "n_matched_files": len(filtered_files),
                    }
                )

                if len(organ_candidates) >= max_candidates_per_organ:
                    break

                time.sleep(RATE_LIMIT_DELAY)
            except Exception as exc:
                logger.warning("Gap scan failed for %s in organ=%s: %s", gse.gse_id, organ, exc)

        candidates_by_organ[organ] = organ_candidates

    return {
        "covered_organs": sorted(covered_organs),
        "target_organs": sorted(targets),
        "missing_organs": missing_organs,
        "format_filters": format_filters,
        "modality": modality,
        "organism": organism,
        "candidates_by_organ": candidates_by_organ,
    }
