"""
GEO-DataHub: Theme Organization Module
======================================
Create isolated, theme-oriented views of downloaded datasets.
Default behavior is symlink-based to avoid data duplication.
"""

import os
import json
import shutil
import logging
from typing import Dict, List, Tuple

from .geo_classifier import classify_domain

logger = logging.getLogger("geo_theme")


ORGAN_KEYWORDS: Dict[str, List[str]] = {
    "lung": ["lung", "pulmonary"],
    "breast": ["breast", "mammary"],
    "colon": ["colon", "colorectal", "rectal", "intestine", "intestinal"],
    "liver": ["liver", "hepatic", "hepatocellular"],
    "brain": ["brain", "glioma", "glioblastoma", "astrocytoma", "cns"],
    "pancreas": ["pancreas", "pancreatic"],
    "gastric": ["gastric", "stomach"],
    "hematologic": ["leukemia", "lymphoma", "myeloma", "bone marrow", "hematologic", "aml", "cll"],
    "prostate": ["prostate"],
    "ovary": ["ovary", "ovarian"],
    "kidney": ["kidney", "renal"],
    "bladder": ["bladder", "urothelial"],
    "skin": ["skin", "melanoma"],
    "head_neck": ["head and neck", "head-neck", "hnscc", "nasopharyngeal", "oral squamous"],
    "esophagus": ["esophagus", "esophageal"],
    "thyroid": ["thyroid"],
    "cervix": ["cervical", "cervix"],
    "uterus": ["uterine", "endometrial", "endometrium"],
}


def infer_organ_from_text(text: str) -> str:
    text_low = (text or "").lower()
    for organ, keywords in ORGAN_KEYWORDS.items():
        if any(keyword in text_low for keyword in keywords):
            return organ
    return "unknown"


def _read_dataset_meta(gse_dir: str) -> dict:
    meta_path = os.path.join(gse_dir, "dataset_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _infer_domain_and_organ(meta: dict) -> Tuple[str, str]:
    geo = meta.get("geo_metadata", {}) if isinstance(meta, dict) else {}
    text = " ".join([
        str(geo.get("title", "")),
        str(geo.get("summary", "")),
        str(geo.get("overall_design", "")),
    ])

    domain, _ = classify_domain(
        str(geo.get("title", "")),
        str(geo.get("summary", "")),
        str(geo.get("overall_design", "")),
    )
    organ = infer_organ_from_text(text)

    return domain or "unknown", organ


def organize_downloads_by_theme(
    downloads_dir: str,
    thematic_root: str,
    mode: str = "symlink",
) -> dict:
    """
    Organize downloads under thematic root:
      {thematic_root}/{domain}/{organ}/{GSE}/

    mode:
      - symlink (default, recommended)
      - copy
      - move
    """
    if mode not in {"symlink", "copy", "move"}:
        raise ValueError("mode must be one of: symlink, copy, move")

    os.makedirs(thematic_root, exist_ok=True)

    gse_dirs = [
        os.path.join(downloads_dir, item)
        for item in sorted(os.listdir(downloads_dir))
        if item.startswith("GSE") and os.path.isdir(os.path.join(downloads_dir, item))
    ]

    linked = 0
    skipped = 0
    entries = []

    for gse_dir in gse_dirs:
        gse_id = os.path.basename(gse_dir)
        meta = _read_dataset_meta(gse_dir)
        domain, organ = _infer_domain_and_organ(meta)

        target_parent = os.path.join(thematic_root, domain, organ)
        os.makedirs(target_parent, exist_ok=True)
        target = os.path.join(target_parent, gse_id)

        if os.path.exists(target):
            skipped += 1
            entries.append({"gse_id": gse_id, "domain": domain, "organ": organ, "target": target, "status": "exists"})
            continue

        if mode == "symlink":
            os.symlink(os.path.abspath(gse_dir), target)
        elif mode == "copy":
            shutil.copytree(gse_dir, target)
        elif mode == "move":
            shutil.move(gse_dir, target)

        linked += 1
        entries.append({"gse_id": gse_id, "domain": domain, "organ": organ, "target": target, "status": mode})

    summary = {
        "total_gse": len(gse_dirs),
        "created": linked,
        "skipped": skipped,
        "mode": mode,
        "thematic_root": os.path.abspath(thematic_root),
        "entries": entries,
    }

    report_path = os.path.join(thematic_root, "theme_organization_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logger.info("Theme organization report saved: %s", report_path)
    return summary
