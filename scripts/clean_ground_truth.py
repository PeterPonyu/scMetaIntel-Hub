#!/usr/bin/env python3
"""One-time cleanup of ground truth JSON files.

Applies the same cleaning pipeline from evaluate.py directly to the source
ground truth files on disk, so downstream benchmarks don't need runtime cleaning.

Changes applied per file:
1. Remove non-disease terms from diseases (healthy, control, normal, etc.)
2. Remove abbreviations/timepoints/numbers from diseases
3. Move disease terms from tissues to diseases
4. Remove cell line entries and sample barcodes from tissues
5. Extract diseases from title+summary when diseases list is empty
6. Extract cell types from title+summary when cell_types list is empty
7. Fix encoding issues (mojibake)
8. Normalize case for known diseases
"""

import json
import glob
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scmetaintel.evaluate import (
    clean_tissue_list,
    clean_disease_list,
    extract_diseases_from_text,
    extract_cell_types_from_text,
)


def fix_encoding(text: str) -> str:
    """Fix common mojibake patterns."""
    replacements = {
        "\xe2\x80\x99".encode().decode("latin-1"): "'",
        "naieve": "naive",
        "helathy": "healthy",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def clean_file(path: Path) -> dict:
    """Clean a single ground truth JSON file. Returns change summary."""
    with open(path) as f:
        doc = json.load(f)

    changes = {}
    title = doc.get("title", "")
    summary = doc.get("summary", "")

    # Fix encoding in title/summary
    new_title = fix_encoding(title)
    new_summary = fix_encoding(summary)
    if new_title != title:
        doc["title"] = new_title
        changes["title_encoding"] = True
    if new_summary != summary:
        doc["summary"] = new_summary
        changes["summary_encoding"] = True

    # Get original field values
    orig_tissues = list(doc.get("tissues", []) or [])
    orig_diseases = list(doc.get("diseases", []) or [])
    orig_cell_types = list(doc.get("cell_types", []) or [])

    # Step 1: Clean tissues — remove cell lines, sample barcodes, move disease terms
    diseases = list(orig_diseases)
    clean_t, diseases = clean_tissue_list(list(orig_tissues), diseases)

    # Step 2: Clean diseases — remove non-disease terms
    clean_d = clean_disease_list(diseases)

    # Step 3: Extract diseases from text if empty
    clean_d = extract_diseases_from_text(
        doc.get("title", ""), doc.get("summary", ""), clean_d
    )

    # Step 4: Extract cell types from text if empty
    clean_ct = extract_cell_types_from_text(
        doc.get("title", ""), doc.get("summary", ""), list(orig_cell_types)
    )

    # Track changes
    if clean_t != orig_tissues:
        changes["tissues"] = f"{len(orig_tissues)} -> {len(clean_t)}"
    if clean_d != orig_diseases:
        changes["diseases"] = f"{len(orig_diseases)} -> {len(clean_d)}"
    if clean_ct != orig_cell_types:
        changes["cell_types"] = f"{len(orig_cell_types)} -> {len(clean_ct)}"

    if changes:
        doc["tissues"] = clean_t
        doc["diseases"] = clean_d
        doc["cell_types"] = clean_ct
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return changes


def main():
    gt_dir = PROJECT_ROOT / "benchmarks" / "ground_truth"
    files = sorted(gt_dir.glob("GSE*.json"))
    print(f"Found {len(files)} ground truth files")

    total_changed = 0
    tissue_changed = 0
    disease_changed = 0
    celltype_changed = 0
    encoding_fixed = 0

    for path in files:
        changes = clean_file(path)
        if changes:
            total_changed += 1
            if "tissues" in changes:
                tissue_changed += 1
            if "diseases" in changes:
                disease_changed += 1
            if "cell_types" in changes:
                celltype_changed += 1
            if "title_encoding" in changes or "summary_encoding" in changes:
                encoding_fixed += 1

    print(f"\nResults:")
    print(f"  Files modified: {total_changed}/{len(files)}")
    print(f"  Tissues cleaned: {tissue_changed}")
    print(f"  Diseases cleaned: {disease_changed}")
    print(f"  Cell types enriched: {celltype_changed}")
    print(f"  Encoding fixed: {encoding_fixed}")

    # Verify: recount field coverage
    n_tissues = n_diseases = n_cell_types = 0
    for path in files:
        doc = json.load(open(path))
        if doc.get("tissues"):
            n_tissues += 1
        if doc.get("diseases"):
            n_diseases += 1
        if doc.get("cell_types"):
            n_cell_types += 1

    print(f"\nPost-cleanup coverage:")
    print(f"  Tissues: {n_tissues}/{len(files)} ({100*n_tissues/len(files):.1f}%)")
    print(f"  Diseases: {n_diseases}/{len(files)} ({100*n_diseases/len(files):.1f}%)")
    print(f"  Cell types: {n_cell_types}/{len(files)} ({100*n_cell_types/len(files):.1f}%)")


if __name__ == "__main__":
    main()
