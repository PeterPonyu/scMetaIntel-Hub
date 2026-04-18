"""
Metadata enrichment for scMetaIntel-Hub.

This merges:
- the dataclass-rich enrichment flow from standalone scMetaIntel
- the benchmark-friendly `build_enriched_document()` / `enrich_all()` helpers
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import requests

from .config import get_config
from .models import CharacteristicsSummary, EnrichedStudy, PubMedInfo, SampleMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TISSUE_KEYS = {"tissue", "tissue type", "organ", "tissue/organ", "cell source",
               "anatomical site", "body site", "sample source", "tissue source"}
DISEASE_KEYS = {"disease", "disease state", "disease status", "diagnosis",
                "condition", "pathology", "disease type", "clinical diagnosis",
                "health status"}
CELL_TYPE_KEYS = {"cell type", "cell_type", "celltype", "cell population",
                  "cell lineage", "sorted population", "facs gate"}
TREATMENT_KEYS = {"treatment", "drug", "perturbation", "stimulus", "compound",
                  "intervention", "treated with"}
SEX_KEYS = {"sex", "gender"}
AGE_KEYS = {"age", "age at diagnosis", "age (years)", "donor age"}
DONOR_KEYS = {"donor", "patient", "subject", "individual", "donor id", "patient id", "subject id"}

SERIES_TYPE_TO_MODALITY = {
    "expression profiling by high throughput sequencing": "scRNA-seq",
    "non-coding rna profiling by high throughput sequencing": "ncRNA-seq",
    "genome binding/occupancy profiling by high throughput sequencing": "ChIP-seq/ATAC-seq",
    "methylation profiling by high throughput sequencing": "Bisulfite-seq",
    "snp genotyping by high throughput sequencing": "WGS/WES",
    "other": "other",
}


# ---------------------------------------------------------------------------
# GEO / PubMed fetching
# ---------------------------------------------------------------------------


def fetch_series_soft(gse_id: str) -> str:
    cfg = get_config()
    params = {"acc": gse_id, "targ": "self", "view": "brief", "form": "text"}
    resp = requests.get(cfg.services.geo_query_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def fetch_geo_soft(gse_id: str) -> str:
    cfg = get_config()
    params = {"acc": gse_id, "targ": "gsm", "view": "brief", "form": "text"}
    resp = requests.get(cfg.services.geo_query_url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_series_soft(soft_text: str) -> dict:
    info = {
        "title": "",
        "summary": "",
        "overall_design": "",
        "organism": "",
        "series_type": "",
        "platform": "",
        "pubmed_ids": [],
        "submission_date": "",
        "n_samples": 0,
    }
    for line in soft_text.splitlines():
        line = line.strip()
        if line.startswith("!Series_title"):
            info["title"] = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_summary"):
            val = line.split("=", 1)[-1].strip()
            info["summary"] = (info["summary"] + " " + val).strip() if info["summary"] else val
        elif line.startswith("!Series_overall_design"):
            val = line.split("=", 1)[-1].strip()
            info["overall_design"] = (info["overall_design"] + " " + val).strip() if info["overall_design"] else val
        elif line.startswith("!Series_sample_organism"):
            info["organism"] = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_type"):
            val = line.split("=", 1)[-1].strip()
            info["series_type"] = (info["series_type"] + "; " + val).strip("; ") if info["series_type"] else val
        elif line.startswith("!Series_platform_id"):
            info["platform"] = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_pubmed_id"):
            pmid = line.split("=", 1)[-1].strip()
            if pmid and pmid not in info["pubmed_ids"]:
                info["pubmed_ids"].append(pmid)
        elif line.startswith("!Series_submission_date"):
            info["submission_date"] = line.split("=", 1)[-1].strip()
        elif line.startswith("!Series_sample_count"):
            try:
                info["n_samples"] = int(line.split("=", 1)[-1].strip())
            except ValueError:
                pass
        elif line.startswith("!Series_sample_id"):
            info["n_samples"] += 1
    return info


def parse_sample_soft(soft_text: str) -> dict[str, SampleMeta]:
    samples: dict[str, SampleMeta] = {}
    current_gsm = None
    current = None
    for line in soft_text.splitlines():
        line = line.strip()
        if line.startswith("^SAMPLE"):
            if current_gsm and current:
                samples[current_gsm] = current
            current_gsm = line.split("=", 1)[-1].strip()
            current = SampleMeta(gsm_id=current_gsm)
        elif current is not None:
            if line.startswith("!Sample_title"):
                current.title = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_source_name_ch1"):
                current.source = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_organism_ch1"):
                current.organism = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_platform_id"):
                current.platform = line.split("=", 1)[-1].strip()
            elif line.startswith("!Sample_characteristics_ch1"):
                raw = line.split("=", 1)[-1].strip()
                if ": " in raw:
                    key, val = raw.split(": ", 1)
                    current.characteristics[key.strip().lower()] = val.strip()
                elif "=" in raw and not raw.startswith("!"):
                    key, val = raw.split("=", 1)
                    current.characteristics[key.strip().lower()] = val.strip()
                else:
                    idx = len([k for k in current.characteristics if k.startswith("attr_")])
                    current.characteristics[f"attr_{idx}"] = raw
    if current_gsm and current:
        samples[current_gsm] = current
    return samples


def fetch_pubmed(pmids: list[str]) -> list[PubMedInfo]:
    if not pmids:
        return []
    cfg = get_config()
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "xml", "retmode": "xml"}
    try:
        resp = requests.get(f"{cfg.services.ncbi_eutils_base}/efetch.fcgi", params=params, timeout=60)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception as e:
        logger.warning(f"PubMed fetch failed for {pmids}: {e}")
        return []

    out = []
    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        title_elem = article.find(".//ArticleTitle")
        pmid = pmid_elem.text if pmid_elem is not None else ""
        title = title_elem.text if title_elem is not None else ""
        abstract_parts = []
        for abs_text in article.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            abstract_parts.append(f"{label}: {text}" if label else text)
        mesh_terms = [m.text for m in article.findall(".//MeshHeading/DescriptorName") if m.text]
        keywords = [k.text for k in article.findall(".//Keyword") if k.text]
        out.append(PubMedInfo(pmid=pmid, title=title or "", abstract=" ".join(abstract_parts), mesh_terms=mesh_terms, keywords=keywords))
    return out


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def summarize_characteristics(samples: dict[str, SampleMeta]) -> CharacteristicsSummary:
    tissues, diseases, cell_types, treatments = set(), set(), set(), set()
    organisms, sexes, donors = set(), set(), set()
    ages = []

    for s in samples.values():
        if s.organism:
            organisms.add(s.organism)
        for key, val in s.characteristics.items():
            k = key.lower().strip()
            v = val.strip()
            if not v or v.lower() in {"na", "n/a", "none", "unknown", "--"}:
                continue
            if k in TISSUE_KEYS:
                tissues.add(v.lower())
            elif k in DISEASE_KEYS:
                diseases.add(v.lower())
            elif k in CELL_TYPE_KEYS:
                cell_types.add(v.lower())
            elif k in TREATMENT_KEYS:
                treatments.add(v.lower())
            elif k in SEX_KEYS:
                sexes.add(v.lower())
            elif k in AGE_KEYS:
                ages.append(v)
            elif k in DONOR_KEYS:
                donors.add(v)
        if s.source and s.source.lower() not in {"na", "n/a", "none", ""}:
            tissues.add(s.source.lower())

    numeric_ages = []
    for a in ages:
        m = re.search(r"(\d+)", a)
        if m:
            numeric_ages.append(int(m.group(1)))
    age_range = f"{min(numeric_ages)}-{max(numeric_ages)}" if numeric_ages else ""

    return CharacteristicsSummary(
        tissues=sorted(tissues),
        diseases=sorted(diseases),
        cell_types=sorted(cell_types),
        treatments=sorted(treatments),
        organisms=sorted(organisms),
        donor_count=len(donors),
        sex=sorted(sexes),
        age_range=age_range,
    )


def derive_domain_modalities(series_type: str, title: str = "", summary: str = "") -> tuple[str, list[str]]:
    modalities = []
    if series_type:
        for part in series_type.split(";"):
            mod = SERIES_TYPE_TO_MODALITY.get(part.strip().lower())
            if mod and mod not in modalities:
                modalities.append(mod)
    text = (title + " " + summary).lower()
    sc_signals = {
        "scrna": "scRNA-seq",
        "single-cell rna": "scRNA-seq",
        "single cell rna": "scRNA-seq",
        "10x genomics": "scRNA-seq",
        "drop-seq": "scRNA-seq",
        "smart-seq": "scRNA-seq",
        "scatac": "scATAC-seq",
        "single-cell atac": "scATAC-seq",
        "single cell atac": "scATAC-seq",
        "cite-seq": "CITE-seq",
        "multiome": "Multiome",
        "spatial transcriptom": "Spatial",
        "visium": "Spatial",
        "merfish": "Spatial",
        "slide-seq": "Spatial",
    }
    for signal, mod in sc_signals.items():
        if signal in text and mod not in modalities:
            modalities.append(mod)

    domain = ""
    if any(k in text for k in ("cancer", "tumor", "tumour", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma", "glioma", "blastoma")):
        domain = "cancer"
    elif any(k in text for k in ("development", "embryo", "fetal", "organogenesis", "differentiation")):
        domain = "development"
    elif any(k in text for k in ("immune", "immunity", "inflammation", "autoimmune")):
        domain = "immunology"
    elif any(k in text for k in ("neurodegen", "alzheimer", "parkinson")):
        domain = "neurodegeneration"
    elif "covid" in text or "sars-cov" in text:
        domain = "infectious_disease"
    return domain, modalities


def load_existing_geo_meta(geo_datahub: Path, gse_id: str) -> dict | None:
    meta_path = geo_datahub / "downloads" / gse_id / "dataset_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def load_gse_ids(index_path: Optional[Path] = None) -> List[str]:
    cfg = get_config()
    path = index_path or cfg.paths.accession_index()
    if not path.exists():
        return []
    with open(path) as f:
        idx = json.load(f)
    if "gse_folder_index" in idx:
        return sorted(idx.get("gse_folder_index", {}).keys())
    gse_ids = set()
    for key, value in idx.items():
        if isinstance(key, str) and key.startswith("GSE"):
            gse_ids.add(key)
        if isinstance(value, dict):
            for subkey in value.keys():
                if isinstance(subkey, str) and subkey.startswith("GSE"):
                    gse_ids.add(subkey)
    return sorted(gse_ids)


# ---------------------------------------------------------------------------
# Main enrichment flows
# ---------------------------------------------------------------------------


def enrich_single_gse(gse_id: str, geo_datahub: Path | None = None) -> EnrichedStudy:
    cfg = get_config()
    logger.info(f"Enriching {gse_id} ...")

    try:
        series_soft = fetch_series_soft(gse_id)
        series_info = parse_series_soft(series_soft)
        time.sleep(cfg.services.rate_limit_delay)
    except Exception as e:
        logger.warning(f"Failed to fetch series SOFT for {gse_id}: {e}")
        series_info = {"title": "", "summary": "", "overall_design": "", "organism": "", "series_type": "", "platform": "", "pubmed_ids": [], "submission_date": "", "n_samples": 0}

    try:
        sample_soft = fetch_geo_soft(gse_id)
        samples = parse_sample_soft(sample_soft)
        time.sleep(cfg.services.rate_limit_delay)
    except Exception as e:
        logger.warning(f"Failed to fetch sample SOFT for {gse_id}: {e}")
        samples = {}

    char_summary = summarize_characteristics(samples)

    pubmed_info = None
    if series_info.get("pubmed_ids"):
        try:
            pubs = fetch_pubmed(series_info["pubmed_ids"])
            if pubs:
                pubmed_info = pubs[0]
            time.sleep(cfg.services.rate_limit_delay)
        except Exception as e:
            logger.warning(f"PubMed fetch failed for {gse_id}: {e}")

    files_summary = {}
    geo_datahub = geo_datahub or cfg.paths.geo_datahub_root
    if geo_datahub and geo_datahub.exists():
        existing = load_existing_geo_meta(geo_datahub, gse_id)
        if existing:
            geo_meta = existing.get("geo_metadata", {})
            if not series_info["title"] and geo_meta.get("title"):
                series_info["title"] = geo_meta["title"]
            if not series_info["summary"] and geo_meta.get("summary"):
                series_info["summary"] = geo_meta["summary"]
            if not series_info["series_type"] and geo_meta.get("series_type"):
                series_info["series_type"] = geo_meta["series_type"]
            files_summary = existing.get("summary", {})

    domain, modalities = derive_domain_modalities(series_info["series_type"], series_info["title"], series_info["summary"])

    return EnrichedStudy(
        gse_id=gse_id,
        title=series_info["title"],
        summary=series_info["summary"],
        overall_design=series_info["overall_design"],
        organism=series_info["organism"] or (char_summary.organisms[0] if char_summary.organisms else ""),
        platform=series_info["platform"],
        series_type=series_info["series_type"],
        n_samples=series_info["n_samples"] or len(samples),
        submission_date=series_info["submission_date"],
        domain=domain,
        modalities=modalities,
        samples=samples,
        characteristics_summary=char_summary,
        pubmed=pubmed_info,
        files_summary=files_summary,
    )


def build_enriched_document(gse_id: str) -> Dict:
    study = enrich_single_gse(gse_id)
    data = study.to_dict()
    data["tissues"] = study.characteristics_summary.tissues
    data["diseases"] = study.characteristics_summary.diseases
    data["cell_types"] = study.characteristics_summary.cell_types
    data["sample_count"] = study.n_samples
    data["document_text"] = study.to_search_text()
    if study.pubmed:
        data["pubmed"] = [study.pubmed.__dict__]
    else:
        data["pubmed"] = []
    return data


def enrich_batch(gse_ids: list[str], output_dir: Path, geo_datahub: Path | None = None, skip_existing: bool = True) -> list[EnrichedStudy]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for gse_id in gse_ids:
        out_path = output_dir / f"{gse_id}.json"
        if skip_existing and out_path.exists():
            try:
                with open(out_path) as f:
                    study = EnrichedStudy.from_dict(json.load(f))
                results.append(study)
                continue
            except Exception:
                pass
        try:
            study = enrich_single_gse(gse_id, geo_datahub)
            with open(out_path, "w") as f:
                json.dump(study.to_dict(), f, indent=2, default=str, ensure_ascii=False)
            results.append(study)
        except Exception as e:
            logger.error(f"Failed to enrich {gse_id}: {e}")
    return results


def enrich_all(output_dir: Optional[Path] = None, index_path: Optional[Path] = None, skip_existing: bool = True) -> List[Dict]:
    cfg = get_config()
    out = output_dir or cfg.paths.enriched_dir
    out.mkdir(parents=True, exist_ok=True)
    docs = []
    for gse_id in load_gse_ids(index_path):
        outfile = out / f"{gse_id}.json"
        if skip_existing and outfile.exists():
            with open(outfile) as f:
                docs.append(json.load(f))
            continue
        try:
            doc = build_enriched_document(gse_id)
            with open(outfile, "w") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
            docs.append(doc)
        except Exception as e:
            logger.error(f"Failed to enrich {gse_id}: {e}")
    return docs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enrich GEO metadata for scMetaIntel-Hub")
    parser.add_argument("--geo-datahub", type=Path, default=None, help="Path to external GEO-DataHub root")
    parser.add_argument("--gse-list", nargs="+", help="Specific GSE IDs to enrich")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = get_config()
    output_dir = args.output_dir or cfg.paths.enriched_dir

    if args.gse_list:
        studies = enrich_batch(args.gse_list, output_dir, geo_datahub=args.geo_datahub, skip_existing=not args.no_skip)
        print(f"Saved {len(studies)} enriched studies to {output_dir}")
        return

    studies = enrich_batch(load_gse_ids(), output_dir, geo_datahub=args.geo_datahub, skip_existing=not args.no_skip)
    print(f"Saved {len(studies)} enriched studies to {output_dir}")


if __name__ == "__main__":
    main()
