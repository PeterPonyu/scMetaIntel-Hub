"""
GEO-DataHub: GEO Search Module
===============================
Search NCBI GEO database for single-cell datasets matching user criteria.

Uses NCBI E-utilities API:
  - esearch: search for GEO DataSets (GDS) / Series (GSE)
  - esummary: get summaries for found records
  - GEOparse: parse detailed SOFT files

Author: GEO-DataHub Pipeline
"""

import os
import re
import json
import time
import logging
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import requests
import yaml

# ============================================================
# Configuration
# ============================================================
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"
ELINK_URL = f"{EUTILS_BASE}/elink.fcgi"
GEO_QUERY_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

# Rate limit: NCBI allows ~3 requests/sec without API key, ~10 with key
RATE_LIMIT_DELAY = 0.35  # seconds between requests

logger = logging.getLogger("geo_search")


# ============================================================
# Data Classes
# ============================================================
@dataclass
class GEOSeriesInfo:
    """Information about a GEO Series (GSE)."""
    gse_id: str
    title: str = ""
    summary: str = ""
    overall_design: str = ""
    organism: str = ""
    platform: str = ""
    n_samples: int = 0
    series_type: str = ""
    submission_date: str = ""
    supplementary_files: List[str] = field(default_factory=list)
    pubmed_ids: List[str] = field(default_factory=list)
    ftp_link: str = ""
    
    def __str__(self):
        return (
            f"[{self.gse_id}] {self.title}\n"
            f"  Organism: {self.organism} | Samples: {self.n_samples}\n"
            f"  Type: {self.series_type}\n"
            f"  Date: {self.submission_date}\n"
            f"  Supplementary files: {len(self.supplementary_files)}"
        )


# ============================================================
# Core Search Functions
# ============================================================
def build_search_query(config: dict) -> str:
    """
    Build an NCBI E-utilities search query from config parameters.
    
    The query targets the GDS database with optional organism and type filters.
    """
    parts = []
    
    # Main query
    base_query = config.get("query", "single cell RNA-seq")
    parts.append(f"({base_query})")
    
    # Organism filter
    organism = config.get("organism", "")
    if organism:
        parts.append(f'"{organism}"[Organism]')
    
    # Dataset type filter
    dataset_type = config.get("dataset_type", "")
    if dataset_type:
        parts.append(f'"{dataset_type}"[DataSet Type]')
    
    # Combine with AND
    full_query = " AND ".join(parts)
    logger.info(f"Search query: {full_query}")
    return full_query


def esearch_geo(query: str, max_results: int = 20, retstart: int = 0) -> dict:
    """
    Search GEO DataSets (GDS) using NCBI E-utilities esearch.
    
    Returns dict with 'count' (total), 'ids' (list of GDS IDs).
    """
    params = {
        "db": "gds",
        "term": query,
        "retmax": max_results,
        "retstart": retstart,
        "retmode": "xml",
        "usehistory": "y",
    }
    
    logger.info(f"Searching GEO: retmax={max_results}, retstart={retstart}")
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    
    root = ET.fromstring(resp.text)
    
    count = int(root.findtext("Count", "0"))
    ids = [id_elem.text for id_elem in root.findall(".//IdList/Id") if id_elem.text]
    
    # Extract WebEnv and QueryKey for history server
    web_env = root.findtext("WebEnv", "")
    query_key = root.findtext("QueryKey", "")
    
    logger.info(f"Found {count} total results, retrieved {len(ids)} IDs")
    
    return {
        "count": count,
        "ids": ids,
        "web_env": web_env,
        "query_key": query_key,
    }


def esummary_gds(gds_ids: List[str]) -> List[dict]:
    """
    Get summary information for GDS IDs using esummary.
    
    Returns list of summary dicts.
    """
    if not gds_ids:
        return []
    
    # Process in batches of 50
    all_summaries = []
    for i in range(0, len(gds_ids), 50):
        batch = gds_ids[i:i+50]
        params = {
            "db": "gds",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        
        resp = requests.get(ESUMMARY_URL, params=params, timeout=30)
        resp.raise_for_status()
        
        root = ET.fromstring(resp.text)
        
        for doc_sum in root.findall("DocSum"):
            summary = _parse_docsummary(doc_sum)
            if summary:
                all_summaries.append(summary)
        
        time.sleep(RATE_LIMIT_DELAY)
    
    return all_summaries


def _parse_docsummary(doc_sum) -> Optional[dict]:
    """Parse a single DocSum XML element into a dict."""
    info = {}
    
    uid = doc_sum.findtext("Id", "")
    info["uid"] = uid
    
    for item in doc_sum.findall("Item"):
        name = item.get("Name", "")
        item_type = item.get("Type", "")
        
        if item_type == "String":
            info[name] = item.text or ""
        elif item_type == "Integer":
            info[name] = int(item.text or "0")
        elif item_type == "List":
            info[name] = [sub.text for sub in item.findall("Item") if sub.text]
    
    return info


def gds_to_gse(summaries: List[dict]) -> List[GEOSeriesInfo]:
    """
    Convert GDS summary records to GEOSeriesInfo objects.
    
    GDS records contain GSE accessions; we extract and deduplicate them.
    """
    gse_map: Dict[str, GEOSeriesInfo] = {}
    
    for s in summaries:
        # Try to extract GSE accession from the Accession field
        accession = s.get("Accession", "")
        gse_id = ""
        
        # GDS accessions look like "GDS1234", GSE ones like "GSE1234"
        if accession.startswith("GSE"):
            gse_id = accession
        else:
            # Try to find GSE in the entry type or relation fields
            entry_type = s.get("entryType", "")
            if entry_type == "GSE":
                gse_id = f"GSE{s.get('uid', '')}"
            else:
                # For GDS records, the GSE is often in the GPL/GSE reference
                gse_match = re.search(r'GSE\d+', str(s))
                if gse_match:
                    gse_id = gse_match.group()
                else:
                    # Construct from UID - GDS UIDs for GSE series start with 200
                    uid = s.get("uid", "")
                    if uid.startswith("200"):
                        gse_id = f"GSE{uid[3:]}"
                    else:
                        continue
        
        if not gse_id or gse_id in gse_map:
            continue
        
        # Build FTP link
        gse_num = gse_id.replace("GSE", "")
        gse_nnn = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSEnnn"
        ftp_link = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_nnn}/{gse_id}/"
        
        info = GEOSeriesInfo(
            gse_id=gse_id,
            title=s.get("title", ""),
            summary=s.get("summary", ""),
            organism=s.get("taxon", ""),
            platform=s.get("GPL", ""),
            n_samples=s.get("n_samples", 0),
            series_type=s.get("gdsType", ""),
            submission_date=s.get("PDAT", ""),
            supplementary_files=[],
            pubmed_ids=s.get("PubMedIds", []),
            ftp_link=ftp_link,
        )
        
        gse_map[gse_id] = info
    
    return list(gse_map.values())


def fetch_supplementary_file_list(gse_id: str) -> List[str]:
    """
    Fetch the list of supplementary files for a GSE series from NCBI FTP.
    
    Returns list of file URLs.
    """
    gse_num = gse_id.replace("GSE", "")
    gse_nnn = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSEnnn"
    suppl_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_nnn}/{gse_id}/suppl/"
    
    try:
        resp = requests.get(suppl_url, timeout=30)
        if resp.status_code == 404:
            logger.warning(f"No supplementary files found for {gse_id}")
            return []
        resp.raise_for_status()
        
        # Parse the directory listing HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        
        files = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href and not href.startswith("/") and not href.startswith("?") and not href.startswith("http"):
                files.append(f"{suppl_url}{href}")
        
        logger.info(f"{gse_id}: found {len(files)} supplementary files")
        return files
        
    except Exception as e:
        logger.error(f"Error fetching file list for {gse_id}: {e}")
        return []


def fetch_sample_supplementary_urls(gse_id: str) -> List[str]:
    """
    Fetch per-sample (GSM) supplementary file URLs from GEO SOFT metadata.

    This reveals what individual files are inside the _RAW.tar archive.
    For example, a tar-only GSE might have sample files like:
      GSM0000001_filtered_feature_bc_matrix.h5.gz  (convertible)
      GSM0000001_something.bw                      (NOT convertible)

    The URLs returned here are the *filenames* (as URLs) that GEO records
    for each sample; they are used purely for format classification —
    the actual download still uses the series-level tar.

    Returns:
        List of sample-level supplementary file URLs/paths.
    """
    sample_files: List[str] = []
    try:
        # Fetch full SOFT metadata including samples
        url = f"{GEO_QUERY_URL}?acc={gse_id}&targ=gsm&form=text&view=brief"
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()

        for line in resp.text.split("\n"):
            line = line.strip()
            if line.startswith("!Sample_supplementary_file"):
                # Format: !Sample_supplementary_file = ftp://...
                # or      !Sample_supplementary_file_1 = ftp://...
                if "=" in line:
                    val = line.split("=", 1)[1].strip()
                    if val and val.lower() != "none":
                        sample_files.append(val)

        logger.info(
            "%s: found %d sample-level supplementary file entries",
            gse_id, len(sample_files),
        )
    except Exception as e:
        logger.warning("Could not fetch sample-level files for %s: %s", gse_id, e)

    return sample_files


def is_single_cell_data(files: List[str]) -> bool:
    """
    Heuristic check if supplementary files look like single-cell data.
    
    Looks for common 10x/single-cell file patterns.
    """
    sc_patterns = [
        r'matrix\.mtx',
        r'barcodes\.tsv',
        r'features\.tsv',
        r'genes\.tsv',
        r'filtered_feature_bc_matrix',
        r'raw_feature_bc_matrix',
        r'filtered_gene_bc_matrices',
        r'\.h5$',
        r'\.h5ad$',
        r'\.loom$',
        r'10[xX]',
        r'cellranger',
    ]
    
    file_str = " ".join(files).lower()
    for pattern in sc_patterns:
        if re.search(pattern, file_str, re.IGNORECASE):
            return True
    
    return False


def classify_supplementary_files(files: List[str]) -> dict:
    """
    Classify supplementary files by type for download prioritization.
    
    Returns dict with categories: 'h5', 'h5ad', 'mtx', 'csv', 'loom', 'tar', 'other'
    """
    categories = {
        "h5": [],
        "h5ad": [],
        "mtx": [],
        "csv_tsv": [],
        "loom": [],
        "tar": [],
        "other": [],
    }
    
    for f in files:
        fname = f.lower()
        if fname.endswith(".h5ad") or fname.endswith(".h5ad.gz"):
            categories["h5ad"].append(f)
        elif fname.endswith(".h5") or fname.endswith(".h5.gz"):
            categories["h5"].append(f)
        elif "matrix.mtx" in fname or "barcodes.tsv" in fname or "features.tsv" in fname or "genes.tsv" in fname:
            categories["mtx"].append(f)
        elif any(fname.endswith(ext) for ext in [".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz"]):
            categories["csv_tsv"].append(f)
        elif fname.endswith(".loom") or fname.endswith(".loom.gz"):
            categories["loom"].append(f)
        elif any(fname.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz"]):
            categories["tar"].append(f)
        else:
            categories["other"].append(f)
    
    return categories


# ============================================================
# Main Search Pipeline
# ============================================================
def search_geo(config: dict) -> List[GEOSeriesInfo]:
    """
    Full search pipeline: query GEO -> get summaries -> enrich with file info.
    
    Args:
        config: Search configuration dict (from config.yaml 'search' section)
    
    Returns:
        List of GEOSeriesInfo objects with supplementary file information
    """
    # Step 1: Build query and search
    query = build_search_query(config)
    max_results = config.get("max_results", 20)
    
    search_result = esearch_geo(query, max_results=max_results)
    
    if not search_result["ids"]:
        logger.warning("No results found for query.")
        return []
    
    logger.info(f"Total matching records: {search_result['count']}")
    
    # Step 2: Get summaries
    time.sleep(RATE_LIMIT_DELAY)
    summaries = esummary_gds(search_result["ids"])
    
    # Step 3: Convert to GSE info
    gse_list = gds_to_gse(summaries)
    logger.info(f"Found {len(gse_list)} unique GSE series")
    
    # Step 4: Fetch supplementary file lists and classify
    for gse_info in gse_list:
        time.sleep(RATE_LIMIT_DELAY)
        gse_info.supplementary_files = fetch_supplementary_file_list(gse_info.gse_id)
    
    # Step 5: Filter by minimum samples if specified  
    min_samples = config.get("min_samples", 1)
    gse_list = [g for g in gse_list if g.n_samples >= min_samples]
    
    return gse_list


def search_geo_direct(gse_ids: List[str]) -> List[GEOSeriesInfo]:
    """
    Directly look up specific GSE accessions (bypass search).
    
    Args:
        gse_ids: List of GSE accession numbers, e.g. ["GSE123456", "GSE789012"]
    
    Returns:
        List of GEOSeriesInfo objects
    """
    results = []
    
    for gse_id in gse_ids:
        gse_id = gse_id.strip().upper()
        if not gse_id.startswith("GSE"):
            logger.warning(f"Skipping invalid accession: {gse_id}")
            continue
        
        logger.info(f"Looking up {gse_id}...")
        
        # Build FTP link
        gse_num = gse_id.replace("GSE", "")
        gse_nnn = f"GSE{gse_num[:-3]}nnn" if len(gse_num) > 3 else "GSEnnn"
        ftp_link = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse_nnn}/{gse_id}/"
        
        # Get supplementary files
        files = fetch_supplementary_file_list(gse_id)
        
        # Try to get metadata from GEO query page
        meta = _fetch_gse_metadata(gse_id)
        
        info = GEOSeriesInfo(
            gse_id=gse_id,
            title=meta["title"],
            summary=meta["summary"],
            overall_design=meta["overall_design"],
            organism=meta["organism"],
            series_type=meta["series_type"],
            platform=meta["platform"],
            ftp_link=ftp_link,
            supplementary_files=files,
        )
        
        results.append(info)
        time.sleep(RATE_LIMIT_DELAY)
    
    return results


def _fetch_gse_metadata(gse_id: str) -> dict:
    """Fetch detailed metadata for a GSE from NCBI GEO SOFT format."""
    meta = {
        "title": "", "summary": "", "overall_design": "",
        "organism": "", "series_type": "", "platform": "",
    }
    try:
        url = f"{GEO_QUERY_URL}?acc={gse_id}&targ=self&form=text"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        summaries = []
        designs = []
        types = []
        
        for line in resp.text.split("\n"):
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            val = val.strip()
            key = key.strip()
            
            if key == "!Series_title":
                meta["title"] = val
            elif key == "!Series_summary":
                summaries.append(val)
            elif key == "!Series_overall_design":
                designs.append(val)
            elif key == "!Series_sample_organism":
                meta["organism"] = val
            elif key == "!Series_type":
                types.append(val)
            elif key == "!Series_platform_id":
                meta["platform"] = val
        
        meta["summary"] = " ".join(summaries)
        meta["overall_design"] = " ".join(designs)
        meta["series_type"] = "; ".join(types)
        
    except Exception as e:
        logger.warning(f"Could not fetch metadata for {gse_id}: {e}")
    
    return meta


# ============================================================
# Output / Display
# ============================================================
def print_search_results(gse_list: List[GEOSeriesInfo]):
    """Pretty-print search results."""
    print(f"\n{'='*80}")
    print(f" GEO Search Results: {len(gse_list)} datasets found")
    print(f"{'='*80}")
    
    for i, gse in enumerate(gse_list, 1):
        file_cats = classify_supplementary_files(gse.supplementary_files)
        sc_flag = "✓ SC" if is_single_cell_data(gse.supplementary_files) else "? Unknown"
        
        print(f"\n[{i}] {gse.gse_id} — {gse.title[:80]}")
        print(f"    Organism: {gse.organism}")
        print(f"    Samples: {gse.n_samples} | Type: {gse.series_type}")
        print(f"    Date: {gse.submission_date}")
        print(f"    Data format: {sc_flag}")
        
        # Show file type breakdown
        type_counts = {k: len(v) for k, v in file_cats.items() if v}
        if type_counts:
            print(f"    Files: {type_counts}")
        
        print(f"    FTP: {gse.ftp_link}")
    
    print(f"\n{'='*80}\n")


def save_search_results(gse_list: List[GEOSeriesInfo], output_path: str):
    """Save search results to JSON."""
    data = [asdict(g) for g in gse_list]
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Search results saved to {output_path}")


def load_search_results(input_path: str) -> List[GEOSeriesInfo]:
    """Load search results from JSON."""
    with open(input_path, "r") as f:
        data = json.load(f)
    
    return [GEOSeriesInfo(**d) for d in data]


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Search GEO for single-cell datasets")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--gse", nargs="+", help="Directly lookup specific GSE accessions")
    parser.add_argument("--output", default=None, help="Output JSON path for search results")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    if args.gse:
        # Direct lookup mode
        gse_list = search_geo_direct(args.gse)
    else:
        # Config-based search mode
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        gse_list = search_geo(config.get("search", {}))
    
    # Display results
    print_search_results(gse_list)
    
    # Save results
    output_path = args.output or "search_results.json"
    save_search_results(gse_list, output_path)
    
    return gse_list


if __name__ == "__main__":
    main()
