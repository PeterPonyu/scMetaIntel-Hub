"""
GEO-DataHub: Smart Classifier Module
=====================================
Classifies GEO supplementary files by:
  1. Data Format     — filtered_feature_bc_matrix.h5, filtered_peak_bc_matrix.h5,
                       MTX triplet, h5ad, loom, csv/tsv, fragments.tsv.gz, etc.
  2. Data Modality   — scRNA-seq, scATAC-seq, Multiome (RNA+ATAC), CITE-seq, etc.
  3. Processing Level— filtered, raw, processed
  4. Biological Domain — cancer, development, normal tissue, disease (non-cancer)

Based on analysis of real GEO naming conventions:
  scRNA  : *filtered_feature_bc_matrix.h5, *_matrix.mtx.gz + barcodes + features
  scATAC : *filtered_peak_bc_matrix.h5, *_fragments.tsv.gz, *peak_matrix*
  Multiome: both RNA + ATAC files present, or title/summary mentions multiome
  Cancer : title/summary keywords (tumor, cancer, carcinoma, melanoma, ...)
  Dev    : title/summary keywords (development, fetal, embryo, differentiation, ...)

Author: GEO-DataHub Pipeline
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("geo_classifier")


# ============================================================
# File Format Classification
# ============================================================

@dataclass
class FileClassification:
    """Classification result for a single supplementary file."""
    url: str
    filename: str
    # Format
    format_type: str = "unknown"          # h5_10x, h5ad, mtx, csv, tsv, loom, fragments, rds, tar, other
    # Modality signal from filename
    modality_signal: str = "unknown"      # rna, atac, multiome, cite, unknown
    # Processing level
    processing_level: str = "unknown"     # filtered, raw, processed, unknown
    # Specific 10x matrix type
    matrix_type: str = ""                 # feature_bc_matrix, peak_bc_matrix, raw_feature_bc_matrix, etc.
    # Is this a count matrix (vs metadata/annotation)?
    is_count_matrix: bool = False
    # File size in bytes (if known)
    file_size: Optional[int] = None


# --- Format type patterns ---
FORMAT_PATTERNS = [
    # h5ad
    (r'\.h5ad(\.gz)?$', 'h5ad'),
    # 10x HDF5 — feature bc matrix (RNA)
    (r'filtered_feature_bc_matrix\.h5$', 'h5_10x_filtered_feature'),
    (r'raw_feature_bc_matrix\.h5$', 'h5_10x_raw_feature'),
    # 10x HDF5 — peak bc matrix (ATAC)
    (r'filtered_peak_bc_matrix\.h5$', 'h5_10x_filtered_peak'),
    (r'raw_peak_bc_matrix\.h5$', 'h5_10x_raw_peak'),
    # 10x HDF5 — tf bc matrix (ATAC motif)
    (r'filtered_tf_bc_matrix\.h5$', 'h5_10x_filtered_tf'),
    # Generic h5
    (r'\.h5$', 'h5_generic'),
    # MTX sparse matrix components
    (r'matrix\.mtx(\.gz)?$', 'mtx_matrix'),
    (r'barcodes\.tsv(\.gz)?$', 'mtx_barcodes'),
    (r'(features|genes)\.tsv(\.gz)?$', 'mtx_features'),
    # ATAC fragments
    (r'fragments\.tsv(\.gz)?$', 'fragments'),
    # Loom
    (r'\.loom(\.gz)?$', 'loom'),
    # RDS (R objects — can't convert with scanpy)
    (r'\.rds(\.gz)?$', 'rds'),
    # Tar archives
    (r'\.(tar\.gz|tgz|tar)$', 'tar'),
    # CSV/TSV count matrices
    (r'\.(csv|csv\.gz)$', 'csv'),
    (r'\.(tsv|tsv\.gz|txt|txt\.gz)$', 'tsv'),
]

# --- Modality signal patterns from filenames ---
MODALITY_PATTERNS = [
    # ATAC signals
    (r'(atac|ATAC|chromatin|peak_bc_matrix|peak_matrix|fragments\.tsv|chromVAR)', 'atac'),
    # RNA signals
    (r'(rna|RNA|feature_bc_matrix|gene_bc_matrix|UMI_count|gene_count|gene_expression|spliced|unspliced)', 'rna'),
    # Multiome signals
    (r'(multiome|multi.?ome|10x.?arc|joint)', 'multiome'),
    # CITE-seq signals
    (r'(cite|CITE|ADT|HTO|antibody)', 'cite'),
]

# --- Processing level patterns ---
PROCESSING_PATTERNS = [
    (r'(filtered|filt_)', 'filtered'),
    (r'(raw_feature|raw_peak|raw_tf|RawCount)', 'raw'),
    (r'(processed|normalized|log.?count|scaled)', 'processed'),
]

# --- Count matrix identification ---
COUNT_MATRIX_PATTERNS = [
    r'matrix\.mtx', r'feature_bc_matrix', r'peak_bc_matrix', r'tf_bc_matrix',
    r'count', r'UMI', r'expression', r'\.h5ad', r'\.h5$', r'\.loom',
    r'gene_activit', r'peak_matrix',
]

# Non-count files (metadata, annotations, etc.)
NON_COUNT_PATTERNS = [
    r'annotation', r'metadata', r'cell_meta', r'sample_qc', r'filelist\.txt',
    r'README', r'cell_type', r'cluster_name', r'contig_annotation',
    r'_qc\.', r'hashtag',
]


def classify_file(url: str) -> FileClassification:
    """
    Classify a single GEO supplementary file URL.
    
    Returns a FileClassification with format, modality signal, processing level, etc.
    """
    filename = url.rstrip("/").split("/")[-1].lower()
    
    clf = FileClassification(url=url, filename=filename)
    
    # 1. Determine format type (use first matching pattern)
    for pattern, fmt in FORMAT_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            clf.format_type = fmt
            break
    
    # 2. Determine modality signal
    for pattern, modality in MODALITY_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            clf.modality_signal = modality
            break
    
    # 3. Determine processing level
    for pattern, level in PROCESSING_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            clf.processing_level = level
            break
    
    # 4. Identify specific matrix type
    if "filtered_feature_bc_matrix" in filename:
        clf.matrix_type = "filtered_feature_bc_matrix"
        if clf.modality_signal == "unknown":
            clf.modality_signal = "rna"
    elif "raw_feature_bc_matrix" in filename:
        clf.matrix_type = "raw_feature_bc_matrix"
        if clf.modality_signal == "unknown":
            clf.modality_signal = "rna"
    elif "filtered_peak_bc_matrix" in filename:
        clf.matrix_type = "filtered_peak_bc_matrix"
        if clf.modality_signal == "unknown":
            clf.modality_signal = "atac"
    elif "raw_peak_bc_matrix" in filename:
        clf.matrix_type = "raw_peak_bc_matrix"
        if clf.modality_signal == "unknown":
            clf.modality_signal = "atac"
    elif "filtered_tf_bc_matrix" in filename:
        clf.matrix_type = "filtered_tf_bc_matrix"
        if clf.modality_signal == "unknown":
            clf.modality_signal = "atac"
    
    # 5. Is it a count matrix?
    clf.is_count_matrix = any(re.search(p, filename, re.IGNORECASE) for p in COUNT_MATRIX_PATTERNS)
    if any(re.search(p, filename, re.IGNORECASE) for p in NON_COUNT_PATTERNS):
        clf.is_count_matrix = False
    
    return clf


def classify_files(urls: List[str]) -> List[FileClassification]:
    """Classify a list of supplementary file URLs."""
    return [classify_file(url) for url in urls]


# ============================================================
# Dataset-Level Classification
# ============================================================

@dataclass
class DatasetClassification:
    """Dataset-level classification for a GSE series."""
    gse_id: str
    # Available formats
    has_h5_10x_filtered_feature: bool = False    # filtered_feature_bc_matrix.h5
    has_h5_10x_raw_feature: bool = False         # raw_feature_bc_matrix.h5
    has_h5_10x_filtered_peak: bool = False       # filtered_peak_bc_matrix.h5
    has_h5_10x_raw_peak: bool = False            # raw_peak_bc_matrix.h5
    has_h5ad: bool = False
    has_mtx_triplet: bool = False
    has_loom: bool = False
    has_csv_tsv: bool = False
    has_fragments: bool = False
    has_tar: bool = False
    has_rds: bool = False
    
    # Detected modalities
    modalities: Set[str] = field(default_factory=set)  # rna, atac, multiome, cite
    
    # Biological domain (from metadata)
    domain: str = "unknown"           # cancer, development, normal, disease, unknown
    domain_keywords_matched: List[str] = field(default_factory=list)
    
    # Metadata
    organism: str = ""
    series_type: str = ""             # Expression profiling, Genome binding, etc.
    
    # File classifications
    file_classifications: List[FileClassification] = field(default_factory=list)
    
    def available_formats_str(self) -> str:
        """Human-readable string of available formats."""
        fmts = []
        if self.has_h5_10x_filtered_feature: fmts.append("filtered_feature_bc.h5")
        if self.has_h5_10x_raw_feature: fmts.append("raw_feature_bc.h5")
        if self.has_h5_10x_filtered_peak: fmts.append("filtered_peak_bc.h5")
        if self.has_h5_10x_raw_peak: fmts.append("raw_peak_bc.h5")
        if self.has_h5ad: fmts.append("h5ad")
        if self.has_mtx_triplet: fmts.append("MTX-triplet")
        if self.has_loom: fmts.append("loom")
        if self.has_csv_tsv: fmts.append("csv/tsv")
        if self.has_fragments: fmts.append("fragments.tsv")
        if self.has_tar: fmts.append("tar(内含样本)")
        if self.has_rds: fmts.append("rds(R)")
        return ", ".join(fmts) if fmts else "none"
    
    def modalities_str(self) -> str:
        return ", ".join(sorted(self.modalities)) if self.modalities else "unknown"

    def has_convertible_matrix(self) -> bool:
        """Return True if at least one file is in a format convertible to h5ad.

        Convertible formats include h5ad, HDF5 10x matrices, MTX triplet,
        loom, csv/tsv count matrices, and tar archives (which may contain
        inner matrices).  Formats that *cannot* produce a proper
        cell-by-gene h5ad without external tooling are excluded:
          - rds         (requires R / Seurat)
          - fragments   (ATAC fragment file only, no count matrix)
          - bigwig/bw   (genome-browser track, not a matrix)
          - RSEM / bulk (not single-cell)

        For ambiguous plain-text formats (csv, tsv) the file must also be
        flagged as ``is_count_matrix``; generic metadata/filelist files
        are excluded.
        """
        # Formats that are unambiguously matrix containers
        STRONG_FORMATS = CONVERTIBLE_FORMAT_TYPES - {"csv", "tsv"}
        for clf in self.file_classifications:
            if clf.format_type in STRONG_FORMATS:
                return True
            # csv/tsv only counts if flagged as a count matrix
            if clf.format_type in ("csv", "tsv") and clf.is_count_matrix:
                return True
        return False


# Formats that can be converted to a proper cell×gene h5ad by the pipeline.
# NOTE: 'tar' is deliberately excluded – a tar archive is just a container
#       whose inner files may or may not be convertible.  The pipeline probes
#       sample-level supplementary files to determine actual content formats.
CONVERTIBLE_FORMAT_TYPES: Set[str] = {
    "h5ad",
    "h5_10x_filtered_feature", "h5_10x_raw_feature",
    "h5_10x_filtered_peak", "h5_10x_raw_peak",
    "h5_10x_filtered_tf",
    "h5_generic",
    "mtx_matrix",      # MTX triplet (matrix component)
    "loom",
    "csv", "tsv",       # count-matrix tables
}


def classify_dataset_files(gse_id: str, file_urls: List[str]) -> DatasetClassification:
    """
    Classify all supplementary files for a dataset to determine
    available formats and modalities.
    """
    dc = DatasetClassification(gse_id=gse_id)
    dc.file_classifications = classify_files(file_urls)
    
    mtx_parts = set()  # track mtx components
    
    for clf in dc.file_classifications:
        fmt = clf.format_type
        
        # Format flags
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
            # Could be 10x — check matrix type
            if clf.matrix_type == "filtered_feature_bc_matrix":
                dc.has_h5_10x_filtered_feature = True
            elif clf.matrix_type == "filtered_peak_bc_matrix":
                dc.has_h5_10x_filtered_peak = True
        elif fmt in ("mtx_matrix", "mtx_barcodes", "mtx_features"):
            mtx_parts.add(fmt)
        elif fmt == "fragments":
            dc.has_fragments = True
        elif fmt == "loom":
            dc.has_loom = True
        elif fmt in ("csv", "tsv"):
            dc.has_csv_tsv = True
        elif fmt == "tar":
            dc.has_tar = True
        elif fmt == "rds":
            dc.has_rds = True
        
        # Modality signals
        if clf.modality_signal != "unknown":
            dc.modalities.add(clf.modality_signal)
    
    # MTX triplet requires at least matrix + barcodes + features
    if len(mtx_parts) >= 2 and "mtx_matrix" in mtx_parts:
        dc.has_mtx_triplet = True
    
    # Infer modality from formats if not detected
    if not dc.modalities:
        if dc.has_h5_10x_filtered_feature or dc.has_h5_10x_raw_feature:
            dc.modalities.add("rna")
        if dc.has_h5_10x_filtered_peak or dc.has_h5_10x_raw_peak or dc.has_fragments:
            dc.modalities.add("atac")
    
    # If both RNA and ATAC, could be multiome
    if "rna" in dc.modalities and "atac" in dc.modalities:
        dc.modalities.add("multiome")
    
    return dc


# ============================================================
# Biological Domain Classification
# ============================================================

CANCER_KEYWORDS = [
    r'cancer', r'tumor', r'tumour', r'carcinoma', r'melanoma', r'leukemia',
    r'leukaemia', r'lymphoma', r'glioma', r'glioblastoma', r'neuroblastoma',
    r'sarcoma', r'adenocarcinoma', r'hepatocellular', r'cholangiocarcinoma',
    r'mesothelioma', r'myeloma', r'neoplasm', r'malignant', r'metasta',
    r'oncogen', r'basal\s*cell', r'squamous\s*cell', r'pancreatic\s*ductal',
    r'\bCRC\b', r'\bHCC\b', r'\bNSCLC\b', r'\bSCLC\b', r'\bAML\b', r'\bALL\b',
    r'\bCLL\b', r'\bGBM\b', r'\bPDAC\b', r'\bTNBC\b', r'\bRCC\b',
    r'triple.negative', r'breast\s*cancer', r'lung\s*cancer', r'colon\s*cancer',
    r'colorectal', r'ovarian\s*cancer', r'prostate\s*cancer', r'bladder\s*cancer',
    r'gastric\s*cancer', r'liver\s*cancer', r'renal\s*cell',
    r'tumor\s*microenvir', r'immune\s*evasion', r'checkpoint',
]

DEVELOPMENT_KEYWORDS = [
    r'develop', r'embryo', r'fetal', r'foetal', r'fetus', r'organogenesis',
    r'differentiat', r'lineage\s*tracing', r'cell\s*fate', r'progenitor',
    r'stem\s*cell', r'pluripoten', r'iPSC', r'\bESC\b', r'hPSC',
    r'gastrulat', r'neurulat', r'somitogenesis', r'morphogenesis',
    r'maturation', r'specification', r'commitment', r'reprogramm',
    r'organoid', r'blastocyst', r'morula', r'zygote', r'implantation',
    r'hematopoie', r'haematopoie', r'erythropoie', r'myogenesis',
    r'chondrogenesis', r'osteogenesis', r'adipogenesis', r'angiogenesis',
    r'neurogenesis', r'gliogenesis', r'synaptogenesis',
    r'cell\s*atlas.*fetal', r'fetal.*atlas',
    r'time.?course', r'pseudo.?time', r'trajectory',
]

NORMAL_TISSUE_KEYWORDS = [
    r'normal\s*tissue', r'healthy\s*donor', r'healthy\s*adult', r'homeosta',
    r'tissue\s*atlas', r'cell\s*atlas', r'reference\s*atlas', r'cell\s*census',
    r'human\s*cell\s*landscape', r'human\s*cell\s*atlas',
    r'adult\s*tissue', r'normal\s*colon', r'normal\s*brain', r'normal\s*lung',
    r'normal\s*liver', r'normal\s*kidney', r'normal\s*heart',
    r'steady.?state', r'resting\s*state',
    r'peripheral\s*blood.*healthy', r'PBMC.*healthy',
]

DISEASE_NON_CANCER_KEYWORDS = [
    r'alzheimer', r'parkinson', r'huntington', r'ALS\b', r'sclerosis',
    r'fibrosis', r'cirrhosis', r'hepatitis', r'diabetes', r'obesity',
    r'asthma', r'COPD', r'arthritis', r'lupus', r'autoimmune',
    r'inflammatory', r'IBD\b', r'crohn', r'colitis', r'COVID',
    r'infection', r'sepsis', r'pneumonia', r'influenza', r'viral',
    r'injury', r'ischemia', r'infarction', r'stroke', r'trauma',
    r'neuropathy', r'retinopathy', r'cardiomyopathy', r'nephropathy',
    r'degenerat', r'dysplasia', r'atrophy',
]


def classify_domain(title: str, summary: str, overall_design: str = "") -> tuple:
    """
    Classify biological domain from GEO metadata text.
    
    Returns (domain_label, matched_keywords) where domain_label is one of:
    'cancer', 'development', 'normal', 'disease', 'unknown'
    
    Priority: cancer > development > disease > normal > unknown
    (Cancer takes priority because cancer papers often also mention development terms)
    """
    text = f"{title} {summary} {overall_design}".lower()
    
    matches = {}
    
    for kw in CANCER_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            matches.setdefault("cancer", []).append(kw)
    
    for kw in DEVELOPMENT_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            matches.setdefault("development", []).append(kw)
    
    for kw in DISEASE_NON_CANCER_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            matches.setdefault("disease", []).append(kw)
    
    for kw in NORMAL_TISSUE_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            matches.setdefault("normal", []).append(kw)
    
    # Scoring: count matches, cancer > disease > development > normal
    scores = {k: len(v) for k, v in matches.items()}
    
    if not scores:
        return "unknown", []
    
    # Cancer dominates if it has >= 2 keyword matches, or if it's the top scorer
    if scores.get("cancer", 0) >= 2:
        return "cancer", matches.get("cancer", [])
    
    # If only 1 cancer keyword but development has more, it's likely developmental
    if scores.get("cancer", 0) == 1 and scores.get("development", 0) > 2:
        return "development", matches.get("development", [])
    
    # Otherwise pick the top scorer
    best = max(scores, key=scores.get)
    return best, matches.get(best, [])


def classify_modality_from_metadata(
    title: str, summary: str, series_type: str, overall_design: str = ""
) -> Set[str]:
    """
    Detect data modalities from GEO metadata text and series type.
    
    Returns set of modalities: {'rna', 'atac', 'multiome', 'cite'}
    """
    text = f"{title} {summary} {overall_design}".lower()
    modalities = set()
    
    # RNA-seq signals  
    rna_patterns = [
        r'scRNA', r'snRNA', r'single.cell\s*RNA', r'single.nucleus\s*RNA',
        r'10x\s*(genomics|chromium).*RNA', r'drop.?seq', r'smart.?seq',
        r'CEL.?seq', r'MARS.?seq', r'inDrop', r'RNA.?seq',
        r'transcriptom', r'gene\s*expression',
    ]
    for p in rna_patterns:
        if re.search(p, text, re.IGNORECASE):
            modalities.add("rna")
            break
    
    # ATAC-seq signals
    atac_patterns = [
        r'scATAC', r'snATAC', r'single.cell\s*ATAC', r'single.nucleus\s*ATAC',
        r'chromatin\s*accessib', r'ATAC.?seq', r'open\s*chromatin',
    ]
    for p in atac_patterns:
        if re.search(p, text, re.IGNORECASE):
            modalities.add("atac")
            break
    
    # Multiome / Multi-omics signals
    multi_patterns = [
        r'multiome', r'multi.?ome', r'multi.?modal', r'multi.?omic',
        r'10x.*ARC', r'joint\s*profiling', r'paired.*RNA.*ATAC',
        r'simultaneous.*RNA.*ATAC', r'co.?assay',
    ]
    for p in multi_patterns:
        if re.search(p, text, re.IGNORECASE):
            modalities.add("multiome")
            modalities.add("rna")
            modalities.add("atac")
            break
    
    # CITE-seq / protein signals
    cite_patterns = [
        r'CITE.?seq', r'REAP.?seq', r'ECCITE', r'antibody.?derived\s*tag',
        r'surface\s*protein', r'ADT\b', r'protein\s*expression',
    ]
    for p in cite_patterns:
        if re.search(p, text, re.IGNORECASE):
            modalities.add("cite")
            break
    
    # Series type fallback
    if "Expression profiling by high throughput sequencing" in series_type:
        if "rna" not in modalities and "cite" not in modalities:
            modalities.add("rna")
    if "Genome binding/occupancy profiling by high throughput sequencing" in series_type:
        if "atac" not in modalities:
            modalities.add("atac")
    
    return modalities


# ============================================================
# File URL Filtering
# ============================================================

# CLI-friendly format filter names → format_type values they accept
FORMAT_FILTER_MAP = {
    "filtered_feature_h5": {"h5_10x_filtered_feature"},
    "raw_feature_h5":      {"h5_10x_raw_feature"},
    "filtered_peak_h5":    {"h5_10x_filtered_peak"},
    "raw_peak_h5":         {"h5_10x_raw_peak"},
    "h5ad":                {"h5ad"},
    "h5":                  {"h5_10x_filtered_feature", "h5_10x_raw_feature",
                            "h5_10x_filtered_peak", "h5_10x_raw_peak",
                            "h5_generic"},
    "mtx":                 {"mtx_matrix", "mtx_barcodes", "mtx_features"},
    "loom":                {"loom"},
    "csv":                 {"csv"},
    "tsv":                 {"tsv"},
    "fragments":           {"fragments"},
    "tar":                 {"tar"},
    "count_matrix":        None,  # Special: uses is_count_matrix flag
}


def filter_files_by_format(
    classifications: List[FileClassification],
    format_filters: List[str],
) -> List[FileClassification]:
    """
    Filter file classifications by format type.
    
    Args:
        classifications: List of FileClassification objects
        format_filters: List of format filter names (keys from FORMAT_FILTER_MAP)
    
    Returns:
        Filtered list of FileClassification objects
    """
    if not format_filters:
        return classifications
    
    # Build set of accepted format_types
    accepted_types = set()
    use_count_matrix_flag = False
    
    for ff in format_filters:
        ff = ff.lower().strip()
        if ff in FORMAT_FILTER_MAP:
            if FORMAT_FILTER_MAP[ff] is None:
                use_count_matrix_flag = True
            else:
                accepted_types.update(FORMAT_FILTER_MAP[ff])
        else:
            logger.warning(f"Unknown format filter: {ff}")
    
    result = []
    for clf in classifications:
        if clf.format_type in accepted_types:
            result.append(clf)
        elif use_count_matrix_flag and clf.is_count_matrix:
            result.append(clf)
    
    return result


def filter_files_by_modality(
    classifications: List[FileClassification],
    modality_filter: str,
) -> List[FileClassification]:
    """
    Filter file classifications by modality signal in filename.
    
    Args:
        classifications: List of FileClassification objects
        modality_filter: 'rna', 'atac', 'multiome', 'cite', or 'any'
    
    Returns:
        Filtered list
    """
    if not modality_filter or modality_filter == "any":
        return classifications
    
    result = []
    for clf in classifications:
        if clf.modality_signal == modality_filter:
            result.append(clf)
        elif clf.modality_signal == "unknown":
            # Keep unknown-modality files (could be either)
            result.append(clf)
    
    return result


# ============================================================
# Utility Functions
# ============================================================
def print_classification_report(dc: DatasetClassification):
    """Print a detailed classification report for a dataset."""
    print(f"\n  {'─'*56}")
    print(f"  [{dc.gse_id}]")
    print(f"  Modalities: {dc.modalities_str()}")
    print(f"  Domain:     {dc.domain} {dc.domain_keywords_matched[:3] if dc.domain_keywords_matched else ''}")
    print(f"  Formats:    {dc.available_formats_str()}")
    
    # File breakdown
    if dc.file_classifications:
        print(f"  Files ({len(dc.file_classifications)}):")
        for clf in dc.file_classifications:
            mod_tag = f"[{clf.modality_signal}]" if clf.modality_signal != "unknown" else ""
            lvl_tag = f"({clf.processing_level})" if clf.processing_level != "unknown" else ""
            count_tag = "★" if clf.is_count_matrix else " "
            print(f"    {count_tag} {clf.format_type:30s} {mod_tag:12s} {lvl_tag:12s} {clf.filename}")


def get_available_format_filters() -> str:
    """Return a readable list of supported format filter names."""
    lines = ["Available format filters:"]
    for name, types in FORMAT_FILTER_MAP.items():
        if types:
            lines.append(f"  {name:25s} → matches: {', '.join(sorted(types))}")
        else:
            lines.append(f"  {name:25s} → matches files flagged as count matrices")
    return "\n".join(lines)
